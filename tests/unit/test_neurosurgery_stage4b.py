import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from aegis_lab.editing.neurosurgery.stage4b_contract import (
    OperationKind,
    ExecutionSemantics,
    ComponentType,
    ConflictSeverity,
    ConflictType,
    NeurosurgeryError,
    ValidationError,
    ConflictingOperationsError,
    InvalidCompositionPathError,
    UnsupportedOperationError,
    UnifiedOperation,
    UnifiedPlan,
    ConflictDetector,
    IndexRemapper,
    ModelTopology,
    OperationComposer,
    ComposedPipeline,
    ArchitectureCapabilities,
    AdapterCapabilityRegistry,
    inspect_parameter_aliases,
    get_alias_groups,
    inspect_module_aliases,
    detect_conflicts,
    validate_no_conflicts,
    compose_operations,
    get_capability_matrix,
    check_adapter_capability,
    validate_adapter_capability,
    PlanRuntimeContext,
    PlanExecutionResult,
    PlanExecutor,
    PlanApplicator,
    apply_plan,
    execute_plan,
)


class MockTiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 8)
        self.lm_head = nn.Linear(8, 10, bias=False)
        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight
        self.config = SimpleNamespace(model_type="llama")

    def forward(self, x):
        return self.lm_head(self.embed_tokens(x))


class MockSharedLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        shared_layer = nn.Linear(4, 4)
        self.layers = nn.ModuleList([shared_layer, shared_layer])


class MockToyTransformer(nn.Module):
    def __init__(self, num_layers: int = 4):
        super().__init__()
        class TinyMLP(nn.Module):
            def __init__(self, in_features=4, hidden_features=8):
                super().__init__()
                self.gate_proj = nn.Linear(in_features, hidden_features, bias=False)
                self.up_proj = nn.Linear(in_features, hidden_features, bias=False)
                self.down_proj = nn.Linear(hidden_features, in_features, bias=False)

            def forward(self, x):
                return self.down_proj(self.gate_proj(x) * self.up_proj(x))

        class TinyAttn(nn.Module):
            def __init__(self, hidden_dim=4, num_heads=4, num_kv_heads=2, head_dim=2):
                super().__init__()
                self.num_heads = num_heads
                self.num_key_value_heads = num_kv_heads
                self.head_dim = head_dim
                self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
                self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
                self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
                self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

            def forward(self, x):
                q = self.q_proj(x)
                return self.o_proj(q)

        class TinyLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = TinyMLP()
                self.self_attn = TinyAttn()

            def forward(self, x):
                return x + self.self_attn(x) + self.mlp(x)

        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer() for _ in range(num_layers)])
        self.config = SimpleNamespace(
            model_type="llama",
            num_hidden_layers=num_layers,
            intermediate_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


class MockMoETransformer(nn.Module):
    def __init__(self, num_layers: int = 2, num_experts: int = 8, hidden_dim: int = 4):
        super().__init__()
        class TinyMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = nn.Linear(hidden_dim, num_experts, bias=False)
                self.experts = nn.ModuleList([
                    nn.Sequential(nn.Linear(hidden_dim, 8), nn.Linear(8, hidden_dim))
                    for _ in range(num_experts)
                ])

            def forward(self, x):
                weights = torch.softmax(self.router(x), dim=-1)
                out = torch.zeros_like(x)
                for i, exp in enumerate(self.experts):
                    out = out + weights[..., i:i+1] * exp(x)
                return out

        class TinyMoELayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = None
                self.self_attn = None
                self.block_sparse_moe = TinyMoE()

            def forward(self, x):
                return x + self.block_sparse_moe(x)

        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyMoELayer() for _ in range(num_layers)])
        self.config = SimpleNamespace(
            model_type="mixtral",
            num_hidden_layers=num_layers,
            num_experts=num_experts,
            num_local_experts=num_experts,
            num_experts_per_tok=2,
        )

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


class TestUnifiedOperationContract(unittest.TestCase):
    def test_all_seven_operation_kinds_constructible(self):
        op_mask = UnifiedOperation.create_mask(
            target_path="model.layers.0.mlp",
            component_type=ComponentType.MLP_CHANNEL,
            target_indices=[0, 1],
            mask_value=0.0,
            execution_semantics=ExecutionSemantics.RUNTIME_ONLY,
            export_supported=False,
        )
        self.assertEqual(op_mask.kind, OperationKind.MASK)
        self.assertEqual(op_mask.execution_semantics, ExecutionSemantics.RUNTIME_ONLY)
        self.assertFalse(op_mask.export_supported)

        op_scale = UnifiedOperation.create_scale(
            target_path="model.layers.0.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            scale_factor=1.5,
            execution_semantics=ExecutionSemantics.PERSISTENT,
            export_supported=True,
        )
        self.assertEqual(op_scale.kind, OperationKind.SCALE)
        self.assertEqual(op_scale.parameters["scale_factor"], 1.5)

        op_clamp = UnifiedOperation.create_clamp(
            target_path="model.layers.0.self_attn",
            component_type=ComponentType.ATTENTION_HEAD,
            min_value=-2.0,
            max_value=2.0,
            execution_semantics=ExecutionSemantics.RUNTIME_ONLY,
            export_supported=False,
        )
        self.assertEqual(op_clamp.kind, OperationKind.CLAMP)
        self.assertEqual(op_clamp.parameters["min_value"], -2.0)
        self.assertEqual(op_clamp.parameters["max_value"], 2.0)

        op_project = UnifiedOperation.create_project(
            target_path="model.layers.0.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            direction=[0.5, 0.5],
            preserve_subspace=True,
        )
        self.assertEqual(op_project.kind, OperationKind.PROJECT)
        self.assertTrue(op_project.parameters["preserve_subspace"])

        op_replace = UnifiedOperation.create_replace(
            target_path="model.layers.1.mlp.gate_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            replacement_value=0.0,
        )
        self.assertEqual(op_replace.kind, OperationKind.REPLACE)

        op_lora = UnifiedOperation.create_low_rank_delta(
            target_path="model.layers.0.self_attn.o_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            rank=4,
            alpha=2.0,
        )
        self.assertEqual(op_lora.kind, OperationKind.LOW_RANK_DELTA)
        self.assertEqual(op_lora.parameters["rank"], 4)

        op_remove = UnifiedOperation.create_physical_remove(
            target_path="model.layers.2",
            component_type=ComponentType.FULL_LAYER,
            layer_index=2,
        )
        self.assertEqual(op_remove.kind, OperationKind.PHYSICAL_REMOVE)
        self.assertEqual(op_remove.layer_index, 2)

    def test_runtime_only_with_export_supported_rejected(self):
        with self.assertRaisesRegex(ValidationError, "cannot have export_supported=True"):
            UnifiedOperation(
                kind=OperationKind.MASK,
                target_path="model.layers.0.mlp",
                component_type=ComponentType.MLP_CHANNEL,
                execution_semantics=ExecutionSemantics.RUNTIME_ONLY,
                export_supported=True,
            )

    def test_operation_kind_validation(self):
        # Scale missing scale_factor
        with self.assertRaisesRegex(ValidationError, "requires 'scale_factor'"):
            UnifiedOperation(
                kind=OperationKind.SCALE,
                target_path="model.layers.0.mlp.down_proj.weight",
                component_type=ComponentType.WEIGHT_TENSOR,
                execution_semantics=ExecutionSemantics.PERSISTENT,
                export_supported=True,
                parameters={},
            )

        # Clamp with min_value > max_value
        with self.assertRaisesRegex(ValidationError, "min_value > max_value"):
            UnifiedOperation(
                kind=OperationKind.CLAMP,
                target_path="model.layers.0.mlp",
                component_type=ComponentType.MLP_CHANNEL,
                execution_semantics=ExecutionSemantics.RUNTIME_ONLY,
                export_supported=False,
                parameters={"min_value": 5.0, "max_value": 1.0},
            )

        # Project missing direction
        with self.assertRaisesRegex(ValidationError, "requires 'direction'"):
            UnifiedOperation(
                kind=OperationKind.PROJECT,
                target_path="model.layers.0.mlp.down_proj.weight",
                component_type=ComponentType.WEIGHT_TENSOR,
                execution_semantics=ExecutionSemantics.PERSISTENT,
                export_supported=True,
                parameters={},
            )

        # Low-rank delta missing rank
        with self.assertRaisesRegex(ValidationError, "requires positive 'rank'"):
            UnifiedOperation(
                kind=OperationKind.LOW_RANK_DELTA,
                target_path="model.layers.0.mlp.down_proj.weight",
                component_type=ComponentType.WEIGHT_TENSOR,
                execution_semantics=ExecutionSemantics.PERSISTENT,
                export_supported=True,
                parameters={"rank": 0},
            )

        # Replace missing replacement value
        with self.assertRaisesRegex(ValidationError, "requires replacement value"):
            UnifiedOperation(
                kind=OperationKind.REPLACE,
                target_path="model.layers.0.mlp.down_proj.weight",
                component_type=ComponentType.WEIGHT_TENSOR,
                execution_semantics=ExecutionSemantics.PERSISTENT,
                export_supported=True,
                parameters={},
            )

        # Physical remove missing indices
        with self.assertRaisesRegex(ValidationError, "must specify target_indices"):
            UnifiedOperation(
                kind=OperationKind.PHYSICAL_REMOVE,
                target_path="root_tensor",
                component_type=ComponentType.WEIGHT_TENSOR,
                execution_semantics=ExecutionSemantics.PERSISTENT,
                export_supported=True,
                layer_index=None,
                target_indices=None,
                parameters={},
            )

    def test_operation_serialization_roundtrip(self):
        op = UnifiedOperation.create_scale(
            target_path="model.layers.3.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            scale_factor=0.75,
            target_indices=[2, 4],
            order=1,
            metadata={"source": "search_frontier"},
        )
        d = op.to_dict()
        recovered = UnifiedOperation.from_dict(d)
        self.assertEqual(op.kind, recovered.kind)
        self.assertEqual(op.target_path, recovered.target_path)
        self.assertEqual(op.layer_index, recovered.layer_index)
        self.assertEqual(op.parameters["scale_factor"], recovered.parameters["scale_factor"])
        self.assertEqual(op.target_indices, recovered.target_indices)
        self.assertEqual(op.order, recovered.order)
        self.assertEqual(op.metadata["source"], recovered.metadata["source"])

    def test_unified_plan_yaml_roundtrip(self):
        op1 = UnifiedOperation.create_scale(
            target_path="model.layers.0.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            scale_factor=1.2,
            order=1,
        )
        op2 = UnifiedOperation.create_physical_remove(
            target_path="model.layers.1",
            component_type=ComponentType.FULL_LAYER,
            layer_index=1,
            order=2,
        )
        plan = UnifiedPlan(
            version=4,
            model_type="llama",
            operations=[op1, op2],
            metadata={"experiment": "stage4b_testing"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "plan.yaml"
            plan.to_yaml(file_path)

            loaded_from_file = UnifiedPlan.from_yaml(file_path)
            self.assertEqual(len(loaded_from_file.operations), 2)
            self.assertEqual(loaded_from_file.model_type, "llama")
            self.assertEqual(loaded_from_file.version, 4)

            # Also load from string
            yaml_content = file_path.read_text(encoding="utf-8")
            loaded_from_str = UnifiedPlan.from_yaml(yaml_content)
            self.assertEqual(len(loaded_from_str.operations), 2)

    def test_plan_validation_unsupported_version(self):
        with self.assertRaisesRegex(ValidationError, "Unsupported plan version: 3"):
            UnifiedPlan(version=3, operations=[]).validate()


class TestSharedWeightAndAliasingConflictDetection(unittest.TestCase):
    def test_inspect_parameter_aliases_tied_weights(self):
        model = MockTiedModel()
        aliases = inspect_parameter_aliases(model)
        self.assertIn("lm_head.weight", aliases)
        self.assertIn("embed_tokens.weight", aliases)
        self.assertEqual(
            set(aliases["lm_head.weight"]),
            {"lm_head.weight", "embed_tokens.weight"},
        )
        groups = get_alias_groups(model)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], {"lm_head.weight", "embed_tokens.weight"})

    def test_inspect_module_aliases_shared_layers(self):
        model = MockSharedLayerModel()
        mod_aliases = inspect_module_aliases(model)
        self.assertIn("layers.0", mod_aliases)
        self.assertIn("layers.1", mod_aliases)
        self.assertEqual(set(mod_aliases["layers.0"]), {"layers.0", "layers.1"})

    def test_detect_tied_weight_destruction(self):
        model = MockTiedModel()
        op_destructive = UnifiedOperation.create_physical_remove(
            target_path="lm_head.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            target_indices=[0, 1],
        )
        conflicts = detect_conflicts([op_destructive], model=model)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].conflict_type, ConflictType.TIED_WEIGHT_DESTRUCTION)
        self.assertEqual(conflicts[0].severity, ConflictSeverity.ERROR)

        with self.assertRaises(ConflictingOperationsError):
            validate_no_conflicts([op_destructive], model=model)

    def test_detect_undefined_composition_order_aliased(self):
        model = MockTiedModel()
        op1 = UnifiedOperation.create_scale(
            target_path="lm_head.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            scale_factor=0.9,
            order=None,
        )
        op2 = UnifiedOperation.create_project(
            target_path="embed_tokens.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            direction=[1.0] * 8,
            order=None,
        )
        conflicts = detect_conflicts([op1, op2], model=model)
        conflict_types = [c.conflict_type for c in conflicts]
        self.assertIn(ConflictType.UNDEFINED_COMPOSITION_ORDER, conflict_types)

        with self.assertRaises(ConflictingOperationsError):
            validate_no_conflicts([op1, op2], model=model)

    def test_detect_undefined_composition_order_hierarchical(self):
        op1 = UnifiedOperation.create_scale(
            target_path="model.layers.0.mlp",
            component_type=ComponentType.MLP_CHANNEL,
            scale_factor=0.5,
            order=None,
        )
        op2 = UnifiedOperation.create_scale(
            target_path="model.layers.0.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            scale_factor=0.8,
            order=None,
        )
        conflicts = detect_conflicts([op1, op2])
        self.assertTrue(any(c.conflict_type == ConflictType.UNDEFINED_COMPOSITION_ORDER for c in conflicts))

    def test_explicit_order_resolves_conflict(self):
        model = MockTiedModel()
        op1 = UnifiedOperation.create_scale(
            target_path="lm_head.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            scale_factor=0.9,
            order=1,
        )
        op2 = UnifiedOperation.create_project(
            target_path="embed_tokens.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            direction=[1.0] * 8,
            order=2,
        )
        conflicts = detect_conflicts([op1, op2], model=model)
        self.assertEqual(len(conflicts), 0)
        # Should not raise
        validate_no_conflicts([op1, op2], model=model)

    def test_incompatible_execution_semantics_conflict(self):
        model = MockTiedModel()
        op1 = UnifiedOperation.create_scale(
            target_path="lm_head.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            scale_factor=0.9,
            execution_semantics=ExecutionSemantics.RUNTIME_ONLY,
            export_supported=False,
            order=1,
        )
        op2 = UnifiedOperation.create_scale(
            target_path="embed_tokens.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            scale_factor=1.1,
            execution_semantics=ExecutionSemantics.PERSISTENT,
            export_supported=True,
            order=2,
        )
        conflicts = detect_conflicts([op1, op2], model=model)
        self.assertTrue(any(c.conflict_type == ConflictType.INCOMPATIBLE_SEMANTICS for c in conflicts))

    def test_independent_weights_no_conflict(self):
        op1 = UnifiedOperation.create_scale(
            target_path="model.layers.0.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            scale_factor=0.5,
        )
        op2 = UnifiedOperation.create_scale(
            target_path="model.layers.1.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            scale_factor=0.8,
        )
        conflicts = detect_conflicts([op1, op2])
        self.assertEqual(len(conflicts), 0)


class TestArbitraryEditCompositionAndIndexRemapping(unittest.TestCase):
    def test_single_layer_removal_remaps_indices_and_paths(self):
        op1 = UnifiedOperation.create_physical_remove(
            target_path="model.layers.1",
            component_type=ComponentType.FULL_LAYER,
            layer_index=1,
            order=1,
        )
        op2 = UnifiedOperation.create_scale(
            target_path="model.layers.2.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            layer_index=2,
            scale_factor=0.7,
            order=2,
        )
        op3 = UnifiedOperation.create_scale(
            target_path="model.layers.3.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            layer_index=3,
            scale_factor=0.9,
            order=3,
        )

        composer = OperationComposer(ModelTopology(num_layers=4))
        pipeline = composer.compose([op1, op2, op3])

        # op2: original layer 2 should remap to current layer 1
        remapped_op2 = pipeline.remapped_operations[1]
        self.assertEqual(remapped_op2.layer_index, 1)
        self.assertEqual(remapped_op2.target_path, "model.layers.1.mlp.down_proj.weight")

        # op3: original layer 3 should remap to current layer 2
        remapped_op3 = pipeline.remapped_operations[2]
        self.assertEqual(remapped_op3.layer_index, 2)
        self.assertEqual(remapped_op3.target_path, "model.layers.2.mlp.down_proj.weight")

    def test_multi_layer_removal_chained(self):
        op1 = UnifiedOperation.create_physical_remove(
            target_path="model.layers.0",
            component_type=ComponentType.FULL_LAYER,
            layer_index=0,
            order=1,
        )
        op2 = UnifiedOperation.create_physical_remove(
            target_path="model.layers.2",
            component_type=ComponentType.FULL_LAYER,
            layer_index=2,
            order=2,
        )
        op3 = UnifiedOperation.create_scale(
            target_path="model.layers.3.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            layer_index=3,
            scale_factor=0.5,
            order=3,
        )

        composer = OperationComposer(ModelTopology(num_layers=5))
        pipeline = composer.compose([op1, op2, op3])

        # Layers 0 and 2 removed from [0, 1, 2, 3, 4]
        # Remaining: original [1, 3, 4] -> current [0, 1, 2]
        # Original 3 maps to current 1
        remapped_op3 = pipeline.remapped_operations[2]
        self.assertEqual(remapped_op3.layer_index, 1)
        self.assertEqual(remapped_op3.target_path, "model.layers.1.mlp.down_proj.weight")

    def test_targeting_removed_layer_rejected(self):
        op1 = UnifiedOperation.create_physical_remove(
            target_path="model.layers.1",
            component_type=ComponentType.FULL_LAYER,
            layer_index=1,
            order=1,
        )
        op2 = UnifiedOperation.create_scale(
            target_path="model.layers.1.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            layer_index=1,
            scale_factor=0.5,
            order=2,
        )

        composer = OperationComposer(ModelTopology(num_layers=4))
        with self.assertRaisesRegex(InvalidCompositionPathError, "already physically removed"):
            composer.compose([op1, op2])

    def test_mlp_channel_slicing_and_remapping(self):
        # Layer 0 MLP intermediate size = 8 channels [0..7]
        op1 = UnifiedOperation.create_physical_remove(
            target_path="model.layers.0.mlp",
            component_type=ComponentType.MLP_CHANNEL,
            layer_index=0,
            kept_indices=[0, 2, 4, 6],
            order=1,
        )
        op2 = UnifiedOperation.create_scale(
            target_path="model.layers.0.mlp",
            component_type=ComponentType.MLP_CHANNEL,
            layer_index=0,
            scale_factor=1.2,
            target_indices=[0, 4],
            order=2,
        )

        composer = OperationComposer(ModelTopology(num_layers=4, mlp_intermediate_sizes={0: 8}))
        pipeline = composer.compose([op1, op2])

        # Original channels 0 and 4 should be remapped to current channels 0 and 2
        remapped_op2 = pipeline.remapped_operations[1]
        self.assertEqual(remapped_op2.target_indices, [0, 2])
        self.assertEqual(remapped_op2.metadata["original_target_indices"], [0, 4])

    def test_targeting_removed_channel_rejected(self):
        op1 = UnifiedOperation.create_physical_remove(
            target_path="model.layers.0.mlp",
            component_type=ComponentType.MLP_CHANNEL,
            layer_index=0,
            kept_indices=[0, 2, 4, 6],
            order=1,
        )
        op2 = UnifiedOperation.create_scale(
            target_path="model.layers.0.mlp",
            component_type=ComponentType.MLP_CHANNEL,
            layer_index=0,
            scale_factor=1.2,
            target_indices=[1, 4],
            order=2,
        )

        composer = OperationComposer(ModelTopology(num_layers=4, mlp_intermediate_sizes={0: 8}))
        with self.assertRaisesRegex(InvalidCompositionPathError, "has been removed"):
            composer.compose([op1, op2])

    def test_attention_group_slicing_and_remapping(self):
        # 4 KV groups [0, 1, 2, 3], keep [1, 3] -> current [0, 1]
        op1 = UnifiedOperation.create_physical_remove(
            target_path="model.layers.0.self_attn",
            component_type=ComponentType.ATTENTION_GROUP,
            layer_index=0,
            kept_indices=[1, 3],
            order=1,
        )
        op2 = UnifiedOperation.create_scale(
            target_path="model.layers.0.self_attn",
            component_type=ComponentType.ATTENTION_GROUP,
            layer_index=0,
            scale_factor=0.9,
            target_indices=[3],
            order=2,
        )

        composer = OperationComposer(ModelTopology(num_layers=2, num_key_value_heads={0: 4}))
        pipeline = composer.compose([op1, op2])

        remapped_op2 = pipeline.remapped_operations[1]
        self.assertEqual(remapped_op2.target_indices, [1])

    def test_moe_expert_pruning_and_remapping(self):
        # 8 experts [0..7], keep [0, 2, 5, 7] -> current [0, 1, 2, 3]
        op1 = UnifiedOperation.create_physical_remove(
            target_path="model.layers.0.block_sparse_moe",
            component_type=ComponentType.MOE_EXPERT,
            layer_index=0,
            kept_indices=[0, 2, 5, 7],
            order=1,
        )
        op2 = UnifiedOperation.create_scale(
            target_path="model.layers.0.block_sparse_moe",
            component_type=ComponentType.MOE_EXPERT,
            layer_index=0,
            scale_factor=1.05,
            target_indices=[5],
            order=2,
        )

        composer = OperationComposer(ModelTopology(num_layers=2, num_experts={0: 8}))
        pipeline = composer.compose([op1, op2])

        remapped_op2 = pipeline.remapped_operations[1]
        self.assertEqual(remapped_op2.target_indices, [2])

    def test_removing_all_components_rejected(self):
        remapper = IndexRemapper(initial_count=4, domain="layer")
        with self.assertRaisesRegex(InvalidCompositionPathError, "remaining count cannot be 0"):
            remapper.remove_indices([0, 1, 2, 3])

    def test_ambiguous_composition_order_rejected(self):
        op1 = UnifiedOperation.create_scale(
            target_path="model.layers.0.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            scale_factor=0.8,
            order=1,
        )
        op2 = UnifiedOperation.create_scale(
            target_path="model.layers.1.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            scale_factor=0.9,
            order=1,  # Duplicate order index
        )
        composer = OperationComposer(ModelTopology(num_layers=2))
        with self.assertRaisesRegex(InvalidCompositionPathError, "duplicate order indices"):
            composer.compose([op1, op2])

    def test_compose_operations_with_model_topology_discovery(self):
        model = MockToyTransformer()
        op = UnifiedOperation.create_scale(
            target_path="model.layers.0.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            scale_factor=0.5,
        )
        pipeline = compose_operations([op], model=model)
        self.assertEqual(len(pipeline.remapped_operations), 1)


class TestAdapterCapabilityMatrix(unittest.TestCase):
    def setUp(self):
        self.matrix = get_capability_matrix()

    def test_default_architectures_registered(self):
        for arch in ("llama", "mixtral", "qwen", "qwen_moe", "gemma"):
            caps = self.matrix.get(arch)
            self.assertIsNotNone(caps)
            self.assertEqual(caps.architecture, arch)

    def test_architecture_alias_resolution(self):
        self.assertEqual(self.matrix.resolve_architecture_name("mistral"), "llama")
        self.assertEqual(self.matrix.resolve_architecture_name("qwen2"), "qwen")
        self.assertEqual(self.matrix.resolve_architecture_name("qwen3"), "qwen")
        self.assertEqual(self.matrix.resolve_architecture_name("gemma2"), "gemma")
        self.assertEqual(self.matrix.resolve_architecture_name("gemma3_text"), "gemma")

    def test_llama_supports_mlp_attention_layers_weights(self):
        op_mlp = UnifiedOperation.create_physical_remove(
            target_path="model.layers.0.mlp",
            component_type=ComponentType.MLP_CHANNEL,
            layer_index=0,
            kept_indices=[0, 1],
        )
        op_attn = UnifiedOperation.create_scale(
            target_path="model.layers.0.self_attn",
            component_type=ComponentType.ATTENTION_GROUP,
            scale_factor=1.1,
        )
        op_layer = UnifiedOperation.create_physical_remove(
            target_path="model.layers.1",
            component_type=ComponentType.FULL_LAYER,
            layer_index=1,
        )
        res = check_adapter_capability("llama", [op_mlp, op_attn, op_layer])
        self.assertTrue(res.supported)
        self.assertEqual(len(res.violations), 0)
        # Should not raise
        validate_adapter_capability("llama", [op_mlp, op_attn, op_layer])

    def test_llama_rejects_moe_operations(self):
        op_moe = UnifiedOperation.create_physical_remove(
            target_path="model.layers.0.block_sparse_moe",
            component_type=ComponentType.MOE_EXPERT,
            layer_index=0,
            kept_indices=[0, 1],
        )
        res = check_adapter_capability("llama", [op_moe])
        self.assertFalse(res.supported)
        self.assertEqual(len(res.violations), 1)
        self.assertIn("does not support component type 'moe_expert'", res.violations[0])

        with self.assertRaises(UnsupportedOperationError):
            validate_adapter_capability("llama", [op_moe])

    def test_mixtral_supports_moe_operations(self):
        op_moe = UnifiedOperation.create_physical_remove(
            target_path="model.layers.0.block_sparse_moe",
            component_type=ComponentType.MOE_EXPERT,
            layer_index=0,
            kept_indices=[0, 1],
        )
        res = check_adapter_capability("mixtral", [op_moe])
        self.assertTrue(res.supported)

    def test_gemma_supports_dense_rejects_moe(self):
        op_mlp = UnifiedOperation.create_scale(
            target_path="model.layers.0.mlp",
            component_type=ComponentType.MLP_CHANNEL,
            scale_factor=0.8,
        )
        op_moe = UnifiedOperation.create_physical_remove(
            target_path="model.layers.0.moe",
            component_type=ComponentType.MOE_EXPERT,
            layer_index=0,
            kept_indices=[0, 1],
        )
        res_mlp = check_adapter_capability("gemma", [op_mlp])
        self.assertTrue(res_mlp.supported)

        res_moe = check_adapter_capability("gemma", [op_moe])
        self.assertFalse(res_moe.supported)

    def test_custom_architecture_registration(self):
        custom_registry = AdapterCapabilityRegistry()
        custom_registry.register(
            ArchitectureCapabilities(
                architecture="custom_net",
                adapter_name="custom_adapter",
                supported_operations={OperationKind.SCALE, OperationKind.MASK},
                supported_components={ComponentType.WEIGHT_TENSOR},
            )
        )
        op_scale = UnifiedOperation.create_scale(
            target_path="weight_0",
            component_type=ComponentType.WEIGHT_TENSOR,
            scale_factor=0.5,
        )
        op_clamp = UnifiedOperation.create_clamp(
            target_path="weight_0",
            component_type=ComponentType.WEIGHT_TENSOR,
            min_value=-1.0,
        )
        res_scale = custom_registry.check_plan_support("custom_net", [op_scale])
        self.assertTrue(res_scale.supported)

        res_clamp = custom_registry.check_plan_support("custom_net", [op_clamp])
        self.assertFalse(res_clamp.supported)

    def test_model_instance_capability_lookup(self):
        model = MockToyTransformer()
        op = UnifiedOperation.create_scale(
            target_path="model.layers.0.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            scale_factor=1.0,
        )
        res = check_adapter_capability(model, [op])
        self.assertTrue(res.supported)
        self.assertEqual(res.architecture, "llama")


class TestPlanExecutor(unittest.TestCase):
    def test_plan_executor_scale_persistent(self):
        model = MockToyTransformer()
        x = torch.randn(2, 4)
        orig_out = model(x)
        orig_weight = model.model.layers[0].mlp.down_proj.weight.data.clone()

        op = UnifiedOperation.create_scale(
            target_path="model.layers.0.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            scale_factor=0.5,
            execution_semantics=ExecutionSemantics.PERSISTENT,
        )
        result = apply_plan(model, [op])
        self.assertTrue(result.success)
        self.assertIn("model.layers.0.mlp.down_proj.weight", result.modified_parameters)
        self.assertTrue(torch.allclose(model.model.layers[0].mlp.down_proj.weight.data, orig_weight * 0.5))

        new_out = model(x)
        self.assertEqual(new_out.shape, (2, 4))
        self.assertFalse(torch.allclose(orig_out, new_out))

    def test_plan_executor_scale_runtime(self):
        model = MockToyTransformer()
        x = torch.randn(2, 4)
        orig_out = model(x)
        orig_param_data = model.model.layers[0].mlp.down_proj.weight.data.clone()

        op = UnifiedOperation.create_scale(
            target_path="model.layers.0.mlp",
            component_type=ComponentType.MLP_CHANNEL,
            scale_factor=0.1,
            execution_semantics=ExecutionSemantics.RUNTIME_ONLY,
            export_supported=False,
        )

        with apply_plan(model, [op]) as result:
            self.assertTrue(result.success)
            self.assertIsNotNone(result.runtime_context)
            self.assertTrue(result.runtime_context.is_active)
            runtime_out = model(x)
            self.assertFalse(torch.allclose(orig_out, runtime_out))
            self.assertTrue(torch.equal(model.model.layers[0].mlp.down_proj.weight.data, orig_param_data))

        reverted_out = model(x)
        self.assertTrue(torch.allclose(orig_out, reverted_out))

    def test_plan_executor_clamp_persistent_and_runtime(self):
        # Persistent clamp
        model1 = MockToyTransformer()
        op_clamp_p = UnifiedOperation.create_clamp(
            target_path="model.layers.0.mlp.gate_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            min_value=-0.01,
            max_value=0.01,
            execution_semantics=ExecutionSemantics.PERSISTENT,
            export_supported=True,
        )
        res_p = apply_plan(model1, [op_clamp_p])
        self.assertTrue(res_p.success)
        w = model1.model.layers[0].mlp.gate_proj.weight.data
        self.assertTrue(torch.all(w >= -0.010001))
        self.assertTrue(torch.all(w <= 0.010001))

        # Runtime clamp
        model2 = MockToyTransformer()
        x = torch.randn(2, 4)
        op_clamp_r = UnifiedOperation.create_clamp(
            target_path="model.layers.0.mlp",
            component_type=ComponentType.MLP_CHANNEL,
            min_value=-0.05,
            max_value=0.05,
            execution_semantics=ExecutionSemantics.RUNTIME_ONLY,
            export_supported=False,
        )
        with apply_plan(model2, [op_clamp_r]):
            out = model2(x)
            self.assertEqual(out.shape, (2, 4))

    def test_plan_executor_mask_persistent_and_runtime(self):
        # Persistent mask on targeted channels
        model1 = MockToyTransformer()
        op_mask_p = UnifiedOperation.create_mask(
            target_path="model.layers.0.mlp.gate_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            target_indices=[0, 1],
            mask_value=0.0,
            execution_semantics=ExecutionSemantics.PERSISTENT,
            export_supported=True,
        )
        res_p = apply_plan(model1, [op_mask_p])
        self.assertTrue(res_p.success)
        w = model1.model.layers[0].mlp.gate_proj.weight.data
        self.assertTrue(torch.all(w[0] == 0.0))
        self.assertTrue(torch.all(w[1] == 0.0))

        # Runtime mask
        model2 = MockToyTransformer()
        x = torch.randn(2, 4)
        orig_out = model2(x)
        op_mask_r = UnifiedOperation.create_mask(
            target_path="model.layers.0.mlp",
            component_type=ComponentType.MLP_CHANNEL,
            target_indices=[0, 1],
            mask_value=0.0,
            execution_semantics=ExecutionSemantics.RUNTIME_ONLY,
            export_supported=False,
        )
        with apply_plan(model2, [op_mask_r]):
            out = model2(x)
            self.assertFalse(torch.allclose(orig_out, out))

    def test_plan_executor_project_persistent(self):
        model = MockToyTransformer()
        w_before = model.model.layers[0].self_attn.o_proj.weight.data.clone()
        direction = torch.randn(w_before.shape[1]).tolist()

        op_proj = UnifiedOperation.create_project(
            target_path="model.layers.0.self_attn.o_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            direction=direction,
            preserve_subspace=False,
            execution_semantics=ExecutionSemantics.PERSISTENT,
            export_supported=True,
        )
        res = apply_plan(model, [op_proj])
        self.assertTrue(res.success)
        w_after = model.model.layers[0].self_attn.o_proj.weight.data
        self.assertFalse(torch.allclose(w_before, w_after))

    def test_plan_executor_replace_persistent_and_runtime(self):
        model = MockToyTransformer()
        op_rep = UnifiedOperation.create_replace(
            target_path="model.layers.0.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            replacement_value=0.42,
            target_indices=[0],
            execution_semantics=ExecutionSemantics.PERSISTENT,
        )
        res = apply_plan(model, [op_rep])
        self.assertTrue(res.success)
        w = model.model.layers[0].mlp.down_proj.weight.data
        self.assertTrue(torch.allclose(w[0], torch.tensor(0.42)))

    def test_plan_executor_low_rank_delta_persistent(self):
        model = MockToyTransformer()
        w_before = model.model.layers[0].mlp.down_proj.weight.data.clone()
        out_dim, in_dim = w_before.shape
        rank = 2
        u = torch.randn(out_dim, rank).tolist()
        v = torch.randn(rank, in_dim).tolist()

        op_lora = UnifiedOperation.create_low_rank_delta(
            target_path="model.layers.0.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            rank=rank,
            alpha=1.0,
            u=u,
            v=v,
        )
        res = apply_plan(model, [op_lora])
        self.assertTrue(res.success)
        expected_delta = (1.0 / rank) * (torch.tensor(u) @ torch.tensor(v))
        w_after = model.model.layers[0].mlp.down_proj.weight.data
        self.assertTrue(torch.allclose(w_after, w_before + expected_delta, atol=1e-5))

    def test_plan_executor_physical_remove_layer(self):
        model = MockToyTransformer(num_layers=4)
        x = torch.randn(2, 4)
        self.assertEqual(len(model.model.layers), 4)

        op_drop = UnifiedOperation.create_physical_remove(
            target_path="model.layers.1",
            component_type=ComponentType.FULL_LAYER,
            layer_index=1,
        )
        res = apply_plan(model, [op_drop])
        self.assertTrue(res.success)
        self.assertEqual(len(model.model.layers), 3)
        self.assertEqual(model.config.num_hidden_layers, 3)
        out = model(x)
        self.assertEqual(out.shape, (2, 4))

    def test_plan_executor_physical_remove_mlp_channels(self):
        model = MockToyTransformer()
        x = torch.randn(2, 4)
        mlp = model.model.layers[0].mlp
        self.assertEqual(mlp.gate_proj.weight.shape[0], 8)

        op_mlp = UnifiedOperation.create_physical_remove(
            target_path="model.layers.0.mlp",
            component_type=ComponentType.MLP_CHANNEL,
            layer_index=0,
            kept_indices=[0, 2, 4, 6],
        )
        res = apply_plan(model, [op_mlp])
        self.assertTrue(res.success)
        self.assertEqual(mlp.gate_proj.weight.shape[0], 4)
        self.assertEqual(mlp.up_proj.weight.shape[0], 4)
        self.assertEqual(mlp.down_proj.weight.shape[1], 4)
        self.assertEqual(model.config.intermediate_size, 4)

        out = model(x)
        self.assertEqual(out.shape, (2, 4))

    def test_plan_executor_physical_remove_attention_groups(self):
        model = MockToyTransformer()
        x = torch.randn(2, 4)
        attn = model.model.layers[0].self_attn
        self.assertEqual(attn.num_heads, 4)
        self.assertEqual(attn.num_key_value_heads, 2)

        op_attn = UnifiedOperation.create_physical_remove(
            target_path="model.layers.0.self_attn",
            component_type=ComponentType.ATTENTION_GROUP,
            layer_index=0,
            kept_indices=[1],
        )
        res = apply_plan(model, [op_attn])
        self.assertTrue(res.success)
        self.assertEqual(attn.num_heads, 2)
        self.assertEqual(attn.num_key_value_heads, 1)
        self.assertEqual(attn.q_proj.weight.shape[0], 4)
        self.assertEqual(attn.k_proj.weight.shape[0], 2)
        self.assertEqual(attn.v_proj.weight.shape[0], 2)
        self.assertEqual(attn.o_proj.weight.shape[1], 4)

        out = model(x)
        self.assertEqual(out.shape, (2, 4))

    def test_plan_executor_physical_remove_moe_experts(self):
        model = MockMoETransformer(num_layers=2, num_experts=8)
        x = torch.randn(2, 4)
        self.assertEqual(len(model.model.layers[0].block_sparse_moe.experts), 8)

        op_moe = UnifiedOperation.create_physical_remove(
            target_path="model.layers.0.block_sparse_moe",
            component_type=ComponentType.MOE_EXPERT,
            layer_index=0,
            kept_indices=[0, 2, 5, 7],
        )
        res = apply_plan(model, [op_moe])
        self.assertTrue(res.success)
        moe = model.model.layers[0].block_sparse_moe
        self.assertEqual(len(moe.experts), 4)
        self.assertEqual(moe.router.weight.shape[0], 4)
        self.assertEqual(model.config.num_experts, 4)

        new_out = model(x)
        self.assertEqual(new_out.shape, (2, 4))

    def test_plan_executor_composed_pipeline_end_to_end(self):
        model = MockToyTransformer(num_layers=4)
        x = torch.randn(2, 4)

        # 1. Remove layer 1
        op1 = UnifiedOperation.create_physical_remove(
            target_path="model.layers.1",
            component_type=ComponentType.FULL_LAYER,
            layer_index=1,
            order=1,
        )
        # 2. Slice MLP channels on original layer 2 (which becomes layer 1)
        op2 = UnifiedOperation.create_physical_remove(
            target_path="model.layers.2.mlp",
            component_type=ComponentType.MLP_CHANNEL,
            layer_index=2,
            kept_indices=[0, 2, 4, 6],
            order=2,
        )
        # 3. Slice Attention groups on original layer 3 (which becomes layer 2)
        op3 = UnifiedOperation.create_physical_remove(
            target_path="model.layers.3.self_attn",
            component_type=ComponentType.ATTENTION_GROUP,
            layer_index=3,
            kept_indices=[0],
            order=3,
        )
        # 4. Scale down_proj on original layer 0
        op4 = UnifiedOperation.create_scale(
            target_path="model.layers.0.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            layer_index=0,
            scale_factor=0.8,
            order=4,
        )

        plan = UnifiedPlan(version=4, model_type="llama", operations=[op1, op2, op3, op4])
        result = apply_plan(model, plan)

        self.assertTrue(result.success)
        self.assertEqual(len(model.model.layers), 3)
        self.assertEqual(model.model.layers[1].mlp.gate_proj.weight.shape[0], 4)
        self.assertEqual(model.model.layers[2].self_attn.num_heads, 2)

        out = model(x)
        self.assertEqual(out.shape, (2, 4))

    def test_plan_executor_tied_weight_destruction_aborts_before_write(self):
        model = MockTiedModel()
        orig_weight = model.lm_head.weight.data.clone()

        op = UnifiedOperation.create_physical_remove(
            target_path="lm_head.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            target_indices=[0, 1],
        )

        with self.assertRaises(ConflictingOperationsError):
            apply_plan(model, [op])

        self.assertTrue(torch.equal(model.lm_head.weight.data, orig_weight))

    def test_plan_executor_unsupported_architecture_aborts_before_write(self):
        model = MockToyTransformer()
        orig_weight = model.model.layers[0].mlp.down_proj.weight.data.clone()

        op_unsupported = UnifiedOperation.create_physical_remove(
            target_path="model.layers.0.block_sparse_moe",
            component_type=ComponentType.MOE_EXPERT,
            layer_index=0,
            kept_indices=[0, 1],
        )

        with self.assertRaises(UnsupportedOperationError):
            apply_plan(model, [op_unsupported])

        self.assertTrue(torch.equal(model.model.layers[0].mlp.down_proj.weight.data, orig_weight))

    def test_plan_executor_invalid_composition_path_aborts_before_write(self):
        model = MockToyTransformer()
        orig_num_layers = len(model.model.layers)

        op1 = UnifiedOperation.create_physical_remove(
            target_path="model.layers.1",
            component_type=ComponentType.FULL_LAYER,
            layer_index=1,
            order=1,
        )
        op2 = UnifiedOperation.create_scale(
            target_path="model.layers.1.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            layer_index=1,
            scale_factor=0.5,
            order=2,
        )

        # Pre-flight conflict detector catches overlapping mutation on removed target
        with self.assertRaises((ConflictingOperationsError, InvalidCompositionPathError)):
            apply_plan(model, [op1, op2])

        # Even with validate_conflicts=False, OperationComposer catches invalid path
        executor = PlanExecutor(validate_conflicts=False)
        with self.assertRaises(InvalidCompositionPathError):
            executor.apply(model, [op1, op2])

        self.assertEqual(len(model.model.layers), orig_num_layers)

    def test_plan_executor_unselected_parameters_remain_strictly_identical(self):
        model = MockToyTransformer()
        initial_params = {name: p.data.clone() for name, p in model.named_parameters()}

        op = UnifiedOperation.create_scale(
            target_path="model.layers.0.mlp.down_proj.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            scale_factor=0.77,
        )
        apply_plan(model, [op])

        for name, p in model.named_parameters():
            if name == "model.layers.0.mlp.down_proj.weight":
                self.assertFalse(torch.equal(p.data, initial_params[name]))
            else:
                self.assertTrue(torch.equal(p.data, initial_params[name]), f"Parameter {name} was unexpectedly modified")

    def test_plan_executor_save_reload_parity(self):
        model = MockToyTransformer(num_layers=4)
        x = torch.randn(2, 4)

        op1 = UnifiedOperation.create_physical_remove(
            target_path="model.layers.1",
            component_type=ComponentType.FULL_LAYER,
            layer_index=1,
            order=1,
        )
        op2 = UnifiedOperation.create_physical_remove(
            target_path="model.layers.2.mlp",
            component_type=ComponentType.MLP_CHANNEL,
            layer_index=2,
            kept_indices=[0, 2, 4, 6],
            order=2,
        )
        apply_plan(model, [op1, op2])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model_state.pt"
            torch.save(model.state_dict(), save_path)

            reloaded_model = MockToyTransformer(num_layers=3)
            # Reconfigure layer 1 MLP intermediate size to 4 to receive sliced weights
            class ReconfiguredMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gate_proj = nn.Linear(4, 4, bias=False)
                    self.up_proj = nn.Linear(4, 4, bias=False)
                    self.down_proj = nn.Linear(4, 4, bias=False)

                def forward(self, inp):
                    return self.down_proj(self.gate_proj(inp) * self.up_proj(inp))

            reloaded_model.model.layers[1].mlp = ReconfiguredMLP()

            state = torch.load(save_path, map_location="cpu", weights_only=True)
            reloaded_model.load_state_dict(state)

            out_reloaded = reloaded_model(x)
            self.assertEqual(out_reloaded.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
