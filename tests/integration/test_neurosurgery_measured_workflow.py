"""Offline HF integration, explicitly separate from trained-model qualification."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
import yaml
from transformers import LlamaForCausalLM

from aegis_lab.editing.neurosurgery.cli_extended import main, build_extended_parser
from aegis_lab.editing.neurosurgery.operator_workflow import run_operator
from aegis_lab.editing.neurosurgery.measured import bind_files, verify_binding, recover_model
from aegis_lab.editing.neurosurgery.pipeline_runner import (
    CampaignPipelineRunner, CampaignConfig, ModelConfig, CampaignThresholds, ThresholdGateError)
from aegis_lab.editing.neurosurgery.preview import build_preview
from aegis_lab.editing.neurosurgery.apply import apply_to_model
from aegis_lab.editing.neurosurgery.stage4b_contract import ConflictingOperationsError
from tests.unit.test_neurosurgery_pipeline_runner import setup_test_campaign_env


class TestMeasuredWorkflow(unittest.TestCase):
    def test_reloadable_workflow_recovery_and_heldout_validation(self):
        with tempfile.TemporaryDirectory() as tmp, torch.random.fork_rng(devices=[]):
            root = Path(tmp)
            torch.manual_seed(42)
            model, base, keep, drop, _ = setup_test_campaign_env(root)
            heldout = root / 'heldout.txt'
            heldout.write_text('keep prompt 16\nkeep prompt 17\n')
            plan = root / 'plan.yaml'
            plan.write_text(yaml.safe_dump(dict(version=4, drop_layers=[1], structured={}, ablation={})))
            args = build_extended_parser().parse_args(['workflow', '--model', str(base),
                '--out', str(root / 'run'), '--keep', str(keep), '--drop', str(drop),
                '--validation-keep', str(heldout), '--plan', str(plan),
                '--stages', 'inspect,plan,apply,recover,validate,export', '--max-recovery-steps', '2'])
            report = run_operator(args, {'max_mean_kl': 10})
            self.assertTrue(report['heldout_validation_passed'])
            self.assertFalse(report['qualified'])
            self.assertEqual(report['completed_steps']['recover']['training_steps'], 2)
            reloaded = LlamaForCausalLM.from_pretrained(root / 'run' / 'exported_runtime')
            self.assertEqual(reloaded.config.num_hidden_layers, 2)
            inputs = model.tokenizer('keep prompt 2', return_tensors='pt')
            with torch.no_grad():
                output = reloaded.generate(**inputs, max_new_tokens=2, use_cache=True)
            self.assertGreater(output.shape[1], inputs['input_ids'].shape[1])
            verify_binding(json.loads((root / 'run' / 'workflow_manifest.json').read_text())['binding'])

    def test_dry_run_requires_real_geometry_and_never_writes_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, base, _, _, _ = setup_test_campaign_env(root)
            plan = root / 'plan.yaml'
            plan.write_text(yaml.safe_dump(dict(version=3, drop_layers=[99])))
            args = build_extended_parser().parse_args(['workflow', '--model', str(base),
                '--out', str(root / 'run'), '--plan', str(plan), '--stages', 'plan', '--dry-run'])
            with self.assertRaises(IndexError):
                run_operator(args, {})
            self.assertFalse((root / 'run' / 'candidate').exists())

    def test_campaign_resume_rejects_changed_data(self):
        from aegis_lab.editing.neurosurgery.pipeline_runner import DatasetSpec, StepConfig
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, base, keep, drop, _ = setup_test_campaign_env(root)
            cfg = CampaignConfig(campaign_id="binding", output_dir=str(root / "run"),
                model=ModelConfig(path=str(base), model_instance=model),
                datasets={"keep": DatasetSpec(name="keep", path=str(keep), kind="keep"),
                          "drop": DatasetSpec(name="drop", path=str(drop), kind="drop")},
                steps=[StepConfig(name="profile")])
            CampaignPipelineRunner(cfg).run()
            keep.write_text(keep.read_text() + '{"text": "new prompt"}\n')
            with self.assertRaisesRegex(ValueError, "changed"):
                CampaignPipelineRunner(cfg).run(resume=True)

    def test_noop_restoration_manifest_requires_identical_hashes(self):
        from aegis_lab.editing.neurosurgery.stage4a_provenance import RestorationManifest
        manifest = RestorationManifest(source_checkpoint="source", candidate_checkpoint="candidate",
            source_hashes={"weight": "same"}, candidate_hashes={"weight": "same"}, edited_tensors=[])
        self.assertEqual(manifest.edited_tensors, [])
        with self.assertRaises(ValueError):
            RestorationManifest(source_checkpoint="source", candidate_checkpoint="candidate",
                source_hashes={"weight": "same"}, candidate_hashes={"weight": "changed"}, edited_tensors=[])

    def test_input_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'keep.txt'
            p.write_text('one two')
            binding = bind_files({'keep': p})
            p.write_text('other bytes')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_binding(binding)

    def test_missing_or_nonfinite_required_campaign_metrics_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = CampaignConfig(campaign_id='gates', output_dir=tmp,
                                 model=ModelConfig(path=tmp),
                                 thresholds=CampaignThresholds(max_mean_kl=0.02))
            runner = CampaignPipelineRunner(cfg)
            with self.assertRaises(ThresholdGateError):
                runner._evaluate_thresholds()
            runner.metrics['quantize'] = {'mean_kl_drift': float('nan')}
            with self.assertRaises(ThresholdGateError):
                runner._evaluate_thresholds()

    def test_tied_structural_targets_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, _, _, _, _ = setup_test_campaign_env(root)
            model.model.layers[1].mlp = model.model.layers[0].mlp
            plan = root / 'plan.yaml'
            plan.write_text(yaml.safe_dump(dict(version=3, drop_layers=[0])))
            before = list(model.model.layers)
            with self.assertRaisesRegex(ConflictingOperationsError, 'tied|shared|alias'):
                apply_to_model(model, None, str(plan))
            self.assertEqual(list(model.model.layers), before)

    def test_middle_layer_drop_updates_cache_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, _, _, _, _ = setup_test_campaign_env(root)
            plan = root / 'plan.yaml'
            plan.write_text(yaml.safe_dump(dict(version=3, drop_layers=[1])))
            preview = build_preview(model, str(plan))
            self.assertEqual(preview['contract']['layer_remapping'], {0: 0, 1: None, 2: 1})
            apply_to_model(model, None, str(plan))
            self.assertEqual([l.self_attn.layer_idx for l in model.model.layers], [0, 1])
            with torch.no_grad():
                model.generate(**model.tokenizer('keep prompt 1', return_tensors='pt'),
                               max_new_tokens=2, use_cache=True)

    def test_unified_plan_remaps_original_layer_targets(self):
        from aegis_lab.editing.neurosurgery.stage4b_contract import (
            UnifiedPlan, UnifiedOperation, OperationKind, ComponentType, ExecutionSemantics)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, _, _, _, _ = setup_test_campaign_env(root)
            original = model.model.layers[2].mlp.down_proj.weight.detach().clone()
            plan = UnifiedPlan(operations=[
                UnifiedOperation(kind=OperationKind.PHYSICAL_REMOVE, target_path="model.layers.0",
                    component_type=ComponentType.FULL_LAYER, execution_semantics=ExecutionSemantics.PERSISTENT,
                    export_supported=True, layer_index=0, order=0),
                UnifiedOperation(kind=OperationKind.SCALE, target_path="model.layers.2.mlp.down_proj.weight",
                    component_type=ComponentType.WEIGHT_TENSOR, execution_semantics=ExecutionSemantics.PERSISTENT,
                    export_supported=True, layer_index=2, parameters={"scale_factor": 0.5}, order=1),
            ])
            path = root / "unified.yaml"
            plan.to_yaml(path)
            preview = build_preview(model, str(path))
            self.assertEqual(preview["operations"][1]["layer_index"], 1)
            self.assertTrue(torch.equal(model.model.layers[2].mlp.down_proj.weight, original))
            apply_to_model(model, None, str(path))
            self.assertTrue(torch.equal(model.model.layers[1].mlp.down_proj.weight, original * 0.5))
            self.assertEqual([layer.self_attn.layer_idx for layer in model.model.layers], [0, 1])

    def test_unified_runtime_hooks_cannot_be_exported_as_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, _, _, _, _ = setup_test_campaign_env(root)
            path = root / "runtime.yaml"
            path.write_text(yaml.safe_dump({"version": 4, "operations": [{"kind": "scale",
                "target_path": "model.layers.0.mlp.down_proj.weight", "component_type": "weight_tensor",
                "execution_semantics": "runtime_only", "export_supported": False,
                "parameters": {"scale_factor": 0.5}}]}))
            with self.assertRaisesRegex(ValueError, "persistent"):
                build_preview(model, str(path))

    def test_change_targets_are_supervised_and_prompts_are_masked(self):
        from aegis_lab.editing.neurosurgery.measured import training_batches, load_change_examples
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, _, _, _, _ = setup_test_campaign_env(root)
            change = root / "change.json"
            change.write_text(json.dumps([{"prompt": "keep prompt", "target": "drop 2"}]))
            batch = training_batches(model.tokenizer, model, load_change_examples(change))[0]
            self.assertEqual(batch["labels"][0, :2].tolist(), [-100, -100])
            self.assertEqual(batch["labels"][0, 2:].tolist(), model.tokenizer.encode("drop 2", add_special_tokens=False))
            self.assertFalse(torch.equal(batch["labels"], batch["input_ids"]))

    def test_library_rollback_restores_structure_and_original_weights(self):
        from aegis_lab.editing.neurosurgery.workflow import NeurosurgeryWorkflow, WorkflowConfig
        from aegis_lab.editing.neurosurgery.stage4a_provenance import EvaluationReport, SlicedMetricsReport
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, _, _, _, _ = setup_test_campaign_env(root)
            original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
            plan = root / "plan.yaml"
            plan.write_text(yaml.safe_dump(dict(version=3, drop_layers=[1])))
            def edit(current):
                apply_to_model(current, None, str(plan))
            def evaluate(current, split):
                score = 1.0 if current.config.num_hidden_layers == 3 else 0.5
                def report(kind, value):
                    return SlicedMetricsReport(kind=kind, sample_count=1, overall_score=value,
                        worst_slice_score=value, worst_slice_damage=1 - value,
                        worst_slice_domain="fixture", domain_metrics={})
                return EvaluationReport(keep_report=report("keep", score), drop_report=report("drop", 1.0))
            workflow = NeurosurgeryWorkflow(WorkflowConfig(model=model, output_dir=root / "workflow",
                candidate_editor_fn=edit, custom_evaluator=evaluate, enable_export=False))
            result = workflow.run()
            self.assertFalse(result.success)
            self.assertTrue(result.rollback_report.restored)
            self.assertEqual(model.config.num_hidden_layers, 3)
            self.assertEqual(set(model.state_dict()), set(original))
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, original[name]))

    def test_installed_operator_chain_binds_producers_and_rejects_tampering(self):
        from aegis_lab.editing.neurosurgery.cli import main as core_main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, source, keep, drop, _ = setup_test_campaign_env(root)
            core_main(['profile-mlp', '--model', str(source), '--keep', str(keep), '--drop', str(drop),
                       '--out', str(root / 'profile'), '--device', 'cpu'])
            profile = root / 'profile' / 'mlp_profile.pt'
            audit = json.loads((root / 'profile' / 'operator_audit.json').read_text())
            self.assertEqual(audit['status'], 'completed')
            core_main(['optimize', '--model', str(source), '--keep', str(keep), '--mlp-profile', str(profile),
                       '--mlp-ratios', '1,0.5', '--max-mean-kl', '10', '--min-top1-agreement', '0',
                       '--max-trials', '4', '--out', str(root / 'opt'), '--device', 'cpu'])
            audit = json.loads((root / 'opt' / 'operator_audit.json').read_text())
            self.assertIn('mlp_profile', audit['upstream'])
            plan = root / 'opt' / 'optimized_plan.yaml'
            core_main(['apply', '--model', str(source), '--plan', str(plan),
                       '--out', str(root / 'candidate'), '--device', 'cpu'])
            audit = json.loads((root / 'candidate' / 'operator_audit.json').read_text())
            self.assertEqual(audit['unbound_legacy_artifacts'], [])
            self.assertIn('plan', audit['upstream'])
            profile.write_bytes(profile.read_bytes() + b'tampered')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                core_main(['search-mlp', '--model', str(source), '--keep', str(keep), '--profile', str(profile),
                           '--out', str(root / 'rejected'), '--device', 'cpu'])
            self.assertFalse((root / 'rejected').exists())

    def test_audit_preserves_model_lineage_through_model_free_plan_generation(self):
        from argparse import Namespace
        from aegis_lab.editing.neurosurgery.operator_audit import audited_dispatch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            (source / 'weights').write_bytes(b'original')
            profile = root / 'profile.json'
            def write_profile(args):
                Path(args.out).write_text('{}')
            audited_dispatch(Namespace(cmd='profile', model=str(source), out=str(profile)), write_profile)
            plan = root / 'plan.yaml'
            def write_plan(args):
                Path(args.out).write_text('version: 4\ndrop_layers: []\n')
            audited_dispatch(Namespace(cmd='plan', profile=str(profile), out=str(plan)), write_plan)
            audit = json.loads(plan.with_name(plan.name + '.operator_audit.json').read_text())
            self.assertEqual(audit['source_model_binding']['files'], bind_files({'model': source})['model']['files'])
            (source / 'weights').write_bytes(b'changed')
            called = []
            with self.assertRaisesRegex(ValueError, 'model binding mismatch'):
                audited_dispatch(Namespace(cmd='apply', model=str(source), plan=str(plan),
                    out=str(root / 'candidate')), lambda args: called.append(args))
            self.assertEqual(called, [])
            self.assertFalse((root / 'candidate').exists())

    def test_boolean_benchmark_option_is_audited_without_becoming_an_input_path(self):
        from argparse import Namespace
        from aegis_lab.editing.neurosurgery.operator_audit import audited_dispatch
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'export'
            def export(args):
                Path(args.out).mkdir()
                (Path(args.out) / 'result.json').write_text('{}')
            audited_dispatch(Namespace(cmd='export-runtime', benchmark=True, out=str(output)), export)
            audit = json.loads((output / 'operator_audit.json').read_text())
            self.assertTrue(audit['options']['benchmark'])
            self.assertNotIn('benchmark', audit['inputs'])
            self.assertEqual(audit['status'], 'completed')

    def test_tokenizer_binding_includes_processing_and_chat_semantics(self):
        from aegis_lab.editing.neurosurgery.stage4a_provenance import compute_tokenizer_hash
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, _, _, _, _ = setup_test_campaign_env(root)
            tokenizer = model.tokenizer
            initial = compute_tokenizer_hash(tokenizer)
            tokenizer(['keep prompt', 'keep'], padding=True, truncation=True)
            self.assertEqual(compute_tokenizer_hash(tokenizer), initial)
            tokenizer.chat_template = '{{messages}}'
            self.assertNotEqual(compute_tokenizer_hash(tokenizer), initial)

    def test_advanced_dispatches_core_commands(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, source, _, _, _ = setup_test_campaign_env(root)
            with patch('aegis_lab.editing.neurosurgery.inventory.run_inventory', return_value={}) as inventory:
                self.assertEqual(main(['inventory', '--model', str(source), '--out', str(root / "inventory"), '--device', 'cpu']), 0)
                inventory.assert_called_once()
            audit = json.loads((root / "inventory" / "operator_audit.json").read_text())
            self.assertEqual(audit["status"], "completed")
            self.assertIn("model", audit["inputs"])



if __name__ == '__main__':
    unittest.main()
