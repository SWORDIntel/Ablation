"""Apply Stage 4B alias and composition checks to resolved core geometry."""
from .stage4b_contract import (UnifiedOperation, OperationKind, ComponentType,
    ExecutionSemantics, validate_no_conflicts, OperationComposer, ModelTopology)


def validate_core_contract(model, resolved):
    operations = []
    for order, item in enumerate(resolved):
        directional = item['kind'] == 'directional_weight_edit'
        full_layer = item['kind'] == 'transformer_layer_remove'
        operations.append(UnifiedOperation(
            kind=OperationKind.PROJECT if directional else OperationKind.PHYSICAL_REMOVE,
            target_path=item.get('parameter') or item['module'],
            component_type=ComponentType.FULL_LAYER if full_layer else ComponentType.WEIGHT_TENSOR,
            execution_semantics=ExecutionSemantics.PERSISTENT, export_supported=True,
            layer_index=item['layer'], operation_id=f'core_{order}', order=order,
            parameters={'direction': 'checksummed profile'} if directional else
                       {'kept_indices': item.get('kept_indices', [])},
        ))
    validate_no_conflicts(operations, model=model)
    composed = OperationComposer(ModelTopology.from_model(model)).compose(operations)
    return dict(operations=[op.to_dict() for op in composed.remapped_operations],
                layer_remapping=composed.original_to_current_layer_map)


def preview_unified_plan(model, payload, plan_path):
    """Resolve arbitrary Stage 4B composition on an isolated copy before mutation."""
    import copy
    from .stage4b_contract import UnifiedPlan, apply_plan
    from .common import file_sha256, parameter_bytes
    if any(payload.get(key) for key in ("ablation", "structured", "drop_layers")):
        raise ValueError("Do not mix unified operations with fixed-order core plan sections")
    plan = UnifiedPlan.from_dict(payload)
    if any(op.execution_semantics != ExecutionSemantics.PERSISTENT or not op.export_supported
           for op in plan.operations):
        raise ValueError("Checkpoint application requires persistent, exportable operations")
    candidate = copy.deepcopy(model)
    result = apply_plan(candidate, plan)
    if not result.success:
        raise ValueError("Unified plan preflight failed")
    report = dict(version=1, model_type=getattr(getattr(model, 'config', None), 'model_type', type(model).__name__),
                  source_model_bytes=parameter_bytes(model), plan=str(plan_path),
                  plan_sha256=file_sha256(plan_path), operation_order=[op.operation_id for op in result.applied_operations],
                  operations=[op.to_dict() for op in result.applied_operations],
                  contract=dict(modified_parameters=result.modified_parameters, details=result.details),
                  summary=dict(operation_count=len(result.applied_operations),
                               parameters_removed=sum(p.numel() for p in model.parameters()) -
                                                  sum(p.numel() for p in candidate.parameters()),
                               estimated_bytes_removed=parameter_bytes(model) - parameter_bytes(candidate)))
    del candidate
    return report
