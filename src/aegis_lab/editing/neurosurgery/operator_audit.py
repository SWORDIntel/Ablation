"""Mandatory byte-level audit for installed operator commands."""
import datetime
import json
import os
from pathlib import Path
import uuid

from .common import file_sha256, LOG
from .measured import bind_files, verify_binding

INPUT_ROLES = {
    'model', 'base', 'candidate', 'teacher', 'baseline_model', 'source_dir',
    'keep', 'drop', 'change', 'distill', 'validation_keep', 'profile', 'plan',
    'search', 'greedy_search', 'mlp_profile', 'mlp_search', 'attention_profile',
    'attention_search', 'moe_profile', 'moe_search', 'selectors', 'config',
    'workload', 'benchmark', 'calibration_data', 'search_space',
}
MODEL_ROLES = {'model', 'base', 'candidate', 'teacher', 'baseline_model'}


def _audit_path(output, run_id):
    if output:
        path = Path(output)
        # The command determines whether --out is a file or a directory.
        if path.is_file():
            return path.with_name(path.name + '.operator_audit.json')
        if path.is_dir() or not path.suffix:
            return path / 'operator_audit.json'
        return path.with_name(path.name + '.operator_audit.json')
    root = Path(os.environ.get('AEGIS_AUDIT_DIR', str(Path.home() / '.local' / 'state' / 'aegis-neurosurgery')))
    return root / f'{run_id}.json'


def _producer_audit(path):
    candidates = [path.with_name(path.name + '.operator_audit.json'), path.parent / 'operator_audit.json']
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def implementation_binding():
    root = Path(__file__).parent
    return {str(path.relative_to(root)): file_sha256(path) for path in sorted(root.rglob('*.py'))}


def audited_dispatch(args, dispatch):
    """Bind every declared local input and verify it before accepting an operator result.

    Remote checkpoints must be downloaded to a local directory first. Lower-level
    numerical APIs are not independent operator executions; workflows/campaigns
    must retain their own input and state manifests.
    """
    namespace = vars(args)
    import yaml
    paths = {}
    for role in INPUT_ROLES:
        value = namespace.get(role)
        if value is None or isinstance(value, bool):
            continue
        if not isinstance(value, (str, Path)):
            raise ValueError(f'Installed operator input {role} must be a local path')
        path = Path(value)
        if not path.exists():
            raise ValueError(f'Operator input {role} does not exist locally: {path}')
        paths[role] = path
    if 'config' in paths:
        config = yaml.safe_load(paths['config'].read_text())
        if isinstance(config, dict):
            for role in ('model', 'keep', 'drop', 'change', 'validation_keep', 'plan', 'profile'):
                if config.get(role) and role not in paths:
                    paths[role] = Path(config[role])
    # Bind every artifact resolved by the actual core plan convention.
    if 'plan' in paths:
        plan = yaml.safe_load(paths['plan'].read_text())
        if isinstance(plan, dict):
            for kind, section in (plan.get('structured') or {}).items():
                if section.get('enabled'):
                    raw = Path(section['selection'])
                    paths[f'selection_{kind}'] = raw if raw.is_absolute() else paths['plan'].parent / raw
    output = namespace.get('out')
    if output:
        target = Path(output).resolve()
        for role, source in paths.items():
            source = source.resolve()
            if target == source or (source.is_dir() and source in target.parents):
                raise ValueError(f'Output overlaps operator input {role}; use a separate destination')
    inputs = bind_files(paths)
    upstream = {}
    inherited_model = None
    unbound = []
    for role, path in paths.items():
        if role in MODEL_ROLES or path.is_dir() or role in {'keep', 'drop', 'change', 'validation_keep',
                                                          'distill', 'config', 'workload', 'benchmark',
                                                          'calibration_data', 'selectors', 'search_space'}:
            continue
        audit_path = _producer_audit(path)
        if audit_path is None:
            unbound.append(role)
            continue
        prior = json.loads(audit_path.read_text())
        if prior.get('status') != 'completed':
            raise ValueError(f'Producer audit for {role} did not complete')
        expected = prior.get('outputs', {}).get(path.name)
        if expected is None or expected != file_sha256(path):
            raise ValueError(f'Producer artifact checksum mismatch for {role}')
        source_model = prior.get('inputs', {}).get('model') or prior.get('source_model_binding')
        if source_model and inputs.get('model') and source_model['files'] != inputs['model']['files']:
            raise ValueError(f'Producer model binding mismatch for {role}')
        if source_model:
            if inherited_model and inherited_model['files'] != source_model['files']:
                raise ValueError('Plan inputs were produced from different source models')
            inherited_model = source_model
        upstream[role] = dict(audit_sha256=file_sha256(audit_path), audit=str(audit_path))
    import torch
    import transformers
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    audit = dict(version=1, command=args.cmd, inputs=inputs, upstream=upstream,
                 source_model_binding=inputs.get('model') or inherited_model,
                 unbound_legacy_artifacts=unbound, implementation=implementation_binding(),
                 options={key: value for key, value in namespace.items()
                          if isinstance(value, (str, int, float, bool, type(None), list))},
                 torch=torch.__version__, transformers=transformers.__version__,
                 started_at=started, status='running')
    run_id = uuid.uuid4().hex
    fallback = _audit_path(None, run_id)
    try:
        result = dispatch(args)
        verify_binding(inputs)
        if implementation_binding() != audit['implementation']:
            raise ValueError('Operator implementation changed during execution')
        for role, prior in upstream.items():
            if file_sha256(prior['audit']) != prior['audit_sha256']:
                raise ValueError(f'Producer audit changed during execution: {role}')
        audit['status'] = 'completed'
        audit['result_passed'] = result.get('passed') if isinstance(result, dict) else None
        audit['outputs'] = {}
        if output:
            target = Path(output)
            files = ([target] if target.is_file() else [p for p in target.rglob('*') if p.is_file()])
            # Artifacts are keyed by relative path, with root filenames directly usable by consumers.
            audit['outputs'] = {str(p.name if target.is_file() else p.relative_to(target)): file_sha256(p)
                                for p in files if not p.name.endswith('operator_audit.json')}
        destination = _audit_path(output, run_id)
        return result
    except BaseException as exc:
        audit['status'] = 'failed'
        audit['error'] = f'{type(exc).__name__}: {exc}'
        destination = fallback
        raise
    finally:
        audit['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')
        LOG.info('Operator audit: %s', destination)


def callable_binding(handler):
    import hashlib
    import inspect
    if handler is None:
        return None
    try:
        source = inspect.getsource(handler)
    except (OSError, TypeError):
        raise ValueError('Custom operator handlers must have inspectable source for audit binding')
    return dict(module=handler.__module__, name=handler.__qualname__,
                source_sha256=hashlib.sha256(source.encode()).hexdigest())
