from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class TopologyManifest:
    schema_version: str
    job_id: str
    source_model_ref: str
    architecture_family: str
    adapter_name: str
    parameter_count: int
    editable_modules: List[str]
    routing_characteristics: Dict[str, Any]
    moe_enabled: bool
    multimodal_enabled: bool
    hardware_profile: str
    created_at: str

@dataclass
class StageManifest:
    schema_version: str
    job_id: str
    stage_id: str
    stage_name: str
    input_artifacts: List[str]
    output_artifacts: List[str]
    config_hash: str
    worker_type: str
    device_plan: str
    device_participation: List[Dict[str, Any]]
    timings: Dict[str, float]
    result_summary: Dict[str, Any]
    created_at: str

@dataclass
class FinalManifest:
    schema_version: str
    job_id: str
    project_id: str
    source_model_hash: str
    topology_manifest_hash: str
    atom_hashes: List[str]
    delta_hashes: List[str]
    calibration_set_hash: str
    quantization_profile: str
    high_precision_verification_summary: Dict[str, Any]
    post_quant_verification_summary: Dict[str, Any]
    device_participation_summary: Dict[str, Any]
    operator_gate_summary: Dict[str, Any]
    rollback_points: List[str]
    export_artifacts: List[str]
    created_at: str
