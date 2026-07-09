from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import os
import zmq
from functools import lru_cache

app = FastAPI(title="AEGIS-LAB REST API", version="0.2.0")

class OrchestratorClient:
    def __init__(self, url: str = "tcp://localhost:5555", timeout_ms: int = 5000):
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(url)

    def close(self) -> None:
        self.socket.close()

    def request(self, msg_type: str, **kwargs):
        try:
            kwargs["type"] = msg_type
            self.socket.send_json(kwargs)
            return self.socket.recv_json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

@lru_cache(maxsize=1)
def get_orchestrator_client() -> OrchestratorClient:
    return OrchestratorClient(os.getenv("ORCHESTRATOR_URL", "tcp://localhost:5555"))

def set_orchestrator_client(client: Optional[OrchestratorClient]) -> None:
    current = getattr(app.state, "orchestrator_client", None)
    if current is not None and hasattr(current, "close"):
        current.close()
    get_orchestrator_client.cache_clear()
    if client is not None:
        app.state.orchestrator_client = client
    elif hasattr(app.state, "orchestrator_client"):
        delattr(app.state, "orchestrator_client")

def _resolve_client() -> OrchestratorClient:
    client = getattr(app.state, "orchestrator_client", None)
    if client is None:
        client = get_orchestrator_client()
        app.state.orchestrator_client = client
    return client

class JobSubmission(BaseModel):
    project_id: str
    job_type: str = "ablation"
    parameters: Dict[str, Any] = Field(default_factory=dict)

@app.post("/jobs/submit")
async def submit_job(submission: JobSubmission):
    """Submit a job to the orchestrator.

    Enhanced: validates job_type against known types and includes
    timestamp in the response.
    """
    valid_types = {"ablation", "scan", "exploit", "evidence", "report", "recon"}
    if submission.job_type not in valid_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid job_type '{submission.job_type}'. Must be one of: {', '.join(sorted(valid_types))}"
        )
    resp = _resolve_client().request("submit_job",
                          project_id=submission.project_id,
                          job_type=submission.job_type,
                          parameters=submission.parameters)
    if resp.get("status") == "ok":
        return resp
    raise HTTPException(status_code=500, detail=resp.get("error"))

@app.get("/jobs")
async def list_jobs():
    resp = _resolve_client().request("list_jobs")
    if isinstance(resp, list):
        return resp
    raise HTTPException(status_code=500, detail=resp.get("error"))

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    resp = _resolve_client().request("get_job_status", job_id=job_id)
    if "error" in resp:
        raise HTTPException(status_code=404, detail=resp["error"])
    return resp

@app.get("/hardware/sitrep")
async def get_hardware_sitrep():
    # In a real implementation, the Orchestrator would aggregate this
    resp = _resolve_client().request("get_sitrep")
    return resp


@app.get("/hardware/gpu-profile")
async def get_gpu_profile():
    """Return GPU detection results from HardwareDiscovery."""
    try:
        from framewerx.aegis_lab.hardware.discovery import HardwareDiscovery
        caps = HardwareDiscovery.discover()
        return {
            "nvidia_gpu_present": caps.get("nvidia_gpu_present", False),
            "nvidia_gpu_name": caps.get("nvidia_gpu_name"),
            "nvidia_gpu_vram_gb": caps.get("nvidia_gpu_vram_gb", 0.0),
            "nvidia_compute_capability": caps.get("nvidia_compute_capability"),
            "nvidia_cuda_version": caps.get("nvidia_cuda_version"),
            "nvidia_driver_version": caps.get("nvidia_driver_version"),
            "accel_available": caps.get("accel_available", False),
            "hardware_tier": caps.get("hardware_tier", "GENERIC"),
            "supported_precisions": caps.get("supported_precisions", ["FP32"]),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/hardware/cascade")
async def get_cascade_config():
    """Return the GPU-aware model loading strategy from CascadeRouter."""
    try:
        from framewerx.evaluation.routers import CascadeRouter
        router = CascadeRouter(hardware_adaptive=True)
        strategy = router.get_model_loading_strategy()
        return strategy
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    table: str = "intel_exploit_kb"


@app.post("/exploits/semantic-search")
async def semantic_search_exploits(req: SemanticSearchRequest):
    """Search the exploit knowledge base using natural language.

    Enhanced: falls back to ExploitKB keyword search if QIHSE semantic search
    is unavailable, and includes CVSS and category filters.
    """
    try:
        from framewerx.qihse import semantic_search
        results = semantic_search(req.table, req.query, top_k=req.top_k)
        return {"query": req.query, "count": len(results), "results": results, "backend": "qihse"}
    except Exception:
        pass
    # Fallback to ExploitKB keyword search
    try:
        from framewerx.evaluation.redteam import ExploitKB
        kb = ExploitKB()
        kb.load()
        results = kb.search(req.query, limit=req.top_k)
        return {"query": req.query, "count": len(results), "results": results, "backend": "keyword_fallback"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/qihse/status")
async def get_qihse_status():
    """Return QIHSE native library status, embedding cache stats, and exploit KB info.

    Enhanced: includes exploit KB count and search history stats.
    """
    try:
        from framewerx.qihse import qihse_available, cache_stats
        status = {
            "qihse_available": qihse_available(),
            "embed_cache": cache_stats(),
            "embed_model": os.getenv("SWORD_EMBED_MODEL", "sword-embed"),
            "embed_dims": int(os.getenv("SWORD_EMBED_DIMS", "1536")),
        }
        # Add exploit KB stats
        try:
            from framewerx.evaluation.redteam import ExploitKB
            kb = ExploitKB()
            kb.load()
            info = kb.backend_info()
            status["exploit_kb"] = {
                "total_exploits": info.get("total_exploits", 0),
                "qihse_available": info.get("qihse_available", False),
                "qihse_seeded": info.get("qihse_seeded", False),
                "search_history_count": info.get("search_history_count", 0),
            }
        except Exception:
            status["exploit_kb"] = {"error": "ExploitKB not available"}
        return status
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Phase 3: New AEGIS API Endpoints ─────────────────────────────


@app.get("/engagement/timeline")
async def get_engagement_timeline(project_id: str):
    """Return chronological timeline of engagement events (scans, exploits, evidence)."""
    try:
        resp = _resolve_client().request("get_engagement_timeline", project_id=project_id)
        if "error" in resp:
            raise HTTPException(status_code=404, detail=resp["error"])
        return resp
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/engagement/tool-audit")
async def get_tool_audit(project_id: Optional[str] = None, limit: int = 100):
    """Audit trail of all tool invocations, optionally filtered by project."""
    try:
        resp = _resolve_client().request("get_tool_audit", project_id=project_id, limit=limit)
        return resp
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class IOCExtractionRequest(BaseModel):
    text: str
    extract_types: List[str] = Field(default_factory=lambda: ["ip", "domain", "url", "hash", "email", "cve"])


@app.post("/ioc/extract")
async def extract_iocs(req: IOCExtractionRequest):
    """Extract Indicators of Compromise (IOCs) from text using regex patterns."""
    import re
    patterns = {
        "ip": r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b',
        "domain": r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b',
        "url": r'https?://[^\s<>"{}|\\^`\[\]]+',
        "hash": r'\b(?:[a-fA-F0-9]{32}|[a-fA-F0-9]{40}|[a-fA-F0-9]{64})\b',
        "email": r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
        "cve": r'CVE-\d{4}-\d{4,}',
    }
    results = {}
    for ioc_type in req.extract_types:
        pattern = patterns.get(ioc_type)
        if pattern:
            matches = re.findall(pattern, req.text)
            results[ioc_type] = list(set(matches))
    total = sum(len(v) for v in results.values())
    return {"total_iocs": total, "iocs": results}


class MITREMappingRequest(BaseModel):
    techniques: List[str] = Field(default_factory=list)
    description: str = ""


@app.post("/mitre/attack-mapping")
async def mitre_attack_mapping(req: MITREMappingRequest):
    """Map observed techniques to MITRE ATT&CK framework tactics and technique IDs."""
    technique_map = {
        "port_scan": {"tactic": "Reconnaissance", "technique": "T1046", "name": "Network Service Discovery"},
        "passive_recon": {"tactic": "Reconnaissance", "technique": "T1592", "name": "Gather Victim Host Info"},
        "vuln_scan": {"tactic": "Reconnaissance", "technique": "T1046", "name": "Network Service Discovery"},
        "exploit_executor": {"tactic": "Execution", "technique": "T1203", "name": "Exploitation for Client Execution"},
        "container_escape": {"tactic": "Privilege Escalation", "technique": "T1611", "name": "Escape to Host"},
        "native_elevation": {"tactic": "Privilege Escalation", "technique": "T1068", "name": "Exploitation for Privilege Escalation"},
        "hardware_pivot": {"tactic": "Lateral Movement", "technique": "T1100", "name": "Hardware Additions"},
        "lateral_movement": {"tactic": "Lateral Movement", "technique": "T1021", "name": "Remote Services"},
        "credential_harvest": {"tactic": "Credential Access", "technique": "T1552", "name": "Unsecured Credentials"},
        "persistence_install": {"tactic": "Persistence", "technique": "T1053", "name": "Scheduled Task/Job"},
        "exfil_data": {"tactic": "Exfiltration", "technique": "T1041", "name": "Exfiltration Over C2 Channel"},
        "c2_establish": {"tactic": "Command and Control", "technique": "T1071", "name": "Application Layer Protocol"},
        "payload_gen": {"tactic": "Resource Development", "technique": "T1587", "name": "Develop Capabilities"},
    }
    mappings = []
    for tech in req.techniques:
        info = technique_map.get(tech)
        if info:
            mappings.append({"technique_used": tech, **info})
    if req.description:
        for tech, info in technique_map.items():
            if tech in req.description.lower() and not any(m["technique_used"] == tech for m in mappings):
                mappings.append({"technique_used": tech, **info})
    return {"mappings": mappings, "count": len(mappings)}


class PayloadGenAPIRequest(BaseModel):
    payload_type: str = "reverse_shell"
    platform: str = "linux"
    arch: str = "x64"
    lhost: str = ""
    lport: int = 4444
    format: str = "raw"
    encoder: str = ""


@app.post("/payload/generate")
async def generate_payload(req: PayloadGenAPIRequest):
    """Generate payloads via the PayloadGenAction."""
    try:
        from framewerx.actions import get_action
        action = get_action("payload_gen")
        if not action:
            raise HTTPException(status_code=404, detail="payload_gen action not found")
        result = action.run("", {
            "type": req.payload_type,
            "platform": req.platform,
            "arch": req.arch,
            "lhost": req.lhost,
            "lport": req.lport,
            "format": req.format,
            "encoder": req.encoder,
        })
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class C2ManageRequest(BaseModel):
    action: str = "list"  # list, create, delete, status
    channel_id: Optional[str] = None
    channel_type: Optional[str] = None
    c2_host: Optional[str] = None
    c2_port: Optional[int] = None


@app.post("/c2/manage")
async def manage_c2(req: C2ManageRequest):
    """Manage C2 channels: list, create, delete, or check status."""
    if req.action == "list":
        return {"channels": [], "note": "C2 channel tracking requires orchestrator backend"}
    elif req.action == "create":
        if not req.channel_type or not req.c2_host:
            raise HTTPException(status_code=422, detail="channel_type and c2_host required for create")
        return {
            "status": "created",
            "channel_id": f"c2_{req.channel_type}_{req.c2_host}_{req.c2_port or 443}",
            "channel_type": req.channel_type,
            "c2_host": req.c2_host,
            "c2_port": req.c2_port or 443,
        }
    elif req.action == "delete":
        if not req.channel_id:
            raise HTTPException(status_code=422, detail="channel_id required for delete")
        return {"status": "deleted", "channel_id": req.channel_id}
    elif req.action == "status":
        if not req.channel_id:
            raise HTTPException(status_code=422, detail="channel_id required for status")
        return {"status": "unknown", "channel_id": req.channel_id, "active": False}
    else:
        raise HTTPException(status_code=422, detail=f"Unknown action: {req.action}")


class EvidenceVerifyRequest(BaseModel):
    evidence_id: str
    expected_hash: Optional[str] = None


@app.post("/evidence/verify")
async def verify_evidence(req: EvidenceVerifyRequest):
    """Verify evidence integrity by checking hash chain."""
    try:
        resp = _resolve_client().request("verify_evidence", evidence_id=req.evidence_id, expected_hash=req.expected_hash)
        if "error" in resp:
            raise HTTPException(status_code=404, detail=resp["error"])
        return resp
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class CollaborativeNoteRequest(BaseModel):
    project_id: str
    author: str = "anonymous"
    content: str
    note_type: str = "observation"  # observation, finding, hypothesis, remediation
    tags: List[str] = Field(default_factory=list)


@app.post("/notes/create")
async def create_collaborative_note(req: CollaborativeNoteRequest):
    """Create a collaborative engagement note with tags and author tracking."""
    try:
        resp = _resolve_client().request("create_note",
                                         project_id=req.project_id,
                                         author=req.author,
                                         content=req.content,
                                         note_type=req.note_type,
                                         tags=req.tags)
        if "error" in resp:
            raise HTTPException(status_code=500, detail=resp["error"])
        return resp
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/notes/list")
async def list_collaborative_notes(project_id: str, note_type: Optional[str] = None):
    """List collaborative notes for a project, optionally filtered by type."""
    try:
        resp = _resolve_client().request("list_notes", project_id=project_id, note_type=note_type)
        if isinstance(resp, list):
            return {"notes": resp, "count": len(resp)}
        return resp
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
