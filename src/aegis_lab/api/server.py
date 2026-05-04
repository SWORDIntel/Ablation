from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os
import zmq
from functools import lru_cache

app = FastAPI(title="AEGIS-LAB REST API", version="0.1.0")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
