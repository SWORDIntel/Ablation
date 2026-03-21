from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import zmq
import json

app = FastAPI(title="AEGIS-LAB REST API", version="0.1.0")

class OrchestratorClient:
    def __init__(self, url="tcp://localhost:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(url)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)

    def request(self, msg_type, **kwargs):
        try:
            kwargs["type"] = msg_type
            self.socket.send_json(kwargs)
            return self.socket.recv_json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

client = OrchestratorClient(os.getenv("ORCHESTRATOR_URL", "tcp://localhost:5555"))

class JobSubmission(BaseModel):
    project_id: str
    job_type: str = "ablation"
    parameters: Dict[str, Any] = {}

@app.post("/jobs/submit")
async def submit_job(submission: JobSubmission):
    resp = client.request("submit_job", 
                          project_id=submission.project_id, 
                          job_type=submission.job_type,
                          parameters=submission.parameters)
    if resp.get("status") == "ok":
        return resp
    raise HTTPException(status_code=500, detail=resp.get("error"))

@app.get("/jobs")
async def list_jobs():
    resp = client.request("list_jobs")
    if isinstance(resp, list):
        return resp
    raise HTTPException(status_code=500, detail=resp.get("error"))

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    resp = client.request("get_job_status", job_id=job_id)
    if "error" in resp:
        raise HTTPException(status_code=404, detail=resp["error"])
    return resp

@app.get("/hardware/sitrep")
async def get_hardware_sitrep():
    # In a real implementation, the Orchestrator would aggregate this
    resp = client.request("get_sitrep")
    return resp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
