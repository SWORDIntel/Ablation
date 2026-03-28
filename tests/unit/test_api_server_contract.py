import asyncio
import unittest

from fastapi import HTTPException

from aegis_lab.api import server


class FakeClient:
    def __init__(self):
        self.calls = []

    def request(self, msg_type, **kwargs):
        self.calls.append((msg_type, kwargs))
        if msg_type == "submit_job":
            return {"status": "ok", "job_id": "job-123"}
        if msg_type == "list_jobs":
            return [{"job_id": "job-123", "status": "pending"}]
        if msg_type == "get_job_status":
            return {"job_id": kwargs["job_id"], "status": "running"}
        return {"status": "error", "error": "unknown"}


class TestAPIServerContract(unittest.TestCase):
    def tearDown(self):
        server.set_orchestrator_client(None)

    def test_submit_job_uses_injected_client(self):
        fake_client = FakeClient()
        server.set_orchestrator_client(fake_client)

        response = asyncio.run(
            server.submit_job(
                server.JobSubmission(project_id="proj-1", job_type="ablation")
            )
        )

        self.assertEqual(response["job_id"], "job-123")
        self.assertEqual(fake_client.calls[0][0], "submit_job")
        self.assertEqual(fake_client.calls[0][1]["project_id"], "proj-1")

    def test_get_job_status_raises_not_found_for_error_payload(self):
        class ErrorClient:
            def request(self, msg_type, **kwargs):
                return {"status": "error", "error": "missing job"}

        server.set_orchestrator_client(ErrorClient())

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(server.get_job_status("job-missing"))

        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
