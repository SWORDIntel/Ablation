import unittest
import os
import shutil
import tempfile
from aegis_lab.state.db import AegisState

class TestDbQihse(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "QIHSE", "qihse", "libqihse.so"))
        # Ensure we skip if lib doesn't exist
        if not os.path.exists(self.lib_path):
            self.skipTest(f"QIHSE library not found at {self.lib_path}")
        self.state = AegisState(self.test_dir, self.lib_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_job_upsert_deduplication(self):
        job_id = "test-job-1"
        self.state.create_job(job_id, "project-a", "ablation")
        
        # Initial status
        job = self.state.get_job(job_id)
        self.assertEqual(job["status"], "pending")
        
        # Update status
        self.state.update_job(job_id, {"status": "running"})
        job = self.state.get_job(job_id)
        self.assertEqual(job["status"], "running")
        
        # All jobs count should still be 1 after deduplication
        all_jobs = self.state.get_jobs()
        self.assertEqual(len(all_jobs), 1)

    def test_stage_filtering(self):
        job_a = "job-a"
        job_b = "job-b"
        self.state.create_stage("stage-1", job_a, "atom_clean", 1)
        self.state.create_stage("stage-2", job_b, "atom_clean", 1)
        
        stages_a = self.state.get_stages(job_a)
        self.assertEqual(len(stages_a), 1)
        self.assertEqual(stages_a[0]["job_id"], job_a)

if __name__ == "__main__":
    unittest.main()
