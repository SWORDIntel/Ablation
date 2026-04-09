import os
import sys
import unittest
import tempfile
import shutil
from aegis_lab.state.db import AegisState

class TestAegisState(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.storage_root = os.path.join(self.test_dir, "state")
        
        # Path to libqihse.so
        self.lib_path = "../../../../../home/john/Documents/MEMSHADOW/QIHSE/qihse/libqihse.so"
        if not os.path.exists(self.lib_path):
            self.skipTest(f"libqihse.so not found at {self.lib_path}")
            
        self.state = AegisState(self.storage_root, self.lib_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_create_and_get_job(self):
        job = self.state.create_job("job-123", "proj-abc", "ablation")
        self.assertEqual(job["job_id"], "job-123")
        self.assertEqual(job["status"], "pending")
        
        jobs = self.state.get_jobs()
        self.assertTrue(any(j["job_id"] == "job-123" for j in jobs))

    def test_stages(self):
        self.state.create_stage("stage-1", "job-123", "intake", 1)
        self.state.create_stage("stage-2", "job-123", "probe", 2)
        
        stages = self.state.get_stages("job-123")
        self.assertEqual(len(stages), 2)
        self.assertTrue(any(s["stage_name"] == "intake" for s in stages))

if __name__ == "__main__":
    unittest.main()
