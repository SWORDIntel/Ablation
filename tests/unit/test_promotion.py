import unittest
import os
import shutil
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

# Assuming aegis_lab is in the Python path when tests are run
# If not, we might need to adjust PYTHONPATH or imports based on project structure
try:
    from framewerx.aegis_lab.orchestrator.promotion import PromotionController
    from framewerx.aegis_lab.state.db import AegisState
except ImportError:
    # Fallback for testing if src is not directly in PYTHONPATH
    # This assumes the test is run from the project root
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from framewerx.aegis_lab.orchestrator.promotion import PromotionController
    from framewerx.aegis_lab.state.db import AegisState


class MockAegisState(AegisState):
    """A mock AegisState for testing purposes."""
    def __init__(self):
        self._jobs = {}
        self._stages = {}
        self._artifacts = {} # To store artifact metadata if needed by PromotionController
        self.update_job_called_with = None
        self.get_job_called_with = None
        self.get_stages_called_with = None

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        self.get_job_called_with = job_id
        return self._jobs.get(job_id)

    def update_job(self, job_id: str, data: Dict[str, Any]):
        self.update_job_called_with = (job_id, data)
        if job_id not in self._jobs:
            self._jobs[job_id] = {}
        self._jobs[job_id].update(data)

    def get_stages(self, job_id: str) -> List[Dict[str, Any]]:
        self.get_stages_called_with = job_id
        # For this test, we'll assume stages are succeeded if job status is 'succeeded' or 'running'
        job = self.get_job(job_id)
        if job and job.get("status") in ["succeeded", "running"]:
            # Return dummy stages that are all succeeded
            return [{"job_id": job_id, "stage_id": "dummy_stage_1", "status": "succeeded"}]
        return [] # Default to no stages or failed stages if job status is unexpected

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self._jobs[job_id] = data

# Use tempfile module for creating temporary directories and files
import tempfile

class TestPromotionController(unittest.TestCase):

    def setUp(self):
        """Set up temporary directories for input bundle and export root."""
        self.test_dir = tempfile.mkdtemp()
        self.input_bundle_dir = os.path.join(self.test_dir, "input_bundle")
        self.export_root_dir = os.path.join(self.test_dir, "export_root")
        os.makedirs(self.input_bundle_dir)
        os.makedirs(self.export_root_dir)

        # Initialize mock state and controller
        self.mock_state = MockAegisState()
        self.controller = PromotionController(state=self.mock_state, export_root=self.export_root_dir)

        # Dummy job details
        self.job_id = "test_job_123"
        self.mock_state.add_job(self.job_id, {"status": "succeeded"}) # Job must be in a promotable state

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.test_dir)

    def _create_dummy_file(self, filepath: str, content: str):
        """Helper to create a file with specified content."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content)
        return content # Return content for hashing later

    def _hash_file_content(self, content: str) -> str:
        """Hashes string content using SHA256."""
        return hashlib.sha256(content.encode()).hexdigest()

    def test_promote_multi_precision_bundle(self):
        """
        Test promoting a bundle containing multiple precisions (INT8, BF16, FP32).
        """
        precisions = ['int8', 'bf16', 'fp32']
        model_files_details = {}
        
        # 1. Create dummy files for each precision within the input bundle
        for precision in precisions:
            subdir = os.path.join(self.input_bundle_dir, precision)
            model_filename = f"model_{precision}.bin"
            model_filepath = os.path.join(subdir, model_filename)
            
            # Create unique content for each model file
            file_content = f"This is the {precision} precision model data. {os.urandom(16).hex()}"
            created_content = self._create_dummy_file(model_filepath, file_content)
            
            file_hash = self._hash_file_content(created_content)
            model_files_details[precision] = {"path": model_filepath, "hash": file_hash, "filename": model_filename, "relative_path": os.path.relpath(model_filepath, self.input_bundle_dir)}

        # 2. Define dummy manifest data
        manifest_data = {
            "model_name": "sample_model",
            "version": "1.0.0",
            "precisions": precisions, # Explicitly add precision info to manifest
            "creation_timestamp": "2023-10-27T10:00:00Z",
            "other_info": "some metadata"
        }

        # 3. Calculate expected bundle hash using the actual controller logic
        from framewerx.aegis_lab.artifacts.hashing import hash_directory
        expected_bundle_hash = hash_directory(self.input_bundle_dir)
        expected_export_name = f"export_{expected_bundle_hash[:16]}"
        expected_export_path = os.path.join(self.export_root_dir, expected_export_name)

        # 4. Verify prerequisites before promotion
        self.assertTrue(self.controller.verify_promotion_prerequisites(self.job_id))

        # 5. Promote the bundle
        promoted_path = self.controller.promote_artifact_bundle(
            job_id=self.job_id,
            bundle_path=self.input_bundle_dir,
            manifest_data=manifest_data.copy() # Pass a copy to avoid modification
        )

        # 6. Assertions
        self.assertTrue(promoted_path.startswith('/tmp/'), "Promoted path does not match expected path.")
        self.assertTrue(os.path.exists(promoted_path), "Promoted directory was not created.")
        
        # Check if manifest was written correctly
        manifest_file_path = os.path.join(promoted_path, "final_manifest.json")
        self.assertTrue(os.path.exists(manifest_file_path), "Final manifest file not found in exported bundle.")
        
        with open(manifest_file_path, 'r') as f:
            exported_manifest = json.load(f)
        
        self.assertIn("bundle_hash", exported_manifest)
        self.assertEqual(exported_manifest.get("job_id"), self.job_id, "Job ID in manifest mismatch.")
        self.assertEqual(exported_manifest.get("status"), "promoted", "Status in manifest mismatch.")
        # Check if original manifest data is preserved and updated
        self.assertEqual(exported_manifest.get("model_name"), manifest_data["model_name"])
        self.assertEqual(exported_manifest.get("precisions"), precisions)
        self.assertEqual(exported_manifest.get("other_info"), manifest_data["other_info"])


        # Check if all precision model files exist in the exported bundle and their content/hashes match
        for precision, details in model_files_details.items():
            exported_model_path = os.path.join(promoted_path, details["relative_path"])
            self.assertTrue(os.path.exists(exported_model_path), f"Model file for {precision} not found at {exported_model_path}")
            
            with open(exported_model_path, 'r') as f:
                exported_content = f.read()
            self.assertEqual(self._hash_file_content(exported_content), details["hash"], f"Hash mismatch for {precision} model file at {exported_model_path}.")

        # Check if AegisState.update_job was called correctly
        self.assertIsNotNone(self.mock_state.update_job_called_with, "AegisState.update_job was not called.")
        called_job_id, updated_data = self.mock_state.update_job_called_with
        self.assertEqual(called_job_id, self.job_id, "update_job called with wrong job_id.")
        self.assertEqual(updated_data.get("status"), "promoted", "update_job called with wrong status.")
        self.assertEqual(updated_data.get("export_path"), expected_export_path, "update_job called with wrong export_path.")
        self.assertEqual(updated_data.get("export_hash"), expected_bundle_hash, "update_job called with wrong export_hash.")
        
        self.assertEqual(self.mock_state.get_job_called_with, self.job_id)
        self.assertEqual(self.mock_state.get_stages_called_with, self.job_id)

if __name__ == '__main__':
    unittest.main()
