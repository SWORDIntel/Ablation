import os
import tempfile
import unittest
from pathlib import Path

from framewerx.aegis_lab.state.db import AegisState
from framewerx.aegis_lab.artifacts.store import ArtifactStore
from framewerx.aegis_lab.atoms.registry import AtomRegistry

class TestAtomRegistry(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_root = os.path.join(self.temp_dir.name, "state")
        self.artifact_root = os.path.join(self.temp_dir.name, "artifacts")
        
        # Find libqihse.so
        self.lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "QIHSE", "qihse", "libqihse.so"))
        if not os.path.exists(self.lib_path):
            self.skipTest(f"libqihse.so not found at {self.lib_path}")

        self.state = AegisState(self.storage_root, self.lib_path)
        self.store = ArtifactStore(self.artifact_root, state=self.state)
        self.registry = AtomRegistry(self.state, self.store)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_atom_registry_workflow(self):
        # Put a dummy artifact in the store
        dummy_file = os.path.join(self.temp_dir.name, "dummy_tensor.pt")
        with open(dummy_file, "w") as f:
            f.write("dummy_data")
        artifact_hash = self.store.put_file(dummy_file)

        # Register an atom
        semantic_vector = [0.1, 0.2, 0.3, 0.4] + [0.0] * 124

        atom = self.registry.register_atom(
            atom_id="atom_123",
            atom_type="TargetAtom",
            content_hash="hash_content",
            architecture_scope="transformer",
            prompt_suite_hash="hash_suite",
            extraction_config_hash="hash_config",
            validation_status="valid",
            artifact_id=artifact_hash,
            semantic_vector=semantic_vector
        )

        self.assertEqual(atom["atom_id"], "atom_123")
        self.assertEqual(atom["validation_status"], "valid")

        # Retrieve it
        fetched = self.registry.get_atom_record("atom_123")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["atom_id"], "atom_123")

        # Search
        results = self.registry.search_atoms(semantic_vector, top_k=5)
        self.assertGreater(len(results), 0)
        self.assertTrue(any(r["atom_id"] == "atom_123" for r in results))

        # Fetch artifact
        dest_path = os.path.join(self.temp_dir.name, "retrieved_tensor.pt")
        self.registry.retrieve_atom_artifact("atom_123", dest_path)

        self.assertTrue(os.path.exists(dest_path))
        with open(dest_path, "r") as f:
            self.assertEqual(f.read(), "dummy_data")

if __name__ == "__main__":
    unittest.main()
