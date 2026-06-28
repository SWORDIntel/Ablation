import unittest
import os
import shutil
import tempfile
from aegis_lab.hardware.vault import ModelVault

class TestModelVaultSecurity(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.vault = ModelVault()
        self.vault.vault_path = self.test_dir
        self.vault.unlock_csme()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_store_path_traversal_blocked(self):
        malicious_atom_id = "../../etc/passwd"
        data = b"malicious data"

        with self.assertRaises(ValueError) as cm:
            self.vault.store_atom_securely(malicious_atom_id, data)
        self.assertIn("Path traversal detected", str(cm.exception))

        # Ensure no file was created outside
        outside_file = os.path.join(self.test_dir, malicious_atom_id + ".enc")
        self.assertFalse(os.path.exists(outside_file))

    def test_retrieve_path_traversal_blocked(self):
        # Manually create a file outside
        outside_path = os.path.join(os.path.dirname(self.test_dir), "outside.enc")
        with open(outside_path, "wb") as f:
            f.write(b"secret")

        try:
            malicious_atom_id = "../outside"
            data = self.vault.retrieve_atom_securely(malicious_atom_id)
            self.assertIsNone(data)
        finally:
            if os.path.exists(outside_path):
                os.remove(outside_path)

    def test_legitimate_access(self):
        atom_id = "valid-atom-id"
        data = b"some-secure-data"

        self.assertTrue(self.vault.store_atom_securely(atom_id, data))
        self.assertEqual(self.vault.retrieve_atom_securely(atom_id), data)

    def test_locked_vault_denies_access(self):
        self.vault.is_locked = True
        self.assertFalse(self.vault.store_atom_securely("id", b"data"))
        self.assertIsNone(self.vault.retrieve_atom_securely("id"))

if __name__ == "__main__":
    unittest.main()
