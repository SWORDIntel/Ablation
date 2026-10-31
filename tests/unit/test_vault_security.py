"""ModelVault path isolation tests; do not substitute for cryptographic security tests."""
import os
from pathlib import Path
import stat
import tempfile
import unittest

from aegis_lab.hardware.vault import ModelVault


class TestVaultFilesystemIsolation(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.directory = self.base / "vault"
        self.directory.mkdir(mode=0o700)
        self.vault = ModelVault.__new__(ModelVault)
        self.vault.vault_path = str(self.directory)
        self.vault.is_locked = False
        # Keep file-path tests independent of historical software/MEI crypto.
        self.vault._encrypt = lambda value: b"test:" + value
        self.vault._decrypt = lambda value: value[5:]

    def test_roundtrip_and_permissions(self):
        self.assertTrue(self.vault.store_atom_securely("a-b_c.1", b"hello"))
        self.assertEqual(self.vault.retrieve_atom_securely("a-b_c.1"), b"hello")
        path = self.directory / "a-b_c.1.enc"
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertIsNone(self.vault.retrieve_atom_securely("missing"))

    def test_rejects_traversal_and_non_flat_ids(self):
        outside = self.base / "outside.enc"
        outside.write_bytes(b"unchanged")
        for bad in ("../outside", "../outside/../outside", "/tmp/x", "dir/x", "dir\\x",
                    "", ".", "..", "a..b", "a" * 129, "\x00", "🧪", "a b", None, 42):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(ValueError):
                    self.vault.store_atom_securely(bad, b"overwrite")
                with self.assertRaises(ValueError):
                    self.vault.retrieve_atom_securely(bad)
        self.assertEqual(outside.read_bytes(), b"unchanged")

    def test_symlink_cannot_redirect_read_or_write(self):
        outside = self.base / "outside.enc"
        outside.write_bytes(b"outside")
        (self.directory / "link.enc").symlink_to(outside)
        with self.assertRaises(OSError):
            self.vault.store_atom_securely("link", b"overwritten")
        with self.assertRaises(OSError):
            self.vault.retrieve_atom_securely("link")
        self.assertEqual(outside.read_bytes(), b"outside")

    def test_hardlink_cannot_redirect_write(self):
        outside = self.base / "outside.enc"
        outside.write_bytes(b"outside")
        os.link(outside, self.directory / "linked.enc")
        with self.assertRaises(OSError):
            self.vault.store_atom_securely("linked", b"overwritten")
        self.assertEqual(outside.read_bytes(), b"outside")

    def test_rejects_symlinked_vault_directory(self):
        link = self.base / "vault-link"
        link.symlink_to(self.directory, target_is_directory=True)
        self.vault.vault_path = str(link)
        with self.assertRaises(OSError):
            self.vault.store_atom_securely("atom", b"value")

    def test_tightens_directory_permissions(self):
        os.chmod(self.directory, 0o755)
        self.vault.store_atom_securely("atom", b"value")
        self.assertEqual(stat.S_IMODE(self.directory.stat().st_mode), 0o700)

    def test_locked_vault_remains_closed(self):
        self.vault.is_locked = True
        self.assertFalse(self.vault.store_atom_securely("atom", b"value"))
        self.assertIsNone(self.vault.retrieve_atom_securely("atom"))


if __name__ == "__main__":
    unittest.main()
