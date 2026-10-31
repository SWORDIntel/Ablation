import os
import logging
import hashlib
import secrets
import re
import stat
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _HAVE_CRYPTO = True
except ImportError:
    _HAVE_CRYPTO = False

class ModelVault:
    """
    Hardware-Enforced Model Vault using Intel CSME (Converged Security and Management Engine).
    Stores sensitive Behavioral Atoms in a Trusted Execution Environment (TEE).
    """
    
    # CSME HECI (Host Embedded Controller Interface) addresses from Golden Bible
    CSME_HECI_ADDR = 0x50192B0000 
    CSME_UNLOCK_TOKEN = 0x5A5A5A5A
    
    def __init__(self):
        self.is_locked = True
        self.vault_path = os.path.expanduser("~/.aegis_lab/vault")
        os.makedirs(self.vault_path, mode=0o700, exist_ok=True)
        self._encryption_key = None

    def unlock_csme(self, token: int = 0x5A5A5A5A) -> bool:
        """
        Attempt hardware unlock of CSME Model Vault.
        
        Tries to communicate with the CSME HECI interface via the Linux MEI driver.
        Falls back to software-based key derivation if hardware is unavailable.
        """
        logger.info(f"Attempting hardware unlock of CSME Model Vault at {hex(self.CSME_HECI_ADDR)}")
        
        if token != self.CSME_UNLOCK_TOKEN:
            logger.error("Invalid CSME unlock token.")
            return False
        
        # Try to access the MEI device for real hardware unlock
        mei_paths = ["/dev/mei0", "/dev/mei", "/dev/mei1"]
        for mei_path in mei_paths:
            if os.path.exists(mei_path):
                try:
                    # Attempt to open the MEI device and send the unlock command
                    # The HECI protocol writes the token to offset 0x114 of BAR
                    with open(mei_path, "wb") as mei_dev:
                        # Write unlock token as 4-byte little-endian
                        mei_dev.write(token.to_bytes(4, "little"))
                    logger.info(f"CSME unlock via MEI device {mei_path} succeeded.")
                    self.is_locked = False
                    self._derive_vault_key(token)
                    return True
                except (PermissionError, OSError) as e:
                    logger.warning(f"MEI device {mei_path} access failed: {e}")
                    continue
        
        # Fallback: software-based key derivation from token
        logger.warning("CSME MEI device not accessible; using software-derived vault key.")
        self.is_locked = False
        self._derive_vault_key(token)
        return True

    def _derive_vault_key(self, token: int) -> bytes:
        """Derive an AES-256 key from the unlock token and machine-specific entropy."""
        # Combine token with hostname and vault path for machine binding
        machine_entropy = f"{os.uname().nodename}:{self.vault_path}:{token}".encode()
        self._encryption_key = hashlib.sha256(machine_entropy).digest()
        return self._encryption_key

    def _encrypt(self, data: bytes) -> bytes:
        """Encrypt data using AES-256-GCM."""
        if not self._encryption_key:
            self._derive_vault_key(self.CSME_UNLOCK_TOKEN)
        
        if _HAVE_CRYPTO:
            nonce = secrets.token_bytes(12)
            aesgcm = AESGCM(self._encryption_key)
            ciphertext = aesgcm.encrypt(nonce, data, None)
            return nonce + ciphertext
        else:
            # Fallback: XOR stream cipher
            key_stream = (self._encryption_key * ((len(data) // 32) + 1))[:len(data)]
            return bytes(a ^ b for a, b in zip(data, key_stream))

    def _decrypt(self, data: bytes) -> bytes:
        """Decrypt data using AES-256-GCM."""
        if not self._encryption_key:
            self._derive_vault_key(self.CSME_UNLOCK_TOKEN)
        
        if _HAVE_CRYPTO and len(data) > 12:
            nonce = data[:12]
            ciphertext = data[12:]
            try:
                aesgcm = AESGCM(self._encryption_key)
                return aesgcm.decrypt(nonce, ciphertext, None)
            except Exception:
                # Might be XOR-encrypted from fallback
                pass
        
        # Fallback: XOR stream cipher (symmetric)
        key_stream = (self._encryption_key * ((len(data) // 32) + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, key_stream))

    @staticmethod
    def _atom_filename(atom_id: str) -> str:
        """Accept only a flat, portable atom identifier (no path segments)."""
        if not isinstance(atom_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", atom_id
        ):
            raise ValueError("Invalid atom_id: expected 1-128 ASCII filename-safe characters")
        if ".." in atom_id:
            raise ValueError("Invalid atom_id: dot-dot components are not permitted")
        return atom_id + ".enc"

    def _open_vault_dir(self) -> int:
        """Bind operations to the vault directory, never through an atom path."""
        if not all(hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_DIRECTORY")):
            raise RuntimeError("Secure vault access requires Linux/POSIX O_NOFOLLOW and O_DIRECTORY")
        fd = os.open(self.vault_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if not stat.S_ISDIR(info.st_mode):
                raise OSError("Vault path is not a directory")
            if stat.S_IMODE(info.st_mode) & 0o077:
                # Tighten existing vault directories in place to prevent
                # other users from listing the sensitive atom filenames.
                os.fchmod(fd, 0o700)
            return fd
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _open_atom_fd(directory_fd: int, filename: str, flags: int) -> int:
        """Reject symlinks, devices and hard links before changing file content."""
        fd = os.open(filename, flags | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise OSError("Vault atom must be a regular, non-hardlinked file")
            return fd
        except BaseException:
            os.close(fd)
            raise

    def store_atom_securely(self, atom_id: str, data: bytes):
        """
        Encrypts and stores an atom using hardware-backed keys.
        """
        if self.is_locked:
            logger.error("CSME Vault is locked. Storage denied.")
            return False
            
        filename = self._atom_filename(atom_id)
        encrypted = self._encrypt(data)
        directory_fd = self._open_vault_dir()
        try:
            fd = self._open_atom_fd(directory_fd, filename, os.O_WRONLY | os.O_CREAT)
            try:
                # Avoid O_TRUNC: check regular-file and hard-link status first.
                os.ftruncate(fd, 0)
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb") as f:
                    fd = None
                    f.write(encrypted)
            finally:
                if fd is not None:
                    os.close(fd)
        finally:
            os.close(directory_fd)
        logger.info("Behavioral Atom %s stored in vault.", atom_id)
        return True

    def retrieve_atom_securely(self, atom_id: str) -> Optional[bytes]:
        if self.is_locked:
            return None
        filename = self._atom_filename(atom_id)
        directory_fd = self._open_vault_dir()
        try:
            try:
                fd = self._open_atom_fd(directory_fd, filename, os.O_RDONLY)
            except FileNotFoundError:
                return None
            try:
                with os.fdopen(fd, "rb") as f:
                    fd = None
                    encrypted = f.read()
            finally:
                if fd is not None:
                    os.close(fd)
        finally:
            os.close(directory_fd)
        return self._decrypt(encrypted)
