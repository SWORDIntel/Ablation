import ctypes
import os
import logging
from enum import IntEnum

class QihseVectorDBBackend(IntEnum):
    FAISS = 0
    CHROMA = 1
    QDRANT = 2
    INMEMORY = 3
    AUTO = 4

class QihseUmaMigrationPolicy(IntEnum):
    MIGRATE_ON_ACCESS = 0
    MIGRATE_PREFETCH = 1
    MIGRATE_EXPLICIT = 2
    MIGRATE_LAZY = 3

logger = logging.getLogger(__name__)

class QIHSE:
    def __init__(self, lib_path = None):
        # In-memory storage for test emulation
        self._db_storage = {} # handle -> {id -> (vector, metadata)}
        self._pinning = {}    # handle -> {id -> priority}
        self._policy = {}     # handle -> policy

        if lib_path is None:
            possible_paths = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../../native/vpu_core/target/release/libvpu_core.so")),
                "/home/john/Documents/MEMSHADOW/QIHSE/qihse/libqihse.so",
                "/usr/local/lib/libvpu_core.so"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    lib_path = path
                    break
        
        if lib_path and os.path.exists(lib_path):
            try:
                self.lib = ctypes.CDLL(lib_path)
                logger.info("Successfully loaded native QIHSE library from %s", lib_path)
            except Exception as e:
                logger.error("Failed to load native library: %s", e)
                self.lib = None
        else:
            logger.warning("QIHSE native library not found. Falling back to in-memory mode.")
            self.lib = None

    def create_vector_db(self, backend, db_path):
        handle = id(db_path)
        self._db_storage[handle] = {}
        return handle

    def add_to_vector_db(self, handle, vectors, metadata_list):
        if handle not in self._db_storage:
            self._db_storage[handle] = {}
        
        for i, (vec, meta) in enumerate(zip(vectors, metadata_list)):
            vec_id = len(self._db_storage[handle]) + i
            self._db_storage[handle][vec_id] = (vec, meta)

    def search_vector_db(self, handle, vector, top_k):
        results = []
        if handle in self._db_storage:
            for vec_id, (v, m) in self._db_storage[handle].items():
                results.append({"id": vec_id, "metadata": m})
                if len(results) >= top_k:
                    break
        return results

    def set_priority_pinning(self, handle, vector_ids, priority):
        if handle not in self._pinning:
            self._pinning[handle] = {}
        for vid in vector_ids:
            self._pinning[handle][vid] = priority

    def set_migration_policy(self, handle, policy):
        self._policy[handle] = policy
