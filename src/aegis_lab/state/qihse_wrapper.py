import ctypes
import os
from enum import IntEnum
from typing import List

class QihseDataType(IntEnum):
    INT64 = 0
    UINT64 = 1
    DOUBLE = 2
    STRING = 3
    BINARY = 4
    STRUCT = 5
    CUSTOM = 6

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

class QihseConfig(ctypes.Structure):
    _fields_ = [
        ("dummy", ctypes.c_byte * 4096)
    ]

class QihseVectorResult(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("score", ctypes.c_float),
        ("vector", ctypes.POINTER(ctypes.c_float)),
        ("vector_dims", ctypes.c_size_t),
        ("metadata", ctypes.c_void_p),
        ("metadata_size", ctypes.c_size_t)
    ]

class QihseVectorQuery(ctypes.Structure):
    _fields_ = [
        ("query_vector", ctypes.POINTER(ctypes.c_float)),
        ("vector_dims", ctypes.c_size_t),
        ("top_k", ctypes.c_size_t),
        ("similarity_threshold", ctypes.c_float),
        ("include_vectors", ctypes.c_bool),
        ("include_metadata", ctypes.c_bool)
    ]

class QihseNpuCache(ctypes.Structure):
    _fields_ = [
        ("cache_size_mb", ctypes.c_size_t),
        ("line_size_bytes", ctypes.c_size_t),
        ("associativity", ctypes.c_size_t),
        ("hit_latency_ns", ctypes.c_double),
        ("miss_penalty_ns", ctypes.c_double),
        ("gna_enabled", ctypes.c_bool),
        ("gna_workgroup_size", ctypes.c_size_t)
    ]

class QIHSE:
    def __init__(self, lib_path):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Could not find {lib_path}")
        self.lib = ctypes.CDLL(lib_path)
        
        # Define argtypes and restypes
        self.lib.qihse_context_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.qihse_context_create.restype = ctypes.c_int
        
        self.lib.qihse_memory_manager_create.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.qihse_memory_manager_create.restype = ctypes.c_void_p
        
        self.lib.qihse_uma_create.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.qihse_uma_create.restype = ctypes.c_void_p

        self.lib.qihse_uma_init_meteor_lake_npu_cache.argtypes = [ctypes.POINTER(QihseNpuCache)]
        self.lib.qihse_uma_init_meteor_lake_npu_cache.restype = ctypes.c_bool

        self.lib.qihse_uma_optimize_for_cache_size.argtypes = [ctypes.c_void_p, ctypes.POINTER(QihseNpuCache)]
        self.lib.qihse_uma_optimize_for_cache_size.restype = ctypes.c_bool

        self.lib.qihse_uma_set_priority_pinning.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_int]
        self.lib.qihse_uma_set_priority_pinning.restype = ctypes.c_bool
        
        self.lib.qihse_config_init.argtypes = [ctypes.POINTER(QihseConfig), ctypes.c_int, ctypes.c_size_t]
        self.lib.qihse_config_init.restype = ctypes.c_int
        
        self.lib.qihse_vector_db_create.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p]
        self.lib.qihse_vector_db_create.restype = ctypes.c_void_p
        
        self.lib.qihse_vector_db_add_vectors.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, 
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64), 
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)
        ]
        self.lib.qihse_vector_db_add_vectors.restype = ctypes.c_bool
        
        self.lib.qihse_vector_db_search.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(QihseVectorQuery), 
            ctypes.POINTER(QihseVectorResult), ctypes.c_size_t
        ]
        self.lib.qihse_vector_db_search.restype = ctypes.c_int

        # Add hardware-accelerated hashing
        try:
            self.lib.qihse_intel_hw_hash.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int
            ]
            self.lib.qihse_intel_hw_hash.restype = ctypes.c_int
            self.has_hw_hash = True
        except AttributeError:
            self.has_hw_hash = False

        # Initialize global state
        self.ctx = ctypes.c_void_p()
        self.lib.qihse_context_create(None, ctypes.byref(self.ctx))
        
        # Apply Lying E820 Memory Map Protection
        from aegis_lab.hardware.discovery import HardwareDiscovery
        npu_bar = HardwareDiscovery.check_npu_bar()
        if npu_bar.get("conflict"):
            os.environ["QIHSE_E820_SAFE_MODE"] = "1"
            
        self.mem_manager = self.lib.qihse_memory_manager_create(self.ctx, b"UMA")
        self.uma = self.lib.qihse_uma_create(self.mem_manager, QihseUmaMigrationPolicy.MIGRATE_EXPLICIT)

        # Hardware-aware optimization
        self.npu_cache = QihseNpuCache()
        if self.lib.qihse_uma_init_meteor_lake_npu_cache(ctypes.byref(self.npu_cache)):
            self.lib.qihse_uma_optimize_for_cache_size(self.uma, ctypes.byref(self.npu_cache))

    def create_vector_db(self, backend=QihseVectorDBBackend.INMEMORY, db_path=None):
        path_bytes = db_path.encode('utf-8') if db_path else None
        return self.lib.qihse_vector_db_create(backend, self.uma, path_bytes)

    def add_to_vector_db(self, db_handle, vectors, metadata_list):
        num_vectors = len(vectors)
        if num_vectors == 0:
            return True
        
        vector_dims = len(vectors[0])
        vectors_flat = (ctypes.c_float * (num_vectors * vector_dims))(*[f for v in vectors for f in v])
        
        metadata_ptrs = (ctypes.c_void_p * num_vectors)()
        metadata_sizes = (ctypes.c_size_t * num_vectors)()
        
        self._metadata_blobs = getattr(self, '_metadata_blobs', [])
        for i, meta in enumerate(metadata_list):
            if isinstance(meta, str):
                meta = meta.encode('utf-8')
            blob = (ctypes.c_byte * len(meta)).from_buffer_copy(meta)
            self._metadata_blobs.append(blob)
            metadata_ptrs[i] = ctypes.cast(ctypes.pointer(blob), ctypes.c_void_p)
            metadata_sizes[i] = len(meta)
            
        return self.lib.qihse_vector_db_add_vectors(
            db_handle, vectors_flat, num_vectors, vector_dims, 
            None, metadata_ptrs, metadata_sizes
        )

    def search_vector_db(self, db_handle, query_vector, top_k=10):
        dims = len(query_vector)
        q_vec = (ctypes.c_float * dims)(*query_vector)
        query = QihseVectorQuery(
            query_vector=ctypes.cast(q_vec, ctypes.POINTER(ctypes.c_float)),
            vector_dims=dims,
            top_k=top_k,
            similarity_threshold=-1.0, # Use -1.0 to get everything if using dummy vectors
            include_vectors=False,
            include_metadata=True
        )
        
        results = (QihseVectorResult * top_k)()
        count = self.lib.qihse_vector_db_search(db_handle, ctypes.byref(query), results, top_k)
        
        final_results = []
        for i in range(count):
            res = results[i]
            meta = None
            if res.metadata:
                meta = ctypes.string_at(res.metadata, res.metadata_size)
            final_results.append({
                "id": res.id,
                "score": res.score,
                "metadata": meta
            })
        return final_results

    def set_priority_pinning(self, db_handle, vector_ids: List[int], priority: int = 1):
        """
        Dynamically pin high-priority vectors to the 128MB NPU SRAM cache.
        """
        count = len(vector_ids)
        ids_array = (ctypes.c_uint64 * count)(*vector_ids)
        return self.lib.qihse_uma_set_priority_pinning(db_handle, ids_array, count, priority)

    def sha256(self, data: bytes) -> str:
        """Computes SHA256 hash, using hardware acceleration if available."""
        if self.has_hw_hash:
            hash_out = (ctypes.c_byte * 32)()
            res = self.lib.qihse_intel_hw_hash(
                data, len(data), ctypes.byref(hash_out), 0 # 0 = SHA-256
            )
            if res == 0:
                return "".join(f"{b & 0xFF:02x}" for b in hash_out)
        
        # Fallback to hashlib
        import hashlib
        return hashlib.sha256(data).hexdigest()
