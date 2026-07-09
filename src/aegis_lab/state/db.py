import json
import os
import uuid
import logging
import threading
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from .qihse_wrapper import QIHSE
from enum import IntEnum
from framewerx.state.db import QihseKVStore, _DEFAULT_STORAGE_ROOT as _FW_DEFAULT_ROOT
from framewerx.state.qihse_paths import resolve_qihse_lib_path

class QihseVectorDBBackend(IntEnum):
    FAISS = 0
    CHROMA = 1
    QDRANT = 2
    INMEMORY = 3
    AUTO = 4

InMemoryQIHSE = QIHSE

logger = logging.getLogger(__name__)

class QihseStore:
    """
    Unified QIHSE storage abstraction handling persistence, versioning, 
    and vector-based retrieval for a specific table/entity type.
    """
    def __init__(self, qihse: QIHSE, table_name: str, db_path: str):
        self.qihse = qihse
        self.table_name = table_name
        self.db_path = db_path
        self._lock = threading.RLock()
        logger.info(f"Initializing QIHSE Store: {table_name} at {db_path}")
        self.db_handle = self.qihse.create_vector_db(
            backend=QihseVectorDBBackend.AUTO, 
            db_path=db_path
        )
        # Default vector for non-vectorized metadata (state domain)
        self.default_vector = [1.0] * 128

    def upsert(self, pk_field: str, data: Dict[str, Any], vector: Optional[List[float]] = None):
        """
        Inserts or updates a record. QIHSE is append-only, so we rely on 
        timestamp-based deduplication during retrieval.
        """
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            if 'created_at' not in data:
                data['created_at'] = now
            data['updated_at'] = now
            
            target_vector = vector if vector is not None else self.default_vector
            metadata = json.dumps(data)
            
            self.qihse.add_to_vector_db(self.db_handle, [target_vector], [metadata])
            return data

    def query(self, 
              filters: Optional[Dict[str, Any]] = None, 
              vector: Optional[List[float]] = None, 
              top_k: int = 5000,
              pk_field: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Queries the store. If a vector is provided, performs semantic search.
        Otherwise, performs a broad search for all items in the domain.
        """
        with self._lock:
            target_vector = vector if vector is not None else self.default_vector
            results = self.qihse.search_vector_db(self.db_handle, target_vector, top_k=top_k)
            
            items_map = {}
            for res in results:
                if res['metadata']:
                    try:
                        if isinstance(res['metadata'], bytes):
                            meta_str = res['metadata'].decode('utf-8')
                        else:
                            meta_str = res['metadata']
                        item = json.loads(meta_str)
                        
                        # Deduplicate by primary key (latest update wins)
                        # Use provided pk_field, or try to infer, or fallback to internal ID
                        pk = None
                        if pk_field:
                            pk = item.get(pk_field)
                        
                        if pk is None:
                            # Inference logic for known tables
                            pk = item.get('stage_id') or item.get('atom_id') or \
                                 item.get('artifact_id') or item.get('run_id') or \
                                 item.get('snapshot_id') or item.get('eval_id') or \
                                 item.get('log_id') or item.get('job_id') or \
                                 item.get('id') or res['id']
                        
                        if pk in items_map:
                            existing_ts = items_map[pk].get('updated_at', '')
                            new_ts = item.get('updated_at', '')
                            if new_ts > existing_ts:
                                items_map[pk] = item
                        else:
                            items_map[pk] = item
                    except (json.JSONDecodeError, KeyError):
                        continue
            
            all_items = list(items_map.values())
            
            # Apply filters
            if filters:
                filtered = []
                for item in all_items:
                    match = True
                    for k, v in filters.items():
                        if item.get(k) != v:
                            match = False
                            break
                    if match:
                        filtered.append(item)
                return filtered
                
            return all_items

_AEGIS_DEFAULT_ROOT = os.environ.get(
    "AEGIS_STATE_ROOT",
    os.path.join(os.path.expanduser("~"), ".framewerx", "aegis"),
)


class StateDatabase:
    """
    Manager for multiple Stores, maintaining the 'Source of Truth'.
    Uses QIHSE for atoms/intel tables and SQLite for structured state.
    """
    def __init__(self, storage_root: str = _AEGIS_DEFAULT_ROOT, lib_path: str = None):
        self.storage_root = storage_root
        os.makedirs(storage_root, exist_ok=True)
        if lib_path is None:
            lib_path = resolve_qihse_lib_path()
        require_native = os.getenv("AEGIS_REQUIRE_NATIVE_QIHSE", "").lower() in {"1", "true", "yes"}
        try:
            self.qihse = QIHSE(lib_path)
        except FileNotFoundError:
            if require_native:
                raise
            logger.warning("QIHSE native library missing at %s; falling back to in-memory store.", lib_path)
            self.qihse = InMemoryQIHSE(lib_path)
        self.stores: Dict[str, Union[QihseStore, QihseKVStore]] = {}
        self._stores_lock = threading.Lock()

    def _get_store(self, table_name: str) -> Union[QihseStore, QihseKVStore]:
        with self._stores_lock:
            if table_name not in self.stores:
                if table_name == "atoms" or table_name.startswith(("intel_", "vuln_")):
                    db_path = os.path.join(self.storage_root, f"{table_name}.qihse")
                    self.stores[table_name] = QihseStore(self.qihse, table_name, db_path)
                else:
                    kv_path = os.path.join(self.storage_root, f"{table_name}.qkv")
                    self.stores[table_name] = QihseKVStore(table_name, kv_path)
            return self.stores[table_name]

    def upsert(self, table_name: str, pk_field: str, pk_value: Any, data: Dict[str, Any], vector: Optional[List[float]] = None):
        # Ensure pk_value matches pk_field in data for consistency
        data[pk_field] = pk_value
        return self._get_store(table_name).upsert(pk_field, data, vector)

    def insert(self, table_name: str, data: Dict[str, Any], vector: Optional[List[float]] = None):
        """Legacy alias for upsert to support existing registry calls."""
        # Find a suitable PK field
        pk_field = next((k for k in data.keys() if k.endswith('_id')), 'id')
        return self._get_store(table_name).upsert(pk_field, data, vector)

    def query(self, table_name: str, filters: Optional[Dict[str, Any]] = None, vector: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        return self._get_store(table_name).query(filters, vector)

    def list_all(self, table_name: str) -> List[Dict[str, Any]]:
        """Legacy alias for query without filters."""
        return self._get_store(table_name).query()

class AegisState:
    def __init__(self, storage_root: Optional[str] = None, lib_path: Optional[str] = None):
        if not storage_root:
            storage_root = _AEGIS_DEFAULT_ROOT
        if not lib_path:
            lib_path = resolve_qihse_lib_path()
        logger.info(f"AegisState initializing — storage: {storage_root}, lib: {lib_path}")
        self.db = StateDatabase(storage_root, lib_path)
        self.qihse = self.db.qihse
        
    # --- Jobs ---
    def create_job(self, job_id: str, project_id: str, job_type: str, priority: int = 50) -> Dict[str, Any]:
        job_data = {
            "job_id": job_id,
            "project_id": project_id,
            "job_type": job_type,
            "status": "pending",
            "priority": priority,
            "current_stage_id": None
        }
        return self.db.upsert("jobs", "job_id", job_id, job_data)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        jobs = self.db.query("jobs", {"job_id": job_id})
        return jobs[0] if jobs else None

    def get_jobs(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self.db._get_store("jobs").query(filters, pk_field="job_id")

    def update_job(self, job_id: str, updates: Dict[str, Any]):
        job = self.get_job(job_id)
        if job:
            job.update(updates)
            self.db.upsert("jobs", "job_id", job_id, job)

    # --- Stages ---
    def create_stage(self, stage_id: str, job_id: str, stage_name: str, ordinal: int) -> Dict[str, Any]:
        stage_data = {
            "stage_id": stage_id,
            "job_id": job_id,
            "stage_name": stage_name,
            "ordinal": ordinal,
            "status": "pending"
        }
        return self.db.upsert("stages", "stage_id", stage_id, stage_data)

    def get_stage(self, stage_id: str) -> Optional[Dict[str, Any]]:
        stages = self.db.query("stages", {"stage_id": stage_id})
        return stages[0] if stages else None

    def get_stages(self, job_id_or_filters) -> List[Dict[str, Any]]:
        if isinstance(job_id_or_filters, dict):
            return self.db._get_store("stages").query(job_id_or_filters, pk_field="stage_id")
        return self.db._get_store("stages").query({"job_id": job_id_or_filters}, pk_field="stage_id")

    def get_all_stages(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self.db._get_store("stages").query(filters, pk_field="stage_id")

    def update_stage(self, stage_id: str, updates: Dict[str, Any]):
        stages = self.db.query("stages", {"stage_id": stage_id})
        if stages:
            stage = stages[0]
            stage.update(updates)
            self.db.upsert("stages", "stage_id", stage_id, stage)

    # --- Atoms (Behavioral Atom Registry) ---
    def register_atom(self, atom_data: Dict[str, Any], semantic_vector: Optional[List[float]] = None):
        return self.db.upsert("atoms", "atom_id", atom_data["atom_id"], atom_data, vector=semantic_vector)

    def get_atom(self, atom_id: str) -> Optional[Dict[str, Any]]:
        atoms = self.db.query("atoms", {"atom_id": atom_id})
        return atoms[0] if atoms else None

    def search_atoms(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        return self.db.query("atoms", vector=query_vector, filters=None) # QihseStore.query handles top_k internally or we can expose it

    # --- Artifacts (Metadata mapping to content hashes) ---
    def register_artifact(self, artifact_id: str, metadata: Dict[str, Any]):
        metadata["artifact_id"] = artifact_id
        return self.db.upsert("artifacts", "artifact_id", artifact_id, metadata)

    def get_artifact_metadata(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        results = self.db.query("artifacts", {"artifact_id": artifact_id})
        return results[0] if results else None

    # --- Sentinel Runs (NPU watchdog telemetry) ---
    def record_sentinel_run(self, run_id: str, telemetry: Dict[str, Any]):
        telemetry["run_id"] = run_id
        return self.db.upsert("sentinel_runs", "run_id", run_id, telemetry)

    def get_sentinel_runs(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self.db.query("sentinel_runs", filters)

    # --- Hardware Snapshots (Historical thermal/usage data) ---
    def record_hardware_snapshot(self, snapshot_id: str, snapshot_data: Dict[str, Any]):
        snapshot_data["snapshot_id"] = snapshot_id
        return self.db.upsert("hardware_snapshots", "snapshot_id", snapshot_id, snapshot_data)

    def get_hardware_snapshots(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self.db.query("hardware_snapshots", filters)

    # --- Evaluation Scores (Performance, Elo, Adversarial, Integrity) ---
    def record_evaluation_score(self, eval_id: str, eval_data: Dict[str, Any]):
        """
        Records an evaluation score for a specific job or artifact.
        
        Args:
            eval_id: Unique identifier for this evaluation run.
            eval_data: Dictionary containing scores and metadata.
        """
        eval_data["eval_id"] = eval_id
        return self.db.upsert("evaluation_scores", "eval_id", eval_id, eval_data)

    def get_evaluation_scores(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves evaluation scores, optionally filtered by job_id, artifact_id, etc.
        """
        return self.db.query("evaluation_scores", filters)

    # --- Distributed Logs ---
    def add_log(self, log_entry: Dict[str, Any]):
        log_id = f"log-{uuid.uuid4().hex[:8]}"
        log_entry["log_id"] = log_id
        return self.db.upsert("logs", "log_id", log_id, log_entry)

    def get_logs(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self.db.query("logs", filters)
