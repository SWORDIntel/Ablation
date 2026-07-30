import time
import logging
import threading
from typing import Dict, Any, List, Optional
from aegis_lab.hardware.telemetry import LevelZeroTelemetry
from aegis_lab.state.db import AegisState

logger = logging.getLogger(__name__)

class HardwareOptimizer:
    """
    Dynamic UMA Memory Policy Tuning daemon.
    Feedback loop between hardware telemetry and QIHSE memory management.
    """
    
    def __init__(self, state: AegisState, telemetry: LevelZeroTelemetry, interval: int = 2):
        self.state = state
        self.telemetry = telemetry
        self.interval = interval
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._pinned_vectors: Dict[str, List[int]] = {} # store_name -> list of pinned IDs
        
        # Thresholds from 'Golden Bible' performance truths
        self.POWER_THRESHOLD_WATTS = 45.0
        self.BANDWIDTH_THRESHOLD_PCT = 80.0

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Hardware Optimizer daemon started.")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while self.running:
            try:
                stats = self.telemetry.get_metrics()
                if not stats:
                    time.sleep(self.interval)
                    continue
                
                # Check for pressure points
                power = stats.get("gpu_power_watts", 0.0)
                # Note: bandwidth might need more complex calculation in telemetry.py
                # but we'll assume a simplified metric for now
                bandwidth = stats.get("memory_bandwidth_util", 0.0) 
                
                if power > self.POWER_THRESHOLD_WATTS or bandwidth > self.BANDWIDTH_THRESHOLD_PCT:
                    logger.warning(f"High hardware pressure detected (Power: {power}W, BW: {bandwidth}%). Optimizing UMA...")
                    self._optimize_memory_layout()
                else:
                    # Pressure dropped, unpin to free SRAM
                    if self._pinned_vectors:
                        logger.info("Hardware pressure normalized. Releasing NPU SRAM pins...")
                        self._release_npu_cache()
                    
            except Exception as e:
                logger.error(f"Hardware Optimizer error: {e}")
            
            time.sleep(self.interval)

    def _optimize_memory_layout(self):
        """
        Identify 'hot' behavioral atoms and pin them to the 128MB NPU SRAM cache,
        applying explicit migration policies.
        """
        from aegis_lab.state.qihse_wrapper import QihseUmaMigrationPolicy
        
        # Query recently accessed or important atoms
        atom_store = self.state.db._get_store("atoms")
        
        # For demo, let's assume we want to pin the top 10 most recent atoms
        results = self.state.qihse.search_vector_db(atom_store.db_handle, [1.0]*128, top_k=10)
        vector_ids = [res["id"] for res in results]
        
        if vector_ids:
            logger.info(f"Pinning {len(vector_ids)} vectors to NPU cache using MIGRATE_PREFETCH policy.")
            # Explicitly set the migration policy to PREFETCH for hot atoms
            self.state.qihse.set_priority_pinning(atom_store.db_handle, vector_ids, priority=2)
            # Applying policy migration as per architecture spec
            self.state.qihse.set_migration_policy(atom_store.db_handle, QihseUmaMigrationPolicy.MIGRATE_PREFETCH)
            self._pinned_vectors["atoms"] = vector_ids

    def _release_npu_cache(self):
        """
        Release all pinned vectors to free up NPU SRAM.
        """
        for store_name, vector_ids in self._pinned_vectors.items():
            store = self.state.db._get_store(store_name)
            if store:
                logger.debug(f"Unpinning {len(vector_ids)} vectors from {store_name} store.")
                self.state.qihse.set_priority_pinning(store.db_handle, vector_ids, priority=0)
        
        self._pinned_vectors.clear()
