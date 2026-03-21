import logging
from typing import Any

from aegis_lab.artifacts.store import ArtifactStore

logger = logging.getLogger(__name__)

class RollbackController:
    """
    Implements rollback snapshots using the content-addressed artifact store.
    Adheres to the AEGIS-LAB Rollback Controller Algorithm (Section 13.2).
    """
    def __init__(self, artifact_store: ArtifactStore, db_session: Any):
        self.artifact_store = artifact_store
        self.db = db_session

    def rollback_to_point(self, rollback_point_id: str) -> bool:
        """
        Executes the rollback algorithm:
        1. resolve requested rollback point
        2. verify rollback point manifest hash
        3. verify artifact presence
        4. restore active manifest pointer
        5. mark superseded artifacts as non-active
        6. write rollback event log
        
        Rollback must not rely on inverse tensor arithmetic alone. 
        It must restore a known-good manifest-defined state.
        """
        logger.info(f"Initiating rollback to point: {rollback_point_id}")
        
        # 1. Resolve requested rollback point
        rollback_point = self.db.get_rollback_point(rollback_point_id)
        if not rollback_point:
            logger.error(f"Rollback point {rollback_point_id} not found.")
            return False
            
        manifest_hash = rollback_point.get('manifest_id')
        artifact_hash = rollback_point.get('artifact_id')
        
        # 2. Verify rollback point manifest hash
        if manifest_hash and not self.artifact_store.exists(manifest_hash):
             logger.error(f"Manifest hash {manifest_hash} missing from artifact store.")
             return False
             
        # 3. Verify artifact presence
        if artifact_hash and not self.artifact_store.exists(artifact_hash):
             logger.error(f"Target artifact {artifact_hash} missing from artifact store.")
             return False
             
        # 4. Restore active manifest pointer
        try:
            self.db.update_active_manifest(manifest_hash)
        except Exception as e:
            logger.error(f"Failed to restore active manifest pointer: {e}")
            return False
            
        # 5. Mark superseded artifacts as non-active
        try:
            self.db.mark_superseded_artifacts_inactive(rollback_point_id)
        except Exception as e:
            logger.error(f"Failed to mark superseded artifacts inactive: {e}")
            return False
            
        # 6. Write rollback event log
        try:
            self.db.log_rollback_event(
                rollback_point_id=rollback_point_id,
                manifest_hash=manifest_hash,
                artifact_hash=artifact_hash,
                status="SUCCESS"
            )
        except Exception as e:
            logger.error(f"Failed to write rollback event log: {e}")
            
        logger.info(f"Successfully rolled back to {rollback_point_id}. Manifest-defined state restored.")
        return True
