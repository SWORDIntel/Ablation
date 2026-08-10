import logging
from typing import Dict, Any, List, Optional
from aegis_lab.state.db import AegisState

logger = logging.getLogger(__name__)

class LeaderboardManager:
    """
    Unified Leaderboard for ranking ablation runs.
    Aggregates Method 1-5 scores into a single 'Ablation Integrity Score'.
    """
    
    def __init__(self, state: AegisState):
        self.state = state

    def record_score(self, job_id: str, metric_name: str, value: float):
        """
        Records a specific metric score in the QIHSE state.
        """
        score_data = {
            "job_id": job_id,
            "metric": metric_name,
            "value": value
        }
        # Assuming 'evaluation_scores' table exists in our QihseStore
        self.state.db.upsert("evaluation_scores", "job_metric_id", f"{job_id}_{metric_name}", score_data)

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """
        Retrieves and ranks all jobs based on aggregated metrics.
        """
        all_scores = self.state.db.query("evaluation_scores")
        
        # Aggregate by Job ID
        rankings = {}
        for s in all_scores:
            jid = s["job_id"]
            if jid not in rankings:
                rankings[jid] = {"job_id": jid, "total_score": 0.0, "metrics": {}}
            
            rankings[jid]["metrics"][s["metric"]] = s["value"]
            # Weighted aggregation
            weight = 0.2 # Equal weight for simplicity in demo
            rankings[jid]["total_score"] += s["value"] * weight
            
        # Sort by total score
        sorted_ranks = sorted(rankings.values(), key=lambda x: x["total_score"], reverse=True)
        return sorted_ranks
