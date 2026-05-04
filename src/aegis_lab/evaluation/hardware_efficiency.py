import logging
from typing import Dict, Any, List
from aegis_lab.hardware.telemetry import LevelZeroTelemetry

logger = logging.getLogger(__name__)

class EfficiencyRanker:
    """
    Method 4: Performance-per-Watt (Hardware Efficiency).
    Ranks models based on tokens/sec per Watt on MTL-P silicon.
    """
    
    def __init__(self, telemetry: LevelZeroTelemetry):
        self.telemetry = telemetry

    def compute_efficiency_score(self, tokens_per_sec: float) -> float:
        """
        Returns a normalized efficiency score.
        Calculation: (Tokens/Sec) / (GPU_Power_Watts + 1.0)
        """
        metrics = self.telemetry.get_metrics()
        if isinstance(metrics, list):
            metrics = metrics[0] if metrics else {}
        power = metrics.get("gpu_power_watts", 15.0) # Fallback to idle power
        
        if power <= 0: power = 1.0
        
        efficiency = tokens_per_sec / power
        logger.info(f"MTL-P Efficiency Calculated: {efficiency:.4f} T/s/W")
        
        # Normalize score (0.0 - 1.0) based on typical MTL-P limits
        # Max tokens/sec ~100, Max power ~45W -> Max efficiency ~2.2
        normalized = min(efficiency / 2.5, 1.0)
        return normalized
