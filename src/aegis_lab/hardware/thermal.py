import os
import logging
from enum import IntEnum
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class ThermalLevel(IntEnum):
    LOW = 0      # < 50°C
    NORMAL = 1   # 50-75°C
    HIGH = 2     # 75-90°C
    CRITICAL = 3 # > 90°C

class ThermalGuardian:
    """
    Monitors system temperature and provides safety triggers
    to the scheduler and orchestrator.
    """
    
    THERMAL_DIR = "/sys/class/thermal"
    
    def __init__(self, high_threshold: float = 75.0, critical_threshold: float = 90.0):
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold
        self.zones = self._discover_zones()
        self._last_temp = 0.0
        self._last_check = 0.0

    def _discover_zones(self) -> List[str]:
        zones = []
        if not os.path.exists(self.THERMAL_DIR):
            logger.warning("Thermal sysfs not found. Thermal monitoring disabled.")
            return zones
            
        for d in os.listdir(self.THERMAL_DIR):
            if d.startswith("thermal_zone"):
                zones.append(os.path.join(self.THERMAL_DIR, d))
        return zones

    def get_max_temperature(self) -> float:
        import time
        now = time.time()
        if now - self._last_check < 1.0:
            return self._last_temp
            
        max_temp = 0.0
        for zone in self.zones:
            try:
                with open(os.path.join(zone, "temp"), "r") as f:
                    # Temp is usually in millidegrees Celsius
                    temp = float(f.read().strip()) / 1000.0
                    if temp > max_temp:
                        max_temp = temp
            except (IOError, ValueError):
                continue
                
        self._last_temp = max_temp
        self._last_check = now
        return max_temp

    def get_thermal_level(self) -> ThermalLevel:
        temp = self.get_max_temperature()
        
        if temp >= self.critical_threshold:
            return ThermalLevel.CRITICAL
        if temp >= self.high_threshold:
            return ThermalLevel.HIGH
        if temp >= 50.0:
            return ThermalLevel.NORMAL
        return ThermalLevel.LOW

    def get_status(self) -> Dict[str, Any]:
        temp = self.get_max_temperature()
        level = self.get_thermal_level()
        return {
            "temperature": temp,
            "level": level.name,
            "safe_to_compute": level < ThermalLevel.CRITICAL,
            "throttling_recommended": level >= ThermalLevel.HIGH
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tg = ThermalGuardian()
    print("Thermal Status:", tg.get_status())
