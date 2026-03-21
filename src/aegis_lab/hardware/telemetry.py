import ctypes
import os
import time
import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# --- Level Zero Types & Constants ---
ze_result_t = ctypes.c_int
zes_driver_handle_t = ctypes.c_void_p
zes_device_handle_t = ctypes.c_void_p
zes_pwr_handle_t = ctypes.c_void_p
zes_freq_handle_t = ctypes.c_void_p
zes_engine_handle_t = ctypes.c_void_p
zes_mem_handle_t = ctypes.c_void_p

ZE_RESULT_SUCCESS = 0

ZE_STRUCTURE_TYPE_SYSMAN_POWER_PROPERTIES = 0x1001000b
ZE_STRUCTURE_TYPE_SYSMAN_POWER_ENERGY_COUNTER = 0x1001000c
ZE_STRUCTURE_TYPE_SYSMAN_FREQ_PROPERTIES = 0x10010007
ZE_STRUCTURE_TYPE_SYSMAN_FREQ_STATE = 0x10010008
ZE_STRUCTURE_TYPE_SYSMAN_ENGINE_PROPERTIES = 0x10010005
ZE_STRUCTURE_TYPE_SYSMAN_ENGINE_STATS = 0x10010006
ZE_STRUCTURE_TYPE_SYSMAN_MEMORY_PROPERTIES = 0x10010001
ZE_STRUCTURE_TYPE_SYSMAN_MEMORY_STATE = 0x10010002

class zes_power_properties_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("onSubdevice", ctypes.c_bool),
        ("subdeviceId", ctypes.c_uint32),
        ("canControl", ctypes.c_bool),
        ("isEnergyThresholdSupported", ctypes.c_bool),
        ("defaultLimit", ctypes.c_int32),
        ("minLimit", ctypes.c_int32),
        ("maxLimit", ctypes.c_int32),
    ]

class zes_power_energy_counter_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("energy", ctypes.c_uint64),
        ("timestamp", ctypes.c_uint64),
    ]

class zes_freq_properties_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("onSubdevice", ctypes.c_bool),
        ("subdeviceId", ctypes.c_uint32),
        ("type", ctypes.c_int), # zes_freq_domain_t
        ("canControl", ctypes.c_bool),
        ("isThrottleEventSupported", ctypes.c_bool),
        ("min", ctypes.c_double),
        ("max", ctypes.c_double),
    ]

class zes_freq_state_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("currentVoltage", ctypes.c_double),
        ("request", ctypes.c_double),
        ("tdp", ctypes.c_double),
        ("efficient", ctypes.c_double),
        ("actual", ctypes.c_double),
        ("throttleReasons", ctypes.c_int),
    ]

class zes_engine_properties_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_int), # zes_engine_group_t
        ("onSubdevice", ctypes.c_bool),
        ("subdeviceId", ctypes.c_uint32),
    ]

class zes_engine_stats_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("activeTime", ctypes.c_uint64),
        ("timestamp", ctypes.c_uint64),
    ]

class zes_mem_properties_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_int), # zes_mem_type_t
        ("onSubdevice", ctypes.c_bool),
        ("subdeviceId", ctypes.c_uint32),
        ("physicalSize", ctypes.c_uint64),
        ("busWidth", ctypes.c_int32),
        ("numChannels", ctypes.c_int32),
    ]

class zes_mem_state_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("free", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
    ]

class LevelZeroTelemetry:
    """
    Python bindings for Level Zero Sysman API to extract
    hardware telemetry for GPU/NPU on Intel Meteor Lake.
    """
    
    _lib = None
    
    def __init__(self):
        self._initialized = False
        self.devices = []
        self._load_library()
        if self._lib:
            self._init_sysman()

    def _load_library(self):
        paths = [
            "/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1",
            "/usr/lib/x86_64-linux-gnu/libze_loader.so.1",
            "libze_loader.so.1",
            "libze_intel_gpu.so.1"
        ]
        for path in paths:
            try:
                self._lib = ctypes.CDLL(path)
                logger.info(f"Loaded Level Zero library: {path}")
                break
            except OSError:
                continue
        
        if not self._lib:
            logger.warning("Level Zero library not found.")

    def _init_sysman(self):
        try:
            # zesInit
            res = self._lib.zesInit(0)
            if res != ZE_RESULT_SUCCESS:
                logger.error(f"zesInit failed with error {res}")
                return

            # Get drivers
            count = ctypes.c_uint32(0)
            self._lib.zesDriverGet(ctypes.byref(count), None)
            if count.value == 0:
                logger.warning("No Level Zero drivers found.")
                return
            
            drivers = (zes_driver_handle_t * count.value)()
            self._lib.zesDriverGet(ctypes.byref(count), drivers)
            
            # Get devices for the first driver (usually enough for MTL-P)
            for i in range(count.value):
                dev_count = ctypes.c_uint32(0)
                self._lib.zesDeviceGet(drivers[i], ctypes.byref(dev_count), None)
                if dev_count.value > 0:
                    dev_handles = (zes_device_handle_t * dev_count.value)()
                    self._lib.zesDeviceGet(drivers[i], ctypes.byref(dev_count), dev_handles)
                    for j in range(dev_count.value):
                        self.devices.append(dev_handles[j])
            
            self._initialized = len(self.devices) > 0
            logger.info(f"Initialized Sysman with {len(self.devices)} devices.")
        except Exception as e:
            logger.error(f"Failed to initialize Sysman: {e}")

    def get_metrics(self) -> List[Dict[str, Any]]:
        """Returns telemetry metrics for all discovered devices."""
        if not self._initialized:
            return []
            
        all_metrics = []
        for hDevice in self.devices:
            metrics = {
                "power_w": 0.0,
                "freq_mhz": 0.0,
                "utilization": 0.0,
                "memory_used_mb": 0.0,
                "memory_total_mb": 0.0,
                "occupancy": 0.0
            }
            
            # 1. Power Usage
            try:
                p_count = ctypes.c_uint32(0)
                self._lib.zesDeviceEnumPowerDomains(hDevice, ctypes.byref(p_count), None)
                if p_count.value > 0:
                    p_handles = (zes_pwr_handle_t * p_count.value)()
                    self._lib.zesDeviceEnumPowerDomains(hDevice, ctypes.byref(p_count), p_handles)
                    
                    # Get energy for the first domain (usually package/GPU)
                    energy1 = zes_power_energy_counter_t(stype=ZE_STRUCTURE_TYPE_SYSMAN_POWER_ENERGY_COUNTER)
                    self._lib.zesPowerGetEnergyCounter(p_handles[0], ctypes.byref(energy1))
                    time.sleep(0.1)
                    energy2 = zes_power_energy_counter_t(stype=ZE_STRUCTURE_TYPE_SYSMAN_POWER_ENERGY_COUNTER)
                    self._lib.zesPowerGetEnergyCounter(p_handles[0], ctypes.byref(energy2))
                    
                    # Power in Watts = (Joules2 - Joules1) / (Time2 - Time1)
                    # Energy is in micro-joules, timestamp is in micro-seconds
                    de = energy2.energy - energy1.energy
                    dt = energy2.timestamp - energy1.timestamp
                    if dt > 0:
                        metrics["power_w"] = (de / dt) # (uJ / us) = Watts
            except:
                pass

            # 2. Frequency
            try:
                f_count = ctypes.c_uint32(0)
                self._lib.zesDeviceEnumFrequencyDomains(hDevice, ctypes.byref(f_count), None)
                if f_count.value > 0:
                    f_handles = (zes_freq_handle_t * f_count.value)()
                    self._lib.zesDeviceEnumFrequencyDomains(hDevice, ctypes.byref(f_count), f_handles)
                    
                    state = zes_freq_state_t(stype=ZE_STRUCTURE_TYPE_SYSMAN_FREQ_STATE)
                    self._lib.zesFrequencyGetState(f_handles[0], ctypes.byref(state))
                    metrics["freq_mhz"] = state.actual
            except:
                pass

            # 3. Engine Utilization (Occupancy)
            try:
                e_count = ctypes.c_uint32(0)
                self._lib.zesDeviceEnumEngineGroups(hDevice, ctypes.byref(e_count), None)
                if e_count.value > 0:
                    e_handles = (zes_engine_handle_t * e_count.value)()
                    self._lib.zesDeviceEnumEngineGroups(hDevice, ctypes.byref(e_count), e_handles)
                    
                    # Average utilization across all engines
                    total_util = 0.0
                    for k in range(e_count.value):
                        stats1 = zes_engine_stats_t(stype=ZE_STRUCTURE_TYPE_SYSMAN_ENGINE_STATS)
                        self._lib.zesEngineGetStats(e_handles[k], ctypes.byref(stats1))
                        time.sleep(0.01)
                        stats2 = zes_engine_stats_t(stype=ZE_STRUCTURE_TYPE_SYSMAN_ENGINE_STATS)
                        self._lib.zesEngineGetStats(e_handles[k], ctypes.byref(stats2))
                        
                        da = stats2.activeTime - stats1.activeTime
                        dt = stats2.timestamp - stats1.timestamp
                        if dt > 0:
                            total_util += (da / dt)
                    
                    metrics["utilization"] = (total_util / e_count.value) * 100.0
                    metrics["occupancy"] = metrics["utilization"] # synonym in this context
            except:
                pass

            # 4. Memory / SRAM
            try:
                m_count = ctypes.c_uint32(0)
                self._lib.zesDeviceEnumMemoryModules(hDevice, ctypes.byref(m_count), None)
                if m_count.value > 0:
                    m_handles = (zes_mem_handle_t * m_count.value)()
                    self._lib.zesDeviceEnumMemoryModules(hDevice, ctypes.byref(m_count), m_handles)
                    
                    state = zes_mem_state_t(stype=ZE_STRUCTURE_TYPE_SYSMAN_MEMORY_STATE)
                    self._lib.zesMemoryGetState(m_handles[0], ctypes.byref(state))
                    metrics["memory_total_mb"] = state.size / (1024 * 1024)
                    metrics["memory_used_mb"] = (state.size - state.free) / (1024 * 1024)
            except:
                pass
                
            all_metrics.append(metrics)
            
        return all_metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    telemetry = LevelZeroTelemetry()
    while True:
        results = telemetry.get_metrics()
        print(f"Metrics: {results}")
        time.sleep(1)
