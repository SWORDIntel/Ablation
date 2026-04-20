import ctypes
import os

class TelemetryProcessor:
    def __init__(self, lib_path: str):
        self.lib = ctypes.CDLL(lib_path)
        # Assuming simple init/process interface
        self.lib.process_telemetry.argtypes = [ctypes.c_char_p]
        self.lib.process_telemetry.restype = ctypes.c_int

    def process(self, data_stream: bytes) -> int:
        return self.lib.process_telemetry(data_stream)
