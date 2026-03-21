import os
import subprocess
import logging
from typing import List

logger = logging.getLogger(__name__)

class CudaBridge:
    """
    Manages the execution of CUDA-based binaries on Intel hardware via ZLUDA.
    """
    
    def __init__(self, zluda_path: str = "/usr/local/bin/zluda"):
        self.zluda_path = os.getenv("ZLUDA_BIN", zluda_path)
        self.is_available = os.path.exists(self.zluda_path)

    def wrap_command(self, cmd: List[str]) -> List[str]:
        """
        Wraps a standard CUDA command with the ZLUDA translation layer.
        Example: ['nvidia-smi'] -> ['zluda', '--', 'nvidia-smi']
        """
        if not self.is_available:
            logger.warning("ZLUDA not found. Attempting raw command execution (likely to fail).")
            return cmd
            
        return [self.zluda_path, "--"] + cmd

    def run_cuda_kernel(self, binary_path: str, args: List[str]):
        """
        Executes a CUDA binary targeting the Intel GPU.
        """
        full_cmd = self.wrap_command([binary_path] + args)
        logger.info(f"Executing CUDA-on-Intel: {' '.join(full_cmd)}")
        
        env = os.environ.copy()
        env["ZE_ENABLE_SYSMAN"] = "1" # Required for Level Zero metrics
        
        return subprocess.run(full_cmd, env=env, capture_output=True, text=True)
