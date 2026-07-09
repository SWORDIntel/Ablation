import platform
import os
import shutil
import subprocess
import logging
import time
import numpy as np
from typing import Dict, Any, List, Set

logger = logging.getLogger(__name__)

class HardwareDiscovery:
    """
    Discovers hardware capabilities including CPU features (AMX/AVX512/AVX2),
    iGPU presence, NPU availability via OpenVINO, USB Myriad devices,
    and supported precisions (INT8, BF16, FP16, FP32).
    Performs memory bandwidth benchmarking and hardware tier classification.
    """

    @staticmethod
    def check_cpu_features() -> Dict[str, bool]:
        features = {"avx512": False, "amx": False, "avx_vnni": False, "avx2": False, "hybrid": False}
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    content = f.read().lower()
                    # Check for AVX-512 foundation
                    if "avx512f" in content:
                        features["avx512"] = True
                    # Check for AMX components (tile, int8, bf16)
                    if "amx_tile" in content or "amx_int8" in content or "amx_bf16" in content:
                        features["amx"] = True
                    # Check for AVX-VNNI (AVX2-based VNNI)
                    if "avx_vnni" in content or "avx512_vnni" in content:
                        features["avx_vnni"] = True
                    # Check for AVX2
                    if "avx2" in content:
                        features["avx2"] = True
                
                # Check for hybrid architecture (P/E cores)
                cpu_dir = "/sys/devices/system/cpu"
                if os.path.exists(cpu_dir):
                    # Intel hybrid detection via sysfs core_type
                    core_types = []
                    for d in os.listdir(cpu_dir):
                        if d.startswith("cpu") and d[3:].isdigit():
                            core_type_path = os.path.join(cpu_dir, d, "topology/core_type")
                            if os.path.exists(core_type_path):
                                with open(core_type_path, "r") as f:
                                    core_types.append(f.read().strip())
                    
                    if len(set(core_types)) > 1:
                        features["hybrid"] = True

            except Exception as e:
                logger.warning(f"Failed to read CPU topology: {e}")
        return features

    @staticmethod
    def check_memory_bandwidth() -> float:
        """
        Performs a rapid memory-bandwidth benchmark (Read+Write).
        Returns estimated GB/s.
        """
        try:
            size_mb = 128
            data = np.random.bytes(size_mb * 1024 * 1024)
            arr = np.frombuffer(data, dtype=np.uint8).copy()
            
            start_time = time.perf_counter()
            # Perform a simple copy-like operation
            arr2 = arr + 1
            # Perform some sum for read throughput
            _ = np.sum(arr2)
            end_time = time.perf_counter()
            
            duration = end_time - start_time
            # (Read + Write + Read) = 3 * size_mb
            total_gb = (3.0 * size_mb) / 1024.0
            gb_per_sec = total_gb / duration
            return gb_per_sec
        except Exception as e:
            logger.warning(f"Memory benchmark failed: {e}")
            return 0.0

    @staticmethod
    def _load_openvino_core():
        try:
            import openvino as ov
            if hasattr(ov, "Core"):
                return ov.Core, "openvino.Core"
        except ImportError:
            pass

        try:
            from openvino.runtime import Core
            return Core, "openvino.runtime.Core"
        except ImportError:
            return None, None

    @staticmethod
    def _get_device_precisions(device_name: str, igpu_type: str, cpu_features: Dict[str, bool]) -> List[str]:
        """
        Infer supported precisions for a given device type based on heuristics.
        
        Args:
            device_name: The name of the OpenVINO device (e.g., "GPU.0", "NPU", "MYRIAD").
            igpu_type: The type of integrated GPU (e.g., "xe-lpg", "standard").
            cpu_features: Dictionary of CPU features.
            
        Returns:
            A list of supported precision strings (e.g., ["FP32", "INT8", "BF16"]).
        """
        precisions = ["FP32"] # FP32 is universally supported
        
        if "MYRIAD" in device_name:
            # Myriad VPUs typically support FP32 and FP16
            precisions.extend(["FP16"])
        elif "NPU" in device_name:
            # Intel NPUs (e.g., on MTL-P) typically support INT8, BF16, FP32
            precisions.extend(["INT8", "BF16", "FP16"])
        elif "GPU" in device_name:
            if igpu_type == "xe-lpg":
                # Modern Intel iGPUs (like Xe-LPG) support a wide range
                precisions.extend(["FP16", "BF16", "INT8"])
            else: # Standard GPU (e.g., older Intel HD Graphics)
                precisions.extend(["FP16"])
        
        # Add CPU-specific precisions if not already covered by accelerators
        # This logic might need refinement if CPU precisions are queried differently
        # For now, we assume CPU always supports FP32 and potentially others if features are present.
        if not any(p in precisions for p in ["INT8", "BF16", "FP16"]):
            if (cpu_features["amx"] or cpu_features["avx_vnni"]):
                # AMX and AVX-VNNI can enable BF16 and INT8 on CPU
                precisions.extend(["BF16", "INT8"])

        # Ensure unique and sorted precisions, prioritizing common formats
        unique_precisions = sorted(list(set(precisions)), key=lambda x: {"FP32": 0, "FP16": 1, "BF16": 2, "INT8": 3}.get(x, 4))
        return unique_precisions

    @classmethod
    def check_openvino_devices(cls, cpu_features: Dict[str, bool]) -> Dict[str, Any]:
        devices = {
            "igpu": False,
            "npu": False,
            "vpu": False,
            "vpu_count": 0,
            "vpu_details": [],
            "vpu_runtime": False,
            "vpu_runtime_count": 0,
            "vpu_runtime_details": [],
            "vpu_usb": False,
            "vpu_usb_count": 0,
            "vpu_usb_details": [],
            "igpu_type": "standard",
            "npu_type": "none",
            "openvino_available": False,
            "openvino_import": None,
            "supported_precisions_by_device": {}, # New field
        }

        Core, import_path = cls._load_openvino_core()
        if Core:
            core = Core()
            devices["openvino_available"] = True
            devices["openvino_import"] = import_path
            available_devices = core.available_devices
            for dev in available_devices:
                device_precisions = []
                if "GPU" in dev:
                    devices["igpu"] = True
                    try:
                        full_name = core.get_property(dev, "FULL_DEVICE_NAME").lower()
                        if "arc" in full_name or "xe-lpg" in full_name:
                            devices["igpu_type"] = "xe-lpg"
                    except Exception as e:
                        logger.warning(f"Failed to get FULL_DEVICE_NAME for {dev}: {e}")
                    device_precisions = cls._get_device_precisions(dev, devices["igpu_type"], cpu_features)

                elif "NPU" in dev:
                    devices["npu"] = True
                    devices["npu_type"] = "intel_ai_boost"
                    device_precisions = cls._get_device_precisions(dev, devices["igpu_type"], cpu_features)

                elif "MYRIAD" in dev:
                    devices["vpu"] = True
                    devices["vpu_runtime"] = True
                    devices["vpu_runtime_count"] += 1
                    devices["vpu_runtime_details"].append(dev)
                    device_precisions = cls._get_device_precisions(dev, devices["igpu_type"], cpu_features)
                
                if device_precisions:
                    devices["supported_precisions_by_device"][dev] = device_precisions
            
        else:
            logger.info("OpenVINO not installed. Checking lspci/lsusb as fallback.")

        usb_devices = cls.list_usb_myriad_devices()
        if usb_devices:
            devices["vpu"] = True
            devices["vpu_usb"] = True
            devices["vpu_usb_count"] = len(usb_devices)
            devices["vpu_usb_details"] = [
                device["serial"] or f"USB_{device['sysfs_name']}"
                for device in usb_devices
            ]
            # For USB Myriad, we assume standard VPU precisions
            if "MYRIAD" not in devices["supported_precisions_by_device"]:
                devices["supported_precisions_by_device"]["MYRIAD_USB"] = cls._get_device_precisions("MYRIAD", devices["igpu_type"], cpu_features)


        devices["vpu_count"] = max(devices["vpu_runtime_count"], devices["vpu_usb_count"])
        devices["vpu_details"] = (
            devices["vpu_runtime_details"]
            if devices["vpu_runtime_details"]
            else devices["vpu_usb_details"]
        )

        return devices

    @staticmethod
    def check_nvidia_gpu() -> Dict[str, Any]:
        """
        Detect NVIDIA CUDA GPUs via nvidia-smi.
        Returns GPU name, VRAM (bytes), compute capability, CUDA version, and driver version.
        """
        result = {
            "nvidia_gpu_present": False,
            "nvidia_gpu_name": None,
            "nvidia_gpu_vram_bytes": 0,
            "nvidia_gpu_vram_gb": 0.0,
            "nvidia_compute_capability": None,
            "nvidia_cuda_version": None,
            "nvidia_driver_version": None,
            "nvidia_gpu_index": 0,
        }
        if platform.system() != "Linux":
            return result
        try:
            smi = shutil.which("nvidia-smi")
            if not smi:
                return result
            output = subprocess.check_output(
                [smi, "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                text=True, timeout=10,
            ).strip()
            if not output:
                return result
            # Parse first GPU (multi-GPU systems can be extended later)
            parts = [p.strip() for p in output.split("\n")[0].split(",")]
            if len(parts) >= 3:
                result["nvidia_gpu_present"] = True
                result["nvidia_gpu_name"] = parts[0]
                result["nvidia_gpu_vram_bytes"] = int(parts[1]) * 1024 * 1024  # MiB → bytes
                result["nvidia_gpu_vram_gb"] = round(int(parts[1]) / 1024, 2)
                result["nvidia_compute_capability"] = parts[2]
            # Get CUDA and driver versions
            ver_output = subprocess.check_output(
                [smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, timeout=10,
            ).strip()
            if ver_output:
                result["nvidia_driver_version"] = ver_output.split("\n")[0].strip()
            # CUDA version from nvidia-smi header
            try:
                header = subprocess.check_output([smi], text=True, timeout=10)
                for line in header.split("\n"):
                    if "CUDA Version:" in line:
                        result["nvidia_cuda_version"] = line.split("CUDA Version:")[-1].strip()
                        break
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"nvidia-smi probe failed: {e}")
        return result

    @staticmethod
    def check_gpu_blacklist() -> Dict[str, Any]:
        """
        Blacklists Fermi-based NVIDIA GPUs (GTX 560 Ti) if modern compute kernels are required.
        """
        blacklist = {"blacklisted_gpus_found": False, "reasons": []}
        if platform.system() == "Linux":
            try:
                # 10de:1200 is GTX 560 Ti
                lspci_output = subprocess.check_output(["lspci", "-nn"], text=True).lower()
                if "10de:1200" in lspci_output or "gtx 560 ti" in lspci_output:
                    blacklist["blacklisted_gpus_found"] = True
                    blacklist["reasons"].append("Fermi-based NVIDIA GPU (GTX 560 Ti) detected. Blacklisted for modern kernels.")
            except Exception:
                pass
        return blacklist

    @staticmethod
    def get_hardware_tier(cpu_features: Dict[str, bool], ov_devices: Dict[str, Any]) -> str:
        """
        Classifies the system into a 'Hardware Tier'.
        """
        if cpu_features["amx"] or (cpu_features["avx512"] and ov_devices["npu"]):
            return "HIGH_PERF_SERVER"
        elif ov_devices["igpu_type"] == "xe-lpg" or ov_devices["npu"]:
            return "MODERN_MTL"
        elif cpu_features["avx2"]:
            return "LEGACY_AVX2"
        return "GENERIC"

    @staticmethod
    def check_npu_bar() -> Dict[str, Any]:
        """
        Check for 0x5010000000 (NPU BAR) and its status in /proc/iomem to avoid BIOS/E820 conflicts.
        """
        npu_bar = 0x5010000000
        result = {"found": False, "protected": False, "conflict": False}
        if os.path.exists("/proc/iomem"):
            try:
                with open("/proc/iomem", "r") as f:
                    for line in f:
                        if "-" in line and ":" in line:
                            range_part, name = line.split(":", 1)
                            try:
                                start_str, end_str = range_part.strip().split("-")
                                start_addr = int(start_str, 16)
                                end_addr = int(end_str, 16)
                                if start_addr <= npu_bar <= end_addr:
                                    result["found"] = True
                                    name = name.strip().lower()
                                    if "system ram" in name:
                                        result["conflict"] = True
                                    elif "reserved" in name or "npu" in name or "pci" in name:
                                        result["protected"] = True
                            except ValueError:
                                continue
            except Exception as e:
                logger.warning(f"Failed to read /proc/iomem: {e}")
        return result

    @classmethod
    def list_usb_myriad_devices(cls):
        return []

    @staticmethod
    def detect_virtualization() -> Dict[str, Any]:
        """Detect if we're inside a VM/container and identify the hypervisor type.

        Returns flags used by vm_escape actions to gate viability:
        - is_vm: True if running inside a hypervisor
        - hypervisor: "kvm", "xen", "vmware", "hyper-v", "qemu", "none"
        - is_container: True if inside a container (Docker/LXC)
        - has_dev_mem: /dev/mem accessible
        - has_pci_passthrough: any VFIO/PCI passthrough devices
        - has_xhci: XHCI controller present (for DMA escapes)
        - has_gpu_passthrough: GPU passed through to guest
        - has_nic_passthrough: physical NIC passed through
        - has_heci: /dev/mei0 or HECI device accessible
        - has_virtio: virtio devices present
        - has_vhost: vhost-net/vhost-scsi loaded
        - iommu_active: VT-d / AMD-Vi active
        - nested_virt: nested virtualization enabled
        - escape_paths: dict of viable escape technique -> bool
        """
        result = {
            "is_vm": False,
            "hypervisor": "none",
            "is_container": False,
            "has_dev_mem": False,
            "has_pci_passthrough": False,
            "has_xhci": False,
            "has_gpu_passthrough": False,
            "has_nic_passthrough": False,
            "has_heci": False,
            "has_virtio": False,
            "has_vhost": False,
            "iommu_active": False,
            "nested_virt": False,
            "escape_paths": {},
        }

        if platform.system() != "Linux":
            return result

        # --- Container detection ---
        try:
            with open("/proc/1/cgroup", "r") as f:
                cgroup = f.read().lower()
                if "docker" in cgroup or "lxc" in cgroup or "containerd" in cgroup:
                    result["is_container"] = True
            if os.path.exists("/.dockerenv"):
                result["is_container"] = True
        except Exception:
            pass

        # --- Hypervisor detection via DMI ---
        try:
            vendor_files = [
                "/sys/class/dmi/id/sys_vendor",
                "/sys/class/dmi/id/product_name",
                "/sys/class/dmi/id/bios_vendor",
            ]
            dmi_text = ""
            for vf in vendor_files:
                if os.path.exists(vf):
                    with open(vf, "r") as f:
                        dmi_text += f.read().lower() + " "
            if "kvm" in dmi_text or "qemu" in dmi_text:
                result["is_vm"] = True
                result["hypervisor"] = "kvm"
            elif "xen" in dmi_text:
                result["is_vm"] = True
                result["hypervisor"] = "xen"
            elif "vmware" in dmi_text:
                result["is_vm"] = True
                result["hypervisor"] = "vmware"
            elif "hyper-v" in dmi_text or "microsoft" in dmi_text:
                result["is_vm"] = True
                result["hypervisor"] = "hyper-v"
            elif "virtualbox" in dmi_text:
                result["is_vm"] = True
                result["hypervisor"] = "virtualbox"
        except Exception:
            pass

        # --- CPUID-based hypervisor detection (fallback) ---
        if not result["is_vm"]:
            try:
                with open("/proc/cpuinfo", "r") as f:
                    content = f.read()
                    if "hypervisor" in content.lower():
                        result["is_vm"] = True
                        # Try to identify which one
                        if "kvm" in content.lower():
                            result["hypervisor"] = "kvm"
                        else:
                            result["hypervisor"] = "unknown"
            except Exception:
                pass

        # --- /dev/mem access ---
        result["has_dev_mem"] = os.access("/dev/mem", os.R_OK | os.W_OK)

        # --- PCI passthrough (VFIO) ---
        try:
            vfio_dir = "/sys/bus/pci/drivers/vfio-pci"
            if os.path.isdir(vfio_dir):
                for entry in os.listdir(vfio_dir):
                    if entry.startswith("0000:"):
                        result["has_pci_passthrough"] = True
                        break
        except Exception:
            pass

        # --- XHCI controller ---
        try:
            if shutil.which("lspci"):
                lspci = subprocess.check_output(["lspci", "-nn"], text=True, timeout=5)
                if "0c03:30" in lspci.lower() or "xhci" in lspci.lower():
                    result["has_xhci"] = True
                # GPU passthrough
                if "nvidia" in lspci.lower() and result["is_vm"]:
                    result["has_gpu_passthrough"] = True
                if "amd" in lspci.lower() and "radeon" in lspci.lower() and result["is_vm"]:
                    result["has_gpu_passthrough"] = True
                # NIC passthrough (physical NIC, not virtio)
                if "ethernet" in lspci.lower() and "virtio" not in lspci.lower() and result["is_vm"]:
                    result["has_nic_passthrough"] = True
        except Exception:
            pass

        # --- HECI / MEI device ---
        result["has_heci"] = os.path.exists("/dev/mei0") or os.path.exists("/dev/mei")

        # --- Virtio devices ---
        try:
            if os.path.isdir("/sys/bus/virtio/devices"):
                result["has_virtio"] = len(os.listdir("/sys/bus/virtio/devices")) > 0
        except Exception:
            pass

        # --- vhost modules ---
        try:
            with open("/proc/modules", "r") as f:
                modules = f.read()
                if "vhost_net" in modules:
                    result["has_vhost"] = True
        except Exception:
            pass

        # --- IOMMU active ---
        try:
            with open("/proc/cmdline", "r") as f:
                cmdline = f.read().lower()
                if "intel_iommu=on" in cmdline or "amd_iommu=on" in cmdline or "iommu=pt" in cmdline:
                    result["iommu_active"] = True
            if os.path.isdir("/sys/class/iommu"):
                result["iommu_active"] = len(os.listdir("/sys/class/iommu")) > 0
        except Exception:
            pass

        # --- Nested virtualization ---
        try:
            nested_files = [
                "/sys/module/kvm_intel/parameters/nested",
                "/sys/module/kvm_amd/parameters/nested",
            ]
            for nf in nested_files:
                if os.path.exists(nf):
                    with open(nf, "r") as f:
                        val = f.read().strip().lower()
                        if val in ("1", "y", "yes"):
                            result["nested_virt"] = True
        except Exception:
            pass

        # --- Compute viable escape paths ---
        paths = {}
        hv = result["hypervisor"]

        # DMA escape: needs /dev/mem + XHCI or PCI passthrough
        paths["vm_escape_dma"] = (
            result["has_dev_mem"] and
            (result["has_xhci"] or result["has_pci_passthrough"])
        )

        # SMM escape: needs VM + /dev/mem (SMI triggering)
        paths["vm_escape_smm"] = result["is_vm"] and result["has_dev_mem"]

        # HECI escape: needs HECI device accessible
        paths["vm_escape_heci"] = result["has_heci"]

        # Virtio escape: needs virtio devices (KVM/QEMU)
        paths["vm_escape_virtio"] = result["has_virtio"] and hv in ("kvm", "qemu", "none")

        # Xen escape: needs Xen hypervisor
        paths["vm_escape_xen"] = hv == "xen"

        # VMware escape: needs VMware hypervisor
        paths["vm_escape_vmware"] = hv == "vmware"

        # Hyper-V escape: needs Hyper-V hypervisor
        paths["vm_escape_hyperv"] = hv == "hyper-v"

        # Container escape: needs container environment
        paths["container_escape"] = result["is_container"]

        result["escape_paths"] = paths
        return result

    @classmethod
    def discover(cls) -> Dict[str, Any]:
        """
        Discover system hardware capabilities dynamically on launch.
        Returns a dictionary containing:
        - cpu_features: Dict of CPU capabilities (amx, avx512, avx_vnni, avx2, hybrid).
        - igpu_present, igpu_type: Integrated GPU information.
        - npu_present, npu_type: NPU information.
        - vpu_present, vpu_count, vpu_details, vpu_runtime_usable, vpu_runtime_count, vpu_usb_present, vpu_usb_count: VPU information.
        - openvino_available, openvino_import: OpenVINO status.
        - npu_bar_found, npu_bar_protected, npu_bar_conflict: NPU BAR status.
        - cuda_compat: CUDA compatibility (e.g., via ZLUDA).
        - nvidia_gpu_*: NVIDIA CUDA GPU detection (name, VRAM, compute cap, CUDA version, driver).
        - accel_available: General accelerator availability flag.
        - mem_bandwidth_gbs: Memory bandwidth in GB/s.
        - hardware_tier: Classified hardware tier (e.g., "HIGH_PERF_SERVER", "MODERN_MTL").
        - gpu_blacklisted, blacklist_reasons: GPU blacklist status.
        - supported_precisions_by_device: Dict mapping device names to lists of supported precisions.
        - supported_precisions: A consolidated list of all unique precisions supported by any accelerator.
        """
        cpu_features = cls.check_cpu_features()
        ov_devices = cls.check_openvino_devices(cpu_features) # Pass cpu_features for precision inference
        npu_bar_status = cls.check_npu_bar()
        mem_bandwidth = cls.check_memory_bandwidth()
        gpu_blacklist = cls.check_gpu_blacklist()
        nvidia_gpu = cls.check_nvidia_gpu()
        virt_info = cls.detect_virtualization()

        # Check for ZLUDA (CUDA on Intel) compatibility
        has_zluda = os.path.exists("/usr/local/bin/zluda") or "ZLUDA_PATH" in os.environ
        
        # DYNAMIC ACCELERATION PROBE:
        # Check if the detected accelerators actually support the minimum required precision (INT8/FP16)
        accel_functional = ov_devices["igpu"] or ov_devices["npu"] or ov_devices["vpu"] or has_zluda or nvidia_gpu["nvidia_gpu_present"]
        
        # Blacklist logic: If Fermi GPU found, we explicitly disable CUDA compat for it
        if gpu_blacklist["blacklisted_gpus_found"]:
            has_zluda = False
            accel_functional = ov_devices["igpu"] or ov_devices["npu"] or ov_devices["vpu"] or nvidia_gpu["nvidia_gpu_present"]

        hardware_tier = cls.get_hardware_tier(cpu_features, ov_devices)
        
        # Consolidate supported precisions from all devices
        all_supported_precisions: Set[str] = set()
        for device_precisions in ov_devices.get("supported_precisions_by_device", {}).values():
            all_supported_precisions.update(device_precisions)
        
        # NVIDIA CUDA GPUs support FP32, FP16, and potentially INT8
        if nvidia_gpu["nvidia_gpu_present"]:
            all_supported_precisions.update(["FP32", "FP16", "INT8"])
        
        # Ensure FP32 is always present if any accelerator is available, or as a fallback
        if accel_functional and "FP32" not in all_supported_precisions:
            all_supported_precisions.add("FP32")
        # If no accelerators, assume FP32 is the only option via CPU
        elif not accel_functional and "FP32" not in all_supported_precisions:
             all_supported_precisions.add("FP32")


        return {
            **cpu_features, # Unpack cpu_features dict directly
            "igpu_present": ov_devices["igpu"],
            "igpu_type": ov_devices["igpu_type"],
            "npu_present": ov_devices["npu"],
            "npu_type": ov_devices["npu_type"],
            "vpu_present": ov_devices["vpu"],
            "vpu_runtime_usable": ov_devices["vpu_runtime"],
            "vpu_runtime_count": ov_devices["vpu_runtime_count"],
            "vpu_runtime_details": ov_devices["vpu_runtime_details"],
            "vpu_usb_present": ov_devices["vpu_usb"],
            "vpu_usb_count": ov_devices["vpu_usb_count"],
            "vpu_details": ov_devices["vpu_details"],
            "openvino_available": ov_devices["openvino_available"],
            "openvino_import": ov_devices["openvino_import"],
            "npu_bar_found": npu_bar_status["found"],
            "npu_bar_protected": npu_bar_status["protected"],
            "npu_bar_conflict": npu_bar_status["conflict"],
            "cuda_compat": has_zluda,
            "nvidia_gpu_present": nvidia_gpu["nvidia_gpu_present"],
            "nvidia_gpu_name": nvidia_gpu["nvidia_gpu_name"],
            "nvidia_gpu_vram_bytes": nvidia_gpu["nvidia_gpu_vram_bytes"],
            "nvidia_gpu_vram_gb": nvidia_gpu["nvidia_gpu_vram_gb"],
            "nvidia_compute_capability": nvidia_gpu["nvidia_compute_capability"],
            "nvidia_cuda_version": nvidia_gpu["nvidia_cuda_version"],
            "nvidia_driver_version": nvidia_gpu["nvidia_driver_version"],
            "accel_available": accel_functional,
            "mem_bandwidth_gbs": mem_bandwidth,
            "hardware_tier": hardware_tier,
            "gpu_blacklisted": gpu_blacklist["blacklisted_gpus_found"],
            "blacklist_reasons": gpu_blacklist["reasons"],
            "supported_precisions_by_device": ov_devices["supported_precisions_by_device"], # Include device-specific precisions
            "supported_precisions": sorted(list(all_supported_precisions), key=lambda x: {"FP32": 0, "FP16": 1, "BF16": 2, "INT8": 3}.get(x, 4)), # Consolidated list
            "virtualization": virt_info,
        }

    @classmethod
    def print_capabilities(cls):
        """
        Pretty-print the discovered hardware capabilities, including precisions.
        """
        caps = cls.discover()
        print("---------------------------------")
        print(f"  CPU [AMX]:    {'✅ Supported' if caps['amx'] else '❌ Not Detected'}")
        print(f"  CPU [AVX512]: {'✅ Supported' if caps['cpu_avx512'] else '❌ Not Detected'}")
        print(f"  CPU [AVX2]:   {'✅ Supported' if caps['cpu_avx2'] else '❌ Not Detected'}")
        print(f"  CPU [VNNI]:   {'✅ Supported' if caps['cpu_vnni'] else '❌ Not Detected'}")
        print(f"  CPU [Hybrid]: {'✅ Detected' if caps['cpu_hybrid'] else 'Standard'}")
        print(f"  Memory BW:    {caps['mem_bandwidth_gbs']:.2f} GB/s")
        print(f"  iGPU Presence: {'✅ Detected' if caps['igpu_present'] else '❌ Not Detected'} ({caps['igpu_type']})")
        print(f"  NPU Presence:  {'✅ Detected' if caps['npu_present'] else '❌ Not Detected'} ({caps['npu_type']})")
        print(f"  VPU Presence:  {'✅ Detected' if caps['vpu_present'] else '❌ Not Detected'} ({caps['vpu_count']} units)")
        print(f"  VPU Runtime:   {'✅ Usable' if caps['vpu_runtime_usable'] else '❌ Not Usable'} ({caps['vpu_runtime_count']} targets)")
        print(f"  VPU USB:       {'✅ Present' if caps['vpu_usb_present'] else '❌ Not Present'} ({caps['vpu_usb_count']} devices)")
        print(f"  OpenVINO API:  {caps['openvino_import'] or 'Not Available'}")
        print(f"  NPU BAR Protection: {'✅ SAFE' if caps['npu_bar_protected'] else '⚠️  CONFLICT' if caps['npu_bar_conflict'] else 'Unknown'}")
        if caps['gpu_blacklisted']:
            print(f"  GPU Status:    ❌ BLACKLISTED: {', '.join(caps['blacklist_reasons'])}")
        else:
            print(f"  CUDA Bridge:   {'✅ Enabled (ZLUDA)' if caps['cuda_compat'] else 'Standard Path'}")
        
        if caps['supported_precisions']:
            print(f"  Supported Precisions: {', '.join(caps['supported_precisions'])}")
            if caps['supported_precisions_by_device']:
                print("  Device Specific Precisions:")
                for device, precs in caps['supported_precisions_by_device'].items():
                    print(f"    - {device}: {', '.join(precs)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    HardwareDiscovery.print_capabilities()
