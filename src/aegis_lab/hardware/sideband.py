import os
import struct
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class SidebandAccess:
    """
    Access Intel IOSF Sideband Fabric via P2SB uncloaking.
    Uses CF8/CFC Port I/O to bypass BIOS locks where possible.
    Reference: Golden Bible Findings (Meteor Lake-P)
    """
    PORT_CF8 = 0xCF8
    PORT_CFC = 0xCFC
    
    # Common P2SB BDFs on Intel systems
    P2SB_BDFS = [(0, 0x1F, 1), (0, 0x1D, 0)]

    def __init__(self):
        self.dev_port = "/dev/port"
        self.dev_mem = "/dev/mem"
        self._sbreg_bar = None
        self._active_bdf = None

    def _outl(self, port: int, val: int):
        with open(self.dev_port, 'wb', buffering=0) as f:
            f.seek(port)
            f.write(struct.pack('<I', val))

    def _inl(self, port: int) -> int:
        with open(self.dev_port, 'rb', buffering=0) as f:
            f.seek(port)
            return struct.unpack('<I', f.read(4))[0]

    def _pci_read32(self, bus: int, dev: int, fn: int, off: int) -> int:
        addr = (1 << 31) | (bus << 16) | (dev << 11) | (fn << 8) | (off & 0xFC)
        try:
            self._outl(self.PORT_CF8, addr)
            return self._inl(self.PORT_CFC)
        except Exception as e:
            logger.debug(f"PCI read failed at {bus:02x}:{dev:02x}.{fn} offset {off:02x}: {e}")
            return 0xFFFFFFFF

    def _pci_write32(self, bus: int, dev: int, fn: int, off: int, val: int):
        addr = (1 << 31) | (bus << 16) | (dev << 11) | (fn << 8) | (off & 0xFC)
        try:
            self._outl(self.PORT_CF8, addr)
            self._outl(self.PORT_CFC, val)
        except Exception as e:
            logger.debug(f"PCI write failed at {bus:02x}:{dev:02x}.{fn} offset {off:02x}: {e}")

    def uncloak_p2sb(self) -> bool:
        """
        Uncloak P2SB bridge to reveal SBREG_BAR.
        """
        if not os.access(self.dev_port, os.R_OK | os.W_OK):
            logger.warning("No R/W access to /dev/port. Cannot uncloak P2SB.")
            return False

        for bdf in self.P2SB_BDFS:
            bus, dev, fn = bdf
            vid_did = self._pci_read32(bus, dev, fn, 0)
            if (vid_did & 0xFFFF) == 0x8086:
                self._active_bdf = bdf
                break
        
        if not self._active_bdf:
            logger.error("Intel P2SB bridge not found.")
            return False

        bus, dev, fn = self._active_bdf
        # P2SB Hide bit is typically bit 24 of register 0xE0
        val = self._pci_read32(bus, dev, fn, 0xE0)
        if val & 0x01000000:
            logger.info(f"P2SB at {bus:02x}:{dev:02x}.{fn} is hidden. Attempting to uncloak...")
            self._pci_write32(bus, dev, fn, 0xE0, val & ~0x01000000)
            # Verify
            val = self._pci_read32(bus, dev, fn, 0xE0)
            if val & 0x01000000:
                logger.error("Failed to uncloak P2SB. It might be hardware-locked by BIOS.")
                return False
            logger.info("P2SB uncloaked successfully.")

        # Read BAR0 (SBREG_BAR)
        self._sbreg_bar = self._pci_read32(bus, dev, fn, 0x10) & ~0xF
        if self._sbreg_bar == 0 or self._sbreg_bar == 0xFFFFFFF0:
            logger.error(f"Invalid SBREG_BAR: 0x{self._sbreg_bar:08X}")
            return False
            
        logger.info(f"SBREG_BAR discovered: 0x{self._sbreg_bar:08X}")
        return True

    def read_sideband(self, port_id: int, offset: int) -> Optional[int]:
        """
        Read from a sideband port using SBREG_BAR.
        Addr = SBREG_BAR + (port_id << 16) + offset
        """
        if not self._sbreg_bar and not self.uncloak_p2sb():
            return None
        
        target_addr = self._sbreg_bar + (port_id << 16) + offset
        
        if not os.access(self.dev_mem, os.R_OK):
            logger.warning("No read access to /dev/mem. Cannot read sideband MMIO.")
            return None
            
        import mmap
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            page_addr = target_addr & ~(page_size - 1)
            page_off = target_addr & (page_size - 1)
            
            with open(self.dev_mem, "rb") as f:
                mm = mmap.mmap(f.fileno(), page_size, mmap.MAP_SHARED, mmap.PROT_READ, offset=page_addr)
                val = struct.unpack('<I', mm[page_off:page_off+4])[0]
                mm.close()
                return val
        except Exception as e:
            logger.error(f"Failed to read sideband port 0x{port_id:02X} offset 0x{offset:04X} at 0x{target_addr:08X}: {e}")
            return None

    def get_dci_status(self) -> Optional[int]:
        """
        Access Port 0xAF (DCI - Direct Connect Interface).
        Returns the DCI status/control register value.
        """
        return self.read_sideband(0xAF, 0x0)

    def get_csme_status(self) -> Optional[int]:
        """
        Access Port 0xB8 (CSME - Converged Security and Manageability Engine).
        Returns the HFSTS1 register value (offset 0x40).
        """
        return self.read_sideband(0xB8, 0x40)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sb = SidebandAccess()
    if sb.uncloak_p2sb():
        dci = sb.get_dci_status()
        csme = sb.get_csme_status()
        print(f"DCI Status (Port 0xAF): 0x{dci:08X}" if dci is not None else "DCI: Failed to read")
        print(f"CSME HFSTS1 (Port 0xB8): 0x{csme:08X}" if csme is not None else "CSME: Failed to read")
    else:
        print("P2SB uncloak failed.")
