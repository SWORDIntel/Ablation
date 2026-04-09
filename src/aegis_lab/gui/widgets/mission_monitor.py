from textual.app import ComposeResult
from textual.containers import Grid, Container, Vertical
from textual.widgets import Static, Label
from aegis_lab.gui.widgets.cockpit_widgets import RadialGauge, LedIndicator

class MissionMonitor(Container):
    def compose(self) -> ComposeResult:
        with Grid(id="cockpit-grid"):
            with Vertical(id="telemetry"):
                yield Label("Mission Telemetry")
                yield RadialGauge(0.0, "Progress")
                yield Static("Duration: 00:00", id="duration")
            with Vertical(id="system-health"):
                yield Label("System Health")
                yield LedIndicator("Thermal", True)
                yield LedIndicator("VPU", True)
            yield Static("Message Log", id="message-log")

    def update_status(self, progress: float, status: str, duration: str, thermal_ok: bool, vpu_ok: bool):
        self.query_one(RadialGauge).value = progress
        self.query_one("#duration", Static).update(f"Duration: {duration}")
        # Note: In a full implementation, these would query and update specific widgets
        self.query_one("#message-log", Static).update(f"Log: {status}")
