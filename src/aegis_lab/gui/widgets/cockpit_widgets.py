from textual.widget import Widget
from textual.widgets import Static
from math import cos, sin, pi

class RadialGauge(Widget):
    def __init__(self, value: float, label: str):
        super().__init__()
        self.value = value  # 0.0 to 1.0
        self.label = label

    def render(self) -> str:
        # Simple ASCII circular representation
        theta = self.value * 2 * pi - pi / 2
        char = "●" if self.value < 0.9 else "★"
        return f"{self.label}: {int(self.value*100)}%\n[{char:^10}]"

class LedIndicator(Widget):
    def __init__(self, label: str, active: bool):
        super().__init__()
        self.label = label
        self.active = active

    def render(self) -> str:
        color = "green" if self.active else "red"
        return f"[{color}]{self.label}[/{color}]"
