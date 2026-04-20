from aegis_lab.editing import StaticIntervention
from typing import Any

class CausalEditor(StaticIntervention):
    def __init__(self, method: str = "ROME"):
        self.method = method
    def apply(self, model: Any) -> Any:
        print(f"Applying {self.method} CausalEditor.")
        return model

class AttnAblation(StaticIntervention):
    def apply(self, model: Any) -> Any:
        print("Applying AttnAblation.")
        return model
