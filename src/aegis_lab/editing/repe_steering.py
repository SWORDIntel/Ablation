from aegis_lab.editing import RuntimeSteering
from typing import Any

class RepeSteering(RuntimeSteering):
    def apply(self, model: Any) -> Any:
        print("Applying RepeSteering.")
        return model

class ContrastiveSteering(RuntimeSteering):
    def apply(self, model: Any) -> Any:
        print("Applying ContrastiveSteering.")
        return model
