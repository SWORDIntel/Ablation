from aegis_lab.editing import RuntimeSteering, TopologicalSurgicalTool
from typing import Any

class RepeSteering(RuntimeSteering):
    def apply(self, model: Any) -> Any:
        print("Applying RepeSteering.")
        return model

class ContrastiveSteering(RuntimeSteering):
    def apply(self, model: Any) -> Any:
        print("Applying ContrastiveSteering.")
        return model

class DynamicSurgicalSteering(TopologicalSurgicalTool):
    def apply(self, model: Any) -> Any:
        print("Applying DynamicSurgicalSteering.")
        return model
