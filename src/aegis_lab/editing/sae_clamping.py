from framewerx.aegis_lab.editing import FeatureIntervention
from typing import Any

class SaeClamping(FeatureIntervention):
    def apply(self, model: Any) -> Any:
        print("Applying SaeClamping.")
        return model

class ConceptErasure(FeatureIntervention):
    def apply(self, model: Any) -> Any:
        print("Applying ConceptErasure (LEACE).")
        return model
