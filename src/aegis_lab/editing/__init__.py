from abc import ABC, abstractmethod
from typing import Any

class Intervention(ABC):
    @abstractmethod
    def apply(self, model: Any) -> Any:
        pass

class StaticIntervention(Intervention):
    pass

class FeatureIntervention(Intervention):
    pass

class RuntimeSteering(Intervention):
    pass

# Exported classes for easier access
from framewerx.aegis_lab.editing.causal_editor import CausalEditor, AttnAblation
from framewerx.aegis_lab.editing.sae_clamping import SaeClamping, ConceptErasure
from framewerx.aegis_lab.editing.repe_steering import RepeSteering, ContrastiveSteering
