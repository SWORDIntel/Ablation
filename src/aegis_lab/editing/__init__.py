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

class TopologicalSurgicalTool(Intervention):
    pass

class GraphPruningTool(Intervention):
    pass

# Exported classes for easier access
from aegis_lab.editing.causal_editor import CausalEditor, AttnAblation, GraphPruner
from aegis_lab.editing.sae_clamping import SaeClamping, ConceptErasure, SurgicalFeatureAblator
from aegis_lab.editing.repe_steering import RepeSteering, ContrastiveSteering, DynamicSurgicalSteering
from aegis_lab.editing.topological import GraphIsolator, PathwayExcavator, AttentionHeadSurgeon
