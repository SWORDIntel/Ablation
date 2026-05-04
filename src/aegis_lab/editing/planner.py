import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class EditPlanner:
    """
    Plans edits, specifically focusing on 'steering-only' edits as per Milestone 5.
    Steering edits apply temporary activation perturbations rather than permanent delta tensors.
    """
    def __init__(self, topology_manifest: Dict[str, Any]):
        self.topology_manifest = topology_manifest

    def plan_steering_edit(self, atoms: List[Dict[str, Any]], edit_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a plan for steering-only edits using provided atoms.
        
        Args:
            atoms: A list of immutable atom metadata records.
            edit_config: Configuration defining layer targeting, steering strength, etc.
            
        Returns:
            A dictionary representing the steering plan, detailing which layers
            receive which vectors with what magnitude.
        """
        logger.info("Planning steering-only edit...")
        
        plan = {
            "type": "steering",
            "layers": {},
            "metadata": {
                "config_used": edit_config,
                "atoms_used": [atom.get("atom_id") for atom in atoms if "atom_id" in atom]
            }
        }
        
        target_layers = edit_config.get("target_layers", [])
        if not target_layers:
             # Default to all editable modules if none specified
             target_layers = self.topology_manifest.get("editable_modules", [])
             
        strength = edit_config.get("steering_strength", 1.0)
        
        for layer in target_layers:
            # Map atoms to steering vectors for this layer
            plan["layers"][layer] = {
                "steering_strength": strength,
                "vectors": [atom.get("content_hash") for atom in atoms]
            }
            
        return plan

    def plan_permanent_edit(self, atoms: List[Dict[str, Any]], edit_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a plan for permanent weight modifications.
        """
        logger.info("Planning permanent edit...")
        return {
            "type": "permanent",
            "layers": edit_config.get("target_layers", self.topology_manifest.get("editable_modules", [])),
            "atoms_used": [atom.get("atom_id") for atom in atoms if "atom_id" in atom]
        }
