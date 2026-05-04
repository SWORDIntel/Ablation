"""
Distributed Model Sharding (Tile-Aware) Planner
Handles large-scale (Opus-scale) model partitioning across heterogeneous compute tiles (NPU/VPU).
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class DeviceTile:
    """Represents a compute tile on a hardware device (e.g., NPU core, VPU stick)."""
    tile_id: str
    device_type: str  # 'NPU', 'VPU', 'CPU'
    memory_capacity_mb: float
    compute_ops: float

@dataclass
class LayerSpec:
    """Specification for a single model layer."""
    name: str
    memory_mb: float
    compute_ops: float
    dependencies: List[str] = field(default_factory=list)

@dataclass
class ModelShard:
    """A partition of the model assigned to a specific tile."""
    shard_id: str
    layers: List[LayerSpec]
    assigned_tile: Optional[DeviceTile] = None
    
    @property
    def total_memory_mb(self) -> float:
        return sum(layer.memory_mb for layer in self.layers)
        
    @property
    def total_compute_ops(self) -> float:
        return sum(layer.compute_ops for layer in self.layers)

@dataclass
class ExecutionPlan:
    """Distributed execution plan for the model."""
    plan_id: str
    shards: List[ModelShard]
    routing_map: Dict[str, str]  # layer_name -> tile_id

class OpusShardingPlanner:
    """
    Planner for distributing Opus-scale models across multiple hardware tiles.
    Handles memory estimation, layer-wise sharding, and device targeting.
    """
    def __init__(self, available_tiles: List[DeviceTile]):
        self.tiles = sorted(available_tiles, key=lambda t: t.memory_capacity_mb, reverse=True)
        if not self.tiles:
            logger.warning("OpusShardingPlanner initialized with no compute tiles.")
        else:
            logger.info(f"Initialized OpusShardingPlanner with {len(self.tiles)} tiles.")

    def _estimate_memory(self, layers: List[LayerSpec]) -> float:
        """Estimate the memory required for a sequence of layers."""
        return sum(layer.memory_mb for layer in layers)

    def partition_topology(self, topology: List[LayerSpec]) -> List[ModelShard]:
        """
        Partitions the model topology into layer-wise shards that fit into tile memory.
        """
        if not self.tiles:
            raise RuntimeError("Cannot partition topology: No tiles available.")

        shards = []
        current_shard_layers = []
        current_memory = 0.0
        
        # Simple assignment strategy: fill largest tiles first
        tile_idx = 0
        current_tile = self.tiles[tile_idx]
        
        for layer in topology:
            if layer.memory_mb > current_tile.memory_capacity_mb:
                raise ValueError(f"Layer '{layer.name}' requires {layer.memory_mb}MB, "
                                 f"exceeding max tile capacity of {current_tile.memory_capacity_mb}MB.")
                
            if current_memory + layer.memory_mb > current_tile.memory_capacity_mb:
                # Close current shard
                shards.append(ModelShard(
                    shard_id=f"shard_{len(shards)}",
                    layers=current_shard_layers,
                    assigned_tile=current_tile
                ))
                
                # Advance to next tile
                tile_idx = (tile_idx + 1) % len(self.tiles)
                current_tile = self.tiles[tile_idx]
                
                # Start new shard
                current_shard_layers = [layer]
                current_memory = layer.memory_mb
            else:
                # Add to current shard
                current_shard_layers.append(layer)
                current_memory += layer.memory_mb
                
        if current_shard_layers:
            shards.append(ModelShard(
                shard_id=f"shard_{len(shards)}",
                layers=current_shard_layers,
                assigned_tile=current_tile
            ))
            
        return shards

    def generate_execution_plan(self, topology: List[LayerSpec], plan_id: str = "opus_plan") -> ExecutionPlan:
        """
        Produces a complete distributed execution plan for Opus-scale models.
        """
        logger.info(f"Generating execution plan '{plan_id}' for topology with {len(topology)} layers.")
        shards = self.partition_topology(topology)
        
        routing_map = {}
        for shard in shards:
            if shard.assigned_tile:
                for layer in shard.layers:
                    routing_map[layer.name] = shard.assigned_tile.tile_id
                    
        plan = ExecutionPlan(
            plan_id=plan_id,
            shards=shards,
            routing_map=routing_map
        )
        logger.info(f"Execution plan '{plan_id}' generated with {len(shards)} shards.")
        return plan
