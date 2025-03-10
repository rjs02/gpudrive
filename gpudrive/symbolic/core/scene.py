"""Symbolic representation of a driving scene."""

from dataclasses import dataclass
from typing import Dict, Set, List, Tuple, Union
import numpy as np

from .relations import SpatialRelation
from .context import RoadContext

@dataclass
class AgentState:
    """Efficient representation of agent state."""
    id: int
    position: np.ndarray  # [x, y]
    velocity: np.ndarray  # [vx, vy] 
    heading: float
    size: np.ndarray  # [length, width]
    agent_type: int
    goal: np.ndarray = None  # [x, y] for ego only
    is_collided: bool = False  # for ego only

class SymbolicScene:
    """Efficient symbolic representation of a driving scene."""
    
    def __init__(self):
        self.ego: AgentState = None
        self.agents: Dict[int, AgentState] = {}
        self._spatial_relations: Dict[Tuple[int, int], Set[SpatialRelation]] = {}
        self._road_contexts: Set[RoadContext] = set()
        
    def update_from_gpudrive_obs(self, 
                                self_obs: Union[Dict, object],
                                partner_obs: List[Dict],
                                road_contexts: Set[RoadContext]):
        """Convert GPUDrive observations to symbolic representation.
        
        Args:
            self_obs: Dictionary or object containing ego vehicle state
            partner_obs: List of dictionaries containing partner vehicle states
            road_contexts: Set of RoadContext enums representing the current road context
        """
        # Update ego state
        self.ego = AgentState(
            id=self_obs['id'] if isinstance(self_obs, dict) else self_obs.id,
            position=np.array([0.0, 0.0]),  # Ego is origin in relative coords
            velocity=np.array([self_obs['speed'] if isinstance(self_obs, dict) else self_obs.speed, 0.0]),
            heading=0.0,  # Ego heading is reference in relative coords
            size=self_obs['size'] if isinstance(self_obs, dict) else np.array([self_obs.vehicle_size.length, self_obs.vehicle_size.width]),
            agent_type=1,  # Vehicle type
            goal=self_obs['goal'] if isinstance(self_obs, dict) else None,
            is_collided=self_obs['is_collided'] if isinstance(self_obs, dict) else False
        )
        
        # Update other agents
        self.agents.clear()
        for partner in partner_obs:
            if partner is not None:  # Skip invalid agents
                agent = AgentState(
                    id=partner['id'] if isinstance(partner, dict) else partner.id,
                    position=partner['position'] if isinstance(partner, dict) else np.array([partner.position.x, partner.position.y]),
                    velocity=np.array([partner['speed'] if isinstance(partner, dict) else partner.speed, 0.0]),
                    heading=partner['heading'] if isinstance(partner, dict) else partner.heading,
                    size=partner['size'] if isinstance(partner, dict) else np.array([partner.vehicle_size.length, partner.vehicle_size.width]),
                    agent_type=partner['type'] if isinstance(partner, dict) else int(partner.type)
                )
                self.agents[agent.id] = agent
                
        # Compute spatial relations
        self._compute_spatial_relations()
        
        # Update road context
        self._road_contexts = road_contexts
    
    def _compute_spatial_relations(self):
        """Efficiently compute spatial relations between agents."""
        self._spatial_relations.clear()
        
        # Pre-compute ego vectors
        ego_heading = np.array([np.cos(self.ego.heading), 
                              np.sin(self.ego.heading)])
        ego_right = np.array([-ego_heading[1], ego_heading[0]])
        
        for agent_id, agent in self.agents.items():
            rel_pos = agent.position  # Already in ego frame
            rel_vel = agent.velocity - self.ego.velocity
            
            # Compute basic relations using dot products
            relations = set()
            
            # Forward/backward relations
            forward_proj = np.dot(rel_pos, ego_heading)
            if forward_proj > 0:
                relations.add(SpatialRelation.AHEAD)
            else:
                relations.add(SpatialRelation.BEHIND)
                
            # Left/right relations
            right_proj = np.dot(rel_pos, ego_right)
            if right_proj > 0:
                relations.add(SpatialRelation.RIGHT)
            else:
                relations.add(SpatialRelation.LEFT)
                
            # Approaching/yielding relations
            closing_speed = np.dot(rel_vel, rel_pos) / np.linalg.norm(rel_pos)
            if closing_speed > 0.5:  # Threshold in m/s
                relations.add(SpatialRelation.APPROACHING)
            elif closing_speed < -0.5:
                relations.add(SpatialRelation.YIELDING)
                
            self._spatial_relations[(self.ego.id, agent_id)] = relations
    
    def has_relation(self, agent1_id: int, agent2_id: int, 
                    relation: SpatialRelation) -> bool:
        """Check if a spatial relation exists between two agents."""
        relations = self._spatial_relations.get((agent1_id, agent2_id))
        return relations is not None and relation in relations
    
    def get_nearby_agents(self) -> List[AgentState]:
        """Get list of all agents in observation range."""
        return list(self.agents.values()) 