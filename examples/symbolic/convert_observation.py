"""Test script demonstrating conversion of GPUDrive observations to symbolic representation."""

import os
from pathlib import Path
import torch
import numpy as np

# Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != 'gpudrive':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)

from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.symbolic.core.scene import SymbolicScene
from gpudrive.symbolic.core.relations import SpatialRelation
from gpudrive.symbolic.core.context import RoadContext

MAX_NUM_OBJECTS = 64
NUM_WORLDS = 1  # We'll focus on a single world for testing
UNIQUE_SCENES = 1
device = "cpu"
# Constants for observation indices
EGO_STATE_DIM = 6  # speed, vehicle_length, vehicle_width, rel_goal_x, rel_goal_y, is_collided
PARTNER_OBS_DIM = 6  # speed, rel_pos_x, rel_pos_y, orientation, vehicle_length, vehicle_width
ROAD_TYPE_CLASSES = 7  # Number of road type classes for one-hot encoding

def extract_ego_state(obs_tensor: torch.Tensor, agent_idx: int = 0) -> dict:
    """Extract ego state from observation tensor.
    
    Args:
        obs_tensor: Tensor of shape [num_worlds, max_agent_count, obs_dim]
        agent_idx: Index of the ego agent (default 0)
    """
    ego_obs = obs_tensor[0, agent_idx, :EGO_STATE_DIM]
    print("\nEgo observation tensor:", ego_obs)
    
    return {
        'id': agent_idx,
        'speed': float(ego_obs[0]),
        'position': np.array([0.0, 0.0]),  # Ego is at origin in relative coords
        'velocity': np.array([float(ego_obs[0]), 0.0]),  # Using speed as velocity magnitude
        'heading': 0.0,  # Ego heading is reference
        'size': np.array([float(ego_obs[1]), float(ego_obs[2])]),  # length, width
        'goal': np.array([float(ego_obs[3]), float(ego_obs[4])]),  # relative goal position
        'is_collided': bool(ego_obs[5])
    }

def extract_partner_info(obs_tensor: torch.Tensor, agent_idx: int) -> dict:
    """Extract partner information for a single agent.
    
    Args:
        obs_tensor: Tensor of shape [num_worlds, max_agent_count, obs_dim]
        agent_idx: Index of the partner agent
    """
    # Partner observations start after ego state
    start_idx = EGO_STATE_DIM
    partner_obs = obs_tensor[0, agent_idx, start_idx:start_idx + PARTNER_OBS_DIM]
    
    # Skip if all zeros (no agent)
    if torch.all(partner_obs == 0):
        return None
        
    print(f"\nPartner {agent_idx} observation tensor:", partner_obs)
    
    return {
        'id': agent_idx,
        'speed': float(partner_obs[0]),
        'position': np.array([float(partner_obs[1]), float(partner_obs[2])]),  # rel_pos_x, rel_pos_y
        'heading': float(partner_obs[3]),  # relative heading
        'size': np.array([float(partner_obs[4]), float(partner_obs[5])]),  # length, width
        'type': 2  # partner type (vehicle)
    }

def extract_road_context(obs_tensor: torch.Tensor, agent_idx: int = 0) -> set:
    """Extract road context from observation tensor.
    
    Args:
        obs_tensor: Tensor of shape [num_worlds, max_agent_count, obs_dim]
        agent_idx: Index of the agent (default 0)
    """
    # Road context starts after ego state and partner observations
    start_idx = EGO_STATE_DIM + MAX_NUM_OBJECTS * PARTNER_OBS_DIM
    road_obs = obs_tensor[0, agent_idx, start_idx:]
    
    print("\nRoad observation tensor:", road_obs[:20])  # Print first 20 values
    
    contexts = set()
    
    # Extract road type from one-hot encoding (if present)
    if len(road_obs) >= ROAD_TYPE_CLASSES:
        road_type = torch.argmax(road_obs[-ROAD_TYPE_CLASSES:])
        print("Road type:", road_type)
        
        # Map road type to context
        if road_type == 1:  # Example mapping, adjust based on actual encoding
            contexts.add(RoadContext.IN_LANE)
        elif road_type == 2:
            contexts.add(RoadContext.IN_INTERSECTION)
    
    return contexts

def print_symbolic_scene(scene: SymbolicScene):
    """Helper function to print the symbolic scene state."""
    print("\nSymbolic Scene State:")
    print("-" * 50)
    
    print("\nEgo State:")
    print(f"  Position: {scene.ego.position}")
    print(f"  Velocity: {scene.ego.velocity}")
    print(f"  Heading: {scene.ego.heading}")
    print(f"  Size: {scene.ego.size}")
    print(f"  Goal: {scene.ego.goal}")
    print(f"  Collided: {scene.ego.is_collided}")
    
    print("\nOther Agents:")
    for agent_id, agent in scene.agents.items():
        print(f"\nAgent {agent_id}:")
        print(f"  Position (relative): {agent.position}")
        print(f"  Velocity: {agent.velocity}")
        print(f"  Heading (relative): {agent.heading}")
        print(f"  Size: {agent.size}")
        print(f"  Type: {agent.agent_type}")
        
        # Print spatial relations with ego
        relations = scene._spatial_relations.get((scene.ego.id, agent_id), set())
        print(f"  Relations with ego: {[r.name for r in relations]}")
    
    print("\nRoad Context:")
    print(f"  {[ctx.name for ctx in scene._road_contexts]}")

def main(): 
    # Initialize environment
    env_config = EnvConfig(
        steer_actions=torch.round(torch.linspace(-1.0, 1.0, 3), decimals=3),
        accel_actions=torch.round(torch.linspace(-3, 3, 3), decimals=3)
    )
    
    # Create dataloader using example data
    data_loader = SceneDataLoader(
        root="data/processed/examples",
        batch_size=NUM_WORLDS,
        dataset_size=UNIQUE_SCENES,
        sample_with_replacement=False,
        seed=42,
        shuffle=True,
    )
    
    # Create environment
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=data_loader,
        max_cont_agents=MAX_NUM_OBJECTS,
        device=device,
    )
    
    # Reset environment and get initial observation
    obs = env.reset()
    
    # Create symbolic scene
    scene = SymbolicScene()
    
    # Extract ego state
    ego_info = extract_ego_state(obs)
    
    # Extract partner info
    partner_infos = []
    for i in range(1, MAX_NUM_OBJECTS):
        partner_info = extract_partner_info(obs, i)
        if partner_info is not None:
            partner_infos.append(partner_info)
    
    # Extract road context
    road_contexts = extract_road_context(obs)
    
    # Convert to symbolic representation
    scene.update_from_gpudrive_obs(ego_info, partner_infos, road_contexts)
    
    # Print initial symbolic scene state
    print("\nInitial state:")
    print_symbolic_scene(scene)
    
    # Take a random action and observe state change
    rand_action = torch.Tensor(
        [[env.action_space.sample() for _ in range(MAX_NUM_OBJECTS * NUM_WORLDS)]]
    ).reshape(NUM_WORLDS, MAX_NUM_OBJECTS)
    
    env.step_dynamics(rand_action)
    obs = env.get_obs()
    
    # Update symbolic scene with new observations
    ego_info = extract_ego_state(obs)
    partner_infos = []
    for i in range(1, MAX_NUM_OBJECTS):
        partner_info = extract_partner_info(obs, i)
        if partner_info is not None:
            partner_infos.append(partner_info)
    road_contexts = extract_road_context(obs)
            
    scene.update_from_gpudrive_obs(ego_info, partner_infos, road_contexts)
    
    print("\nAfter taking random action:")
    print_symbolic_scene(scene)

if __name__ == "__main__":
    main()
