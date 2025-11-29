"""Generate executable Python Gym environment code from specifications."""

import json
from typing import Dict, List


def generate_env_code(spec: Dict, repo_path: str) -> str:
    """Generate Python code for a Gym environment from a specification.
    
    Args:
        spec: Environment specification dictionary
        repo_path: Path to the original repository (for imports/references)
        
    Returns:
        Python code as string
    """
    env_name = spec.get('name', 'GeneratedEnv').replace('-', '_')
    
    # Extract action space definition
    actions = spec.get('actions', {})
    action_type = actions.get('type', 'discrete')
    action_space_code = _generate_action_space(actions)
    
    # Extract observation space definition
    observations = spec.get('observations', {})
    obs_type = observations.get('type', 'box')
    obs_space_code = _generate_observation_space(observations)
    
    # Extract reward function description
    reward = spec.get('reward', {})
    reward_code = _generate_reward_function(reward)
    
    # Extract termination conditions
    termination = spec.get('termination', {})
    termination_code = _generate_termination_logic(termination)
    
    # Generate the full environment class
    code = f'''"""
{spec.get('description', 'Generated RL Environment')}

Generated from: {repo_path}
Version: {spec.get('version', '1.0.0')}
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, Optional


class {env_name}(gym.Env):
    """RL Environment: {spec.get('description', 'Generated environment')}"""
    
    metadata = {{
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }}
    
    def __init__(self, render_mode: Optional[str] = None):
        """Initialize the environment.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        
        self.render_mode = render_mode
        
        # Define action space
        {action_space_code}
        
        # Define observation space
        {obs_space_code}
        
        # Initialize state
        self.state = None
        self.step_count = 0
        self.max_steps = {termination.get('max_steps', 1000)}
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset state to initial values
        {_generate_reset_logic(observations)}
        
        self.step_count = 0
        
        observation = self._get_observation()
        info = {{"step_count": self.step_count}}
        
        return observation, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Run one timestep of the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation: New observation after action
            reward: Reward for this step
            terminated: Whether episode terminated (success/failure)
            truncated: Whether episode was truncated (timeout)
            info: Additional information
        """
        self.step_count += 1
        
        # Apply action to environment
        {_generate_action_application(actions)}
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = {reward_code}
        
        # Check termination conditions
        terminated, truncated = {termination_code}
        
        info = {{
            "step_count": self.step_count,
            "max_steps": self.max_steps,
        }}
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            # Implement human-readable rendering
            print(f"Step: {{self.step_count}}, State: {{self.state}}")
        elif self.render_mode == "rgb_array":
            # Return RGB array for visualization
            # TODO: Implement actual rendering based on codebase
            return np.zeros((64, 64, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up resources."""
        pass
    
    def _get_observation(self) -> Any:
        """Get current observation based on state.
        
        Returns:
            observation: Current observation
        """
        {_generate_observation_logic(observations)}
    
    # TODO: Implement actual game logic integration
    # These methods need to be implemented based on the codebase:
    # - _apply_action_to_game(action): Apply action to actual game code
    # - _get_game_state(): Extract state from game
    # - _check_game_termination(): Check if game has ended
    
    def _check_success(self) -> bool:
        """Check if episode succeeded.
        
        Returns:
            bool: True if success condition met
        """
        # TODO: Implement success condition based on game
        return False
    
    def _check_failure(self) -> bool:
        """Check if episode failed.
        
        Returns:
            bool: True if failure condition met
        """
        # TODO: Implement failure condition based on game
        return False
''' 
    
    return code


def _generate_action_space(actions: Dict) -> str:
    """Generate action space code."""
    action_type = actions.get('type', 'discrete')
    
    if action_type == 'discrete':
        n = actions.get('space', {}).get('n', 4)
        if isinstance(n, dict):
            # If space is complex, extract the number
            n = 4  # Default
        return f"self.action_space = spaces.Discrete({n})"
    elif action_type == 'continuous':
        space_def = actions.get('space', {})
        if isinstance(space_def, dict):
            shape = space_def.get('shape', [1])
            low = space_def.get('low', -1.0)
            high = space_def.get('high', 1.0)
        else:
            shape = [1]
            low = -1.0
            high = 1.0
        return f"self.action_space = spaces.Box(low={low}, high={high}, shape={shape}, dtype=np.float32)"
    elif action_type == 'multi_discrete':
        nvec = actions.get('space', [4])
        # Handle various formats of multi_discrete space
        if isinstance(nvec, dict):
            # Extract numeric values if it's a dict  
            nvec = [v.get('n', 4) if isinstance(v, dict) else 4 for k, v in nvec.items()]
        elif isinstance(nvec, list):
            # Ensure all elements are integers
            nvec = [int(n) if isinstance(n, (int, float)) else 4 for n in nvec]
        else:
            nvec = [4]
        return f"self.action_space = spaces.MultiDiscrete({nvec})"
    else:
        return "self.action_space = spaces.Discrete(4)  # Default"


def _generate_observation_space(observations: Dict) -> str:
    """Generate observation space code."""
    obs_type = observations.get('type', 'box')
    
    if obs_type == 'discrete':
        n = observations.get('shape', [1])[0] if isinstance(observations.get('shape'), list) else 10
        return f"self.observation_space = spaces.Discrete({n})"
    elif obs_type == 'box':
        shape = observations.get('shape', [10])
        if isinstance(shape, dict):
            # Convert dict shape to list
            shape = list(shape.values())
        low = observations.get('low', 0.0)
        high = observations.get('high', 1.0)
        dtype = observations.get('dtype', 'float32')
        return f"self.observation_space = spaces.Box(low={low}, high={high}, shape={shape}, dtype=np.{dtype})"
    elif obs_type == 'dict':
        shape_dict = observations.get('shape', {})
        if shape_dict:
            # Generate dict space code
            dict_items = []
            for k, v in shape_dict.items():
                size = v if isinstance(v, (int, list)) else 1
                if isinstance(size, list):
                    size = tuple(size)
                else:
                    size = (size,)
                dict_items.append(f'"{k}": spaces.Box(low=0.0, high=1.0, shape={size}, dtype=np.float32)')
            dict_str = ', '.join(dict_items)
            return f"self.observation_space = spaces.Dict({{{dict_str}}})"
        else:
            return "self.observation_space = spaces.Dict({})  # Default empty dict"
    else:
        return "self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)  # Default"


def _generate_reward_function(reward: Dict) -> str:
    """Generate reward calculation code."""
    func_desc = reward.get('description', 'reward = 0.0')
    # Extract key components from description
    if 'composite' in func_desc.lower() or 'weighted' in func_desc.lower():
        # Try to extract weights and components
        components = reward.get('components', {})
        if components:
            # Generate composite reward
            parts = []
            for key, value in components.items():
                if 'weight' in key.lower():
                    var_name = key.replace('_weight', '')
                    parts.append(f"{value} * {var_name}")
            if parts:
                return f"({' + '.join(parts)})"
    
    # Default reward
    return "0.0  # TODO: Implement actual reward calculation"


def _generate_termination_logic(termination: Dict) -> str:
    """Generate termination condition code."""
    conditions = termination.get('conditions', ['timeout'])
    max_steps = termination.get('max_steps', 1000)
    
    logic_parts = []
    
    if 'timeout' in conditions or 'max_steps' in str(conditions):
        logic_parts.append(f"self.step_count >= self.max_steps")
    
    if 'success' in conditions:
        logic_parts.append("self._check_success()")
    
    if 'failure' in conditions:
        logic_parts.append("self._check_failure()")
    
    if logic_parts:
        terminated = " or ".join([p for p in logic_parts if 'success' in p or 'failure' in p] or ["False"])
        truncated = " or ".join([p for p in logic_parts if 'timeout' in p or 'max_steps' in p] or ["False"])
    else:
        terminated = "False"
        truncated = f"self.step_count >= self.max_steps"
    
    return f"({terminated}, {truncated})"


def _generate_reset_logic(observations: Dict) -> str:
    """Generate reset logic code."""
    obs_type = observations.get('type', 'box')
    
    if obs_type == 'dict':
        shape_dict = observations.get('shape', {})
        if shape_dict:
            reset_parts = ["self.state = {}"]
            for key in shape_dict.keys():
                size = shape_dict[key]
                if isinstance(size, (int, list)):
                    if isinstance(size, list):
                        shape = tuple(size)
                    else:
                        shape = (size,)
                    reset_parts.append(f"self.state['{key}'] = np.zeros({shape}, dtype=np.float32)")
            return "\n        ".join(reset_parts)
        else:
            return "self.state = {}"
    
    return "self.state = np.zeros((10,), dtype=np.float32)  # Default state"


def _generate_observation_logic(observations: Dict) -> str:
    """Generate observation extraction code."""
    obs_type = observations.get('type', 'box')
    
    if obs_type == 'dict':
        shape_dict = observations.get('shape', {})
        if shape_dict:
            obs_parts = []
            for key in shape_dict.keys():
                obs_parts.append(f"'{key}': self.state.get('{key}', np.zeros({shape_dict[key] if isinstance(shape_dict[key], int) else 1}, dtype=np.float32))")
            obs_str = ',\n            '.join(obs_parts)
            return f"return {{\n            {obs_str}\n        }}"
    
    return "return self.state.copy()"


def _generate_action_application(actions: Dict) -> str:
    """Generate action application code."""
    return "# TODO: Apply action to game state\n        # self._apply_action_to_game(action)"

