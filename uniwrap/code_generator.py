"""Generate executable Python Gym environment code from specifications."""

import json
from typing import Dict, List, Optional
from pathlib import Path


def generate_env_code(spec: Dict, repo_path: str, game_info: Optional[Dict] = None) -> str:
    """Generate Python code for a Gym environment from a specification.

    Args:
        spec: Environment specification dictionary
        repo_path: Path to the original repository (for imports/references)
        game_info: Game type detection info (from GameDetector)

    Returns:
        Python code as string
    """
    # Check if this is a pygame game - use specialized generator
    controller_info = spec.get('controller', {})
    framework = game_info.get('framework', 'unknown') if game_info else 'unknown'

    if framework == 'pygame' or controller_info.get('type') == 'pygame':
        return _generate_pygame_env_code(spec, repo_path, game_info)

    # Default generator for other game types
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
    
    def __init__(self, render_mode: Optional[str] = None, game_url: Optional[str] = None):
        """Initialize the environment.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            game_url: URL or path to game (auto-detected if None)
        """
        super().__init__()
        
        self.render_mode = render_mode
        
        # Define action space
        {action_space_code}
        
        # Define observation space
        {obs_space_code}
        
        # Initialize game controller
        {_generate_game_controller_init(game_info, repo_path)}
        
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
        
        # Reset game to initial state
        if hasattr(self, 'game_controller'):
            self.game_controller.reset_game()
        
        # Reset state to initial values
        {_generate_reset_logic(observations)}
        
        self.step_count = 0
        
        # Get observation from real game state
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
        
        # Apply action to real game
        if hasattr(self, 'game_controller'):
            self.game_controller.apply_action(action)
        
        # Get new observation from real game state
        observation = self._get_observation()
        
        # Get game state for reward calculation
        game_state = {{}}
        if hasattr(self, 'game_controller'):
            try:
                game_state = self.game_controller.get_game_state()
            except:
                pass
        
        # Calculate reward using real game state
        {reward_code}
        
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
        if hasattr(self, 'game_controller'):
            self.game_controller.close_game()
    
    def _get_observation(self) -> Any:
        """Get current observation from real game state.
        
        Returns:
            observation: Current observation
        """
        # Get real game state
        if hasattr(self, 'game_controller'):
            game_state = self.game_controller.get_game_state()
            # Update internal state with real values
            {_generate_state_update_from_game(observations)}
        
        {_generate_observation_logic(observations)}
    
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
    func_desc = reward.get('description', '')
    reward_range = reward.get('range', [0.0, 1.0])
    components = reward.get('components', {})
    
    # Generate reward logic based on description
    reward_lines = []
    
    # Extract any numeric components from the reward spec
    if components:
        reward_lines.append("reward_components = []")
        has_components = False
        
        for key, value in components.items():
            if isinstance(value, (int, float)):
                has_components = True
                if 'weight' in key.lower():
                    # It's a weight - extract the variable name
                    var_name = key.replace('_weight', '')
                    reward_lines.append(f"# {var_name} component")
                    # Map to game state keys
                    game_key = _map_to_game_state_key(var_name)
                    reward_lines.append(f"{var_name}_val = game_state.get('{game_key}', 0.0)")
                    reward_lines.append(f"reward_components.append({value} * {var_name}_val)")
                elif 'penalty' in key.lower():
                    # It's a penalty - check game state
                    condition_var = key.replace('_penalty', '')
                    game_key = _map_to_game_state_key(condition_var)
                    reward_lines.append(f"# {key}")
                    reward_lines.append(f"# Check {condition_var} condition from game state")
                    reward_lines.append(f"if game_state.get('{game_key}', 0) < 0.5:  # Adjust threshold as needed")
                    reward_lines.append(f"    reward_components.append({value})")
                elif 'bonus' in key.lower():
                    # It's a bonus - check game state
                    condition_var = key.replace('_bonus', '')
                    game_key = _map_to_game_state_key(condition_var)
                    reward_lines.append(f"# {key}")
                    reward_lines.append(f"# Check {condition_var} condition from game state")
                    reward_lines.append(f"if game_state.get('{game_key}', 0) > 0.8:  # Adjust threshold as needed")
                    reward_lines.append(f"    reward_components.append({value})")
        
        if has_components:
            reward_lines.append("reward = sum(reward_components) if reward_components else 0.0")
        else:
            reward_lines = ["reward = 0.0  # No valid components found"]
    else:
        # Generate simple reward based on range
        min_r, max_r = reward_range[0], reward_range[1]
        if min_r == 0 and max_r == 1:
            # Standard reward
            reward_lines.append("# Simple progress-based reward")
            reward_lines.append("progress = float(self.step_count) / self.max_steps")
            reward_lines.append("reward = progress")
        else:
            # Scaled reward
            reward_lines.append(f"# Scaled reward in range [{min_r}, {max_r}]")
            reward_lines.append(f"progress = float(self.step_count) / self.max_steps")
            reward_lines.append(f"reward = {min_r} + progress * ({max_r} - {min_r})")
    
    # Add step penalty if mentioned in description
    if 'step' in func_desc.lower() and 'penalty' in func_desc.lower():
        reward_lines.append("reward -= 0.1  # Step penalty")
    
    return "\n        ".join(reward_lines)


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
    return "# Action applied via game_controller.apply_action()"


def _generate_game_controller_init(game_info: Optional[Dict], repo_path: str) -> str:
    """Generate game controller initialization code."""
    if not game_info or game_info['type'] == 'unknown':
        # Default to web controller
        return f"""# Initialize web game controller
        from uniwrap.game_controllers.web_controller import WebGameController
        import os

        # Determine game URL
        if game_url:
            url = game_url
        elif os.path.exists('{repo_path}'):
            # Local file path
            url = f"file://{{os.path.abspath('{repo_path}')}}/index.html"
        else:
            # Try to start dev server or use localhost
            url = "http://localhost:3000"

        self.game_controller = WebGameController(url, framework='react', headless=True)
        self.game_controller.start_game()"""

    game_type = game_info['type']
    framework = game_info.get('framework', 'unknown')

    if game_type == 'web':
        url = game_info.get('config', {}).get('url')
        if not url:
            port = game_info.get('config', {}).get('port', 3000)
            url = f"http://localhost:{port}"

        return f"""# Initialize web game controller
        from uniwrap.game_controllers.web_controller import WebGameController

        # Determine game URL
        if game_url:
            url = game_url
        else:
            url = "{url}"

        self.game_controller = WebGameController(url, framework='{framework}', headless=True)
        self.game_controller.start_game()"""

    elif game_type == 'desktop' and framework == 'pygame':
        entry_point = game_info.get('entry_point', 'main.py')
        game_subtype = game_info.get('config', {}).get('game_type', 'generic')

        # Use specialized controller for known game types
        if game_subtype == 'snake':
            return f"""# Initialize pygame snake controller
        from uniwrap.game_controllers.pygame_controller import PygameSnakeController

        headless = render_mode is None or render_mode != 'human'
        self.game_controller = PygameSnakeController(
            game_path='{repo_path}/{entry_point}',
            headless=headless,
            fps=30 if not headless else 1000  # Fast for training, slow for viewing
        )
        self.game_controller.start_game()"""
        else:
            return f"""# Initialize pygame controller
        from uniwrap.game_controllers.pygame_controller import PygameController

        headless = render_mode is None or render_mode != 'human'
        self.game_controller = PygameController(
            game_path='{repo_path}/{entry_point}',
            headless=headless
        )
        self.game_controller.start_game()"""

    else:
        # Fallback to web
        return f"""# Initialize web game controller (fallback)
        from uniwrap.game_controllers.web_controller import WebGameController

        url = game_url or "http://localhost:3000"
        self.game_controller = WebGameController(url, framework='vanilla', headless=True)
        self.game_controller.start_game()"""


def _map_to_game_state_key(var_name: str) -> str:
    """Map variable name to common game state keys."""
    var_lower = var_name.lower()
    if 'slice' in var_lower or 'quality' in var_lower:
        return 'slice_quality'
    elif 'toast' in var_lower or 'level' in var_lower:
        return 'toast_level'
    elif 'butter' in var_lower or 'coverage' in var_lower:
        return 'butter_coverage'
    elif 'phase' in var_lower or 'state' in var_lower:
        return 'phase'
    else:
        # Return normalized key
        return var_name.lower().replace('_', '')


def _generate_state_update_from_game(observations: Dict) -> str:
    """Generate code to update state from game controller."""
    obs_type = observations.get('type', 'box')

    if obs_type == 'dict':
        shape_dict = observations.get('shape', {})
        if shape_dict:
            update_parts = []
            for key in shape_dict.keys():
                # Map common game state keys
                game_key = key.replace('_', '').lower()
                if 'slice' in key.lower() or 'quality' in key.lower():
                    update_parts.append(f"self.state['{key}'] = game_state.get('slice_quality', 0.0)")
                elif 'toast' in key.lower() or 'level' in key.lower():
                    update_parts.append(f"self.state['{key}'] = game_state.get('toast_level', 0.0)")
                elif 'butter' in key.lower() or 'coverage' in key.lower():
                    update_parts.append(f"self.state['{key}'] = game_state.get('butter_coverage', 0.0)")
                else:
                    # Generic mapping
                    update_parts.append(f"self.state['{key}'] = game_state.get('{game_key}', 0.0)")
            return "\n            ".join(update_parts) if update_parts else "# No state updates"

    return "# Update state from game_state if needed"


def _generate_pygame_env_code(spec: Dict, repo_path: str, game_info: Optional[Dict] = None) -> str:
    """Generate specialized environment code for pygame games.

    This generates a clean, working environment that integrates with PygameController.
    """
    env_name = spec.get('name', 'pygame_env').replace('-', '_')
    description = spec.get('description', 'Pygame RL Environment')
    version = spec.get('version', '1.0.0')

    # Get observations info
    observations = spec.get('observations', {})
    features = observations.get('features', [
        'danger_up', 'danger_down', 'danger_left', 'danger_right',
        'dir_up', 'dir_down', 'dir_left', 'dir_right',
        'food_dir_x', 'food_dir_y', 'distance_to_food'
    ])
    obs_shape = observations.get('shape', [len(features)])
    if isinstance(obs_shape, list):
        obs_shape = obs_shape[0] if obs_shape else len(features)

    # Get reward components
    reward = spec.get('reward', {})
    components = reward.get('components', {
        'food_reward': 10.0,
        'death_penalty': -10.0,
        'closer_to_food': 1.0,
        'away_from_food': -1.0,
        'step_penalty': -0.01
    })

    # Get termination info
    termination = spec.get('termination', {})
    max_steps = termination.get('max_steps', 1000)

    # Get controller info
    controller = spec.get('controller', {})
    game_type = controller.get('game_type', 'snake')

    # Build observation extraction code
    obs_extraction = []
    for feature in features:
        obs_extraction.append(f"            state.get('{feature}', 0.0),")
    obs_extraction_str = '\n'.join(obs_extraction)

    # Build reward calculation code
    reward_code_lines = []
    if 'food_reward' in components:
        reward_code_lines.append(f"        if action_result.get('ate_food', False):")
        reward_code_lines.append(f"            reward += {components['food_reward']}")
    if 'death_penalty' in components:
        reward_code_lines.append(f"        if game_state.get('game_over', 0.0) > 0.5:")
        reward_code_lines.append(f"            reward += {components['death_penalty']}")
    if 'closer_to_food' in components and 'away_from_food' in components:
        reward_code_lines.append(f"        # Distance-based reward shaping")
        reward_code_lines.append(f"        current_distance = game_state.get('distance_to_food', 1.0)")
        reward_code_lines.append(f"        if self._prev_distance is not None:")
        reward_code_lines.append(f"            if current_distance < self._prev_distance:")
        reward_code_lines.append(f"                reward += {components['closer_to_food']}")
        reward_code_lines.append(f"            elif current_distance > self._prev_distance:")
        reward_code_lines.append(f"                reward += {components['away_from_food']}")
        reward_code_lines.append(f"        self._prev_distance = current_distance")
    if 'step_penalty' in components:
        reward_code_lines.append(f"        reward += {components['step_penalty']}  # Step penalty")
    reward_code_str = '\n'.join(reward_code_lines) if reward_code_lines else "        reward = 0.0"

    # Select appropriate controller
    if game_type == 'snake':
        controller_class = 'PygameSnakeController'
    else:
        controller_class = 'PygameController'

    code = f'''"""
{description}

Generated from: {repo_path}
Version: {version}
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, Optional


class {env_name}(gym.Env):
    """Gymnasium environment for pygame game.

    This environment wraps a pygame game using the {controller_class}.
    """

    metadata = {{
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }}

    def __init__(self, render_mode: Optional[str] = None):
        """Initialize the environment.

        Args:
            render_mode: 'human' for visual display, 'rgb_array' for pixel output, None for headless
        """
        super().__init__()

        self.render_mode = render_mode

        # Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = spaces.Discrete(4)

        # Observation space: {obs_shape} features normalized to [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=({obs_shape},), dtype=np.float32
        )

        # Initialize game controller
        from uniwrap.game_controllers.pygame_controller import {controller_class}

        headless = render_mode is None or render_mode != 'human'
        self.game_controller = {controller_class}(
            headless=headless,
            fps=30 if not headless else 0  # No fps limit for training
        )
        self.game_controller.start_game()

        # Track previous distance for reward shaping
        self._prev_distance = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.game_controller.reset_game()
        self._prev_distance = None

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Apply action
        action_result = self.game_controller.apply_action(action)

        # Get new state
        game_state = self.game_controller.get_game_state()
        observation = self._get_observation()

        # Calculate reward
        reward = 0.0
{reward_code_str}

        # Check termination
        terminated = game_state.get('game_over', 0.0) > 0.5
        truncated = False

        info = self._get_info()
        info['ate_food'] = action_result.get('ate_food', False)

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Extract observation from game state."""
        state = self.game_controller.get_game_state()

        obs = np.array([
{obs_extraction_str}
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> Dict:
        """Get additional info dictionary."""
        state = self.game_controller.get_game_state()
        return {{
            'score': int(state.get('score', 0) * 50),
            'snake_length': int(state.get('snake_length', 0) * 100),
        }}

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self.game_controller.get_frame()

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'game_controller'):
            self.game_controller.close_game()
'''

    return code

