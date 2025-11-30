"""
RL Agent module - Trains and runs RL agents on generated environments.

Uses Stable Baselines3 for training PPO agents.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    BaseCallback = object  # Placeholder for type checking when SB3 not available


class TrainingCallback(BaseCallback):
    """Callback for streaming training progress with visualization."""

    def __init__(self, progress_callback: Optional[Callable] = None, env_instance=None, verbose=0):
        if SB3_AVAILABLE:
            super().__init__(verbose)
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.env_instance = env_instance  # For getting visual frames
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = 0
        self.current_length = 0
        self.frame_skip = 0  # Send frame every N steps

    def _on_step(self) -> bool:
        # Track rewards
        reward = self.locals.get('rewards', [0])[0]
        self.current_rewards += reward
        self.current_length += 1
        self.frame_skip += 1

        # Send frame update every 10 steps for visualization
        if self.progress_callback and self.frame_skip >= 10:
            self.frame_skip = 0
            frame_data = {
                'type': 'training_frame',
                'step': int(self.current_length),
                'total_timesteps': int(self.num_timesteps),
                'reward': float(reward),
                'episode_reward': float(self.current_rewards),
            }

            # Try to get visual data from environment
            if self.env_instance is not None:
                try:
                    # Get the underlying env from DummyVecEnv
                    env = self.env_instance.envs[0]
                    if hasattr(env, 'snake_body') and hasattr(env, 'food_pos'):
                        # Snake-like game
                        frame_data['snake_body'] = [list(pos) for pos in env.snake_body]
                        frame_data['food_pos'] = list(env.food_pos)
                        frame_data['score'] = getattr(env, 'score', 0)
                    elif hasattr(env, 'render'):
                        # Try to get RGB observation
                        obs = self.locals.get('new_obs', self.locals.get('obs', None))
                        if obs is not None and len(obs.shape) >= 2:
                            frame_data['obs'] = obs[0] if len(obs.shape) > 2 else obs
                except:
                    pass

            self.progress_callback(frame_data)

        # Check for episode end
        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.episode_rewards.append(self.current_rewards)
            self.episode_lengths.append(self.current_length)

            if self.progress_callback:
                self.progress_callback({
                    'type': 'episode_complete',
                    'episode': len(self.episode_rewards),
                    'reward': float(self.current_rewards),
                    'length': int(self.current_length),
                    'avg_reward': float(np.mean(self.episode_rewards[-10:])) if self.episode_rewards else 0.0,
                    'total_timesteps': int(self.num_timesteps)
                })

            self.current_rewards = 0
            self.current_length = 0

        return True


class RLAgent:
    """RL Agent that can train on and play any Gymnasium environment."""

    def __init__(self, env_class, model_path: Optional[Path] = None):
        """
        Initialize the RL agent.

        Args:
            env_class: The environment class to use
            model_path: Optional path to a saved model
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")

        self.env_class = env_class
        self.model_path = model_path
        self.model = None
        self.env = None

    def train(
        self,
        total_timesteps: int = 10000,
        progress_callback: Optional[Callable] = None,
        save_path: Optional[Path] = None,
        continue_training: bool = False
    ) -> Dict[str, Any]:
        """
        Train the agent on the environment.

        Args:
            total_timesteps: Number of timesteps to train for
            progress_callback: Optional callback for progress updates
            save_path: Optional path to save the trained model
            continue_training: If True and model already loaded, continue training it

        Returns:
            Dict with training results
        """
        # Create vectorized environment
        self.env = DummyVecEnv([lambda: self.env_class(render_mode=None)])

        # Create or continue PPO model
        if continue_training and self.model is not None:
            # Continue training existing model - just set the new environment
            self.model.set_env(self.env)
        else:
            # Create new PPO model
            self.model = PPO(
                "MlpPolicy",
                self.env,
                verbose=0,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
            )

        # Setup callback with env instance for visualization
        callback = TrainingCallback(progress_callback=progress_callback, env_instance=self.env)

        # Train
        if progress_callback:
            progress_callback({'type': 'training_start', 'total_timesteps': total_timesteps})

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False
        )

        # Save model if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(save_path))

        results = {
            'total_timesteps': total_timesteps,
            'episodes_completed': len(callback.episode_rewards),
            'avg_reward': float(np.mean(callback.episode_rewards)) if callback.episode_rewards else 0,
            'max_reward': float(np.max(callback.episode_rewards)) if callback.episode_rewards else 0,
            'avg_length': float(np.mean(callback.episode_lengths)) if callback.episode_lengths else 0,
            'final_avg_reward': float(np.mean(callback.episode_rewards[-10:])) if len(callback.episode_rewards) >= 10 else float(np.mean(callback.episode_rewards)) if callback.episode_rewards else 0,
            'episode_rewards': callback.episode_rewards,
            'episode_lengths': callback.episode_lengths,
        }

        if progress_callback:
            progress_callback({'type': 'training_complete', **results})

        return results

    def load(self, model_path: Path):
        """Load a trained model."""
        self.model = PPO.load(str(model_path))
        self.model_path = model_path

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict an action given an observation.

        Args:
            observation: The current observation
            deterministic: Whether to use deterministic actions

        Returns:
            The predicted action
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)

    def evaluate(
        self,
        num_episodes: int = 10,
        render: bool = False,
        frame_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the trained agent.

        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render the environment
            frame_callback: Optional callback for each frame (for visualization)

        Returns:
            Dict with evaluation results
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        env = self.env_class(render_mode="human" if render else None)

        episode_rewards = []
        episode_lengths = []
        episode_scores = []

        for ep in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0

            if frame_callback:
                frame_callback({
                    'type': 'episode_start',
                    'episode': ep + 1,
                    'snake_body': [list(pos) for pos in env.snake_body] if hasattr(env, 'snake_body') else [],
                    'food_pos': list(env.food_pos) if hasattr(env, 'food_pos') else []
                })

            while True:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1

                if frame_callback:
                    frame_callback({
                        'type': 'frame',
                        'episode': ep + 1,
                        'step': episode_length,
                        'snake_body': [list(pos) for pos in env.snake_body] if hasattr(env, 'snake_body') else [],
                        'food_pos': list(env.food_pos) if hasattr(env, 'food_pos') else [],
                        'score': info.get('score', 0),
                        'reward': float(reward),
                        'total_reward': float(episode_reward),
                        'action': int(action),
                        'terminated': terminated
                    })

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_scores.append(info.get('score', 0))

            if frame_callback:
                frame_callback({
                    'type': 'episode_end',
                    'episode': ep + 1,
                    'reward': float(episode_reward),
                    'length': episode_length,
                    'score': info.get('score', 0)
                })

        env.close()

        return {
            'num_episodes': num_episodes,
            'avg_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'avg_length': float(np.mean(episode_lengths)),
            'avg_score': float(np.mean(episode_scores)),
            'max_score': int(np.max(episode_scores)),
            'episodes': [
                {'episode': i+1, 'reward': float(r), 'length': int(l), 'score': int(s)}
                for i, (r, l, s) in enumerate(zip(episode_rewards, episode_lengths, episode_scores))
            ]
        }

    def close(self):
        """Clean up resources."""
        if self.env is not None:
            self.env.close()
            self.env = None


def generate_training_graphs(
    episode_rewards: list,
    episode_lengths: list,
    env_name: str,
    output_dir: Optional[Path] = None,
    graph_type: str = "training"
) -> Optional[Path]:
    """Generate training/evaluation curves graph.

    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        env_name: Name of the environment
        output_dir: Directory to save graph (default: training_graphs/)
        graph_type: Type of graph - "training" or "evaluation"

    Returns:
        Path to saved graph file, or None if failed
    """
    if not episode_rewards:
        return None

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "training_graphs"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        title_type = "Evaluation" if graph_type == "evaluation" else "Training"
        fig.suptitle(f'{title_type} Results: {env_name}', fontsize=14, fontweight='bold')

        episodes = list(range(1, len(episode_rewards) + 1))

        # 1. Episode Rewards
        ax1 = axes[0, 0]
        ax1.plot(episodes, episode_rewards, 'b-', alpha=0.6, label='Episode Reward')
        # Moving average
        window = min(10, len(episode_rewards))
        if len(episode_rewards) >= window:
            moving_avg = []
            for i in range(len(episode_rewards)):
                start = max(0, i - window + 1)
                moving_avg.append(np.mean(episode_rewards[start:i+1]))
            ax1.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window} ep)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Episode Lengths
        ax2 = axes[0, 1]
        ax2.plot(episodes, episode_lengths, 'g-', alpha=0.6, label='Episode Length')
        if len(episode_lengths) >= window:
            moving_avg_len = []
            for i in range(len(episode_lengths)):
                start = max(0, i - window + 1)
                moving_avg_len.append(np.mean(episode_lengths[start:i+1]))
            ax2.plot(episodes, moving_avg_len, 'orange', linewidth=2, label=f'Moving Avg ({window} ep)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Lengths (Survival)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Reward Distribution
        ax3 = axes[1, 0]
        ax3.hist(episode_rewards, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax3.axvline(np.mean(episode_rewards), color='red', linestyle='--', label=f'Mean: {np.mean(episode_rewards):.2f}')
        ax3.axvline(np.median(episode_rewards), color='orange', linestyle='--', label=f'Median: {np.median(episode_rewards):.2f}')
        ax3.set_xlabel('Reward')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Reward Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Summary Stats
        ax4 = axes[1, 1]
        ax4.axis('off')

        summary_type = "Evaluation" if graph_type == "evaluation" else "Training"
        stats_text = f"""
{summary_type} Summary
================

Episodes Completed: {len(episode_rewards)}

Reward Statistics:
  Mean:    {np.mean(episode_rewards):>10.2f}
  Std:     {np.std(episode_rewards):>10.2f}
  Min:     {np.min(episode_rewards):>10.2f}
  Max:     {np.max(episode_rewards):>10.2f}

  First 10 avg: {np.mean(episode_rewards[:10]) if len(episode_rewards) >= 10 else np.mean(episode_rewards):>7.2f}
  Last 10 avg:  {np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards):>7.2f}

Episode Length Statistics:
  Mean:    {np.mean(episode_lengths):>10.1f} steps
  Max:     {np.max(episode_lengths):>10.0f} steps

Learning Signal:
  Reward improving: {'Yes ✓' if len(episode_rewards) >= 20 and np.mean(episode_rewards[-10:]) > np.mean(episode_rewards[:10]) else 'No ✗'}
  Non-zero rewards: {sum(1 for r in episode_rewards if r != 0)} / {len(episode_rewards)}
"""
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save graph
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_path = output_dir / f"{env_name}_{graph_type}_{timestamp}.png"
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()

        return graph_path

    except Exception as e:
        print(f"Warning: Could not generate training graph: {e}")
        return None


def get_models_dir() -> Path:
    """Get the models directory."""
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def get_model_path(env_name: str) -> Path:
    """Get the default (latest) model save path for an environment."""
    return get_models_dir() / f"{env_name}_ppo.zip"


def get_metadata_path(model_path: Path) -> Path:
    """Get the metadata JSON path for a model."""
    return model_path.with_suffix('.json')


def save_model_metadata(model_path: Path, env_class, env_name: str,
                        total_timesteps: int, continued_from: Optional[str] = None):
    """Save metadata about the model and environment it was trained on."""
    import json
    from datetime import datetime

    # Get environment info
    env_info = {}
    try:
        # Create a temp instance to get observation/action space info
        temp_env = env_class()
        env_info['observation_space_shape'] = list(temp_env.observation_space.shape)
        env_info['action_space_n'] = int(temp_env.action_space.n) if hasattr(temp_env.action_space, 'n') else None
        temp_env.close()
    except Exception as e:
        env_info['error'] = str(e)

    # Try to get environment file hash for version tracking
    env_file_hash = None
    try:
        import hashlib
        env_path = Path(__file__).parent.parent / "environments" / env_name
        for py_file in env_path.glob("*.py"):
            if not py_file.name.startswith("_"):
                content = py_file.read_bytes()
                env_file_hash = hashlib.md5(content).hexdigest()[:12]
                env_info['env_file'] = py_file.name
                break
    except:
        pass

    metadata = {
        'env_name': env_name,
        'env_file_hash': env_file_hash,
        'observation_shape': env_info.get('observation_space_shape'),
        'action_space_n': env_info.get('action_space_n'),
        'total_timesteps': total_timesteps,
        'continued_from': continued_from,
        'trained_at': datetime.now().isoformat(),
        'model_file': model_path.name,
    }

    metadata_path = get_metadata_path(model_path)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def load_model_metadata(model_path: Path) -> Optional[Dict[str, Any]]:
    """Load metadata for a model."""
    import json
    metadata_path = get_metadata_path(model_path)
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def check_model_env_compatibility(model_path: Path, env_class) -> Dict[str, Any]:
    """Check if a model is compatible with the current environment."""
    metadata = load_model_metadata(model_path)
    if not metadata:
        return {'compatible': None, 'reason': 'No metadata found for model'}

    try:
        temp_env = env_class()
        current_obs_shape = list(temp_env.observation_space.shape)
        current_action_n = int(temp_env.action_space.n) if hasattr(temp_env.action_space, 'n') else None
        temp_env.close()
    except Exception as e:
        return {'compatible': None, 'reason': f'Could not create environment: {e}'}

    model_obs_shape = metadata.get('observation_shape')
    model_action_n = metadata.get('action_space_n')

    if model_obs_shape and model_obs_shape != current_obs_shape:
        return {
            'compatible': False,
            'reason': f'Observation shape mismatch: model expects {model_obs_shape}, env has {current_obs_shape}',
            'model_obs_shape': model_obs_shape,
            'current_obs_shape': current_obs_shape
        }

    if model_action_n and model_action_n != current_action_n:
        return {
            'compatible': False,
            'reason': f'Action space mismatch: model expects {model_action_n}, env has {current_action_n}',
        }

    return {'compatible': True, 'metadata': metadata}


def get_versioned_model_path(env_name: str) -> Path:
    """Get a new versioned model path (doesn't overwrite existing)."""
    models_dir = get_models_dir()
    base_path = models_dir / f"{env_name}_ppo.zip"

    # If no existing model, use base path
    if not base_path.exists():
        return base_path

    # Find next version number
    version = 2
    while True:
        versioned_path = models_dir / f"{env_name}_ppo_v{version}.zip"
        if not versioned_path.exists():
            return versioned_path
        version += 1


def get_latest_model_path(env_name: str) -> Optional[Path]:
    """Get the latest (highest version) model path for an environment."""
    models_dir = get_models_dir()
    base_path = models_dir / f"{env_name}_ppo.zip"

    # Find all versions
    versions = []
    if base_path.exists():
        versions.append((1, base_path))

    for f in models_dir.glob(f"{env_name}_ppo_v*.zip"):
        try:
            v = int(f.stem.split("_v")[-1])
            versions.append((v, f))
        except ValueError:
            continue

    if not versions:
        return None

    # Return highest version
    versions.sort(key=lambda x: x[0], reverse=True)
    return versions[0][1]


def list_model_versions(env_name: str) -> list:
    """List all model versions for an environment with metadata."""
    models_dir = get_models_dir()
    base_path = models_dir / f"{env_name}_ppo.zip"

    versions = []
    if base_path.exists():
        import os
        size = os.path.getsize(base_path) / 1024
        metadata = load_model_metadata(base_path)
        versions.append({
            "version": 1,
            "path": base_path,
            "size_kb": size,
            "metadata": metadata
        })

    for f in models_dir.glob(f"{env_name}_ppo_v*.zip"):
        try:
            v = int(f.stem.split("_v")[-1])
            size = os.path.getsize(f) / 1024
            metadata = load_model_metadata(f)
            versions.append({
                "version": v,
                "path": f,
                "size_kb": size,
                "metadata": metadata
            })
        except ValueError:
            continue

    versions.sort(key=lambda x: x["version"])
    return versions


def train_agent(
    env_class,
    env_name: str,
    total_timesteps: int = 10000,
    progress_callback: Optional[Callable] = None,
    continue_training: bool = False
) -> Dict[str, Any]:
    """
    Train an agent on an environment.

    Args:
        env_class: The environment class
        env_name: Name of the environment (for saving)
        total_timesteps: Training timesteps
        progress_callback: Optional progress callback
        continue_training: If True, load existing model and continue training

    Returns:
        Training results dict
    """
    agent = RLAgent(env_class)

    # Determine save path (versioned to avoid overwriting)
    if continue_training:
        # Continue from latest model
        existing_model = get_latest_model_path(env_name)
        if existing_model:
            agent.load(existing_model)
            save_path = get_versioned_model_path(env_name)
        else:
            # No existing model, start fresh
            save_path = get_model_path(env_name)
            continue_training = False  # Can't continue without existing model
    else:
        # New training - save to versioned path
        save_path = get_versioned_model_path(env_name)

    continued_from_path = str(existing_model) if continue_training and existing_model else None

    results = agent.train(
        total_timesteps=total_timesteps,
        progress_callback=progress_callback,
        save_path=save_path,
        continue_training=continue_training
    )

    # Save metadata about the model and environment
    metadata = save_model_metadata(
        model_path=save_path,
        env_class=env_class,
        env_name=env_name,
        total_timesteps=total_timesteps,
        continued_from=continued_from_path
    )

    results['model_path'] = str(save_path)
    results['continued_from'] = continued_from_path
    results['metadata'] = metadata
    agent.close()

    return results


def evaluate_trained_agent(
    env_class,
    env_name: str,
    num_episodes: int = 10,
    frame_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a trained agent.

    Args:
        env_class: The environment class
        env_name: Name of the environment
        num_episodes: Number of evaluation episodes
        frame_callback: Optional frame callback for visualization

    Returns:
        Evaluation results dict
    """
    model_path = get_model_path(env_name)

    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found at {model_path}. Train first!")

    agent = RLAgent(env_class)
    agent.load(model_path)

    results = agent.evaluate(
        num_episodes=num_episodes,
        frame_callback=frame_callback
    )

    agent.close()

    return results
