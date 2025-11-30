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


class TrainingCallback(BaseCallback):
    """Callback for streaming training progress with visualization."""

    def __init__(self, progress_callback: Optional[Callable] = None, env_instance=None, verbose=0):
        super().__init__(verbose)
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
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Train the agent on the environment.

        Args:
            total_timesteps: Number of timesteps to train for
            progress_callback: Optional callback for progress updates
            save_path: Optional path to save the trained model

        Returns:
            Dict with training results
        """
        # Create vectorized environment
        self.env = DummyVecEnv([lambda: self.env_class(render_mode=None)])

        # Create PPO model
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


def get_model_path(env_name: str) -> Path:
    """Get the default model save path for an environment."""
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir / f"{env_name}_ppo.zip"


def train_agent(
    env_class,
    env_name: str,
    total_timesteps: int = 10000,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Convenience function to train an agent.

    Args:
        env_class: The environment class
        env_name: Name of the environment (for saving)
        total_timesteps: Training timesteps
        progress_callback: Optional progress callback

    Returns:
        Training results dict
    """
    agent = RLAgent(env_class)
    save_path = get_model_path(env_name)

    results = agent.train(
        total_timesteps=total_timesteps,
        progress_callback=progress_callback,
        save_path=save_path
    )

    results['model_path'] = str(save_path)
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
