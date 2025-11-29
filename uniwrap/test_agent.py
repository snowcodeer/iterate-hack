"""Test agent runner for generated environments with evaluation tracking."""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


class RandomAgent:
    """Simple random agent for testing environments."""
    
    def __init__(self, action_space):
        """Initialize agent with action space.
        
        Args:
            action_space: Gymnasium action space
        """
        self.action_space = action_space
    
    def select_action(self, observation):
        """Select a random action.
        
        Args:
            observation: Current observation
            
        Returns:
            action: Random action from action space
        """
        return self.action_space.sample()


class EpisodeStats:
    """Track statistics for a single episode."""
    
    def __init__(self):
        self.rewards = []
        self.actions = []
        self.steps = 0
        self.terminated = False
        self.truncated = False
        self.total_reward = 0.0
        self.start_time = time.time()
        self.end_time = None
    
    def add_step(self, action, reward):
        """Add a step to the episode stats."""
        self.actions.append(action if isinstance(action, (int, float)) else action.tolist())
        self.rewards.append(float(reward))
        self.steps += 1
        self.total_reward += reward
    
    def finish(self, terminated, truncated):
        """Mark episode as finished."""
        self.terminated = terminated
        self.truncated = truncated
        self.end_time = time.time()
    
    def to_dict(self) -> Dict:
        """Convert stats to dictionary."""
        return {
            "steps": self.steps,
            "total_reward": float(self.total_reward),
            "mean_reward": float(np.mean(self.rewards)) if self.rewards else 0.0,
            "max_reward": float(np.max(self.rewards)) if self.rewards else 0.0,
            "min_reward": float(np.min(self.rewards)) if self.rewards else 0.0,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "duration": self.end_time - self.start_time if self.end_time else 0.0,
        }


class EnvironmentEvaluator:
    """Evaluate agents in environments and track results."""
    
    def __init__(self, env_class, env_path: str, num_episodes: int = 10):
        """Initialize evaluator.
        
        Args:
            env_class: Environment class to test
            env_path: Path to environment module (for output directory)
            num_episodes: Number of episodes to run
        """
        self.env_class = env_class
        self.env_path = Path(env_path)
        self.num_episodes = num_episodes
        self.env_name = env_class.__name__
        
        # Create outputs directory
        self.output_dir = self.env_path.parent / "outputs" / self.env_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_evaluation(self, agent: Optional[Any] = None) -> Dict:
        """Run evaluation episodes.
        
        Args:
            agent: Agent to test (defaults to RandomAgent)
            
        Returns:
            Evaluation results dictionary
        """
        print(f"\n{'='*70}")
        print(f"Evaluating: {self.env_name}")
        print(f"{'='*70}")
        
        # Create environment
        env = self.env_class()
        
        # Create agent if not provided
        if agent is None:
            agent = RandomAgent(env.action_space)
        
        # Track all episode stats
        all_episodes = []
        
        # Run episodes
        for episode in range(self.num_episodes):
            print(f"\nEpisode {episode + 1}/{self.num_episodes}...")
            
            episode_stats = EpisodeStats()
            obs, info = env.reset()
            done = False
            
            while not done:
                action = agent.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_stats.add_step(action, reward)
                done = terminated or truncated
            
            episode_stats.finish(terminated, truncated)
            all_episodes.append(episode_stats)
            
            print(f"  Steps: {episode_stats.steps}, "
                  f"Total Reward: {episode_stats.total_reward:.2f}, "
                  f"Terminated: {episode_stats.terminated}")
        
        # Aggregate results
        results = self._aggregate_results(all_episodes)
        
        # Save results
        self._save_results(results, all_episodes)
        
        # Print summary
        self._print_summary(results)
        
        env.close()
        
        return results
    
    def _aggregate_results(self, episodes: List[EpisodeStats]) -> Dict:
        """Aggregate statistics across episodes."""
        total_rewards = [ep.total_reward for ep in episodes]
        steps = [ep.steps for ep in episodes]
        durations = [ep.end_time - ep.start_time for ep in episodes]
        
        return {
            "env_name": self.env_name,
            "num_episodes": self.num_episodes,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_reward": {
                    "mean": float(np.mean(total_rewards)),
                    "std": float(np.std(total_rewards)),
                    "min": float(np.min(total_rewards)),
                    "max": float(np.max(total_rewards)),
                },
                "steps": {
                    "mean": float(np.mean(steps)),
                    "std": float(np.std(steps)),
                    "min": int(np.min(steps)),
                    "max": int(np.max(steps)),
                },
                "duration": {
                    "mean": float(np.mean(durations)),
                    "total": float(np.sum(durations)),
                },
                "termination": {
                    "terminated": sum(1 for ep in episodes if ep.terminated),
                    "truncated": sum(1 for ep in episodes if ep.truncated),
                }
            },
            "episodes": [ep.to_dict() for ep in episodes]
        }
    
    def _save_results(self, results: Dict, episodes: List[EpisodeStats]):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary JSON
        summary_file = self.output_dir / f"eval_{timestamp}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save detailed episode data
        detailed_file = self.output_dir / f"eval_{timestamp}_detailed.json"
        detailed_data = {
            "env_name": self.env_name,
            "timestamp": results["timestamp"],
            "episodes": [ep.to_dict() for ep in episodes]
        }
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        
        # Save simple CSV for quick analysis
        csv_file = self.output_dir / f"eval_{timestamp}_results.csv"
        with open(csv_file, 'w') as f:
            f.write("episode,steps,total_reward,mean_reward,terminated,truncated\n")
            for i, ep in enumerate(episodes):
                stats = ep.to_dict()
                f.write(f"{i},{stats['steps']},{stats['total_reward']:.4f},"
                       f"{stats['mean_reward']:.4f},{stats['terminated']},{stats['truncated']}\n")
        
        print(f"\n✓ Results saved to: {self.output_dir}")
        print(f"  - Summary: {summary_file.name}")
        print(f"  - Detailed: {detailed_file.name}")
        print(f"  - CSV: {csv_file.name}")
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        stats = results["statistics"]
        
        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Environment: {self.env_name}")
        print(f"Episodes: {self.num_episodes}")
        print()
        print("Total Reward:")
        print(f"  Mean: {stats['total_reward']['mean']:.4f} ± {stats['total_reward']['std']:.4f}")
        print(f"  Range: [{stats['total_reward']['min']:.4f}, {stats['total_reward']['max']:.4f}]")
        print()
        print("Steps per Episode:")
        print(f"  Mean: {stats['steps']['mean']:.2f} ± {stats['steps']['std']:.2f}")
        print(f"  Range: [{stats['steps']['min']}, {stats['steps']['max']}]")
        print()
        print("Termination:")
        print(f"  Terminated: {stats['termination']['terminated']}/{self.num_episodes}")
        print(f"  Truncated: {stats['termination']['truncated']}/{self.num_episodes}")
        print(f"{'='*70}")


def test_all_environments(repo_name: str, num_episodes: int = 10, visualize: bool = False):
    """Test all environments for a given repository.
    
    Args:
        repo_name: Name of repository (subdirectory in environments/)
        num_episodes: Number of episodes to run per environment
        visualize: Whether to generate visualizations after testing
    """
    import importlib
    
    # Import the repository module
    try:
        repo_module = importlib.import_module(f"environments.{repo_name}")
    except ImportError as e:
        print(f"Error: Could not import environments.{repo_name}")
        print(f"  {e}")
        return
    
    # Get all environment classes
    env_classes = []
    for name in repo_module.__all__:
        env_class = getattr(repo_module, name)
        env_classes.append(env_class)
    
    print(f"\n{'='*70}")
    print(f"Testing {len(env_classes)} environments from {repo_name}")
    print(f"{'='*70}")
    
    # Test each environment
    results = {}
    for env_class in env_classes:
        # Get the environment file path
        env_module = importlib.import_module(f"environments.{repo_name}.{env_class.__name__}")
        env_path = env_module.__file__
        
        evaluator = EnvironmentEvaluator(env_class, env_path, num_episodes)
        results[env_class.__name__] = evaluator.run_evaluation()
    
    # Print overall summary
    print(f"\n\n{'='*70}")
    print(f"ALL ENVIRONMENTS TESTED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"Repository: {repo_name}")
    print(f"Environments tested: {len(env_classes)}")
    print()
    for env_name, result in results.items():
        print(f"  ✓ {env_name}")
        print(f"    Mean reward: {result['statistics']['total_reward']['mean']:.4f}")
        print(f"    Mean steps: {result['statistics']['steps']['mean']:.2f}")
    print(f"{'='*70}")
    
    # Generate visualizations if requested
    if visualize:
        try:
            from uniwrap.visualize import generate_visualizations
            print("\n")
            generate_visualizations(repo_name)
        except ImportError:
            print("\n⚠ Visualization skipped: matplotlib not installed")
            print("  Install with: pip install matplotlib")

