"""Visualization tools for environment evaluation results."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class EvaluationVisualizer:
    """Create visualizations from evaluation results."""
    
    def __init__(self, repo_name: str):
        """Initialize visualizer.
        
        Args:
            repo_name: Repository name (subdirectory in environments/)
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
        
        self.repo_name = repo_name
        self.base_path = Path("environments") / repo_name / "outputs"
        self.viz_path = self.base_path / "visualizations"
        self.viz_path.mkdir(parents=True, exist_ok=True)
    
    def load_results(self, env_name: str) -> Optional[Dict]:
        """Load evaluation results for an environment.
        
        Args:
            env_name: Environment name
            
        Returns:
            Results dictionary or None if not found
        """
        env_path = self.base_path / env_name
        if not env_path.exists():
            return None
        
        # Find most recent summary file
        summary_files = list(env_path.glob("eval_*_summary.json"))
        if not summary_files:
            return None
        
        latest = max(summary_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest, 'r') as f:
            return json.load(f)
    
    def plot_single_environment(self, env_name: str):
        """Create plots for a single environment.
        
        Args:
            env_name: Environment name
        """
        results = self.load_results(env_name)
        if not results:
            print(f"No results found for {env_name}")
            return
        
        episodes = results['episodes']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{env_name} - Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Reward over episodes
        ax = axes[0, 0]
        rewards = [ep['total_reward'] for ep in episodes]
        ax.plot(rewards, marker='o', linewidth=2, markersize=6)
        ax.axhline(y=np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title('Total Reward per Episode', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Steps over episodes
        ax = axes[0, 1]
        steps = [ep['steps'] for ep in episodes]
        ax.plot(steps, marker='s', color='green', linewidth=2, markersize=6)
        ax.axhline(y=np.mean(steps), color='r', linestyle='--', label=f'Mean: {np.mean(steps):.1f}')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Steps', fontsize=12)
        ax.set_title('Steps per Episode', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 3. Reward distribution
        ax = axes[1, 0]
        ax.hist(rewards, bins=min(10, len(rewards)), color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=np.mean(rewards), color='r', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(x=np.median(rewards), color='g', linestyle='--', linewidth=2, label='Median')
        ax.set_xlabel('Total Reward', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Reward Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Statistics summary (text)
        ax = axes[1, 1]
        ax.axis('off')
        
        stats = results['statistics']
        summary_text = f"""
        Environment: {env_name}
        Episodes: {results['num_episodes']}
        
        Total Reward:
          Mean: {stats['total_reward']['mean']:.4f}
          Std:  {stats['total_reward']['std']:.4f}
          Min:  {stats['total_reward']['min']:.4f}
          Max:  {stats['total_reward']['max']:.4f}
        
        Steps:
          Mean: {stats['steps']['mean']:.2f}
          Std:  {stats['steps']['std']:.2f}
          Min:  {stats['steps']['min']}
          Max:  {stats['steps']['max']}
        
        Termination:
          Terminated: {stats['termination']['terminated']}/{results['num_episodes']}
          Truncated:  {stats['termination']['truncated']}/{results['num_episodes']}
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.viz_path / f"{env_name}_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_file}")
    
    def compare_environments(self, env_names: Optional[List[str]] = None):
        """Create comparison plots across environments.
        
        Args:
            env_names: List of environment names to compare (all if None)
        """
        # Load all results
        if env_names is None:
            env_names = [d.name for d in self.base_path.iterdir() 
                        if d.is_dir() and d.name != 'visualizations']
        
        all_results = {}
        for env_name in env_names:
            results = self.load_results(env_name)
            if results:
                all_results[env_name] = results
        
        if not all_results:
            print("No results found to compare")
            return
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.repo_name} - Environment Comparison', fontsize=16, fontweight='bold')
        
        # 1. Mean reward comparison
        ax = axes[0, 0]
        names = list(all_results.keys())
        means = [all_results[name]['statistics']['total_reward']['mean'] for name in names]
        stds = [all_results[name]['statistics']['total_reward']['std'] for name in names]
        
        x = np.arange(len(names))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([n[:20] + '...' if len(n) > 20 else n for n in names], 
                          rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Mean Total Reward', fontsize=12)
        ax.set_title('Mean Reward Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Mean steps comparison
        ax = axes[0, 1]
        means = [all_results[name]['statistics']['steps']['mean'] for name in names]
        stds = [all_results[name]['statistics']['steps']['std'] for name in names]
        
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([n[:20] + '...' if len(n) > 20 else n for n in names], 
                          rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Mean Steps', fontsize=12)
        ax.set_title('Mean Steps Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Reward distribution comparison
        ax = axes[1, 0]
        for i, name in enumerate(names):
            episodes = all_results[name]['episodes']
            rewards = [ep['total_reward'] for ep in episodes]
            ax.violinplot([rewards], positions=[i], widths=0.7, showmeans=True, showmedians=True)
        
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n[:20] + '...' if len(n) > 20 else n for n in names], 
                          rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Total Reward Distribution', fontsize=12)
        ax.set_title('Reward Distributions', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Success rate comparison
        ax = axes[1, 1]
        terminated = [all_results[name]['statistics']['termination']['terminated'] for name in names]
        truncated = [all_results[name]['statistics']['termination']['truncated'] for name in names]
        total = [all_results[name]['num_episodes'] for name in names]
        
        term_pct = [100 * t / tot for t, tot in zip(terminated, total)]
        trunc_pct = [100 * t / tot for t, tot in zip(truncated, total)]
        
        width = 0.35
        ax.bar(x - width/2, term_pct, width, label='Terminated', alpha=0.8, color='forestgreen', edgecolor='black')
        ax.bar(x + width/2, trunc_pct, width, label='Truncated', alpha=0.8, color='coral', edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels([n[:20] + '...' if len(n) > 20 else n for n in names], 
                          rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Termination Type', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.viz_path / f"{self.repo_name}_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_file}")
    
    def generate_dashboard(self):
        """Generate complete dashboard with all environments."""
        print(f"\n{'='*70}")
        print(f"Generating visualizations for {self.repo_name}")
        print(f"{'='*70}\n")
        
        # Find all environments
        env_names = [d.name for d in self.base_path.iterdir() 
                    if d.is_dir() and d.name != 'visualizations']
        
        if not env_names:
            print("No evaluation results found")
            return
        
        # Plot individual environments
        print("Creating individual environment plots...")
        for env_name in env_names:
            self.plot_single_environment(env_name)
        
        # Create comparison plot
        print("\nCreating comparison plot...")
        self.compare_environments(env_names)
        
        print(f"\n{'='*70}")
        print(f"✓ All visualizations saved to:")
        print(f"  {self.viz_path.absolute()}")
        print(f"{'='*70}")
        
        return self.viz_path


def generate_visualizations(repo_name: str):
    """Generate all visualizations for a repository.
    
    Args:
        repo_name: Repository name
    """
    try:
        visualizer = EvaluationVisualizer(repo_name)
        visualizer.generate_dashboard()
    except ImportError as e:
        print(f"Error: {e}")
        print("Install matplotlib with: pip install matplotlib")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

