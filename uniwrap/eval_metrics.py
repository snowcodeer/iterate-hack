"""
Evaluation metrics and graph generation for RL training.

Generates training curves, performance comparisons, and evaluation reports.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving files
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TrainingMetricsCollector:
    """Collects and stores training metrics during training."""

    def __init__(self, env_name: str):
        self.env_name = env_name
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_scores: List[float] = []
        self.timesteps: List[int] = []
        self.training_start_time: Optional[datetime] = None
        self.training_end_time: Optional[datetime] = None

    def on_episode_complete(self, data: Dict):
        """Called when an episode completes during training."""
        self.episode_rewards.append(data.get('reward', 0))
        self.episode_lengths.append(data.get('length', 0))
        self.timesteps.append(data.get('total_timesteps', 0))
        if 'score' in data:
            self.episode_scores.append(data['score'])

    def on_training_start(self):
        """Called when training starts."""
        self.training_start_time = datetime.now()

    def on_training_complete(self):
        """Called when training completes."""
        self.training_end_time = datetime.now()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.episode_rewards:
            return {'error': 'No training data collected'}

        return {
            'env_name': self.env_name,
            'total_episodes': len(self.episode_rewards),
            'total_timesteps': self.timesteps[-1] if self.timesteps else 0,
            'training_duration_seconds': (
                (self.training_end_time - self.training_start_time).total_seconds()
                if self.training_start_time and self.training_end_time else None
            ),
            'rewards': {
                'mean': float(np.mean(self.episode_rewards)),
                'std': float(np.std(self.episode_rewards)),
                'min': float(np.min(self.episode_rewards)),
                'max': float(np.max(self.episode_rewards)),
                'final_10_mean': float(np.mean(self.episode_rewards[-10:])) if len(self.episode_rewards) >= 10 else float(np.mean(self.episode_rewards)),
            },
            'lengths': {
                'mean': float(np.mean(self.episode_lengths)),
                'std': float(np.std(self.episode_lengths)),
                'min': int(np.min(self.episode_lengths)),
                'max': int(np.max(self.episode_lengths)),
            },
            'scores': {
                'mean': float(np.mean(self.episode_scores)) if self.episode_scores else 0,
                'max': float(np.max(self.episode_scores)) if self.episode_scores else 0,
            } if self.episode_scores else None
        }

    def save(self, output_dir: Path):
        """Save metrics to JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data = {
            'env_name': self.env_name,
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_summary(),
            'episode_data': {
                'rewards': self.episode_rewards,
                'lengths': self.episode_lengths,
                'scores': self.episode_scores,
                'timesteps': self.timesteps,
            }
        }

        filepath = output_dir / f"{self.env_name}_training_metrics.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath


def generate_training_graphs(
    metrics: TrainingMetricsCollector,
    output_dir: Path,
    show_plot: bool = False
) -> List[Path]:
    """
    Generate training graphs from collected metrics.

    Args:
        metrics: TrainingMetricsCollector with training data
        output_dir: Directory to save graphs
        show_plot: Whether to display plots interactively

    Returns:
        List of paths to generated graph files
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, cannot generate graphs")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []
    env_name = metrics.env_name

    # Style settings
    plt.style.use('seaborn-v0_8-darkgrid') if 'seaborn-v0_8-darkgrid' in plt.style.available else None

    # 1. Evaluation Graphs
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Evaluation Results: {env_name}', fontsize=14, fontweight='bold')

    # Plot 1: Episode Rewards
    ax1 = axes[0, 0]
    episodes = range(1, len(metrics.episode_rewards) + 1)
    ax1.plot(episodes, metrics.episode_rewards, alpha=0.6, color='blue', marker='o', markersize=4, label='Episode Reward')

    # Add moving average
    window = min(10, len(metrics.episode_rewards) // 5 + 1)
    if len(metrics.episode_rewards) >= window:
        moving_avg = np.convolve(metrics.episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(metrics.episode_rewards) + 1), moving_avg,
                 color='red', linewidth=2, label=f'Moving Avg ({window} ep)')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Game Score per Episode (if available, otherwise reward vs length scatter)
    ax2 = axes[0, 1]
    if metrics.episode_scores and any(s != 0 for s in metrics.episode_scores):
        # Show game scores
        ax2.plot(episodes, metrics.episode_scores, alpha=0.6, color='green', marker='o', markersize=4)
        if len(metrics.episode_scores) >= window:
            score_avg = np.convolve(metrics.episode_scores, np.ones(window)/window, mode='valid')
            ax2.plot(range(window, len(metrics.episode_scores) + 1), score_avg,
                     color='darkgreen', linewidth=2, label=f'Moving Avg')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Score')
        ax2.set_title('Game Score per Episode')
        ax2.legend()
    else:
        # Show reward vs episode length (scatter plot) - useful correlation
        scatter = ax2.scatter(metrics.episode_lengths, metrics.episode_rewards,
                             c=episodes, cmap='viridis', s=60, alpha=0.7, edgecolors='black')
        ax2.set_xlabel('Episode Length (steps)')
        ax2.set_ylabel('Reward')
        ax2.set_title('Reward vs Survival Time')
        # Add colorbar to show episode progression
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Episode #')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Episode Lengths
    ax3 = axes[1, 0]
    ax3.plot(episodes, metrics.episode_lengths, alpha=0.6, color='purple', marker='o', markersize=4)
    if len(metrics.episode_lengths) >= window:
        length_avg = np.convolve(metrics.episode_lengths, np.ones(window)/window, mode='valid')
        ax3.plot(range(window, len(metrics.episode_lengths) + 1), length_avg,
                 color='darkviolet', linewidth=2, label=f'Moving Avg')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.set_title('Episode Length (Survival)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Reward Distribution
    ax4 = axes[1, 1]
    n_bins = min(30, max(5, len(metrics.episode_rewards) // 2))
    ax4.hist(metrics.episode_rewards, bins=n_bins, color='steelblue', edgecolor='black', alpha=0.7)
    ax4.axvline(np.mean(metrics.episode_rewards), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(metrics.episode_rewards):.1f}')
    ax4.axvline(np.median(metrics.episode_rewards), color='orange', linestyle='--',
                linewidth=2, label=f'Median: {np.median(metrics.episode_rewards):.1f}')
    ax4.set_xlabel('Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Reward Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    filepath = output_dir / f"{env_name}_eval_curves.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    generated_files.append(filepath)

    if show_plot:
        plt.show()
    plt.close()

    # 2. Performance Summary Card
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    summary = metrics.get_summary()

    # Create text summary
    text_lines = [
        f"Environment: {env_name}",
        f"{'─' * 40}",
        f"Evaluation Statistics",
        f"{'─' * 40}",
        f"Total Episodes: {summary['total_episodes']}",
        f"Total Timesteps: {summary['total_timesteps']:,}",
        f"",
        f"Reward Statistics",
        f"{'─' * 40}",
        f"Mean Reward: {summary['rewards']['mean']:.2f} (+/- {summary['rewards']['std']:.2f})",
        f"Min Reward: {summary['rewards']['min']:.2f}",
        f"Max Reward: {summary['rewards']['max']:.2f}",
        f"Final 10 Episodes Mean: {summary['rewards']['final_10_mean']:.2f}",
        f"",
        f"Episode Length",
        f"{'─' * 40}",
        f"Mean Length: {summary['lengths']['mean']:.1f} steps",
        f"Min/Max: {summary['lengths']['min']} / {summary['lengths']['max']} steps",
    ]

    if summary.get('scores'):
        text_lines.extend([
            f"",
            f"Game Score",
            f"{'─' * 40}",
            f"Mean Score: {summary['scores']['mean']:.1f}",
            f"Max Score: {summary['scores']['max']:.1f}",
        ])

    text = '\n'.join(text_lines)
    ax.text(0.1, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_title(f'Evaluation Summary: {env_name}', fontsize=14, fontweight='bold', pad=20)

    filepath2 = output_dir / f"{env_name}_summary.png"
    plt.savefig(filepath2, dpi=150, bbox_inches='tight')
    generated_files.append(filepath2)

    if show_plot:
        plt.show()
    plt.close()

    return generated_files


def generate_comparison_graph(
    metrics_list: List[Dict],
    output_dir: Path,
    title: str = "Agent Comparison"
) -> Optional[Path]:
    """
    Generate a comparison graph for multiple trained agents.

    Args:
        metrics_list: List of metrics dicts from different environments/runs
        output_dir: Directory to save graph
        title: Graph title

    Returns:
        Path to generated graph file
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, cannot generate graphs")
        return None

    if not metrics_list:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Bar chart: Mean rewards
    ax1 = axes[0]
    names = [m.get('env_name', f'Agent {i}') for i, m in enumerate(metrics_list)]
    means = [m.get('rewards', {}).get('mean', 0) for m in metrics_list]
    stds = [m.get('rewards', {}).get('std', 0) for m in metrics_list]

    x = np.arange(len(names))
    bars = ax1.bar(x, means, yerr=stds, capsize=5, color='steelblue', edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Mean Reward Comparison')
    ax1.grid(True, alpha=0.3, axis='y')

    # Bar chart: Max rewards
    ax2 = axes[1]
    max_rewards = [m.get('rewards', {}).get('max', 0) for m in metrics_list]
    bars2 = ax2.bar(x, max_rewards, color='forestgreen', edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Max Reward')
    ax2.set_title('Max Reward Comparison')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    filepath = output_dir / "agent_comparison.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def evaluate_and_graph(
    env_name: str,
    env_class,
    num_episodes: int = 20,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run evaluation episodes and generate graphs.

    Args:
        env_name: Name of the environment
        env_class: Environment class
        num_episodes: Number of evaluation episodes
        output_dir: Directory to save results

    Returns:
        Dict with evaluation results and graph paths
    """
    from uniwrap.rl_agent import RLAgent, get_model_path

    output_dir = output_dir or Path("evals") / env_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = get_model_path(env_name)

    if not model_path.exists():
        return {'error': f'No trained model found at {model_path}'}

    # Initialize metrics collector
    metrics = TrainingMetricsCollector(env_name)

    # Run evaluation
    agent = RLAgent(env_class)
    agent.load(model_path)

    env = env_class(render_mode=None)

    print(f"\nEvaluating {env_name} for {num_episodes} episodes...")

    cumulative_steps = 0
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            if terminated or truncated:
                break

        cumulative_steps += episode_length
        score = info.get('score', 0)
        metrics.episode_rewards.append(episode_reward)
        metrics.episode_lengths.append(episode_length)
        metrics.episode_scores.append(score)
        metrics.timesteps.append(cumulative_steps)  # Cumulative steps

        print(f"  Episode {ep+1:3d}: Reward={episode_reward:7.1f}, Length={episode_length:4d}, Score={score}")

    env.close()
    agent.close()

    # Save metrics
    metrics_file = metrics.save(output_dir)

    # Generate graphs
    graph_files = generate_training_graphs(metrics, output_dir)

    summary = metrics.get_summary()

    print(f"\n{'=' * 50}")
    print(f"Evaluation Complete: {env_name}")
    print(f"{'=' * 50}")
    print(f"Episodes:     {summary['total_episodes']}")
    print(f"Mean Reward:  {summary['rewards']['mean']:.2f} (+/- {summary['rewards']['std']:.2f})")
    print(f"Max Reward:   {summary['rewards']['max']:.2f}")
    print(f"Mean Length:  {summary['lengths']['mean']:.1f} steps")
    if summary.get('scores'):
        print(f"Mean Score:   {summary['scores']['mean']:.1f}")
        print(f"Max Score:    {summary['scores']['max']:.1f}")
    print(f"\nGraphs saved to: {output_dir}")
    for gf in graph_files:
        print(f"  - {gf.name}")

    return {
        'summary': summary,
        'metrics_file': str(metrics_file),
        'graph_files': [str(gf) for gf in graph_files],
        'output_dir': str(output_dir)
    }


def load_and_compare_evals(eval_dirs: List[Path], output_dir: Path) -> Optional[Path]:
    """
    Load evaluation results from multiple directories and create comparison.

    Args:
        eval_dirs: List of directories containing evaluation results
        output_dir: Directory to save comparison graph

    Returns:
        Path to comparison graph
    """
    metrics_list = []

    for eval_dir in eval_dirs:
        eval_dir = Path(eval_dir)
        # Find metrics file
        metrics_files = list(eval_dir.glob("*_training_metrics.json"))
        if metrics_files:
            with open(metrics_files[0]) as f:
                data = json.load(f)
                metrics_list.append(data.get('summary', {}))

    if metrics_list:
        return generate_comparison_graph(metrics_list, output_dir)

    return None
