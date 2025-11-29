#!/usr/bin/env python3
"""
Run and evaluate a generated environment with visual rendering.

Usage:
    python run_eval.py snake_pygame          # Run with random agent, visual mode
    python run_eval.py snake_pygame --headless --episodes 100  # Headless evaluation
    python run_eval.py snake_pygame --human  # Human playable mode
"""

import argparse
import sys
import time
import importlib
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def run_random_agent(env, num_episodes=10, render=True, fps=10):
    """Run random agent and collect metrics."""
    results = {
        'episodes': [],
        'total_reward': 0,
        'total_steps': 0,
        'total_score': 0,
        'max_score': 0,
    }

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0

        while True:
            if render:
                env.render()
                time.sleep(1.0 / fps)  # Control playback speed

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_steps += 1

            if terminated or truncated:
                break

        score = info.get('score', 0)
        results['episodes'].append({
            'episode': episode + 1,
            'steps': episode_steps,
            'reward': episode_reward,
            'score': score,
        })
        results['total_reward'] += episode_reward
        results['total_steps'] += episode_steps
        results['total_score'] += score
        results['max_score'] = max(results['max_score'], score)

        print(f"Episode {episode + 1}: steps={episode_steps}, reward={episode_reward:.1f}, score={score}")

    return results


def run_human_mode(env):
    """Let human play the game using keyboard."""
    import pygame

    print("\n=== Human Play Mode ===")
    print("Controls: Arrow keys or WASD to move")
    print("Press Q or ESC to quit\n")

    obs, info = env.reset()
    total_score = 0
    episode = 1

    running = True
    while running:
        env.render()

        # Process pygame events for keyboard input
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key in (pygame.K_UP, pygame.K_w):
                    action = 0
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    action = 1
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    action = 2
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    action = 3

        if not running:
            break

        # If no key pressed, continue in current direction (no action change)
        if action is None:
            action = env.action_space.sample()  # Or keep last direction

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            score = info.get('score', 0)
            total_score += score
            print(f"Episode {episode} ended! Score: {score}")
            episode += 1
            obs, info = env.reset()

        time.sleep(0.1)  # Slow down for human play

    print(f"\nTotal episodes: {episode - 1}, Total score: {total_score}")
    return total_score


def main():
    parser = argparse.ArgumentParser(description="Run and evaluate generated environments")
    parser.add_argument('env_name', type=str, help='Name of the environment module (e.g., snake_pygame)')
    parser.add_argument('--episodes', '-n', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--headless', action='store_true', help='Run without rendering')
    parser.add_argument('--human', action='store_true', help='Human playable mode')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second for rendering')

    args = parser.parse_args()

    # Import the environment
    try:
        env_module = importlib.import_module(f'environments.{args.env_name}')
        # Get the first exported class
        env_class_name = env_module.__all__[0] if hasattr(env_module, '__all__') else None
        if env_class_name:
            env_class = getattr(env_module, env_class_name)
        else:
            # Try to find a class that ends with 'Env'
            for name in dir(env_module):
                if name.endswith('Env') and not name.startswith('_'):
                    env_class = getattr(env_module, name)
                    break
            else:
                raise ImportError(f"Could not find environment class in {args.env_name}")

        print(f"Loaded environment: {env_class.__name__}")
    except ImportError as e:
        print(f"Error: Could not import environment '{args.env_name}': {e}")
        sys.exit(1)

    # Create environment
    render_mode = None if args.headless else "human"
    env = env_class(render_mode=render_mode)

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print()

    try:
        if args.human:
            run_human_mode(env)
        else:
            print(f"Running {args.episodes} episodes with random agent...")
            print("=" * 50)

            results = run_random_agent(
                env,
                num_episodes=args.episodes,
                render=not args.headless,
                fps=args.fps
            )

            print("=" * 50)
            print(f"\n=== Evaluation Results ===")
            print(f"Episodes: {args.episodes}")
            print(f"Average steps: {results['total_steps'] / args.episodes:.1f}")
            print(f"Average reward: {results['total_reward'] / args.episodes:.1f}")
            print(f"Average score: {results['total_score'] / args.episodes:.2f}")
            print(f"Max score: {results['max_score']}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()


if __name__ == '__main__':
    main()
