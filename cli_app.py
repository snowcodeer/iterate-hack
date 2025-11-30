#!/usr/bin/env python3
"""
UniWrap CLI - Terminal app for generating and training RL environments.
"""

import os
import sys
import time
import importlib
from pathlib import Path

# Add venv site-packages
venv_site = Path(__file__).parent / "venv" / "lib"
for p in venv_site.glob("python*/site-packages"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Apply nest_asyncio early to allow Playwright sync API with asyncio
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Load env vars
from dotenv import load_dotenv
load_dotenv()


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    print("\033[96m" + "=" * 60 + "\033[0m")
    print("\033[96m  🎮 UniWrap - RL Environment Generator\033[0m")
    print("\033[96m" + "=" * 60 + "\033[0m")
    print()


def print_menu(visual_mode: bool):
    print("\033[93mOptions:\033[0m")
    print("  1. Generate environment from URL")
    print("  2. List environments")
    print("  3. Test environment (random agent)")
    print("  4. Train RL agent")
    print("  5. Test trained agent")
    print("  6. Evaluate & generate graphs")
    print("  7. Analyze & improve environment (AI supervisor)")
    print("  8. Delete environment")
    visual_status = "\033[92mON\033[0m" if visual_mode else "\033[91mOFF\033[0m"
    print(f"  v. Toggle visual mode [{visual_status}]")
    print("  q. Quit")
    print()


def list_environments():
    """List all generated environments."""
    env_dir = Path("environments")
    if not env_dir.exists():
        print("\033[91mNo environments directory found.\033[0m")
        return []

    envs = []
    for d in env_dir.iterdir():
        if d.is_dir() and not d.name.startswith('_'):
            init_file = d / "__init__.py"
            if init_file.exists():
                # Try to get the class name
                class_name = "Unknown"
                try:
                    content = init_file.read_text()
                    if '__all__' in content:
                        import ast
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Assign):
                                for target in node.targets:
                                    if isinstance(target, ast.Name) and target.id == '__all__':
                                        if isinstance(node.value, ast.List) and node.value.elts:
                                            # Handle both ast.Str (old) and ast.Constant (new)
                                            first_el = node.value.elts[0]
                                            if hasattr(first_el, 's'):
                                                class_name = first_el.s
                                            elif hasattr(first_el, 'value'):
                                                class_name = first_el.value
                                            break
                except Exception as e:
                    pass
                envs.append((d.name, class_name))

    return envs


def generate_environment(url: str, hints: str = "", feedback: str = ""):
    """Generate an environment from a URL."""
    print(f"\n\033[94mGenerating environment from: {url}\033[0m")
    if hints:
        print(f"\033[94mHints: {hints}\033[0m")
    if feedback:
        print(f"\033[94mFeedback: {feedback}\033[0m")
    print()

    try:
        import tempfile
        import subprocess
        import re
        from pathlib import Path
        from urllib.parse import urlparse

        from uniwrap.game_detector import GameDetector
        from uniwrap.repo_summarizer import summarize_repo
        from uniwrap.env_designer import generate_env_spec, generate_pygame_env_code, generate_web_game_env_code
        from uniwrap.code_generator import generate_env_code
        from uniwrap.spec_validator import validate_spec

        # Build combined prompt with hints and feedback
        extra_context = ""
        if hints:
            extra_context += f"\n\nUser hints about this game:\n{hints}"
        if feedback:
            extra_context += f"\n\nPrevious attempt feedback - please fix these issues:\n{feedback}"

        is_url = url.startswith('http://') or url.startswith('https://')
        is_github = 'github.com' in url.lower() if is_url else False

        output_base = Path("environments")
        output_base.mkdir(exist_ok=True)

        if is_github:
            # Clone GitHub repo
            print("🔍 Cloning repository...")
            github_repo_name = url.rstrip('/').split('/')[-1].replace('.git', '')
            temp_dir = tempfile.mkdtemp(prefix='uniwrap_')
            try:
                subprocess.run(['git', 'clone', url, temp_dir], check=True, capture_output=True)
                repo_path = temp_dir
                print(f"   Cloned to: {temp_dir}")
            except subprocess.CalledProcessError as e:
                print(f"\033[91m❌ Failed to clone: {e}\033[0m")
                return None

            # Detect game type
            print("🔍 Detecting game type...")
            detector = GameDetector()
            game_info = detector.detect_game_type(repo_path)
            print(f"   Type: {game_info.get('type', 'unknown')}")
            print(f"   Framework: {game_info.get('framework', 'unknown')}")

            # Summarize repo
            print("\n📖 Analyzing codebase...")
            repo_summary = summarize_repo(repo_path)
            print("   Done")

            repo_name = github_repo_name.replace('-', '_').replace('.', '_').lower()

            # Handle pygame games
            if game_info.get('framework') == 'pygame' and game_info.get('entry_point'):
                print("\n💻 Generating pygame environment...")
                game_file_path = Path(repo_path) / game_info['entry_point']
                with open(game_file_path, 'r', encoding='utf-8') as f:
                    game_code = f.read()

                python_code = generate_pygame_env_code(
                    game_code=game_code,
                    repo_summary=repo_summary + extra_context
                )
            else:
                # Generate spec then code for other types
                print("\n🎨 Generating environment spec...")
                spec = generate_env_spec(repo_summary + extra_context, game_info=game_info)
                print(f"   Name: {spec.get('name', 'unknown')}")

                print("\n💻 Generating code...")
                python_code = generate_env_code(spec, repo_path, game_info=game_info)

        elif is_url:
            # Web game URL
            print("🔍 Fetching web game info...")
            try:
                import urllib.request
                with urllib.request.urlopen(url, timeout=30) as response:
                    page_html = response.read().decode('utf-8', errors='ignore')[:50000]
            except Exception as e:
                print(f"   Warning: Could not fetch page: {e}")
                page_html = f"URL: {url}\nCould not fetch page content."

            page_info = f"""
URL: {url}

Page HTML preview (first 5000 chars):
{page_html[:5000]}
"""
            parsed = urlparse(url)
            # Include path in name to distinguish different games on same domain
            domain = parsed.netloc.replace('.', '_').replace('-', '_').lower()
            if domain.startswith('www_'):
                domain = domain[4:]
            # Add path component if present (e.g., /space-waves -> _space_waves)
            path_part = parsed.path.strip('/').replace('/', '_').replace('-', '_').lower()
            if path_part:
                repo_name = f"{domain}_{path_part}"
            else:
                repo_name = domain

            print("\n💻 Generating web game environment...")
            python_code = generate_web_game_env_code(
                game_url=url,
                page_info=page_info,
                extra_context=extra_context
            )

        else:
            # Local path
            repo_path = os.path.abspath(url)
            if not os.path.exists(repo_path):
                print(f"\033[91m❌ Path does not exist: {repo_path}\033[0m")
                return None

            print("🔍 Detecting game type...")
            detector = GameDetector()
            game_info = detector.detect_game_type(repo_path)
            print(f"   Type: {game_info.get('type', 'unknown')}")

            print("\n📖 Analyzing codebase...")
            repo_summary = summarize_repo(repo_path)

            repo_name = Path(repo_path).name.replace('-', '_').replace('.', '_').lower()

            if game_info.get('framework') == 'pygame' and game_info.get('entry_point'):
                print("\n💻 Generating pygame environment...")
                game_file_path = Path(repo_path) / game_info['entry_point']
                with open(game_file_path, 'r', encoding='utf-8') as f:
                    game_code = f.read()

                python_code = generate_pygame_env_code(
                    game_code=game_code,
                    repo_summary=repo_summary + extra_context
                )
            else:
                print("\n🎨 Generating environment spec...")
                spec = generate_env_spec(repo_summary + extra_context, game_info=game_info)

                print("\n💻 Generating code...")
                python_code = generate_env_code(spec, repo_path, game_info=game_info)

        # Extract class name from generated code
        class_match = re.search(r'class\s+(\w+)\s*\(', python_code)
        env_name = class_match.group(1) if class_match else f"{repo_name}Env"

        # Check if environment already exists and handle versioning
        output_path = output_base / repo_name
        if output_path.exists():
            # Find existing versions
            existing_versions = [repo_name]
            for d in output_base.iterdir():
                if d.is_dir() and d.name.startswith(f"{repo_name}_v"):
                    existing_versions.append(d.name)

            print(f"\n\033[93m⚠️  Environment '{repo_name}' already exists!\033[0m")
            print("Existing versions:")
            for v in sorted(existing_versions):
                print(f"  • {v}")

            print("\nOptions:")
            print("  1. Overwrite existing")
            print("  2. Create new version")
            print("  b. Cancel")
            version_choice = input("Select [1]: ").strip().lower() or "1"

            if version_choice == 'b':
                print("\033[91mCancelled.\033[0m")
                return None
            elif version_choice == '2':
                # Find next version number
                version = 2
                while (output_base / f"{repo_name}_v{version}").exists():
                    version += 1
                repo_name = f"{repo_name}_v{version}"
                output_path = output_base / repo_name
                print(f"\n   Creating new version: {repo_name}")

        # Write environment files
        output_path.mkdir(parents=True, exist_ok=True)

        # Create __init__.py
        init_file = output_path / '__init__.py'
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(f'"""Generated RL Environment for {repo_name}."""\n\n')
            f.write(f"from .{env_name.lower()} import {env_name}\n")
            f.write(f'\n__all__ = ["{env_name}"]\n')

        # Write environment code
        python_file = output_path / f"{env_name.lower()}.py"
        with open(python_file, 'w', encoding='utf-8') as f:
            f.write(python_code)

        # Update top-level __init__.py
        top_init = output_base / '__init__.py'
        if not top_init.exists():
            with open(top_init, 'w', encoding='utf-8') as f:
                f.write('"""UniWrap Generated Environments."""\n\n')

        with open(top_init, 'r', encoding='utf-8') as f:
            content = f.read()

        import_line = f"from . import {repo_name}\n"
        if import_line not in content:
            with open(top_init, 'a', encoding='utf-8') as f:
                if not content.endswith('\n'):
                    f.write('\n')
                f.write(import_line)

        print(f"\n\033[92m✅ Environment generated: {env_name}\033[0m")
        print(f"   Path: {output_path}")
        print(f"\n   Usage:")
        print(f"   from environments.{repo_name} import {env_name}")
        print(f"   env = {env_name}()")

        return repo_name

    except Exception as e:
        print(f"\n\033[91m❌ Error: {e}\033[0m")
        import traceback
        traceback.print_exc()
        return None


def test_environment(env_name: str, episodes: int = 5, use_trained: bool = False, visual: bool = True):
    """Test an environment with random or trained agent."""
    print(f"\n\033[94mTesting environment: {env_name}\033[0m")
    print(f"Agent: {'Trained PPO' if use_trained else 'Random'}")
    print(f"Episodes: {episodes}")
    print(f"Visual: {visual}")
    print()

    try:
        # Check if this is a web-based environment BEFORE importing
        env_path = Path(f'environments/{env_name}')
        is_web_env = False
        for py_file in env_path.glob('*.py'):
            if py_file.name.startswith('_'):
                continue
            try:
                content = py_file.read_text()
                if 'playwright' in content.lower() or 'sync_playwright' in content:
                    is_web_env = True
                    break
            except:
                continue

        # Initialize pygame BEFORE importing the environment module
        # This prevents circular import issues with pygame.font
        if not is_web_env:
            try:
                import pygame
                pygame.init()
                # Font module may fail on Python 3.14+ with pygame 2.6.x - that's OK
                try:
                    if hasattr(pygame, 'font') and pygame.font:
                        pygame.font.init()
                except (NotImplementedError, ImportError):
                    pass  # Font not available, continue without it
            except Exception as e:
                print(f"   Note: pygame pre-init: {e}")

        # Clear module cache
        module_name = f'environments.{env_name}'
        for key in list(sys.modules.keys()):
            if key.startswith(f'environments.{env_name}'):
                del sys.modules[key]

        # Import environment
        env_module = importlib.import_module(f'environments.{env_name}')
        env_class_name = env_module.__all__[0] if hasattr(env_module, '__all__') else None
        if not env_class_name:
            for name in dir(env_module):
                if name.endswith('Env') and not name.startswith('_'):
                    env_class_name = name
                    break

        env_class = getattr(env_module, env_class_name)

        # Load trained agent if requested
        agent = None
        if use_trained:
            from uniwrap.rl_agent import RLAgent, get_model_path
            model_path = get_model_path(env_name)
            if not model_path.exists():
                print(f"\033[91m❌ No trained model found at {model_path}\033[0m")
                print("Train the agent first with option 4.")
                return
            agent = RLAgent(env_class)
            agent.load(model_path)
            print(f"Loaded model from: {model_path}")

        # Determine render mode
        if visual:
            render_mode = "human"  # Shows window/browser
            print("\n\033[93m🎮 Opening game window...\033[0m")
            if is_web_env:
                print("   (Browser window will open)")
            else:
                print("   (Pygame window will open)")
        else:
            render_mode = None

        # Run episodes
        env = env_class(render_mode=render_mode)
        all_rewards = []
        all_lengths = []
        all_scores = []

        for ep in range(episodes):
            obs, info = env.reset()
            episode_reward = 0
            steps = 0

            print(f"\n--- Episode {ep + 1} ---")

            while True:
                if agent:
                    action = agent.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1

                # Render if visual mode
                if visual:
                    env.render()
                    # Small delay so we can see what's happening
                    time.sleep(0.03 if is_web_env else 0.05)

                # Show progress every 50 steps
                if steps % 50 == 0:
                    score = info.get('score', info.get('survival_time', 0))
                    print(f"  Step {steps}: reward={reward:.2f}, total={episode_reward:.2f}, score={score}")

                if terminated or truncated:
                    break

            score = info.get('score', info.get('survival_time', 0))
            all_rewards.append(episode_reward)
            all_lengths.append(steps)
            all_scores.append(score if isinstance(score, (int, float)) else 0)

            print(f"  \033[93mDone: {steps} steps, reward={episode_reward:.2f}, score={score}\033[0m")

        env.close()
        if agent:
            agent.close()

        # Summary
        print("\n" + "=" * 40)
        print("\033[92mResults:\033[0m")
        print(f"  Avg Reward: {sum(all_rewards) / len(all_rewards):.2f}")
        print(f"  Max Reward: {max(all_rewards):.2f}")
        print(f"  Avg Score:  {sum(all_scores) / len(all_scores):.2f}")
        print(f"  Max Score:  {max(all_scores)}")

        # Generate evaluation graphs if we have enough data
        if len(all_rewards) >= 3 and use_trained:
            try:
                from uniwrap.rl_agent import generate_training_graphs
                graph_path = generate_training_graphs(
                    episode_rewards=all_rewards,
                    episode_lengths=all_lengths,
                    env_name=env_name,
                    graph_type="evaluation"
                )
                if graph_path:
                    print(f"\n📊 Evaluation graph saved: {graph_path}")
            except Exception as e:
                print(f"   (Graph generation failed: {e})")

    except Exception as e:
        print(f"\n\033[91m❌ Error: {e}\033[0m")
        import traceback
        traceback.print_exc()


def train_agent(env_name: str, timesteps: int = 10000, visual: bool = False, continue_training: bool = False):
    """Train an RL agent on an environment."""
    print(f"\n\033[94mTraining agent on: {env_name}\033[0m")
    print(f"Timesteps: {timesteps:,}")
    print(f"Visual: {visual}")
    print(f"Continue training: {continue_training}")
    print()

    try:
        from uniwrap.rl_agent import train_agent as do_train, get_latest_model_path, get_model_path

        # Check if this is a web-based environment
        env_path = Path(f'environments/{env_name}')
        is_web_env = False
        for py_file in env_path.glob('*.py'):
            if py_file.name.startswith('_'):
                continue
            try:
                content = py_file.read_text()
                if 'playwright' in content.lower() or 'sync_playwright' in content:
                    is_web_env = True
                    break
            except:
                continue

        # Initialize pygame BEFORE importing the environment module
        if not is_web_env:
            try:
                import pygame
                pygame.init()
                # Font module may fail on Python 3.14+ with pygame 2.6.x - that's OK
                try:
                    if hasattr(pygame, 'font') and pygame.font:
                        pygame.font.init()
                except (NotImplementedError, ImportError):
                    pass  # Font not available, continue without it
            except Exception as e:
                print(f"   Note: pygame pre-init: {e}")

        # Clear module cache
        module_name = f'environments.{env_name}'
        for key in list(sys.modules.keys()):
            if key.startswith(f'environments.{env_name}'):
                del sys.modules[key]

        # Import environment
        env_module = importlib.import_module(f'environments.{env_name}')
        env_class_name = env_module.__all__[0] if hasattr(env_module, '__all__') else None
        if not env_class_name:
            for name in dir(env_module):
                if name.endswith('Env') and not name.startswith('_'):
                    env_class_name = name
                    break

        env_class = getattr(env_module, env_class_name)

        # Check if continuing from existing model
        if continue_training:
            existing = get_latest_model_path(env_name)
            if existing:
                print(f"\033[93mContinuing from: {existing}\033[0m")
            else:
                print("\033[91mNo existing model found, starting fresh.\033[0m")
                continue_training = False

        if visual:
            print("\n\033[93m🎮 Opening game window for visual training...\033[0m")
            print("   (Training will be slower but you can watch the agent learn)")
            # For visual training, we'll wrap the env class to use human render mode
            original_class = env_class
            class VisualEnvClass(original_class):
                def __init__(self, *args, **kwargs):
                    kwargs['render_mode'] = 'human'
                    super().__init__(*args, **kwargs)
                def step(self, action):
                    result = super().step(action)
                    self.render()
                    time.sleep(0.02)  # Small delay to see what's happening
                    return result
            env_class = VisualEnvClass

        print("Initializing PPO agent...")

        # Progress callback
        def progress(data):
            if data['type'] == 'episode_complete':
                ep = data['episode']
                r = data['reward']
                avg = data['avg_reward']
                ts = data['total_timesteps']
                pct = (ts / timesteps) * 100
                print(f"  Episode {ep:3d} | Reward: {r:7.1f} | Avg(10): {avg:7.1f} | Progress: {pct:5.1f}%")
            elif data['type'] == 'training_start':
                print("Training started...")
            elif data['type'] == 'training_frame':
                pass  # Silent frame updates

        results = do_train(
            env_class=env_class,
            env_name=env_name,
            total_timesteps=timesteps,
            progress_callback=progress,
            continue_training=continue_training
        )

        print()
        print("\033[92m✅ Training complete!\033[0m")
        print(f"   Episodes:    {results['episodes_completed']}")
        print(f"   Avg Reward:  {results['avg_reward']:.2f}")
        print(f"   Max Reward:  {results['max_reward']:.2f}")
        print(f"   Model saved: {results['model_path']}")
        if results.get('continued_from'):
            print(f"   Continued from: {results['continued_from']}")

        # Generate training graphs
        if results.get('episode_rewards'):
            from uniwrap.rl_agent import generate_training_graphs
            graph_path = generate_training_graphs(
                episode_rewards=results['episode_rewards'],
                episode_lengths=results['episode_lengths'],
                env_name=env_name
            )
            if graph_path:
                print(f"   Training graph: {graph_path}")
                # Try to open the graph
                try:
                    import subprocess
                    subprocess.run(['open', str(graph_path)], check=False)
                except:
                    pass

    except Exception as e:
        print(f"\n\033[91m❌ Error: {e}\033[0m")
        import traceback
        traceback.print_exc()


def evaluate_agent(env_name: str, episodes: int = 20):
    """Evaluate a trained agent and generate graphs."""
    print(f"\n\033[94mEvaluating agent: {env_name}\033[0m")
    print(f"Episodes: {episodes}")
    print()

    try:
        from uniwrap.eval_metrics import evaluate_and_graph

        # Check if this is a web-based environment
        env_path = Path(f'environments/{env_name}')
        is_web_env = False
        for py_file in env_path.glob('*.py'):
            if py_file.name.startswith('_'):
                continue
            try:
                content = py_file.read_text()
                if 'playwright' in content.lower() or 'sync_playwright' in content:
                    is_web_env = True
                    break
            except:
                continue

        # Initialize pygame if needed
        if not is_web_env:
            try:
                import pygame
                pygame.init()
                try:
                    if hasattr(pygame, 'font') and pygame.font:
                        pygame.font.init()
                except (NotImplementedError, ImportError):
                    pass
            except Exception as e:
                pass

        # Clear module cache and import
        for key in list(sys.modules.keys()):
            if key.startswith(f'environments.{env_name}'):
                del sys.modules[key]

        env_module = importlib.import_module(f'environments.{env_name}')
        env_class_name = env_module.__all__[0] if hasattr(env_module, '__all__') else None
        if not env_class_name:
            for name in dir(env_module):
                if name.endswith('Env') and not name.startswith('_'):
                    env_class_name = name
                    break

        env_class = getattr(env_module, env_class_name)

        # Check model-environment compatibility
        from uniwrap.rl_agent import get_latest_model_path, check_model_env_compatibility
        model_path = get_latest_model_path(env_name)
        if model_path:
            compat = check_model_env_compatibility(model_path, env_class)
            if compat.get('compatible') == False:
                print(f"\033[91m⚠️  Model-Environment Mismatch!\033[0m")
                print(f"   {compat['reason']}")
                print(f"\n   The model was trained on a different version of the environment.")
                print(f"   Options: Retrain with current env, or restore old env version.")
                input("\nPress Enter to continue anyway (will likely fail)...")

        # Run evaluation
        output_dir = Path("evals") / env_name
        results = evaluate_and_graph(
            env_name=env_name,
            env_class=env_class,
            num_episodes=episodes,
            output_dir=output_dir
        )

        if 'error' in results:
            print(f"\033[91m❌ Error: {results['error']}\033[0m")
        else:
            print(f"\n\033[92m✅ Evaluation complete!\033[0m")
            print(f"   Graphs saved to: {results['output_dir']}")

            # Try to open the graphs
            try:
                import subprocess
                for gf in results.get('graph_files', []):
                    subprocess.run(['open', gf], check=False)
            except:
                pass

    except Exception as e:
        print(f"\n\033[91m❌ Error: {e}\033[0m")
        import traceback
        traceback.print_exc()


def evaluate_all_agents(trained_envs: list):
    """Evaluate all trained agents and create comparison."""
    print(f"\n\033[94mEvaluating all {len(trained_envs)} trained agents...\033[0m")
    print()

    all_summaries = []
    eval_dirs = []

    for env_name, _ in trained_envs:
        print(f"\n{'='*50}")
        print(f"Evaluating: {env_name}")
        print(f"{'='*50}")

        try:
            from uniwrap.eval_metrics import evaluate_and_graph

            # Check if web env
            env_path = Path(f'environments/{env_name}')
            is_web_env = False
            for py_file in env_path.glob('*.py'):
                if py_file.name.startswith('_'):
                    continue
                try:
                    content = py_file.read_text()
                    if 'playwright' in content.lower():
                        is_web_env = True
                        break
                except:
                    continue

            if not is_web_env:
                try:
                    import pygame
                    pygame.init()
                except:
                    pass

            # Clear and import
            for key in list(sys.modules.keys()):
                if key.startswith(f'environments.{env_name}'):
                    del sys.modules[key]

            env_module = importlib.import_module(f'environments.{env_name}')
            env_class_name = env_module.__all__[0] if hasattr(env_module, '__all__') else None
            if not env_class_name:
                for name in dir(env_module):
                    if name.endswith('Env') and not name.startswith('_'):
                        env_class_name = name
                        break

            env_class = getattr(env_module, env_class_name)

            output_dir = Path("evals") / env_name
            results = evaluate_and_graph(
                env_name=env_name,
                env_class=env_class,
                num_episodes=20,
                output_dir=output_dir
            )

            if 'summary' in results:
                all_summaries.append(results['summary'])
                eval_dirs.append(output_dir)

        except Exception as e:
            print(f"\033[91m❌ Failed to evaluate {env_name}: {e}\033[0m")

    # Generate comparison graph
    if len(all_summaries) > 1:
        print(f"\n{'='*50}")
        print("Generating comparison graph...")
        print(f"{'='*50}")

        try:
            from uniwrap.eval_metrics import generate_comparison_graph
            comparison_dir = Path("evals")
            comparison_path = generate_comparison_graph(all_summaries, comparison_dir, "Trained Agents Comparison")
            if comparison_path:
                print(f"\n\033[92m✅ Comparison graph saved to: {comparison_path}\033[0m")
                try:
                    import subprocess
                    subprocess.run(['open', str(comparison_path)], check=False)
                except:
                    pass
        except Exception as e:
            print(f"\033[91m❌ Failed to generate comparison: {e}\033[0m")

    print(f"\n\033[92m✅ All evaluations complete!\033[0m")
    print(f"   Results saved to: evals/")


def main():
    last_url = ""
    last_env = ""
    visual_mode = True  # Visual mode on by default

    while True:
        clear_screen()
        print_header()

        # Show existing environments
        envs = list_environments()
        if envs:
            print("\033[92mExisting environments:\033[0m")
            for name, cls in envs:
                # Check if trained
                from uniwrap.rl_agent import get_model_path
                model_exists = get_model_path(name).exists()
                trained_marker = " \033[93m[trained]\033[0m" if model_exists else ""
                print(f"  • {name} ({cls}){trained_marker}")
            print()

        print_menu(visual_mode)

        choice = input("\033[96mChoice: \033[0m").strip().lower()

        if choice == 'q':
            print("\nGoodbye! 👋")
            break

        elif choice == '1':
            # Generate environment
            url = input("\nEnter URL (GitHub repo or game URL): ").strip()
            if not url:
                continue
            last_url = url

            hints = input("Hints for the AI (optional, press Enter to skip): ").strip()

            env_name = generate_environment(url, hints=hints)
            if env_name:
                last_env = env_name

            input("\nPress Enter to continue...")

        elif choice == '2':
            # List environments with details
            if not envs:
                print("\n\033[91mNo environments found.\033[0m")
            else:
                print("\n\033[92m" + "=" * 50 + "\033[0m")
                print("\033[92mEnvironment Details\033[0m")
                print("\033[92m" + "=" * 50 + "\033[0m")

                from uniwrap.rl_agent import get_model_path

                for name, cls in envs:
                    print(f"\n\033[96m{name}\033[0m")
                    print(f"  Class: {cls}")

                    # Check for trained model
                    model_path = get_model_path(name)
                    if model_path.exists():
                        import os
                        size = os.path.getsize(model_path) / 1024
                        print(f"  Model: \033[92m✓ Trained\033[0m ({size:.1f} KB)")
                    else:
                        print(f"  Model: \033[91m✗ Not trained\033[0m")

                    # Show env file path
                    env_path = Path(f"environments/{name}")
                    py_files = list(env_path.glob("*.py"))
                    main_file = [f for f in py_files if not f.name.startswith("_")]
                    if main_file:
                        print(f"  File: {main_file[0]}")

                    # Try to detect env type (web vs pygame)
                    for py_file in py_files:
                        if py_file.name.startswith("_"):
                            continue
                        try:
                            content = py_file.read_text()
                            if 'playwright' in content.lower():
                                print(f"  Type: \033[93mWeb Game (Playwright)\033[0m")
                            elif 'pygame' in content.lower():
                                print(f"  Type: \033[93mPygame\033[0m")
                            break
                        except:
                            pass

                print()
            input("\nPress Enter to continue...")

        elif choice == '3':
            # Test with random agent
            if not envs:
                print("\n\033[91mNo environments available. Generate one first.\033[0m")
                input("\nPress Enter to continue...")
                continue

            print("\nAvailable environments:")
            for i, (name, cls) in enumerate(envs, 1):
                print(f"  {i}. {name} ({cls})")
            print("  b. Back to main menu")

            selection = input("\nSelect environment (number or 'b'): ").strip().lower()
            if selection == 'b':
                continue

            try:
                idx = int(selection) - 1
                if 0 <= idx < len(envs):
                    env_name = envs[idx][0]
                    episodes = int(input("Number of episodes [3]: ") or "3")
                    test_environment(env_name, episodes=episodes, use_trained=False, visual=visual_mode)
                    last_env = env_name
            except (ValueError, IndexError):
                print("\033[91mInvalid selection.\033[0m")

            input("\nPress Enter to continue...")

        elif choice == '4':
            # Train RL agent
            if not envs:
                print("\n\033[91mNo environments available. Generate one first.\033[0m")
                input("\nPress Enter to continue...")
                continue

            print("\nAvailable environments:")
            for i, (name, cls) in enumerate(envs, 1):
                print(f"  {i}. {name} ({cls})")
            print("  b. Back to main menu")

            selection = input("\nSelect environment (number or 'b'): ").strip().lower()
            if selection == 'b':
                continue

            try:
                idx = int(selection) - 1
                if 0 <= idx < len(envs):
                    env_name = envs[idx][0]

                    # Check if model already exists
                    from uniwrap.rl_agent import get_latest_model_path, list_model_versions
                    existing_model = get_latest_model_path(env_name)
                    continue_training = False

                    if existing_model:
                        versions = list_model_versions(env_name)
                        print(f"\n\033[93mExisting models found:\033[0m")
                        for v in versions:
                            meta = v.get('metadata')
                            obs_info = ""
                            if meta and meta.get('observation_shape'):
                                obs_info = f" [obs: {meta['observation_shape']}]"
                            print(f"  v{v['version']}: {v['path'].name} ({v['size_kb']:.1f} KB){obs_info}")

                        print("\nTraining mode:")
                        print("  1. Start fresh (new model)")
                        print("  2. Continue training (from latest)")
                        print("  b. Back")
                        mode = input("Select [1]: ").strip().lower() or "1"
                        if mode == 'b':
                            continue
                        elif mode == '2':
                            continue_training = True

                    print("\nTraining steps:")
                    print("  1. 5,000 (quick test)")
                    print("  2. 10,000 (normal)")
                    print("  3. 25,000 (better)")
                    print("  4. 50,000 (good)")
                    print("  5. 100,000 (best)")
                    print("  b. Back")
                    steps_choice = input("Select [2]: ").strip().lower() or "2"
                    if steps_choice == 'b':
                        continue
                    steps_map = {'1': 5000, '2': 10000, '3': 25000, '4': 50000, '5': 100000}
                    timesteps = steps_map.get(steps_choice, 10000)

                    train_agent(env_name, timesteps=timesteps, visual=visual_mode, continue_training=continue_training)
                    last_env = env_name
            except (ValueError, IndexError):
                print("\033[91mInvalid selection.\033[0m")

            input("\nPress Enter to continue...")

        elif choice == '5':
            # Test trained agent
            if not envs:
                print("\n\033[91mNo environments available.\033[0m")
                input("\nPress Enter to continue...")
                continue

            # Filter to only trained envs
            from uniwrap.rl_agent import get_model_path
            trained_envs = [(n, c) for n, c in envs if get_model_path(n).exists()]

            if not trained_envs:
                print("\n\033[91mNo trained agents available. Train one first (option 4).\033[0m")
                input("\nPress Enter to continue...")
                continue

            print("\nTrained environments:")
            for i, (name, cls) in enumerate(trained_envs, 1):
                print(f"  {i}. {name} ({cls})")
            print("  b. Back to main menu")

            selection = input("\nSelect environment (number or 'b'): ").strip().lower()
            if selection == 'b':
                continue

            try:
                idx = int(selection) - 1
                if 0 <= idx < len(trained_envs):
                    env_name = trained_envs[idx][0]
                    episodes = int(input("Number of episodes [3]: ") or "3")
                    test_environment(env_name, episodes=episodes, use_trained=True, visual=visual_mode)
                    last_env = env_name
            except (ValueError, IndexError):
                print("\033[91mInvalid selection.\033[0m")

            input("\nPress Enter to continue...")

        elif choice == '6':
            # Evaluate and generate graphs
            if not envs:
                print("\n\033[91mNo environments available.\033[0m")
                input("\nPress Enter to continue...")
                continue

            # Filter to only trained envs
            from uniwrap.rl_agent import get_model_path
            trained_envs = [(n, c) for n, c in envs if get_model_path(n).exists()]

            if not trained_envs:
                print("\n\033[91mNo trained agents available. Train one first (option 4).\033[0m")
                input("\nPress Enter to continue...")
                continue

            print("\nTrained environments:")
            for i, (name, cls) in enumerate(trained_envs, 1):
                print(f"  {i}. {name} ({cls})")
            print(f"  {len(trained_envs) + 1}. Evaluate ALL trained agents")
            print("  b. Back to main menu")

            selection = input("\nSelect environment (number or 'b'): ").strip().lower()
            if selection == 'b':
                continue

            try:
                idx = int(selection) - 1

                if idx == len(trained_envs):
                    # Evaluate all
                    evaluate_all_agents(trained_envs)
                elif 0 <= idx < len(trained_envs):
                    env_name = trained_envs[idx][0]
                    episodes = int(input("Number of evaluation episodes [20]: ") or "20")
                    evaluate_agent(env_name, episodes=episodes)
                    last_env = env_name
            except (ValueError, IndexError):
                print("\033[91mInvalid selection.\033[0m")

            input("\nPress Enter to continue...")

        elif choice == '7':
            # Analyze & improve environment with AI supervisor
            if not envs:
                print("\n\033[91mNo environments available. Generate one first.\033[0m")
                input("\nPress Enter to continue...")
                continue

            print("\nAvailable environments:")
            for i, (name, cls) in enumerate(envs, 1):
                # Check if we have training data
                training_graph_dir = Path("training_graphs")
                has_training = any(training_graph_dir.glob(f"{name}_training_*.png")) if training_graph_dir.exists() else False
                status = " [has training data]" if has_training else ""
                print(f"  {i}. {name} ({cls}){status}")
            print("  b. Back to main menu")

            selection = input("\nSelect environment to analyze (number or 'b'): ").strip().lower()
            if selection == 'b':
                continue

            try:
                idx = int(selection) - 1
                if 0 <= idx < len(envs):
                    env_name = envs[idx][0]

                    print(f"\n\033[94mAnalyzing environment: {env_name}\033[0m")

                    # Read environment code
                    env_path = Path(f'environments/{env_name}')
                    env_code = None
                    env_file = None
                    for py_file in env_path.glob('*.py'):
                        if not py_file.name.startswith('_'):
                            env_code = py_file.read_text()
                            env_file = py_file
                            break

                    if not env_code:
                        print("\033[91mCould not find environment code.\033[0m")
                        input("\nPress Enter to continue...")
                        continue

                    # Check for training results
                    print("\nChecking for training data...")
                    from uniwrap.rl_agent import get_latest_model_path, load_model_metadata

                    model_path = get_latest_model_path(env_name)
                    training_results = {}

                    if model_path:
                        metadata = load_model_metadata(model_path)
                        if metadata:
                            print(f"  Found model: {model_path.name}")
                            print(f"  Trained with obs shape: {metadata.get('observation_shape')}")

                    # Option to do quick training for analysis
                    print("\nOptions:")
                    print("  1. Quick train (1000 steps) then analyze")
                    print("  2. Analyze code only (no training data)")
                    print("  b. Back")
                    analyze_choice = input("Select [1]: ").strip() or "1"

                    if analyze_choice == 'b':
                        continue
                    elif analyze_choice == '1':
                        # Quick training
                        print("\n🏃 Running quick training (1000 steps)...")

                        # Import and run training
                        for key in list(sys.modules.keys()):
                            if key.startswith(f'environments.{env_name}'):
                                del sys.modules[key]

                        env_module = importlib.import_module(f'environments.{env_name}')
                        env_class_name = env_module.__all__[0] if hasattr(env_module, '__all__') else None
                        if not env_class_name:
                            for n in dir(env_module):
                                if n.endswith('Env') and not n.startswith('_'):
                                    env_class_name = n
                                    break
                        env_class = getattr(env_module, env_class_name)

                        from uniwrap.rl_agent import RLAgent
                        agent = RLAgent(env_class)

                        episode_rewards = []
                        episode_lengths = []

                        def collect_callback(data):
                            if data['type'] == 'episode_complete':
                                episode_rewards.append(data['reward'])
                                episode_lengths.append(data['length'])
                                ep = data['episode']
                                r = data['reward']
                                print(f"  Episode {ep:3d} | Reward: {r:7.1f}")

                        try:
                            agent.train(
                                total_timesteps=1000,
                                progress_callback=collect_callback,
                                save_path=None  # Don't save
                            )
                        except Exception as e:
                            print(f"\033[91mTraining error: {e}\033[0m")
                        finally:
                            agent.close()

                        training_results = {
                            'episode_rewards': episode_rewards,
                            'episode_lengths': episode_lengths
                        }

                    # Run analysis
                    print("\n🔍 Analyzing environment with AI supervisor...")
                    from uniwrap.env_supervisor import analyze_environment, print_analysis_report, quick_reward_check, improve_environment

                    # Quick check first
                    if training_results.get('episode_rewards'):
                        quick = quick_reward_check(training_results['episode_rewards'])
                        if quick['status'] == 'issues_found':
                            print("\n⚡ Quick Check Results:")
                            for issue in quick['issues']:
                                print(f"  {issue['severity'].upper()}: {issue['message']}")

                    # Full AI analysis
                    analysis = analyze_environment(
                        env_code=env_code,
                        training_results=training_results
                    )

                    print_analysis_report(analysis)

                    # Offer to improve
                    if analysis.get('quality_score', 10) < 7 or analysis.get('issues'):
                        print("\n\033[93mWould you like the AI to suggest improvements?\033[0m")
                        print("  1. Yes, generate improved environment")
                        print("  2. No, just show analysis")
                        improve_choice = input("Select [2]: ").strip() or "2"

                        if improve_choice == '1':
                            feedback = input("\nAny specific feedback or requirements? (Enter to skip): ").strip()

                            print("\n🔧 Generating improved environment...")
                            improved_code = improve_environment(
                                env_code=env_code,
                                analysis=analysis,
                                user_feedback=feedback
                            )

                            # Show diff summary
                            print("\n📝 Improvements generated!")
                            print(f"   Original: {len(env_code)} chars")
                            print(f"   Improved: {len(improved_code)} chars")

                            # Ask to save
                            print("\nOptions:")
                            print("  1. Save as new version (e.g., env_v2)")
                            print("  2. Overwrite current")
                            print("  3. Show code (don't save)")
                            print("  4. Cancel")
                            save_choice = input("Select [1]: ").strip() or "1"

                            if save_choice == '1':
                                # Save as new version
                                version = 2
                                while Path(f'environments/{env_name}_v{version}').exists():
                                    version += 1
                                new_env_name = f"{env_name}_v{version}"
                                new_env_path = Path(f'environments/{new_env_name}')
                                new_env_path.mkdir(parents=True, exist_ok=True)

                                # Write improved code
                                new_file = new_env_path / env_file.name
                                new_file.write_text(improved_code)

                                # Write __init__.py
                                init_content = env_path.joinpath('__init__.py').read_text()
                                (new_env_path / '__init__.py').write_text(
                                    init_content.replace(env_name, new_env_name)
                                )

                                # Update top-level __init__.py
                                top_init = Path('environments/__init__.py')
                                with open(top_init, 'a') as f:
                                    f.write(f"from . import {new_env_name}\n")

                                print(f"\n\033[92m✅ Saved as: {new_env_name}\033[0m")
                                print(f"   Path: {new_env_path}")

                            elif save_choice == '2':
                                # Overwrite
                                confirm = input("Are you sure you want to overwrite? (yes/no): ").strip().lower()
                                if confirm == 'yes':
                                    env_file.write_text(improved_code)
                                    print(f"\n\033[92m✅ Overwritten: {env_file}\033[0m")
                                else:
                                    print("Cancelled.")

                            elif save_choice == '3':
                                print("\n" + "=" * 60)
                                print("IMPROVED CODE:")
                                print("=" * 60)
                                print(improved_code[:3000])
                                if len(improved_code) > 3000:
                                    print(f"\n... ({len(improved_code) - 3000} more chars)")

            except (ValueError, IndexError) as e:
                print(f"\033[91mInvalid selection: {e}\033[0m")

            input("\nPress Enter to continue...")

        elif choice == '8':
            # Delete environment
            if not envs:
                print("\n\033[91mNo environments available.\033[0m")
                input("\nPress Enter to continue...")
                continue

            print("\nAvailable environments:")
            for i, (name, cls) in enumerate(envs, 1):
                # Check for associated models
                model_path = Path(f'models/{name}_ppo.zip')
                has_model = " [has trained model]" if model_path.exists() else ""
                print(f"  {i}. {name} ({cls}){has_model}")
            print("  b. Back to main menu")

            selection = input("\nSelect environment to delete (number or 'b'): ").strip().lower()
            if selection == 'b':
                continue

            try:
                idx = int(selection) - 1
                if 0 <= idx < len(envs):
                    env_name = envs[idx][0]
                    env_path = Path(f'environments/{env_name}')

                    # Show what will be deleted
                    print(f"\n\033[93mThis will delete:\033[0m")
                    print(f"  - Environment folder: {env_path}")

                    # Check for associated files
                    model_path = Path(f'models/{env_name}_ppo.zip')
                    metadata_path = Path(f'models/{env_name}_ppo.json')
                    graph_paths = list(Path('training_graphs').glob(f'{env_name}_*.png')) if Path('training_graphs').exists() else []

                    if model_path.exists():
                        print(f"  - Trained model: {model_path}")
                    if metadata_path.exists():
                        print(f"  - Model metadata: {metadata_path}")
                    if graph_paths:
                        print(f"  - Training graphs: {len(graph_paths)} file(s)")

                    # Confirm deletion
                    confirm = input("\n\033[91mType 'delete' to confirm: \033[0m").strip().lower()

                    if confirm == 'delete':
                        import shutil

                        # Delete environment folder
                        if env_path.exists():
                            shutil.rmtree(env_path)
                            print(f"  ✓ Deleted {env_path}")

                        # Ask about associated files
                        if model_path.exists() or metadata_path.exists() or graph_paths:
                            delete_associated = input("Delete associated model and graphs too? (y/n) [n]: ").strip().lower()
                            if delete_associated == 'y':
                                if model_path.exists():
                                    model_path.unlink()
                                    print(f"  ✓ Deleted {model_path}")
                                if metadata_path.exists():
                                    metadata_path.unlink()
                                    print(f"  ✓ Deleted {metadata_path}")
                                for gp in graph_paths:
                                    gp.unlink()
                                    print(f"  ✓ Deleted {gp}")

                        print(f"\n\033[92m✅ Environment '{env_name}' deleted.\033[0m")

                        # Refresh env list
                        envs = list_environments()
                    else:
                        print("Cancelled.")

            except (ValueError, IndexError):
                print("\033[91mInvalid selection.\033[0m")

            input("\nPress Enter to continue...")

        elif choice == 'v':
            # Toggle visual mode
            visual_mode = not visual_mode
            status = "ON" if visual_mode else "OFF"
            print(f"\n\033[93mVisual mode: {status}\033[0m")
            time.sleep(0.5)

        else:
            print("\033[91mInvalid choice.\033[0m")
            time.sleep(1)


if __name__ == "__main__":
    main()
