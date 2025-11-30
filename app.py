#!/usr/bin/env python3
"""
UniWrap Web App - Generate and evaluate RL environments from any game.

Run with: python app.py
Then open: http://localhost:8080
"""

import os
import sys
import json
import subprocess
import threading
import queue
import time
import tempfile
import importlib
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add venv site-packages to path if running outside venv
venv_site = project_root / 'venv' / 'lib'
if venv_site.exists():
    for pydir in venv_site.iterdir():
        if pydir.name.startswith('python'):
            site_packages = pydir / 'site-packages'
            if site_packages.exists() and str(site_packages) not in sys.path:
                sys.path.insert(0, str(site_packages))
            break

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Global state for streaming logs
generation_logs = {}
evaluation_results = {}
eval_stream_params = {}  # Store params for SSE evaluation streams


class UniWrapAPI(BaseHTTPRequestHandler):
    """HTTP handler for UniWrap web app."""

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def _send_json(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_html(self, html):
        """Send HTML response."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def _handle_eval_stream(self, job_id):
        """Handle Server-Sent Events for live evaluation streaming."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('X-Accel-Buffering', 'no')
        self.end_headers()

        print(f"[SSE] Starting eval stream for job {job_id}")

        # Get params from the stream queue
        if job_id not in eval_stream_params:
            self.wfile.write(b'data: {"error": "Invalid job ID"}\n\n')
            self.wfile.flush()
            return

        params = eval_stream_params[job_id]
        episodes = params.get('episodes', 10)

        # Check request type
        if params.get('type') == 'web':
            self._handle_web_eval_stream(job_id, params)
            return
        elif params.get('type') == 'train':
            self._handle_train_stream(job_id, params)
            return
        elif params.get('type') == 'trained_eval':
            self._handle_trained_eval_stream(job_id, params)
            return

        env_name = params['env_name']

        try:
            # Clear cached module to ensure fresh import
            module_name = f'environments.{env_name}'
            if module_name in sys.modules:
                del sys.modules[module_name]

            # Also clear the submodule
            for key in list(sys.modules.keys()):
                if key.startswith(f'environments.{env_name}'):
                    del sys.modules[key]

            # Import the environment fresh
            env_module = importlib.import_module(f'environments.{env_name}')

            # Check if this is a web-based environment (uses playwright)
            env_path = Path(f'environments/{env_name}')
            is_web_env = False
            for py_file in env_path.glob('*.py'):
                content = py_file.read_text()
                if 'playwright' in content.lower() or 'sync_playwright' in content:
                    is_web_env = True
                    break

            if is_web_env:
                # Run web-based environment evaluation in a separate thread
                self._handle_playwright_env_stream(job_id, params, env_module)
                return

            env_class_name = env_module.__all__[0] if hasattr(env_module, '__all__') else None
            if not env_class_name:
                for name in dir(env_module):
                    if name.endswith('Env') and not name.startswith('_'):
                        env_class_name = name
                        break

            env_class = getattr(env_module, env_class_name)
            env = env_class(render_mode=None)

            # Get game dimensions
            game_width = getattr(env, 'frame_size_x', 720)
            game_height = getattr(env, 'frame_size_y', 480)
            grid_size = getattr(env, 'grid_size', 10)

            # Send initial config
            config_data = json.dumps({
                'type': 'config',
                'game_width': game_width,
                'game_height': game_height,
                'grid_size': grid_size,
                'total_episodes': episodes
            })
            self.wfile.write(f'data: {config_data}\n\n'.encode())
            self.wfile.flush()

            # Run evaluation with streaming
            all_results = []
            import time as time_module

            for ep in range(episodes):
                obs, info = env.reset()
                episode_reward = 0
                episode_steps = 0
                max_steps = 500  # Limit steps per episode

                print(f"[SSE] Episode {ep+1}/{episodes} starting - snake at {env.snake_body[0] if hasattr(env, 'snake_body') else 'N/A'}")

                # Send episode start
                start_data = json.dumps({
                    'type': 'episode_start',
                    'episode': ep + 1,
                    'snake_body': [list(pos) for pos in env.snake_body] if hasattr(env, 'snake_body') else [],
                    'food_pos': list(env.food_pos) if hasattr(env, 'food_pos') else []
                })
                self.wfile.write(f'data: {start_data}\n\n'.encode())
                self.wfile.flush()

                while episode_steps < max_steps:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    episode_steps += 1

                    # Send frame update (every step for visualization)
                    frame_data = json.dumps({
                        'type': 'frame',
                        'episode': ep + 1,
                        'step': episode_steps,
                        'snake_body': [list(pos) for pos in env.snake_body] if hasattr(env, 'snake_body') else [],
                        'food_pos': list(env.food_pos) if hasattr(env, 'food_pos') else [],
                        'score': info.get('score', 0),
                        'reward': round(reward, 3),
                        'total_reward': round(episode_reward, 2),
                        'action': int(action),
                        'terminated': terminated
                    })
                    self.wfile.write(f'data: {frame_data}\n\n'.encode())
                    self.wfile.flush()

                    # Small delay to allow browser to render (10ms)
                    time_module.sleep(0.01)

                    if terminated or truncated:
                        break

                # Send episode end
                score = info.get('score', 0)
                all_results.append({
                    'episode': ep + 1,
                    'steps': episode_steps,
                    'reward': round(episode_reward, 2),
                    'score': score
                })

                end_data = json.dumps({
                    'type': 'episode_end',
                    'episode': ep + 1,
                    'steps': episode_steps,
                    'reward': round(episode_reward, 2),
                    'score': score
                })
                self.wfile.write(f'data: {end_data}\n\n'.encode())
                self.wfile.flush()

                print(f"[SSE] Episode {ep+1} ended - steps={episode_steps}, score={score}")

            env.close()
            print(f"[SSE] All {episodes} episodes complete")

            # Send final results
            total_reward = sum(r['reward'] for r in all_results)
            total_steps = sum(r['steps'] for r in all_results)
            total_score = sum(r['score'] for r in all_results)
            max_score = max(r['score'] for r in all_results)

            final_data = json.dumps({
                'type': 'complete',
                'episodes': all_results,
                'avg_reward': round(total_reward / episodes, 2),
                'avg_steps': round(total_steps / episodes, 1),
                'avg_score': round(total_score / episodes, 2),
                'max_score': max_score
            })
            self.wfile.write(f'data: {final_data}\n\n'.encode())
            self.wfile.flush()
            print(f"[SSE] Sent complete message")

        except Exception as e:
            print(f"[SSE] Error: {e}")
            import traceback
            traceback.print_exc()
            error_data = json.dumps({'type': 'error', 'error': str(e)})
            self.wfile.write(f'data: {error_data}\n\n'.encode())
            self.wfile.flush()

        # Cleanup
        if job_id in eval_stream_params:
            del eval_stream_params[job_id]

    def _handle_web_eval_stream(self, job_id: str, params: dict):
        """Handle Server-Sent Events for web game evaluation streaming."""
        game_url = params['url']
        episodes = params['episodes']

        try:
            from uniwrap.web_game_env import create_web_game_env
            import base64
            from PIL import Image
            import io

            # Create the web game environment
            env = create_web_game_env(game_url, render_mode=None)

            # Send initial config
            config_data = json.dumps({
                'type': 'config',
                'game_url': game_url,
                'game_width': env.width,
                'game_height': env.height,
                'total_episodes': episodes,
                'is_web_game': True
            })
            self.wfile.write(f'data: {config_data}\n\n'.encode())
            self.wfile.flush()

            # Run evaluation with streaming
            all_results = []

            for ep in range(episodes):
                obs, info = env.reset()
                episode_reward = 0
                episode_steps = 0

                # Send episode start with screenshot
                screenshot_b64 = self._obs_to_base64(obs)
                start_data = json.dumps({
                    'type': 'episode_start',
                    'episode': ep + 1,
                    'screenshot': screenshot_b64,
                    'score': 0
                })
                self.wfile.write(f'data: {start_data}\n\n'.encode())
                self.wfile.flush()

                max_steps = 1000
                while episode_steps < max_steps:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    episode_steps += 1

                    # Send frame update every few steps to avoid overwhelming the browser
                    if episode_steps % 3 == 0 or terminated:
                        screenshot_b64 = self._obs_to_base64(obs)
                        frame_data = json.dumps({
                            'type': 'frame',
                            'episode': ep + 1,
                            'step': episode_steps,
                            'screenshot': screenshot_b64,
                            'score': info.get('score', 0),
                            'reward': round(reward, 3),
                            'total_reward': round(episode_reward, 2),
                            'action': int(action),
                            'terminated': terminated
                        })
                        self.wfile.write(f'data: {frame_data}\n\n'.encode())
                        self.wfile.flush()

                    if terminated or truncated:
                        break

                # Send episode end
                score = info.get('score', 0)
                all_results.append({
                    'episode': ep + 1,
                    'steps': episode_steps,
                    'reward': round(episode_reward, 2),
                    'score': score
                })

                end_data = json.dumps({
                    'type': 'episode_end',
                    'episode': ep + 1,
                    'steps': episode_steps,
                    'reward': round(episode_reward, 2),
                    'score': score
                })
                self.wfile.write(f'data: {end_data}\n\n'.encode())
                self.wfile.flush()

            env.close()

            # Send final results
            total_reward = sum(r['reward'] for r in all_results)
            total_steps = sum(r['steps'] for r in all_results)
            total_score = sum(r['score'] for r in all_results)
            max_score = max(r['score'] for r in all_results) if all_results else 0

            final_data = json.dumps({
                'type': 'complete',
                'episodes': all_results,
                'avg_reward': round(total_reward / episodes, 2) if episodes > 0 else 0,
                'avg_steps': round(total_steps / episodes, 1) if episodes > 0 else 0,
                'avg_score': round(total_score / episodes, 2) if episodes > 0 else 0,
                'max_score': max_score
            })
            self.wfile.write(f'data: {final_data}\n\n'.encode())
            self.wfile.flush()

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_data = json.dumps({'type': 'error', 'error': str(e)})
            self.wfile.write(f'data: {error_data}\n\n'.encode())
            self.wfile.flush()

        # Cleanup
        if job_id in eval_stream_params:
            del eval_stream_params[job_id]

    def _obs_to_base64(self, obs) -> str:
        """Convert observation array to base64 encoded JPEG."""
        from PIL import Image
        import io
        import base64

        # Handle different observation shapes
        if len(obs.shape) == 2:
            # Grayscale without channel dim (H, W)
            img = Image.fromarray(obs, mode='L')
        elif obs.shape[-1] == 1:
            # Grayscale with channel dim (H, W, 1)
            img = Image.fromarray(obs.squeeze(), mode='L')
        else:
            # RGB (H, W, 3)
            img = Image.fromarray(obs, mode='RGB')

        # Encode as JPEG
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=70)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _handle_playwright_env_stream(self, job_id: str, params: dict, env_module):
        """Handle evaluation of Playwright-based environments in a separate thread."""
        import concurrent.futures
        from queue import Queue
        import base64

        env_name = params['env_name']
        episodes = params.get('episodes', 10)

        print(f"[SSE] Running Playwright environment {env_name} in thread")

        # Find environment class
        env_class_name = env_module.__all__[0] if hasattr(env_module, '__all__') else None
        if not env_class_name:
            for name in dir(env_module):
                if name.endswith('Env') and not name.startswith('_'):
                    env_class_name = name
                    break

        env_class = getattr(env_module, env_class_name)

        # Queue to receive results from thread
        result_queue = Queue()

        def run_evaluation():
            """Run the evaluation in a separate thread where Playwright sync API works."""
            try:
                env = env_class(render_mode=None)

                # Get dimensions from observation space
                obs_shape = env.observation_space.shape
                game_width = obs_shape[1] if len(obs_shape) >= 2 else 84
                game_height = obs_shape[0] if len(obs_shape) >= 2 else 84

                result_queue.put({
                    'type': 'config',
                    'game_width': game_width,
                    'game_height': game_height,
                    'total_episodes': episodes,
                    'is_web_game': True
                })

                all_results = []

                for ep in range(episodes):
                    obs, info = env.reset()
                    episode_reward = 0
                    episode_steps = 0
                    max_steps = 500

                    # Send episode start with screenshot
                    result_queue.put({
                        'type': 'episode_start',
                        'episode': ep + 1,
                        'obs': obs,
                        'score': info.get('score', info.get('survival_time', 0))
                    })

                    while episode_steps < max_steps:
                        action = env.action_space.sample()
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_reward += reward
                        episode_steps += 1

                        # Send frame update every few steps
                        if episode_steps % 5 == 0 or terminated or truncated:
                            result_queue.put({
                                'type': 'frame',
                                'episode': ep + 1,
                                'step': episode_steps,
                                'obs': obs,
                                'score': info.get('score', info.get('survival_time', 0)),
                                'reward': float(reward),
                                'total_reward': float(episode_reward),
                                'action': int(action),
                                'terminated': terminated
                            })

                        if terminated or truncated:
                            break

                    score = info.get('score', info.get('survival_time', 0))
                    all_results.append({
                        'episode': ep + 1,
                        'steps': episode_steps,
                        'reward': round(float(episode_reward), 2),
                        'score': round(float(score), 2) if isinstance(score, float) else score
                    })

                    result_queue.put({
                        'type': 'episode_end',
                        'episode': ep + 1,
                        'steps': episode_steps,
                        'reward': round(float(episode_reward), 2),
                        'score': round(float(score), 2) if isinstance(score, float) else score
                    })

                env.close()

                # Calculate final results
                total_reward = sum(r['reward'] for r in all_results)
                total_steps = sum(r['steps'] for r in all_results)
                total_score = sum(r['score'] for r in all_results)
                max_score = max(r['score'] for r in all_results) if all_results else 0

                result_queue.put({
                    'type': 'complete',
                    'episodes': all_results,
                    'avg_reward': round(total_reward / episodes, 2) if episodes > 0 else 0,
                    'avg_steps': round(total_steps / episodes, 1) if episodes > 0 else 0,
                    'avg_score': round(total_score / episodes, 2) if episodes > 0 else 0,
                    'max_score': max_score
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                result_queue.put({'type': 'error', 'error': str(e)})

        # Start evaluation in thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_evaluation)

            # Stream results as they come in
            while True:
                try:
                    msg = result_queue.get(timeout=60)

                    # Convert obs to base64 if present
                    if 'obs' in msg:
                        msg['screenshot'] = self._obs_to_base64(msg['obs'])
                        del msg['obs']

                    data = json.dumps(msg)
                    self.wfile.write(f'data: {data}\n\n'.encode())
                    self.wfile.flush()

                    if msg['type'] in ('complete', 'error'):
                        break

                except Exception as e:
                    # Timeout or other error
                    if future.done():
                        break
                    continue

        # Cleanup
        if job_id in eval_stream_params:
            del eval_stream_params[job_id]

    def _handle_train_stream(self, job_id: str, params: dict):
        """Handle Server-Sent Events for RL training progress."""
        env_name = params['env_name']
        timesteps = params.get('timesteps', 10000)

        print(f"[TRAIN] Starting training for {env_name} with {timesteps} timesteps")

        try:
            from uniwrap.rl_agent import RLAgent, get_model_path

            # Import environment
            module_name = f'environments.{env_name}'
            if module_name in sys.modules:
                del sys.modules[module_name]
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

            # Create a test env to get dimensions
            test_env = env_class(render_mode=None)
            game_width = getattr(test_env, 'game_width', 720)
            game_height = getattr(test_env, 'game_height', 480)
            grid_size = getattr(test_env, 'grid_size', 10)
            test_env.close()

            # Send config first
            config_data = json.dumps({
                'type': 'config',
                'game_width': game_width,
                'game_height': game_height,
                'grid_size': grid_size
            })
            self.wfile.write(f'data: {config_data}\n\n'.encode())
            self.wfile.flush()

            # Progress callback to stream updates
            def progress_callback(data):
                # Handle numpy arrays in obs
                if 'obs' in data:
                    obs = data['obs']
                    if hasattr(obs, 'shape'):
                        data['screenshot'] = self._obs_to_base64(obs)
                    del data['obs']

                msg = json.dumps(data)
                try:
                    self.wfile.write(f'data: {msg}\n\n'.encode())
                    self.wfile.flush()
                except:
                    pass

            # Train
            agent = RLAgent(env_class)
            save_path = get_model_path(env_name)

            results = agent.train(
                total_timesteps=timesteps,
                progress_callback=progress_callback,
                save_path=save_path
            )

            results['model_path'] = str(save_path)
            agent.close()

            print(f"[TRAIN] Training complete: {results}")

        except Exception as e:
            print(f"[TRAIN] Error: {e}")
            import traceback
            traceback.print_exc()
            error_data = json.dumps({'type': 'error', 'error': str(e)})
            self.wfile.write(f'data: {error_data}\n\n'.encode())
            self.wfile.flush()

        # Cleanup
        if job_id in eval_stream_params:
            del eval_stream_params[job_id]

    def _handle_trained_eval_stream(self, job_id: str, params: dict):
        """Handle Server-Sent Events for trained agent evaluation."""
        env_name = params['env_name']
        episodes = params.get('episodes', 10)

        print(f"[EVAL] Starting trained agent evaluation for {env_name}")

        try:
            from uniwrap.rl_agent import RLAgent, get_model_path
            import time as time_module

            # Check if model exists
            model_path = get_model_path(env_name)
            if not model_path.exists():
                error_data = json.dumps({
                    'type': 'error',
                    'error': f'No trained model found for {env_name}. Train first!'
                })
                self.wfile.write(f'data: {error_data}\n\n'.encode())
                self.wfile.flush()
                return

            # Import environment
            module_name = f'environments.{env_name}'
            if module_name in sys.modules:
                del sys.modules[module_name]
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

            # Load agent
            agent = RLAgent(env_class)
            agent.load(model_path)

            # Get game dimensions for visualization
            env = env_class(render_mode=None)
            game_width = getattr(env, 'frame_size_x', 720)
            game_height = getattr(env, 'frame_size_y', 480)
            grid_size = getattr(env, 'grid_size', 10)

            # Send config
            config_data = json.dumps({
                'type': 'config',
                'game_width': game_width,
                'game_height': game_height,
                'grid_size': grid_size,
                'total_episodes': episodes,
                'agent_type': 'trained'
            })
            self.wfile.write(f'data: {config_data}\n\n'.encode())
            self.wfile.flush()

            # Run evaluation
            all_results = []

            for ep in range(episodes):
                obs, info = env.reset()
                episode_reward = 0
                episode_steps = 0
                max_steps = 500

                print(f"[EVAL] Episode {ep+1}/{episodes} starting")

                # Send episode start
                start_data = json.dumps({
                    'type': 'episode_start',
                    'episode': ep + 1,
                    'snake_body': [list(pos) for pos in env.snake_body] if hasattr(env, 'snake_body') else [],
                    'food_pos': list(env.food_pos) if hasattr(env, 'food_pos') else []
                })
                self.wfile.write(f'data: {start_data}\n\n'.encode())
                self.wfile.flush()

                while episode_steps < max_steps:
                    # Use trained agent's prediction
                    action = agent.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    episode_steps += 1

                    # Send frame update
                    frame_data = json.dumps({
                        'type': 'frame',
                        'episode': ep + 1,
                        'step': episode_steps,
                        'snake_body': [list(pos) for pos in env.snake_body] if hasattr(env, 'snake_body') else [],
                        'food_pos': list(env.food_pos) if hasattr(env, 'food_pos') else [],
                        'score': info.get('score', 0),
                        'reward': round(float(reward), 3),
                        'total_reward': round(float(episode_reward), 2),
                        'action': int(action),
                        'terminated': terminated
                    })
                    self.wfile.write(f'data: {frame_data}\n\n'.encode())
                    self.wfile.flush()

                    time_module.sleep(0.01)

                    if terminated or truncated:
                        break

                # Send episode end
                score = info.get('score', 0)
                all_results.append({
                    'episode': ep + 1,
                    'steps': episode_steps,
                    'reward': round(float(episode_reward), 2),
                    'score': score
                })

                end_data = json.dumps({
                    'type': 'episode_end',
                    'episode': ep + 1,
                    'steps': episode_steps,
                    'reward': round(float(episode_reward), 2),
                    'score': score
                })
                self.wfile.write(f'data: {end_data}\n\n'.encode())
                self.wfile.flush()

                print(f"[EVAL] Episode {ep+1} ended - steps={episode_steps}, score={score}")

            env.close()
            agent.close()

            # Send final results
            total_reward = sum(r['reward'] for r in all_results)
            total_steps = sum(r['steps'] for r in all_results)
            total_score = sum(r['score'] for r in all_results)
            max_score = max(r['score'] for r in all_results)

            final_data = json.dumps({
                'type': 'complete',
                'episodes': all_results,
                'avg_reward': round(total_reward / episodes, 2),
                'avg_steps': round(total_steps / episodes, 1),
                'avg_score': round(total_score / episodes, 2),
                'max_score': max_score,
                'agent_type': 'trained'
            })
            self.wfile.write(f'data: {final_data}\n\n'.encode())
            self.wfile.flush()

            print(f"[EVAL] Trained agent evaluation complete")

        except Exception as e:
            print(f"[EVAL] Error: {e}")
            import traceback
            traceback.print_exc()
            error_data = json.dumps({'type': 'error', 'error': str(e)})
            self.wfile.write(f'data: {error_data}\n\n'.encode())
            self.wfile.flush()

        # Cleanup
        if job_id in eval_stream_params:
            del eval_stream_params[job_id]

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path

        if path == '/' or path == '/index.html':
            self._send_html(get_index_html())

        elif path == '/api/status':
            self._send_json({'status': 'ready', 'environments': list_environments()})

        elif path == '/api/environments':
            self._send_json({'environments': list_environments()})

        elif path.startswith('/api/logs/'):
            job_id = path.split('/')[-1]
            logs = generation_logs.get(job_id, [])
            self._send_json({'logs': logs})

        elif path.startswith('/api/eval-results/'):
            job_id = path.split('/')[-1]
            results = evaluation_results.get(job_id, {})
            self._send_json(results)

        elif path.startswith('/api/eval-stream/'):
            # Server-Sent Events for live evaluation
            job_id = path.split('/')[-1]
            self._handle_eval_stream(job_id)

        elif path.startswith('/api/train-stream/'):
            # Server-Sent Events for training progress
            job_id = path.split('/')[-1]
            self._handle_train_stream(job_id)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests."""
        path = urlparse(self.path).path
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'

        try:
            data = json.loads(post_data.decode())
        except json.JSONDecodeError:
            data = {}

        if path == '/api/generate':
            result = handle_generate(data)
            self._send_json(result, 200 if result.get('success') else 500)

        elif path == '/api/evaluate':
            result = handle_evaluate(data)
            self._send_json(result, 200 if result.get('success') else 500)

        elif path == '/api/start-eval-stream':
            # Start a streaming evaluation - returns job_id for SSE connection
            env_name = data.get('env_name', '')
            episodes = data.get('episodes', 10)
            if not env_name:
                self._send_json({'success': False, 'error': 'No environment specified'})
            else:
                job_id = str(int(time.time() * 1000))
                eval_stream_params[job_id] = {'env_name': env_name, 'episodes': episodes}
                self._send_json({'success': True, 'job_id': job_id})

        elif path == '/api/start-web-eval':
            # Start a web game evaluation - returns job_id for SSE connection
            game_url = data.get('url', '')
            episodes = data.get('episodes', 5)
            if not game_url:
                self._send_json({'success': False, 'error': 'No URL specified'})
            else:
                job_id = str(int(time.time() * 1000))
                eval_stream_params[job_id] = {'url': game_url, 'episodes': episodes, 'type': 'web'}
                self._send_json({'success': True, 'job_id': job_id})

        elif path == '/api/run-episode':
            result = handle_run_episode(data)
            self._send_json(result, 200 if result.get('success') else 500)

        elif path == '/api/start-training':
            # Start training an RL agent - returns job_id for SSE connection
            env_name = data.get('env_name', '')
            timesteps = data.get('timesteps', 10000)
            if not env_name:
                self._send_json({'success': False, 'error': 'No environment specified'})
            else:
                job_id = str(int(time.time() * 1000))
                eval_stream_params[job_id] = {
                    'env_name': env_name,
                    'timesteps': timesteps,
                    'type': 'train'
                }
                self._send_json({'success': True, 'job_id': job_id})

        elif path == '/api/start-trained-eval':
            # Start evaluation with trained agent - returns job_id for SSE connection
            env_name = data.get('env_name', '')
            episodes = data.get('episodes', 10)
            if not env_name:
                self._send_json({'success': False, 'error': 'No environment specified'})
            else:
                job_id = str(int(time.time() * 1000))
                eval_stream_params[job_id] = {
                    'env_name': env_name,
                    'episodes': episodes,
                    'type': 'trained_eval'
                }
                self._send_json({'success': True, 'job_id': job_id})

        elif path == '/api/check-model':
            # Check if a trained model exists for an environment
            env_name = data.get('env_name', '')
            if not env_name:
                self._send_json({'success': False, 'error': 'No environment specified'})
            else:
                from uniwrap.rl_agent import get_model_path
                model_path = get_model_path(env_name)
                self._send_json({
                    'success': True,
                    'has_model': model_path.exists(),
                    'model_path': str(model_path) if model_path.exists() else None
                })

        else:
            self.send_response(404)
            self.end_headers()


def list_environments():
    """List all generated environments."""
    envs = []
    env_dir = Path('environments')
    if env_dir.exists():
        for subdir in env_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('_'):
                init_file = subdir / '__init__.py'
                if init_file.exists():
                    # Try to get the class name
                    try:
                        content = init_file.read_text()
                        for line in content.split('\n'):
                            if 'from' in line and 'import' in line:
                                class_name = line.split('import')[-1].strip()
                                envs.append({
                                    'name': subdir.name,
                                    'class': class_name,
                                    'path': str(subdir)
                                })
                                break
                    except:
                        envs.append({'name': subdir.name, 'path': str(subdir)})
    return envs


def handle_generate(data):
    """Handle environment generation request."""
    url = data.get('url', '').strip()
    if not url:
        return {'success': False, 'error': 'No URL provided'}

    job_id = str(int(time.time() * 1000))
    generation_logs[job_id] = []

    def log(msg):
        generation_logs[job_id].append(msg)
        print(msg)

    try:
        log(f"Starting generation for: {url}")

        # Build command - use venv python if available
        project_root = Path(__file__).parent
        venv_python = project_root / 'venv' / 'bin' / 'python'
        python_exe = str(venv_python) if venv_python.exists() else sys.executable

        cmd = [python_exe, '-m', 'uniwrap', url]

        # Get API key from environment
        env = os.environ.copy()
        api_key = env.get('ANTHROPIC_API_KEY', '')
        if not api_key:
            # Try to load from .env file
            env_file = project_root / '.env'
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        if line.startswith('ANTHROPIC_API_KEY='):
                            api_key = line.split('=', 1)[1].strip().strip('"\'')
                            env['ANTHROPIC_API_KEY'] = api_key
                            break

        if not api_key:
            return {'success': False, 'error': 'ANTHROPIC_API_KEY not configured. Set it in .env file or as environment variable.', 'job_id': job_id}

        # Run UniWrap
        log("Running UniWrap...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(project_root)
        )

        output_lines = []
        for line in process.stdout:
            line = line.strip()
            if line:
                log(line)
                output_lines.append(line)

        process.wait()

        if process.returncode != 0:
            return {
                'success': False,
                'error': 'Generation failed',
                'output': '\n'.join(output_lines),
                'job_id': job_id
            }

        log("Generation complete!")

        # Find generated environment
        envs = list_environments()

        return {
            'success': True,
            'output': '\n'.join(output_lines),
            'environments': envs,
            'job_id': job_id
        }

    except Exception as e:
        log(f"Error: {str(e)}")
        return {'success': False, 'error': str(e), 'job_id': job_id}


def handle_evaluate(data):
    """Handle evaluation request."""
    env_name = data.get('env_name', '')
    episodes = data.get('episodes', 10)

    if not env_name:
        return {'success': False, 'error': 'No environment specified'}

    job_id = str(int(time.time() * 1000))

    try:
        # Import the environment
        env_module = importlib.import_module(f'environments.{env_name}')
        importlib.reload(env_module)  # Reload in case it changed

        env_class_name = env_module.__all__[0] if hasattr(env_module, '__all__') else None
        if not env_class_name:
            for name in dir(env_module):
                if name.endswith('Env') and not name.startswith('_'):
                    env_class_name = name
                    break

        if not env_class_name:
            return {'success': False, 'error': 'Could not find environment class'}

        env_class = getattr(env_module, env_class_name)

        # Create headless environment
        env = env_class(render_mode=None)

        # Run evaluation
        results = {
            'episodes': [],
            'total_reward': 0,
            'total_steps': 0,
            'total_score': 0,
            'max_score': 0,
        }

        for ep in range(episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_steps = 0

            while True:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_steps += 1

                if terminated or truncated:
                    break

            score = info.get('score', 0)
            results['episodes'].append({
                'episode': ep + 1,
                'steps': episode_steps,
                'reward': round(episode_reward, 2),
                'score': score
            })
            results['total_reward'] += episode_reward
            results['total_steps'] += episode_steps
            results['total_score'] += score
            results['max_score'] = max(results['max_score'], score)

        env.close()

        # Calculate averages
        results['avg_reward'] = round(results['total_reward'] / episodes, 2)
        results['avg_steps'] = round(results['total_steps'] / episodes, 1)
        results['avg_score'] = round(results['total_score'] / episodes, 2)

        evaluation_results[job_id] = results

        return {
            'success': True,
            'job_id': job_id,
            'results': results
        }

    except Exception as e:
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def handle_run_episode(data):
    """Run a single episode and return step-by-step data for visualization."""
    env_name = data.get('env_name', '')
    max_steps = data.get('max_steps', 500)

    if not env_name:
        return {'success': False, 'error': 'No environment specified'}

    try:
        # Import the environment
        env_module = importlib.import_module(f'environments.{env_name}')
        importlib.reload(env_module)

        env_class_name = env_module.__all__[0]
        env_class = getattr(env_module, env_class_name)

        # Create environment
        env = env_class(render_mode=None)

        # Get game dimensions if available
        game_width = getattr(env, 'frame_size_x', 720)
        game_height = getattr(env, 'frame_size_y', 480)
        grid_size = getattr(env, 'grid_size', 10)

        # Run one episode
        obs, info = env.reset()
        frames = []
        total_reward = 0

        # Capture initial state
        frames.append({
            'step': 0,
            'snake_body': [list(pos) for pos in env.snake_body] if hasattr(env, 'snake_body') else [],
            'food_pos': list(env.food_pos) if hasattr(env, 'food_pos') else [],
            'score': 0,
            'reward': 0,
            'total_reward': 0,
            'action': -1,
            'direction': getattr(env, 'direction', 'RIGHT'),
            'terminated': False
        })

        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Capture game state for visualization
            frame = {
                'step': step + 1,
                'snake_body': [list(pos) for pos in env.snake_body] if hasattr(env, 'snake_body') else [],
                'food_pos': list(env.food_pos) if hasattr(env, 'food_pos') else [],
                'score': info.get('score', 0),
                'reward': round(reward, 3),
                'total_reward': round(total_reward, 2),
                'action': int(action),
                'direction': getattr(env, 'direction', ''),
                'terminated': terminated
            }
            frames.append(frame)

            if terminated or truncated:
                break

        env.close()

        return {
            'success': True,
            'frames': frames,
            'game_width': game_width,
            'game_height': game_height,
            'grid_size': grid_size,
            'final_score': info.get('score', 0),
            'total_reward': round(total_reward, 2),
            'total_steps': len(frames) - 1
        }

    except Exception as e:
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def get_index_html():
    """Return the main HTML page."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UniWrap - RL Environment Generator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            min-height: 100vh;
        }

        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 20px 40px;
            border-bottom: 1px solid #2a2a4a;
        }

        .header h1 {
            font-size: 1.8em;
            background: linear-gradient(135deg, #00d9ff, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
        }

        .header p {
            color: #888;
            margin-top: 5px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }

        @media (max-width: 900px) {
            .grid { grid-template-columns: 1fr; }
        }

        .card {
            background: #1a1a2e;
            border-radius: 16px;
            padding: 25px;
            border: 1px solid #2a2a4a;
        }

        .card h2 {
            font-size: 1.3em;
            margin-bottom: 20px;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .card h2 span {
            font-size: 1.2em;
        }

        .input-group {
            margin-bottom: 20px;
        }

        .input-group label {
            display: block;
            margin-bottom: 8px;
            color: #aaa;
            font-size: 0.9em;
        }

        input[type="text"], input[type="number"], select {
            width: 100%;
            padding: 12px 15px;
            background: #0f0f1a;
            border: 1px solid #2a2a4a;
            border-radius: 8px;
            color: #fff;
            font-size: 15px;
            transition: border-color 0.3s;
        }

        input:focus, select:focus {
            outline: none;
            border-color: #00d9ff;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .btn-primary {
            background: linear-gradient(135deg, #00d9ff, #a855f7);
            color: #fff;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0, 217, 255, 0.3);
        }

        .btn-secondary {
            background: #2a2a4a;
            color: #fff;
        }

        .btn-secondary:hover {
            background: #3a3a5a;
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }

        .log-box {
            background: #0a0a14;
            border-radius: 8px;
            padding: 15px;
            height: 200px;
            overflow-y: auto;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
            line-height: 1.6;
            border: 1px solid #1a1a2e;
        }

        .log-line {
            color: #0f0;
        }

        .log-line.error {
            color: #f55;
        }

        .env-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .env-item {
            background: #0f0f1a;
            border: 1px solid #2a2a4a;
            border-radius: 8px;
            padding: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: border-color 0.3s;
        }

        .env-item:hover {
            border-color: #00d9ff;
        }

        .env-item .name {
            font-weight: 600;
            color: #fff;
        }

        .env-item .class {
            color: #888;
            font-size: 0.85em;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-top: 20px;
        }

        .stat-box {
            background: #0f0f1a;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }

        .stat-box .value {
            font-size: 1.8em;
            font-weight: 700;
            color: #00d9ff;
        }

        .stat-box .label {
            font-size: 0.8em;
            color: #888;
            margin-top: 5px;
        }

        .episode-chart {
            height: 150px;
            background: #0f0f1a;
            border-radius: 8px;
            margin-top: 20px;
            display: flex;
            align-items: flex-end;
            padding: 10px;
            gap: 2px;
        }

        .episode-bar {
            flex: 1;
            background: linear-gradient(to top, #00d9ff, #a855f7);
            border-radius: 2px 2px 0 0;
            min-height: 5px;
            transition: height 0.3s;
        }

        .loading {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #888;
        }

        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid #2a2a4a;
            border-top-color: #00d9ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .hidden { display: none !important; }

        .success-msg {
            background: rgba(0, 255, 150, 0.1);
            border: 1px solid rgba(0, 255, 150, 0.3);
            color: #0f6;
            padding: 12px 15px;
            border-radius: 8px;
            margin-top: 15px;
        }

        .error-msg {
            background: rgba(255, 50, 50, 0.1);
            border: 1px solid rgba(255, 50, 50, 0.3);
            color: #f55;
            padding: 12px 15px;
            border-radius: 8px;
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎮 UniWrap</h1>
        <p>Generate Gymnasium RL environments from any game repository</p>
    </div>

    <div class="container">
        <div class="grid">
            <!-- Generate Section -->
            <div class="card">
                <h2><span>⚡</span> Generate Environment</h2>

                <div class="input-group">
                    <label>GitHub Repository URL or Game URL</label>
                    <input type="text" id="repoUrl" placeholder="https://github.com/user/game-repo or https://example.com/game">
                </div>

                <div class="input-group">
                    <label>Hints for the AI (optional)</label>
                    <textarea id="userHints" rows="3" placeholder="e.g., 'The game ends when you hit an obstacle. Score is shown in top-right. Press Space to jump, Down to duck. Game over is detected by a restart button appearing.'" style="width: 100%; background: #0a0a14; border: 1px solid #2a2a4a; border-radius: 6px; padding: 10px; color: #fff; font-family: inherit; resize: vertical;"></textarea>
                </div>

                <button class="btn btn-primary" id="generateBtn" onclick="generate()">
                    <span>🚀</span> Generate
                </button>

                <div id="generateLoading" class="loading hidden" style="margin-top: 15px;">
                    <div class="spinner"></div>
                    <span>Analyzing code and generating environment...</span>
                </div>

                <div id="generateResult"></div>

                <!-- Feedback for regeneration -->
                <div id="feedbackSection" class="hidden" style="margin-top: 15px;">
                    <div class="input-group">
                        <label>Feedback - What's wrong? How should it be fixed?</label>
                        <textarea id="feedbackText" rows="2" placeholder="e.g., 'Game over detection not working - the game freezes but episode continues. Try detecting when screen stops changing.'" style="width: 100%; background: #0a0a14; border: 1px solid #2a2a4a; border-radius: 6px; padding: 10px; color: #fff; font-family: inherit; resize: vertical;"></textarea>
                    </div>
                    <button class="btn btn-secondary" onclick="regenerateWithFeedback()">
                        <span>🔄</span> Regenerate with Feedback
                    </button>
                </div>

                <div class="log-box" id="logBox" style="margin-top: 20px;">
                    <div class="log-line">Ready. Enter a GitHub URL or game URL to generate an RL environment.</div>
                </div>
            </div>

            <!-- Environments Section -->
            <div class="card">
                <h2><span>📦</span> Generated Environments</h2>

                <div id="envList" class="env-list">
                    <p style="color: #666;">No environments generated yet.</p>
                </div>

                <button class="btn btn-secondary" onclick="refreshEnvs()" style="margin-top: 15px;">
                    🔄 Refresh
                </button>
            </div>

            <!-- Web Game Section -->
            <div class="card" style="grid-column: span 2;">
                <h2><span>🌐</span> Web Game (Browser-based)</h2>

                <div style="display: flex; gap: 15px; align-items: flex-end;">
                    <div class="input-group" style="flex: 1; margin-bottom: 0;">
                        <label>Game URL</label>
                        <input type="text" id="webGameUrl" placeholder="https://dinosaur-game.io/">
                    </div>

                    <div class="input-group" style="width: 150px; margin-bottom: 0;">
                        <label>Episodes</label>
                        <input type="number" id="webEpisodes" value="5" min="1" max="20">
                    </div>

                    <button class="btn btn-primary" id="webEvalBtn" onclick="runWebEvaluation()">
                        <span>🎮</span> Play & Evaluate
                    </button>
                </div>

                <!-- Live Web Game Visualization -->
                <div id="webGameContainer" class="hidden" style="margin-top: 20px;">
                    <div style="display: flex; gap: 20px;">
                        <div>
                            <img id="webGameImage" width="600" height="150" style="background: #000; border-radius: 8px; border: 2px solid #2a2a4a; object-fit: contain;">
                        </div>
                        <div style="flex: 1;">
                            <div class="stats-grid" style="grid-template-columns: repeat(3, 1fr);">
                                <div class="stat-box">
                                    <div class="value" id="webEpisodeNum">0</div>
                                    <div class="label">Episode</div>
                                </div>
                                <div class="stat-box">
                                    <div class="value" id="webStepNum">0</div>
                                    <div class="label">Step</div>
                                </div>
                                <div class="stat-box">
                                    <div class="value" id="webCurrentScore">0</div>
                                    <div class="label">Score</div>
                                </div>
                            </div>
                            <div style="margin-top: 15px;">
                                <div style="display: flex; justify-content: space-between; color: #888; font-size: 0.85em; margin-bottom: 5px;">
                                    <span>Episode Progress</span>
                                    <span id="webProgressText">0 / 0 episodes</span>
                                </div>
                                <div style="background: #0a0a14; border-radius: 4px; height: 8px; overflow: hidden;">
                                    <div id="webProgressBar" style="background: linear-gradient(90deg, #00d9ff, #a855f7); height: 100%; width: 0%; transition: width 0.3s;"></div>
                                </div>
                            </div>
                            <div id="webStatus" style="margin-top: 10px; color: #888; font-size: 0.9em;">
                                Enter a game URL and click "Play & Evaluate"
                            </div>
                        </div>
                    </div>
                </div>

                <div id="webResults" class="hidden" style="margin-top: 20px;">
                    <div class="stats-grid">
                        <div class="stat-box">
                            <div class="value" id="webAvgReward">-</div>
                            <div class="label">Avg Reward</div>
                        </div>
                        <div class="stat-box">
                            <div class="value" id="webAvgSteps">-</div>
                            <div class="label">Avg Steps</div>
                        </div>
                        <div class="stat-box">
                            <div class="value" id="webAvgScore">-</div>
                            <div class="label">Avg Score</div>
                        </div>
                        <div class="stat-box">
                            <div class="value" id="webMaxScore">-</div>
                            <div class="label">Max Score</div>
                        </div>
                    </div>
                    <div class="episode-chart" id="webEpisodeChart"></div>
                </div>
            </div>

            <!-- Test & Train Agent Section -->
            <div class="card" style="grid-column: span 2;">
                <h2><span>🎮</span> Test & Train Agent</h2>

                <p style="color: #888; margin-bottom: 20px; font-size: 0.9em;">
                    Select an environment, then: <strong>1)</strong> Test with random actions to verify it works,
                    <strong>2)</strong> Train an RL agent to learn the game,
                    <strong>3)</strong> Watch the trained agent play.
                </p>

                <div style="display: flex; gap: 15px; align-items: flex-end; flex-wrap: wrap;">
                    <div class="input-group" style="flex: 1; min-width: 200px; margin-bottom: 0;">
                        <label>Select Environment</label>
                        <select id="evalEnvSelect" onchange="onEnvChange()">
                            <option value="">-- Select an environment --</option>
                        </select>
                    </div>

                    <div class="input-group" style="width: 130px; margin-bottom: 0;">
                        <label>Episodes</label>
                        <input type="number" id="evalEpisodes" value="10" min="1" max="100">
                    </div>

                    <div class="input-group" style="width: 160px; margin-bottom: 0;">
                        <label>Training Steps</label>
                        <select id="trainSteps">
                            <option value="5000">5K (Quick)</option>
                            <option value="10000" selected>10K (Normal)</option>
                            <option value="25000">25K (Better)</option>
                            <option value="50000">50K (Good)</option>
                            <option value="100000">100K (Best)</option>
                        </select>
                    </div>
                </div>

                <!-- Action Buttons -->
                <div style="display: flex; gap: 10px; margin-top: 20px; flex-wrap: wrap;">
                    <button class="btn btn-secondary" id="evalBtn" onclick="runEvaluation()">
                        <span>🎲</span> 1. Test Random
                    </button>

                    <button class="btn btn-primary" id="trainBtn" onclick="startTraining()">
                        <span>🧠</span> 2. Train Agent
                    </button>

                    <button class="btn btn-secondary" id="trainedEvalBtn" onclick="runTrainedEval()" disabled>
                        <span>🎯</span> 3. Watch Trained
                    </button>

                    <span id="modelStatus" style="display: flex; align-items: center; color: #888; font-size: 0.85em; margin-left: 10px;"></span>
                </div>

                <!-- Training Progress -->
                <div id="trainProgressContainer" class="hidden" style="margin-top: 20px;">
                    <div style="background: #0f0f1a; border-radius: 8px; padding: 15px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <span style="color: #feca57; font-weight: 600;">Training in Progress</span>
                            <span id="trainProgressText" style="color: #888;">0%</span>
                        </div>
                        <div style="background: #1a1a2e; border-radius: 4px; height: 8px; overflow: hidden; margin-bottom: 10px;">
                            <div id="trainProgressBar" style="background: linear-gradient(90deg, #ff6b6b, #feca57); height: 100%; width: 0%; transition: width 0.3s;"></div>
                        </div>
                        <div class="stats-grid" style="grid-template-columns: repeat(4, 1fr); gap: 10px;">
                            <div class="stat-box" style="padding: 10px;">
                                <div class="value" id="trainEpisodes" style="font-size: 1.3em;">0</div>
                                <div class="label">Episodes</div>
                            </div>
                            <div class="stat-box" style="padding: 10px;">
                                <div class="value" id="trainTimesteps" style="font-size: 1.3em;">0</div>
                                <div class="label">Timesteps</div>
                            </div>
                            <div class="stat-box" style="padding: 10px;">
                                <div class="value" id="trainAvgReward" style="font-size: 1.3em;">0</div>
                                <div class="label">Avg Reward</div>
                            </div>
                            <div class="stat-box" style="padding: 10px;">
                                <div class="value" id="trainLastReward" style="font-size: 1.3em;">0</div>
                                <div class="label">Last Ep</div>
                            </div>
                        </div>
                        <div id="trainStatus" style="margin-top: 10px; color: #888; font-size: 0.9em;">
                            Initializing...
                        </div>
                    </div>
                </div>

                <!-- Live Game Visualization -->
                <div id="evalGameContainer" class="hidden" style="margin-top: 20px;">
                    <div style="display: flex; gap: 20px;">
                        <div>
                            <canvas id="evalGameCanvas" width="360" height="240" style="background: #000; border-radius: 8px; border: 2px solid #2a2a4a;"></canvas>
                        </div>
                        <div style="flex: 1;">
                            <div id="agentTypeLabel" style="color: #00d9ff; font-weight: 600; margin-bottom: 10px;">Random Agent</div>
                            <div class="stats-grid" style="grid-template-columns: repeat(3, 1fr);">
                                <div class="stat-box">
                                    <div class="value" id="evalEpisodeNum">0</div>
                                    <div class="label">Episode</div>
                                </div>
                                <div class="stat-box">
                                    <div class="value" id="evalStepNum">0</div>
                                    <div class="label">Step</div>
                                </div>
                                <div class="stat-box">
                                    <div class="value" id="evalCurrentScore">0</div>
                                    <div class="label">Score</div>
                                </div>
                            </div>
                            <div style="margin-top: 15px;">
                                <div style="display: flex; justify-content: space-between; color: #888; font-size: 0.85em; margin-bottom: 5px;">
                                    <span>Episode Progress</span>
                                    <span id="evalProgressText">0 / 0 episodes</span>
                                </div>
                                <div style="background: #0a0a14; border-radius: 4px; height: 8px; overflow: hidden;">
                                    <div id="evalProgressBar" style="background: linear-gradient(90deg, #00d9ff, #a855f7); height: 100%; width: 0%; transition: width 0.3s;"></div>
                                </div>
                            </div>
                            <div id="evalStatus" style="margin-top: 10px; color: #888; font-size: 0.9em;">
                                Ready
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Results -->
                <div id="evalResults" class="hidden" style="margin-top: 20px;">
                    <div class="stats-grid">
                        <div class="stat-box">
                            <div class="value" id="statAvgReward">-</div>
                            <div class="label">Avg Reward</div>
                        </div>
                        <div class="stat-box">
                            <div class="value" id="statAvgSteps">-</div>
                            <div class="label">Avg Steps</div>
                        </div>
                        <div class="stat-box">
                            <div class="value" id="statAvgScore">-</div>
                            <div class="label">Avg Score</div>
                        </div>
                        <div class="stat-box">
                            <div class="value" id="statMaxScore">-</div>
                            <div class="label">Max Score</div>
                        </div>
                    </div>

                    <div class="episode-chart" id="episodeChart"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Game visualization state
        let gameWidth = 720;
        let gameHeight = 480;
        let gridSize = 10;
        let totalEpisodes = 0;
        let eventSource = null;
        let isWebGame = false;  // Track if current eval is a web game

        // Load environments on page load
        document.addEventListener('DOMContentLoaded', refreshEnvs);

        async function refreshEnvs() {
            try {
                const res = await fetch('/api/environments');
                const data = await res.json();

                const envList = document.getElementById('envList');
                const evalSelect = document.getElementById('evalEnvSelect');

                if (data.environments.length === 0) {
                    envList.innerHTML = '<p style="color: #666;">No environments generated yet.</p>';
                    evalSelect.innerHTML = '<option value="">-- No environments available --</option>';
                    return;
                }

                envList.innerHTML = data.environments.map(env => `
                    <div class="env-item">
                        <div>
                            <div class="name">${env.name}</div>
                            <div class="class">${env.class || 'Unknown class'}</div>
                        </div>
                        <button class="btn btn-secondary" onclick="selectEnv('${env.name}')">Select</button>
                    </div>
                `).join('');

                const options = '<option value="">-- Select an environment --</option>' +
                    data.environments.map(env =>
                        `<option value="${env.name}">${env.name} (${env.class})</option>`
                    ).join('');

                evalSelect.innerHTML = options;

                // Check if any environment has a trained model and update options
                data.environments.forEach(async env => {
                    try {
                        const res = await fetch('/api/check-model', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ env_name: env.name })
                        });
                        const modelData = await res.json();
                        if (modelData.has_model) {
                            const option = evalSelect.querySelector(`option[value="${env.name}"]`);
                            if (option) {
                                option.textContent = `${env.name} (${env.class}) [trained]`;
                            }
                        }
                    } catch (e) {}
                });

                // Update model status for currently selected env
                onEnvChange();

            } catch (e) {
                console.error('Failed to load environments:', e);
            }
        }

        function selectEnv(name) {
            document.getElementById('evalEnvSelect').value = name;
            onEnvChange();
        }

        async function onEnvChange() {
            const envName = document.getElementById('evalEnvSelect').value;
            const evalBtn = document.getElementById('trainedEvalBtn');
            const modelStatus = document.getElementById('modelStatus');

            if (!envName) {
                evalBtn.disabled = true;
                modelStatus.innerHTML = '';
                return;
            }

            try {
                const res = await fetch('/api/check-model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ env_name: envName })
                });
                const data = await res.json();

                if (data.has_model) {
                    evalBtn.disabled = false;
                    modelStatus.innerHTML = '<span style="color: #0f6;">✓ Trained model available</span>';
                } else {
                    evalBtn.disabled = true;
                    modelStatus.innerHTML = '<span style="color: #888;">No trained model yet</span>';
                }
            } catch (e) {
                evalBtn.disabled = true;
                modelStatus.innerHTML = '';
            }
        }

        function addLog(msg, isError = false) {
            const logBox = document.getElementById('logBox');
            const line = document.createElement('div');
            line.className = 'log-line' + (isError ? ' error' : '');
            line.textContent = msg;
            logBox.appendChild(line);
            logBox.scrollTop = logBox.scrollHeight;
        }

        let lastGeneratedUrl = '';
        let lastGeneratedEnvName = '';

        async function generate() {
            const url = document.getElementById('repoUrl').value.trim();
            const hints = document.getElementById('userHints').value.trim();

            if (!url) {
                alert('Please enter a repository URL or game URL');
                return;
            }

            const btn = document.getElementById('generateBtn');
            const loading = document.getElementById('generateLoading');
            const result = document.getElementById('generateResult');
            const logBox = document.getElementById('logBox');
            const feedbackSection = document.getElementById('feedbackSection');

            btn.disabled = true;
            loading.classList.remove('hidden');
            result.innerHTML = '';
            logBox.innerHTML = '';
            feedbackSection.classList.add('hidden');

            addLog(`Starting generation for: ${url}`);
            if (hints) {
                addLog(`User hints: ${hints}`);
            }

            try {
                const res = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url, hints })
                });

                const data = await res.json();

                if (data.success) {
                    // Show output in log
                    if (data.output) {
                        data.output.split('\\n').forEach(line => {
                            if (line.trim()) addLog(line);
                        });
                    }

                    result.innerHTML = '<div class="success-msg">✅ Environment generated successfully!</div>';
                    refreshEnvs();

                    // Store for regeneration and show feedback section
                    lastGeneratedUrl = url;
                    lastGeneratedEnvName = data.env_name || '';
                    feedbackSection.classList.remove('hidden');
                    document.getElementById('feedbackText').value = '';
                } else {
                    addLog(data.error || 'Unknown error', true);
                    result.innerHTML = `<div class="error-msg">❌ ${data.error || 'Generation failed'}</div>`;

                    // Still show feedback section for retrying
                    lastGeneratedUrl = url;
                    feedbackSection.classList.remove('hidden');
                }

            } catch (e) {
                addLog(e.message, true);
                result.innerHTML = `<div class="error-msg">❌ ${e.message}</div>`;
            } finally {
                btn.disabled = false;
                loading.classList.add('hidden');
            }
        }

        async function regenerateWithFeedback() {
            const feedback = document.getElementById('feedbackText').value.trim();
            const hints = document.getElementById('userHints').value.trim();

            if (!lastGeneratedUrl) {
                alert('No previous generation to regenerate');
                return;
            }

            const btn = document.querySelector('#feedbackSection button');
            const loading = document.getElementById('generateLoading');
            const result = document.getElementById('generateResult');
            const logBox = document.getElementById('logBox');

            btn.disabled = true;
            loading.classList.remove('hidden');
            result.innerHTML = '';
            logBox.innerHTML = '';

            addLog(`Regenerating with feedback...`);
            if (feedback) {
                addLog(`Feedback: ${feedback}`);
            }

            try {
                const res = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        url: lastGeneratedUrl,
                        hints: hints,
                        feedback: feedback,
                        env_name: lastGeneratedEnvName,
                        regenerate: true
                    })
                });

                const data = await res.json();

                if (data.success) {
                    if (data.output) {
                        data.output.split('\\n').forEach(line => {
                            if (line.trim()) addLog(line);
                        });
                    }

                    result.innerHTML = '<div class="success-msg">✅ Environment regenerated successfully!</div>';
                    refreshEnvs();
                    lastGeneratedEnvName = data.env_name || lastGeneratedEnvName;
                } else {
                    addLog(data.error || 'Unknown error', true);
                    result.innerHTML = `<div class="error-msg">❌ ${data.error || 'Regeneration failed'}</div>`;
                }

            } catch (e) {
                addLog(e.message, true);
                result.innerHTML = `<div class="error-msg">❌ ${e.message}</div>`;
            } finally {
                btn.disabled = false;
                loading.classList.add('hidden');
            }
        }

        async function runEvaluation() {
            const envName = document.getElementById('evalEnvSelect').value;
            const episodes = parseInt(document.getElementById('evalEpisodes').value) || 20;

            if (!envName) {
                alert('Please select an environment');
                return;
            }

            const btn = document.getElementById('evalBtn');
            const gameContainer = document.getElementById('evalGameContainer');
            const results = document.getElementById('evalResults');

            btn.disabled = true;
            document.getElementById('trainBtn').disabled = true;
            document.getElementById('trainedEvalBtn').disabled = true;
            gameContainer.classList.remove('hidden');
            results.classList.add('hidden');

            // Set agent type label
            document.getElementById('agentTypeLabel').textContent = 'Random Agent';
            document.getElementById('agentTypeLabel').style.color = '#00d9ff';

            // Close any existing event source
            if (eventSource) {
                eventSource.close();
            }

            try {
                // First, start the evaluation stream
                const res = await fetch('/api/start-eval-stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ env_name: envName, episodes })
                });

                const data = await res.json();
                if (!data.success) {
                    alert(data.error || 'Failed to start evaluation');
                    btn.disabled = false;
                    return;
                }

                const jobId = data.job_id;
                totalEpisodes = episodes;

                // Setup canvas
                const canvas = document.getElementById('evalGameCanvas');
                canvas.width = 360;
                canvas.height = 240;

                document.getElementById('evalStatus').textContent = 'Connecting...';
                document.getElementById('evalProgressText').textContent = `0 / ${episodes} episodes`;
                document.getElementById('evalProgressBar').style.width = '0%';

                // Connect to SSE stream
                eventSource = new EventSource(`/api/eval-stream/${jobId}`);

                eventSource.onmessage = function(event) {
                    const msg = JSON.parse(event.data);

                    if (msg.type === 'config') {
                        gameWidth = msg.game_width;
                        gameHeight = msg.game_height;
                        gridSize = msg.grid_size || 10;
                        totalEpisodes = msg.total_episodes;
                        isWebGame = msg.is_web_game || false;
                        canvas.height = Math.round(360 * gameHeight / gameWidth);
                        document.getElementById('evalStatus').textContent = 'Running evaluation...';
                    }
                    else if (msg.type === 'episode_start') {
                        document.getElementById('evalEpisodeNum').textContent = msg.episode;
                        document.getElementById('evalStepNum').textContent = '0';
                        document.getElementById('evalCurrentScore').textContent = msg.score || '0';
                        document.getElementById('evalStatus').textContent = `Episode ${msg.episode} started`;
                        if (msg.screenshot) {
                            renderScreenshot(msg.screenshot);
                        } else if (msg.snake_body) {
                            renderEvalFrame({
                                snake_body: msg.snake_body,
                                food_pos: msg.food_pos,
                                score: 0
                            });
                        }
                    }
                    else if (msg.type === 'frame') {
                        document.getElementById('evalStepNum').textContent = msg.step;
                        document.getElementById('evalCurrentScore').textContent = msg.score;
                        if (msg.screenshot) {
                            renderScreenshot(msg.screenshot);
                        } else {
                            renderEvalFrame(msg);
                        }
                    }
                    else if (msg.type === 'episode_end') {
                        const progress = (msg.episode / totalEpisodes) * 100;
                        document.getElementById('evalProgressBar').style.width = progress + '%';
                        document.getElementById('evalProgressText').textContent = `${msg.episode} / ${totalEpisodes} episodes`;
                        document.getElementById('evalStatus').textContent = `Episode ${msg.episode} done - Score: ${msg.score}, Steps: ${msg.steps}`;
                    }
                    else if (msg.type === 'complete') {
                        eventSource.close();
                        eventSource = null;
                        btn.disabled = false;
                        document.getElementById('trainBtn').disabled = false;
                        onEnvChange(); // Re-check trained model status

                        // Update final stats
                        document.getElementById('statAvgReward').textContent = msg.avg_reward;
                        document.getElementById('statAvgSteps').textContent = msg.avg_steps;
                        document.getElementById('statAvgScore').textContent = msg.avg_score;
                        document.getElementById('statMaxScore').textContent = msg.max_score;

                        // Draw chart
                        const chart = document.getElementById('episodeChart');
                        const maxReward = Math.max(...msg.episodes.map(e => Math.abs(e.reward)), 1);
                        chart.innerHTML = msg.episodes.map(e => {
                            const height = Math.max(5, (Math.abs(e.reward) / maxReward) * 120);
                            const color = e.reward >= 0 ? '#0f6' : '#f55';
                            return `<div class="episode-bar" style="height: ${height}px; background: ${color};" title="Ep ${e.episode}: ${e.reward}"></div>`;
                        }).join('');

                        document.getElementById('evalStatus').textContent = 'Evaluation complete!';
                        results.classList.remove('hidden');
                    }
                    else if (msg.type === 'error') {
                        eventSource.close();
                        eventSource = null;
                        btn.disabled = false;
                        document.getElementById('trainBtn').disabled = false;
                        onEnvChange();
                        alert(msg.error);
                    }
                };

                eventSource.onerror = function() {
                    eventSource.close();
                    eventSource = null;
                    btn.disabled = false;
                    document.getElementById('trainBtn').disabled = false;
                    onEnvChange();
                    document.getElementById('evalStatus').textContent = 'Connection lost';
                };

            } catch (e) {
                alert(e.message);
                btn.disabled = false;
                document.getElementById('trainBtn').disabled = false;
                onEnvChange();
            }
        }

        function renderEvalFrame(frame) {
            const canvas = document.getElementById('evalGameCanvas');
            const ctx = canvas.getContext('2d');

            // Scale factor
            const scaleX = canvas.width / gameWidth;
            const scaleY = canvas.height / gameHeight;

            // Clear canvas
            ctx.fillStyle = '#0a0a14';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw grid (subtle)
            ctx.strokeStyle = '#1a1a2e';
            ctx.lineWidth = 0.5;
            for (let x = 0; x < gameWidth; x += gridSize) {
                ctx.beginPath();
                ctx.moveTo(x * scaleX, 0);
                ctx.lineTo(x * scaleX, canvas.height);
                ctx.stroke();
            }
            for (let y = 0; y < gameHeight; y += gridSize) {
                ctx.beginPath();
                ctx.moveTo(0, y * scaleY);
                ctx.lineTo(canvas.width, y * scaleY);
                ctx.stroke();
            }

            // Draw food
            if (frame.food_pos && frame.food_pos.length === 2) {
                ctx.fillStyle = '#ff4444';
                ctx.shadowColor = '#ff4444';
                ctx.shadowBlur = 10;
                ctx.fillRect(
                    frame.food_pos[0] * scaleX,
                    frame.food_pos[1] * scaleY,
                    gridSize * scaleX,
                    gridSize * scaleY
                );
                ctx.shadowBlur = 0;
            }

            // Draw snake
            if (frame.snake_body && frame.snake_body.length > 0) {
                frame.snake_body.forEach((segment, i) => {
                    if (i === 0) {
                        // Head - brighter
                        ctx.fillStyle = '#00ff88';
                        ctx.shadowColor = '#00ff88';
                        ctx.shadowBlur = 8;
                    } else {
                        // Body - gradient
                        const brightness = Math.max(0.3, 1 - (i / frame.snake_body.length) * 0.7);
                        ctx.fillStyle = `rgba(0, ${Math.round(255 * brightness)}, ${Math.round(136 * brightness)}, 1)`;
                        ctx.shadowBlur = 0;
                    }
                    ctx.fillRect(
                        segment[0] * scaleX + 1,
                        segment[1] * scaleY + 1,
                        gridSize * scaleX - 2,
                        gridSize * scaleY - 2
                    );
                });
                ctx.shadowBlur = 0;
            }
        }

        function renderScreenshot(base64Data) {
            // Render screenshot to the eval canvas
            const canvas = document.getElementById('evalGameCanvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            img.onload = function() {
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.src = 'data:image/jpeg;base64,' + base64Data;
        }

        // === Web Game Evaluation Functions ===

        let webEventSource = null;

        async function runWebEvaluation() {
            const gameUrl = document.getElementById('webGameUrl').value.trim();
            const episodes = parseInt(document.getElementById('webEpisodes').value) || 5;

            if (!gameUrl) {
                alert('Please enter a game URL');
                return;
            }

            const btn = document.getElementById('webEvalBtn');
            const gameContainer = document.getElementById('webGameContainer');
            const results = document.getElementById('webResults');

            btn.disabled = true;
            gameContainer.classList.remove('hidden');
            results.classList.add('hidden');

            // Close any existing event source
            if (webEventSource) {
                webEventSource.close();
            }

            try {
                // Start the web evaluation stream
                const res = await fetch('/api/start-web-eval', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: gameUrl, episodes })
                });

                const data = await res.json();
                if (!data.success) {
                    alert(data.error || 'Failed to start evaluation');
                    btn.disabled = false;
                    return;
                }

                const jobId = data.job_id;

                document.getElementById('webStatus').textContent = 'Launching browser...';
                document.getElementById('webProgressText').textContent = `0 / ${episodes} episodes`;
                document.getElementById('webProgressBar').style.width = '0%';

                // Connect to SSE stream
                webEventSource = new EventSource(`/api/eval-stream/${jobId}`);

                webEventSource.onmessage = function(event) {
                    const msg = JSON.parse(event.data);

                    if (msg.type === 'config') {
                        document.getElementById('webStatus').textContent = 'Running evaluation...';
                        // Update image size based on game dimensions
                        const img = document.getElementById('webGameImage');
                        img.width = msg.game_width || 600;
                        img.height = msg.game_height || 150;
                    }
                    else if (msg.type === 'episode_start') {
                        document.getElementById('webEpisodeNum').textContent = msg.episode;
                        document.getElementById('webStepNum').textContent = '0';
                        document.getElementById('webCurrentScore').textContent = '0';
                        document.getElementById('webStatus').textContent = `Episode ${msg.episode} - Playing...`;
                        if (msg.screenshot) {
                            document.getElementById('webGameImage').src = 'data:image/jpeg;base64,' + msg.screenshot;
                        }
                    }
                    else if (msg.type === 'frame') {
                        document.getElementById('webStepNum').textContent = msg.step;
                        document.getElementById('webCurrentScore').textContent = msg.score;
                        if (msg.screenshot) {
                            document.getElementById('webGameImage').src = 'data:image/jpeg;base64,' + msg.screenshot;
                        }
                    }
                    else if (msg.type === 'episode_end') {
                        const progress = (msg.episode / episodes) * 100;
                        document.getElementById('webProgressBar').style.width = progress + '%';
                        document.getElementById('webProgressText').textContent = `${msg.episode} / ${episodes} episodes`;
                        document.getElementById('webStatus').textContent = `Episode ${msg.episode} done - Score: ${msg.score}, Steps: ${msg.steps}`;
                    }
                    else if (msg.type === 'complete') {
                        webEventSource.close();
                        webEventSource = null;
                        btn.disabled = false;

                        // Update final stats
                        document.getElementById('webAvgReward').textContent = msg.avg_reward;
                        document.getElementById('webAvgSteps').textContent = msg.avg_steps;
                        document.getElementById('webAvgScore').textContent = msg.avg_score;
                        document.getElementById('webMaxScore').textContent = msg.max_score;

                        // Draw chart
                        const chart = document.getElementById('webEpisodeChart');
                        const maxReward = Math.max(...msg.episodes.map(e => Math.abs(e.reward)), 1);
                        chart.innerHTML = msg.episodes.map(e => {
                            const height = Math.max(5, (Math.abs(e.reward) / maxReward) * 120);
                            const color = e.reward >= 0 ? '#0f6' : '#f55';
                            return `<div class="episode-bar" style="height: ${height}px; background: ${color};" title="Ep ${e.episode}: ${e.reward}"></div>`;
                        }).join('');

                        document.getElementById('webStatus').textContent = 'Evaluation complete!';
                        results.classList.remove('hidden');
                    }
                    else if (msg.type === 'error') {
                        webEventSource.close();
                        webEventSource = null;
                        btn.disabled = false;
                        document.getElementById('webStatus').textContent = 'Error: ' + msg.error;
                        alert(msg.error);
                    }
                };

                webEventSource.onerror = function() {
                    webEventSource.close();
                    webEventSource = null;
                    btn.disabled = false;
                    document.getElementById('webStatus').textContent = 'Connection lost';
                };

            } catch (e) {
                alert(e.message);
                btn.disabled = false;
            }
        }

        // === RL Training Functions ===

        let trainEventSource = null;
        let trainTotalTimesteps = 10000;

        async function startTraining() {
            const envName = document.getElementById('evalEnvSelect').value;
            const timesteps = parseInt(document.getElementById('trainSteps').value) || 10000;

            if (!envName) {
                alert('Please select an environment');
                return;
            }

            const btn = document.getElementById('trainBtn');
            const evalBtn = document.getElementById('trainedEvalBtn');
            const testBtn = document.getElementById('evalBtn');
            const progressContainer = document.getElementById('trainProgressContainer');
            const gameContainer = document.getElementById('evalGameContainer');

            btn.disabled = true;
            evalBtn.disabled = true;
            testBtn.disabled = true;
            progressContainer.classList.remove('hidden');
            gameContainer.classList.add('hidden');
            document.getElementById('evalResults').classList.add('hidden');

            // Reset stats
            document.getElementById('trainEpisodes').textContent = '0';
            document.getElementById('trainTimesteps').textContent = '0';
            document.getElementById('trainAvgReward').textContent = '0';
            document.getElementById('trainLastReward').textContent = '0';
            document.getElementById('trainProgressBar').style.width = '0%';
            document.getElementById('trainProgressText').textContent = '0%';
            document.getElementById('trainStatus').textContent = 'Starting training...';

            trainTotalTimesteps = timesteps;

            // Close any existing event source
            if (trainEventSource) {
                trainEventSource.close();
            }

            try {
                // Start the training stream
                const res = await fetch('/api/start-training', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ env_name: envName, timesteps })
                });

                const data = await res.json();
                if (!data.success) {
                    alert(data.error || 'Failed to start training');
                    btn.disabled = false;
                    return;
                }

                const jobId = data.job_id;
                document.getElementById('trainStatus').textContent = 'Initializing PPO agent...';

                // Show game visualization during training
                gameContainer.classList.remove('hidden');
                document.getElementById('agentTypeLabel').textContent = 'Training PPO Agent';
                document.getElementById('agentTypeLabel').style.color = '#ff6b6b';

                // Setup canvas
                const canvas = document.getElementById('evalGameCanvas');
                canvas.width = 360;
                canvas.height = 240;

                // Connect to SSE stream
                trainEventSource = new EventSource(`/api/eval-stream/${jobId}`);

                trainEventSource.onmessage = function(event) {
                    const msg = JSON.parse(event.data);

                    if (msg.type === 'config') {
                        gameWidth = msg.game_width;
                        gameHeight = msg.game_height;
                        gridSize = msg.grid_size || 10;
                        canvas.height = Math.round(360 * gameHeight / gameWidth);
                    }
                    else if (msg.type === 'training_start') {
                        document.getElementById('trainStatus').textContent = 'Training in progress...';
                    }
                    else if (msg.type === 'training_frame') {
                        // Update live visualization during training
                        document.getElementById('evalStepNum').textContent = msg.step;
                        document.getElementById('evalCurrentScore').textContent = msg.score || 0;

                        if (msg.screenshot) {
                            renderScreenshot(msg.screenshot);
                        } else if (msg.snake_body) {
                            renderEvalFrame(msg);
                        }
                    }
                    else if (msg.type === 'episode_complete') {
                        document.getElementById('trainEpisodes').textContent = msg.episode;
                        document.getElementById('trainTimesteps').textContent = msg.total_timesteps.toLocaleString();
                        document.getElementById('trainAvgReward').textContent = msg.avg_reward.toFixed(1);
                        document.getElementById('trainLastReward').textContent = msg.reward.toFixed(1);
                        document.getElementById('evalEpisodeNum').textContent = msg.episode;

                        const progress = (msg.total_timesteps / trainTotalTimesteps) * 100;
                        document.getElementById('trainProgressBar').style.width = Math.min(progress, 100) + '%';
                        document.getElementById('trainProgressText').textContent = Math.min(progress, 100).toFixed(1) + '%';
                        document.getElementById('evalProgressBar').style.width = Math.min(progress, 100) + '%';
                        document.getElementById('evalProgressText').textContent = Math.min(progress, 100).toFixed(1) + '%';
                        document.getElementById('trainStatus').textContent = `Episode ${msg.episode}: reward=${msg.reward.toFixed(1)}, avg=${msg.avg_reward.toFixed(1)}`;
                    }
                    else if (msg.type === 'training_complete') {
                        trainEventSource.close();
                        trainEventSource = null;
                        btn.disabled = false;
                        evalBtn.disabled = false;
                        testBtn.disabled = false;

                        document.getElementById('trainProgressBar').style.width = '100%';
                        document.getElementById('trainProgressText').textContent = '100%';
                        document.getElementById('trainStatus').textContent = `Training complete! Final avg reward: ${msg.final_avg_reward.toFixed(2)}`;
                        document.getElementById('modelStatus').innerHTML = '<span style="color: #0f6;">✓ Trained model available</span>';

                        // Refresh envs to update the trained status
                        refreshEnvs();
                    }
                    else if (msg.type === 'error') {
                        trainEventSource.close();
                        trainEventSource = null;
                        btn.disabled = false;
                        testBtn.disabled = false;
                        onEnvChange();
                        document.getElementById('trainStatus').textContent = 'Error: ' + msg.error;
                        alert(msg.error);
                    }
                };

                trainEventSource.onerror = function() {
                    trainEventSource.close();
                    trainEventSource = null;
                    btn.disabled = false;
                    testBtn.disabled = false;
                    onEnvChange();
                    document.getElementById('trainStatus').textContent = 'Connection lost';
                };

            } catch (e) {
                alert(e.message);
                btn.disabled = false;
                testBtn.disabled = false;
                onEnvChange();
            }
        }

        let trainedEvalSource = null;

        async function runTrainedEval() {
            const envName = document.getElementById('evalEnvSelect').value;
            const episodes = parseInt(document.getElementById('evalEpisodes').value) || 10;

            if (!envName) {
                alert('Please select an environment');
                return;
            }

            const btn = document.getElementById('trainedEvalBtn');
            const trainBtn = document.getElementById('trainBtn');
            const testBtn = document.getElementById('evalBtn');
            const gameContainer = document.getElementById('evalGameContainer');
            const results = document.getElementById('evalResults');
            const progressContainer = document.getElementById('trainProgressContainer');

            btn.disabled = true;
            trainBtn.disabled = true;
            testBtn.disabled = true;
            gameContainer.classList.remove('hidden');
            results.classList.add('hidden');
            progressContainer.classList.add('hidden');

            // Set agent type label
            document.getElementById('agentTypeLabel').textContent = 'Trained PPO Agent';
            document.getElementById('agentTypeLabel').style.color = '#feca57';

            // Close any existing event source
            if (trainedEvalSource) {
                trainedEvalSource.close();
            }

            try {
                // Start the trained evaluation stream
                const res = await fetch('/api/start-trained-eval', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ env_name: envName, episodes })
                });

                const data = await res.json();
                if (!data.success) {
                    alert(data.error || 'Failed to start evaluation');
                    btn.disabled = false;
                    trainBtn.disabled = false;
                    testBtn.disabled = false;
                    return;
                }

                const jobId = data.job_id;
                totalEpisodes = episodes;

                // Setup canvas
                const canvas = document.getElementById('evalGameCanvas');
                canvas.width = 360;
                canvas.height = 240;

                document.getElementById('evalStatus').textContent = 'Loading trained model...';
                document.getElementById('evalProgressText').textContent = `0 / ${episodes} episodes`;
                document.getElementById('evalProgressBar').style.width = '0%';

                // Connect to SSE stream
                trainedEvalSource = new EventSource(`/api/eval-stream/${jobId}`);

                trainedEvalSource.onmessage = function(event) {
                    const msg = JSON.parse(event.data);

                    if (msg.type === 'config') {
                        gameWidth = msg.game_width;
                        gameHeight = msg.game_height;
                        gridSize = msg.grid_size;
                        totalEpisodes = msg.total_episodes;
                        canvas.height = Math.round(360 * gameHeight / gameWidth);
                        document.getElementById('evalStatus').textContent = 'Running trained agent...';
                    }
                    else if (msg.type === 'episode_start') {
                        document.getElementById('evalEpisodeNum').textContent = msg.episode;
                        document.getElementById('evalStepNum').textContent = '0';
                        document.getElementById('evalCurrentScore').textContent = '0';
                        document.getElementById('evalStatus').textContent = `Episode ${msg.episode} - Trained agent playing`;
                        renderEvalFrame({
                            snake_body: msg.snake_body,
                            food_pos: msg.food_pos,
                            score: 0
                        });
                    }
                    else if (msg.type === 'frame') {
                        document.getElementById('evalStepNum').textContent = msg.step;
                        document.getElementById('evalCurrentScore').textContent = msg.score;
                        renderEvalFrame(msg);
                    }
                    else if (msg.type === 'episode_end') {
                        const progress = (msg.episode / totalEpisodes) * 100;
                        document.getElementById('evalProgressBar').style.width = progress + '%';
                        document.getElementById('evalProgressText').textContent = `${msg.episode} / ${totalEpisodes} episodes`;
                        document.getElementById('evalStatus').textContent = `Episode ${msg.episode} done - Score: ${msg.score}, Steps: ${msg.steps}`;
                    }
                    else if (msg.type === 'complete') {
                        trainedEvalSource.close();
                        trainedEvalSource = null;
                        btn.disabled = false;
                        trainBtn.disabled = false;
                        testBtn.disabled = false;

                        // Update final stats
                        document.getElementById('statAvgReward').textContent = msg.avg_reward;
                        document.getElementById('statAvgSteps').textContent = msg.avg_steps;
                        document.getElementById('statAvgScore').textContent = msg.avg_score;
                        document.getElementById('statMaxScore').textContent = msg.max_score;

                        // Draw chart
                        const chart = document.getElementById('episodeChart');
                        const maxReward = Math.max(...msg.episodes.map(e => Math.abs(e.reward)), 1);
                        chart.innerHTML = msg.episodes.map(e => {
                            const height = Math.max(5, (Math.abs(e.reward) / maxReward) * 120);
                            const color = e.reward >= 0 ? '#0f6' : '#f55';
                            return `<div class="episode-bar" style="height: ${height}px; background: ${color};" title="Ep ${e.episode}: ${e.reward}"></div>`;
                        }).join('');

                        document.getElementById('evalStatus').textContent = 'Trained agent evaluation complete!';
                        results.classList.remove('hidden');
                    }
                    else if (msg.type === 'error') {
                        trainedEvalSource.close();
                        trainedEvalSource = null;
                        btn.disabled = false;
                        trainBtn.disabled = false;
                        testBtn.disabled = false;
                        document.getElementById('evalStatus').textContent = 'Error: ' + msg.error;
                        alert(msg.error);
                    }
                };

                trainedEvalSource.onerror = function() {
                    trainedEvalSource.close();
                    trainedEvalSource = null;
                    btn.disabled = false;
                    trainBtn.disabled = false;
                    testBtn.disabled = false;
                    document.getElementById('evalStatus').textContent = 'Connection lost';
                };

            } catch (e) {
                alert(e.message);
                btn.disabled = false;
                trainBtn.disabled = false;
                testBtn.disabled = false;
            }
        }
    </script>
</body>
</html>'''


def main():
    """Run the web server."""
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), UniWrapAPI)

    print(f"""
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   🎮 UniWrap Web App                                  ║
║                                                       ║
║   Running at: http://localhost:{port:<5}                 ║
║                                                       ║
║   Open this URL in your browser to:                   ║
║   • Generate RL environments from GitHub repos        ║
║   • Evaluate environments with random agents          ║
║   • View performance metrics                          ║
║                                                       ║
║   Press Ctrl+C to stop                                ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
""")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
