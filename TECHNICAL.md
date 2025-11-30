# UniWrap Technical Details

This document covers the technical architecture and implementation details of UniWrap.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (cli_app.py)                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────┐  ┌────────────────┐    │
│  │   Environment    │  │  RL Agent    │  │ AI Supervisor  │    │
│  │    Generator     │  │  (SB3 PPO)   │  │ (Claude API)   │    │
│  │   (Claude API)   │  │              │  │                │    │
│  └──────────────────┘  └──────────────┘  └────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Generated Gymnasium Environments             │  │
│  │  ┌─────────────────┐  ┌─────────────────────────────┐    │  │
│  │  │  Pygame Envs    │  │     Web Game Envs           │    │  │
│  │  │  (embedded)     │  │     (Playwright)            │    │  │
│  │  └─────────────────┘  └─────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Environment Generator (`uniwrap/env_designer.py`)

Generates Gymnasium environments from game URLs using Claude AI.

**Supported Game Types:**
- **Web Games**: Browser-based games wrapped via Playwright
- **Pygame Games**: Python pygame games with embedded game logic

**Generation Process:**
1. Fetch game page info (title, canvas elements, instructions)
2. Send to Claude with specialized prompts
3. Claude generates complete `gym.Env` subclass
4. **Validate and auto-fix** generated code
5. Save to `environments/<game_name>/`

**Key Prompts:**
- `WEB_GAME_PROMPT`: Guidance for web game wrapping (Playwright, canvas detection, game over detection)
- `PYGAME_CODE_PROMPT`: Guidance for pygame embedding (font handling, render modes)

**Code Validation Pipeline:**
```python
# 1. Generate code from Claude
code = client.call_claude(prompt)

# 2. Auto-fix common issues (e.g., missing self. prefix)
code = fix_common_code_issues(code)

# 3. Validate syntax using AST
is_valid, error = validate_python_code(code)
if not is_valid:
    print(f"⚠️ Syntax error: {error}")
```

**Auto-Fix Function:**
```python
def fix_common_code_issues(code: str) -> str:
    # Find all self.VARNAME definitions
    defined_vars = re.findall(r'self\.([A-Z_]+)\s*=', code)

    # Fix missing self. prefix for uppercase instance variables
    # e.g., TILE -> self.TILE when self.TILE is defined
    for var in defined_vars:
        pattern = rf'(?<!self\.)({var})(?![a-zA-Z_])'
        # Replace with self.VAR if not already prefixed
```

### 2. RL Agent (`uniwrap/rl_agent.py`)

Trains PPO agents using Stable Baselines3.

**Auto Policy Selection:**
```python
if isinstance(obs_space, spaces.Dict):
    policy = "MultiInputPolicy"      # Dict observations
elif isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 3:
    policy = "CnnPolicy"             # Image observations (H, W, C)
else:
    policy = "MlpPolicy"             # Flat vector observations
```

**Training Configuration:**
- Algorithm: PPO (Proximal Policy Optimization)
- Learning rate: 3e-4
- Batch size: 64
- N steps: 2048
- Entropy coefficient: 0.01

**Outputs:**
- Model: `models/<env_name>_ppo.zip` (or `_v2.zip`, `_v3.zip`, etc.)
- Metadata: `models/<env_name>_ppo.json`
- Live graph: `training_graphs/<env_name>_live_*.png`
- Detailed graph: `training_graphs/<env_name>_training_*.png`

**Live Training Visualization:**
During training, a matplotlib window opens showing real-time progress:
- Episode rewards with moving average (10 episodes)
- Episode lengths (survival time)
- Auto-scaling axes as training progresses
- Graph is saved when training completes

```python
# Set up live plotting
plt.ion()  # Enable interactive mode
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Update after each episode
def progress(data):
    if data['type'] == 'episode_complete':
        episode_rewards_live.append(data['reward'])
        reward_line.set_data(episodes, episode_rewards_live)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
```

### 3. Model Version Management

Models are automatically versioned to preserve training history.

**Version Naming:**
```
models/
├── snake_pygame_ppo.zip      # v1 (original)
├── snake_pygame_ppo.json     # v1 metadata
├── snake_pygame_ppo_v2.zip   # v2 (continued or fresh)
├── snake_pygame_ppo_v2.json  # v2 metadata
├── snake_pygame_ppo_v3.zip   # v3
└── snake_pygame_ppo_v3.json  # v3 metadata
```

**Metadata Contents:**
```json
{
  "env_name": "snake_pygame",
  "env_file_hash": "a1b2c3d4e5f6",
  "observation_shape": [20, 10],
  "action_space_n": 4,
  "total_timesteps": 25000,
  "continued_from": "models/snake_pygame_ppo.zip",
  "trained_at": "2025-11-30T14:30:00",
  "model_file": "snake_pygame_ppo_v2.zip"
}
```

**Training Modes:**
1. **Start Fresh**: Creates new version with random weights
   ```python
   save_path = get_versioned_model_path(env_name)  # Returns next available version
   ```

2. **Continue Training**: Loads latest weights, saves as new version
   ```python
   existing_model = get_latest_model_path(env_name)
   agent.load(existing_model)
   save_path = get_versioned_model_path(env_name)  # New version
   ```

**Version Selection (Testing):**
```python
versions = list_model_versions(env_name)
# Returns: [
#   {"version": 1, "path": Path(...), "size_kb": 245, "metadata": {...}},
#   {"version": 2, "path": Path(...), "size_kb": 312, "metadata": {...}},
# ]
```

### 4. AI Supervisor (`uniwrap/env_supervisor.py`)

Analyzes and improves environments automatically.

**Analysis Pipeline:**
1. Run quick training (1000 steps, ~10-30 episodes)
2. Collect metrics (rewards, episode lengths, variance)
3. Fetch game info from URL to understand mechanics
4. Send to Claude for analysis
5. Generate improved environment code

**Common Issues Detected:**
- Premature game over (short episodes)
- Flat rewards (no learning signal)
- Incorrect action mapping
- Missing grace period at startup

**Improvement Categories:**
- `[GAME_OVER]`: Fix termination detection
- `[REWARD]`: Fix reward shaping
- `[ACTION]`: Fix action space/mapping
- `[OBSERVATION]`: Fix observation extraction
- `[PERFORMANCE]`: Reduce observation size, optimize

### 5. LLM Client (`uniwrap/llm_client.py`)

Claude API wrapper with retry logic.

```python
client = ClaudeClient(api_key=os.getenv("ANTHROPIC_API_KEY"))
response = client.call_claude(
    prompt="...",
    model="claude-sonnet-4-5-20250929",
    max_tokens=8192  # Enough for complete environment code
)
```

## Environment Structure

### Generated Environment Files

```
environments/<game_name>/
├── __init__.py          # Exports the environment class
├── <game>env.py         # Main environment implementation
└── CHANGELOG.txt        # AI supervisor changes (if improved)
```

### Gymnasium Interface

All environments implement the standard Gymnasium interface:

```python
class GameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, headless=True):
        self.observation_space = spaces.Box(...)  # or Dict, Discrete
        self.action_space = spaces.Discrete(n)

    def reset(self, seed=None, options=None):
        return observation, info

    def step(self, action):
        return observation, reward, terminated, truncated, info

    def render(self):
        return frame  # or None for human mode

    def close(self):
        # Cleanup resources
```

## Web Game Environments

### Browser Automation (Playwright)

```python
def _start_browser(self):
    self.playwright = sync_playwright().start()
    self.browser = self.playwright.chromium.launch(headless=self.headless)
    self.page = self.browser.new_context().new_page()
    self.page.goto(self.game_url, wait_until="domcontentloaded")
```

### Game Element Detection

Priority order for finding the game canvas:
1. Specific selectors: `canvas#game`, `#game-container canvas`
2. Largest canvas on page
3. Fallback bounding box

### Game Over Detection

Multiple methods with conservative thresholds:

1. **Grace Period**: Skip first 30 steps (loading/startup)
2. **JavaScript State**: `window.gameOver`, `window.game.isOver`
3. **Text Detection**: "game over", "try again" in DOM
4. **Frame Freeze**: 30+ identical frames (not 8!)

```python
def _is_game_over(self):
    if self.steps_taken < 30:  # Grace period
        return False

    # JavaScript detection (most reliable)
    game_state = self._extract_game_state()
    if game_state.get("gameOver"):
        return True

    # Frame freeze detection (conservative)
    if self.identical_frame_count >= 30:
        return True

    return False
```

### Reward Design

For survival/dodge games:
```python
# Time-based survival reward that increases with difficulty
survival_reward = 0.01 + (self.time_alive / 5000)
reward += survival_reward

# Score bonus (if available)
if score_delta > 0:
    reward += score_delta * 0.1

# Death penalty (normalized)
if terminated:
    reward = -1.0
```

## Pygame Environments

### How Pygame Environments Work

Pygame environments embed game logic directly rather than controlling an external game:

1. **Game State**: All game variables (positions, scores, etc.) are stored as instance variables
2. **Step Function**: Executes one game tick, applying the action and updating state
3. **Observations**: Can be either raw game state arrays or rendered screenshots
4. **Rendering**: Uses pygame surfaces for visual display in "human" mode

### Render Modes

- `"human"`: Opens pygame window, calls `pygame.display.flip()`
- `"rgb_array"`: Returns numpy array via `pygame.surfarray.pixels3d()`
- `None`: Headless mode for training (no rendering)

## Environment Versioning

Environments are versioned when improved by the AI supervisor:

```
environments/
├── snake_pygame/       # Original (v1)
├── snake_pygame_v2/    # First improvement
├── snake_pygame_v3/    # Second improvement
└── ...
```

Each version has a `CHANGELOG.txt` documenting changes:
```
snake_pygame_v2 - Improved by AI Supervisor
==================================================
Based on: snake_pygame

CHANGES MADE:
- [GAME_OVER] Added 30-step grace period
- [REWARD] Added survival time bonus
- [OBSERVATION] Reduced to grayscale for faster training
```

## Complete Training Pipeline

```
1. Generate Environment
   └─> environments/<name>/<name>env.py

2. Train Agent (with live graph)
   └─> models/<name>_ppo.zip (v1)
   └─> models/<name>_ppo.json (metadata)
   └─> training_graphs/<name>_live_*.png
   └─> training_graphs/<name>_training_*.png

3. Continue Training (optional)
   └─> Loads v1 weights
   └─> models/<name>_ppo_v2.zip (v2)
   └─> Metadata tracks: continued_from: v1

4. AI Supervisor (if needed)
   └─> Analyze training results
   └─> Generate improved environment
   └─> environments/<name>_v2/

5. Test Specific Version
   └─> Select model v1, v2, or v3
   └─> Run episodes with selected weights
```

## Configuration

### Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...  # Required for AI features
```

### Dependencies

See `requirements.txt`:
- `gymnasium>=0.29.0` - RL environment interface
- `stable-baselines3>=2.0.0` - PPO training
- `playwright>=1.40.0` - Browser automation
- `pygame>=2.5.0` - Game rendering
- `anthropic>=0.34.0` - Claude API
- `matplotlib>=3.5.0` - Live training graphs

## Performance Considerations

### Observation Size

Smaller observations = faster training:
- Web games: 200x125 grayscale (25K values) vs 800x500 RGB (1.2M values)
- Pygame: Use game state arrays when possible, not full screenshots

### Headless Mode

Always train with `headless=True`:
- No window rendering overhead
- 5-10x faster training
- Use visual mode only for debugging

### Episode Length

Short episodes indicate problems:
- < 30 steps: Premature game over detection
- 30-50 steps: Check grace period
- 100+ steps: Healthy training

## Troubleshooting

### Model-Environment Mismatch
Each model stores `observation_shape` in metadata. If you modify an environment's observation space, you need to train a new model - the old one won't work.

### Flat Rewards
The AI supervisor detects this by checking reward variance. If all episodes have identical rewards, it indicates:
- Game never actually started
- Reward function returns constant value
- Score extraction not working

### Matplotlib Backend Issues
The live graph tries multiple backends in order: TkAgg, Qt5Agg, MacOSX, GTK3Agg. If none work, training continues without live visualization.
