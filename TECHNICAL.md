# UniWrap Technical Details

This document covers the technical architecture and implementation details of UniWrap.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (cli_app.py)                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Environment  │  │  RL Agent    │  │   AI Supervisor      │  │
│  │  Generator   │  │  (SB3 PPO)   │  │   (Claude API)       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
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
4. Save to `environments/<game_name>/`

**Key Prompts:**
- `WEB_GAME_PROMPT`: Guidance for web game wrapping (Playwright, canvas detection, game over detection)
- `PYGAME_CODE_PROMPT`: Guidance for pygame embedding (font handling, render modes)

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
- Model: `models/<env_name>_ppo.zip`
- Metadata: `models/<env_name>_ppo.json`
- Training graphs: `training_graphs/<env_name>_training_*.png`

### 3. AI Supervisor (`uniwrap/env_supervisor.py`)

Analyzes and improves environments automatically.

**Analysis Pipeline:**
1. Run quick training (5000 steps, ~34 episodes)
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

### 4. LLM Client (`uniwrap/llm_client.py`)

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

### Font Handling (Python 3.14+ Compatibility)

```python
# Robust font handling pattern
try:
    font = pygame.font.Font(None, 36)
    text = font.render("Score: 100", True, (255, 255, 255))
    screen.blit(text, (10, 10))
except Exception:
    try:
        font = pygame.font.SysFont('arial', 36)
        text = font.render("Score: 100", True, (255, 255, 255))
        screen.blit(text, (10, 10))
    except Exception:
        pass  # Skip text - game still works
```

### Render Modes

- `"human"`: Opens pygame window, calls `pygame.display.flip()`
- `"rgb_array"`: Returns numpy array via `pygame.surfarray.pixels3d()`
- `None`: Headless mode for training (no rendering)

## Versioning System

Environments are versioned when improved:

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

## Training Pipeline

```
1. Generate Environment
   └─> environments/<name>/<name>env.py

2. Train Agent
   └─> models/<name>_ppo.zip
   └─> training_graphs/<name>_training_*.png

3. Evaluate (optional)
   └─> Check rewards, episode lengths

4. AI Supervisor (if needed)
   └─> Analyze training results
   └─> Generate improved environment
   └─> environments/<name>_v2/

5. Repeat training on improved environment
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
- `matplotlib>=3.5.0` - Training graphs

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
