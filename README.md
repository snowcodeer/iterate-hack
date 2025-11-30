# UniWrap

**UniWrap: Automatic RL Environment Generator**

UniWrap automatically generates Gymnasium RL environments from web games and pygame applications. Just paste a URL - UniWrap uses Claude AI to analyze the game and create a complete, trainable environment with proper reward functions, observations, and game-over detection.

## Key Features

### Zero-Config Environment Generation
- **Web Games**: Paste any browser game URL → get a trainable environment
- **Pygame Games**: Paste a GitHub repo URL → game logic is embedded directly
- **Auto-Detection**: Automatically identifies game type, controls, and mechanics

### Live Training Visualization
- **Real-time Graphs**: Watch episode rewards and survival time update live during training
- **Moving Averages**: 10-episode rolling average shows learning progress
- **Auto-saved**: Graphs saved to `training_graphs/` when training completes

### Model Version Management
- **Version History**: Each training run creates a new model version (v1, v2, v3...)
- **Continue Training**: Load previous weights and train more steps
- **Version Selection**: Choose which model version to test or evaluate
- **Metadata Tracking**: Each model stores training date, timesteps, observation shape

### AI Supervisor (Automatic Debugging)
- **Analyzes Training**: Runs quick training to collect metrics
- **Identifies Issues**: Detects flat rewards, premature termination, wrong actions
- **Generates Fixes**: Creates improved environment code automatically
- **Documents Changes**: Saves CHANGELOG.txt explaining each fix

## Installation

1. Clone the repository:
```bash
git clone https://github.com/snowcodeer/iterate-hack.git
cd iterate-hack
```

2. Create virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set your Anthropic API key:
```bash
cp .env.example .env
# Edit .env and add your API key: ANTHROPIC_API_KEY=sk-ant-...
```

## Quick Start

Run the interactive CLI:
```bash
python cli_app.py
```

On first run, the CLI will automatically:
- Create required directories (`models/`, `training_graphs/`, `environments/`)
- Install Playwright browsers if needed for web game support

## CLI Menu

```
============================================================
  🎮 UniWrap - RL Environment Generator
============================================================

Existing environments:
  • snake_pygame (SnakeEnv) [trained]
  • dinosaur_game_io (DinoGameEnv)

Options:
  1. Generate environment from URL    - Create env from web game or pygame repo
  2. List environments                - Show all generated environments
  3. Test environment (random agent)  - Test with random actions
  4. Train RL agent                   - Train PPO agent on environment
  5. Test trained agent               - Run trained agent with visualization
  6. Evaluate & generate graphs       - Full evaluation with metrics
  7. Analyze & improve (AI supervisor)- AI analyzes and fixes environment issues
  8. Delete environment               - Remove environment and associated files
  v. Toggle visual mode               - Show/hide game window during training
  q. Quit
```

## Detailed Workflow

### 1. Generate Environment

Provide a URL to a web game or pygame GitHub repository:

```
> 1
Enter game URL: https://github.com/user/snake-game
Hints for the AI (optional): Arrow keys to move, eat food to grow

🔍 Cloning repository...
🔍 Detecting game type... pygame
📖 Analyzing codebase...
💻 Generating pygame environment...

✅ Environment generated: SnakeEnv
   Path: environments/snake_game/
```

### 2. Train Agent with Live Graph

Select an environment and watch training progress in real-time:

```
> 4
Select environment: 1

Existing models found:
  v1: 10000 steps @ 2025-11-30
  v2: 25000 steps @ 2025-11-30 (from v1)

Training mode:
  1. Start fresh (creates new version)
  2. Continue from latest (loads weights, saves as new version)
Select [1]: 2

   Will load: snake_pygame_ppo_v2.zip
   Will save as: new version (v3)

Training steps:
  1. 5,000 (quick test)
  2. 10,000 (normal)
  3. 25,000 (better)
  4. 50,000 (good)
  5. 100,000 (best)
Select [2]: 3

Initializing PPO agent...
   📊 Live training graph opened
Training started...
  Episode   1 | Reward:   -50.0 | Avg(10):   -50.0 | Steps:   23 | Progress:   0.5%
  Episode   2 | Reward:   -45.0 | Avg(10):   -47.5 | Steps:   31 | Progress:   1.1%
  ...
```

A matplotlib window opens showing:
- **Left plot**: Episode rewards (blue) with 10-episode moving average (red)
- **Right plot**: Episode length/survival time (green)

### 3. Test Trained Agent (with Version Selection)

Choose which model version to test:

```
> 5

Trained environments:
  1. snake_pygame (SnakeEnv) - 3 model(s), latest: 25000 steps @ 2025-11-30

Select environment: 1

Multiple models available for snake_pygame:
  1. v1: snake_pygame_ppo.zip (245KB)
     Trained: 2025-11-30 10:23 | Steps: 10000 | Obs: [20, 10]
  2. v2: snake_pygame_ppo_v2.zip (312KB)
     Trained: 2025-11-30 11:45 | Steps: 25000 | Obs: [20, 10]
  3. v3: snake_pygame_ppo_v3.zip (356KB)
     Trained: 2025-11-30 14:30 | Steps: 50000 | Obs: [20, 10]

Select model version [1-3] or Enter for latest: 3

   Using model: snake_pygame_ppo_v3.zip
Number of episodes [3]: 5
```

### 4. AI Supervisor (Recommended for Debugging)

If training isn't working well, the AI supervisor analyzes and fixes issues:

```
> 7
Select environment to analyze: 1

🏃 Running quick training (1000 steps)...
  Episode   1 | Reward:    -9.4
  Episode   2 | Reward:    -9.4
  ...

🔍 Analyzing environment with AI supervisor...

============================================================
ENVIRONMENT ANALYSIS REPORT
============================================================

📊 Quality Score: 3/10
📈 Learning Detected: No ✗

⚠️  Issues Found (4):
  1. 🔴 [CRITICAL] FLAT REWARDS - All episodes have identical -9.4 reward
  2. 🔴 [CRITICAL] PREMATURE TERMINATION - Episodes end at exactly 30 steps
  3. 🟠 [MAJOR] ACTION SPACE - Only space bar, but game needs arrow keys
  4. 🟡 [MINOR] OBSERVATION SIZE - 800x500 RGB is slow to process

🔧 Generating improved environment...

CHANGES MADE:
- [GAME_OVER] Added 60-frame threshold (was 30)
- [GAME_OVER] Added game state detection via JavaScript
- [REWARD] Added survival time reward (+0.01 per step)
- [ACTION] Expanded to arrow keys (Discrete(5))
- [PERFORMANCE] Reduced observation to 400x250

✅ Saved as: snake_pygame_v2
   Path: environments/snake_pygame_v2/
   Changelog: environments/snake_pygame_v2/CHANGELOG.txt
```

## Project Structure

```
uniwrap/
├── cli_app.py              # Main interactive CLI
├── uniwrap/
│   ├── env_designer.py     # Environment generation prompts
│   ├── env_supervisor.py   # AI supervisor for improvements
│   ├── rl_agent.py         # Training, evaluation, model versioning
│   ├── llm_client.py       # Claude API client
│   └── ...
├── environments/           # Generated environments
│   ├── snake_pygame/
│   │   ├── __init__.py
│   │   └── snakeenv.py
│   ├── snake_pygame_v2/    # Improved version
│   │   ├── __init__.py
│   │   ├── snakeenv.py
│   │   └── CHANGELOG.txt   # Documents AI supervisor fixes
│   └── ...
├── models/                 # Trained models (versioned)
│   ├── snake_pygame_ppo.zip      # v1
│   ├── snake_pygame_ppo.json     # v1 metadata
│   ├── snake_pygame_ppo_v2.zip   # v2
│   ├── snake_pygame_ppo_v2.json  # v2 metadata
│   └── ...
└── training_graphs/        # Training visualizations
    ├── snake_pygame_live_20251130_143052.png
    └── snake_pygame_training_20251130_143052.png
```

## How UniWrap Compares

| Feature | UniWrap | OpenAI Retro | Unity ML-Agents |
|---------|---------|--------------|-----------------|
| **Game Source** | Any web game URL or pygame code | ROM files (Atari, NES, SNES, Genesis) | Unity game projects |
| **Setup Required** | Just provide a URL | Requires ROMs + manual integration files | Requires Unity SDK integration |
| **Environment Generation** | AI-generated from scratch | Pre-built, requires memory mapping | Manual C# scripting |
| **Auto-Improvement** | AI supervisor fixes issues automatically | None | None |
| **Live Training Graphs** | Yes, real-time matplotlib | No | TensorBoard (separate) |
| **Model Versioning** | Automatic with metadata | Manual | Manual |
| **Game Types** | Modern web games, pygame | ~1000 classic retro games | Unity-built games only |
| **Learning Curve** | Minimal - paste URL, train | Moderate - ROM setup, integration | High - Unity + ML-Agents SDK |

### UniWrap's Unique Value

1. **Zero-Config Environment Generation**: Paste a URL, get a trainable RL environment. No manual reward engineering, observation extraction, or game-over detection required.

2. **Live Training Visualization**: Real-time graphs show learning progress as it happens - no waiting until training ends to see if it's working.

3. **AI Supervisor Agent**: Automatically analyzes training runs and fixes common issues:
   - Premature episode termination
   - Flat or sparse rewards
   - Incorrect action mappings
   - Each improvement is documented in `CHANGELOG.txt`

4. **Model Version Management**: Every training run is preserved with full metadata. Continue training from any version, compare performance across versions.

5. **Works with Any Web Game**: No ROMs, no SDKs, no game engine integration. If it runs in a browser or uses pygame, UniWrap can wrap it.

## Supported Game Types

### Web Games
UniWrap uses Playwright to:
- Open the game in a headless browser
- Capture screenshots as observations
- Send keyboard inputs (space, arrows, WASD)
- Detect game-over via JavaScript state, DOM text, or frame freeze

### Pygame Games
UniWrap analyzes pygame source code and generates:
- Embedded game logic (no external process)
- Direct game state as observations (faster than screenshots)
- Action-to-control mapping

## Requirements

- Python 3.8+
- Anthropic API key (Claude)
- Chrome/Chromium (auto-installed via Playwright for web games)

## Dependencies

Core:
- `gymnasium` - RL environment interface
- `stable-baselines3` - PPO training algorithm
- `anthropic` - Claude AI API

Game Support:
- `playwright` - Browser automation for web games
- `pygame` - Game rendering for pygame environments

Visualization:
- `matplotlib` - Live training graphs
- `numpy`, `pillow` - Data processing

## Troubleshooting

### Training reward is flat
Run the AI supervisor (option 7) - it will diagnose and fix reward issues.

### Episodes are too short
Usually means game-over detection is too aggressive. The supervisor can fix this by adding grace periods and adjusting thresholds.

### Model doesn't match environment
Each model stores the observation shape it was trained on. If you modify an environment, train a new model version.

### Pygame font errors on Python 3.14+
This is a known pygame issue. UniWrap uses `pygame._freetype` as a workaround - fonts should work automatically.
