# UniWrap

**UniWrap: Automatic RL Environment Generator**

UniWrap automatically generates Gymnasium RL environments from web games and pygame applications. It uses Claude AI to analyze games and create complete, trainable environments with proper reward functions, observations, and game-over detection.

## Features

- **Web Game Support**: Automatically wraps browser-based games using Playwright
- **Pygame Support**: Wraps pygame games by analyzing source code
- **RL Training**: Train PPO agents on generated environments using Stable Baselines3
- **AI Supervisor**: Analyzes environment quality and suggests improvements
- **Training Graphs**: Automatically generates visualizations of training progress
- **Environment Versioning**: Track and manage multiple versions of environments

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
- Create required directories (`models/`, `training_graphs/`, etc.)
- Install Playwright browsers if needed for web game support

## CLI Options

```
============================================================
  🎮 UniWrap - RL Environment Generator
============================================================

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

## Workflow

### 1. Generate Environment

Provide a URL to a web game or pygame GitHub repository:

```
> 1
Enter game URL: https://example.com/game
```

UniWrap will:
- Detect the game type (web or pygame)
- Analyze the game mechanics
- Generate a complete Gymnasium environment

### 2. Train Agent

Select an environment and training duration:

```
> 4
Select environment: 1
Training steps: 10,000
```

Training generates:
- PPO model saved to `models/`
- Training graphs saved to `training_graphs/`

### 3. AI Supervisor (Recommended)

If training isn't working well, use the AI supervisor:

```
> 7
Select environment to analyze: 1
```

The supervisor will:
- Run quick training to collect data
- Analyze reward function and game-over detection
- Identify issues (flat rewards, premature termination, etc.)
- Generate improved environment code
- Save changes with a CHANGELOG.txt explaining fixes

## Project Structure

```
uniwrap/
├── cli_app.py              # Main interactive CLI
├── uniwrap/
│   ├── env_designer.py     # Environment generation prompts
│   ├── env_supervisor.py   # AI supervisor for improvements
│   ├── rl_agent.py         # Training and evaluation
│   ├── llm_client.py       # Claude API client
│   └── ...
├── environments/           # Generated environments
│   ├── snake_pygame/
│   │   ├── __init__.py
│   │   └── snakeeaterenv.py
│   ├── functorial_com_v2/
│   │   ├── __init__.py
│   │   ├── stardodgeenv.py
│   │   └── CHANGELOG.txt   # AI supervisor changes
│   └── ...
├── models/                 # Trained models
│   ├── snake_pygame_ppo.zip
│   └── snake_pygame_ppo.json  # Model metadata
└── training_graphs/        # Training visualizations
    └── snake_pygame_training_*.png
```

## Environment Generation

### Web Games

UniWrap uses Playwright to:
- Open the game in a browser
- Capture screenshots as observations
- Send keyboard inputs (space, arrows)
- Detect game-over conditions

### Pygame Games

UniWrap analyzes pygame source code and generates:
- Embedded game logic in the environment
- Proper observation extraction
- Action-to-control mapping

## AI Supervisor

The supervisor identifies and fixes common issues:

**Reward Issues:**
- Flat rewards (no learning signal)
- Sparse rewards (only on game over)
- Missing intermediate feedback

**Game-Over Detection Issues:**
- Premature termination (episodes too short)
- Missing grace period at start
- Unreliable color detection
- Too-aggressive frame comparison

When improvements are made, a `CHANGELOG.txt` is saved documenting:
```
CHANGES MADE:
- [GAME_OVER] Added 30-step grace period
- [GAME_OVER] Increased frame threshold from 8 to 30
- [REWARD] Added survival reward (+0.01 per step)
- [REWARD] Added score-based reward
```

## How UniWrap Compares

| Feature | UniWrap | OpenAI Retro | Unity ML-Agents |
|---------|---------|--------------|-----------------|
| **Game Source** | Any web game URL or pygame code | ROM files (Atari, NES, SNES, Genesis) | Unity game projects |
| **Setup Required** | Just provide a URL | Requires ROMs + manual integration files | Requires Unity SDK integration |
| **Environment Generation** | AI-generated from scratch | Pre-built, requires memory mapping | Manual C# scripting |
| **Auto-Improvement** | AI supervisor fixes issues automatically | None | None |
| **Game Types** | Modern web games, pygame | ~1000 classic retro games | Unity-built games only |
| **Learning Curve** | Minimal - paste URL, train | Moderate - ROM setup, integration | High - Unity + ML-Agents SDK |
| **Legal Concerns** | Uses publicly accessible web games | ROM licensing issues | None (your own games) |

### UniWrap's Unique Value

1. **Zero-Config Environment Generation**: Paste a URL, get a trainable RL environment. No manual reward engineering, observation extraction, or game-over detection required.

2. **AI Supervisor Agent**: Automatically analyzes training runs and fixes common issues:
   - Premature episode termination
   - Flat or sparse rewards
   - Incorrect action mappings
   - Each improvement is documented in `CHANGELOG.txt`

3. **Iterative Refinement**: Environments evolve (v1 → v2 → v3) with each supervisor pass, progressively improving until the agent learns effectively.

4. **Works with Any Web Game**: No ROMs, no SDKs, no game engine integration. If it runs in a browser, UniWrap can wrap it.

## Requirements

- Python 3.8+
- Anthropic API key
- Chrome/Chromium (for web games)

## Dependencies

- `gymnasium` - RL environment interface
- `stable-baselines3` - PPO training
- `playwright` - Browser automation
- `pygame` - Game rendering
- `anthropic` - Claude AI API
- `matplotlib` - Training graphs
- `numpy`, `pillow` - Data processing
