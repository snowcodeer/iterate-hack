# UniWrap

**UniWrap: Automatic RL Environment Generator**

UniWrap reads any codebase and automatically generates a family of RL environment specifications (actions, observations, rewards, termination conditions) as JSON, enabling RL methods to be tested on software without manual environment design.

## Overview

Today, RL environments are handcrafted. UniWrap is an agent that reads any codebase (or web game URL) and automatically generates a fully-integrated RL environment wrapper with real game integration. No simulated mode - it connects directly to the actual game using standard industry patterns (browser automation for web games, direct integration for native games).

**Key Features:**
- **Automatic Game Detection**: Detects web games, desktop games, APIs
- **Real Integration**: Uses Playwright for web games, direct imports for Python games
- **Single Wrapper**: Generates one optimized environment (not multiple variants)
- **Evaluation Agent**: Automatically assesses wrapper quality
- **URL Support**: Works with GitHub repos or direct web game URLs

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Anthropic API key (choose one method):

   **Option A: Using .env file (recommended)**
   ```bash
   cp .env.example .env
   # Edit .env and add your API key
   ```

   **Option B: Using environment variable**
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

   Get your API key from: https://console.anthropic.com/

## Usage

Run UniWrap on any codebase:

```bash
python -m uniwrap <repo_path> [options]
```

### Options

- `repo_path`: Path to codebase, GitHub repo URL, or web game URL (required)
- `--output` / `-o`: Output directory for environment files (default: `environments/`)
- `--format`: Output format - `code` (Python files), `json` (specs), or `both` (default: `code`)
- `--evaluate`: Run evaluation agent to assess wrapper quality
- `--model`: Claude model to use (default: claude-sonnet-4-5-20250929)
- `--api-key`: Anthropic API key (default: uses `ANTHROPIC_API_KEY` env var)

### Examples

**Local Repository:**
```bash
python -m uniwrap /path/to/codebase --evaluate
```

**GitHub Repository:**
```bash
python -m uniwrap https://github.com/user/repo --evaluate
```

**Web Game URL:**
```bash
python -m uniwrap http://localhost:3000 --evaluate
```

This will create:
```
environments/
└── repo_name/
    ├── __init__.py
    └── generated_env.py  # Single optimized wrapper
```

## Output

UniWrap generates executable Python Gymnasium environment files organized by repository. Each repository gets its own subdirectory with:

- **Python environment files**: One file per generated environment variant
- **Package structure**: Proper `__init__.py` files for easy imports
- **Documentation**: Inline comments and docstrings

### Output Structure

```
environments/
├── __init__.py                    # Top-level package
└── <repo_name>/                   # Repository-specific subdirectory
    ├── __init__.py                # Package with all environments
    ├── env_variant_1.py           # First environment variant
    ├── env_variant_2.py           # Second environment variant
    └── env_variant_3.py           # Third environment variant
```

### Using Generated Environments

```python
from environments.<repo_name> import <env_name>

env = <env_name>()
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Example

After running `python -m uniwrap bread-2d --variants 3`:

```python
from environments.bread2d import toast_optimization_v1

env = toast_optimization_v1()
obs, info = env.reset()
# ... use the environment
```

## Project Structure

```
uniwrap/
├── __init__.py
├── __main__.py
├── llm_client.py          # Anthropic Claude API wrapper
├── repo_summarizer.py     # Analyzes and summarizes codebase
├── env_designer.py        # Orchestrates LLM calls, generates multiple specs
├── cli.py                 # Command-line interface
├── spec_validator.py      # Validates generated JSON specs
└── utils.py               # Helper functions (file reading, JSON handling)
```

## How It Works

1. **Game Type Detection**: Automatically detects if it's a web game, desktop game, or API
2. **Repository Analysis**: Analyzes codebase structure, README, and game mechanics
3. **Environment Design**: Claude generates a single optimized RL environment specification
4. **Real Integration**: Code generator creates Gymnasium wrapper with actual game controller integration:
   - **Web Games**: Uses Playwright for browser automation
   - **Python Games**: Direct import and method calls
   - **APIs**: HTTP client integration
5. **Evaluation**: Optional evaluation agent assesses wrapper quality and identifies issues
6. **Output**: Single optimized environment file ready to use

## Testing

### Test Generated Environments

Test all environments for a repository with a random agent:

```bash
python test_environments.py <repo_name> --episodes 10 --visualize
```

This will:
- Run a random agent in each environment
- Generate evaluation metrics (rewards, steps, termination stats)
- Save results to `environments/<repo_name>/outputs/<env_name>/`
- Generate visualizations (with `--visualize` flag)

### Visualization

Generate visual analysis and comparison plots:

```bash
# Auto-generate during testing
python test_environments.py bread2d --episodes 10 --visualize

# Or generate separately
python visualize_results.py bread2d
```

**Generated visualizations:**
- Individual environment analysis (4-panel plots)
  - Reward over episodes
  - Steps over episodes
  - Reward distribution
  - Statistics summary
- Cross-environment comparison
  - Mean reward comparison
  - Mean steps comparison
  - Reward distributions (violin plots)
  - Termination type breakdown

Saved to: `environments/<repo_name>/outputs/visualizations/`

### Output Files

For each environment, three files are generated:
- `eval_<timestamp>_summary.json` - Aggregated statistics
- `eval_<timestamp>_detailed.json` - Per-episode data
- `eval_<timestamp>_results.csv` - CSV format for analysis

Plus visualizations (with `--visualize` flag):
- `<env_name>_analysis.png` - Individual environment plots
- `<repo_name>_comparison.png` - Cross-environment comparison

### Example

```bash
# Test bread2d environments with 5 episodes each and generate visualizations
python test_environments.py bread2d --episodes 5 --visualize

# Results saved to:
# environments/bread2d/outputs/
# ├── toast_quality_optimization_v1/
# │   ├── eval_20251129_184410_summary.json
# │   ├── eval_20251129_184410_detailed.json
# │   └── eval_20251129_184410_results.csv
# └── visualizations/
#     ├── toast_quality_optimization_v1_analysis.png
#     ├── bread2d_toast_optimization_v1_analysis.png
#     ├── bread2d_toasting_optimization_analysis.png
#     └── bread2d_comparison.png
```

### Evaluation Metrics

Generated reports include:
- **Total Reward**: Mean, std, min, max across episodes
- **Steps per Episode**: Distribution of episode lengths
- **Termination**: Count of terminated vs truncated episodes
- **Duration**: Time taken per episode
- **Per-Episode Data**: Detailed statistics for each episode

## Quick Component Tests

Test individual components without API calls:

```bash
python test_uniwrap.py
```

This tests:
- Utility functions (file reading, directory tree, JSON parsing)
- Repository summarization
- Spec validation
- Sample repository handling

### Full Pipeline Test

Test the complete pipeline with a sample repository:

```bash
# Make sure ANTHROPIC_API_KEY is set
export ANTHROPIC_API_KEY="your-api-key-here"

# Test with the included test repository
python -m uniwrap test_repo --variants 1 --output test_output.json

# Or test with any codebase
python -m uniwrap /path/to/your/codebase --variants 3
```

### Test with Your Own Codebase

1. **Set your API key:**
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

2. **Run on any repository:**
   ```bash
   python -m uniwrap /path/to/repo --variants 3 --output env_specs.json
   ```

3. **Check the output:**
   ```bash
   cat env_specs.json | python -m json.tool
   ```

### Testing Individual Components

You can also test components in Python:

```python
from uniwrap.repo_summarizer import summarize_repo
from uniwrap.spec_validator import validate_spec

# Test summarization
summary = summarize_repo("/path/to/repo")
print(summary)

# Test validation
spec = {...}  # Your spec dict
is_valid, errors = validate_spec(spec)
```

## Requirements

- Python 3.8+
- Anthropic API key
- `anthropic` package
