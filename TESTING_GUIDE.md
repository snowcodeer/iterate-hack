# Environment Testing Guide

## Overview

UniWrap includes a built-in agent testing system that runs agents in generated environments and produces detailed evaluation reports.

## Quick Start

```bash
# Test all environments for a repository
python test_environments.py bread2d --episodes 10
```

## Command Line Options

```bash
python test_environments.py <repo_name> [options]

Required:
  repo_name              Repository name (subdirectory in environments/)

Optional:
  --episodes, -e N       Number of episodes to run per environment (default: 10)
```

## Output Structure

Results are saved to `environments/<repo_name>/outputs/<env_name>/`:

```
environments/
└── bread2d/
    ├── outputs/
    │   ├── env_1/
    │   │   ├── eval_TIMESTAMP_summary.json
    │   │   ├── eval_TIMESTAMP_detailed.json
    │   │   └── eval_TIMESTAMP_results.csv
    │   ├── env_2/
    │   │   └── ...
    │   └── env_3/
    │       └── ...
    ├── env_1.py
    ├── env_2.py
    └── env_3.py
```

## Output Files

### 1. Summary JSON (`eval_*_summary.json`)

Aggregated statistics across all episodes:

```json
{
  "env_name": "toast_quality_optimization_v1",
  "num_episodes": 10,
  "timestamp": "2025-11-29T18:44:10",
  "statistics": {
    "total_reward": {
      "mean": 15.3,
      "std": 2.1,
      "min": 10.0,
      "max": 20.5
    },
    "steps": {
      "mean": 50.2,
      "std": 5.3,
      "min": 42,
      "max": 58
    },
    "duration": {
      "mean": 0.082,
      "total": 0.82
    },
    "termination": {
      "terminated": 7,
      "truncated": 3
    }
  }
}
```

### 2. Detailed JSON (`eval_*_detailed.json`)

Per-episode statistics:

```json
{
  "env_name": "toast_quality_optimization_v1",
  "timestamp": "2025-11-29T18:44:10",
  "episodes": [
    {
      "steps": 45,
      "total_reward": 18.5,
      "mean_reward": 0.41,
      "max_reward": 1.2,
      "min_reward": -0.1,
      "terminated": true,
      "truncated": false,
      "duration": 0.075
    }
  ]
}
```

### 3. CSV Results (`eval_*_results.csv`)

Tabular format for quick analysis:

```csv
episode,steps,total_reward,mean_reward,terminated,truncated
0,45,18.5000,0.4111,True,False
1,52,16.2000,0.3115,True,False
2,48,19.1000,0.3979,True,False
```

## Console Output

During execution, you'll see real-time progress:

```
======================================================================
Testing 3 environments from bread2d
======================================================================

======================================================================
Evaluating: toast_quality_optimization_v1
======================================================================

Episode 1/10...
  Steps: 45, Total Reward: 18.50, Terminated: True

Episode 2/10...
  Steps: 52, Total Reward: 16.20, Terminated: True

...

✓ Results saved to: environments/bread2d/outputs/toast_quality_optimization_v1
  - Summary: eval_20251129_184410_summary.json
  - Detailed: eval_20251129_184410_detailed.json
  - CSV: eval_20251129_184410_results.csv

======================================================================
EVALUATION SUMMARY
======================================================================
Environment: toast_quality_optimization_v1
Episodes: 10

Total Reward:
  Mean: 17.2000 ± 2.1000
  Range: [10.0000, 20.5000]

Steps per Episode:
  Mean: 50.20 ± 5.30
  Range: [42, 58]

Termination:
  Terminated: 7/10
  Truncated: 3/10
======================================================================
```

## Using Custom Agents

To test with a custom agent instead of RandomAgent:

```python
import sys
sys.path.insert(0, '.')

from uniwrap.test_agent import EnvironmentEvaluator
from environments.bread2d import toast_quality_optimization_v1

class MyAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def select_action(self, observation):
        # Your agent logic here
        return self.action_space.sample()

# Create evaluator
env_class = toast_quality_optimization_v1
env_path = "environments/bread2d/toast_quality_optimization_v1.py"
evaluator = EnvironmentEvaluator(env_class, env_path, num_episodes=10)

# Run with custom agent
agent = MyAgent(env_class().action_space)
results = evaluator.run_evaluation(agent)
```

## Analyzing Results

### Using Python

```python
import json
import pandas as pd

# Load summary
with open('environments/bread2d/outputs/env_1/eval_*_summary.json') as f:
    summary = json.load(f)
    print(f"Mean reward: {summary['statistics']['total_reward']['mean']}")

# Load CSV
df = pd.read_csv('environments/bread2d/outputs/env_1/eval_*_results.csv')
print(df.describe())
```

### Using Command Line

```bash
# View summary stats
cat environments/bread2d/outputs/*/eval_*_summary.json | jq '.statistics.total_reward'

# View CSV data
column -t -s, environments/bread2d/outputs/*/eval_*_results.csv | less
```

## Metrics Explained

- **Total Reward**: Cumulative reward over the episode
- **Mean Reward**: Average reward per step
- **Steps**: Number of steps taken in the episode
- **Terminated**: Episode ended due to success/failure condition
- **Truncated**: Episode ended due to max steps limit
- **Duration**: Wall-clock time for the episode

## Troubleshooting

### "Cannot import environments.repo_name"

Make sure the environment was generated:
```bash
python -m uniwrap <repo_path> --output environments
```

### "No outputs directory"

Outputs are created automatically during testing. Check permissions on the environments directory.

### Zero rewards in all episodes

This is expected for environments with placeholder reward functions (TODO comments in generated code). Implement actual reward logic in the environment file to see meaningful results.

