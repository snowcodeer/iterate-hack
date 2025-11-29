# UniWrap Visualization Guide

## Overview

UniWrap includes powerful visualization tools that automatically generate plots and comparison charts from evaluation results.

## Quick Start

```bash
# Generate visualizations from existing results
python visualize_results.py bread2d

# Or auto-generate during testing
python test_environments.py bread2d --episodes 10 --visualize
```

## Visualization Types

### 1. Individual Environment Analysis

Four-panel plot for each environment showing:

**Top Left: Reward over Episodes**
- Line plot of total reward per episode
- Mean reward line (red dashed)
- Helps identify learning trends or consistency

**Top Right: Steps over Episodes**
- Line plot of steps taken per episode
- Mean steps line (red dashed)
- Shows episode length variation

**Bottom Left: Reward Distribution**
- Histogram of total rewards
- Mean and median indicators
- Visualizes reward spread

**Bottom Right: Statistics Summary**
- Text panel with key metrics
- Mean, std, min, max for rewards and steps
- Termination statistics

**File:** `<env_name>_analysis.png`

### 2. Cross-Environment Comparison

Four-panel comparison across all environments:

**Top Left: Mean Reward Comparison**
- Bar chart with error bars (std)
- Compare average performance

**Top Right: Mean Steps Comparison**
- Bar chart with error bars (std)
- Compare episode lengths

**Bottom Left: Reward Distributions**
- Violin plots for each environment
- Shows full distribution shape
- Medians and means visible

**Bottom Right: Termination Type**
- Stacked bar chart
- Shows % terminated vs truncated
- Indicates completion patterns

**File:** `<repo_name>_comparison.png`

## Usage Examples

### Generate all visualizations

```bash
python visualize_results.py bread2d
```

Output:
```
======================================================================
Generating visualizations for bread2d
======================================================================

Creating individual environment plots...
  ✓ Saved: bread2d_toasting_optimization_analysis.png
  ✓ Saved: bread2d_toast_optimization_v1_analysis.png
  ✓ Saved: toast_quality_optimization_v1_analysis.png

Creating comparison plot...
  ✓ Saved: bread2d_comparison.png

======================================================================
✓ All visualizations saved to:
  /path/to/environments/bread2d/outputs/visualizations
======================================================================
```

### Auto-generate during testing

```bash
python test_environments.py bread2d --episodes 10 --visualize
```

This runs tests AND generates visualizations automatically.

## Output Location

All visualizations are saved to:
```
environments/<repo_name>/outputs/visualizations/
├── <env_name_1>_analysis.png
├── <env_name_2>_analysis.png
├── <env_name_3>_analysis.png
└── <repo_name>_comparison.png
```

## Interpreting the Plots

### Reward Plots

**Increasing trend**: Agent may be learning (or environment has increasing difficulty)
**Flat line**: Consistent performance or no learning
**High variance**: Stochastic environment or unstable agent
**Zero rewards**: Placeholder reward function (implement actual rewards)

### Steps Plots

**Constant steps**: Hitting max_steps limit (truncated episodes)
**Decreasing steps**: Agent solving faster (if terminated early)
**Increasing steps**: Agent taking longer to complete/fail

### Distribution Plots

**Normal distribution**: Well-behaved stochastic rewards
**Bimodal**: Two distinct outcomes (success/failure modes)
**Uniform**: Highly random or placeholder rewards
**Single spike**: Deterministic outcomes

### Termination Comparison

**High terminated %**: Episodes ending due to success/failure conditions
**High truncated %**: Episodes hitting max_steps limit
**All truncated**: May need to adjust max_steps or termination logic

## Command Line Options

### visualize_results.py

```bash
python visualize_results.py <repo_name>

Required:
  repo_name              Repository name (subdirectory in environments/)
```

### test_environments.py

```bash
python test_environments.py <repo_name> [options]

Required:
  repo_name              Repository name

Optional:
  --episodes, -e N       Number of episodes (default: 10)
  --visualize, -v        Generate visualizations after testing
```

## Requirements

Visualization requires matplotlib:

```bash
pip install matplotlib
# or
pip install -r requirements.txt
```

If matplotlib is not installed, you'll see:
```
⚠ Visualization skipped: matplotlib not installed
  Install with: pip install matplotlib
```

## Programmatic Usage

```python
from uniwrap.visualize import EvaluationVisualizer

# Create visualizer
viz = EvaluationVisualizer("bread2d")

# Generate all plots
viz.generate_dashboard()

# Or create specific plots
viz.plot_single_environment("toast_quality_optimization_v1")
viz.compare_environments()
```

## Customization

To customize plots, edit `uniwrap/visualize.py`:

- Change colors: Modify color parameters in plot calls
- Adjust layout: Change `figsize` in `plt.subplots()`
- Add metrics: Extend the `EvaluationVisualizer` class
- Change DPI: Modify `dpi` parameter in `plt.savefig()`

## Troubleshooting

### "No results found"

Make sure you've run evaluations first:
```bash
python test_environments.py bread2d --episodes 5
```

### "matplotlib not installed"

Install the visualization dependencies:
```bash
pip install matplotlib numpy
```

### Plots look crowded

For many environments, the comparison plot may be crowded. The code automatically:
- Rotates x-axis labels 45°
- Truncates long names to 20 chars
- Adjusts layout with `tight_layout()`

### Want different metrics

Edit the summary statistics in `EvaluationVisualizer._aggregate_results()` to add custom metrics.

## Tips

1. **Run multiple evaluations**: Each run creates timestamped files, allowing you to track improvements over time

2. **Compare across repositories**: Create visualizations for multiple repos to compare different codebases

3. **Use CSV for custom analysis**: The CSV files can be loaded into pandas, Excel, or R for custom analysis

4. **Combine with training**: Run evaluations periodically during training and visualize progress

5. **Share results**: PNG files are easy to share in reports, presentations, or documentation

