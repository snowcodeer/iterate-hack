"""
Environment Supervisor - Evaluates environment quality and suggests improvements.

This module provides an AI-powered supervisor that:
1. Analyzes training results to detect issues
2. Reviews environment code for common problems
3. Suggests specific improvements
4. Can auto-improve with human approval
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from uniwrap.llm_client import ClaudeClient


EVALUATION_PROMPT = """You are an expert RL environment designer reviewing a generated environment.

## Environment Code
```python
{env_code}
```

## Training Results
- Episodes completed: {episodes}
- Average reward: {avg_reward:.4f}
- Max reward: {max_reward:.4f}
- Min reward: {min_reward:.4f}
- Reward std: {reward_std:.4f}
- Non-zero rewards: {non_zero_rewards} / {total_episodes}
- First 10 episodes avg: {first_10_avg:.4f}
- Last 10 episodes avg: {last_10_avg:.4f}
- Average episode length: {avg_length:.1f} steps

## Your Task
Analyze this environment and training results. Identify issues and suggest improvements.

Consider these common problems:

### REWARD ISSUES:
1. FLAT REWARDS - Agent gets same reward regardless of action (no learning signal)
2. SPARSE REWARDS - Only rewards on game over (too infrequent to learn from)
3. REWARD SCALE - Rewards too large/small causing training instability
4. NO INTERMEDIATE FEEDBACK - No shaping rewards to guide learning
5. GAME STATE EXTRACTION - Not properly reading game state for rewards

### GAME OVER DETECTION ISSUES (CRITICAL FOR WEB GAMES):
6. PREMATURE TERMINATION - Game ends too early due to false positive detection
   - Look for: identical_frame_count threshold < 30 (should be 30+, not 8)
   - Look for: no grace period at start (should skip first 30 steps)
   - Look for: unreliable color detection (red pixels, etc.)
7. NO GRACE PERIOD - Game over checked during startup/loading
   - Fix: Add "if self.steps_taken < 30: return False" at start of _is_game_over()
8. AGGRESSIVE FRAME DETECTION - Too few identical frames triggers game over
   - 8 frames is TOO FEW - causes false positives
   - Should require 30+ identical frames (~1 second)

### OTHER ISSUES:
9. OBSERVATION ISSUES - Observations not capturing relevant game info
10. ACTION MAPPING - Actions not properly mapped to game controls

## IMPORTANT: Episode Length Analysis
- If avg episode length is very SHORT (< 50 steps), suspect PREMATURE GAME OVER
- Short episodes + low rewards often means game over detection is broken
- Check _is_game_over() for the issues listed above

Output your analysis as JSON:
```json
{{
    "quality_score": <1-10 score>,
    "is_learning": <true/false - is there evidence the agent is learning?>,
    "issues": [
        {{
            "type": "<issue type>",
            "severity": "<critical/major/minor>",
            "description": "<what's wrong>",
            "location": "<which function/line>",
            "suggestion": "<how to fix>"
        }}
    ],
    "reward_analysis": {{
        "has_learning_signal": <true/false>,
        "is_reward_shaped": <true/false>,
        "reward_range_ok": <true/false>,
        "issues": ["<list of reward-specific issues>"]
    }},
    "recommended_changes": [
        {{
            "priority": <1-3, 1=highest>,
            "change": "<description of change>",
            "code_hint": "<pseudo-code or specific suggestion>"
        }}
    ],
    "summary": "<2-3 sentence summary of environment quality and main issues>"
}}
```

Be specific and actionable in your suggestions. Focus on the reward function first - without good rewards, the agent cannot learn."""


IMPROVEMENT_PROMPT = """You are an expert RL environment designer. Improve this environment based on the analysis.

## Current Environment Code
```python
{env_code}
```

## Analysis Results
{analysis}

## User Feedback
{user_feedback}

## Your Task
Generate an IMPROVED version of this environment that fixes the identified issues.

### REWARD FUNCTION FIXES:
1. Fix the reward function to provide meaningful learning signal
2. Add intermediate/shaping rewards
3. Properly extract game state for rewards
4. Ensure actions have different expected outcomes
5. Scale rewards to roughly -1.0 to +1.0 range

### GAME OVER DETECTION FIXES (CRITICAL):
If the analysis mentions short episodes or premature termination, fix _is_game_over():

1. ADD GRACE PERIOD - Skip checks during startup:
   ```python
   def _is_game_over(self, observation):
       # Don't check during startup/loading
       if self.steps_taken < 30:
           return False
       # ... rest of detection
   ```

2. INCREASE FRAME THRESHOLD - Require 30+ identical frames, NOT 8:
   ```python
   # BAD (causes false positives):
   if self.identical_frame_count >= 8:
       return True

   # GOOD (conservative):
   if self.identical_frame_count >= 30:
       return True
   ```

3. REMOVE UNRELIABLE COLOR DETECTION:
   - Remove any "red_pixels > X" checks
   - Remove color-based game over detection
   - These cause false positives from UI elements

4. PREFER TEXT DETECTION:
   ```python
   try:
       game_over = self.page.evaluate('''() => {{
           const text = document.body.innerText.toLowerCase();
           return text.includes('game over') || text.includes('try again');
       }}''')
       if game_over:
           return True
   except:
       pass
   ```

### IMPORTANT RULES:
- Keep the same class name and overall structure
- Don't change the observation space or action space unless necessary
- Add comments explaining reward logic and game over detection
- Make rewards scale between -1.0 and +1.0 approximately

## OUTPUT FORMAT
Output your response in TWO sections:

1. First, a brief CHANGELOG section listing what you changed and why (3-10 bullet points)
2. Then the complete improved Python code

Format:
```
=== CHANGELOG ===
- [REWARD] Added intermediate reward for X because Y
- [GAME_OVER] Increased frame threshold from 8 to 30 to prevent false positives
- [GAME_OVER] Added grace period to skip checks during first 30 steps
- [FIX] Removed unreliable red pixel detection
...

=== CODE ===
<complete python code here>
```

Be specific about what you changed and why. The user wants to understand your improvements."""


def analyze_environment(
    env_code: str,
    training_results: Dict[str, Any],
    model: str = "claude-sonnet-4-5-20250929",
    api_key: str = None
) -> Dict[str, Any]:
    """
    Analyze an environment's quality based on its code and training results.

    Args:
        env_code: The environment source code
        training_results: Results from training (episode_rewards, etc.)
        model: Claude model to use
        api_key: Optional API key

    Returns:
        Analysis dict with quality score, issues, and recommendations
    """
    import numpy as np

    episode_rewards = training_results.get('episode_rewards', [])
    episode_lengths = training_results.get('episode_lengths', [])

    if not episode_rewards:
        return {
            'quality_score': 0,
            'is_learning': False,
            'issues': [{'type': 'NO_DATA', 'severity': 'critical',
                       'description': 'No training data available'}],
            'summary': 'Cannot analyze - no training results provided.'
        }

    # Calculate statistics
    avg_reward = float(np.mean(episode_rewards))
    max_reward = float(np.max(episode_rewards))
    min_reward = float(np.min(episode_rewards))
    reward_std = float(np.std(episode_rewards))
    non_zero = sum(1 for r in episode_rewards if r != 0)
    first_10 = float(np.mean(episode_rewards[:10])) if len(episode_rewards) >= 10 else avg_reward
    last_10 = float(np.mean(episode_rewards[-10:])) if len(episode_rewards) >= 10 else avg_reward
    avg_length = float(np.mean(episode_lengths)) if episode_lengths else 0

    prompt = EVALUATION_PROMPT.format(
        env_code=env_code,
        episodes=len(episode_rewards),
        avg_reward=avg_reward,
        max_reward=max_reward,
        min_reward=min_reward,
        reward_std=reward_std,
        non_zero_rewards=non_zero,
        total_episodes=len(episode_rewards),
        first_10_avg=first_10,
        last_10_avg=last_10,
        avg_length=avg_length
    )

    client = ClaudeClient(api_key=api_key)
    response = client.call_claude(prompt, model=model)

    # Parse JSON from response
    try:
        # Find JSON in response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            analysis = json.loads(json_str)
        else:
            analysis = {'error': 'Could not parse JSON response', 'raw': response}
    except json.JSONDecodeError as e:
        analysis = {'error': f'JSON parse error: {e}', 'raw': response}

    # Add computed stats
    analysis['training_stats'] = {
        'episodes': len(episode_rewards),
        'avg_reward': avg_reward,
        'max_reward': max_reward,
        'min_reward': min_reward,
        'reward_std': reward_std,
        'non_zero_pct': (non_zero / len(episode_rewards)) * 100 if episode_rewards else 0,
        'improvement': last_10 - first_10 if len(episode_rewards) >= 20 else 0
    }

    return analysis


def improve_environment(
    env_code: str,
    analysis: Dict[str, Any],
    user_feedback: str = "",
    model: str = "claude-sonnet-4-5-20250929",
    api_key: str = None
) -> Dict[str, str]:
    """
    Generate improved environment code based on analysis.

    Args:
        env_code: Current environment source code
        analysis: Analysis results from analyze_environment()
        user_feedback: Additional user feedback/requirements
        model: Claude model to use
        api_key: Optional API key

    Returns:
        Dict with 'code' (improved code) and 'changelog' (summary of changes)
    """
    # Format analysis for prompt
    analysis_text = json.dumps(analysis, indent=2, default=str)

    prompt = IMPROVEMENT_PROMPT.format(
        env_code=env_code,
        analysis=analysis_text,
        user_feedback=user_feedback or "No additional feedback."
    )

    client = ClaudeClient(api_key=api_key)
    response = client.call_claude(prompt, model=model)

    # Parse changelog and code sections
    changelog = ""
    code = response.strip()

    # Try to extract changelog
    if "=== CHANGELOG ===" in response:
        parts = response.split("=== CODE ===")
        if len(parts) >= 2:
            changelog_part = parts[0]
            code_part = parts[1]

            # Extract changelog content
            if "=== CHANGELOG ===" in changelog_part:
                changelog = changelog_part.split("=== CHANGELOG ===")[1].strip()

            code = code_part.strip()

    # Clean up code
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]

    code = code.strip()

    # Add header comment with changelog if we have one
    if changelog:
        header = f'''"""
Environment improved by AI Supervisor
=====================================

CHANGES MADE:
{changelog}
"""

'''
        # Only add header if code doesn't already have a docstring
        if not code.startswith('"""') and not code.startswith("'''"):
            code = header + code

    return {
        'code': code,
        'changelog': changelog
    }


def quick_reward_check(episode_rewards: List[float], episode_lengths: List[int] = None) -> Dict[str, Any]:
    """
    Quick heuristic check of reward quality without LLM.

    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths (optional)

    Returns:
        Dict with quick analysis results
    """
    import numpy as np

    if not episode_rewards:
        return {'status': 'error', 'message': 'No rewards to analyze'}

    rewards = np.array(episode_rewards)

    issues = []

    # Check for very short episodes (suggests premature game over)
    if episode_lengths:
        avg_length = np.mean(episode_lengths)
        if avg_length < 30:
            issues.append({
                'type': 'PREMATURE_GAME_OVER',
                'severity': 'critical',
                'message': f'Episodes are very short (avg {avg_length:.0f} steps) - likely premature game over detection. Check _is_game_over() for: identical_frame_count < 30, missing grace period, or unreliable color detection.'
            })
        elif avg_length < 50:
            issues.append({
                'type': 'SHORT_EPISODES',
                'severity': 'major',
                'message': f'Episodes are short (avg {avg_length:.0f} steps) - may indicate game over detection issues or very difficult game.'
            })

    # Check for flat rewards (no variance)
    if np.std(rewards) < 0.001:
        issues.append({
            'type': 'FLAT_REWARDS',
            'severity': 'critical',
            'message': 'All episodes have nearly identical rewards - agent has no learning signal'
        })

    # Check for all-zero rewards
    if np.all(rewards == 0):
        issues.append({
            'type': 'ZERO_REWARDS',
            'severity': 'critical',
            'message': 'All rewards are zero - reward function not working'
        })

    # Check for no improvement
    if len(rewards) >= 20:
        first_half = np.mean(rewards[:len(rewards)//2])
        second_half = np.mean(rewards[len(rewards)//2:])
        if second_half <= first_half:
            issues.append({
                'type': 'NO_IMPROVEMENT',
                'severity': 'major',
                'message': 'No improvement in rewards over training - may need better reward shaping'
            })

    # Check for extreme values
    if np.max(np.abs(rewards)) > 100:
        issues.append({
            'type': 'REWARD_SCALE',
            'severity': 'minor',
            'message': 'Reward values seem large - consider scaling to [-1, 1] range'
        })

    # Check for sparse rewards
    non_zero_pct = (np.sum(rewards != 0) / len(rewards)) * 100
    if non_zero_pct < 10:
        issues.append({
            'type': 'SPARSE_REWARDS',
            'severity': 'major',
            'message': f'Only {non_zero_pct:.1f}% of episodes have non-zero rewards - too sparse'
        })

    return {
        'status': 'ok' if not issues else 'issues_found',
        'issues': issues,
        'stats': {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
            'non_zero_pct': non_zero_pct
        }
    }


def print_analysis_report(analysis: Dict[str, Any]):
    """Print a formatted analysis report."""
    print("\n" + "=" * 60)
    print("ENVIRONMENT ANALYSIS REPORT")
    print("=" * 60)

    if 'error' in analysis:
        print(f"\n❌ Error: {analysis['error']}")
        return

    # Quality score
    score = analysis.get('quality_score', 'N/A')
    is_learning = analysis.get('is_learning', False)
    print(f"\n📊 Quality Score: {score}/10")
    print(f"📈 Learning Detected: {'Yes ✓' if is_learning else 'No ✗'}")

    # Summary
    if 'summary' in analysis:
        print(f"\n📝 Summary: {analysis['summary']}")

    # Issues
    issues = analysis.get('issues', [])
    if issues:
        print(f"\n⚠️  Issues Found ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            severity = issue.get('severity', 'unknown')
            severity_icon = {'critical': '🔴', 'major': '🟠', 'minor': '🟡'}.get(severity, '⚪')
            print(f"  {i}. {severity_icon} [{severity.upper()}] {issue.get('type', 'Unknown')}")
            print(f"     {issue.get('description', '')}")
            if issue.get('suggestion'):
                print(f"     💡 {issue['suggestion']}")

    # Recommendations
    recommendations = analysis.get('recommended_changes', [])
    if recommendations:
        print(f"\n🔧 Recommended Changes:")
        for rec in sorted(recommendations, key=lambda x: x.get('priority', 99)):
            priority = rec.get('priority', '?')
            print(f"  [P{priority}] {rec.get('change', '')}")
            if rec.get('code_hint'):
                print(f"       Code: {rec['code_hint'][:100]}...")

    # Training stats
    stats = analysis.get('training_stats', {})
    if stats:
        print(f"\n📉 Training Statistics:")
        print(f"  Episodes: {stats.get('episodes', 'N/A')}")
        print(f"  Avg Reward: {stats.get('avg_reward', 0):.4f}")
        print(f"  Reward Range: [{stats.get('min_reward', 0):.2f}, {stats.get('max_reward', 0):.2f}]")
        print(f"  Non-zero: {stats.get('non_zero_pct', 0):.1f}%")
        print(f"  Improvement: {stats.get('improvement', 0):+.4f}")

    print("\n" + "=" * 60)
