"""Evaluation agent to assess wrapper/environment quality."""

import ast
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional
import sys


def evaluate_environment(
    env_path: Path,
    env_name: str,
    game_info: Dict,
    max_episodes: int = 3
) -> Dict:
    """Evaluate a generated environment wrapper.
    
    Args:
        env_path: Path to the environment Python file
        env_name: Name of the environment class
        game_info: Game type detection info
        max_episodes: Number of test episodes to run
        
    Returns:
        Dict with evaluation results:
        - quality_score: float (0-1)
        - issues: List of issue descriptions
        - test_results: Dict with test outcomes
    """
    issues = []
    test_results = {}
    quality_score = 1.0
    
    # 1. Static code analysis
    try:
        with open(env_path, 'r') as f:
            code = f.read()
        
        tree = ast.parse(code)
        
        # Check for required methods
        required_methods = ['reset', 'step', 'render', 'close']
        class_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == env_name:
                class_def = node
                break
        
        if class_def:
            method_names = {node.name for node in class_def.body if isinstance(node, ast.FunctionDef)}
            missing_methods = set(required_methods) - method_names
            if missing_methods:
                issues.append(f"Missing required methods: {', '.join(missing_methods)}")
                quality_score -= 0.2 * len(missing_methods)
        
        # Check for game controller integration
        if 'game_controller' not in code:
            issues.append("No game controller integration found")
            quality_score -= 0.3
        elif 'TODO' in code and 'game_controller' in code:
            issues.append("Game controller integration has TODO comments")
            quality_score -= 0.1
        
        # Check for simulated/placeholder code
        placeholder_patterns = [
            'np.random.uniform',
            'TODO: Replace',
            'placeholder',
            'simulated'
        ]
        placeholder_count = sum(1 for pattern in placeholder_patterns if pattern in code)
        if placeholder_count > 0:
            issues.append(f"Found {placeholder_count} placeholder/simulated code sections")
            quality_score -= 0.1 * min(placeholder_count, 3)
        
    except Exception as e:
        issues.append(f"Code analysis failed: {e}")
        quality_score -= 0.2
    
    # 2. Try to import and instantiate
    try:
        spec = importlib.util.spec_from_file_location(env_name, env_path)
        if spec is None or spec.loader is None:
            issues.append("Failed to create module spec")
            quality_score -= 0.3
        else:
            module = importlib.util.module_from_spec(spec)
            sys.modules[env_name] = module
            spec.loader.exec_module(module)
            
            env_class = getattr(module, env_name, None)
            if env_class is None:
                issues.append(f"Environment class '{env_name}' not found in module")
                quality_score -= 0.3
            else:
                # Try to instantiate (may fail if game not available)
                try:
                    env = env_class()
                    test_results['instantiation'] = 'success'
                    
                    # Try reset
                    try:
                        obs, info = env.reset()
                        test_results['reset'] = 'success'
                        
                        # Try step
                        try:
                            action = env.action_space.sample()
                            obs, reward, terminated, truncated, info = env.step(action)
                            test_results['step'] = 'success'
                            test_results['reward_type'] = type(reward).__name__
                            test_results['observation_type'] = type(obs).__name__
                            
                            # Check if reward is meaningful (not always zero)
                            if reward == 0:
                                issues.append("Reward is zero - may indicate integration issue")
                                quality_score -= 0.1
                            
                        except Exception as e:
                            test_results['step'] = f'failed: {e}'
                            issues.append(f"Step() failed: {e}")
                            quality_score -= 0.2
                            
                    except Exception as e:
                        test_results['reset'] = f'failed: {e}'
                        issues.append(f"Reset() failed: {e}")
                        quality_score -= 0.2
                    
                    # Try close
                    try:
                        env.close()
                        test_results['close'] = 'success'
                    except Exception as e:
                        test_results['close'] = f'failed: {e}'
                        quality_score -= 0.1
                        
                except Exception as e:
                    test_results['instantiation'] = f'failed: {e}'
                    issues.append(f"Failed to instantiate environment: {e}")
                    quality_score -= 0.3
                    
    except Exception as e:
        issues.append(f"Import failed: {e}")
        quality_score -= 0.2
    
    # 3. Check game type compatibility
    if game_info.get('type') == 'web' and 'WebGameController' not in code:
        issues.append("Web game detected but WebGameController not used")
        quality_score -= 0.2
    
    # Normalize quality score
    quality_score = max(0.0, min(1.0, quality_score))
    
    return {
        'quality_score': quality_score,
        'issues': issues,
        'test_results': test_results,
        'game_info': game_info
    }

