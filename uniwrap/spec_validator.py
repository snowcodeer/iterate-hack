"""JSON spec validation for generated environment specifications."""

from typing import Dict, List, Tuple


def validate_spec(spec: Dict) -> Tuple[bool, List[str]]:
    """Validate an environment spec dictionary.
    
    Args:
        spec: Environment specification dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required top-level fields
    required_fields = ['name', 'actions', 'observations', 'reward', 'termination']
    for field in required_fields:
        if field not in spec:
            errors.append(f"Missing required field: {field}")
    
    # Validate actions
    if 'actions' in spec:
        actions = spec['actions']
        if not isinstance(actions, dict):
            errors.append("'actions' must be a dictionary")
        else:
            if 'type' not in actions:
                errors.append("'actions.type' is required")
            elif actions['type'] not in ['discrete', 'continuous', 'multi_discrete']:
                errors.append(f"Invalid actions.type: {actions['type']}")
    
    # Validate observations
    if 'observations' in spec:
        observations = spec['observations']
        if not isinstance(observations, dict):
            errors.append("'observations' must be a dictionary")
        else:
            if 'type' not in observations:
                errors.append("'observations.type' is required")
            elif observations['type'] not in ['box', 'discrete', 'dict']:
                errors.append(f"Invalid observations.type: {observations['type']}")
    
    # Validate reward
    if 'reward' in spec:
        reward = spec['reward']
        if not isinstance(reward, dict):
            errors.append("'reward' must be a dictionary")
        else:
            if 'function' not in reward:
                errors.append("'reward.function' is required")
    
    # Validate termination
    if 'termination' in spec:
        termination = spec['termination']
        if not isinstance(termination, dict):
            errors.append("'termination' must be a dictionary")
        else:
            if 'conditions' not in termination:
                errors.append("'termination.conditions' is required")
            elif not isinstance(termination['conditions'], list):
                # Auto-fix: convert to list if it's a string or other type
                if isinstance(termination['conditions'], str):
                    termination['conditions'] = [termination['conditions']]
                elif termination['conditions'] is None:
                    termination['conditions'] = []
                else:
                    # Try to convert other types
                    try:
                        termination['conditions'] = list(termination['conditions'])
                    except (TypeError, ValueError):
                        errors.append("'termination.conditions' must be a list, string, or convertible to list")
    
    # Validate metadata (optional but should be dict if present)
    if 'metadata' in spec and not isinstance(spec['metadata'], dict):
        errors.append("'metadata' must be a dictionary if present")
    
    is_valid = len(errors) == 0
    return is_valid, errors

