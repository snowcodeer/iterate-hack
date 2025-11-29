"""Environment spec generation using Claude LLM."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from uniwrap.llm_client import ClaudeClient
from uniwrap.utils import clean_json_response


PROMPT_TEMPLATE = """You are an expert in reinforcement learning environment design.

Your task: Given the following codebase summary, propose ONE RL environment
specification as valid JSON only. The goal is to define an environment WITH
A step() FUNCTION (Gym-style).

Define:
- name: A unique identifier for this environment variant
- version: Spec version (e.g., "1.0.0")
- description: Brief description of what this environment tests
- actions: What can an agent do? (type: discrete/continuous/multi_discrete, space definition, description)
- observations: What should be returned each step? (type: box/discrete/dict, shape, dtype, description)
- reward: How to measure progress? (function description, range, description)
- termination: When the episode ends? (conditions: success/failure/timeout, max_steps, description)
- metadata: Additional info (episode_length, reset_conditions, additional_info)

Focus on a specific aspect of the codebase. Consider different approaches:
- Reward functions (correctness, efficiency, coverage, performance)
- Action spaces (discrete choices vs continuous parameters)
- Observation strategies (code state vs execution traces vs metrics)

Output ONLY a valid JSON object (not an array). No explanations, no markdown, just the JSON object.

Codebase summary:

{repo_summary}
"""


def _generate_single_spec(
    repo_summary: str,
    agent_id: int,
    model: str = "claude-sonnet-4-5-20250929",
    api_key: str = None
) -> Dict:
    """Generate a single environment specification (called in parallel).
    
    Args:
        repo_summary: Summarized codebase information
        agent_id: Identifier for this agent (for logging)
        model: Claude model to use
        api_key: Optional API key (uses env var if not provided)
        
    Returns:
        Environment specification dictionary
        
    Raises:
        ValueError: If JSON parsing fails
        Exception: If Claude API call fails
    """
    # Format the prompt
    prompt = PROMPT_TEMPLATE.format(repo_summary=repo_summary)
    
    # Call Claude
    client = ClaudeClient(api_key=api_key)
    response_text = client.call_claude(prompt, model=model)
    
    # Clean and parse JSON response
    json_text = clean_json_response(response_text)
    
    try:
        spec = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Agent {agent_id}: Failed to parse JSON response from Claude: {e}\n"
            f"Response text: {response_text[:500]}"
        )
    
    # Ensure it's a dict (not a list)
    if isinstance(spec, list):
        if len(spec) > 0:
            spec = spec[0]
        else:
            raise ValueError(f"Agent {agent_id}: Received empty list from Claude")
    
    if not isinstance(spec, dict):
        raise ValueError(f"Agent {agent_id}: Expected dict, got {type(spec)}")
    
    return spec


def generate_env_specs(
    repo_summary: str, 
    num_variants: int = 3,
    model: str = "claude-sonnet-4-5-20250929",
    api_key: str = None
) -> List[Dict]:
    """Generate environment specifications from a codebase summary using parallel agents.
    
    Args:
        repo_summary: Summarized codebase information
        num_variants: Number of parallel agents to run (each generates one variant)
        model: Claude model to use
        api_key: Optional API key (uses env var if not provided)
        
    Returns:
        List of environment specification dictionaries
        
    Raises:
        ValueError: If JSON parsing fails
        Exception: If Claude API call fails
    """
    specs = []
    errors = []
    
    # Run agents in parallel
    with ThreadPoolExecutor(max_workers=num_variants) as executor:
        # Submit all tasks
        future_to_agent = {
            executor.submit(_generate_single_spec, repo_summary, i+1, model, api_key): i+1
            for i in range(num_variants)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_agent):
            agent_id = future_to_agent[future]
            try:
                spec = future.result()
                specs.append(spec)
            except Exception as e:
                errors.append(f"Agent {agent_id}: {str(e)}")
    
    # If we have errors but some succeeded, warn but continue
    if errors and specs:
        import warnings
        warnings.warn(f"Some agents failed: {', '.join(errors)}")
    
    # If all failed, raise an error
    if not specs:
        error_msg = "All agents failed to generate specs"
        if errors:
            error_msg += f": {', '.join(errors)}"
        raise Exception(error_msg)
    
    return specs

