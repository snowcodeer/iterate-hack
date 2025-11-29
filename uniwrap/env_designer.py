"""Environment spec generation using Claude LLM."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from uniwrap.llm_client import ClaudeClient
from uniwrap.utils import clean_json_response


PROMPT_TEMPLATE = """You are an expert in reinforcement learning environment design.

Your task: Given the following codebase summary, propose ONE RL environment
specification as valid JSON only. The goal is to define an environment WITH
A step() FUNCTION (Gym-style) that can be immediately implemented.

Define:
- name: A unique identifier for this environment variant (use snake_case)
- version: Spec version (e.g., "1.0.0")
- description: Brief description of what this environment tests
- actions: What can an agent do? (type: discrete/continuous/multi_discrete, space definition, description)
- observations: What should be returned each step? (type: box/discrete/dict, shape, dtype, description)
- reward: How to measure progress? IMPORTANT: Include concrete numeric components
  - function: Description of reward function
  - range: [min, max] values
  - description: Detailed explanation
  - components: Dictionary of numeric components with CONCRETE VALUES:
    * For weighted components: "quality_weight": 0.3, "speed_weight": 0.5
    * For penalties: "error_penalty": -10, "timeout_penalty": -5
    * For bonuses: "perfect_bonus": 20
    * Example reward.components: {{"quality_weight": 0.4, "speed_weight": 0.3, "error_penalty": -10, "perfect_bonus": 25}}
- termination: When the episode ends? (conditions: success/failure/timeout, max_steps, description)
- metadata: Additional info (episode_length, reset_conditions, additional_info)

Focus on a specific aspect of the codebase. Consider different approaches:
- Reward functions (correctness, efficiency, coverage, performance)
- Action spaces (discrete choices vs continuous parameters)
- Observation strategies (code state vs execution traces vs metrics)

CRITICAL: The reward.components field MUST contain numeric values that can be directly used in code.
Example: {{"correctness_weight": 0.5, "efficiency_weight": 0.3, "timeout_penalty": -5}}

Output ONLY a valid JSON object (not an array). No explanations, no markdown, just the JSON object.

Codebase summary:

{repo_summary}
"""


WEB_GAME_PROMPT = """You are an expert in reinforcement learning and browser automation.

Your task: Create a Gymnasium environment that can play a web-based game using Playwright for browser automation.

The game URL is: {game_url}

Here is information about the web page:
{page_info}

You must create a COMPLETE, WORKING Gymnasium environment that:
1. Uses Playwright to open the game in a browser
2. Sends keyboard inputs to control the game (space, arrow keys, etc.)
3. Captures screenshots and/or reads game state from the DOM
4. Extracts observations (game state, score, positions if visible)
5. Calculates rewards based on survival time, score, etc.
6. Detects game over conditions

IMPORTANT REQUIREMENTS:
- Use playwright.sync_api for browser automation
- The environment should work headlessly for training
- Support render_mode="human" to show the browser
- Use keyboard.press() for discrete actions like jump/duck
- For observations, you can use:
  - Screenshots converted to numpy arrays (for CNN-based agents)
  - DOM element positions/values if accessible
  - Simple features like "is obstacle near", "current speed", etc.
- For the Chrome Dino game specifically:
  - Actions: 0=do nothing, 1=jump (space), 2=duck (down arrow)
  - The game runs in a canvas element
  - Score is usually visible in the DOM
  - Game over is detected when the game stops/shows restart

Output a complete Python file with:
1. All necessary imports (gymnasium, playwright, numpy, PIL if needed)
2. A gym.Env subclass with __init__, reset, step, render, close methods
3. Proper action space (Discrete) and observation space (Box for screenshot or features)
4. Browser lifecycle management

The class name should be based on the game (e.g., DinoGameEnv, ChromeDinoEnv).

Output ONLY the Python code. No explanations, no markdown code blocks, just the raw Python code starting with imports."""


PYGAME_CODE_PROMPT = """You are an expert in reinforcement learning and pygame game development.

Your task: Analyze the following pygame game code and generate a COMPLETE, WORKING Gymnasium environment that wraps this game for RL training.

You must:
1. Understand how the game works (state variables, game loop, controls, win/lose conditions)
2. Extract the game logic into a controllable format
3. Create a gym.Env class that can run the game headlessly and step through it

IMPORTANT REQUIREMENTS:
- The environment must be SELF-CONTAINED - it should embed/adapt the game logic, not import the original file
- Extract key game variables (positions, scores, game over conditions) for observations
- Map discrete actions to game controls (e.g., key presses)
- Design a reward function based on game objectives
- Handle game reset properly
- Support both headless (training) and visual (human) render modes

Output a complete Python file with:
1. All necessary imports
2. A gym.Env subclass with __init__, reset, step, render, close methods
3. Embedded game logic (adapted from the original code)
4. Proper observation and action spaces
5. Reward calculation based on game events

The class name should be the game name in snake_case (e.g., snake_game -> SnakeGameEnv).

Here is the game code to wrap:

{game_code}

Additional context about the repository:
{repo_summary}

Output ONLY the Python code. No explanations, no markdown code blocks, just the raw Python code starting with imports."""


def _generate_single_spec(
    repo_summary: str,
    agent_id: int,
    model: str = "claude-sonnet-4-5-20250929",
    api_key: str = None,
    prompt_template: str = None
) -> Dict:
    """Generate a single environment specification (called in parallel).

    Args:
        repo_summary: Summarized codebase information
        agent_id: Identifier for this agent (for logging)
        model: Claude model to use
        api_key: Optional API key (uses env var if not provided)
        prompt_template: Optional custom prompt template

    Returns:
        Environment specification dictionary

    Raises:
        ValueError: If JSON parsing fails
        Exception: If Claude API call fails
    """
    # Format the prompt
    template = prompt_template or PROMPT_TEMPLATE
    prompt = template.format(repo_summary=repo_summary)
    
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


def generate_env_spec(
    repo_summary: str,
    game_info: Dict = None,
    model: str = "claude-sonnet-4-5-20250929",
    api_key: str = None
) -> Dict:
    """Generate a single environment specification from a codebase summary.

    Args:
        repo_summary: Summarized codebase information
        game_info: Game type detection info (optional)
        model: Claude model to use
        api_key: Optional API key (uses env var if not provided)

    Returns:
        Environment specification dictionary

    Raises:
        ValueError: If JSON parsing fails
        Exception: If Claude API call fails
    """
    # Choose prompt template based on game type
    framework = game_info.get('framework', 'unknown') if game_info else 'unknown'
    game_subtype = game_info.get('config', {}).get('game_type', 'generic') if game_info else 'generic'

    prompt_template = PROMPT_TEMPLATE
    # Enhance prompt with game type info if available
    enhanced_summary = repo_summary
    if game_info and game_info.get('type') != 'unknown':
        game_type_section = f"""

=== Game Type Information ===
Type: {game_info.get('type', 'unknown')}
Framework: {game_info.get('framework', 'unknown')}
Integration Method: {game_info.get('integration_method', 'browser')}
Entry Point: {game_info.get('entry_point', 'N/A')}

IMPORTANT: This is a {game_info.get('type', 'web')} game. Design the environment to work with real game integration.
The environment will use {game_info.get('integration_method', 'browser')} to interact with the actual game.
Ensure actions, observations, and rewards map to real game mechanics.

"""
        enhanced_summary = game_type_section + repo_summary

    # Generate single spec
    try:
        spec = _generate_single_spec(enhanced_summary, 1, model, api_key, prompt_template=prompt_template)
        # Add controller info from game_info
        if game_info and framework == 'pygame':
            spec['controller'] = {
                'type': 'pygame',
                'game_type': game_subtype
            }
        return spec
    except Exception as e:
        raise Exception(f"Failed to generate environment spec: {e}")


def generate_pygame_env_code(
    game_code: str,
    repo_summary: str,
    model: str = "claude-sonnet-4-5-20250929",
    api_key: str = None
) -> str:
    """Generate complete pygame environment code using Claude.

    This function has Claude analyze the game code and generate a complete,
    self-contained Gymnasium environment that embeds the game logic.

    Args:
        game_code: The pygame game source code
        repo_summary: Summary of the repository for context
        model: Claude model to use
        api_key: Optional API key

    Returns:
        Complete Python code for the environment
    """
    prompt = PYGAME_CODE_PROMPT.format(
        game_code=game_code,
        repo_summary=repo_summary
    )

    client = ClaudeClient(api_key=api_key)
    response_text = client.call_claude(prompt, model=model)

    # Clean up the response - remove markdown code blocks if present
    code = response_text.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]

    return code.strip()


def generate_web_game_env_code(
    game_url: str,
    page_info: str,
    model: str = "claude-sonnet-4-5-20250929",
    api_key: str = None
) -> str:
    """Generate environment code for a web-based game.

    This function has Claude analyze a web game and generate a Gymnasium
    environment that uses Playwright to control it.

    Args:
        game_url: URL of the web game
        page_info: Information about the page (from web fetch)
        model: Claude model to use
        api_key: Optional API key

    Returns:
        Complete Python code for the environment
    """
    prompt = WEB_GAME_PROMPT.format(
        game_url=game_url,
        page_info=page_info
    )

    client = ClaudeClient(api_key=api_key)
    response_text = client.call_claude(prompt, model=model)

    # Clean up the response - remove markdown code blocks if present
    code = response_text.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]

    return code.strip()

