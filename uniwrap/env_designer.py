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

{extra_context}

You must create a COMPLETE, WORKING Gymnasium environment that:
1. Uses Playwright to open the game in a browser
2. Sends keyboard inputs to control the game (space, arrow keys, etc.)
3. Captures screenshots as observations (RGB or grayscale numpy arrays)
4. Calculates rewards based on survival time, score increase, etc.
5. RELIABLY detects game over conditions

CRITICAL REQUIREMENTS FOR GAME OVER DETECTION:
- Many games don't expose game state via JavaScript
- Use VISUAL DETECTION as the PRIMARY method:
  - Compare consecutive screenshots - if they're nearly identical for 5+ frames, the game is likely over
  - Look for dramatic color changes (red flash, game over screen)
  - Track if the game elements stop moving
- As BACKUP, try JavaScript/DOM detection:
  - Look for game over elements: '.game-over', '.crashed', '.restart', '[class*="game-over"]'
  - Try accessing game internals if available (Runner.instance_.crashed for Chrome Dino)
- ALWAYS implement visual detection as fallback

SCREENSHOT AND OBSERVATION:
- Use page.screenshot(clip={{...}}) to capture just the game canvas/area
- Find the canvas/game element bounding box first
- Use a reasonable resolution (300-600px width) - not too small (hard to see), not too large (slow)
- Convert to numpy array using PIL
- For training efficiency, grayscale is fine; for visualization, use RGB

BROWSER SETUP:
- Use larger viewport (1024x768 or more) so game isn't cropped
- Scroll game element into view: element.scroll_into_view_if_needed()
- Click on game area to focus before starting
- Handle cookie consent popups that may block the game

REWARD DESIGN:
- +0.1 per step survived (survival reward)
- +score_delta for score increases
- -1.0 penalty when game over detected
- Keep rewards small and balanced

Output a complete Python file with:
1. All necessary imports (gymnasium, playwright, numpy, PIL, time)
2. A gym.Env subclass with __init__, reset, step, render, close methods
3. Discrete action space (typically: 0=nothing, 1=jump/space, 2=duck/down)
4. Box observation space for screenshots
5. _start_browser() that handles consent popups
6. _get_canvas_screenshot() that clips to game area
7. _is_game_over() with BOTH visual AND JavaScript detection

The class name should be based on the game URL (e.g., DinoGameEnv for dinosaur-game.io).

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
    api_key: str = None,
    extra_context: str = ""
) -> str:
    """Generate environment code for a web-based game.

    This function has Claude analyze a web game and generate a Gymnasium
    environment that uses Playwright to control it.

    Args:
        game_url: URL of the web game
        page_info: Information about the page (from web fetch)
        model: Claude model to use
        api_key: Optional API key
        extra_context: Additional hints/feedback from user

    Returns:
        Complete Python code for the environment
    """
    # Build extra context section
    context_section = ""
    if extra_context:
        context_section = f"""
=== USER PROVIDED INFORMATION ===
{extra_context}

IMPORTANT: Pay close attention to the user's hints above. They have tested this game and know how it works.
"""

    prompt = WEB_GAME_PROMPT.format(
        game_url=game_url,
        page_info=page_info,
        extra_context=context_section
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

