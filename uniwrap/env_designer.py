"""Environment spec generation using Claude LLM."""

import json
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from uniwrap.llm_client import ClaudeClient
from uniwrap.utils import clean_json_response


def validate_python_code(code: str) -> Tuple[bool, Optional[str]]:
    """Validate Python code for syntax errors.

    Args:
        code: Python source code to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"


def fix_common_code_issues(code: str) -> str:
    """Attempt to fix common code generation issues.

    Args:
        code: Python source code

    Returns:
        Fixed code
    """
    import re

    # Fix missing self. prefix for common instance variables
    # Pattern: self.X = ... Y ... where Y is an uppercase variable without self.
    # This catches cases like: self.GAME_RES = (self.W * TILE, ...) -> self.TILE

    # Find all self.VARNAME definitions
    defined_vars = set(re.findall(r'self\.([A-Z_]+)\s*=', code))

    # For each defined var, check if it's used without self. prefix
    for var in defined_vars:
        # Pattern: not preceded by 'self.' but the variable name appears
        # Look for the variable used without self. (but not in string literals)
        pattern = rf'(?<!self\.)(?<!["\'])(?<![a-zA-Z_])({var})(?![a-zA-Z_])'

        # Check if the variable is used without self. in assignments
        matches = list(re.finditer(pattern, code))
        for match in matches:
            # Skip if this is the definition itself or in a string
            pos = match.start()
            # Check if we're inside a string by counting quotes
            before = code[:pos]
            if before.count('"') % 2 == 0 and before.count("'") % 2 == 0:
                # Not in a string, replace with self.VAR
                code = code[:pos] + 'self.' + code[pos:]

    return code


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

=== CRITICAL: IDENTIFYING THE GAME ELEMENT ===

Web game pages often have many elements (ads, menus, navigation, social buttons). You MUST identify the ACTUAL GAME:

1. LOOK FOR THE GAME CANVAS/IFRAME:
   - Canvas elements: 'canvas', '#game-canvas', '#gameCanvas', '.game-canvas'
   - Iframe embeds: 'iframe[src*="game"]', 'iframe.game-frame', '#game-iframe'
   - Game containers: '#game', '#game-container', '.game-wrapper', '[id*="game"]'
   - Unity/WebGL: 'canvas#unity-canvas', '#gameContainer canvas'

2. IDENTIFY BY SIZE AND POSITION:
   - Games are usually the LARGEST canvas/iframe on the page
   - Games are typically centered or prominently placed
   - Ads are usually in sidebars, headers, or fixed positions
   - Look for elements with reasonable game dimensions (400-1200px wide, 300-800px tall)

3. EXCLUDE NON-GAME ELEMENTS:
   - Ads: '[class*="ad"]', '[id*="ad"]', 'ins.adsbygoogle', '[data-ad]'
   - Social: '[class*="social"]', '[class*="share"]', '.fb-like'
   - Navigation: 'header', 'nav', 'footer', '.menu', '.sidebar'
   - Popups: '.modal', '.popup', '.overlay', '[class*="cookie"]'

4. DETECTION STRATEGY in _start_browser():
   ```python
   # Try specific game selectors first
   game_selectors = [
       'canvas#game', 'canvas.game', '#game canvas',
       'iframe[src*="game"]', '#game-container canvas',
       'canvas:not([class*="ad"])'  # canvas that's not an ad
   ]

   for selector in game_selectors:
       try:
           element = self.page.locator(selector).first
           if element.is_visible():
               bbox = element.bounding_box()
               # Verify it's a reasonable game size (not a tiny ad)
               if bbox and bbox['width'] > 300 and bbox['height'] > 200:
                   self.game_element = element
                   break
       except:
           continue

   # Fallback: find largest canvas
   if not self.game_element:
       canvases = self.page.locator('canvas').all()
       largest = None
       largest_area = 0
       for c in canvases:
           bbox = c.bounding_box()
           if bbox:
               area = bbox['width'] * bbox['height']
               if area > largest_area:
                   largest_area = area
                   largest = c
       self.game_element = largest
   ```

5. HIDE DISTRACTING ELEMENTS:
   ```python
   def _hide_overlays(self):
       self.page.evaluate('''() => {{
           const hide = [
               '[class*="ad"]', '[id*="ad"]', 'ins.adsbygoogle',
               '[class*="social"]', '[class*="share"]',
               '[class*="cookie"]', '[class*="consent"]',
               '.overlay:not(#game)', 'header', 'footer'
           ];
           hide.forEach(sel => {{
               document.querySelectorAll(sel).forEach(el => {{
                   if (!el.closest('#game') && !el.closest('.game')) {{
                       el.style.display = 'none';
                   }}
               }});
           }});
       }}''')
   ```

=== GAME OVER DETECTION (CRITICAL - AVOID FALSE POSITIVES) ===

Premature game over detection is a COMMON BUG that ruins training. The agent thinks the game ended when it didn't!

1. GRACE PERIOD AT START:
   - Skip game over checks for the first 30-50 steps
   - Games often have loading screens, start animations, or need a click to begin
   - These look like "stillness" but aren't game over
   ```python
   def _is_game_over(self, observation):
       # Don't check game over during startup
       if self.steps_taken < 30:
           return False
   ```

2. IDENTICAL FRAME DETECTION - USE HIGH THRESHOLD:
   - 8 identical frames is TOO FEW - brief pauses are normal
   - Use 30+ identical frames (~1 second at 30fps) minimum
   - Some games have natural pauses, menus, or slow moments
   ```python
   # BAD: Too aggressive
   if self.identical_frame_count >= 8:  # WRONG - causes false positives
       return True

   # GOOD: Conservative threshold
   if self.identical_frame_count >= 30:  # ~1 second of no change
       return True
   ```

3. AVOID UNRELIABLE COLOR DETECTION:
   - DON'T use "red pixel count" or similar heuristics
   - Game elements, UI, particles can trigger false positives
   - Red doesn't always mean game over - could be health bar, enemies, effects
   ```python
   # BAD - Don't do this:
   if red_pixels > 5000:
       return True  # WRONG - too many false positives
   ```

4. PREFER TEXT/DOM DETECTION:
   - Check for actual "game over" text on the page
   - Look for restart buttons appearing
   - Check game state variables via JavaScript if possible
   ```python
   # GOOD - Check for actual game over text
   try:
       game_over_text = self.page.evaluate('''() => {{
           const text = document.body.innerText.toLowerCase();
           return text.includes('game over') || text.includes('try again') || text.includes('play again');
       }}''')
       if game_over_text:
           return True
   except:
       pass
   ```

5. GAME STATE VIA JAVASCRIPT (BEST):
   - Some games expose state variables
   - Check canvas context, game objects, or global variables
   ```python
   try:
       is_over = self.page.evaluate('() => window.gameOver || window.game?.isOver || false')
       if is_over:
           return True
   except:
       pass
   ```

6. COMPLETE _is_game_over() TEMPLATE:
   ```python
   def _is_game_over(self, observation: np.ndarray) -> bool:
       # Grace period - don't check during game startup
       if self.steps_taken < 30:
           return False

       # Track frame changes
       frame_hash = hashlib.md5(observation.tobytes()).hexdigest()
       if frame_hash == self.last_frame_hash:
           self.identical_frame_count += 1
       else:
           self.identical_frame_count = 0
       self.last_frame_hash = frame_hash

       # Only trigger on LONG stillness (30+ frames = ~1 second)
       if self.identical_frame_count >= 30:
           return True

       # Check for game over text (reliable)
       try:
           game_over = self.page.evaluate('''() => {{
               const text = document.body.innerText.toLowerCase();
               return text.includes('game over') || text.includes('try again');
           }}''')
           if game_over:
               return True
       except:
           pass

       return False
   ```

=== SCREENSHOT AND OBSERVATION ===
- Use page.screenshot(clip={{...}}) to capture ONLY the game element
- Get bounding box from the identified game element
- Resolution 400-600px wide is good
- Convert to numpy array using PIL

=== BROWSER SETUP ===
- Use wait_until="domcontentloaded" (NOT "networkidle" - ads never stop loading)
- Viewport: 1280x800 or larger
- Scroll game into view and click to focus
- Call _hide_overlays() after page load
- Handle cookie consent popups

=== REWARD DESIGN (CRITICAL FOR LEARNING) ===

The agent MUST receive meaningful feedback to learn. A flat "+0.1 per step" teaches nothing!

1. UNDERSTAND THE GAME OBJECTIVE FIRST:
   - What is the player trying to do? (survive, collect, avoid, reach goal)
   - What actions lead to success? (jumping at right time, moving to safe spots)
   - What causes failure? (collision, falling, timeout)

2. DESIGN INTERMEDIATE/SHAPING REWARDS:
   Many games don't have frequent score updates. You MUST create intermediate rewards:

   For AVOIDANCE games (dodge obstacles, survive):
   ```python
   # Track player position relative to dangers
   # Reward for maintaining safe distance from obstacles
   # Reward for successful dodges (obstacle passed without collision)
   # Small penalty for risky positions (too close to danger)
   ```

   For COLLECTION games (collect coins, eat food):
   ```python
   # Reward for moving toward collectibles (distance decreased)
   # Penalty for moving away from collectibles
   # Big reward for actually collecting
   ```

   For MOVEMENT games (reach goal, navigate):
   ```python
   # Reward for progress toward goal
   # Penalty for going backward
   # Reward for speed/efficiency
   ```

3. EXTRACT GAME STATE FOR REWARDS:
   - Use JavaScript to read game variables if possible
   - Track player position changes between frames
   - Detect near-misses (close to obstacle but survived)
   - Count successful actions (jumps that avoided something)

4. REWARD STRUCTURE TEMPLATE:
   ```python
   def _calculate_reward(self):
       reward = 0.0

       # Base survival reward (small)
       reward += 0.01

       # Score-based reward (if available)
       score_delta = self.current_score - self.last_score
       if score_delta > 0:
           reward += score_delta * 1.0

       # Action effectiveness reward
       # e.g., if agent took action and something good happened
       if self.action_was_useful:
           reward += 0.5

       # Position-based shaping (game specific)
       # e.g., reward for being in safe zone, penalty for danger zone

       # Penalty for game over
       if self.game_over:
           reward = -1.0

       return reward
   ```

5. SPECIFIC GAME PATTERNS:

   JUMPING GAMES (Dino, Flappy Bird style):
   - Reward for each obstacle successfully passed
   - Track obstacles and detect when player clears them
   - Penalize unnecessary jumps (wastes opportunity)

   DODGING GAMES (Star Dodge, Asteroids style):
   - Reward for time survived (but scale with difficulty)
   - Reward for active dodging (position changed to avoid threat)
   - Track "close calls" - near misses deserve bonus

   CLICK/TAP GAMES:
   - Reward must be tied to the ACTION having an effect
   - If clicking does nothing useful, no reward
   - Reward when click causes positive game state change

6. AVOID THESE MISTAKES:
   - DON'T give same reward regardless of action (agent won't learn)
   - DON'T only reward on game over (too sparse)
   - DON'T make rewards too large (causes instability)
   - DO scale rewards to roughly -1.0 to +1.0 range
   - DO give MORE reward for ACTIVE good play than passive survival

Output a complete Python file with:
1. All necessary imports (gymnasium, numpy, PIL, time, typing)
2. nest_asyncio import and apply() BEFORE playwright import:
   ```python
   try:
       import nest_asyncio
       nest_asyncio.apply()
   except ImportError:
       pass
   from playwright.sync_api import sync_playwright
   ```
3. A gym.Env subclass with __init__, reset, step, render, close methods
4. Discrete action space (0=nothing, 1=jump/action, optionally 2=duck/alt)
5. Box observation space for screenshots
6. _start_browser() with INTELLIGENT GAME ELEMENT DETECTION
7. _hide_overlays() to remove ads and distractions
8. _get_canvas_screenshot() that clips to the GAME element only
9. _is_game_over() with visual detection

The class name should be based on the game URL path (e.g., SpaceWavesEnv for /space-waves).

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

=== CRITICAL: PYGAME FONT HANDLING (Python 3.14+) ===

Python 3.14+ has circular import issues with pygame.font. Use pygame._freetype instead:

```python
import os

# Import pygame._freetype BEFORE pygame to avoid circular import
try:
    import pygame._freetype as freetype
    FREETYPE_AVAILABLE = True
except ImportError:
    FREETYPE_AVAILABLE = False
    freetype = None

import pygame

# System font paths
SYSTEM_FONT_PATHS = [
    '/System/Library/Fonts/Helvetica.ttc',  # macOS
    '/System/Library/Fonts/Supplemental/Arial.ttf',  # macOS
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
    'C:\\\\Windows\\\\Fonts\\\\arial.ttf',  # Windows
]

def get_system_font_path():
    for path in SYSTEM_FONT_PATHS:
        if os.path.exists(path):
            return path
    return None

# In render() method:
def render(self):
    if self.render_mode != "human" or self.screen is None:
        return None

    self.screen.fill((0, 0, 0))
    # ... draw game objects ...

    # Font rendering with freetype (works on Python 3.14)
    try:
        if FREETYPE_AVAILABLE:
            font_path = get_system_font_path()
            if font_path and freetype:
                freetype.init()
                ft_font = freetype.Font(font_path, 36)
                # freetype.render() returns (surface, rect)
                text_surf, _ = ft_font.render('Score: 100', pygame.Color('white'))
                self.screen.blit(text_surf, (10, 10))
        else:
            # Fallback for older Python
            font = pygame.font.Font(None, 36)
            text = font.render('Score: 100', True, (255, 255, 255))
            self.screen.blit(text, (10, 10))
    except Exception:
        pass  # Skip text if fonts unavailable

    pygame.display.flip()
```

IMPORTANT: Import pygame._freetype BEFORE pygame to avoid circular import errors.

=== CRITICAL: CODE CORRECTNESS RULES ===

Your generated code MUST be syntactically correct and runnable. Common mistakes to AVOID:

1. ALWAYS use `self.` prefix for instance variables:
   - WRONG: `self.GAME_RES = (self.W * TILE, self.H * TILE)`
   - RIGHT: `self.GAME_RES = (self.W * self.TILE, self.H * self.TILE)`

2. Define variables BEFORE using them:
   - WRONG: Using `self.field` before it's assigned in __init__
   - RIGHT: Initialize all instance variables at the start of __init__

3. All methods must have `self` as first parameter:
   - WRONG: `def _check_collision():`
   - RIGHT: `def _check_collision(self):`

4. Import all modules you use:
   - If using `deepcopy`, add `from copy import deepcopy`
   - If using `randrange`, add `from random import randrange`

5. Consistent variable naming - don't mix styles:
   - Pick either `game_over` or `gameOver`, not both

BEFORE outputting code, mentally trace through __init__ and verify every variable reference is valid.

Output a complete Python file with:
1. All necessary imports
2. A gym.Env subclass with __init__, reset, step, render, close methods
3. Embedded game logic (adapted from the original code)
4. Proper observation and action spaces
5. Reward calculation based on game events
6. ROBUST FONT HANDLING with try/except (see above)
7. ALL instance variables use `self.` prefix consistently

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

    code = code.strip()

    # Validate and fix common issues
    code = fix_common_code_issues(code)

    is_valid, error = validate_python_code(code)
    if not is_valid:
        print(f"   ⚠️  Generated code has syntax error: {error}")
        print("   Attempting to continue anyway...")

    return code


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

    code = code.strip()

    # Validate and fix common issues
    code = fix_common_code_issues(code)

    is_valid, error = validate_python_code(code)
    if not is_valid:
        print(f"   ⚠️  Generated code has syntax error: {error}")
        print("   Attempting to continue anyway...")

    return code



def generate_environment_code(
    game_info: Dict,
    game_type: str = "web",
    hints: str = "",
    model: str = "claude-sonnet-4-5-20250929",
    api_key: str = None
) -> str:
    """
    Unified function to generate environment code from game info.
    Used by the agent orchestrator.

    Args:
        game_info: Dict with game information from browsing
        game_type: "web" or "pygame"
        hints: Additional user hints
        model: Claude model to use
        api_key: Optional API key

    Returns:
        Complete Python environment code
    """
    import json

    # Format game info as page_info string
    page_info = f"""
URL: {game_info.get('url', 'Unknown')}
Title: {game_info.get('title', 'Unknown')}
Has Canvas: {game_info.get('has_canvas', False)}
Canvas IDs: {game_info.get('canvas_ids', [])}

Instructions Found:
{json.dumps(game_info.get('instructions', []), indent=2)}

Page Text Preview:
{game_info.get('text_preview', '')[:1500]}

Inline Script Preview:
{game_info.get('inline_scripts_preview', '')[:1000]}
"""

    if game_type == "pygame":
        # For pygame, we need the actual code - for now, use web game approach
        # with pygame-specific prompt
        return generate_pygame_env_code(
            code_files={},  # Would need actual code
            main_file="game.py",
            model=model,
            api_key=api_key
        )
    else:
        return generate_web_game_env_code(
            game_url=game_info.get('url', ''),
            page_info=page_info,
            model=model,
            api_key=api_key,
            extra_context=hints
        )
