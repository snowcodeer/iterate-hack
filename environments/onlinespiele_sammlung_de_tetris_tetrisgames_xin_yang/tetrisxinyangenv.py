import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import time
from typing import Optional, Tuple
import hashlib
import io

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

from playwright.sync_api import sync_playwright, Page, Browser


class TetrisXinYangEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, headless: bool = False):
        super().__init__()
        
        self.render_mode = render_mode
        self.headless = headless
        self.game_url = "https://www.onlinespiele-sammlung.de/tetris/tetrisgames/xin-yang/"
        
        # Action space: 0=noop, 1=left, 2=right, 3=rotate, 4=down, 5=drop
        self.action_space = spaces.Discrete(6)
        
        # Observation space: RGB screenshot
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(400, 300, 3), dtype=np.uint8
        )
        
        # Browser objects
        self.playwright = None
        self.browser = None
        self.page = None
        self.game_element = None
        
        # Game state tracking
        self.steps_taken = 0
        self.last_frame_hash = None
        self.identical_frame_count = 0
        self.current_score = 0
        self.last_score = 0
        self.lines_cleared = 0
        self.last_lines_cleared = 0
        self.level = 1
        self.last_level = 1
        self.game_started = False
        
        # Grid state tracking for reward shaping
        self.last_grid_height = 0
        self.pieces_placed = 0
        self.last_pieces_placed = 0

    def _start_browser(self):
        if self.playwright is None:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=self.headless)
            context = self.browser.new_context(viewport={"width": 1280, "height": 900})
            self.page = context.new_page()
            
            # Navigate to game
            self.page.goto(self.game_url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(2)
            
            # Handle any overlays or popups
            self._hide_overlays()
            
            # Identify game element - this is an HTML table-based tetris game
            # The game board is in a div with id="Holder"
            try:
                holder = self.page.locator("#Holder").first
                if holder.is_visible():
                    self.game_element = holder
                else:
                    # Fallback to body
                    self.game_element = self.page.locator("body").first
            except:
                self.game_element = self.page.locator("body").first
            
            # Scroll game into view
            if self.game_element:
                self.game_element.scroll_into_view_if_needed()
                time.sleep(0.5)
            
            # Click to focus the game
            try:
                self.page.click("body")
                time.sleep(0.5)
            except:
                pass
            
            # Start the game if there's a start button
            try:
                start_selectors = ['input[value="Start"]', 'input[value="start"]', 'button:has-text("Start")']
                for selector in start_selectors:
                    try:
                        if self.page.locator(selector).is_visible():
                            self.page.click(selector)
                            self.game_started = True
                            time.sleep(0.5)
                            break
                    except:
                        continue
            except:
                pass

    def _hide_overlays(self):
        try:
            self.page.evaluate('''() => {
                const hide = [
                    '[class*="ad"]:not(#Holder):not(#Holder *)', 
                    '[id*="ad"]:not(#Holder):not(#Holder *)', 
                    'ins.adsbygoogle',
                    '[class*="social"]', 
                    '[class*="share"]',
                    '[class*="cookie"]', 
                    '[class*="consent"]',
                    'header:not(#Holder):not(#Holder *)', 
                    'footer:not(#Holder):not(#Holder *)',
                    'nav:not(#Holder):not(#Holder *)'
                ];
                hide.forEach(sel => {
                    try {
                        document.querySelectorAll(sel).forEach(el => {
                            el.style.display = 'none';
                        });
                    } catch(e) {}
                });
            }''')
        except:
            pass

    def _get_canvas_screenshot(self) -> np.ndarray:
        try:
            if self.game_element:
                bbox = self.game_element.bounding_box()
                if bbox:
                    screenshot_bytes = self.page.screenshot(
                        clip={
                            "x": bbox["x"],
                            "y": bbox["y"],
                            "width": bbox["width"],
                            "height": bbox["height"]
                        }
                    )
                else:
                    screenshot_bytes = self.page.screenshot()
            else:
                screenshot_bytes = self.page.screenshot()
            
            image = Image.open(io.BytesIO(screenshot_bytes))
            image = image.resize((300, 400), Image.Resampling.LANCZOS)
            obs = np.array(image)
            
            if obs.shape != (400, 300, 3):
                obs = np.zeros((400, 300, 3), dtype=np.uint8)
            
            return obs
        except Exception as e:
            return np.zeros((400, 300, 3), dtype=np.uint8)

    def _extract_game_state(self):
        try:
            state = self.page.evaluate('''() => {
                const scoreElem = document.querySelector('p.score1, p.score2, p.score3, p.score4');
                let score = 0;
                if (scoreElem) {
                    const text = scoreElem.innerText;
                    const match = text.match(/Score[:\\s]*([0-9]+)/i);
                    if (match) {
                        score = parseInt(match[1]);
                    }
                }
                
                const levelElem = document.querySelector('p.level');
                let level = 1;
                if (levelElem) {
                    const text = levelElem.innerText;
                    const match = text.match(/Level[:\\s]*([0-9]+)/i);
                    if (match) {
                        level = parseInt(match[1]);
                    }
                }
                
                const bodyText = document.body.innerText.toLowerCase();
                const gameOver = bodyText.includes('game over') || 
                                bodyText.includes('gameover') ||
                                document.querySelector('p.gameover') !== null;
                
                const lines = (bodyText.match(/lines[:\\s]*([0-9]+)/i) || [0, 0])[1];
                
                return {
                    score: score,
                    level: level,
                    gameOver: gameOver,
                    lines: parseInt(lines) || 0
                };
            }''')
            
            self.current_score = state.get('score', 0)
            self.level = state.get('level', 1)
            self.lines_cleared = state.get('lines', 0)
            
            return state.get('gameOver', False)
        except:
            return False

    def _is_game_over(self, observation: np.ndarray) -> bool:
        # Grace period - don't check during startup
        if self.steps_taken < 30:
            return False
        
        # Check game state via DOM
        try:
            is_over = self._extract_game_state()
            if is_over:
                return True
        except:
            pass
        
        # Track frame changes
        frame_hash = hashlib.md5(observation.tobytes()).hexdigest()
        if frame_hash == self.last_frame_hash:
            self.identical_frame_count += 1
        else:
            self.identical_frame_count = 0
        self.last_frame_hash = frame_hash
        
        # Only trigger on LONG stillness (50+ frames)
        if self.identical_frame_count >= 50:
            return True
        
        return False

    def _calculate_reward(self) -> float:
        reward = 0.0
        
        # Base survival reward (small)
        reward += 0.01
        
        # Score increase reward (primary reward)
        score_delta = self.current_score - self.last_score
        if score_delta > 0:
            reward += score_delta * 0.01
        
        # Lines cleared reward (important milestone)
        lines_delta = self.lines_cleared - self.last_lines_cleared
        if lines_delta > 0:
            # More lines at once = combo bonus
            if lines_delta == 1:
                reward += 0.5
            elif lines_delta == 2:
                reward += 1.5
            elif lines_delta == 3:
                reward += 3.0
            elif lines_delta >= 4:
                reward += 5.0
        
        # Level up reward
        level_delta = self.level - self.last_level
        if level_delta > 0:
            reward += 2.0
        
        # Update tracking
        self.last_score = self.current_score
        self.last_lines_cleared = self.lines_cleared
        self.last_level = self.level
        
        return reward

    def _send_action(self, action: int):
        try:
            if action == 0:
                # No operation
                pass
            elif action == 1:
                # Left arrow
                self.page.keyboard.press("ArrowLeft")
            elif action == 2:
                # Right arrow
                self.page.keyboard.press("ArrowRight")
            elif action == 3:
                # Rotate (Up arrow or Space)
                self.page.keyboard.press("ArrowUp")
            elif action == 4:
                # Soft drop (Down arrow)
                self.page.keyboard.press("ArrowDown")
            elif action == 5:
                # Hard drop (Space)
                self.page.keyboard.press("Space")
            
            time.sleep(0.05)
        except:
            pass

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        if self.page is None:
            self._start_browser()
        else:
            # Reload page to restart game
            try:
                self.page.reload(wait_until="domcontentloaded", timeout=10000)
                time.sleep(1)
                self._hide_overlays()
                
                # Try to start the game
                try:
                    if self.page.locator('input[value="Start"]').is_visible():
                        self.page.click('input[value="Start"]')
                        time.sleep(0.5)
                except:
                    pass
                
                # Click to focus
                self.page.click("body")
                time.sleep(0.5)
            except:
                self._start_browser()
        
        # Reset tracking variables
        self.steps_taken = 0
        self.last_frame_hash = None
        self.identical_frame_count = 0
        self.current_score = 0
        self.last_score = 0
        self.lines_cleared = 0
        self.last_lines_cleared = 0
        self.level = 1
        self.last_level = 1
        self.last_grid_height = 0
        self.pieces_placed = 0
        self.last_pieces_placed = 0
        
        # Get initial observation
        observation = self._get_canvas_screenshot()
        info = {}
        
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.steps_taken += 1
        
        # Send action to game
        self._send_action(action)
        
        # Small delay for game to update
        time.sleep(0.05)
        
        # Get observation
        observation = self._get_canvas_screenshot()
        
        # Extract game state
        self._extract_game_state()
        
        # Check if game over
        terminated = self._is_game_over(observation)
        truncated = False
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Penalty for game over
        if terminated:
            reward = -1.0
        
        info = {
            "score": self.current_score,
            "level": self.level,
            "lines": self.lines_cleared,
            "steps": self.steps_taken
        }
        
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_canvas_screenshot()
        elif self.render_mode == "human":
            return None

    def close(self):
        if self.page:
            self.page.close()
            self.page = None
        if self.browser:
            self.browser.close()
            self.browser = None
        if self.playwright:
            self.playwright.stop()
            self.playwright = None