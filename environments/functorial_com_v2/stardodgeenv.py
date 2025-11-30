import gymnasium as gym
from gymnasium import spaces
import numpy as np
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from PIL import Image
import io
import time

class StarDodgeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, headless=False):
        super().__init__()
        
        self.render_mode = render_mode
        self.headless = headless
        self.game_url = "https://functorial.com/star-dodge-clone/"
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(500, 800, 3), dtype=np.uint8
        )
        
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.canvas_bbox = None
        
        self.current_frame = None
        self.previous_frames = []
        self.identical_frame_count = 0
        self.steps_taken = 0
        self.last_score = 0
        self.game_started = False
        
        self._start_browser()

    def _start_browser(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context(
            viewport={"width": 1280, "height": 900}
        )
        self.page = self.context.new_page()
        
        self.page.goto(self.game_url, wait_until="networkidle", timeout=30000)
        time.sleep(2)
        
        try:
            canvas = self.page.locator("#canvas")
            canvas.wait_for(timeout=5000)
            canvas.scroll_into_view_if_needed()
            time.sleep(0.5)
            
            self.canvas_bbox = canvas.bounding_box()
            if self.canvas_bbox is None:
                raise RuntimeError("Canvas bounding box is None")
            
            canvas.click()
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error locating canvas: {e}")
            self.canvas_bbox = {"x": 240, "y": 100, "width": 800, "height": 500}

    def _get_canvas_screenshot(self):
        if self.canvas_bbox:
            screenshot_bytes = self.page.screenshot(
                clip={
                    "x": self.canvas_bbox["x"],
                    "y": self.canvas_bbox["y"],
                    "width": self.canvas_bbox["width"],
                    "height": self.canvas_bbox["height"]
                }
            )
        else:
            screenshot_bytes = self.page.screenshot()
        
        image = Image.open(io.BytesIO(screenshot_bytes))
        image = image.convert("RGB")
        image = image.resize((800, 500), Image.Resampling.LANCZOS)
        return np.array(image)

    def _is_game_over(self):
        """
        Detect game over state with conservative thresholds to avoid false positives.
        Uses multiple detection methods with proper grace periods.
        """
        # GRACE PERIOD: Don't check during startup to avoid false positives
        # Games often have loading screens or initial animations
        if self.steps_taken < 30:
            return False
        
        # METHOD 1: Text-based detection (most reliable)
        # Look for common game over phrases in the page text
        try:
            game_over_text = self.page.evaluate("""
                () => {
                    const bodyText = document.body.innerText.toLowerCase();
                    const gameOverPhrases = [
                        'game over',
                        'gameover', 
                        'try again',
                        'you died',
                        'you crashed',
                        'play again',
                        'restart'
                    ];
                    
                    for (const phrase of gameOverPhrases) {
                        if (bodyText.includes(phrase)) {
                            return true;
                        }
                    }
                    
                    // Check for game over elements by class or ID
                    const gameOverElements = document.querySelectorAll(
                        '.game-over, .gameover, #game-over, #gameover, .game-over-screen'
                    );
                    
                    for (const elem of gameOverElements) {
                        const style = window.getComputedStyle(elem);
                        // Element exists and is visible
                        if (style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0') {
                            return true;
                        }
                    }
                    
                    return false;
                }
            """)
            
            if game_over_text:
                return True
        except Exception as e:
            # Don't crash on JavaScript errors, just continue to other methods
            pass
        
        # METHOD 2: Frame freeze detection (conservative threshold)
        # If frames are identical for extended period, game likely ended
        # Using 30 frames (1 second at 30fps) to be very conservative
        if self.current_frame is not None and len(self.previous_frames) > 0:
            frame_diff = np.mean(np.abs(
                self.current_frame.astype(float) - self.previous_frames[-1].astype(float)
            ))
            
            # Threshold of 1.0 is conservative - allows for small UI animations
            if frame_diff < 1.0:
                self.identical_frame_count += 1
            else:
                self.identical_frame_count = 0
            
            # Require 30+ consecutive identical frames (not just 5)
            # This prevents false positives from brief pauses or loading
            if self.identical_frame_count >= 30:
                return True
        
        return False

    def _get_score(self):
        """Extract score from the game UI."""
        try:
            score = self.page.evaluate("""
                () => {
                    // Method 1: Look for score in text content
                    const textElements = document.querySelectorAll('*');
                    for (let elem of textElements) {
                        const text = elem.textContent || '';
                        const match = text.match(/score[:\s]*(\d+)/i);
                        if (match) {
                            return parseInt(match[1]);
                        }
                    }
                    
                    // Method 2: Look for score by ID or class
                    const scoreElem = document.querySelector('#score, .score, [class*="score"]');
                    if (scoreElem) {
                        const scoreText = scoreElem.textContent.replace(/\D/g, '');
                        if (scoreText) {
                            return parseInt(scoreText);
                        }
                    }
                    
                    return 0;
                }
            """)
            return score if score else 0
        except:
            return 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.page:
            try:
                self.page.reload(wait_until="networkidle", timeout=10000)
            except:
                self.page.reload()
            
            time.sleep(2)
            
            try:
                canvas = self.page.locator("#canvas")
                canvas.click()
                time.sleep(0.3)
            except:
                pass
        
        self.steps_taken = 0
        self.last_score = 0
        self.identical_frame_count = 0
        self.previous_frames = []
        self.game_started = False
        
        time.sleep(0.5)
        
        observation = self._get_canvas_screenshot()
        self.current_frame = observation
        
        return observation, {}

    def step(self, action):
        self.steps_taken += 1
        
        # Execute action: 0 = do nothing, 1 = press space
        if action == 1:
            self.page.keyboard.down("Space")
            time.sleep(0.05)
            self.page.keyboard.up("Space")
        
        # Wait for frame update (30 FPS)
        time.sleep(0.033)
        
        observation = self._get_canvas_screenshot()
        
        # Update frame history for game over detection
        self.previous_frames.append(self.current_frame)
        if len(self.previous_frames) > 10:
            self.previous_frames.pop(0)
        self.current_frame = observation
        
        # Get current score
        current_score = self._get_score()
        score_delta = current_score - self.last_score
        self.last_score = current_score
        
        # REWARD FUNCTION:
        # Primary signal: survival time (dodge game focuses on staying alive)
        # Secondary signal: score increases (if game tracks score)
        # Penalty: death (strong negative to avoid early termination)
        
        reward = 0.0
        
        # Base survival reward: 0.01 per step (encourages staying alive)
        # This provides continuous positive feedback for not dying
        reward += 0.01
        
        # Score increase reward: Large bonus for improving score
        # Multiplier of 100 makes score changes very significant
        if score_delta > 0:
            reward += score_delta * 100
        
        # Score-based survival bonus: Higher scores = better play
        # Reward scales with current score to encourage sustained performance
        if current_score > 0:
            reward += 0.001 * current_score
        
        # Check for game over
        terminated = self._is_game_over()
        
        # Death penalty: Strong negative reward to discourage dying
        # -10.0 is significant but not overwhelming given positive rewards
        if terminated:
            reward = -10.0
        
        # Truncate very long episodes
        truncated = False
        if self.steps_taken > 10000:
            truncated = True
        
        info = {
            "score": current_score,
            "steps": self.steps_taken,
            "identical_frames": self.identical_frame_count
        }
        
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.current_frame
        elif self.render_mode == "human":
            return None

    def close(self):
        if self.page:
            self.page.close()
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()