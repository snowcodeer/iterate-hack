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
            
            # Click canvas to start/focus
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
        Detect game over state using multiple methods.
        Uses conservative thresholds to avoid false positives.
        """
        
        # GRACE PERIOD: Skip all checks during first 30 steps
        # This prevents false positives during game loading/startup
        if self.steps_taken < 30:
            return False
        
        # METHOD 1: Text-based detection (most reliable)
        try:
            game_over_text = self.page.evaluate("""
                () => {
                    const bodyText = document.body.innerText.toLowerCase();
                    
                    // Check for common game over phrases
                    if (bodyText.includes('game over') || 
                        bodyText.includes('try again') ||
                        bodyText.includes('restart') ||
                        bodyText.includes('crashed')) {
                        return true;
                    }
                    
                    // Check for game over elements by class/id
                    const gameOverElements = document.querySelectorAll(
                        '.game-over, .gameover, #gameOver, [class*="game-over"]'
                    );
                    if (gameOverElements.length > 0) {
                        for (let elem of gameOverElements) {
                            const style = window.getComputedStyle(elem);
                            if (style.display !== 'none' && style.visibility !== 'hidden') {
                                return true;
                            }
                        }
                    }
                    
                    return false;
                }
            """)
            
            if game_over_text:
                return True
        except Exception as e:
            pass
        
        # METHOD 2: Frozen frame detection (conservative threshold)
        # Only trigger if we have enough frame history
        if self.current_frame is not None and len(self.previous_frames) > 0:
            frame_diff = np.mean(np.abs(
                self.current_frame.astype(float) - self.previous_frames[-1].astype(float)
            ))
            
            # Increased threshold from 1.0 to 2.0 for more conservative detection
            if frame_diff < 2.0:
                self.identical_frame_count += 1
            else:
                self.identical_frame_count = 0
            
            # Increased threshold from 5 to 30 frames
            # This means ~1 second of identical frames at 30 FPS
            if self.identical_frame_count >= 30:
                return True
        
        return False

    def _get_score(self):
        """
        Extract current score from the game.
        Uses multiple methods for robustness.
        """
        try:
            score = self.page.evaluate("""
                () => {
                    // Method 1: Look for score in text content
                    const textElements = document.querySelectorAll('*');
                    for (let elem of textElements) {
                        const text = elem.textContent || '';
                        
                        // Try different score patterns
                        let match = text.match(/score[:\s]*(\d+)/i);
                        if (match) {
                            return parseInt(match[1]);
                        }
                        
                        match = text.match(/points[:\s]*(\d+)/i);
                        if (match) {
                            return parseInt(match[1]);
                        }
                    }
                    
                    // Method 2: Check for score element by ID/class
                    const scoreElem = document.querySelector('#score, .score, [class*="score"]');
                    if (scoreElem) {
                        const scoreText = scoreElem.textContent.match(/(\d+)/);
                        if (scoreText) {
                            return parseInt(scoreText[1]);
                        }
                    }
                    
                    return 0;
                }
            """)
            return score if score else 0
        except Exception as e:
            return 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.page:
            try:
                self.page.reload(wait_until="networkidle", timeout=10000)
            except:
                self.page.reload()
            
            time.sleep(2)
            
            # Try to click canvas multiple times if needed
            for attempt in range(3):
                try:
                    canvas = self.page.locator("#canvas")
                    canvas.click()
                    time.sleep(0.3)
                    break
                except:
                    if attempt == 2:
                        # Last resort: click center of screen
                        self.page.mouse.click(640, 450)
                    time.sleep(0.2)
        
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
        
        # Execute action
        if action == 1:  # Press space to dodge/jump
            self.page.keyboard.down("Space")
            time.sleep(0.05)
            self.page.keyboard.up("Space")
        # action == 0 is no-op (let gravity/default behavior happen)
        
        # Wait for frame to update
        time.sleep(0.033)  # ~30 FPS
        
        # Get new observation
        observation = self._get_canvas_screenshot()
        
        # Update frame history for game over detection
        self.previous_frames.append(self.current_frame)
        if len(self.previous_frames) > 10:
            self.previous_frames.pop(0)
        
        # Calculate frame difference for movement reward
        frame_diff = 0.0
        if self.current_frame is not None:
            frame_diff = np.mean(np.abs(
                observation.astype(float) - self.current_frame.astype(float)
            ))
        
        self.current_frame = observation
        
        # Get current score
        current_score = self._get_score()
        score_delta = current_score - self.last_score
        self.last_score = current_score
        
        # === REWARD CALCULATION ===
        # Scale rewards to stay roughly in [-1.0, +1.0] range
        
        reward = 0.0
        
        # 1. Survival reward: Small positive reward for staying alive
        #    Encourages agent to keep playing
        reward += 0.01
        
        # 2. Score reward: Main reward signal from game score
        #    Normalized to reasonable scale (reduced from 10x to 1x)
        reward += score_delta * 1.0
        
        # 3. Movement reward: Small reward for visual changes
        #    Encourages active dodging behavior vs staying still
        #    Capped to prevent dominating other rewards
        movement_reward = min(frame_diff / 100.0, 0.05)
        reward += movement_reward
        
        # Check termination
        terminated = self._is_game_over()
        
        # 4. Death penalty: Large negative reward for game over
        #    Makes agent strongly prefer survival
        if terminated:
            reward = -10.0
        
        # Truncation for very long episodes
        truncated = False
        if self.steps_taken > 10000:
            truncated = True
        
        info = {
            "score": current_score,
            "steps": self.steps_taken,
            "identical_frames": self.identical_frame_count,
            "frame_diff": frame_diff,
            "score_delta": score_delta
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