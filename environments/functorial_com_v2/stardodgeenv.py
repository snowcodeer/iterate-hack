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
        self.survival_steps = 0
        
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
        # GRACE PERIOD: Don't check for game over during startup/loading
        # This prevents false positives during initial frames
        if self.steps_taken < 30:
            return False
        
        # PRIMARY METHOD: Text-based detection (most reliable)
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
                    
                    // Check for game over class elements
                    const gameOverElements = document.querySelectorAll(
                        '.game-over, .gameover, .crashed, .restart, [class*="game-over"]'
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
        except:
            pass
        
        # SECONDARY METHOD: Frame stagnation detection (conservative threshold)
        # Only trigger if 30+ consecutive identical frames (1 second at 30fps)
        # This is much more conservative than the original 5 frames
        if self.current_frame is not None and len(self.previous_frames) > 0:
            frame_diff = np.mean(np.abs(self.current_frame.astype(float) - self.previous_frames[-1].astype(float)))
            
            if frame_diff < 1.0:
                self.identical_frame_count += 1
            else:
                self.identical_frame_count = 0
            
            # Only consider game over after 30 identical frames AND significant gameplay
            if self.identical_frame_count >= 30 and self.steps_taken > 60:
                return True
        
        return False

    def _get_score(self):
        try:
            score = self.page.evaluate("""
                () => {
                    // Check for score in text elements
                    const textElements = document.querySelectorAll('*');
                    for (let elem of textElements) {
                        const text = elem.textContent || '';
                        const match = text.match(/score[:\s]*(\d+)/i);
                        if (match) {
                            return parseInt(match[1]);
                        }
                    }
                    
                    // Check for score variable in window
                    if (typeof window.score !== 'undefined') {
                        return window.score;
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
        self.survival_steps = 0
        
        time.sleep(0.5)
        
        observation = self._get_canvas_screenshot()
        self.current_frame = observation
        
        return observation, {}

    def step(self, action):
        self.steps_taken += 1
        self.survival_steps += 1
        
        # Execute action
        if action == 1:
            self.page.keyboard.down("Space")
            time.sleep(0.05)
            self.page.keyboard.up("Space")
        
        time.sleep(0.033)
        
        observation = self._get_canvas_screenshot()
        
        # Update frame history
        self.previous_frames.append(self.current_frame)
        if len(self.previous_frames) > 10:
            self.previous_frames.pop(0)
        self.current_frame = observation
        
        # Get current score
        current_score = self._get_score()
        score_delta = current_score - self.last_score
        self.last_score = current_score
        
        # REWARD FUNCTION:
        # 1. Base survival reward: small positive for staying alive
        # 2. Score increase reward: larger reward for scoring points
        # 3. Death penalty: negative reward for game over
        # Scaled to approximately -1.0 to +1.0 range
        
        reward = 0.0
        
        # Survival reward: +0.01 per step (encourages staying alive)
        # This accumulates to meaningful rewards over time
        reward += 0.01
        
        # Score increase reward: +0.5 per point scored
        # This provides strong signal when player successfully dodges obstacles
        if score_delta > 0:
            reward += score_delta * 0.5
        
        # Check for game over
        terminated = self._is_game_over()
        
        if terminated:
            # Death penalty: -1.0 for dying
            # This creates clear distinction between success and failure
            reward = -1.0
        
        # Optional: Bonus for long survival (milestone rewards)
        # Provides intermediate goals during training
        if self.survival_steps % 100 == 0 and self.survival_steps > 0 and not terminated:
            reward += 0.1
        
        truncated = False
        if self.steps_taken > 10000:
            truncated = True
        
        info = {
            "score": current_score,
            "steps": self.steps_taken,
            "identical_frames": self.identical_frame_count,
            "survival_steps": self.survival_steps
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