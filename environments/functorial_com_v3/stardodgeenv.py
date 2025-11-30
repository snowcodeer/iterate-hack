import gymnasium as gym
from gymnasium import spaces
import numpy as np
from playwright.sync_api import sync_playwright
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

        # Action space: 0 = release space, 1 = hold space (matches game mechanic)
        self.action_space = spaces.Discrete(2)

        # Reduced observation space for faster processing (grayscale, downsampled)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(125, 200, 1), dtype=np.uint8
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
        self.time_alive = 0
        self.last_score = 0
        self.current_score = 0
        self.game_started = False
        self.space_held = False

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

            # Click to start game
            canvas.click()
            time.sleep(0.5)

        except Exception as e:
            print(f"Error locating canvas: {e}")
            self.canvas_bbox = {"x": 240, "y": 100, "width": 800, "height": 500}

    def _get_canvas_screenshot(self):
        """Capture game frame - downsampled grayscale for faster processing."""
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
        # Convert to grayscale and downsample for efficiency
        image = image.convert("L")
        image = image.resize((200, 125), Image.Resampling.LANCZOS)
        return np.array(image)[:, :, np.newaxis]  # Add channel dimension

    def _extract_game_state(self):
        """Extract game state directly from JavaScript - more reliable than pixel parsing."""
        try:
            state = self.page.evaluate("""
                () => {
                    const result = {};

                    // Try to extract score from common variable names
                    if (typeof score !== 'undefined') {
                        result.score = score;
                    } else if (window.game && typeof window.game.score !== 'undefined') {
                        result.score = window.game.score;
                    } else {
                        result.score = 0;
                    }

                    // Try to extract game over state
                    if (typeof gameOver !== 'undefined') {
                        result.gameOver = gameOver;
                    } else if (window.game && typeof window.game.gameOver !== 'undefined') {
                        result.gameOver = window.game.gameOver;
                    } else {
                        result.gameOver = false;
                    }

                    return result;
                }
            """)
            return state
        except Exception as e:
            return {"score": 0, "gameOver": False}

    def _is_game_over(self):
        """
        Conservative game over detection with grace period and multiple methods.
        FIXED: Resets frame counter during grace period to prevent false positives.
        """
        # GRACE PERIOD: Skip checks during startup (30 steps = 1 second at 30fps)
        if self.steps_taken < 30:
            # CRITICAL FIX: Reset counter to prevent accumulation during startup
            self.identical_frame_count = 0
            return False

        # METHOD 1: JavaScript-based state extraction (most reliable)
        try:
            game_state = self._extract_game_state()
            if game_state.get("gameOver", False):
                return True
        except:
            pass

        # METHOD 2: Text-based detection
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
                        'play again'
                    ];

                    for (const phrase of gameOverPhrases) {
                        if (bodyText.includes(phrase)) {
                            return true;
                        }
                    }
                    return false;
                }
            """)

            if game_over_text:
                return True
        except:
            pass

        # METHOD 3: Frame freeze detection (VERY CONSERVATIVE - 60 frames = 2 seconds)
        if self.current_frame is not None and len(self.previous_frames) > 0:
            frame_diff = np.mean(np.abs(
                self.current_frame.astype(float) - self.previous_frames[-1].astype(float)
            ))

            # Higher threshold (5.0) to allow for small animations
            if frame_diff < 5.0:
                self.identical_frame_count += 1
            else:
                self.identical_frame_count = 0

            # Require 60 consecutive identical frames (2 seconds at 30fps)
            if self.identical_frame_count >= 60:
                return True

        return False

    def _get_score(self):
        """Extract score from JavaScript or UI."""
        game_state = self._extract_game_state()
        return game_state.get("score", 0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.page:
            try:
                self.page.reload(wait_until="networkidle", timeout=10000)
            except:
                self.page.reload()

            time.sleep(2)

            # Click canvas to start/focus
            try:
                canvas = self.page.locator("#canvas")
                canvas.click()
                time.sleep(0.3)
            except:
                pass

        # Reset all state tracking variables
        self.steps_taken = 0
        self.time_alive = 0
        self.last_score = 0
        self.current_score = 0
        self.identical_frame_count = 0
        self.previous_frames = []
        self.game_started = False
        self.space_held = False

        # Release space key if it was held
        try:
            self.page.keyboard.up("Space")
        except:
            pass

        time.sleep(0.5)

        observation = self._get_canvas_screenshot()
        self.current_frame = observation

        return observation, {}

    def step(self, action):
        self.steps_taken += 1
        self.time_alive += 1

        # ACTION EXECUTION: Hold space or release space
        # Action 0 = release space, Action 1 = hold space
        # This matches "Hold Space to change direction" mechanic
        if action == 1 and not self.space_held:
            # Start holding space
            self.page.keyboard.down("Space")
            self.space_held = True
        elif action == 0 and self.space_held:
            # Release space
            self.page.keyboard.up("Space")
            self.space_held = False

        # Wait for frame update (30 FPS)
        time.sleep(0.033)

        # Get observation
        observation = self._get_canvas_screenshot()

        # Update frame history for game over detection
        self.previous_frames.append(self.current_frame)
        if len(self.previous_frames) > 5:
            self.previous_frames.pop(0)
        self.current_frame = observation

        # Get current score
        self.current_score = self._get_score()
        score_delta = self.current_score - self.last_score
        self.last_score = self.current_score

        # ===== REWARD FUNCTION =====
        # Primary: Survival time (with increasing difficulty bonus)
        # Secondary: Score increases (if game tracks score)
        # Penalty: Death (but not dominant)
        # Scale: Approximately -1.0 to +1.0 range

        reward = 0.0

        # SURVIVAL REWARD: Base reward that increases with time
        # Longer survival = harder = more reward per frame
        # Scales from 0.01 to ~0.05 over 5000 steps
        survival_reward = 0.01 + (self.time_alive / 5000.0)
        reward += survival_reward

        # SCORE BONUS: Reward for score increases
        if score_delta > 0:
            reward += score_delta * 0.1

        # Check for game over
        terminated = self._is_game_over()

        # DEATH PENALTY: Normalized to -1.0
        if terminated:
            reward = -1.0
            # Release space key on death
            if self.space_held:
                self.page.keyboard.up("Space")
                self.space_held = False

        # Truncate very long episodes
        truncated = False
        if self.steps_taken > 10000:
            truncated = True
            if self.space_held:
                self.page.keyboard.up("Space")
                self.space_held = False

        info = {
            "score": self.current_score,
            "steps": self.steps_taken,
            "time_alive": self.time_alive,
            "identical_frames": self.identical_frame_count,
            "space_held": self.space_held
        }

        # Debug logging every 50 steps
        if self.steps_taken % 50 == 0:
            print(f"[Step {self.steps_taken}] Score: {self.current_score}, Reward: {reward:.4f}, Space: {self.space_held}")

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.current_frame
        elif self.render_mode == "human":
            return None
        return None

    def close(self):
        # Release space key if held
        if self.space_held and self.page:
            try:
                self.page.keyboard.up("Space")
            except:
                pass

        if self.page:
            self.page.close()
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
