import gymnasium as gym
from gymnasium import spaces
import numpy as np
from playwright.sync_api import sync_playwright
from PIL import Image
import io
import time

class DinoGameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, headless=True):
        super().__init__()
        
        self.render_mode = render_mode
        self.headless = headless if render_mode != "human" else False
        
        # Action space: 0=do nothing, 1=jump (space), 2=duck (down arrow)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: grayscale screenshot resized to 84x84
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )
        
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        
        self.game_url = "https://dinosaur-game.io/"
        self.current_score = 0
        self.previous_score = 0
        self.steps_taken = 0
        self.game_over = False
        self.last_action_time = 0
        
    def _start_browser(self):
        """Initialize Playwright and browser"""
        if self.playwright is None:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=self.headless)
            self.context = self.browser.new_context(
                viewport={"width": 800, "height": 400}
            )
            self.page = self.context.new_page()
            
    def _load_game(self):
        """Load the game page and start it"""
        self.page.goto(self.game_url, wait_until="networkidle", timeout=30000)
        time.sleep(2)
        
        # Try to find and click the game canvas or start button
        try:
            # Look for common game start elements
            canvas = self.page.query_selector("canvas")
            if canvas:
                canvas.click()
                time.sleep(0.5)
        except Exception:
            pass
        
        # Press space to start the game
        self.page.keyboard.press("Space")
        time.sleep(0.5)
        
    def _get_screenshot(self):
        """Capture screenshot and convert to numpy array"""
        screenshot_bytes = self.page.screenshot()
        image = Image.open(io.BytesIO(screenshot_bytes))
        
        # Convert to grayscale and resize
        image = image.convert("L")
        image = image.resize((84, 84), Image.Resampling.LANCZOS)
        
        return np.array(image, dtype=np.uint8)
    
    def _get_score(self):
        """Extract current score from the page"""
        try:
            # Try multiple selectors for score
            score_selectors = [
                ".score",
                "#score",
                "[class*='score']",
                "[id*='score']"
            ]
            
            for selector in score_selectors:
                try:
                    score_element = self.page.query_selector(selector)
                    if score_element:
                        score_text = score_element.inner_text()
                        # Extract numbers from text
                        score_value = ''.join(filter(str.isdigit, score_text))
                        if score_value:
                            return int(score_value)
                except Exception:
                    continue
            
            # If no score found in DOM, try to detect from canvas or use steps as proxy
            return self.steps_taken
            
        except Exception:
            return self.steps_taken
    
    def _is_game_over(self):
        """Detect if game is over"""
        try:
            # Check for game over text or restart button
            game_over_selectors = [
                "text=/game over/i",
                "text=/restart/i",
                "text=/play again/i",
                "[class*='game-over']",
                "[class*='gameover']",
                ".restart-button",
                "#restart"
            ]
            
            for selector in game_over_selectors:
                try:
                    element = self.page.query_selector(selector)
                    if element and element.is_visible():
                        return True
                except Exception:
                    continue
            
            # Check if score has stopped increasing for too long
            current_score = self._get_score()
            if self.steps_taken > 50 and current_score == self.previous_score and self.steps_taken % 10 == 0:
                # Additional check: might be game over
                pass
            
            return False
            
        except Exception:
            return False
    
    def _execute_action(self, action):
        """Execute the given action"""
        current_time = time.time()
        
        # Prevent actions too close together
        if current_time - self.last_action_time < 0.05:
            return
        
        if action == 1:  # Jump
            self.page.keyboard.press("Space")
        elif action == 2:  # Duck
            self.page.keyboard.down("ArrowDown")
            time.sleep(0.1)
            self.page.keyboard.up("ArrowDown")
        # action == 0: do nothing
        
        self.last_action_time = current_time
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        if self.page is None:
            self._start_browser()
            self._load_game()
        else:
            # Restart the game
            try:
                # Try to find and click restart button
                restart_selectors = [
                    "text=/restart/i",
                    "text=/play again/i",
                    ".restart-button",
                    "#restart"
                ]
                
                restarted = False
                for selector in restart_selectors:
                    try:
                        restart_button = self.page.query_selector(selector)
                        if restart_button and restart_button.is_visible():
                            restart_button.click()
                            restarted = True
                            break
                    except Exception:
                        continue
                
                if not restarted:
                    # Reload the page
                    self.page.reload(wait_until="networkidle")
                    time.sleep(1)
                    self._load_game()
                else:
                    time.sleep(0.5)
                    
            except Exception:
                # If anything fails, reload the page
                self.page.reload(wait_until="networkidle")
                time.sleep(1)
                self._load_game()
        
        self.current_score = 0
        self.previous_score = 0
        self.steps_taken = 0
        self.game_over = False
        
        time.sleep(0.5)
        observation = self._get_screenshot()
        
        return observation, {}
    
    def step(self, action):
        """Execute one step in the environment"""
        self._execute_action(action)
        
        # Small delay to allow game to update
        time.sleep(0.033)  # ~30 FPS
        
        self.steps_taken += 1
        
        # Get new state
        observation = self._get_screenshot()
        self.previous_score = self.current_score
        self.current_score = self._get_score()
        
        # Check if game is over
        self.game_over = self._is_game_over()
        
        # Calculate reward
        reward = 0.1  # Small reward for survival
        
        # Reward for score increase
        score_delta = self.current_score - self.previous_score
        if score_delta > 0:
            reward += score_delta * 0.5
        
        # Penalty for game over
        if self.game_over:
            reward = -10
        
        terminated = self.game_over
        truncated = False
        
        info = {
            "score": self.current_score,
            "steps": self.steps_taken
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "rgb_array":
            screenshot_bytes = self.page.screenshot()
            image = Image.open(io.BytesIO(screenshot_bytes))
            return np.array(image)
        elif self.render_mode == "human":
            # Browser is already visible in human mode
            pass
    
    def close(self):
        """Clean up resources"""
        if self.page:
            self.page.close()
            self.page = None
        if self.context:
            self.context.close()
            self.context = None
        if self.browser:
            self.browser.close()
            self.browser = None
        if self.playwright:
            self.playwright.stop()
            self.playwright = None