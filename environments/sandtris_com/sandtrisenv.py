import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import io
import time
import hashlib
from typing import Optional, Tuple

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

from playwright.sync_api import sync_playwright, Page, Browser, Playwright

class SandtrisEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, headless: bool = False):
        super().__init__()
        
        self.render_mode = render_mode
        self.headless = headless
        self.game_url = "https://sandtris.com/"
        
        # Action space: 0=nothing, 1=left, 2=right, 3=rotate, 4=down
        self.action_space = spaces.Discrete(5)
        
        # Observation space: RGB screenshot
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(600, 400, 3), dtype=np.uint8
        )
        
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.game_element = None
        
        self.steps_taken = 0
        self.last_frame_hash = None
        self.identical_frame_count = 0
        self.last_score = 0
        self.current_score = 0
        self.last_observation = None
        self.game_started = False
        
        self._start_browser()

    def _start_browser(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        context = self.browser.new_context(viewport={"width": 1280, "height": 800})
        self.page = context.new_page()
        
        self.page.goto(self.game_url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(3)
        
        # Close any dialogs/start screens
        try:
            self.page.evaluate('''() => {
                // Close the start dialog
                const startPage = document.querySelector('#startpage');
                if (startPage && startPage.open) {
                    startPage.close();
                }
                // Try clicking start button if exists
                const startBtn = document.querySelector('button.greenbutton');
                if (startBtn) {
                    startBtn.click();
                }
            }''')
            time.sleep(1)
        except:
            pass
        
        # Find the game canvas
        try:
            canvas = self.page.locator('canvas').first
            if canvas.is_visible():
                bbox = canvas.bounding_box()
                if bbox and bbox['width'] > 300 and bbox['height'] > 300:
                    self.game_element = canvas
        except:
            pass
        
        if not self.game_element:
            canvases = self.page.locator('canvas').all()
            largest = None
            largest_area = 0
            for c in canvases:
                try:
                    bbox = c.bounding_box()
                    if bbox:
                        area = bbox['width'] * bbox['height']
                        if area > largest_area:
                            largest_area = area
                            largest = c
                except:
                    pass
            self.game_element = largest
        
        self._hide_overlays()
        
        # Click on canvas to focus
        if self.game_element:
            try:
                self.game_element.click()
                time.sleep(0.5)
            except:
                pass

    def _hide_overlays(self):
        try:
            self.page.evaluate('''() => {
                const hide = [
                    '.adsbox', '[class*="ad"]', '[id*="ad"]', 'ins.adsbygoogle',
                    '[class*="social"]', '[class*="share"]',
                    '[class*="cookie"]', '[class*="consent"]'
                ];
                hide.forEach(sel => {
                    document.querySelectorAll(sel).forEach(el => {
                        el.style.display = 'none';
                    });
                });
                
                // Hide dialogs
                const dialogs = document.querySelectorAll('dialog');
                dialogs.forEach(d => {
                    if (d.id !== 'pausepage') {
                        try { d.close(); } catch(e) {}
                        d.style.display = 'none';
                    }
                });
            }''')
        except:
            pass

    def _get_canvas_screenshot(self) -> np.ndarray:
        if self.game_element:
            try:
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
                    image = Image.open(io.BytesIO(screenshot_bytes))
                    image = image.resize((400, 600), Image.BILINEAR)
                    return np.array(image)
            except:
                pass
        
        screenshot_bytes = self.page.screenshot()
        image = Image.open(io.BytesIO(screenshot_bytes))
        image = image.resize((400, 600), Image.BILINEAR)
        return np.array(image)

    def _extract_score(self) -> int:
        try:
            score = self.page.evaluate('''() => {
                const scoreText = document.body.innerText;
                const match = scoreText.match(/SCORE[:\\s]+(\\d+)/i);
                if (match) return parseInt(match[1]);
                
                // Try to find score element
                const elements = document.querySelectorAll('*');
                for (let el of elements) {
                    const text = el.innerText || '';
                    if (text.includes('SCORE') || text.includes('Score')) {
                        const m = text.match(/\\d+/);
                        if (m) return parseInt(m[0]);
                    }
                }
                return 0;
            }''')
            return score if score else 0
        except:
            return 0

    def _is_game_over(self, observation: np.ndarray) -> bool:
        if self.steps_taken < 50:
            return False
        
        frame_hash = hashlib.md5(observation.tobytes()).hexdigest()
        if frame_hash == self.last_frame_hash:
            self.identical_frame_count += 1
        else:
            self.identical_frame_count = 0
        self.last_frame_hash = frame_hash
        
        if self.identical_frame_count >= 45:
            return True
        
        try:
            game_over = self.page.evaluate('''() => {
                const text = document.body.innerText.toLowerCase();
                return text.includes('game over') || text.includes('try again') || 
                       text.includes('restart') || text.includes('play again');
            }''')
            if game_over:
                return True
        except:
            pass
        
        return False

    def _calculate_reward(self) -> float:
        reward = 0.0
        
        # Survival reward
        reward += 0.01
        
        # Score-based reward
        score_delta = self.current_score - self.last_score
        if score_delta > 0:
            reward += score_delta * 0.5
        
        self.last_score = self.current_score
        
        return reward

    def _send_action(self, action: int):
        if not self.game_element:
            return
        
        try:
            if action == 1:  # Left
                self.page.keyboard.press('ArrowLeft')
            elif action == 2:  # Right
                self.page.keyboard.press('ArrowRight')
            elif action == 3:  # Rotate
                self.page.keyboard.press('ArrowUp')
            elif action == 4:  # Down faster
                self.page.keyboard.press('ArrowDown')
        except:
            pass

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        if self.page:
            try:
                self.page.reload(wait_until="domcontentloaded", timeout=30000)
                time.sleep(2)
                
                # Close start dialog and start game
                self.page.evaluate('''() => {
                    const startPage = document.querySelector('#startpage');
                    if (startPage && startPage.open) {
                        startPage.close();
                    }
                    const startBtn = document.querySelector('button.greenbutton');
                    if (startBtn) {
                        startBtn.click();
                    }
                    
                    // Call startGame if exists
                    if (typeof startGame === 'function') {
                        startGame();
                    }
                }''')
                time.sleep(1)
                
                self._hide_overlays()
                
                if self.game_element:
                    self.game_element.click()
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"Reset error: {e}")
        
        self.steps_taken = 0
        self.last_frame_hash = None
        self.identical_frame_count = 0
        self.last_score = 0
        self.current_score = 0
        self.game_started = True
        
        observation = self._get_canvas_screenshot()
        self.last_observation = observation
        
        return observation, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.steps_taken += 1
        
        self._send_action(action)
        
        time.sleep(0.05)
        
        observation = self._get_canvas_screenshot()
        self.last_observation = observation
        
        self.current_score = self._extract_score()
        
        reward = self._calculate_reward()
        
        terminated = self._is_game_over(observation)
        truncated = False
        
        if terminated:
            reward = -1.0
        
        info = {
            "score": self.current_score,
            "steps": self.steps_taken
        }
        
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.last_observation
        return None

    def close(self):
        if self.page:
            self.page.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()