import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import io
import time
from typing import Optional, Tuple
import hashlib

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

from playwright.sync_api import sync_playwright, Page, Browser, Playwright

class StarDodgeCloneEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, headless: bool = True):
        super().__init__()
        
        self.render_mode = render_mode
        self.headless = headless
        self.game_url = "https://functorial.com/star-dodge-clone/"
        
        self.action_space = spaces.Discrete(2)  # 0: do nothing, 1: hold space
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(500, 800, 3), dtype=np.uint8
        )
        
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.game_element = None
        
        self.current_score = 0
        self.last_score = 0
        self.steps_taken = 0
        self.last_frame_hash = None
        self.identical_frame_count = 0
        self.game_over = False
        self.last_action = 0
        self.survival_time = 0
        self.space_pressed = False
        
        self._start_browser()

    def _start_browser(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        context = self.browser.new_context(viewport={"width": 1280, "height": 900})
        self.page = context.new_page()
        
        self.page.goto(self.game_url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(3)
        
        try:
            self.game_element = self.page.locator("canvas#canvas").first
            if self.game_element.is_visible():
                bbox = self.game_element.bounding_box()
                if bbox and bbox['width'] > 300 and bbox['height'] > 200:
                    pass
                else:
                    self.game_element = None
        except:
            self.game_element = None
        
        if not self.game_element:
            canvases = self.page.locator("canvas").all()
            largest = None
            largest_area = 0
            for c in canvases:
                try:
                    if c.is_visible():
                        bbox = c.bounding_box()
                        if bbox:
                            area = bbox['width'] * bbox['height']
                            if area > largest_area:
                                largest_area = area
                                largest = c
                except:
                    continue
            self.game_element = largest
        
        self._hide_overlays()
        
        if self.game_element:
            try:
                self.game_element.scroll_into_view_if_needed()
                self.game_element.click()
            except:
                pass
        
        time.sleep(1)

    def _hide_overlays(self):
        try:
            self.page.evaluate('''() => {
                const hide = [
                    '[class*="ad"]', '[id*="ad"]', 'ins.adsbygoogle',
                    '[class*="social"]', '[class*="share"]',
                    '[class*="cookie"]', '[class*="consent"]',
                    '.modal', '.popup'
                ];
                hide.forEach(sel => {
                    document.querySelectorAll(sel).forEach(el => {
                        if (!el.closest('canvas')) {
                            el.style.display = 'none';
                        }
                    });
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
                            'x': bbox['x'],
                            'y': bbox['y'],
                            'width': bbox['width'],
                            'height': bbox['height']
                        }
                    )
                    image = Image.open(io.BytesIO(screenshot_bytes))
                    return np.array(image)
            except:
                pass
        
        screenshot_bytes = self.page.screenshot()
        image = Image.open(io.BytesIO(screenshot_bytes))
        image = image.resize((800, 500))
        return np.array(image)

    def _get_score(self) -> int:
        try:
            score = self.page.evaluate('''() => {
                const canvas = document.getElementById('canvas');
                const ctx = canvas.getContext('2d');
                const imageData = ctx.getImageData(0, 0, 200, 100);
                const data = imageData.data;
                let whitePixels = 0;
                for (let i = 0; i < data.length; i += 4) {
                    if (data[i] > 200 && data[i+1] > 200 && data[i+2] > 200) {
                        whitePixels++;
                    }
                }
                return whitePixels;
            }''')
            return score if score else 0
        except:
            return 0

    def _is_game_over(self, observation: np.ndarray) -> bool:
        frame_hash = hashlib.md5(observation.tobytes()).hexdigest()
        
        if frame_hash == self.last_frame_hash:
            self.identical_frame_count += 1
        else:
            self.identical_frame_count = 0
        
        self.last_frame_hash = frame_hash
        
        if self.identical_frame_count >= 8:
            return True
        
        try:
            red_pixels = np.sum((observation[:, :, 0] > 200) & 
                               (observation[:, :, 1] < 100) & 
                               (observation[:, :, 2] < 100))
            
            if red_pixels > 5000:
                return True
        except:
            pass
        
        try:
            game_over_text = self.page.evaluate('''() => {
                return document.body.innerText.toLowerCase().includes('game over');
            }''')
            if game_over_text:
                return True
        except:
            pass
        
        return False

    def _calculate_reward(self) -> float:
        reward = 0.0
        
        reward += 0.02
        
        score_delta = self.current_score - self.last_score
        if score_delta > 0:
            reward += score_delta * 0.001
        
        self.survival_time += 1
        if self.survival_time % 50 == 0:
            reward += 0.1
        
        if self.game_over:
            reward = -1.0
        
        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        if self.space_pressed:
            try:
                self.page.keyboard.up("Space")
                self.space_pressed = False
            except:
                pass
        
        try:
            self.page.reload(wait_until="domcontentloaded", timeout=30000)
            time.sleep(2)
            
            if self.game_element:
                try:
                    self.game_element.click()
                except:
                    pass
            
            time.sleep(0.5)
        except:
            self.close()
            self._start_browser()
        
        self.current_score = 0
        self.last_score = 0
        self.steps_taken = 0
        self.last_frame_hash = None
        self.identical_frame_count = 0
        self.game_over = False
        self.last_action = 0
        self.survival_time = 0
        
        observation = self._get_canvas_screenshot()
        info = {"score": self.current_score, "survival_time": self.survival_time}
        
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.steps_taken += 1
        
        try:
            if action == 1 and not self.space_pressed:
                self.page.keyboard.down("Space")
                self.space_pressed = True
            elif action == 0 and self.space_pressed:
                self.page.keyboard.up("Space")
                self.space_pressed = False
        except:
            pass
        
        time.sleep(0.03)
        
        observation = self._get_canvas_screenshot()
        
        self.last_score = self.current_score
        self.current_score = self._get_score()
        
        self.game_over = self._is_game_over(observation)
        
        reward = self._calculate_reward()
        
        terminated = self.game_over
        truncated = self.steps_taken >= 10000
        
        info = {
            "score": self.current_score,
            "survival_time": self.survival_time,
            "steps": self.steps_taken
        }
        
        self.last_action = action
        
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_canvas_screenshot()
        return None

    def close(self):
        if self.space_pressed:
            try:
                self.page.keyboard.up("Space")
            except:
                pass
        
        if self.page:
            try:
                self.page.close()
            except:
                pass
        if self.browser:
            try:
                self.browser.close()
            except:
                pass
        if self.playwright:
            try:
                self.playwright.stop()
            except:
                pass