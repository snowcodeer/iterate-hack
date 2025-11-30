import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import io
import time
from collections import deque

# Apply nest_asyncio to allow Playwright sync API
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

class DinosaurGameEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, headless=False):
        super().__init__()
        
        self.render_mode = render_mode
        self.headless = headless
        self.game_url = "https://dinosaur-game.io/"
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(150, 600, 3), dtype=np.uint8
        )
        
        self.browser = None
        self.context = None
        self.page = None
        self.playwright = None
        self.game_element = None
        self.game_bbox = None
        
        self.score = 0
        self.prev_score = 0
        self.steps = 0
        self.last_frames = deque(maxlen=5)
        self.game_started = False
        
    def _start_browser(self):
        if self.playwright is None:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=self.headless)
            self.context = self.browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            self.page = self.context.new_page()
            
            self.page.goto(self.game_url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(2)
            
            self._close_popups()
            self._hide_overlays()
            self._detect_game_element()
            
            if self.game_element:
                self.game_element.scroll_into_view_if_needed()
                time.sleep(0.5)
                try:
                    self.game_element.click(timeout=1000)
                except:
                    pass
            else:
                try:
                    self.page.click('body', timeout=1000)
                except:
                    pass
                    
            time.sleep(1)
    
    def _close_popups(self):
        popup_selectors = [
            '[class*="cookie"]', '[class*="consent"]', '[id*="cookie"]',
            '.modal-close', '.popup-close', '[aria-label*="close"]',
            'button:has-text("Accept")', 'button:has-text("OK")',
            'button:has-text("Close")', '[class*="dismiss"]'
        ]
        
        for selector in popup_selectors:
            try:
                elements = self.page.locator(selector).all()
                for elem in elements:
                    if elem.is_visible():
                        elem.click(timeout=500)
                        time.sleep(0.3)
            except:
                continue
    
    def _hide_overlays(self):
        try:
            self.page.evaluate('''() => {
                const hideSelectors = [
                    '[class*="ad-"]', '[id*="ad-"]', '[class*="advertisement"]',
                    'ins.adsbygoogle', '[data-ad-slot]', '[id*="google_ads"]',
                    '[class*="social-share"]', '[class*="share-button"]',
                    '.fb-like', '.twitter-share', '[class*="follow-us"]',
                    'header:not(.game-header)', 'footer', 'nav',
                    '[class*="sidebar"]', '[class*="navigation"]',
                    '[class*="cookie-banner"]', '[class*="consent-banner"]',
                    '[id*="consent"]', '.overlay:not(#game):not(.game-overlay)'
                ];
                
                hideSelectors.forEach(sel => {
                    try {
                        document.querySelectorAll(sel).forEach(el => {
                            const isGameElement = el.closest('#game') || 
                                                 el.closest('.game-container') ||
                                                 el.closest('[class*="game-"]') ||
                                                 el.id === 'game';
                            if (!isGameElement) {
                                el.style.display = 'none';
                                el.style.visibility = 'hidden';
                            }
                        });
                    } catch(e) {}
                });
            }''')
        except:
            pass
    
    def _detect_game_element(self):
        game_selectors = [
            'canvas.runner-canvas',
            'canvas#runner',
            'canvas.game-canvas',
            '#game canvas',
            '.game-container canvas',
            'canvas[id*="game"]',
            'iframe[src*="game"]',
            'iframe.game-frame'
        ]
        
        for selector in game_selectors:
            try:
                element = self.page.locator(selector).first
                if element.is_visible():
                    bbox = element.bounding_box()
                    if bbox and bbox['width'] > 400 and bbox['height'] > 150:
                        self.game_element = element
                        self.game_bbox = bbox
                        return
            except:
                continue
        
        try:
            canvases = self.page.locator('canvas').all()
            largest = None
            largest_area = 0
            
            for canvas in canvases:
                try:
                    if not canvas.is_visible():
                        continue
                    
                    bbox = canvas.bounding_box()
                    if not bbox:
                        continue
                    
                    class_name = canvas.get_attribute('class') or ''
                    id_name = canvas.get_attribute('id') or ''
                    
                    if 'ad' in class_name.lower() or 'ad' in id_name.lower():
                        continue
                    
                    area = bbox['width'] * bbox['height']
                    if area > largest_area and bbox['width'] > 400 and bbox['height'] > 100:
                        largest_area = area
                        largest = canvas
                        self.game_bbox = bbox
                except:
                    continue
            
            if largest:
                self.game_element = largest
        except:
            pass
    
    def _get_canvas_screenshot(self):
        # Try to refresh game bbox if we have an element
        try:
            if self.game_element:
                bbox = self.game_element.bounding_box()
                if bbox and bbox['width'] > 100 and bbox['height'] > 50:
                    # Validate bbox is within viewport
                    if bbox['x'] >= 0 and bbox['y'] >= 0:
                        self.game_bbox = bbox
        except:
            pass

        # Take screenshot with valid clip or full page
        try:
            if self.game_bbox and self.game_bbox['x'] >= 0 and self.game_bbox['y'] >= 0:
                screenshot_bytes = self.page.screenshot(
                    clip={
                        'x': max(0, self.game_bbox['x']),
                        'y': max(0, self.game_bbox['y']),
                        'width': self.game_bbox['width'],
                        'height': self.game_bbox['height']
                    }
                )
            else:
                screenshot_bytes = self.page.screenshot()
        except:
            # Fallback to full page screenshot
            screenshot_bytes = self.page.screenshot()

        image = Image.open(io.BytesIO(screenshot_bytes))
        image = image.convert('RGB')
        image = image.resize((600, 150), Image.Resampling.LANCZOS)
        obs = np.array(image)

        if obs.shape != (150, 600, 3):
            obs = np.zeros((150, 600, 3), dtype=np.uint8)

        return obs
    
    def _start_game(self):
        try:
            self.page.keyboard.press('Space')
            time.sleep(0.1)
            self.game_started = True
        except:
            pass
    
    def _get_score(self):
        try:
            score_text = self.page.evaluate('''() => {
                const selectors = [
                    '.score', '#score', '[class*="score"]',
                    '[id*="score"]', '.runner-score'
                ];
                
                for (let sel of selectors) {
                    const elem = document.querySelector(sel);
                    if (elem && elem.textContent) {
                        return elem.textContent.trim();
                    }
                }
                
                const canvases = document.querySelectorAll('canvas');
                for (let canvas of canvases) {
                    const ctx = canvas.getContext('2d');
                    if (ctx) {
                        return 'canvas';
                    }
                }
                
                return null;
            }''')
            
            if score_text and score_text != 'canvas':
                score_num = ''.join(filter(str.isdigit, score_text))
                if score_num:
                    return int(score_num)
        except:
            pass
        
        return self.score
    
    def _is_game_over(self):
        try:
            game_over_detected = self.page.evaluate('''() => {
                const gameOverTexts = ['game over', 'gameover', 'game-over', 'retry', 'restart', 'play again'];
                
                const allText = document.body.innerText.toLowerCase();
                for (let text of gameOverTexts) {
                    if (allText.includes(text)) {
                        return true;
                    }
                }
                
                const buttons = document.querySelectorAll('button, [role="button"], .button');
                for (let btn of buttons) {
                    const btnText = btn.textContent.toLowerCase();
                    if (btnText.includes('restart') || btnText.includes('retry') || btnText.includes('play again')) {
                        if (window.getComputedStyle(btn).display !== 'none') {
                            return true;
                        }
                    }
                }
                
                return false;
            }''')
            
            if game_over_detected:
                return True
        except:
            pass
        
        current_frame = self._get_canvas_screenshot()
        self.last_frames.append(current_frame)
        
        if len(self.last_frames) >= 5:
            frames_list = list(self.last_frames)
            all_same = True
            for i in range(1, len(frames_list)):
                if not np.array_equal(frames_list[0], frames_list[i]):
                    all_same = False
                    break
            
            if all_same and self.steps > 10:
                return True
        
        return False
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.page is None:
            self._start_browser()
        else:
            try:
                self.page.reload(wait_until="domcontentloaded", timeout=10000)
                time.sleep(1)
                self._close_popups()
                self._hide_overlays()
                self._detect_game_element()
            except:
                self.close()
                self._start_browser()
        
        self.score = 0
        self.prev_score = 0
        self.steps = 0
        self.last_frames.clear()
        self.game_started = False
        
        self._start_game()
        time.sleep(0.5)
        
        obs = self._get_canvas_screenshot()
        info = {'score': self.score}
        
        return obs, info
    
    def step(self, action):
        if self.page is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        if action == 1:
            self.page.keyboard.press('Space')
        elif action == 2:
            self.page.keyboard.press('ArrowDown')
        
        time.sleep(0.033)
        
        self.steps += 1
        
        obs = self._get_canvas_screenshot()
        
        self.prev_score = self.score
        self.score = self._get_score()
        score_delta = max(0, self.score - self.prev_score)
        
        reward = 0.1
        if score_delta > 0:
            reward += score_delta * 10
        
        terminated = self._is_game_over()
        
        if terminated:
            reward = -10.0
        
        truncated = False
        if self.steps > 10000:
            truncated = True
        
        info = {
            'score': self.score,
            'steps': self.steps
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == 'rgb_array':
            return self._get_canvas_screenshot()
        return None
    
    def close(self):
        if self.page:
            try:
                self.page.close()
            except:
                pass
            self.page = None
        
        if self.context:
            try:
                self.context.close()
            except:
                pass
            self.context = None
        
        if self.browser:
            try:
                self.browser.close()
            except:
                pass
            self.browser = None
        
        if self.playwright:
            try:
                self.playwright.stop()
            except:
                pass
            self.playwright = None