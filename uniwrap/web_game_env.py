"""
Web Game Environment - Wraps browser-based games as Gymnasium environments.

Uses Playwright for browser automation and screen capture.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
import time

try:
    from playwright.sync_api import sync_playwright, Page, Browser
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


class WebGameEnv(gym.Env):
    """
    A Gymnasium environment that wraps a web-based game.

    This environment:
    - Opens a browser and navigates to the game URL
    - Captures screenshots as observations
    - Sends keyboard inputs as actions
    - Extracts game state (score, game over) via JavaScript or screen analysis
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        game_url: str,
        render_mode: Optional[str] = None,
        width: int = 600,
        height: int = 150,
        grayscale: bool = True,
        frame_skip: int = 1,
        game_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the web game environment.

        Args:
            game_url: URL of the game to play
            render_mode: "human" for visible browser, "rgb_array" for headless
            width: Screenshot width
            height: Screenshot height
            grayscale: Whether to convert screenshots to grayscale
            frame_skip: Number of frames to skip between actions
            game_config: Game-specific configuration (start_key, action_keys, etc.)
        """
        super().__init__()

        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright is required. Install with: pip install playwright && playwright install chromium")

        self.game_url = game_url
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.frame_skip = frame_skip
        self.game_config = game_config or {}

        # Default game config
        self.start_key = self.game_config.get('start_key', 'Space')
        self.action_keys = self.game_config.get('action_keys', ['Space', 'ArrowDown'])
        self.score_js = self.game_config.get('score_js', None)
        self.game_over_js = self.game_config.get('game_over_js', None)
        self.restart_js = self.game_config.get('restart_js', None)

        # Action space: discrete actions (0=noop, 1+=action_keys)
        self.action_space = spaces.Discrete(len(self.action_keys) + 1)

        # Observation space: screenshot
        if grayscale:
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(height, width, 1),
                dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(height, width, 3),
                dtype=np.uint8
            )

        # Browser state
        self.playwright = None
        self.browser = None
        self.page = None
        self.score = 0
        self.steps = 0
        self.last_screenshot = None

    def _start_browser(self):
        """Start the browser and navigate to the game."""
        if self.playwright is None:
            self.playwright = sync_playwright().start()

        headless = self.render_mode != "human"
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.page = self.browser.new_page(viewport={"width": self.width + 100, "height": self.height + 100})

        # Navigate to game
        self.page.goto(self.game_url, wait_until="networkidle")
        time.sleep(1)  # Wait for game to fully load

        # Handle cookie consent popups
        self._handle_consent_popups()

        # Click to focus the game area
        self.page.click("body")
        time.sleep(0.5)

    def _handle_consent_popups(self):
        """Try to dismiss common cookie consent popups."""
        consent_selectors = [
            # Common consent button selectors
            'button:has-text("Consent")',
            'button:has-text("Accept")',
            'button:has-text("Accept all")',
            'button:has-text("Accept All")',
            'button:has-text("I agree")',
            'button:has-text("Agree")',
            'button:has-text("OK")',
            'button:has-text("Got it")',
            '[class*="consent"] button',
            '[class*="cookie"] button',
            '[id*="consent"] button',
            '[id*="cookie"] button',
            '.fc-cta-consent',  # Funding Choices
            '#onetrust-accept-btn-handler',  # OneTrust
            '.cc-btn.cc-dismiss',  # Cookie Consent
        ]

        for selector in consent_selectors:
            try:
                button = self.page.locator(selector).first
                if button.is_visible(timeout=500):
                    button.click()
                    time.sleep(0.5)
                    break
            except:
                continue

        # Also try clicking any visible "Consent" text
        try:
            self.page.get_by_role("button", name="Consent").click(timeout=1000)
            time.sleep(0.5)
        except:
            pass

        # Hide annoying overlays (GitHub ribbons, ads, etc.)
        self._hide_overlays()

    def _hide_overlays(self):
        """Hide distracting overlay elements like GitHub ribbons, ads, etc."""
        try:
            self.page.evaluate("""
                () => {
                    // Common overlay selectors to hide
                    const selectorsToHide = [
                        // GitHub ribbons/forks
                        '[class*="github"]',
                        '[id*="github"]',
                        '[href*="github.com"]',
                        'a[href*="fork"]',
                        '[class*="fork"]',
                        '[id*="fork"]',
                        '.github-corner',
                        '.github-ribbon',
                        '.gh-ribbon',

                        // Ads and promotions
                        '[class*="ad-"]',
                        '[class*="ads-"]',
                        '[id*="ad-"]',
                        '[class*="banner"]',
                        '[class*="promo"]',

                        // Social buttons
                        '[class*="social"]',
                        '[class*="share"]',

                        // Fixed/sticky elements that might overlay the game
                        '[class*="overlay"]',
                        '[class*="popup"]',
                        '[class*="modal"]',

                        // Cookie notices (in case they weren't clicked)
                        '[class*="cookie"]',
                        '[class*="gdpr"]',
                        '[class*="consent"]',
                    ];

                    for (const selector of selectorsToHide) {
                        try {
                            const elements = document.querySelectorAll(selector);
                            elements.forEach(el => {
                                // Don't hide the main game canvas
                                if (el.tagName !== 'CANVAS' && !el.querySelector('canvas')) {
                                    el.style.display = 'none';
                                    el.style.visibility = 'hidden';
                                    el.style.opacity = '0';
                                    el.style.pointerEvents = 'none';
                                }
                            });
                        } catch (e) {}
                    }

                    // Also hide any fixed/absolute positioned elements at corners
                    const allElements = document.querySelectorAll('*');
                    allElements.forEach(el => {
                        const style = getComputedStyle(el);
                        if (style.position === 'fixed' || style.position === 'absolute') {
                            const rect = el.getBoundingClientRect();
                            // If it's in a corner and small, likely an overlay
                            if ((rect.top < 100 || rect.bottom > window.innerHeight - 100) &&
                                (rect.left < 100 || rect.right > window.innerWidth - 100) &&
                                rect.width < 200 && rect.height < 200 &&
                                el.tagName !== 'CANVAS') {
                                el.style.display = 'none';
                            }
                        }
                    });
                }
            """)
        except Exception as e:
            pass  # Silently ignore if this fails

    def _close_browser(self):
        """Close the browser."""
        if self.page:
            self.page.close()
            self.page = None
        if self.browser:
            self.browser.close()
            self.browser = None
        if self.playwright:
            self.playwright.stop()
            self.playwright = None

    def _get_screenshot(self) -> np.ndarray:
        """Capture a screenshot of the game area."""
        # Take screenshot
        screenshot_bytes = self.page.screenshot()

        # Convert to numpy array
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(screenshot_bytes))

        # Resize to target dimensions
        img = img.resize((self.width, self.height))

        # Convert to grayscale if needed
        if self.grayscale:
            img = img.convert('L')
            arr = np.array(img)
            arr = arr.reshape(self.height, self.width, 1)
        else:
            img = img.convert('RGB')
            arr = np.array(img)

        self.last_screenshot = arr
        return arr

    def _get_score(self) -> int:
        """Get the current score from the game."""
        if self.score_js:
            try:
                return int(self.page.evaluate(self.score_js))
            except:
                pass
        return self.score

    def _is_game_over(self) -> bool:
        """Check if the game is over."""
        if self.game_over_js:
            try:
                return bool(self.page.evaluate(self.game_over_js))
            except:
                pass
        return False

    def _press_key(self, key: str):
        """Press a key."""
        self.page.keyboard.press(key)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the game."""
        super().reset(seed=seed)

        # Start browser if not running
        if self.page is None:
            self._start_browser()
        else:
            # Restart the game
            if self.restart_js:
                try:
                    result = self.page.evaluate(self.restart_js)
                    if not result:
                        raise Exception("Restart JS returned false")
                    time.sleep(0.3)
                except:
                    # Fallback: reload page
                    self.page.reload()
                    time.sleep(1)
                    self._handle_consent_popups()
                    self.page.click("body")
                    time.sleep(0.5)
            else:
                # Reload the page to restart
                self.page.reload()
                time.sleep(1)
                self._handle_consent_popups()
                self.page.click("body")
                time.sleep(0.5)

        # Start the game
        self._press_key(self.start_key)
        time.sleep(0.3)

        self.score = 0
        self.steps = 0

        obs = self._get_screenshot()
        info = {"score": self.score}

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the game."""
        self.steps += 1

        # Execute action
        if action > 0 and action <= len(self.action_keys):
            key = self.action_keys[action - 1]
            self._press_key(key)

        # Wait for frame(s)
        time.sleep(0.05 * self.frame_skip)

        # Get new state
        obs = self._get_screenshot()
        new_score = self._get_score()
        game_over = self._is_game_over()

        # Calculate reward
        reward = new_score - self.score  # Reward = score increase
        if game_over:
            reward -= 1  # Small penalty for dying
        else:
            reward += 0.01  # Small reward for surviving

        self.score = new_score

        info = {
            "score": self.score,
            "steps": self.steps
        }

        return obs, reward, game_over, False, info

    def render(self):
        """Render the game."""
        if self.render_mode == "rgb_array":
            return self.last_screenshot
        # For "human" mode, the browser window is already visible
        return None

    def close(self):
        """Close the environment."""
        self._close_browser()


class DinoGameEnv(WebGameEnv):
    """
    Environment specifically for the Chrome Dinosaur game at dinosaur-game.io.
    """

    def __init__(self, render_mode: Optional[str] = None):
        game_config = {
            'start_key': 'Space',
            'action_keys': ['Space', 'ArrowDown'],  # Jump, Duck
            'score_js': '''
                (() => {
                    // Try multiple methods to get score
                    // Method 1: Look for score display elements
                    const scoreEls = document.querySelectorAll('[class*="score"], [id*="score"], .distance-meter');
                    for (const el of scoreEls) {
                        const text = el.textContent || el.innerText || '';
                        const num = parseInt(text.replace(/[^0-9]/g, ''));
                        if (!isNaN(num) && num > 0) return num;
                    }
                    // Method 2: Try Runner object (Chrome dino)
                    if (typeof Runner !== 'undefined' && Runner.instance_) {
                        return Math.floor(Runner.instance_.distanceRan) || 0;
                    }
                    return 0;
                })()
            ''',
            'game_over_js': '''
                (() => {
                    // Method 1: Check for visible game-over elements
                    const gameOverSelectors = [
                        '.game-over', '[class*="game-over"]', '[class*="gameover"]',
                        '.crashed', '[class*="crashed"]', '.ended', '[class*="ended"]',
                        '#game-over', '.replay', '[class*="replay"]'
                    ];
                    for (const sel of gameOverSelectors) {
                        const el = document.querySelector(sel);
                        if (el) {
                            const style = getComputedStyle(el);
                            if (style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0') {
                                return true;
                            }
                        }
                    }
                    // Method 2: Check Runner object
                    if (typeof Runner !== 'undefined' && Runner.instance_) {
                        return Runner.instance_.crashed === true;
                    }
                    // Method 3: Check if game canvas shows "GAME OVER" text (some versions)
                    return false;
                })()
            ''',
            'restart_js': '''
                (() => {
                    // Method 1: Use Runner restart
                    if (typeof Runner !== 'undefined' && Runner.instance_) {
                        Runner.instance_.restart();
                        return true;
                    }
                    // Method 2: Click any restart/replay button
                    const btns = document.querySelectorAll('.restart, [class*="restart"], .replay, [class*="replay"], .play-again');
                    for (const btn of btns) {
                        btn.click();
                        return true;
                    }
                    return false;
                })()
            '''
        }

        super().__init__(
            game_url="https://dinosaur-game.io/",
            render_mode=render_mode,
            width=600,
            height=150,
            grayscale=True,
            frame_skip=2,
            game_config=game_config
        )

        self.last_screenshot = None
        self.static_frames = 0

    def _is_game_over(self) -> bool:
        """Check if game is over using JS and visual detection."""
        # First try JavaScript detection
        if self.game_over_js:
            try:
                result = self.page.evaluate(self.game_over_js)
                if result:
                    return True
            except:
                pass

        # Visual detection: if screen hasn't changed for several frames, game is likely over
        # (The dino game freezes on game over)
        if self.last_screenshot is not None:
            current = self._get_screenshot()
            diff = np.mean(np.abs(current.astype(float) - self.last_screenshot.astype(float)))
            if diff < 1.0:  # Very little change
                self.static_frames += 1
                if self.static_frames > 5:  # 5 consecutive static frames = game over
                    return True
            else:
                self.static_frames = 0

        return False

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step with improved game-over detection."""
        self.steps += 1

        # Execute action
        if action > 0 and action <= len(self.action_keys):
            key = self.action_keys[action - 1]
            self._press_key(key)

        # Wait for frame(s)
        time.sleep(0.05 * self.frame_skip)

        # Get new state
        obs = self._get_screenshot()
        new_score = self._get_score()
        game_over = self._is_game_over()

        # Store screenshot for visual detection
        self.last_screenshot = obs.copy()

        # Calculate reward
        reward = new_score - self.score  # Reward = score increase
        if game_over:
            reward -= 1  # Small penalty for dying
        else:
            reward += 0.01  # Small reward for surviving

        self.score = new_score

        info = {
            "score": self.score,
            "steps": self.steps
        }

        return obs, reward, game_over, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset with static frame counter."""
        self.last_screenshot = None
        self.static_frames = 0
        return super().reset(seed=seed, options=options)


# Factory function to create environment based on URL
def create_web_game_env(url: str, render_mode: Optional[str] = None) -> WebGameEnv:
    """
    Create a web game environment based on the URL.

    Recognizes specific games and uses specialized configurations.
    """
    url_lower = url.lower()

    if 'dinosaur' in url_lower or 'dino' in url_lower or 't-rex' in url_lower:
        return DinoGameEnv(render_mode=render_mode)

    # Generic web game
    return WebGameEnv(
        game_url=url,
        render_mode=render_mode,
        width=800,
        height=600,
        grayscale=False
    )
