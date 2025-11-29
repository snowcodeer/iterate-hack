"""Controller for web-based games using browser automation."""

import time
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright, Page, Browser, Playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Playwright = None
    Page = None
    Browser = None

from uniwrap.game_controllers.base import GameController


class WebGameController(GameController):
    """Control web games via browser automation."""
    
    def __init__(self, game_url: str, framework: str = 'react', headless: bool = True):
        """Initialize web game controller.
        
        Args:
            game_url: URL to the game (http://localhost:3000 or file:// path)
            framework: Framework type (react, vue, vanilla, etc.)
            headless: Run browser in headless mode
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "playwright is required for web game integration. "
                "Install with: pip install playwright && playwright install chromium"
            )
        
        self.game_url = game_url
        self.framework = framework
        self.headless = headless
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._game_state_cache: Optional[Dict] = None
    
    def start_game(self) -> bool:
        """Launch game in headless browser."""
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=self.headless)
            self.page = self.browser.new_page()
            
            # Navigate to game
            if self.game_url.startswith('http') or self.game_url.startswith('file://'):
                self.page.goto(self.game_url)
            else:
                # Assume local file path
                file_path = Path(self.game_url).absolute()
                if file_path.exists():
                    self.page.goto(f"file://{file_path}")
                else:
                    raise ValueError(f"Game URL/path not found: {self.game_url}")
            
            # Wait for page to load
            self.page.wait_for_load_state('networkidle', timeout=10000)
            
            # Inject state extraction helpers
            self._inject_state_extraction()
            
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to start web game: {e}")
    
    def _inject_state_extraction(self):
        """Inject JavaScript helpers for state extraction."""
        script = """
        // Create global state extraction function
        window.__RL_GET_STATE__ = function() {
            // Try React first
            if (window.__REACT_DEVTOOLS_GLOBAL_HOOK__) {
                const reactFiber = window.__REACT_DEVTOOLS_GLOBAL_HOOK__.getFiberRoots(1).values().next().value;
                if (reactFiber) {
                    // Traverse React tree to find game state
                    let node = reactFiber;
                    while (node) {
                        if (node.memoizedState) {
                            const state = node.memoizedState;
                            if (state.sliceQuality !== undefined || state.toastLevel !== undefined) {
                                return {
                                    sliceQuality: state.sliceQuality || 0,
                                    toastLevel: state.toastLevel || 0,
                                    butterCoverage: state.butterCoverage || 0,
                                    phase: state.phase || 'START'
                                };
                            }
                        }
                        node = node.child || node.sibling;
                    }
                }
            }
            
            // Fallback: Try to read from window.gameState or similar
            if (window.gameState) {
                return window.gameState;
            }
            
            // Try Vue
            if (window.__VUE__) {
                // Vue state extraction logic
            }
            
            // Default: return empty state
            return { sliceQuality: 0, toastLevel: 0, butterCoverage: 0, phase: 'START' };
        };
        
        // Create action application function
        window.__RL_APPLY_ACTION__ = function(action) {
            // Store action for game to process
            window.__RL_ACTION__ = action;
            window.__RL_ACTION_READY__ = true;
            
            // Trigger action based on framework
            if (window.game && window.game.applyAction) {
                window.game.applyAction(action);
            } else if (window.dispatchEvent) {
                // Try custom event
                window.dispatchEvent(new CustomEvent('rl-action', { detail: action }));
            }
        };
        """
        self.page.evaluate(script)
    
    def apply_action(self, action: Any) -> Dict:
        """Apply action to web game.
        
        Args:
            action: Action from RL agent (can be list, int, or dict)
            
        Returns:
            Dict with action result info
        """
        if not self.page:
            raise RuntimeError("Game not started. Call start_game() first.")
        
        try:
            # Convert action to format game expects
            action_json = action.tolist() if hasattr(action, 'tolist') else action
            
            # Apply action via JavaScript
            result = self.page.evaluate(f"""
                (action) => {{
                    window.__RL_APPLY_ACTION__(action);
                    return {{ success: true }};
                }}
            """, action_json)
            
            # Small delay for action to process
            time.sleep(0.1)
            
            return result
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_game_state(self) -> Dict[str, float]:
        """Extract current game state from web page.
        
        Returns:
            Dict of normalized state variables (0-1)
        """
        if not self.page:
            raise RuntimeError("Game not started. Call start_game() first.")
        
        try:
            state = self.page.evaluate("window.__RL_GET_STATE__()")
            
            # Normalize to 0-1 range
            normalized = {
                'slice_quality': float(state.get('sliceQuality', 0)) / 100.0,
                'toast_level': float(state.get('toastLevel', 0)) / 100.0,
                'butter_coverage': float(state.get('butterCoverage', 0)) / 100.0,
                'phase': state.get('phase', 'START')
            }
            
            self._game_state_cache = normalized
            return normalized
        except Exception as e:
            # Return cached state or defaults
            if self._game_state_cache:
                return self._game_state_cache
            return {
                'slice_quality': 0.0,
                'toast_level': 0.0,
                'butter_coverage': 0.0,
                'phase': 'START'
            }
    
    def reset_game(self) -> bool:
        """Reset game to initial state."""
        if not self.page:
            return False
        
        try:
            # Try clicking restart button
            restart_selectors = [
                'button:has-text("Restart")',
                'button:has-text("Reset")',
                'button:has-text("New Game")',
                '[data-testid="restart"]',
                '.restart-button'
            ]
            
            for selector in restart_selectors:
                try:
                    if self.page.query_selector(selector):
                        self.page.click(selector)
                        time.sleep(0.5)
                        return True
                except:
                    continue
            
            # Fallback: reload page
            self.page.reload()
            self.page.wait_for_load_state('networkidle')
            self._inject_state_extraction()
            return True
        except Exception as e:
            return False
    
    def close_game(self):
        """Close browser and cleanup."""
        try:
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
        except:
            pass
    
    def is_game_running(self) -> bool:
        """Check if game is still running."""
        if not self.page:
            return False
        try:
            return not self.page.is_closed()
        except:
            return False

