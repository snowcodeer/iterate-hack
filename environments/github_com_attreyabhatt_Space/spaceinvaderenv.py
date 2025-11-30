import os
import math
import random
from typing import Optional
from pathlib import Path

try:
    import pygame._freetype as freetype
    FREETYPE_AVAILABLE = True
except ImportError:
    FREETYPE_AVAILABLE = False
    freetype = None

import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces


SYSTEM_FONT_PATHS = [
    '/System/Library/Fonts/Helvetica.ttc',
    '/System/Library/Fonts/Supplemental/Arial.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    'C:\\Windows\\Fonts\\arial.ttf',
]


def get_system_font_path():
    for path in SYSTEM_FONT_PATHS:
        if os.path.exists(path):
            return path
    return None


# Get the directory where this file is located (for loading assets)
ASSET_DIR = Path(__file__).parent


class SpaceInvaderEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.render_mode = render_mode

        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.PLAYER_SIZE = 64
        self.ENEMY_SIZE = 64
        self.BULLET_SIZE = 32

        self.num_of_enemies = 6

        self.observation_space = spaces.Box(
            low=0,
            high=max(self.SCREEN_WIDTH, self.SCREEN_HEIGHT),
            shape=(3 + self.num_of_enemies * 2 + 2,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(4)

        self.screen = None
        self.clock = None

        # Sprite images (loaded when rendering)
        self.player_img = None
        self.enemy_img = None
        self.bullet_img = None
        self.background_img = None
        self.sprites_loaded = False

        self.playerX = 0
        self.playerY = 0
        self.playerX_change = 0

        self.enemyX = []
        self.enemyY = []
        self.enemyX_change = []
        self.enemyY_change = []

        self.bulletX = 0
        self.bulletY = 0
        self.bulletX_change = 0
        self.bulletY_change = 10
        self.bullet_state = "ready"

        self.score_value = 0
        self.game_over = False
        self.steps = 0
        self.max_steps = 5000

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Space Invader")
            self.clock = pygame.time.Clock()
            self._load_sprites()

    def _load_sprites(self):
        """Load sprite images from the asset directory."""
        try:
            from PIL import Image
            import io

            player_path = ASSET_DIR / "player.png"
            enemy_path = ASSET_DIR / "enemy.png"
            bullet_path = ASSET_DIR / "bullet.png"
            background_path = ASSET_DIR / "background.png"

            def load_png(path):
                """Load PNG using PIL and convert to pygame surface."""
                if not path.exists():
                    return None
                # Use PIL to load the PNG, then convert to pygame
                pil_img = Image.open(str(path)).convert('RGBA')
                mode = pil_img.mode
                size = pil_img.size
                data = pil_img.tobytes()
                return pygame.image.fromstring(data, size, mode)

            self.player_img = load_png(player_path)
            self.enemy_img = load_png(enemy_path)
            self.bullet_img = load_png(bullet_path)
            self.background_img = load_png(background_path)

            self.sprites_loaded = True
        except Exception as e:
            print(f"Warning: Could not load sprites: {e}")
            self.sprites_loaded = False

    def _get_obs(self):
        obs = [
            self.playerX / self.SCREEN_WIDTH,
            self.bulletX / self.SCREEN_WIDTH,
            self.bulletY / self.SCREEN_HEIGHT
        ]
        
        for i in range(self.num_of_enemies):
            obs.append(self.enemyX[i] / self.SCREEN_WIDTH)
            obs.append(self.enemyY[i] / self.SCREEN_HEIGHT)
        
        obs.append(1.0 if self.bullet_state == "fire" else 0.0)
        obs.append(self.score_value / 100.0)
        
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.playerX = 370
        self.playerY = 480
        self.playerX_change = 0
        
        self.enemyX = []
        self.enemyY = []
        self.enemyX_change = []
        self.enemyY_change = []
        
        for i in range(self.num_of_enemies):
            self.enemyX.append(random.randint(0, 736))
            self.enemyY.append(random.randint(50, 150))
            self.enemyX_change.append(4)
            self.enemyY_change.append(40)
        
        self.bulletX = 0
        self.bulletY = 480
        self.bulletX_change = 0
        self.bullet_state = "ready"
        
        self.score_value = 0
        self.game_over = False
        self.steps = 0
        
        observation = self._get_obs()
        info = {}
        
        return observation, info

    def step(self, action):
        self.steps += 1
        reward = 0
        
        if action == 0:
            self.playerX_change = -5
        elif action == 1:
            self.playerX_change = 5
        elif action == 2:
            self.playerX_change = 0
        elif action == 3:
            if self.bullet_state == "ready":
                self.bulletX = self.playerX
                self.bullet_state = "fire"
        
        self.playerX += self.playerX_change
        if self.playerX <= 0:
            self.playerX = 0
        elif self.playerX >= 736:
            self.playerX = 736
        
        for i in range(self.num_of_enemies):
            if self.enemyY[i] > 440:
                self.game_over = True
                reward = -100
                break
            
            self.enemyX[i] += self.enemyX_change[i]
            if self.enemyX[i] <= 0:
                self.enemyX_change[i] = 4
                self.enemyY[i] += self.enemyY_change[i]
            elif self.enemyX[i] >= 736:
                self.enemyX_change[i] = -4
                self.enemyY[i] += self.enemyY_change[i]
            
            collision = self._is_collision(self.enemyX[i], self.enemyY[i], self.bulletX, self.bulletY)
            if collision:
                self.bulletY = 480
                self.bullet_state = "ready"
                self.score_value += 1
                reward = 10
                self.enemyX[i] = random.randint(0, 736)
                self.enemyY[i] = random.randint(50, 150)
        
        if self.bulletY <= 0:
            self.bulletY = 480
            self.bullet_state = "ready"
        
        if self.bullet_state == "fire":
            self.bulletY -= self.bulletY_change
        
        reward += 0.001
        
        terminated = self.game_over
        truncated = self.steps >= self.max_steps
        
        observation = self._get_obs()
        info = {"score": self.score_value}
        
        return observation, reward, terminated, truncated, info

    def _is_collision(self, enemyX, enemyY, bulletX, bulletY):
        distance = math.sqrt(math.pow(enemyX - bulletX, 2) + math.pow(enemyY - bulletY, 2))
        if distance < 27:
            return True
        else:
            return False

    def render(self):
        if self.render_mode != "human":
            return None

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Space Invader")
            self.clock = pygame.time.Clock()
            self._load_sprites()

        # Draw background
        if self.background_img is not None:
            self.screen.blit(self.background_img, (0, 0))
        else:
            self.screen.fill((0, 0, 0))

        # Draw player
        if self.player_img is not None:
            self.screen.blit(self.player_img, (self.playerX, self.playerY))
        else:
            player_surface = pygame.Surface((self.PLAYER_SIZE, self.PLAYER_SIZE))
            player_surface.fill((0, 255, 0))
            self.screen.blit(player_surface, (self.playerX, self.playerY))

        # Draw enemies
        for i in range(self.num_of_enemies):
            if self.enemyY[i] < 2000:
                if self.enemy_img is not None:
                    self.screen.blit(self.enemy_img, (self.enemyX[i], self.enemyY[i]))
                else:
                    enemy_surface = pygame.Surface((self.ENEMY_SIZE, self.ENEMY_SIZE))
                    enemy_surface.fill((255, 0, 0))
                    self.screen.blit(enemy_surface, (self.enemyX[i], self.enemyY[i]))

        # Draw bullet
        if self.bullet_state == "fire":
            if self.bullet_img is not None:
                self.screen.blit(self.bullet_img, (self.bulletX + 16, self.bulletY + 10))
            else:
                bullet_surface = pygame.Surface((self.BULLET_SIZE, self.BULLET_SIZE))
                bullet_surface.fill((255, 255, 0))
                self.screen.blit(bullet_surface, (self.bulletX + 16, self.bulletY + 10))

        # Draw score text
        try:
            if FREETYPE_AVAILABLE:
                font_path = get_system_font_path()
                if font_path and freetype:
                    freetype.init()
                    ft_font = freetype.Font(font_path, 32)
                    text_surf, _ = ft_font.render(f'Score: {self.score_value}', pygame.Color('white'))
                    self.screen.blit(text_surf, (10, 10))

                    if self.game_over:
                        ft_font_large = freetype.Font(font_path, 64)
                        over_surf, _ = ft_font_large.render('GAME OVER', pygame.Color('white'))
                        self.screen.blit(over_surf, (200, 250))
            else:
                font = pygame.font.Font(None, 32)
                text = font.render(f'Score: {self.score_value}', True, (255, 255, 255))
                self.screen.blit(text, (10, 10))

                if self.game_over:
                    over_font = pygame.font.Font(None, 64)
                    over_text = over_font.render('GAME OVER', True, (255, 255, 255))
                    self.screen.blit(over_text, (200, 250))
        except Exception:
            pass

        pygame.event.pump()  # Required for window to appear on macOS
        pygame.display.flip()

        if self.clock:
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None