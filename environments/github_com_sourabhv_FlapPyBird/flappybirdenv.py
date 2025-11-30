import os
import sys
from typing import Optional, Tuple
import numpy as np

try:
    import pygame._freetype as freetype
    FREETYPE_AVAILABLE = True
except ImportError:
    FREETYPE_AVAILABLE = False
    freetype = None

import pygame
import gymnasium as gym
from gymnasium import spaces
import random


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


class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.render_mode = render_mode
        
        self.SCREEN_WIDTH = 288
        self.SCREEN_HEIGHT = 512
        self.PIPE_GAP_SIZE = 100
        self.PIPE_WIDTH = 52
        self.BIRD_WIDTH = 34
        self.BIRD_HEIGHT = 24
        self.BASE_Y = int(self.SCREEN_HEIGHT * 0.79)
        self.BIRD_X = int(self.SCREEN_WIDTH * 0.2)
        
        self.FPS = 30
        
        self.PLAYER_MAX_VEL_Y = 10
        self.PLAYER_MIN_VEL_Y = -8
        self.PLAYER_ACC_Y = 1
        self.PLAYER_FLAP_ACC = -9
        
        self.PIPE_VEL_X = -4
        
        self.action_space = spaces.Discrete(2)
        
        self.observation_space = spaces.Box(
            low=np.array([0, -self.SCREEN_HEIGHT, -20, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([
                self.SCREEN_HEIGHT,
                self.SCREEN_HEIGHT * 2,
                20,
                self.SCREEN_WIDTH,
                self.SCREEN_HEIGHT,
                self.SCREEN_WIDTH,
                self.SCREEN_HEIGHT
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        self.screen = None
        self.clock = None
        
        self.bird_y = 0
        self.bird_vel_y = 0
        self.bird_flapped = False
        
        self.pipes = []
        self.score = 0
        self.game_over = False
        
        self.base_x = 0
        self.base_shift = 0
        
        self.frames = 0
        
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird")
            self.clock = pygame.time.Clock()

    def _get_pipe_position(self) -> Tuple[int, int]:
        gap_y = random.randint(int(self.SCREEN_HEIGHT * 0.2), int(self.SCREEN_HEIGHT * 0.6))
        pipe_x = self.SCREEN_WIDTH + 10
        return pipe_x, gap_y

    def _check_collision(self) -> bool:
        if self.bird_y + self.BIRD_HEIGHT >= self.BASE_Y or self.bird_y <= 0:
            return True
        
        bird_rect = pygame.Rect(self.BIRD_X, int(self.bird_y), self.BIRD_WIDTH, self.BIRD_HEIGHT)
        
        for pipe in self.pipes:
            pipe_x, gap_y = pipe
            upper_pipe_rect = pygame.Rect(pipe_x, 0, self.PIPE_WIDTH, gap_y)
            lower_pipe_rect = pygame.Rect(pipe_x, gap_y + self.PIPE_GAP_SIZE, self.PIPE_WIDTH, self.BASE_Y - (gap_y + self.PIPE_GAP_SIZE))
            
            if bird_rect.colliderect(upper_pipe_rect) or bird_rect.colliderect(lower_pipe_rect):
                return True
        
        return False

    def _get_observation(self) -> np.ndarray:
        next_pipe_x = self.SCREEN_WIDTH
        next_pipe_top = 0
        next_pipe_bottom = self.SCREEN_HEIGHT
        next_next_pipe_x = self.SCREEN_WIDTH
        next_next_pipe_top = 0
        
        pipes_ahead = []
        for pipe in self.pipes:
            pipe_x, gap_y = pipe
            if pipe_x + self.PIPE_WIDTH > self.BIRD_X:
                pipes_ahead.append((pipe_x, gap_y))
        
        if len(pipes_ahead) > 0:
            next_pipe_x, gap_y = pipes_ahead[0]
            next_pipe_top = gap_y
            next_pipe_bottom = gap_y + self.PIPE_GAP_SIZE
        
        if len(pipes_ahead) > 1:
            next_next_pipe_x, gap_y = pipes_ahead[1]
            next_next_pipe_top = gap_y
        
        obs = np.array([
            self.bird_y,
            self.bird_vel_y,
            self.bird_vel_y,
            next_pipe_x,
            next_pipe_top,
            next_next_pipe_x,
            next_next_pipe_top
        ], dtype=np.float32)
        
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.bird_y = self.SCREEN_HEIGHT * 0.4
        self.bird_vel_y = 0
        self.bird_flapped = False
        
        self.pipes = []
        pipe_x, gap_y = self._get_pipe_position()
        self.pipes.append((pipe_x, gap_y))
        pipe_x, gap_y = self._get_pipe_position()
        self.pipes.append((pipe_x + self.SCREEN_WIDTH // 2, gap_y))
        
        self.score = 0
        self.game_over = False
        self.base_x = 0
        self.frames = 0
        
        obs = self._get_observation()
        info = {"score": self.score}
        
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        reward = 0.1
        
        if action == 1:
            if self.bird_y > 0:
                self.bird_vel_y = self.PLAYER_FLAP_ACC
                self.bird_flapped = True
        
        if self.bird_vel_y < self.PLAYER_MAX_VEL_Y and not self.bird_flapped:
            self.bird_vel_y += self.PLAYER_ACC_Y
        
        self.bird_flapped = False
        
        self.bird_y += min(self.bird_vel_y, self.BASE_Y - self.bird_y - self.BIRD_HEIGHT)
        
        for i, pipe in enumerate(self.pipes):
            pipe_x, gap_y = pipe
            pipe_x += self.PIPE_VEL_X
            self.pipes[i] = (pipe_x, gap_y)
        
        if len(self.pipes) > 0 and self.pipes[0][0] < -self.PIPE_WIDTH:
            self.pipes.pop(0)
            pipe_x, gap_y = self._get_pipe_position()
            self.pipes.append((pipe_x, gap_y))
        
        for pipe in self.pipes:
            pipe_x, gap_y = pipe
            if pipe_x + self.PIPE_WIDTH > self.BIRD_X and pipe_x < self.BIRD_X:
                middle_of_pipe = pipe_x + self.PIPE_WIDTH // 2
                if self.BIRD_X < middle_of_pipe < self.BIRD_X + abs(self.PIPE_VEL_X) + 1:
                    self.score += 1
                    reward = 10.0
        
        self.base_x = -((-self.base_x + 100) % self.base_shift)
        
        collision = self._check_collision()
        if collision:
            self.game_over = True
            reward = -10.0
        
        self.frames += 1
        
        obs = self._get_observation()
        terminated = self.game_over
        truncated = False
        info = {"score": self.score}
        
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
                pygame.display.set_caption("Flappy Bird")
            else:
                self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
        
        self.screen.fill((135, 206, 250))
        
        for pipe in self.pipes:
            pipe_x, gap_y = pipe
            pygame.draw.rect(self.screen, (0, 255, 0), (pipe_x, 0, self.PIPE_WIDTH, gap_y))
            pygame.draw.rect(self.screen, (0, 200, 0), (pipe_x, 0, self.PIPE_WIDTH, 10))
            pygame.draw.rect(self.screen, (0, 255, 0), (pipe_x, gap_y + self.PIPE_GAP_SIZE, self.PIPE_WIDTH, self.BASE_Y - (gap_y + self.PIPE_GAP_SIZE)))
            pygame.draw.rect(self.screen, (0, 200, 0), (pipe_x, gap_y + self.PIPE_GAP_SIZE, self.PIPE_WIDTH, 10))
        
        pygame.draw.rect(self.screen, (222, 216, 149), (0, self.BASE_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.BASE_Y))
        
        pygame.draw.circle(self.screen, (255, 255, 0), (int(self.BIRD_X + self.BIRD_WIDTH // 2), int(self.bird_y + self.BIRD_HEIGHT // 2)), self.BIRD_HEIGHT // 2)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.BIRD_X + self.BIRD_WIDTH * 0.6), int(self.bird_y + self.BIRD_HEIGHT * 0.4)), 5)
        pygame.draw.circle(self.screen, (0, 0, 0), (int(self.BIRD_X + self.BIRD_WIDTH * 0.65), int(self.bird_y + self.BIRD_HEIGHT * 0.4)), 3)
        
        try:
            if FREETYPE_AVAILABLE:
                font_path = get_system_font_path()
                if font_path and freetype:
                    freetype.init()
                    ft_font = freetype.Font(font_path, 36)
                    text_surf, _ = ft_font.render(f'Score: {self.score}', pygame.Color('white'))
                    self.screen.blit(text_surf, (10, 10))
            else:
                font = pygame.font.Font(None, 36)
                text = font.render(f'Score: {self.score}', True, (255, 255, 255))
                self.screen.blit(text, (10, 10))
        except Exception:
            pass
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        
        return None

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None