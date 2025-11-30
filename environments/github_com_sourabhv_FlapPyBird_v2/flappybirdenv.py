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
        
        # Screen dimensions
        self.SCREEN_WIDTH = 288
        self.SCREEN_HEIGHT = 512
        self.PIPE_GAP_SIZE = 100
        self.PIPE_WIDTH = 52
        self.BIRD_WIDTH = 34
        self.BIRD_HEIGHT = 24
        self.BASE_Y = int(self.SCREEN_HEIGHT * 0.79)
        self.BIRD_X = int(self.SCREEN_WIDTH * 0.2)
        
        self.FPS = 30
        
        # Physics constants
        self.PLAYER_MAX_VEL_Y = 10
        self.PLAYER_MIN_VEL_Y = -8
        self.PLAYER_ACC_Y = 1  # Gravity
        self.PLAYER_FLAP_ACC = -9  # Upward velocity when flapping
        
        self.PIPE_VEL_X = -4
        
        # Action space: 0 = do nothing, 1 = flap
        self.action_space = spaces.Discrete(2)
        
        # Observation space: [bird_y, bird_vel_y, next_pipe_x, next_pipe_top, next_pipe_bottom, next_next_pipe_x, next_next_pipe_top]
        self.observation_space = spaces.Box(
            low=np.array([0, -20, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([
                self.SCREEN_HEIGHT,
                20,
                self.SCREEN_WIDTH,
                self.SCREEN_HEIGHT,
                self.SCREEN_HEIGHT,
                self.SCREEN_WIDTH,
                self.SCREEN_HEIGHT
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        self.screen = None
        self.clock = None
        
        # Game state
        self.bird_y = 0
        self.bird_vel_y = 0
        
        self.pipes = []
        self.score = 0
        self.game_over = False
        
        self.base_x = 0
        self.base_shift = 0
        
        self.frames = 0
        self.passed_pipes = set()  # Track which pipes have been passed for scoring
        
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird")
            self.clock = pygame.time.Clock()

    def _get_pipe_position(self) -> Tuple[int, int]:
        """Generate a new pipe position with random gap height."""
        gap_y = random.randint(int(self.SCREEN_HEIGHT * 0.2), int(self.SCREEN_HEIGHT * 0.6))
        pipe_x = self.SCREEN_WIDTH + 10
        return pipe_x, gap_y

    def _check_collision(self) -> bool:
        """Check if bird collided with ground, ceiling, or pipes."""
        # Check ground and ceiling collision
        if self.bird_y + self.BIRD_HEIGHT >= self.BASE_Y or self.bird_y <= 0:
            return True
        
        bird_rect = pygame.Rect(self.BIRD_X, int(self.bird_y), self.BIRD_WIDTH, self.BIRD_HEIGHT)
        
        # Check pipe collision
        for pipe in self.pipes:
            pipe_x, gap_y = pipe
            # Upper pipe extends from 0 to gap_y
            upper_pipe_rect = pygame.Rect(pipe_x, 0, self.PIPE_WIDTH, gap_y)
            # Lower pipe extends from gap_y + gap_size to base
            lower_pipe_rect = pygame.Rect(pipe_x, gap_y + self.PIPE_GAP_SIZE, 
                                         self.PIPE_WIDTH, self.BASE_Y - (gap_y + self.PIPE_GAP_SIZE))
            
            if bird_rect.colliderect(upper_pipe_rect) or bird_rect.colliderect(lower_pipe_rect):
                return True
        
        return False

    def _get_observation(self) -> np.ndarray:
        """Extract current observation state."""
        next_pipe_x = self.SCREEN_WIDTH
        next_pipe_top = 0
        next_pipe_bottom = self.SCREEN_HEIGHT
        next_next_pipe_x = self.SCREEN_WIDTH
        next_next_pipe_top = 0
        
        # Find pipes ahead of the bird
        pipes_ahead = []
        for pipe in self.pipes:
            pipe_x, gap_y = pipe
            if pipe_x + self.PIPE_WIDTH > self.BIRD_X:
                pipes_ahead.append((pipe_x, gap_y))
        
        # Get next pipe info
        if len(pipes_ahead) > 0:
            next_pipe_x, gap_y = pipes_ahead[0]
            next_pipe_top = gap_y
            next_pipe_bottom = gap_y + self.PIPE_GAP_SIZE
        
        # Get second pipe info
        if len(pipes_ahead) > 1:
            next_next_pipe_x, gap_y = pipes_ahead[1]
            next_next_pipe_top = gap_y
        
        obs = np.array([
            self.bird_y,
            self.bird_vel_y,
            next_pipe_x,
            next_pipe_top,
            next_pipe_bottom,
            next_next_pipe_x,
            next_next_pipe_top
        ], dtype=np.float32)
        
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset bird position and velocity
        self.bird_y = self.SCREEN_HEIGHT * 0.4
        self.bird_vel_y = 0
        
        # Reset pipes - start with two pipes
        self.pipes = []
        pipe_x, gap_y = self._get_pipe_position()
        self.pipes.append((pipe_x, gap_y))
        pipe_x, gap_y = self._get_pipe_position()
        self.pipes.append((pipe_x + self.SCREEN_WIDTH // 2, gap_y))
        
        # Reset game state
        self.score = 0
        self.game_over = False
        self.base_x = 0
        self.frames = 0
        self.passed_pipes = set()
        
        obs = self._get_observation()
        info = {"score": self.score}
        
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        # Reward shaping:
        # - Small survival reward each frame: 0.01
        # - Reward for staying near gap center: up to 0.1
        # - Large reward for passing pipe: 1.0
        # - Death penalty: -1.0
        reward = 0.01  # Base survival reward
        
        # Apply action - flap if action is 1
        if action == 1:
            self.bird_vel_y = self.PLAYER_FLAP_ACC
        
        # Apply gravity (always applies)
        self.bird_vel_y = min(self.bird_vel_y + self.PLAYER_ACC_Y, self.PLAYER_MAX_VEL_Y)
        
        # Update bird position
        self.bird_y += self.bird_vel_y
        # Clamp bird position
        self.bird_y = max(0, min(self.bird_y, self.BASE_Y - self.BIRD_HEIGHT))
        
        # Move pipes
        for i, pipe in enumerate(self.pipes):
            pipe_x, gap_y = pipe
            pipe_x += self.PIPE_VEL_X
            self.pipes[i] = (pipe_x, gap_y)
        
        # Remove off-screen pipes and add new ones
        if len(self.pipes) > 0 and self.pipes[0][0] < -self.PIPE_WIDTH:
            self.pipes.pop(0)
            pipe_x, gap_y = self._get_pipe_position()
            self.pipes.append((pipe_x, gap_y))
        
        # Check if bird passed a pipe and update score
        for i, pipe in enumerate(self.pipes):
            pipe_x, gap_y = pipe
            # Bird passes when it crosses the right edge of the pipe
            if pipe_x + self.PIPE_WIDTH < self.BIRD_X and i not in self.passed_pipes:
                self.passed_pipes.add(i)
                self.score += 1
                reward += 1.0  # Large reward for passing pipe
        
        # Distance-based shaping reward: reward staying near gap center
        if len(self.pipes) > 0:
            next_pipe_x, gap_y = self.pipes[0]
            gap_center = gap_y + self.PIPE_GAP_SIZE / 2
            bird_center = self.bird_y + self.BIRD_HEIGHT / 2
            
            # Calculate normalized distance to gap center (0 = at center, 1 = at edge)
            distance_to_center = abs(bird_center - gap_center)
            max_distance = self.PIPE_GAP_SIZE / 2
            normalized_distance = min(distance_to_center / max_distance, 1.0)
            
            # Reward being close to center (0.1 when centered, 0 at edges)
            center_reward = 0.1 * (1.0 - normalized_distance)
            reward += center_reward
        
        # Check collision
        collision = self._check_collision()
        if collision:
            self.game_over = True
            reward = -1.0  # Death penalty
        
        self.frames += 1
        
        obs = self._get_observation()
        terminated = self.game_over
        truncated = False
        info = {"score": self.score}
        
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the current game state."""
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
        
        # Draw sky
        self.screen.fill((135, 206, 250))
        
        # Draw pipes
        for pipe in self.pipes:
            pipe_x, gap_y = pipe
            # Upper pipe
            pygame.draw.rect(self.screen, (0, 255, 0), (pipe_x, 0, self.PIPE_WIDTH, gap_y))
            pygame.draw.rect(self.screen, (0, 200, 0), (pipe_x, 0, self.PIPE_WIDTH, 10))
            # Lower pipe
            pygame.draw.rect(self.screen, (0, 255, 0), 
                           (pipe_x, gap_y + self.PIPE_GAP_SIZE, 
                            self.PIPE_WIDTH, self.BASE_Y - (gap_y + self.PIPE_GAP_SIZE)))
            pygame.draw.rect(self.screen, (0, 200, 0), 
                           (pipe_x, gap_y + self.PIPE_GAP_SIZE, self.PIPE_WIDTH, 10))
        
        # Draw ground
        pygame.draw.rect(self.screen, (222, 216, 149), 
                        (0, self.BASE_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.BASE_Y))
        
        # Draw bird
        pygame.draw.circle(self.screen, (255, 255, 0), 
                         (int(self.BIRD_X + self.BIRD_WIDTH // 2), 
                          int(self.bird_y + self.BIRD_HEIGHT // 2)), 
                         self.BIRD_HEIGHT // 2)
        # Bird eye
        pygame.draw.circle(self.screen, (255, 255, 255), 
                         (int(self.BIRD_X + self.BIRD_WIDTH * 0.6), 
                          int(self.bird_y + self.BIRD_HEIGHT * 0.4)), 5)
        pygame.draw.circle(self.screen, (0, 0, 0), 
                         (int(self.BIRD_X + self.BIRD_WIDTH * 0.65), 
                          int(self.bird_y + self.BIRD_HEIGHT * 0.4)), 3)
        
        # Draw score - wrapped in try-except for font compatibility
        try:
            if FREETYPE_AVAILABLE:
                font_path = get_system_font_path()
                if font_path and freetype:
                    try:
                        freetype.init()
                        ft_font = freetype.Font(font_path, 36)
                        text_surf, _ = ft_font.render(f'Score: {self.score}', pygame.Color('white'))
                        self.screen.blit(text_surf, (10, 10))
                    except Exception:
                        pass
            else:
                try:
                    font = pygame.font.Font(None, 36)
                    text = font.render(f'Score: {self.score}', True, (255, 255, 255))
                    self.screen.blit(text, (10, 10))
                except Exception:
                    try:
                        font = pygame.font.SysFont('arial', 36)
                        text = font.render(f'Score: {self.score}', True, (255, 255, 255))
                        self.screen.blit(text, (10, 10))
                    except Exception:
                        pass  # Skip text if fonts unavailable
        except Exception:
            pass  # Skip text rendering if any error occurs
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        
        return None

    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None