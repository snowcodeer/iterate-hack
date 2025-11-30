import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from typing import Optional


class SnakeEaterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode: Optional[str] = None, difficulty: int = 25):
        super().__init__()

        # Game configuration
        self.frame_size_x = 720
        self.frame_size_y = 480
        self.grid_size = 10
        self.difficulty = difficulty
        self.render_mode = render_mode

        # Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = spaces.Discrete(4)

        # Observation space: snake head pos (x, y), food pos (x, y), direction (one-hot 4),
        # plus relative positions and danger detection
        # We'll use: [head_x, head_y, food_x, food_y, dir_up, dir_down, dir_left, dir_right,
        #             food_rel_x, food_rel_y, danger_straight, danger_left, danger_right, snake_length]
        self.observation_space = spaces.Box(
            low=0, high=max(self.frame_size_x, self.frame_size_y),
            shape=(14,), dtype=np.float32
        )

        # Pygame initialization
        self.game_window = None
        self.clock = None
        self.font = None

        if self.render_mode == "human":
            pygame.init()
            self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))
            pygame.display.set_caption('Snake Eater')
            self.clock = pygame.time.Clock()
            try:
                self.font = pygame.font.SysFont('consolas', 20)
            except Exception:
                self.font = None  # Font not available on Python 3.14

        # Colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)

        # Game state
        self.snake_pos = None
        self.snake_body = None
        self.food_pos = None
        self.direction = None
        self.score = None
        self.steps = None
        self.max_steps = 1000

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize snake
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]

        # Initialize food
        self.food_pos = self._spawn_food()

        # Initialize direction
        self.direction = 'RIGHT'

        # Initialize score and steps
        self.score = 0
        self.steps = 0

        return self._get_observation(), {}

    def _spawn_food(self):
        """Spawn food at a random position not occupied by snake"""
        while True:
            food_pos = [
                random.randrange(1, (self.frame_size_x // self.grid_size)) * self.grid_size,
                random.randrange(1, (self.frame_size_y // self.grid_size)) * self.grid_size
            ]
            if food_pos not in self.snake_body:
                return food_pos

    def _get_observation(self):
        """Get current observation state"""
        head_x, head_y = self.snake_pos

        # Direction one-hot encoding
        dir_up = 1.0 if self.direction == 'UP' else 0.0
        dir_down = 1.0 if self.direction == 'DOWN' else 0.0
        dir_left = 1.0 if self.direction == 'LEFT' else 0.0
        dir_right = 1.0 if self.direction == 'RIGHT' else 0.0

        # Relative food position
        food_rel_x = self.food_pos[0] - head_x
        food_rel_y = self.food_pos[1] - head_y

        # Danger detection
        danger_straight = self._is_danger_ahead(self.direction)
        danger_left = self._is_danger_ahead(self._get_left_direction(self.direction))
        danger_right = self._is_danger_ahead(self._get_right_direction(self.direction))

        obs = np.array([
            head_x / self.frame_size_x,
            head_y / self.frame_size_y,
            self.food_pos[0] / self.frame_size_x,
            self.food_pos[1] / self.frame_size_y,
            dir_up, dir_down, dir_left, dir_right,
            food_rel_x / self.frame_size_x,
            food_rel_y / self.frame_size_y,
            float(danger_straight),
            float(danger_left),
            float(danger_right),
            len(self.snake_body) / 100.0
        ], dtype=np.float32)

        return obs

    def _get_left_direction(self, direction):
        """Get the left direction relative to current direction"""
        directions = ['UP', 'LEFT', 'DOWN', 'RIGHT']
        idx = directions.index(direction)
        return directions[(idx + 1) % 4]

    def _get_right_direction(self, direction):
        """Get the right direction relative to current direction"""
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        idx = directions.index(direction)
        return directions[(idx + 1) % 4]

    def _is_danger_ahead(self, direction):
        """Check if there's danger in the given direction"""
        test_pos = list(self.snake_pos)

        if direction == 'UP':
            test_pos[1] -= self.grid_size
        elif direction == 'DOWN':
            test_pos[1] += self.grid_size
        elif direction == 'LEFT':
            test_pos[0] -= self.grid_size
        elif direction == 'RIGHT':
            test_pos[0] += self.grid_size

        # Check wall collision
        if (test_pos[0] < 0 or test_pos[0] >= self.frame_size_x or
            test_pos[1] < 0 or test_pos[1] >= self.frame_size_y):
            return True

        # Check self collision
        if test_pos in self.snake_body[1:]:
            return True

        return False

    def step(self, action):
        self.steps += 1

        # Map action to direction change
        action_to_direction = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        change_to = action_to_direction[action]

        # Prevent opposite direction
        if change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        elif change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        elif change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        elif change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'

        # Store previous distance to food
        prev_distance = abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1])

        # Move snake
        if self.direction == 'UP':
            self.snake_pos[1] -= self.grid_size
        elif self.direction == 'DOWN':
            self.snake_pos[1] += self.grid_size
        elif self.direction == 'LEFT':
            self.snake_pos[0] -= self.grid_size
        elif self.direction == 'RIGHT':
            self.snake_pos[0] += self.grid_size

        # Snake body growing mechanism
        self.snake_body.insert(0, list(self.snake_pos))

        reward = 0
        terminated = False

        # Check if food eaten
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.score += 1
            reward = 10
            self.food_pos = self._spawn_food()
        else:
            self.snake_body.pop()
            # Small reward for getting closer to food
            new_distance = abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1])
            if new_distance < prev_distance:
                reward = 0.1
            else:
                reward = -0.1

        # Check game over conditions
        # Out of bounds
        if (self.snake_pos[0] < 0 or self.snake_pos[0] >= self.frame_size_x or
            self.snake_pos[1] < 0 or self.snake_pos[1] >= self.frame_size_y):
            terminated = True
            reward = -10

        # Self collision
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                terminated = True
                reward = -10

        # Timeout
        if self.steps >= self.max_steps:
            terminated = True

        observation = self._get_observation()
        info = {
            'score': self.score,
            'snake_length': len(self.snake_body),
            'steps': self.steps
        }

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def _render_human(self):
        if self.game_window is None:
            pygame.init()
            self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))
            pygame.display.set_caption('Snake Eater')
            self.clock = pygame.time.Clock()
            try:
                self.font = pygame.font.SysFont('consolas', 20)
            except Exception:
                self.font = None

        # Fill background
        self.game_window.fill(self.black)

        # Draw snake
        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, self.green,
                           pygame.Rect(pos[0], pos[1], self.grid_size, self.grid_size))

        # Draw food
        pygame.draw.rect(self.game_window, self.white,
                        pygame.Rect(self.food_pos[0], self.food_pos[1], self.grid_size, self.grid_size))

        # Draw score
        if self.font:
            try:
                score_surface = self.font.render(f'Score : {self.score}', True, self.white)
                score_rect = score_surface.get_rect()
                score_rect.midtop = (self.frame_size_x / 10, 15)
                self.game_window.blit(score_surface, score_rect)
            except Exception:
                pass  # Skip score display if font fails

        pygame.display.update()

        if self.clock:
            self.clock.tick(self.metadata["render_fps"])

    def _render_rgb_array(self):
        if self.game_window is None:
            pygame.init()
            self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))
            try:
                self.font = pygame.font.SysFont('consolas', 20)
            except Exception:
                self.font = None

        # Fill background
        self.game_window.fill(self.black)

        # Draw snake
        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, self.green,
                           pygame.Rect(pos[0], pos[1], self.grid_size, self.grid_size))

        # Draw food
        pygame.draw.rect(self.game_window, self.white,
                        pygame.Rect(self.food_pos[0], self.food_pos[1], self.grid_size, self.grid_size))

        # Draw score
        if self.font:
            try:
                score_surface = self.font.render(f'Score : {self.score}', True, self.white)
                score_rect = score_surface.get_rect()
                score_rect.midtop = (self.frame_size_x / 10, 15)
                self.game_window.blit(score_surface, score_rect)
            except Exception:
                pass  # Skip score display if font fails

        # Convert to RGB array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.game_window)), axes=(1, 0, 2)
        )

    def close(self):
        if self.game_window is not None:
            pygame.quit()
            self.game_window = None
            self.clock = None
            self.font = None