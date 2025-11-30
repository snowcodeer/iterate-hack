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

        # Game constants
        self.frame_size_x = 720
        self.frame_size_y = 480
        self.grid_size = 10
        self.difficulty = difficulty

        # Define action and observation spaces
        # Actions: 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
        self.action_space = spaces.Discrete(4)

        # Observation space: snake head position, direction, food position, and body positions
        # We'll use a simplified observation: [head_x, head_y, food_x, food_y, direction_encoded(4), body_length]
        # Plus relative food position and danger sensors
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.frame_size_x, self.frame_size_y),
            shape=(11,),
            dtype=np.float32
        )

        # Render mode
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)

        # Game state variables
        self.snake_pos = None
        self.snake_body = None
        self.food_pos = None
        self.direction = None
        self.score = None
        self.steps = None
        self.max_steps = 1000

    def _get_obs(self):
        head_x, head_y = self.snake_pos
        food_x, food_y = self.food_pos

        # Direction encoding (one-hot)
        direction_map = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}
        dir_encoded = direction_map[self.direction]

        # Relative food position
        rel_food_x = food_x - head_x
        rel_food_y = food_y - head_y

        # Danger detection (straight, left, right relative to current direction)
        danger_straight = self._is_danger_ahead(self.direction)
        danger_left = self._is_danger_ahead(self._turn_left(self.direction))
        danger_right = self._is_danger_ahead(self._turn_right(self.direction))

        obs = np.array([
            head_x / self.frame_size_x,
            head_y / self.frame_size_y,
            food_x / self.frame_size_x,
            food_y / self.frame_size_y,
            rel_food_x / self.frame_size_x,
            rel_food_y / self.frame_size_y,
            dir_encoded / 3.0,
            len(self.snake_body) / 100.0,
            float(danger_straight),
            float(danger_left),
            float(danger_right)
        ], dtype=np.float32)

        return obs

    def _turn_left(self, direction):
        turns = {'UP': 'LEFT', 'LEFT': 'DOWN', 'DOWN': 'RIGHT', 'RIGHT': 'UP'}
        return turns[direction]

    def _turn_right(self, direction):
        turns = {'UP': 'RIGHT', 'RIGHT': 'DOWN', 'DOWN': 'LEFT', 'LEFT': 'UP'}
        return turns[direction]

    def _is_danger_ahead(self, direction):
        head_x, head_y = self.snake_pos
        test_pos = [head_x, head_y]

        if direction == 'UP':
            test_pos[1] -= self.grid_size
        elif direction == 'DOWN':
            test_pos[1] += self.grid_size
        elif direction == 'LEFT':
            test_pos[0] -= self.grid_size
        elif direction == 'RIGHT':
            test_pos[0] += self.grid_size

        # Check bounds
        if test_pos[0] < 0 or test_pos[0] >= self.frame_size_x:
            return True
        if test_pos[1] < 0 or test_pos[1] >= self.frame_size_y:
            return True

        # Check body collision
        for block in self.snake_body[1:]:
            if test_pos[0] == block[0] and test_pos[1] == block[1]:
                return True

        return False

    def _get_info(self):
        return {
            "score": self.score,
            "snake_length": len(self.snake_body),
            "steps": self.steps
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Initialize game state
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.direction = 'RIGHT'
        self.score = 0
        self.steps = 0

        # Spawn food
        self.food_pos = [
            random.randrange(1, (self.frame_size_x // self.grid_size)) * self.grid_size,
            random.randrange(1, (self.frame_size_y // self.grid_size)) * self.grid_size
        ]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.steps += 1

        # Map action to direction change
        action_to_direction = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        change_to = action_to_direction[action]

        # Prevent opposite direction movement
        if change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        elif change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        elif change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        elif change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'

        # Store previous distance to food for reward shaping
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

        # Check if food is eaten
        reward = 0
        ate_food = False
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.score += 1
            reward = 10
            ate_food = True
            # Spawn new food
            self.food_pos = [
                random.randrange(1, (self.frame_size_x // self.grid_size)) * self.grid_size,
                random.randrange(1, (self.frame_size_y // self.grid_size)) * self.grid_size
            ]
        else:
            self.snake_body.pop()

        # Check game over conditions
        terminated = False

        # Out of bounds
        if (self.snake_pos[0] < 0 or self.snake_pos[0] >= self.frame_size_x or
                self.snake_pos[1] < 0 or self.snake_pos[1] >= self.frame_size_y):
            terminated = True
            reward = -10

        # Body collision
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                terminated = True
                reward = -10

        # Time limit
        truncated = self.steps >= self.max_steps

        # Reward shaping: small reward for getting closer to food
        if not terminated and not ate_food:
            curr_distance = abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1])
            if curr_distance < prev_distance:
                reward = 0.1
            else:
                reward = -0.1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))
            pygame.display.set_caption('Snake Eater')

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.frame_size_x, self.frame_size_y))
        canvas.fill(self.black)

        # Draw snake
        for pos in self.snake_body:
            pygame.draw.rect(canvas, self.green, pygame.Rect(pos[0], pos[1], self.grid_size, self.grid_size))

        # Draw food
        pygame.draw.rect(canvas, self.white, pygame.Rect(self.food_pos[0], self.food_pos[1], self.grid_size, self.grid_size))

        # Draw score (skip if font module unavailable - Python 3.14 compatibility)
        if self.render_mode == "human":
            try:
                font = pygame.font.SysFont('consolas', 20)
                score_surface = font.render(f'Score : {self.score}', True, self.white)
                score_rect = score_surface.get_rect()
                score_rect.midtop = (self.frame_size_x / 10, 15)
                canvas.blit(score_surface, score_rect)
            except (NotImplementedError, ImportError):
                pass  # Font not available, skip score display

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None