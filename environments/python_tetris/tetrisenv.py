import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from copy import deepcopy
from random import choice, randrange
from typing import Optional

class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Game constants
        self.W = 10
        self.H = 20
        self.TILE = 45
        self.GAME_RES = (self.W * self.TILE, self.H * self.TILE)
        self.RES = (750, 940)
        self.FPS = 60
        
        # Tetromino shapes
        self.figures_pos = [
            [(-1, 0), (-2, 0), (0, 0), (1, 0)],  # I
            [(0, -1), (-1, -1), (-1, 0), (0, 0)],  # O
            [(-1, 0), (-1, 1), (0, 0), (0, -1)],  # S
            [(0, 0), (-1, 0), (0, 1), (-1, -1)],  # Z
            [(0, 0), (0, -1), (0, 1), (-1, -1)],  # J
            [(0, 0), (0, -1), (0, 1), (1, -1)],  # L
            [(0, 0), (0, -1), (0, 1), (-1, 0)]   # T
        ]
        
        self.scores = {0: 0, 1: 100, 2: 300, 3: 700, 4: 1500}
        
        # Action space: 0=nothing, 1=left, 2=right, 3=rotate, 4=down
        self.action_space = spaces.Discrete(5)
        
        # Observation space: field (H x W) + current figure positions (4x2) + next figure type
        self.observation_space = spaces.Dict({
            'field': spaces.Box(low=0, high=1, shape=(self.H, self.W), dtype=np.int8),
            'figure': spaces.Box(low=0, high=max(self.H, self.W), shape=(4, 2), dtype=np.int32),
            'next_figure_type': spaces.Discrete(7)
        })
        
        self.render_mode = render_mode
        self.screen = None
        self.game_screen = None
        self.clock = None
        
        # Initialize game state
        self._init_game_state()
        
    def _init_game_state(self):
        """Initialize/reset game state variables"""
        self.field = [[0 for _ in range(self.W)] for _ in range(self.H)]
        self.score = 0
        self.total_lines_cleared = 0
        self.anim_count = 0
        self.anim_speed = 60
        self.anim_limit = 2000
        self.game_over = False
        
        self.current_figure_type = randrange(7)
        self.next_figure_type = randrange(7)
        
        self.figure = self._create_figure(self.current_figure_type)
        self.next_figure = self._create_figure(self.next_figure_type)
        
    def _create_figure(self, figure_type):
        """Create a figure from a type index"""
        fig_pos = self.figures_pos[figure_type]
        return [[x + self.W // 2, y + 1] for x, y in fig_pos]
    
    def _check_borders(self, figure):
        """Check if figure is within bounds and doesn't collide"""
        for block in figure:
            x, y = block
            if x < 0 or x > self.W - 1:
                return False
            elif y > self.H - 1:
                return False
            elif y >= 0 and self.field[y][x]:
                return False
        return True
    
    def _rotate_figure(self, figure):
        """Rotate figure 90 degrees clockwise"""
        center = figure[0]
        rotated = deepcopy(figure)
        for i in range(4):
            x = figure[i][1] - center[1]
            y = figure[i][0] - center[0]
            rotated[i][0] = center[0] - x
            rotated[i][1] = center[1] + y
        return rotated
    
    def _lock_figure(self):
        """Lock current figure to the field"""
        for block in self.figure:
            x, y = block
            if 0 <= y < self.H and 0 <= x < self.W:
                self.field[y][x] = 1
    
    def _clear_lines(self):
        """Clear completed lines and return number of lines cleared"""
        lines_cleared = 0
        new_field = []
        
        for row in range(self.H - 1, -1, -1):
            if all(self.field[row]):
                lines_cleared += 1
            else:
                new_field.append(self.field[row][:])
        
        # Add empty lines at top
        for _ in range(lines_cleared):
            new_field.append([0 for _ in range(self.W)])
        
        new_field.reverse()
        self.field = new_field
        
        return lines_cleared
    
    def _check_game_over(self):
        """Check if game is over (blocks in top row)"""
        return any(self.field[0])
    
    def _get_observation(self):
        """Get current observation"""
        field_array = np.array(self.field, dtype=np.int8)
        figure_array = np.array(self.figure, dtype=np.int32)
        
        return {
            'field': field_array,
            'figure': figure_array,
            'next_figure_type': self.next_figure_type
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment"""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        self._init_game_state()
        
        observation = self._get_observation()
        info = {'score': self.score, 'lines': self.total_lines_cleared}
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.game_over:
            return self._get_observation(), 0, True, False, {'score': self.score, 'lines': self.total_lines_cleared}
        
        reward = 0
        dx = 0
        rotate = False
        speed_up = False
        
        # Map actions
        if action == 1:  # left
            dx = -1
        elif action == 2:  # right
            dx = 1
        elif action == 3:  # rotate
            rotate = True
        elif action == 4:  # down
            speed_up = True
        
        # Move horizontally
        if dx != 0:
            figure_old = deepcopy(self.figure)
            for block in self.figure:
                block[0] += dx
            if not self._check_borders(self.figure):
                self.figure = figure_old
        
        # Rotate
        if rotate:
            figure_old = deepcopy(self.figure)
            self.figure = self._rotate_figure(self.figure)
            if not self._check_borders(self.figure):
                self.figure = figure_old
        
        # Set speed
        current_limit = 100 if speed_up else self.anim_limit
        
        # Move down
        self.anim_count += self.anim_speed
        if self.anim_count > current_limit:
            self.anim_count = 0
            figure_old = deepcopy(self.figure)
            
            for block in self.figure:
                block[1] += 1
            
            if not self._check_borders(self.figure):
                # Lock the figure
                self.figure = figure_old
                self._lock_figure()
                
                # Spawn next figure
                self.current_figure_type = self.next_figure_type
                self.figure = self._create_figure(self.current_figure_type)
                self.next_figure_type = randrange(7)
                self.next_figure = self._create_figure(self.next_figure_type)
                
                self.anim_limit = 2000
                reward += 1  # Reward for placing a piece
                
                # Check for completed lines
                lines_cleared = self._clear_lines()
                if lines_cleared > 0:
                    self.total_lines_cleared += lines_cleared
                    self.anim_speed += 3
                    line_score = self.scores[lines_cleared]
                    self.score += line_score
                    reward += line_score / 100  # Scale down line clear rewards
                
                # Check game over
                if self._check_game_over():
                    self.game_over = True
                    reward = -10  # Penalty for game over
        
        # Small penalty for each step to encourage efficient play
        reward -= 0.01
        
        observation = self._get_observation()
        terminated = self.game_over
        truncated = False
        info = {'score': self.score, 'lines': self.total_lines_cleared}
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return
        
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(self.RES)
            pygame.display.set_caption("Tetris")
        
        if self.game_screen is None:
            self.game_screen = pygame.Surface(self.GAME_RES)
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Create surfaces
        canvas = pygame.Surface(self.RES)
        game_canvas = pygame.Surface(self.GAME_RES)
        
        # Fill backgrounds
        canvas.fill((0, 0, 0))
        game_canvas.fill((20, 20, 40))
        
        # Draw grid
        for x in range(self.W):
            for y in range(self.H):
                rect = pygame.Rect(x * self.TILE, y * self.TILE, self.TILE, self.TILE)
                pygame.draw.rect(game_canvas, (40, 40, 40), rect, 1)
        
        # Draw field
        for y in range(self.H):
            for x in range(self.W):
                if self.field[y][x]:
                    rect = pygame.Rect(x * self.TILE, y * self.TILE, self.TILE - 2, self.TILE - 2)
                    pygame.draw.rect(game_canvas, (100, 200, 100), rect)
        
        # Draw current figure
        for block in self.figure:
            x, y = block
            if 0 <= y < self.H:
                rect = pygame.Rect(x * self.TILE, y * self.TILE, self.TILE - 2, self.TILE - 2)
                pygame.draw.rect(game_canvas, (255, 100, 100), rect)
        
        # Draw next figure
        for block in self.next_figure:
            x, y = block
            rect = pygame.Rect(block[0] * self.TILE + 380, block[1] * self.TILE + 185, 
                             self.TILE - 2, self.TILE - 2)
            pygame.draw.rect(canvas, (100, 100, 255), rect)
        
        # Blit game canvas to main canvas
        canvas.blit(game_canvas, (20, 20))
        
        # Draw text
        try:
            font = pygame.font.Font(None, 36)
        except:
            font = pygame.font.SysFont('arial', 36)
        
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        lines_text = font.render(f'Lines: {self.total_lines_cleared}', True, (255, 255, 255))
        canvas.blit(score_text, (500, 400))
        canvas.blit(lines_text, (500, 450))
        
        if self.render_mode == "human":
            self.screen.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Clean up resources"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None