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

        # Tetris figures definitions
        self.figures_pos = [
            [(-1, 0), (-2, 0), (0, 0), (1, 0)],  # I
            [(0, -1), (-1, -1), (-1, 0), (0, 0)],  # O
            [(-1, 0), (-1, 1), (0, 0), (0, -1)],  # Z
            [(0, 0), (-1, 0), (0, 1), (-1, -1)],  # S
            [(0, 0), (0, -1), (0, 1), (-1, -1)],  # J
            [(0, 0), (0, -1), (0, 1), (1, -1)],  # L
            [(0, 0), (0, -1), (0, 1), (-1, 0)]   # T
        ]

        # Action space: 0=nothing, 1=left, 2=right, 3=rotate, 4=down
        self.action_space = spaces.Discrete(5)

        # Observation space: field + current figure position + next figure type
        # Field: 20x10, Figure info: 4 blocks (x,y) + color, Next figure: type
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.H, self.W),
            dtype=np.uint8
        )

        self.render_mode = render_mode
        self.screen = None
        self.game_sc = None
        self.clock = None

        # Score mapping
        self.scores = {0: 0, 1: 100, 2: 300, 3: 700, 4: 1500}

        self.reset()

    def _get_color(self):
        return (randrange(30, 256), randrange(30, 256), randrange(30, 256))

    def _create_figures(self):
        return [[pygame.Rect(x + self.W // 2, y + 1, 1, 1) for x, y in fig_pos] 
                for fig_pos in self.figures_pos]

    def _check_borders(self):
        for i in range(4):
            if self.figure[i].x < 0 or self.figure[i].x > self.W - 1:
                return False
            elif self.figure[i].y > self.H - 1 or self.field[self.figure[i].y][self.figure[i].x]:
                return False
        return True

    def _is_game_over(self):
        for i in range(self.W):
            if self.field[0][i]:
                return True
        return False

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Initialize pygame if not already done
        if not pygame.get_init():
            pygame.init()

        # Initialize field
        self.field = [[0 for i in range(self.W)] for j in range(self.H)]

        # Initialize figures
        figures = self._create_figures()
        self.figure = deepcopy(choice(figures))
        self.next_figure = deepcopy(choice(figures))
        self.color = self._get_color()
        self.next_color = self._get_color()

        # Game variables
        self.anim_count = 0
        self.anim_speed = 60
        self.anim_limit = 2000
        self.score = 0
        self.total_lines = 0
        self.game_over = False

        return self._get_obs(), {}

    def _get_obs(self):
        # Return field as observation
        obs = np.zeros((self.H, self.W), dtype=np.uint8)
        for y in range(self.H):
            for x in range(self.W):
                if self.field[y][x]:
                    obs[y, x] = 1
        
        # Add current figure to observation
        for i in range(4):
            if 0 <= self.figure[i].y < self.H and 0 <= self.figure[i].x < self.W:
                obs[self.figure[i].y, self.figure[i].x] = 2

        return obs

    def step(self, action):
        if self.game_over:
            return self._get_obs(), 0, True, False, {}

        reward = 0
        dx = 0
        rotate = False
        speed_up = False

        # Map action to controls
        if action == 1:  # left
            dx = -1
        elif action == 2:  # right
            dx = 1
        elif action == 3:  # rotate
            rotate = True
        elif action == 4:  # down
            speed_up = True

        # Move x
        figure_old = deepcopy(self.figure)
        for i in range(4):
            self.figure[i].x += dx
        if not self._check_borders():
            self.figure = deepcopy(figure_old)

        # Move y
        if speed_up:
            current_limit = 100
        else:
            current_limit = self.anim_limit

        self.anim_count += self.anim_speed
        if self.anim_count > current_limit:
            self.anim_count = 0
            figure_old = deepcopy(self.figure)
            for i in range(4):
                self.figure[i].y += 1
            
            if not self._check_borders():
                # Lock figure in place
                for i in range(4):
                    self.field[figure_old[i].y][figure_old[i].x] = self.color
                
                # Spawn new figure
                figures = self._create_figures()
                self.figure = self.next_figure
                self.color = self.next_color
                self.next_figure = deepcopy(choice(figures))
                self.next_color = self._get_color()
                self.anim_limit = 2000
                
                reward += 1  # Reward for placing a piece

        # Rotate
        if rotate:
            center = self.figure[0]
            figure_old = deepcopy(self.figure)
            for i in range(4):
                x = self.figure[i].y - center.y
                y = self.figure[i].x - center.x
                self.figure[i].x = center.x - x
                self.figure[i].y = center.y + y
            if not self._check_borders():
                self.figure = deepcopy(figure_old)

        # Check lines
        line = self.H - 1
        lines_cleared = 0
        for row in range(self.H - 1, -1, -1):
            count = 0
            for i in range(self.W):
                if self.field[row][i]:
                    count += 1
                self.field[line][i] = self.field[row][i]
            if count < self.W:
                line -= 1
            else:
                self.anim_speed += 3
                lines_cleared += 1

        # Compute score and reward
        if lines_cleared > 0:
            line_score = self.scores[lines_cleared]
            self.score += line_score
            reward += line_score
            self.total_lines += lines_cleared

        # Check game over
        terminated = self._is_game_over()
        if terminated:
            self.game_over = True
            reward -= 500  # Penalty for game over

        truncated = False
        info = {
            "score": self.score,
            "lines": self.total_lines,
            "level": self.anim_speed // 60
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode(self.RES)
                pygame.display.set_caption("Tetris")
            if self.game_sc is None:
                self.game_sc = pygame.Surface(self.GAME_RES)
            if self.clock is None:
                self.clock = pygame.time.Clock()

            # Load backgrounds with error handling
            try:
                bg = pygame.Surface(self.RES)
                bg.fill((0, 0, 0))
            except:
                bg = pygame.Surface(self.RES)
                bg.fill((0, 0, 0))

            try:
                game_bg = pygame.Surface(self.GAME_RES)
                game_bg.fill((20, 20, 40))
            except:
                game_bg = pygame.Surface(self.GAME_RES)
                game_bg.fill((20, 20, 40))

            self.screen.blit(bg, (0, 0))
            self.game_sc.blit(game_bg, (0, 0))

            # Draw grid
            for x in range(self.W):
                for y in range(self.H):
                    rect = pygame.Rect(x * self.TILE, y * self.TILE, self.TILE, self.TILE)
                    pygame.draw.rect(self.game_sc, (40, 40, 40), rect, 1)

            # Draw field
            figure_rect = pygame.Rect(0, 0, self.TILE - 2, self.TILE - 2)
            for y, raw in enumerate(self.field):
                for x, col in enumerate(raw):
                    if col:
                        figure_rect.x, figure_rect.y = x * self.TILE, y * self.TILE
                        pygame.draw.rect(self.game_sc, col, figure_rect)

            # Draw current figure
            for i in range(4):
                figure_rect.x = self.figure[i].x * self.TILE
                figure_rect.y = self.figure[i].y * self.TILE
                pygame.draw.rect(self.game_sc, self.color, figure_rect)

            self.screen.blit(self.game_sc, (20, 20))

            # Draw next figure
            for i in range(4):
                figure_rect.x = self.next_figure[i].x * self.TILE + 380
                figure_rect.y = self.next_figure[i].y * self.TILE + 185
                pygame.draw.rect(self.screen, self.next_color, figure_rect)

            # Font rendering with robust error handling
            try:
                main_font = pygame.font.Font(None, 65)
                font = pygame.font.Font(None, 45)
                
                title_tetris = main_font.render('TETRIS', True, pygame.Color('darkorange'))
                title_score = font.render('score:', True, pygame.Color('green'))
                title_record = font.render('record:', True, pygame.Color('purple'))
                score_text = font.render(str(self.score), True, pygame.Color('white'))
                
                self.screen.blit(title_tetris, (485, -10))
                self.screen.blit(title_score, (535, 780))
                self.screen.blit(score_text, (550, 840))
                self.screen.blit(title_record, (525, 650))
            except Exception:
                try:
                    main_font = pygame.font.SysFont('arial', 65)
                    font = pygame.font.SysFont('arial', 45)
                    
                    title_tetris = main_font.render('TETRIS', True, pygame.Color('darkorange'))
                    title_score = font.render('score:', True, pygame.Color('green'))
                    title_record = font.render('record:', True, pygame.Color('purple'))
                    score_text = font.render(str(self.score), True, pygame.Color('white'))
                    
                    self.screen.blit(title_tetris, (485, -10))
                    self.screen.blit(title_score, (535, 780))
                    self.screen.blit(score_text, (550, 840))
                    self.screen.blit(title_record, (525, 650))
                except Exception:
                    pass

            pygame.display.flip()
            if self.clock:
                self.clock.tick(self.FPS)

        elif self.render_mode == "rgb_array":
            if self.game_sc is None:
                self.game_sc = pygame.Surface(self.GAME_RES)
            
            game_bg = pygame.Surface(self.GAME_RES)
            game_bg.fill((20, 20, 40))
            self.game_sc.blit(game_bg, (0, 0))

            # Draw grid
            for x in range(self.W):
                for y in range(self.H):
                    rect = pygame.Rect(x * self.TILE, y * self.TILE, self.TILE, self.TILE)
                    pygame.draw.rect(self.game_sc, (40, 40, 40), rect, 1)

            # Draw field
            figure_rect = pygame.Rect(0, 0, self.TILE - 2, self.TILE - 2)
            for y, raw in enumerate(self.field):
                for x, col in enumerate(raw):
                    if col:
                        figure_rect.x, figure_rect.y = x * self.TILE, y * self.TILE
                        pygame.draw.rect(self.game_sc, col, figure_rect)

            # Draw current figure
            for i in range(4):
                figure_rect.x = self.figure[i].x * self.TILE
                figure_rect.y = self.figure[i].y * self.TILE
                pygame.draw.rect(self.game_sc, self.color, figure_rect)

            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.game_sc)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.game_sc = None
            self.clock = None