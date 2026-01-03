import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

class PongEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode=None, opponent_skill=0.5, fast_mode=True):
        super().__init__()
        
        # Game parameters
        self.width = 600
        self.height = 400
        self.paddle_width = 10
        self.paddle_height = 60
        self.ball_size = 10
        self.paddle_speed = 6  # Increased from 5
        self.ball_speed_x = 5  # Increased from 4
        self.ball_speed_y = 5  # Increased from 4
        self.opponent_skill = opponent_skill
        self.fast_mode = fast_mode  # Skip some physics checks for speed
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)
        
        # Normalized observations for better learning
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0,
            shape=(8,),  # Extended state space
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize positions
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_vel_x = self.ball_speed_x * (1 if np.random.random() > 0.5 else -1)
        self.ball_vel_y = self.ball_speed_y * (np.random.random() * 2 - 1)  # Random angle
        
        self.paddle_y = self.height // 2 - self.paddle_height // 2
        self.opponent_y = self.height // 2 - self.paddle_height // 2
        
        self.score_agent = 0
        self.score_opponent = 0
        self.rally_length = 0  # Track rally length
        self.last_hit_by_agent = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def _get_obs(self):
        """Normalized observations for better learning"""
        return np.array([
            # Ball position (normalized)
            (self.ball_x / self.width) * 2 - 1,
            (self.ball_y / self.height) * 2 - 1,
            # Ball velocity (normalized)
            self.ball_vel_x / self.ball_speed_x,
            self.ball_vel_y / self.ball_speed_y,
            # Paddle positions (normalized)
            (self.paddle_y / self.height) * 2 - 1,
            (self.opponent_y / self.height) * 2 - 1,
            # Distance from ball to paddle (normalized)
            ((self.paddle_y + self.paddle_height/2) - self.ball_y) / self.height,
            # Ball direction indicator (is it coming towards agent?)
            1.0 if self.ball_vel_x > 0 else -1.0
        ], dtype=np.float32)
    
    def _get_info(self):
        return {
            "score_agent": self.score_agent,
            "score_opponent": self.score_opponent,
            "rally_length": self.rally_length
        }
    
    def step(self, action):
        # Move agent paddle
        if action == 1:  # Move up
            self.paddle_y = max(0, self.paddle_y - self.paddle_speed)
        elif action == 2:  # Move down
            self.paddle_y = min(self.height - self.paddle_height, 
                               self.paddle_y + self.paddle_speed)
        
        # Smarter opponent AI with reaction time
        if np.random.random() < self.opponent_skill:
            # Predict where ball will be
            if self.ball_vel_x < 0:  # Ball moving towards opponent
                target_y = self.ball_y
            else:
                # Move to center when ball is away
                target_y = self.height // 2
            
            opponent_center = self.opponent_y + self.paddle_height // 2
            diff = target_y - opponent_center
            
            if abs(diff) > 5:  # Dead zone to prevent jittering
                move_speed = self.paddle_speed * 0.8
                if diff < 0:
                    self.opponent_y = max(0, self.opponent_y - move_speed)
                else:
                    self.opponent_y = min(self.height - self.paddle_height, 
                                         self.opponent_y + move_speed)
        
        # Move ball
        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y
        
        # Ball collision with top/bottom walls
        if self.ball_y <= 0 or self.ball_y >= self.height - self.ball_size:
            self.ball_vel_y *= -1
            self.ball_y = np.clip(self.ball_y, 0, self.height - self.ball_size)
        
        reward = 0
        terminated = False
        
        # OPTIMIZED REWARD SHAPING for faster learning
        
        # 1. Positioning reward - encourage paddle to track ball
        if self.ball_vel_x > 0:  # Ball coming towards agent
            paddle_center = self.paddle_y + self.paddle_height / 2
            distance_to_ball = abs(paddle_center - self.ball_y)
            # Stronger proximity reward when ball is close
            proximity_factor = 1.0 - (self.ball_x / self.width)
            proximity_reward = 0.05 * (1.0 - distance_to_ball / self.height) * proximity_factor
            reward += proximity_reward
        
        # 2. Ball collision with agent paddle (right side)
        if (self.ball_x + self.ball_size >= self.width - self.paddle_width and
            self.ball_vel_x > 0 and
            self.paddle_y <= self.ball_y + self.ball_size and
            self.ball_y <= self.paddle_y + self.paddle_height):
            
            # Calculate hit position for angle control
            hit_pos = (self.ball_y - self.paddle_y) / self.paddle_height
            angle_factor = (hit_pos - 0.5) * 2  # -1 to 1
            
            self.ball_vel_x *= -1.05  # Slight speed increase
            self.ball_vel_y = self.ball_speed_y * angle_factor * 1.5
            self.ball_x = self.width - self.paddle_width - self.ball_size
            
            self.rally_length += 1
            self.last_hit_by_agent = True
            
            # Reward based on rally length (encourage longer rallies)
            hit_reward = 3.0 + min(self.rally_length * 0.5, 5.0)
            reward += hit_reward
        
        # 3. Ball collision with opponent paddle (left side)
        if (self.ball_x <= self.paddle_width and
            self.ball_vel_x < 0 and
            self.opponent_y <= self.ball_y + self.ball_size and
            self.ball_y <= self.opponent_y + self.paddle_height):
            
            hit_pos = (self.ball_y - self.opponent_y) / self.paddle_height
            angle_factor = (hit_pos - 0.5) * 2
            
            self.ball_vel_x *= -1.05
            self.ball_vel_y = self.ball_speed_y * angle_factor * 1.5
            self.ball_x = self.paddle_width
            
            self.rally_length += 1
            self.last_hit_by_agent = False
        
        # 4. Scoring
        if self.ball_x < 0:
            # Agent scored!
            self.score_agent += 1
            base_reward = 10.0
            # Bonus for rally length
            rally_bonus = min(self.rally_length * 1.0, 10.0)
            reward += base_reward + rally_bonus
            terminated = True
            
        elif self.ball_x > self.width:
            # Opponent scored
            self.score_opponent += 1
            penalty = -10.0
            # Extra penalty if agent missed after hitting
            if self.last_hit_by_agent:
                penalty -= 5.0
            reward += penalty
            terminated = True
        
        # 5. Small penalty for staying still when ball is approaching
        if action == 0 and self.ball_vel_x > 0:
            reward -= 0.01
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, False, info
    
    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Pong RL Agent - Advanced Training")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((0, 0, 0))
        
        # Draw paddles
        pygame.draw.rect(canvas, (255, 255, 255), 
                        (self.width - self.paddle_width, self.paddle_y, 
                         self.paddle_width, self.paddle_height))
        pygame.draw.rect(canvas, (255, 255, 255), 
                        (0, self.opponent_y, self.paddle_width, self.paddle_height))
        
        # Draw ball
        pygame.draw.rect(canvas, (255, 255, 255), 
                        (self.ball_x, self.ball_y, self.ball_size, self.ball_size))
        
        # Draw center line
        for i in range(0, self.height, 20):
            pygame.draw.rect(canvas, (100, 100, 100), 
                           (self.width // 2 - 2, i, 4, 10))
        
        # Draw scores
        font = pygame.font.Font(None, 74)
        text_opponent = font.render(str(self.score_opponent), True, (255, 255, 255))
        text_agent = font.render(str(self.score_agent), True, (255, 255, 255))
        canvas.blit(text_opponent, (self.width // 4, 20))
        canvas.blit(text_agent, (3 * self.width // 4, 20))
        
        # Draw rally counter
        small_font = pygame.font.Font(None, 36)
        rally_text = small_font.render(f"Rally: {self.rally_length}", True, (150, 150, 150))
        canvas.blit(rally_text, (self.width // 2 - 50, self.height - 30))
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()