import pygame
import random
import numpy as np

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 100
BALL_SIZE = 10
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class PongGame:
    def __init__(self, render=True):
        self.should_render = render
        if self.should_render:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("NN Pong Simulation")
            self.clock = pygame.time.Clock()

        # Game State
        self.reset_ball()
        self.paddle1_y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2
        self.paddle2_y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2
        self.paddle1_vy = 0
        self.paddle2_vy = 0
        self.paddle_speed = 5

        self.score1 = 0
        self.score2 = 0
        self.game_over = False

    def reset_ball(self):
        self.ball_x = SCREEN_WIDTH // 2
        self.ball_y = SCREEN_HEIGHT // 2
        self.ball_vx = random.choice([-5, 5])
        self.ball_vy = random.uniform(-3, 3)

    def get_state(self):
        """
        Returns the 8-dimension state vector:
        (ball_x, ball_y, ball_vx, ball_vy, paddle1_y, paddle1_vy, paddle2_y, paddle2_vy)
        """
        return np.array([
            self.ball_x, self.ball_y, self.ball_vx, self.ball_vy,
            self.paddle1_y, self.paddle1_vy, self.paddle2_y, self.paddle2_vy
        ], dtype=np.float32)

    def update(self, action1, action2):
        """
        Actions: 0: Up, 1: Stay, 2: Down
        """
        # Paddle 1 movement
        if action1 == 0: self.paddle1_vy = -self.paddle_speed
        elif action1 == 2: self.paddle1_vy = self.paddle_speed
        else: self.paddle1_vy = 0

        # Paddle 2 movement
        if action2 == 0: self.paddle2_vy = -self.paddle_speed
        elif action2 == 2: self.paddle2_vy = self.paddle_speed
        else: self.paddle2_vy = 0

        self.paddle1_y += self.paddle1_vy
        self.paddle2_y += self.paddle2_vy

        # Boundary checks for paddles
        self.paddle1_y = max(0, min(SCREEN_HEIGHT - PADDLE_HEIGHT, self.paddle1_y))
        self.paddle2_y = max(0, min(SCREEN_HEIGHT - PADDLE_HEIGHT, self.paddle2_y))

        # Ball movement
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Wall collisions (top/bottom)
        if self.ball_y <= 0 or self.ball_y >= SCREEN_HEIGHT - BALL_SIZE:
            self.ball_vy *= -1

        # Paddle collisions
        # Left Paddle
        if self.ball_x <= PADDLE_WIDTH:
            if self.paddle1_y <= self.ball_y <= self.paddle1_y + PADDLE_HEIGHT:
                self.ball_vx *= -1
                # Add some angle variation based on where it hits the paddle
                relative_hit = (self.ball_y - (self.paddle1_y + PADDLE_HEIGHT/2)) / (PADDLE_HEIGHT/2)
                self.ball_vy += relative_hit * 2
            else:
                self.score2 += 1
                self.reset_ball()

        # Right Paddle
        if self.ball_x >= SCREEN_WIDTH - PADDLE_WIDTH - BALL_SIZE:
            if self.paddle2_y <= self.ball_y <= self.paddle2_y + PADDLE_HEIGHT:
                self.ball_vx *= -1
                relative_hit = (self.ball_y - (self.paddle2_y + PADDLE_HEIGHT/2)) / (PADDLE_HEIGHT/2)
                self.ball_vy += relative_hit * 2
            else:
                self.score1 += 1
                self.reset_ball()

    def render(self):
        if not self.should_render: return
        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, WHITE, (0, self.paddle1_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(self.screen, WHITE, (SCREEN_WIDTH - PADDLE_WIDTH, self.paddle2_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.ellipse(self.screen, WHITE, (self.ball_x, self.ball_y, BALL_SIZE, BALL_SIZE))
        pygame.display.flip()

def rule_based_ai(game, paddle_num=1):
    """
    Simple rule-based AI that tries to keep the paddle center aligned with the ball.
    """
    state = game.get_state()
    ball_y = state[1]
    paddle_y = state[4] if paddle_num == 1 else state[6]
    paddle_center = paddle_y + PADDLE_HEIGHT / 2

    if ball_y < paddle_center - 10:
        return 0 # Up
    elif ball_y > paddle_center + 10:
        return 2 # Down
    else:
        return 1 # Stay

if __name__ == "__main__":
    game = PongGame(render=True)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action1 = rule_based_ai(game, 1)
        action2 = rule_based_ai(game, 2)

        game.update(action1, action2)
        game.render()
        game.clock.tick(FPS)
    pygame.quit()
