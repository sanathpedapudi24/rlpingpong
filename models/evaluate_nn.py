import torch
import torch.nn as nn
import numpy as np
import os
from simulation.game import PongGame, rule_based_ai
from models.train_nn import PongNet

# --- Configuration ---
MODEL_FILE = "models/nn_model.pth"
NORM_FILE = "models/norm_params.npz"
NUM_TEST_GAMES = 1000
RALLY_TARGET = 50

def evaluate_nn():
    # Load model
    model = PongNet()
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()

    # Load normalization parameters
    norm = np.load(NORM_FILE)
    mean = norm['mean']
    std = norm['std']

    total_rallies = 0
    total_hits = 0
    game_count = 0

    print(f"Evaluating NN model over {NUM_TEST_GAMES} games...")

    while game_count < NUM_TEST_GAMES:
        game = PongGame(render=False)
        hits = 0
        rally_active = True

        while rally_active:
            state = game.get_state()

            # Normalize and predict for paddle 1
            normalized_state = (state - mean) / (std + 1e-8)
            state_tensor = torch.tensor(normalized_state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = model(state_tensor)
                action1 = torch.argmax(output, dim=1).item()

            # Paddle 2 uses rule-based AI to maintain the game
            action2 = rule_based_ai(game, paddle_num=2)

            # Track if it was a hit before the update
            ball_x_before = game.ball_x
            game.update(action1, action2)

            # Check if paddle 1 hit the ball
            if ball_x_before > PADDLE_WIDTH and game.ball_x <= PADDLE_WIDTH:
                hits += 1

            # Check if game is over (ball passed a paddle)
            # Note: our game.update resets the ball automatically on miss
            # We can detect a reset by checking if ball is back at center
            if game.ball_x == 400 and game.ball_y == 300:
                rally_active = False

        total_hits += hits
        total_rallies += 1
        game_count += 1

        if game_count % 100 == 0:
            print(f"Games completed: {game_count}/{NUM_TEST_GAMES} - Avg Hits: {total_hits/game_count:.2f}")

    avg_rally = total_hits / NUM_TEST_GAMES
    print(f"\nEvaluation Results:")
    print(f"Average Rally Length: {avg_rally:.2f}")
    print(f"Target Rally Length: {RALLY_TARGET}")
    print(f"Success: {'YES' if avg_rally >= RALLY_TARGET else 'NO'}")

if __name__ == "__main__":
    from simulation.game import PADDLE_WIDTH
    evaluate_nn()
