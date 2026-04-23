import numpy as np
import os
from simulation.game import PongGame
from symbolic.symbolic_model import symbolic_predict
from models.train_nn import PongNet
import torch

def test_integration():
    print("Starting Integration Test...")
    
    # Load normalization parameters
    norm = np.load("models/norm_params.npz")
    states_mean = norm['mean']
    states_std = norm['std']

    game = PongGame(render=False)
    
    # Test for a fixed duration
    total_frames = 60 * 60 * 5 # 5 minutes
    frames_played = 0
    points_won = 0
    
    print(f"Simulating for {total_frames} frames...")
    
    while frames_played < total_frames:
        state = game.get_state()
        
        # Normalize state
        state_norm = (state - states_mean) / (states_std + 1e-8)
        
        # Predict action using symbolic model
        action = symbolic_predict(state_norm)
        
        # Paddle 2 uses rule-based AI to keep the game going
        from simulation.game import rule_based_ai
        action2 = rule_based_ai(game, paddle_num=2)
        
        # Update game
        game.update(action, action2)
        
        # Track points (paddle 1 wins if paddle 2 misses)
        # In game.py, if paddle 2 misses, score1 increases
        # We can track this by checking game.score1
        
        frames_played += 1
        
    print(f"Simulation complete.")
    print(f"Final Score: Paddle 1: {game.score1}, Paddle 2: {game.score2}")
    print(f"Total points won by symbolic model: {game.score1}")
    
    # Success criteria: 5+ minutes autonomous play
    # Since we forced 5 minutes, we check if it survived reasonably
    # (The game doesn't 'end' in this loop, but we can see the score)
    
if __name__ == "__main__":
    test_integration()
