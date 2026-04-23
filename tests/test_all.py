import numpy as np
import pytest
from simulation.game import PongGame, rule_based_ai
from symbolic.symbolic_model import symbolic_predict
from models.train_nn import PongNet
import torch
import os

def test_physics():
    game = PongGame(render=False)
    initial_x = game.ball_x
    game.update(1, 1)
    assert game.ball_x != initial_x, "Ball should move after update"

def test_symbolic_model_output():
    # Test with a dummy normalized state
    state_norm = np.random.randn(8).astype(np.float32)
    action = symbolic_predict(state_norm)
    assert action in [0, 1, 2], "Symbolic model must output action 0, 1, or 2"

def test_nn_model_output():
    model = PongNet()
    model.load_state_dict(torch.load("models/nn_model.pth"))
    model.eval()
    state_norm = torch.randn(1, 8)
    with torch.no_grad():
        output = model(state_norm)
    assert output.shape == (1, 3), "NN model must output 3 probabilities"

if __name__ == "__main__":
    pytest.main([__file__])
