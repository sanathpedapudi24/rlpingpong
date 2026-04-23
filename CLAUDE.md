# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
A progressive AI system that evolves from a Pong game simulation to a neural network (NN) and finally to an interpretable symbolic regression model. The goal is to extract mathematical equations that replicate superhuman Pong gameplay.

## Technical Stack
- **Language**: Python 3.9+
- **Frameworks**: PyTorch 2.0+, Pygame
- **Libraries**: NumPy, SciPy, scikit-learn, PySR (Symbolic Regression), SymPy
- **Testing**: pytest

## Project Pipeline & Architecture
The project is structured into four phases:
1. **Pong Simulation**: Game engine with physics, state representation, and automated data collection (state-action pairs).
2. **Neural Network Training**: Predicting paddle movement (up/stay/down) using a multi-layer perceptron (MLP) with ReLU and Softmax.
3. **Symbolic Regression**: Distilling the NN's behavior into closed-form mathematical expressions using PySR and Genetic Programming.
4. **Integration & Validation**: Integrating symbolic models back into the game to verify performance (Target: 5+ minutes autonomous gameplay).

## Common Development Tasks
- **Run Tests**: `pytest`
- **Training/Simulation**: Likely executed via Python scripts or Jupyter Notebooks.
- **Physics Validation**: Unit tests for collision and movement logic.
- **Model Evaluation**: Comparing R² and gameplay duration across NN and symbolic models.
