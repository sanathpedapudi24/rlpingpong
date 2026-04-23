# Neural Network Pong to Symbolic Regression

This project implements a pipeline that evolves a Pong game simulation into a neural network, and finally distills that neural network into an interpretable symbolic mathematical model.

## Project Architecture
1. **Simulation**: A Pygame-based Pong engine with physics and a rule-based AI.
2. **Data Collection**: Generation of 500,000+ state-action pairs.
3. **Neural Network**: A PyTorch MLP trained to mimic the rule-based AI.
4. **Symbolic Regression**: Using PySR to find a closed-form equation that approximates the NN.
5. **Integration**: Plugging the symbolic equation back into the game for autonomous play.

## Technical Stack
- Python 3.9+
- PyTorch
- Pygame
- PySR (Symbolic Regression)
- NumPy, SciPy

## How to Run
### 1. Setup
\`\`\`bash
pip install -r requirements.txt # (If applicable)
# Or use the provided venv
source venv/bin/activate
\`\`\`

### 2. Run Simulation
\`\`\`bash
python3 simulation/game.py
\`\`\`

### 3. Run Symbolic Model Test
\`\`\`bash
export PYTHONPATH=\$PYTHONPATH:.
python3 symbolic/integration_test.py
\`\`\`

## Performance Results
- **NN Accuracy**: > 93% Validation Accuracy.
- **Symbolic Model**: Distilled equation: `sin(x5 + sin(x3 + x5*(-1.6924238))) + 0.97586584`
- **Gameplay**: Capable of 5+ minutes of autonomous gameplay.

## Mathematical Representation
The paddle movement is governed by the distilled symbolic equation which maps the normalized state vector $\mathbf{x}$ (8 dimensions) to a continuous value, which is then rounded to the nearest action (Up, Stay, Down).
# rlpingpong
