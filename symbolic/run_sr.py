import torch
import numpy as np
import os
from pysr import PySRRegressor
from models.train_nn import PongNet

# --- Configuration ---
DATA_FILE = "data/pong_dataset.npz"
MODEL_FILE = "models/nn_model.pth"
NORM_FILE = "models/norm_params.npz"
SYMBOLIC_MODEL_FILE = "symbolic/symbolic_model.py"
SAMPLES_FOR_SR = 10000

def generate_nn_samples():
    print("Generating samples from NN for symbolic regression...")
    # Load data and normalization
    data = np.load(DATA_FILE)
    states = data['states']

    norm = np.load(NORM_FILE)
    states_mean = norm['mean']
    states_std = norm['std']

    # Normalize states
    states_norm = (states - states_mean) / (states_std + 1e-8)

    # Load NN model
    model = PongNet()
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()

    # Sample a subset for SR
    indices = np.random.choice(len(states_norm), SAMPLES_FOR_SR, replace=False)
    X_sample = torch.tensor(states_norm[indices], dtype=torch.float32)

    with torch.no_grad():
        # NN outputs probabilities for 3 actions (Up, Stay, Down)
        # For SR, we want to predict the action (argmax) or the probability of the most likely action.
        # Since SR usually works on continuous targets, we'll target the index of the max probability
        # or the continuous value that the NN's final linear layer produces before softmax.
        # Actually, the most interpretable way is to treat the action as a continuous value (0, 1, 2).
        outputs = model(X_sample)
        y_sample = torch.argmax(outputs, dim=1).numpy()

    return states_norm[indices], y_sample

def run_symbolic_regression(X, y):
    print("Running PySR Symbolic Regression...")
    # Define the regressor
    # We use a limited set of operators for interpretability
    model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "sqrt"],
        model_selection="best",
        maxsize=20, # Constraint: < 20 operations
    )

    model.fit(X, y)

    # Save the best equation
    best_eqn = model.sympy()
    print("Best equation found:", best_eqn)

    # Generate Python code from sympy expression
    # We manually convert the sympy expression to a python-compatible lambda
    import sympy
    python_expr = sympy.simplify(best_eqn)

    # Create a mapping for x0, x1... to numpy array indices
    # The symbolic model expects inputs as x0, x1, etc.
    # We will create a function that takes the state_norm array and maps it.

    with open(SYMBOLIC_MODEL_FILE, "w") as f:
        f.write(f"import numpy as np\n")
        f.write(f"import sympy\n\n")
        f.write(f"def symbolic_predict(state_norm):\n")
        f.write(f"    \"\"\"Predicts action (0: Up, 1: Stay, 2: Down) using symbolic equation.\"\"\"\n")
        f.write(f"    # Equation: {best_eqn}\n")
        f.write(f"    try:\n")
        f.write(f"        # Map x0...x7 to state_norm indices\n")
        f.write(f"        locals_dict = {{f'x{{i}}': state_norm[i] for i in range(len(state_norm))}}\n")
        f.write(f"        # Evaluate the expression using sympy's lambdify for performance\n")
        f.write(f"        expr = {repr(best_eqn)}\n")
        f.write(f"        # Convert sympy expr to a function that takes the state_norm array\n")
        f.write(f"        # For simplicity in the final file, we'll use a pre-calculated lambda\n")
        f.write(f"        # But since we are writing the file, we can just use the sympy expression\n")
        f.write(f"        # We'll use a simpler approach: define the lambda explicitly\n")

    # To make it truly usable, we'll use sympy.lambdify
    func = sympy.lambdify([sympy.Symbol(f'x{i}') for i in range(8)], best_eqn, 'numpy')

    # Write a clean, high-performance version of the predictor
    with open(SYMBOLIC_MODEL_FILE, "w") as f:
        f.write(f"import numpy as np\n\n")
        f.write(f"def symbolic_predict(state_norm):\n")
        f.write(f"    \"\"\"Predicts action (0: Up, 1: Stay, 2: Down) using symbolic equation.\"\"\"\n")
        f.write(f"    # Equation: {best_eqn}\n")
        f.write(f"    try:\n")
        f.write(f"        # Use the compiled numpy-compatible expression\n")
        # We write the lambda as a string
        # Since we can't easily write the compiled binary, we use a Python lambda that mirrors the equation
        # We'll replace x0, x1... with state_norm[0], state_norm[1]...
        py_eqn = str(best_eqn)
        for i in range(8):
            py_eqn = py_eqn.replace(f"x{i}", f"state_norm[{i}]")

        f.write(f"        val = {py_eqn}\n")
        f.write(f"        return int(np.clip(round(val), 0, 2))\n")
        f.write(f"    except Exception:\n")
        f.write(f"        return 1\n")

if __name__ == "__main__":
    X, y = generate_nn_samples()
    run_symbolic_regression(X, y)
