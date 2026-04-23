import numpy as np

def symbolic_predict(state_norm):
    """Predicts action (0: Up, 1: Stay, 2: Down) using symbolic equation."""
    # Equation: sin(x5 + sin(x3 + x5*(-1.6924238))) + 0.97586584
    try:
        # Use the compiled numpy-compatible expression
        val = np.sin(state_norm[5] + np.sin(state_norm[3] + state_norm[5]*(-1.6924238))) + 0.97586584
        return int(np.clip(round(val), 0, 2))
    except Exception:
        return 1
