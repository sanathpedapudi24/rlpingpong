import numpy as np
import os
from simulation.game import PongGame, rule_based_ai

# --- Configuration ---
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "pong_dataset.npz")
NUM_SIMULATIONS = 10000
SAMPLES_PER_SIMULATION = 60 * 10  # Assuming average 10 seconds per rally/game
TOTAL_SAMPLES_TARGET = 500000

def collect_data():
    """
    Runs simulations and collects state-action pairs.
    State: (ball_x, ball_y, ball_vx, ball_vy, paddle1_y, paddle1_vy, paddle2_y, paddle2_vy)
    Action: 0 (Up), 1 (Stay), 2 (Down)
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    all_states = []
    all_actions = []

    game = PongGame(render=False)

    print(f"Starting data collection: Target {TOTAL_SAMPLES_TARGET} samples...")

    sim_count = 0
    while len(all_states) < TOTAL_SAMPLES_TARGET and sim_count < NUM_SIMULATIONS:
        # Reset game for a new simulation run
        game.reset_ball()

        # Each simulation run lasts until a certain number of frames or a score change
        # To ensure diverse data, we run for a fixed duration per simulation
        for _ in range(SAMPLES_PER_SIMULATION):
            if len(all_states) >= TOTAL_SAMPLES_TARGET:
                break

            state = game.get_state()
            # We collect data for paddle 1's behavior
            action = rule_based_ai(game, paddle_num=1)

            all_states.append(state)
            all_actions.append(action)

            # Step the game forward
            # Paddle 2 also uses rule-based AI to keep the game going
            action2 = rule_based_ai(game, paddle_num=2)
            game.update(action, action2)

        sim_count += 1
        if sim_count % 100 == 0:
            print(f"Simulation {sim_count}/{NUM_SIMULATIONS} - Samples: {len(all_states)}/{TOTAL_SAMPLES_TARGET}")

    # Convert to numpy arrays
    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.int64)

    # Save to disk
    np.savez_compressed(DATA_FILE, states=states, actions=actions)
    print(f"Dataset saved to {DATA_FILE}. Shape: states={states.shape}, actions={actions.shape}")

if __name__ == "__main__":
    collect_data()
