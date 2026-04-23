import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

# --- Configuration ---
DATA_FILE = "data/pong_dataset.npz"
MODEL_FILE = "models/nn_model.pth"
BATCH_SIZE = 256
LR = 0.001
EPOCHS = 50
VAL_SPLIT = 0.2

class PongNet(nn.Module):
    def __init__(self):
        super(PongNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)

def train():
    if not os.path.exists("models"):
        os.makedirs("models")

    # Load data
    data = np.load(DATA_FILE)
    states = data['states']
    actions = data['actions']

    # Normalization
    states_mean = np.mean(states, axis=0)
    states_std = np.std(states, axis=0)
    states = (states - states_mean) / (states_std + 1e-8)

    # Convert to torch tensors
    X = torch.tensor(states, dtype=torch.float32)
    y = torch.tensor(actions, dtype=torch.long)

    # Split train/val
    dataset = TensorDataset(X, y)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = PongNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0
    patience = 5
    trigger_times = 0

    print(f"Training started. Dataset size: {len(states)}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

        # Early Stopping & Best Model Save
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_FILE)
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered")
                break

    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%. Model saved to {MODEL_FILE}")

    # Save normalization parameters for later use
    np.savez("models/norm_params.npz", mean=states_mean, std=states_std)

if __name__ == "__main__":
    train()
