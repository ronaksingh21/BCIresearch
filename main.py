#!/usr/bin/env python3
#all code was commented and modified using copilot to make it easier for users to understand
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mne
from mne.datasets import sample
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, classification_report
import random
import matplotlib.pyplot as plt

# -----------------------------
# 1. SETUP & CONFIG
# -----------------------------

# Hyperparameters
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 32
SYNTHETIC_SAMPLES = 240    # total synthetic samples (half positive, half negative)
SYNTH_LEN = 226            # length of each synthetic EEG sample
P300_AMP_RANGE = (2.0, 5.0) # amplitude range for simulated P300
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# 2. SYNTHETIC DATA GENERATION
# -----------------------------
def generate_synthetic_eeg_data(n_samples=240, sample_len=226, p300_amp_range=(2,5)):
    """
    Generates synthetic EEG data simulating P300 (positive) vs. non-P300 (negative) signals.
    Returns: (X, y) as numpy arrays
    """
    half = n_samples // 2
    
    # Positive (P300-like) class
    p300_data = []
    for _ in range(half):
        # Baseline noise
        signal = np.random.normal(loc=0.0, scale=0.02, size=sample_len)
        # Simulated P300 peak ~300 ms => let's place it around index ~60-80 if sampling ~200 Hz
        peak_idx = random.randint(60, 80)
        peak_amp = random.uniform(*p300_amp_range) / 10.0   # scale amplitude
        signal[peak_idx:peak_idx+5] += peak_amp
        p300_data.append(signal)
    
    # Negative (non-P300) class
    nonp300_data = []
    for _ in range(half):
        # Baseline noise
        signal = np.random.normal(loc=0.0, scale=0.02, size=sample_len)
        # no strong peak
        nonp300_data.append(signal)
    
    X = np.vstack([p300_data, nonp300_data])  # shape: (n_samples, sample_len)
    y = np.array([1]*half + [0]*half)
    return X, y

# -----------------------------
# 3. LOAD & PREPROCESS REAL DATA (MNE)
# -----------------------------
def load_mne_data():
    """
    Loads MNE sample dataset (EEG subset), returns (X, y) after minimal preprocessing.
    For demonstration, we label a small portion as "positive" and "negative."
    """
    data_path = sample.data_path()
    raw_fname = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_filt-0-40_raw.fif')
    
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.set_eeg_reference(projection=True)
    raw.pick_types(meg=False, eeg=True, eog=True)
    
    # Example: We'll artificially define events 1 => "positive" and 2 => "negative"
    # In practice, you would load real event IDs from raw events.
    events = mne.find_events(raw)
    event_id = dict(Positive=1, Negative=2)  # dummy event dict
    tmin, tmax = -0.2, 0.5
    try:
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                            baseline=(None, 0), preload=True, verbose=False)
    except ValueError:
        print("Warning: No matching events found, returning dummy arrays.")
        return np.empty((0,226)), np.array([])
    
    # Flatten & scale
    X = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    X = X.reshape((X.shape[0], -1))  # Flatten
    y_labels = epochs.events[:,-1] - 1  # => 0 or 1
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_labels

# -----------------------------
# 4. DEFINE A SIMPLE PYTORCH MODEL
# -----------------------------
class SimpleBCIModel(nn.Module):
    def __init__(self, input_dim, hidden1=500, hidden2=1000, hidden3=100, output_dim=2):
        super(SimpleBCIModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.act1 = nn.CELU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden3, output_dim)
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(x)
        return self.soft(x)

# -----------------------------
# 5. TRAIN & EVALUATE
# -----------------------------
def train_epoch(model, optimizer, criterion, X_train, y_train):
    model.train()
    shuffle_idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[shuffle_idx], y_train[shuffle_idx]
    
    num_batches = int(np.ceil(len(X_train)/BATCH_SIZE))
    
    total_loss = 0.0
    for i in range(num_batches):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        xb = torch.tensor(X_train[start:end], dtype=torch.float32)
        yb = torch.tensor(y_train[start:end], dtype=torch.long)
        
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / num_batches

def evaluate_model(model, X_test, y_test):
    model.eval()
    xb = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        preds = model(xb)
        preds_class = torch.argmax(preds, dim=1).numpy()
    return preds_class

# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":
    # 1) Generate Synthetic Data
    X_synth, y_synth = generate_synthetic_eeg_data(
        n_samples=SYNTHETIC_SAMPLES,
        sample_len=SYNTH_LEN,
        p300_amp_range=P300_AMP_RANGE
    )
    
    # 2) Load Real EEG Data from MNE
    X_real, y_real = load_mne_data()
    if len(X_real) == 0:
        print("No real data found. Proceeding with synthetic only.")
    
    # 3) Combine or keep separate for 'hybrid' approach
    # For demonstration, let's do: pre-train on synthetic, then fine-tune on real
    # We'll shape them to the same dimension if necessary:
    input_dim = X_synth.shape[1]
    
    model = SimpleBCIModel(input_dim=input_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # Convert X_synth / y_synth to numpy
    print("Pre-training on synthetic data...")
    EPOCHS_SYNTH = 10
    for epoch in range(EPOCHS_SYNTH):
        loss_val = train_epoch(model, optimizer, criterion, X_synth, y_synth)
        if epoch % 2 == 0:
            print(f"Epoch {epoch}/{EPOCHS_SYNTH}, Loss: {loss_val:.4f}")
    
    # Evaluate on synthetic (optional)
    preds_synth = evaluate_model(model, X_synth, y_synth)
    synth_acc = np.mean(preds_synth == y_synth)
    print(f"Synthetic data accuracy: {synth_acc:.2f}")
    
    # Fine-tuning on real data (if available)
    if len(X_real) > 0:
        print("\nFine-tuning on real EEG data...")
        EPOCHS_REAL = 10
        for epoch in range(EPOCHS_REAL):
            loss_val = train_epoch(model, optimizer, criterion, X_real, y_real)
            if epoch % 2 == 0:
                print(f"Real-data Epoch {epoch}/{EPOCHS_REAL}, Loss: {loss_val:.4f}")
        
        # Evaluate on real data
        preds_real = evaluate_model(model, X_real, y_real)
        real_acc = np.mean(preds_real == y_real)
        print(f"Real data accuracy: {real_acc:.2f}")
        
        # Show confusion matrix & classification report
        cm = confusion_matrix(y_real, preds_real)
        cr = classification_report(y_real, preds_real, target_names=["Negative", "Positive"])
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", cr)
    else:
        print("Skipping real-data fine-tuning since MNE data was not found or events were missing.")
    
    print("\nDone. You can now run `plot_results.py` to visualize confusion matrix or training trends.")
