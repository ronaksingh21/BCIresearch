#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names=("Negative", "Positive"), title="Confusion Matrix"):
    """
    Plots a confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

def plot_training_curves(losses, accuracies):
    """
    Plots training loss and accuracy vs. epochs.
    losses, accuracies : list of floats
    """
    epochs = range(1, len(losses)+1)
    fig, ax1 = plt.subplots()
    
    color = "tab:blue"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(epochs, losses, color=color, marker="o", linestyle="--", label="Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    
    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("Accuracy (%)", color=color)
    ax2.plot(epochs, accuracies, color=color, marker="s", label="Accuracy")
    ax2.tick_params(axis="y", labelcolor=color)
    
    fig.tight_layout()
    plt.title("Training Loss and Accuracy Trends")
    plt.show()

if __name__ == "__main__":
    # EXAMPLE USAGE:
    # Suppose you've logged some data during training or loaded from a file:
    example_losses = [0.9, 0.7, 0.5, 0.3, 0.2]
    example_accs   = [30, 50, 65, 80, 90]

    plot_training_curves(example_losses, example_accs)
    
    # For the confusion matrix example:
    y_true = [0, 0, 1, 1, 0, 1, 1]
    y_pred = [0, 1, 1, 1, 0, 0, 1]
    plot_confusion_matrix(y_true, y_pred, class_names=["Neg","Pos"])
