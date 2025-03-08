# Hybrid EEG Classification for Brain-Computer Interfaces

This repository demonstrates a **hybrid training** approach for **EEG (Electroencephalography) classification** in **Brain-Computer Interface (BCI)** applications. The approach **pre-trains** a neural network on **synthetic EEG data** and **fine-tunes** it with **real-world EEG signals** to achieve higher accuracy, faster convergence, and improved robustness.

## Table of Contents
1. [Overview](#overview)  
2. [Project Structure](#project-structure)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Scripts](#scripts)  
6. [Data](#data)  
7. [Results](#results)  
8. [References](#references)  

---

## 1. Overview
- **Goal**: Enhance EEG classification for real-time BCI applications using **hybrid training**.  
- **Key Features**:
  - Synthetic data generation (P300 vs. non-P300) for baseline learning.  
  - Fine-tuning on **MNE** sample dataset for real-world adaptability.  
  - Comprehensive PyTorch model with **CELU, ReLU** activations and **dropout** to prevent overfitting.  
  - Evaluation metrics: **accuracy**, **confusion matrix**, **precision/recall/F1**.  

---

## 2. Project Structure
my-bci-research/ │
 ├── README.md # This file
  ├── requirements.txt # Python dependencies 
  ├── main.py # Main script: synthetic data gen, real data loading, training, evaluation ├── plot_results.py # Utility script for plotting confusion matrix & training curves └── LICENSE

---

## 3. Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/ronaksingh21/BCIresearch.git
   cd BCIresearch

   Install Python dependencies
pip install -r requirements.txt

run the main script
python main.py

What Happens:
Generates synthetic EEG data simulating P300 vs. non-P300 signals.
Loads real EEG data from the MNE Sample Dataset, performs minimal preprocessing.
Pre-trains a neural network on synthetic data, then fine-tunes on real data if available.
Displays classification performance (accuracy, confusion matrix) in the console.

Step 2: (Optional) Visualize Results
python plot_results.py


Scripts
main.py

Purpose:

Generate synthetic data: Models P300 peaks, Gaussian noise, random spikes.

Load MNE dataset: Gains real EEG examples (MNE Sample Dataset).

Define PyTorch model: (CELU, ReLU activations, Softmax output).

Train model:
Pre-train on synthetic data.

Fine-tune on real data.

Evaluate: Prints accuracy, confusion matrix, classification report.

plot_results.py

Purpose:

Plots confusion matrix using matplotlib or seaborn.

Draws training curves (loss vs. epochs, accuracy vs. epochs).

Currently demonstrates placeholder arrays—modify to load actual training logs from main.py.


6. Data

Synthetic EEG Data

Automatically generated in main.py.

Simulates P300 waveforms around ~300 ms with configurable noise levels.

MNE Sample Dataset

Downloaded automatically by MNE-Python if not already present.

Contains EEG (and MEG) from an adult participant in an auditory-visual experiment.

Preprocessing includes band-pass filtering (1–40 Hz) and artifact rejection via EOG channels.

If you prefer a different dataset, modify load_mne_data() in main.py accordingly.


7. Results
Accuracy: Typically ~75–76% on real MNE data (depending on subset used).

Inference Time: ~3.97 ms per EEG sample (PyTorch CPU) for the final model.

Confusion Matrix: Showcases how many positive vs. negative samples are correctly identified.

E.g
Confusion Matrix:

 [[47  9]

  [ 1 43]]

Classification Report:
               precision    recall  f1-score   support
     ...

