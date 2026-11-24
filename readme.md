# Musicify: GTZAN Music Genre Classification 🎵

An end-to-end pipeline implementing both classic hand-crafted audio feature engineering machine learning baselines and a high-performance deep convolutional neural network (CNN) trained from scratch to classify 10 distinct musical genres.

---

## 📊 Pipeline Summary
- **Dataset:** GTZAN (10 genres, 1000 original clips of 30 seconds each, sampled at 16 kHz mono).
- **Validation Strategy:** Rigorous leak-free song-level split (80% Train, 10% Val, 10% Test) preserving identical class distributions via stratification.
- **Data Augmentation:** Gain Jitter (dB-scale amplitude shift), Time & Frequency SpecAugment Masking, and Mixup augmentation during neural network training.

---

## 🏆 Results & Evaluation

Each test song is scored by averaging predictions across all overlapping 3-second segments (10 windows per song).

| Model | Validation Song Acc | Test Song Acc |
|---|---|---|
| **CNN (Trained from Scratch)** | **86.0%** | **91.0%** |
| **CNN + Random Forest Ensemble** | **88.0%** | **92.0%** |
| **SVM (RBF) Baseline** | 81.0% | 86.0% |
| **Gradient Boosting Baseline** | 81.0% | 81.0% |
| **Random Forest Baseline** | 83.0% | 77.0% |
| **Extra Trees Baseline** | 79.0% | 75.0% |

### CNN Test Confusion Matrix
Accuracy of **91.0%** achieved on the test partition. Major classification patterns include:
- **100% Perfect Recognition:** Classical, Jazz
- **Minor confusion:** Pop vs. Rock/Reggae, Rock vs. Pop/Country

---

## 🛠 Feature Engineering (Classical ML)
We extracted **91 segment-level features** across overlapping 3-second windows, including:
- **Mel-Frequency Cepstral Coefficients (MFCCs)** (20 coefficients: mean, std)
- **Chroma STFT** (12 bins: mean, std)
- **Spectral Centroid, Bandwidth, Rolloff, Flatness, and Contrast**
- **Zero Crossing Rate (ZCR)** & **Root-Mean-Square Energy (RMS)**
- **Global Tempo** (estimated using onset envelope strength)

---

## 🧠 Deep Learning Architecture (GenreCNN)
A custom Pytorch architecture (~1.18M parameters) built from scratch without pre-trained checkpoints:
- **Front-end Input:** Log-mel spectrogram representation ($128 \\times 150$ frames equivalent to 3-second crops).
- **Back-end CNN:** Four sequential 2D convolutional blocks with Batch Normalization, Max Pooling, and Dropout.
- **Aggregation:** Global statistical pool (Time-Mean + Time-Max concat) mapped through a regularized fully connected classification head.
- **Optimizer:** AdamW with OneCycleLR learning rate scheduler peaking at $2 \\times 10^{{-3}}$, Label Smoothing of 0.1, and Weight Decay of 0.05.

---

## 🚀 Hugging Face Hub Integration
The trained model, config, preprocessing weights, metrics, and PyTorch inference pipeline are uploaded and served live on Hugging Face:
🔗 [Hugging Face Model Page](https://huggingface.co/anmolsehgal/gtzan-genre-cnn)
