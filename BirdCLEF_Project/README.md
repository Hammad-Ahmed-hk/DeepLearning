# 🐦 BirdCLEF+ 2026 - Acoustic Species Identification

A Deep Learning project developed for the **BirdCLEF+ 2026 Kaggle Competition**, focused on identifying wildlife species from acoustic recordings collected in the **Pantanal region of South America**.

The project converts 5-second audio recordings into **Mel Spectrograms** and uses pretrained **EfficientNet** models for multilabel species classification.

---

## 📌 Project Overview

BirdCLEF+ 2026 is an acoustic species identification challenge involving birds and other wildlife species.

The goal of this project was to build a deep learning pipeline that could analyze environmental audio recordings and predict the probability of **234 species**.

Our training dataset contained:

* **35,549 labeled audio files**
* **206 species** in the training data
* **234 species** required in the final submission
* Audio sampling rate: **32,000 Hz**
* Audio duration: **5 seconds per training window**
* Audio format: `.ogg`

The project was developed primarily using **Kaggle Notebooks** with GPU acceleration.

---

## 🎯 Objectives

The main objectives of this project were:

* Learn the Kaggle competition environment
* Perform exploratory data analysis on audio data
* Convert audio signals into Mel Spectrograms
* Apply deep learning for acoustic classification
* Experiment with different EfficientNet architectures
* Improve model generalization using data augmentation
* Handle multilabel classification
* Build a CPU-based inference pipeline
* Generate Kaggle-compatible predictions
* Systematically improve the public leaderboard score

---

## 🧠 Machine Learning Approach

The project treats audio classification as an image classification problem.

The overall pipeline is:

```text
Audio Recording
       ↓
Load Audio using Librosa
       ↓
Pad / Trim to 5 Seconds
       ↓
Mel Spectrogram
       ↓
Convert Power to Decibels
       ↓
Normalization
       ↓
Data Augmentation
       ↓
EfficientNet CNN
       ↓
Sigmoid Activation
       ↓
Species Probabilities
       ↓
Kaggle Submission
```

---

## 🎵 Audio Preprocessing

Each audio recording was processed using the following steps:

1. Load audio at **32,000 Hz**
2. Pad or trim the recording to exactly **5 seconds**
3. Generate a Mel Spectrogram
4. Use **128 Mel frequency bins**
5. Convert the power spectrum to decibel scale
6. Normalize using mean and standard deviation

### Spectrogram Configuration

| Parameter       |             Value |
| --------------- | ----------------: |
| Sample Rate     |         32,000 Hz |
| Duration        |         5 seconds |
| Samples         |           160,000 |
| Mel Bins        |               128 |
| Frequency Range | 20 Hz - 16,000 Hz |
| Output Shape    |         128 × 313 |
| Format          |            `.ogg` |

---

## 🔄 Data Augmentation

Several augmentation techniques were tested to improve model generalization.

### Augmentations Used

* Noise Injection
* Time Shift
* Gain Change
* Frequency Masking
* Time Masking
* SpecAugment
* Mixup

These techniques helped the models become less sensitive to background noise and variations in wildlife recordings.

---

## 🤖 Model Architectures

Multiple model versions were trained during the project.

| Version | Model            | Epochs | Main Changes                |     Score |
| ------- | ---------------- | -----: | --------------------------- | --------: |
| V1      | EfficientNet-B0  |      3 | Baseline                    |     0.624 |
| V2      | EfficientNet-B2  |     10 | All data + augmentation     |     0.724 |
| V3      | EfficientNet-B4  |     15 | Mixup + SpecAugment         | **0.738** |
| V4      | EfficientNet-B4  |     15 | Without Mixup               |     0.688 |
| V5      | EfficientNet-B4  |     25 | Extended training           |     0.721 |
| V6      | EfficientNetV2-S |     25 | New architecture            |     0.719 |
| V7      | EfficientNetV2-M |     15 | Custom head + warm restarts |     0.723 |

> **Best public leaderboard score: 0.738**

---

## 🏆 Best Model

The best-performing submitted model was:

### EfficientNet-B4 + Mixup + SpecAugment

Configuration:

* Architecture: **EfficientNet-B4**
* Pretrained weights: **ImageNet**
* Epochs: **15**
* Batch Size: **32**
* Loss: **BCEWithLogitsLoss**
* Optimizer: **AdamW**
* Learning Rate: **3e-4**
* Mixup: Enabled
* SpecAugment: Enabled
* Best Training Loss: **0.0084**
* Public Leaderboard Score: **0.738**

The report shows that adding Mixup produced one of the largest improvements during experimentation.

---

## 📊 Evaluation Metric

The competition uses:

### Macro-Averaged ROC-AUC

ROC-AUC measures how well the model ranks positive examples above negative examples for each species.

The score ranges approximately from:

```text
0.5  → Random performance
1.0  → Perfect performance
```

Our best public leaderboard score was:

```text
0.738
```

The baseline score was:

```text
0.624
```

Improvement:

```text
+0.114
18.2% relative improvement
```

---

## 📈 Performance Improvement

The project followed an iterative experimentation strategy.

```text
EfficientNet-B0
Score: 0.624
       ↓
EfficientNet-B2
Score: 0.724
       ↓
EfficientNet-B4 + Mixup + SpecAugment
Score: 0.738
```

The major improvements came from:

* Using more training data
* Increasing model capacity
* Adding Mixup
* Adding SpecAugment
* Using appropriate regularization
* Experimenting with different architectures

---

## ⚙️ Training Strategy

### Loss Function

The project used:

```text
BCEWithLogitsLoss
```

This was selected because the task is a **multilabel classification problem**, where multiple species can potentially be present in an audio recording.

### Optimizer

```text
AdamW
```

### Learning Rate

```text
3e-4
```

Later experiments also used:

```text
2e-4
```

### Weight Decay

```text
1e-4
```

### Gradient Clipping

```text
max_norm = 1.0
```

### Learning Rate Scheduler

```text
CosineAnnealingLR
```

The training process used Kaggle's **T4 ×2 GPU environment** for model training.

---

## 💻 Technologies Used

### Programming Language

* Python 3.12

### Deep Learning

* PyTorch
* timm
* EfficientNet
* EfficientNetV2

### Audio Processing

* librosa
* Mel Spectrograms
* SpecAugment

### Data Processing

* NumPy
* Pandas

### Machine Learning

* scikit-learn
* ROC-AUC
* Multilabel classification

### Development Platforms

* Kaggle Notebooks
* Google Colab
* Git
* GitHub
* Kaggle CLI

The report lists PyTorch, timm, librosa, pandas, NumPy, and scikit-learn as the main libraries used.

---

## 📂 Dataset Structure

The competition dataset contains the following important files and directories:

```text
BirdCLEF+ 2026/
│
├── train_audio/
│   └── Labeled audio recordings
│
├── train_soundscapes/
│   └── Training soundscape recordings
│
├── test_soundscapes/
│   └── Hidden test recordings
│
├── train.csv
│   └── Training labels and metadata
│
├── train_soundscapes_labels.csv
│   └── Soundscape labels
│
├── sample_submission.csv
│   └── Required submission format
│
├── taxonomy.csv
│   └── Species taxonomy
│
└── recording_location.txt
    └── Recording location information
```

---

## 🔮 Inference Pipeline

During inference, the model runs on **CPU**, following the competition requirements.

The process is:

```text
Test Soundscape
       ↓
Load Audio
       ↓
32 kHz Sampling
       ↓
Split into Overlapping 5-Second Windows
       ↓
Mel Spectrogram
       ↓
Normalization
       ↓
Trained EfficientNet Model
       ↓
Sigmoid
       ↓
234 Species Probabilities
       ↓
Submission CSV
```

Each prediction row contains:

* `row_id`
* 234 species probability columns

The row ID follows the format:

```text
filename_timestamp
```

---

## 💾 Model Persistence

One of the challenges was transferring trained models between Kaggle notebook sessions.

The solution was:

```text
Train Model
    ↓
Save .pth Model Weights
    ↓
Create Private Kaggle Dataset
    ↓
Upload Model Version
    ↓
Load Model in Inference Notebook
    ↓
Generate Predictions
```

The custom Kaggle dataset used for model persistence was:

```text
birdclef-2026-model-v2
```

---

## ⚠️ Challenges Faced

During development, several technical challenges were encountered.

### 1. GPU Quota

Kaggle's free environment provided limited GPU hours.

**Solution:**

* Test small experiments on CPU
* Plan GPU training carefully
* Save and reuse model weights
* Avoid unnecessary training runs

### 2. Offline Inference

Competition inference required CPU execution with Internet disabled.

**Solution:**

Model weights were stored in a private Kaggle dataset and loaded during inference.

### 3. Model Architecture Mismatch

Different model architectures could not directly use incompatible saved weights.

**Solution:**

The exact architecture used during training was recreated during inference.

### 4. Dataset Path Problems

Dataset files were sometimes located in unexpected folders.

**Solution:**

The directory structure was searched programmatically using `os.walk()`.

### 5. Overfitting

Longer training without appropriate regularization reduced leaderboard performance.

**Solution:**

Mixup, SpecAugment, weight decay, and careful training were used.

---

## 📚 Key Lessons Learned

This project provided practical experience in:

* Deep learning
* Transfer learning
* Audio classification
* Multilabel classification
* Mel Spectrograms
* Data augmentation
* Mixup
* SpecAugment
* EfficientNet architectures
* Model persistence
* Kaggle competitions
* GPU resource management
* CPU inference
* Git and GitHub
* Experiment tracking

One important finding was that increasing the number of epochs did not automatically improve leaderboard performance. In some experiments, longer training without suitable regularization resulted in overfitting.

---

## 🚀 Future Improvements

Possible future improvements include:

* Use **Google Perch** or **BirdNET** pretrained audio models
* Apply proper cross-validation
* Include training soundscape data
* Build an ensemble of multiple models
* Apply site and time-based ecological priors
* Use pseudo-labeling on unlabeled soundscapes
* Improve post-processing
* Explore specialized pretrained bird-audio models

These improvements are suggested in the project report as future directions.

---

## 👨‍💻 Team

### Pythonophile

**Team Members:**

* **Hammad Ahmed** - `2023-SE-01`
* **Aina Yousaf** - `2023-SE-29`
* **Amin Tariq** - `2023-SE-33`

**Instructor:** Ahmed Khwaja

**Kaggle Username:** `pythonophile`

**Competition:** BirdCLEF+ 2026

---

## 📅 Project Timeline

| Event                | Date           |
| -------------------- | -------------- |
| Competition Start    | March 11, 2026 |
| Project Started      | May 21, 2026   |
| Entry Deadline       | May 27, 2026   |
| Final Submission     | June 3, 2026   |
| Active Participation | ~13 days       |

---

## 🏅 Results Summary

| Metric               |              Result |
| -------------------- | ------------------: |
| Initial Score        |               0.624 |
| Best Public Score    |           **0.738** |
| Improvement          |          **+0.114** |
| Relative Improvement |           **18.2%** |
| Training Samples     |              35,549 |
| Training Species     |                 206 |
| Submission Species   |                 234 |
| Best Architecture    |     EfficientNet-B4 |
| Best Techniques      | Mixup + SpecAugment |
| Model Versions       |                  6+ |
| Platform             |              Kaggle |

---

## 📖 References

1. BirdCLEF+ 2026 Competition, Kaggle
2. Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.*
3. Park, D. S. et al. (2019). *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.*
4. Zhang, H. et al. (2017). *Mixup: Beyond Empirical Risk Minimization.*
5. McFee, B. et al. (2015). *librosa: Audio and Music Signal Analysis in Python.*
6. Warden, P. & Situnayake, D. (2019). *TinyML.*
7. timm, PyTorch Image Models
8. Cornell Lab of Ornithology BirdCLEF Challenge Documentation

---

## ⭐ Project Highlights

```text
🐦 Acoustic Species Identification
🎵 Audio → Mel Spectrogram
🧠 EfficientNet Deep Learning
🔄 Mixup + SpecAugment
📊 Multilabel Classification
⚡ Kaggle GPU Training
💻 CPU Inference
🏆 Best Public ROC-AUC: 0.738
```

---

## 📌 Disclaimer

This repository is an academic semester project developed as part of participation in the BirdCLEF+ 2026 Kaggle competition. The reported score refers to the project's public leaderboard result.
