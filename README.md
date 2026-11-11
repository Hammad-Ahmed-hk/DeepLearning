# Deep Learning

Two deep learning and computer vision projects built for university coursework and a Kaggle competition.

## Projects

### 🐦 [BirdCLEF+ 2026: Acoustic Species Identification](BirdCLEF_Project)
A pipeline for the BirdCLEF+ 2026 Kaggle competition. It converts 5-second wildlife audio recordings into Mel spectrograms and classifies species with pretrained EfficientNet models (multilabel, 234 species).

- Best public leaderboard score: **0.738** (macro-averaged ROC-AUC)
- Contents: training and starter notebooks, a project report (`.docx`), and a leaderboard screenshot
- Full write-up: [`BirdCLEF_Project/README.md`](BirdCLEF_Project/README.md)

### 🧾 [Quiz 02: Automated Document Segmentation](Quiz_02)
A computer-vision pipeline that detects and crops individual payment slips from one composite scan. It uses adaptive margin removal, Canny edges with morphological closing, and a recursive Sobel-based split for stacked receipts.

- Notebook: [`Quiz_02/Deep_Learning.ipynb`](Quiz_02/Deep_Learning.ipynb)
- Write-up: [`Quiz_02/Readme.md`](Quiz_02/Readme.md)

## Tech stack

Python, PyTorch, librosa (audio), OpenCV, NumPy, Jupyter, Kaggle Notebooks (GPU)

## Running the notebooks

```bash
git clone https://github.com/Hammad-Ahmed-hk/DeepLearning.git
cd DeepLearning
pip install jupyter numpy opencv-python
jupyter notebook
```

For the BirdCLEF notebooks also install `torch` and `librosa`.

The BirdCLEF notebooks were written for Kaggle and expect the competition data, so run them there.

## Author

Hammad Ahmed, [@Hammad-Ahmed-hk](https://github.com/Hammad-Ahmed-hk)
