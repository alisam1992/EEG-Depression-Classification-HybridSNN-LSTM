# EEG-Depression-Classification-HybridSNN-LSTM
Code for "Depression identification using EEG signals via a hybrid of LSTM and spiking neural networks" (Sam et al., IEEE TNSRE 2023).

EEG-Based Depression Classification

"Hybrid of LSTM and Spiking Neural Networks (SNNs)"

This repository contains the full pipeline for the work presented in:

Sam, A., Boostani, R., Hashempour, S., Taghavi, M., & Sanei, S. (2023).
Depression identification using EEG signals via a hybrid of LSTM and spiking neural networks.
IEEE Transactions on Neural Systems and Rehabilitation Engineering, 31, 4725–4737.
DOI: 10.1109/TNSRE.2023.3336467
Zendo:
https://doi.org/10.5281/zenodo.16888569
---

📖 Overview

The algorithm integrates a "Spiking Neural Network (SNN) reservoir" with "LSTM-based classifiers" for EEG-based depression identification. The pipeline consists of three major stages:

1. Preprocessing

   EEG data cleaning and formatting.
   Anonymization and feature extraction.
   Segmentation into epochs using a sliding window with overlap.

2. NeuCube SNN Reservoir

   EEG samples are converted into spike trains using NeuCube.
   The spatio-temporal spiking patterns are encoded across a 3D SNN reservoir.
   The resulting spike outputs are stored in `out_neucube_open.h5`.

3. Classification

   The spike-encoded data are processed with LSTM networks.
   Different tasks supported:

     Regression (depression severity levels).
     Binary classification (e.g., depressed vs. non-depressed).
     Multi-class classification** (non, minimal, moderate, severe).
     Cross-validation (10-fold) ensures robust evaluation.

---

📂 Dataset

The dataset used is publicly available:

J. F. Cavanagh, A. Napolitano, C. Wu, and A. Mueen,
The patient repository for EEG data + computational tools (PRED+CT),
Frontiers Neuroinform., vol. 11, p. 67, Nov. 2017.
[Online]. Available: http://predict.cs.unm.edu/downloads.php

Our preprocessing converts the raw EEG files into spike-encoded samples stored in an HDF5 file (`out_neucube_open.h5`), with:

	"samples" → spike data shaped `(patients × timesteps × neurons)`
	"labels" → depression severity labels

---

⚙️ Requirements

* Python ≥ 3.8
* TensorFlow ≥ 2.8
* Keras ≥ 2.8
* NumPy, Pandas, Scipy
* h5py
* seaborn, matplotlib
* imbalanced-learn
* scikit-learn

Install dependencies:

pip install -r requirements.txt

---

🚀 Pipeline Description

1. Preprocessing

Segmentation: EEG signals are split into 5-second windows with 90% overlap.
Padding/Cropping: Ensures all epochs have equal length.
Label Mapping: Depression labels were mapped into 2 or 4-class categories depending on the task.
HDF5 Storage: Segmented and spike-transformed data are saved in `out_neucube_open.h5`.

2. NeuCube Spiking Reservoir

Raw EEG → spike encoding using NeuCube architecture.
3D SNN reservoir models brain-like spatio-temporal dynamics.
Extracted spike trains serve as input to downstream ML models.

3. Classification (Deep Learning)

Model: A stacked LSTM + Dense feed-forward classifier.

Input shape: (epochs × timesteps × neurons)

Modes:

  Regression (`mode = [0]`)
  Four-class classification (`mode = [4]`)
  Two-class classification (`mode = [i, j]`)

Cross-validation: 10-fold stratified.

Metrics reported: Accuracy, Precision, Recall, F1-score, MSE, RMSE, MAE, R² (depending on task).

Outputs: Results saved as CSV and plots (accuracy curves, confusion matrices, performance metrics).

---

📊 Outputs

All results are automatically saved in the `./model results/` directory:

Accuracy plots (`train_4class.jpg`, `test_4class.jpg`, etc.)
Confusion matrices (`train_4class_CM.jpg`, `test_4class_CM.jpg`)
Performance metrics (CSV):

  Accuracy, sensitivity, specificity
  MSE, RMSE, MAE, R² (for regression)
  Runtime statistics

---

▶️ Running the Code

1. Preprocess EEG into spike-encoded HDF5 (via NeuCube).
2. Run the classification script:

   python preprocessing_clean.py

3. Modify `mode` inside the script to switch between tasks:

   `mode = [0]` → Regression
   `mode = [4]` → 4-class classification
   `mode = [i, j]` → 2-class classification

---

📌 Citation

If you use this code, please cite:

bibtex
@article{sam2023depression,
  title={Depression identification using EEG signals via a hybrid of LSTM and spiking neural networks},
  author={Sam, Ali and Boostani, Reza and Hashempour, Shaghayegh and Taghavi, Maryam and Sanei, Saeid},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  volume={31},
  pages={4725--4737},
  year={2023},
  publisher={IEEE}
}

---

📬 Contact

For questions or collaboration:

Ali Sam – [LinkedIn](https://linkedin.com/in/ali-sam-177937224) | Email: ali.sam1371@gmail.com

---
