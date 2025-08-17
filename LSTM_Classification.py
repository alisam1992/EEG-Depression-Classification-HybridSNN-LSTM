"""
Final Stage: Sequence Classifier on SNN Outputs
-----------------------------------------------
- Loads spike rasters from out_neucube_open.h5 (N, T, 1471), labels (0..3)
- Windows sequences with 5s at 250 Hz and 90% overlap
- Class-balances via class-specific segment counts (as in original)
- Trains an LSTM classifier with 10-fold Stratified CV
- Saves metrics, confusion matrices, and plots

Outputs go to:
  ./model results/
  ./model results/csv results/
"""

from __future__ import annotations

import os
import gc
from time import time
from typing import Tuple, List

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
    f1_score,
    recall_score,
    precision_score,
)

# Keras / TensorFlow (modern API)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.backend import clear_session


# -----------------------------
# Config
# -----------------------------

H5_PATH = "out_neucube_open.h5"           # input from SNN stage
RES_DIR = "./model results"
CSV_DIR = "./model results/csv results"

FS = 250          # Hz
WIN_SEC = 5       # seconds
OLP = 0.9         # 90% overlap

# class-specific number of segments per trial (original settings)
SEGMENTS_PER_CLASS = {0: 30, 1: 83, 2: 48, 3: 232}

# Task mode:
#  - [4]  : 4-class classification (default)
#  - [i,j]: 2-class classification (keeps only classes i and j)
#  - [0]  : regression (not commonly used here; kept for parity)
MODE = [4]

RANDOM_STATE = 42
N_SPLITS = 10
CLASS_NAMES_4 = ["0", "1", "2", "3"]


# -----------------------------
# Utils
# -----------------------------

def ensure_dirs() -> None:
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)


def labels_to_4(labels: np.ndarray) -> np.ndarray:
    """Assumes labels already in 0..3; function kept for parity/clarity."""
    labels = np.asarray(labels).astype(np.int32)
    return labels


def filter_to_two_classes(X: np.ndarray, y: np.ndarray, keep: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only specified classes, reindex them to 0..1 preserving order in keep."""
    mask = np.isin(y, keep)
    X2 = X[mask]
    y2 = y[mask]
    mapping = {keep[0]: 0, keep[1]: 1}
    y2 = np.vectorize(mapping.get)(y2)
    return X2, y2


def one_hot_encoder():
    """Create OneHotEncoder compatible with new & old sklearn versions."""
    try:
        return OneHotEncoder(sparse_output=False)
    except TypeError:
        # older sklearn
        return OneHotEncoder(sparse=False)


# -----------------------------
# Data Loading & Windowing
# -----------------------------

def load_h5_rasters(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spike rasters and labels from H5.
    Returns:
        X: (N, T, 1471) boolean/integer
        y: (N,) int labels in {0,1,2,3}
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing input file: {path}")

    with h5py.File(path, "r") as f:
        X = f["samples"][...]
        y = f["labels"][...]

    # Cast spikes to bool -> float32 later for Keras
    X = np.asarray(X, dtype=bool)
    y = np.asarray(y).astype(np.int32)

    return X, y


def window_rasters_balanced(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a class-balanced windowed dataset:
      - Window size = WIN_SEC * FS
      - Overlap = OLP
      - For each trial, cut into num_samp segments based on its class per SEGMENTS_PER_CLASS
    Returns:
      windowed_x: (num_row, win_size, 1471) float32
      new_labels: (num_row, 1) uint8
    """
    win_size = int(WIN_SEC * FS)
    bias = int((1.0 - OLP) * win_size)

    N, T, F = X.shape
    if bias <= 0 or win_size <= 0 or win_size > T:
        raise ValueError(f"Bad windowing parameters: win_size={win_size}, bias={bias}, T={T}")

    # compute total rows dynamically
    total_rows = 0
    for i in range(N):
        cls = int(y[i])
        num_samp = SEGMENTS_PER_CLASS.get(cls, 0)
        total_rows += num_samp

    windowed_x = np.zeros((total_rows, win_size, F), dtype=np.float32)
    new_labels = np.zeros((total_rows, 1), dtype=np.uint8)

    start = 0
    for i in range(N):
        cls = int(y[i])
        num_samp = SEGMENTS_PER_CLASS.get(cls, 0)
        if num_samp <= 0:
            continue

        seg_data = np.zeros((num_samp, win_size, F), dtype=np.float32)
        start2 = 0
        for j in range(num_samp):
            # guard if we run past the trial length
            end2 = start2 + win_size
            if end2 > T:
                end2 = T
                start2 = max(0, T - win_size)
            seg = X[i, start2:end2, :]
            if seg.shape[0] < win_size:
                # pad last segment by repeating last rows (or zeros)
                pad = np.repeat(seg[-1:, :], win_size - seg.shape[0], axis=0)
                seg = np.vstack([seg, pad])

            seg_data[j] = seg.astype(np.float32)
            start2 += bias

        windowed_x[start:start + num_samp] = seg_data
        new_labels[start:start + num_samp, 0] = cls
        start += num_samp

        gc.collect()

    return windowed_x, new_labels


# -----------------------------
# Model
# -----------------------------

def build_lstm(input_dim: int, timesteps: int, num_class: int) -> Sequential:
    """
    Simple LSTM classifier:
      LSTM(64) -> Dropout(0.2) -> Dense(32, relu) -> Dense(num_class, softmax)
    """
    model = Sequential(name="LSTM_Classifier")
    model.add(LSTM(64,
                   input_shape=(timesteps, input_dim),
                   kernel_initializer=glorot_uniform()))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu", kernel_initializer=glorot_uniform()))
    model.add(Dense(num_class, activation="softmax", kernel_initializer=glorot_uniform()))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# -----------------------------
# Main training/eval
# -----------------------------

def main() -> None:
    ensure_dirs()

    print("Loading data...")
    X, y = load_h5_rasters(H5_PATH)

    print("Preparing data (mode:", MODE, ") ...")
    if len(MODE) == 1 and MODE[0] == 0:
        # Regression path kept for parity; not used typically for 0..3 labels
        Xw, yw = window_rasters_balanced(X, y)
        print("[WARN] Regression selected with 0..3 labels; you likely want classification.")
    elif len(MODE) == 1 and MODE[0] == 4:
        y4 = labels_to_4(y)
        Xw, yw = window_rasters_balanced(X, y4)
    else:
        # 2-class classification
        y4 = labels_to_4(y)
        Xf, yf = filter_to_two_classes(X, y4, MODE)   # keeps only MODE classes
        Xw, yw = window_rasters_balanced(Xf, yf)

    # Shapes
    n_samples, timesteps, num_feats = Xw.shape
    print(f"Windowed shape: X={Xw.shape}, y={yw.shape}")

    # CV setup
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    encoder = one_hot_encoder()

    # Storage
    train_result, test_result = [], []
    train_mse, test_mse = [], []
    train_rmse, test_rmse = [], []
    train_mae, test_mae = [], []
    train_r2, test_r2 = [], []
    run_time: List[float] = []

    if len(MODE) == 1 and MODE[0] == 4:
        train_conf_mat = np.zeros((4, 4), dtype=float)
        test_conf_mat = np.zeros((4, 4), dtype=float)
    elif len(MODE) == 2:
        train_conf_mat = np.zeros((2, 2), dtype=float)
        test_conf_mat = np.zeros((2, 2), dtype=float)
    else:
        train_conf_mat = test_conf_mat = None

    # LSTM expects 3D (no extra channel dim)
    data = Xw.astype(np.float32)
    labels_vec = yw.reshape(-1)

    print("Starting Stratified 10-fold CV...")
    fold = 0
    for train_index, test_index in skf.split(data, labels_vec):
        fold += 1
        print(f"\nFold {fold}/{N_SPLITS}")

        train_x, test_x = data[train_index], data[test_index]
        train_y, test_y = labels_vec[train_index], labels_vec[test_index]

        if len(MODE) == 1 and MODE[0] == 0:  # regression
            num_class = 1
            y_train_in = train_y.astype(np.float32).reshape(-1, 1)
            y_test_in = test_y.astype(np.float32).reshape(-1, 1)
            # For true regression, you'd change model + compile accordingly.
            # Kept for parity with original code.
        else:
            y_train_in = encoder.fit_transform(train_y.reshape(-1, 1))
            y_test_in = encoder.transform(test_y.reshape(-1, 1))
            num_class = y_train_in.shape[1]
            print("num_class:", num_class)

        clear_session()
        model = build_lstm(num_feats, timesteps, num_class)
        model.summary()

        start_time = time()
        if len(MODE) == 1 and MODE[0] == 0:
            # regression path (not typical here)
            history = model.fit(train_x, y_train_in, batch_size=16, epochs=50,
                                validation_data=(test_x, y_test_in), verbose=1)
        else:
            history = model.fit(train_x, y_train_in, batch_size=32, epochs=80,
                                validation_data=(test_x, y_test_in), verbose=0)
        run_time.append(time() - start_time)

        # Predictions
        train_pred_prob = model.predict(train_x, verbose=0)
        test_pred_prob = model.predict(test_x, verbose=0)

        if len(MODE) == 1 and MODE[0] == 0:
            # Regression metrics (unlikely use here)
            train_mse.append(mean_squared_error(y_train_in, train_pred_prob))
            test_mse.append(mean_squared_error(y_test_in, test_pred_prob))
            train_rmse.append(np.sqrt(train_mse[-1]))
            test_rmse.append(np.sqrt(test_mse[-1]))
            train_mae.append(mean_absolute_error(y_train_in, train_pred_prob))
            test_mae.append(mean_absolute_error(y_test_in, test_pred_prob))
            train_r2.append(r2_score(y_train_in, train_pred_prob))
            test_r2.append(r2_score(y_test_in, test_pred_prob))
        else:
            train_pred = np.argmax(train_pred_prob, axis=1)
            test_pred = np.argmax(test_pred_prob, axis=1)

            train_acc = np.mean(train_pred == train_y)
            test_acc = np.mean(test_pred == test_y)
            train_result.append(train_acc)
            test_result.append(test_acc)
            print(f"train acc: {train_acc:.4f} -- test acc: {test_acc:.4f}")

            # Confusion matrices (accumulate)
            if train_conf_mat is not None:
                train_conf_mat += confusion_matrix(train_y, train_pred, labels=range(num_class))
                test_conf_mat += confusion_matrix(test_y, test_pred, labels=range(num_class))

            # Extra metrics
            f1w = f1_score(test_y, test_pred, average="weighted")
            recw = recall_score(test_y, test_pred, average="weighted")
            precw = precision_score(test_y, test_pred, average="weighted")
            print(f"F1(w): {f1w:.4f} | Recall(w): {recw:.4f} | Precision(w): {precw:.4f}")
            # If you want the text report:
            # print(classification_report(test_y, test_pred))

        # cleanup
        del train_x, test_x, y_train_in, y_test_in, model, history
        gc.collect()

    # -----------------------------
    # Summaries & Saving
    # -----------------------------
    ensure_dirs()

    run_time = np.array(run_time).reshape(-1, 1)

    if len(MODE) == 1 and MODE[0] == 0:
        # Regression summary
        train_mse = np.array(train_mse)[:, None]
        test_mse = np.array(test_mse)[:, None]
        train_rmse = np.array(train_rmse)[:, None]
        test_rmse = np.array(test_rmse)[:, None]
        train_mae = np.array(train_mae)[:, None]
        test_mae = np.array(test_mae)[:, None]
        train_r2 = np.array(train_r2)[:, None]
        test_r2 = np.array(test_r2)[:, None]

        print("proposed model train--> mean:", np.mean(train_mse), "std:", np.std(train_mse))
        print("proposed model test-->  mean:", np.mean(test_mse), "std:", np.std(test_mse))

        # Save CSVs
        pd.DataFrame(np.vstack([train_mse, train_mse.mean(), train_mse.std()]), columns=["MSE"]).to_csv(
            f"{CSV_DIR}/train_MSE.csv", index=False)
        pd.DataFrame(np.vstack([test_mse, test_mse.mean(), test_mse.std()]), columns=["MSE"]).to_csv(
            f"{CSV_DIR}/test_MSE.csv", index=False)
        # (Similarly save RMSE/MAE/R2...)
        pd.DataFrame(run_time, columns=["Second"]).to_csv(f"{CSV_DIR}/train_RT.csv", index=False)

        # Plots
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, test_mse.shape[0] + 1), test_mse, ".-")
        plt.xlabel("Fold number"); plt.ylabel("MSE")
        plt.savefig(f"{RES_DIR}/test_MSE.jpg"); plt.close()

    else:
        train_result = np.array(train_result)[:, None]
        test_result = np.array(test_result)[:, None]

        print("proposed model train--> mean:", np.mean(train_result), "std:", np.std(train_result))
        print("proposed model test-->  mean:", np.mean(test_result), "std:", np.std(test_result))

        # Save run time
        pd.DataFrame(np.vstack([run_time, run_time.mean(), run_time.std()]),
                     columns=["Second"]).to_csv(f"{CSV_DIR}/train_CT.csv", index=False)

        # Accuracy plots
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, train_result.shape[0] + 1), train_result, "b.-", linewidth=2)
        plt.xlabel("Fold number"); plt.ylabel("Accuracy"); plt.tight_layout()
        if len(MODE) == 2:
            plt.savefig(f"{RES_DIR}/train_2class({MODE[0]},{MODE[1]})_accuracy.jpg")
        else:
            plt.savefig(f"{RES_DIR}/train_4class.jpg")
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.plot(range(1, test_result.shape[0] + 1), test_result, "b.-", linewidth=2)
        plt.xlabel("Fold number"); plt.ylabel("Accuracy"); plt.tight_layout()
        if len(MODE) == 2:
            plt.savefig(f"{RES_DIR}/test_2class({MODE[0]},{MODE[1]})_accuracy.jpg")
        else:
            plt.savefig(f"{RES_DIR}/test_4class.jpg")
        plt.close()

        # Confusion matrices (averaged over folds)
        if train_conf_mat is not None:
            train_cm_avg = train_conf_mat / N_SPLITS
            test_cm_avg = test_conf_mat / N_SPLITS

            if len(MODE) == 2:
                idx_labels = [str(MODE[0]), str(MODE[1])]
            else:
                idx_labels = CLASS_NAMES_4

            # Save CSVs
            pd.DataFrame(train_cm_avg, index=idx_labels, columns=idx_labels).to_csv(
                f"{CSV_DIR}/train_{'2class(' + str(MODE[0]) + ',' + str(MODE[1]) + ')' if len(MODE)==2 else '4class'}_CM.csv")
            pd.DataFrame(test_cm_avg, index=idx_labels, columns=idx_labels).to_csv(
                f"{CSV_DIR}/test_{'2class(' + str(MODE[0]) + ',' + str(MODE[1]) + ')' if len(MODE)==2 else '4class'}_CM.csv")

            # Heatmaps
            plt.figure(figsize=(7 if len(MODE) == 2 else 10, 4 if len(MODE) == 2 else 7))
            sns.heatmap(pd.DataFrame(train_cm_avg, index=idx_labels, columns=idx_labels),
                        annot=True, cmap="Blues", annot_kws={"size": 12})
            if len(MODE) == 2:
                plt.savefig(f"{RES_DIR}/train_2class({MODE[0]},{MODE[1]})_CM.jpg")
            else:
                plt.savefig(f"{RES_DIR}/train_4class_CM.jpg")
            plt.close()

            plt.figure(figsize=(7 if len(MODE) == 2 else 10, 4 if len(MODE) == 2 else 7))
            sns.heatmap(pd.DataFrame(test_cm_avg, index=idx_labels, columns=idx_labels),
                        annot=True, cmap="Blues", annot_kws={"size": 12})
            if len(MODE) == 2:
                plt.savefig(f"{RES_DIR}/test_2class({MODE[0]},{MODE[1]})_CM.jpg")
            else:
                plt.savefig(f"{RES_DIR}/test_4class_CM.jpg")
            plt.close()

        # Save accuracies
        pd.DataFrame(np.vstack([train_result, train_result.mean(), train_result.std()]),
                     columns=["accuracy"]).to_csv(
            f"{CSV_DIR}/train_{'2class(' + str(MODE[0]) + ',' + str(MODE[1]) + ')' if len(MODE)==2 else '4class'}.csv",
            index=False)
        pd.DataFrame(np.vstack([test_result, test_result.mean(), test_result.std()]),
                     columns=["accuracy"]).to_csv(
            f"{CSV_DIR}/test_{'2class(' + str(MODE[0]) + ',' + str(MODE[1]) + ')' if len(MODE)==2 else '4class'}.csv",
            index=False)

    print("\n[Done] Training & evaluation complete.")


if __name__ == "__main__":
    # Set seeds for some reproducibility (optional)
    np.random.seed(RANDOM_STATE)
    main()
