# src/preprocess.py

import numpy as np
import wfdb
from typing import List, Tuple
from sklearn.model_selection import train_test_split

DATA_PATH = "/Users/umarkhan/Desktop/ECG-HeartbeatClassification/data/physionet.org/files/mitdb/1.0.0/"


# ----------------------------
# 1. Extract beats from a record
# ----------------------------
def extract_beats(record_name: str, window: int = 360, db_path: str = DATA_PATH):
    record = wfdb.rdrecord(f"{db_path}/{record_name}")
    annotation = wfdb.rdann(f"{db_path}/{record_name}", 'atr')

    signal = record.p_signal[:, 0]  # use lead 0
    beats = []
    labels = []

    for i, ann_sample in enumerate(annotation.sample):
        if ann_sample - window < 0 or ann_sample + window >= len(signal):
            continue  # skip beats too close to start/end
        beat = signal[ann_sample - window : ann_sample + window]
        beats.append(beat)
        labels.append(annotation.symbol[i])

    return np.array(beats), np.array(labels)



# ----------------------------
# 2. Load multiple records
# ----------------------------
def load_dataset(records: List[str], window: int = 360):
    all_beats = []
    all_labels = []

    for rec in records:
        print(f"Extracting {rec}...")
        beats, labels = extract_beats(rec, window)
        all_beats.append(beats)
        all_labels.append(labels)

    X = np.concatenate(all_beats, axis=0)
    y = np.concatenate(all_labels, axis=0)

    return X, y


# ----------------------------
# 3. Normalize each beat
# ----------------------------
def normalize(X):
    # z-score per beat
    X_norm = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)
    return X_norm


# ----------------------------
# 4. Train/Val/Test split
# ----------------------------
def split_data(X, y, test_size=0.2, val_size=0.1):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ----------------------------
# 5. Format for CNN & LSTM
# ----------------------------
def prepare_for_cnn(X):
    # CNN expects: (samples, length, channels)
    return X.reshape((X.shape[0], X.shape[1], 1))


def prepare_for_lstm(X):
    # LSTM expects same shape as CNN: (batch, timesteps, features)
    return X.reshape((X.shape[0], X.shape[1], 1))


# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    # MIT-BIH recommended training split
    train_records = [
        "100","101","102","103","104","105","106","107","108","109",
        "111","112","113","114","115","116","117","118","119","121","122","123","124"
    ]
    test_records = ["200","201","202","203","205","207","208","209","210","212","213","214","215","219","220","221","222","223","228","230","231","232","233","234"]

    # Load data
    X_train_raw, y_train_raw = load_dataset(train_records)
    X_test_raw,  y_test_raw  = load_dataset(test_records)

    # Combine for global split
    X = np.concatenate([X_train_raw, X_test_raw])
    y = np.concatenate([y_train_raw, y_test_raw])

    # Normalize
    X = normalize(X)

    # Train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # CNN/LSTM formatting
    X_train_cnn = prepare_for_cnn(X_train)
    X_val_cnn = prepare_for_cnn(X_val)
    X_test_cnn = prepare_for_cnn(X_test)

    X_train_lstm = prepare_for_lstm(X_train)
    X_val_lstm = prepare_for_lstm(X_val)
    X_test_lstm = prepare_for_lstm(X_test)

    print("Final Shapes:")
    print("CNN Train:", X_train_cnn.shape)
    print("CNN Val:", X_val_cnn.shape)
    print("CNN Test:", X_test_cnn.shape)
    print("Labels:", len(np.unique(y)))
