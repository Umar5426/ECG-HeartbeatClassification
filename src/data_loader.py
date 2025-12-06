# src/data_loader.py

import wfdb
import numpy as np
from typing import Tuple, List
import os

DATA_PATH = "/Users/umarkhan/Desktop/ECG-HeartbeatClassification/data/physionet.org/files/mitdb/1.0.0/"

def load_record(record_name: str, db_path: str = DATA_PATH):

    record_path = os.path.join(db_path, record_name)

    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    signal = record.p_signal
    fs = record.fs

    return signal, annotation, fs


def load_multiple_records(record_list: List[str], db_path: str = DATA_PATH):
    data = []
    for rec in record_list:
        print(f"Loading record {rec}...")
        sig, ann, fs = load_record(rec, db_path)
        data.append((sig, ann, fs))
    return data


if __name__ == "__main__":
    signal, ann, fs = load_record("100")
    print("Signal shape:", signal.shape)
    print("Sampling rate:", fs)
    print("First 10 annotation symbols:", ann.symbol[:10])
