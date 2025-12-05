# src/data_loader.py

import wfdb
import numpy as np
from typing import Tuple, List

def load_record(record_name: str, db: str = "mitdb"):
    """
    Load an ECG record and its annotations from PhysioNet using WFDB.
    
    Parameters:
        record_name (str): Example = '100', '101', ...
        db (str): PhysioNet database name (default = 'mitdb')
    
    Returns:
        signal (np.ndarray): Shape (N, channels)
        annotation (wfdb.Annotation)
    """
    # Load ECG signal
    record = wfdb.rdrecord(record_name, pn_dir=db)
    
    # Load annotations (R-peaks + beat types)
    annotation = wfdb.rdann(record_name, 'atr', pn_dir=db)

    signal = record.p_signal  # float values
    fs = record.fs            # sampling frequency = 360 Hz

    return signal, annotation, fs


def load_multiple_records(record_list: List[str], db: str = "mitdb"):
    """
    Load multiple MIT-BIH records.
    Returns list of tuples: (signal, annotation, fs)
    """
    data = []
    for rec in record_list:
        print(f"Loading record {rec}...")
        sig, ann, fs = load_record(rec, db)
        data.append((sig, ann, fs))
    return data


if __name__ == "__main__":
    # Test loading a single record
    signal, ann, fs = load_record("100")
    print("Signal shape:", signal.shape)
    print("Sampling rate:", fs)
    print("First 10 annotation symbols:", ann.symbol[:10])
