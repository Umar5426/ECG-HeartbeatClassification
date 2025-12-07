# src/main.py
"""
Main runner for preprocessing, training and evaluation.
Usage examples:
  python src/main.py preprocess
  python src/main.py train --model cnn
  python src/main.py evaluate --checkpoint checkpoints/best_cnn.pth --model cnn
"""
import argparse
import os
import numpy as np

from preprocess import load_dataset, normalize
from train import get_data, train
from evaluate import evaluate_checkpoint

def do_preprocess():
    print("Running preprocess to sanity-check extraction.")
    train_records = [
        "100","101","102","103","104","105","106","107","108","109",
        "111","112","113","114","115","116","117","118","119","121","122","123","124"
    ]
    test_records = ["200","201","202","203","205","207","208","209","210","212","213","214","215","219","220","221","222","223","228","230","231","232","233","234"]

    Xtr, ytr = load_dataset(train_records)
    Xte, yte = load_dataset(test_records)
    X = np.concatenate([Xtr, Xte])
    y = np.concatenate([ytr, yte])
    X = normalize(X)
    print("Total beats:", X.shape, "unique labels:", np.unique(y))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["preprocess", "train", "evaluate"])
    ap.add_argument("--model", choices=["cnn", "lstm"], default="cnn")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    if args.action == "preprocess":
        do_preprocess()
    elif args.action == "train":
        best_path, label_map, X_test, y_test = train(args.model)
        print("Saved checkpoint:", best_path)
    elif args.action == "evaluate":
        if args.checkpoint is None:
            raise SystemExit("Pass --checkpoint path to evaluate")
        # We'll load the checkpoint and evaluate. We need X_test/y_test from train.get_data() or saved npy.
        # For now we call get_data again to build X_test (this is heavy but consistent)
        train_records = [
            "100","101","102","103","104","105","106","107","108","109",
            "111","112","113","114","115","116","117","118","119","121","122","123","124"
        ]
        test_records = ["200","201","202","203","205","207","208","209","210","212","213","214","215","219","220","221","222","223","228","230","231","232","233","234"]
        # Recreate X/y exactly as in train.get_data (min_count default is 5)
        from train import get_data as _get_data
        X, y, label_map = _get_data(train_records, test_records)
        # split to get test set (same splitting logic as train)
        from sklearn.model_selection import train_test_split
        _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        evaluate_checkpoint(args.checkpoint, X_test, y_test, model_name=args.model)
