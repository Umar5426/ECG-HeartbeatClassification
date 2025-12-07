# src/train.py
import os
import time
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocess import (
    # We call the preprocessing functions to get raw arrays
    load_dataset, normalize
)
from utils import (
    filter_and_encode_labels,
    ECGDataset,
    collate_cnn,
    collate_lstm,
    save_checkpoint
)
from model.cnn import Simple1DCNN
from model.lstm import SimpleLSTM

# training hyperparams
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_data(train_records, test_records, window=360, min_count=5):
    X_tr, y_tr = load_dataset(train_records, window)
    X_te, y_te = load_dataset(test_records, window)

    X = np.concatenate([X_tr, X_te], axis=0)
    y = np.concatenate([y_tr, y_te], axis=0)

    # normalize
    X = normalize(X)

    # filter & encode
    y_encoded, label_map, mask = filter_and_encode_labels(y, min_count=min_count)
    X = X[mask]

    return X, y_encoded, label_map


def train(model_name: str = "cnn"):
    # MIT-BIH split used earlier
    train_records = [
        "100","101","102","103","104","105","106","107","108","109",
        "111","112","113","114","115","116","117","118","119","121","122","123","124"
    ]
    test_records = ["200","201","202","203","205","207","208","209","210","212","213","214","215","219","220","221","222","223","228","230","231","232","233","234"]

    X, y, label_map = get_data(train_records, test_records)
    print("Total beats after filtering:", len(y), "classes:", label_map)

    # split (simple random splits)
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # choose collate and dataset
    if model_name == "cnn":
        collate_fn = collate_cnn
        # dataset returns (L,1) so DataLoader will transform to (batch,1,L) in collate
        train_ds = ECGDataset(X_train, y_train)
        val_ds = ECGDataset(X_val, y_val)
    else:
        collate_fn = collate_lstm
        train_ds = ECGDataset(X_train, y_train)
        val_ds = ECGDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    seq_len = X.shape[1]
    n_classes = len(label_map)

    if model_name == "cnn":
        model = Simple1DCNN(seq_len, n_classes).to(DEVICE)
    else:
        model = SimpleLSTM(seq_len, n_classes).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_path = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # validation
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = model(xb)
                p = torch.argmax(logits, dim=1).cpu().numpy()
                preds.append(p)
                trues.append(yb.cpu().numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        val_acc = accuracy_score(trues, preds)

        avg_loss = float(np.mean(epoch_losses))
        print(f"Epoch {epoch}/{EPOCHS} — loss: {avg_loss:.4f} — val_acc: {val_acc:.4f} — time: {time.time()-t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(CHECKPOINT_DIR, f"best_{model_name}.pth")
            save_checkpoint({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "label_map": label_map,
                "val_acc": val_acc,
                "epoch": epoch
            }, best_path)
            print(f"Saved best checkpoint: {best_path}")

    print("Training complete. Best val acc:", best_val_acc)
    return best_path, label_map, X_test, y_test  # return test set so evaluate can use it


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "lstm"], default="cnn")
    args = ap.parse_args()
    train(args.model)
