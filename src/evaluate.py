# src/evaluate.py
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader

from utils import ECGDataset, collate_cnn, collate_lstm, load_checkpoint
from model.cnn import Simple1DCNN
from model.lstm import SimpleLSTM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_checkpoint(checkpoint_path: str, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "cnn"):
    cp = load_checkpoint(checkpoint_path, DEVICE)
    label_map = cp["label_map"]
    # invert label_map to string names
    int_to_label = {v: k for k, v in label_map.items()}
    n_classes = len(label_map)
    seq_len = X_test.shape[1]

    if model_name == "cnn":
        model = Simple1DCNN(seq_len, n_classes).to(DEVICE)
        collate_fn = collate_cnn
    else:
        model = SimpleLSTM(seq_len, n_classes).to(DEVICE)
        collate_fn = collate_lstm

    model.load_state_dict(cp["model_state_dict"])
    model.eval()

    ds = ECGDataset(X_test, y_test)
    loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            p = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(p)
            trues.append(yb.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    acc = accuracy_score(trues, preds)
    print("Test accuracy:", acc)
    print("Classification report:")
    print(classification_report(trues, preds, target_names=[int_to_label[i] for i in range(n_classes)]))
    print("Confusion matrix:")
    print(confusion_matrix(trues, preds))
    return acc, preds, trues
