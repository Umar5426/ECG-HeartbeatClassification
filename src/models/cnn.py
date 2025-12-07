# src/model/cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple1DCNN(nn.Module):
    """
    Simple 1D CNN for beat classification.
    Input: (batch, 1, seq_len)
    Output: logits over classes
    """
    def __init__(self, seq_len: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # adaptive pooling to fixed features
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (batch, 1, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn3(self.conv3(x)))
        # global pooling
        x = self.pool(x)                # (batch, 128, 1)
        x = x.view(x.size(0), -1)       # (batch, 128)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
