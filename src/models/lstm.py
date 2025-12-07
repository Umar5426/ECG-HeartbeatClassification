# src/model/lstm.py
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    """
    Simple LSTM classifier.
    Input: (batch, seq_len, 1)
    Output: logits over classes
    """
    def __init__(self, seq_len: int, n_classes: int, hidden_size: int = 128, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=n_layers,
                            batch_first=True, bidirectional=True, dropout=dropout if n_layers>1 else 0.0)
        self.pool = nn.AdaptiveAvgPool1d(1)  # will pool across time after transpose
        self.fc = nn.Linear(hidden_size * 2, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, (h_n, c_n) = self.lstm(x)  # out: (batch, seq_len, hidden*2)
        # global average pool across time
        out = out.permute(0, 2, 1)      # (batch, hidden*2, seq_len)
        out = self.pool(out)            # (batch, hidden*2, 1)
        out = out.view(out.size(0), -1) # (batch, hidden*2)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits
