"""
architectures.py
================
The SIX predictor architectures for the Chapter-5 RSRP predictor, copied
*verbatim* from the repository's `UniversalLSTMAdapter` (best2.txt lines 745-910,
identical in lll.ipynb and code1.ipynb). Per the task spec (Part 4) these are
REUSED EXACTLY, not redesigned.

The only change vs. the repo is packaging: the six inner classes are lifted to
module level and exposed through `build_model(name, input_size, output_size)` so
the training script can instantiate any of them with a shared, explicit contract.

Canonical inference contract (the anchor every runner script hardcodes):
    input_size  = 12   (the 12 engineered features, see config.LSTM_FEATURES)
    output_size = 1     (single-step RSRP, normalized)
    H (sequence_length) = 15

NOTE the two loader bugs found during reconciliation (reconciliation.md, Part 1),
which this pipeline AVOIDS by construction but which must be fixed if the repo's
own `UniversalLSTMAdapter.load_model` / `_load_models_adaptive` is ever reused:
  1. Filename parsing `int(name.split('_cell')[1].split('.pth')[0])` crashes on a
     browser-duplicate " (1)" suffix. This pipeline saves clean names.
  2. `detect_architecture` checks 'attention' before 'tcn', so TCN checkpoints
     (which contain a `self.attention` layer) are misdetected as LSTMAttention.
     This pipeline never relies on substring detection; the architecture name is
     always explicit.
"""

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# The six architectures — VERBATIM from the repo's UniversalLSTMAdapter.
# --------------------------------------------------------------------------- #

class ImprovedLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True, dropout=0.2)
        self.lstm3 = nn.LSTM(32, 32, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1, _ = self.lstm1(x)
        x2, _ = self.lstm2(x1)
        x3, _ = self.lstm3(x2)
        x = x3[:, -1, :]
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.relu(self.batch_norm2(self.fc2(x)))
        return self.fc3(x)


class BiLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, 128, 3, batch_first=True,
                              bidirectional=True, dropout=0.2)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        x = lstm_out[:, -1, :]
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.relu(self.batch_norm2(self.fc2(x)))
        return self.fc3(x)


class LSTMAttention(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, 2, batch_first=True, dropout=0.2)
        self.attention = nn.Linear(128, 1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        x = self.relu(self.batch_norm1(self.fc1(context)))
        x = self.relu(self.batch_norm2(self.fc2(x)))
        return self.fc3(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, 3, padding=2, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 64, 3, padding=4, dilation=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.attention = nn.Linear(64, 1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        tcn_out = self.tcn(x).transpose(1, 2)
        attention_weights = torch.softmax(self.attention(tcn_out), dim=1)
        context = torch.sum(attention_weights * tcn_out, dim=1)
        x = self.relu(self.batch_norm1(self.fc1(context)))
        x = self.relu(self.batch_norm2(self.fc2(x)))
        return self.fc3(x)


class SimpleTransformer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_projection = nn.Linear(input_size, 128)
        self.pos_encoding = nn.Parameter(torch.randn(1000, 128))
        encoder_layer = nn.TransformerEncoderLayer(128, 8, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, 3)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.relu(self.batch_norm2(self.fc2(x)))
        return self.fc3(x)


class CNNLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.batch_norm_conv1 = nn.BatchNorm1d(64)
        self.batch_norm_conv2 = nn.BatchNorm1d(128)
        self.lstm = nn.LSTM(128, 128, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.batch_norm_conv1(self.conv1(x)))
        x = self.relu(self.batch_norm_conv2(self.conv2(x)))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.relu(self.batch_norm2(self.fc2(x)))
        return self.fc3(x)


# --------------------------------------------------------------------------- #
# Factory. Keys match the repo's checkpoint-file naming ({Arch}_cell{n}.pth) so
# that models trained here drop straight into the inference loader's convention.
# --------------------------------------------------------------------------- #

ARCHITECTURES = {
    "ImprovedLSTM":  ImprovedLSTM,
    "BiLSTM":        BiLSTM,
    "LSTMAttention": LSTMAttention,
    "TCN":           TCN,
    "Transformer":   SimpleTransformer,
    "CNNLSTM":       CNNLSTM,
}


def build_model(name, input_size, output_size):
    if name not in ARCHITECTURES:
        raise KeyError(f"Unknown architecture '{name}'. "
                       f"Choose from {list(ARCHITECTURES)}")
    return ARCHITECTURES[name](input_size, output_size)
