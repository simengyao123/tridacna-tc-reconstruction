"""Shared inference definitions, identical to the supplied training implementation."""
import numpy as np
import pandas as pd
import torch
from torch import nn
def get_date_segments(dates: pd.Series) -> np.ndarray:
    """Assign a new segment whenever consecutive dates differ by anything other than one day."""
    dates = pd.to_datetime(dates).reset_index(drop=True)
    day_diff = dates.diff().dt.days
    new_segment = day_diff.isna() | (day_diff != 1)
    segment_ids = new_segment.cumsum().to_numpy(dtype=int)
    return segment_ids

def build_sliding_window(X: np.ndarray, y: np.ndarray, dates: pd.Series, left_window: int=5, right_window: int=3) -> tuple[np.ndarray, np.ndarray]:
    """Build endpoint-padded sequences separately within each uninterrupted daily segment. Never bridge date gaps."""
    if len(X) != len(y) or len(X) != len(dates):
        raise ValueError(f'X、y和dates长度不一致：X={len(X)}, y={len(y)}, dates={len(dates)}')
    expected_window = left_window + right_window + 1
    segment_ids = get_date_segments(dates)
    X_seq = []
    y_seq = []
    for segment_id in np.unique(segment_ids):
        segment_indices = np.where(segment_ids == segment_id)[0]
        X_segment = X[segment_indices]
        y_segment = y[segment_indices]
        segment_length = len(X_segment)
        for local_i in range(segment_length):
            start = local_i - left_window
            end = local_i + right_window + 1
            window_parts = []
            if start < 0:
                left_pad_length = -start
                left_pad = np.repeat(X_segment[0:1], left_pad_length, axis=0)
                window_parts.append(left_pad)
                actual_start = 0
            else:
                actual_start = start
            actual_end = min(end, segment_length)
            window_parts.append(X_segment[actual_start:actual_end])
            if end > segment_length:
                right_pad_length = end - segment_length
                right_pad = np.repeat(X_segment[-1:], right_pad_length, axis=0)
                window_parts.append(right_pad)
            window = np.vstack(window_parts)
            if len(window) != expected_window:
                raise RuntimeError(f'窗口长度错误：期望{expected_window}，实际{len(window)}')
            X_seq.append(window)
            y_seq.append(y_segment[local_i])
    return (np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.int64))

class BiLSTMAttention(nn.Module):
    """Two bidirectional LSTM layers with linear temporal attention and a two-class classifier."""

    def __init__(self, input_dim: int, hidden_size: int=32):
        """  init  ."""
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size * 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        lstm_out, _ = self.lstm(x)
        attention_logits = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_logits, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        context = self.dropout(context)
        logits = self.classifier(context)
        return logits
