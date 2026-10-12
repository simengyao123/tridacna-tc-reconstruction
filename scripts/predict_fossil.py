"""Apply the LHJ2 checkpoint and its unchanged scaler to fossil features."""
from project_config import *
import random
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset, DataLoader
SAMPLE_NAME = 'A114'
DATA_PATH = DATA_FILE
TRAINED_MODEL_PATH = MODEL_DIR / 'lhj2_best.pth'
TRAINED_SCALER_PATH = MODEL_DIR / 'scaler_lhj2.pkl'
OUTPUT_DIR = STAGE_OUTPUT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sample_slug = SAMPLE_NAME.lower()
RESULT_CSV_PATH = OUTPUT_DIR / f'{sample_slug}_daily_probability.csv'
RESULT_EXCEL_PATH = OUTPUT_DIR / f'{sample_slug}_daily_probability.xlsx'
REPORT_PATH = OUTPUT_DIR / f'{sample_slug}_classification_report.csv'
CM_FIG_PATH = OUTPUT_DIR / f'{sample_slug}_confusion_matrix.png'
PROB_FIG_PATH = OUTPUT_DIR / f'{sample_slug}_daily_tc_probability.png'
DATE_COL_INDEX = 0
FEATURE_INDICES = FEATURE_INDICES_CONFIG
LABEL_COL_INDEX = -1
LEFT_WINDOW = CFG['left_window']
RIGHT_WINDOW = CFG['right_window']
TOTAL_WINDOW = LEFT_WINDOW + RIGHT_WINDOW + 1
HIDDEN_SIZE = CFG['hidden_size']
BATCH_SIZE = CFG['batch_size']
CLASSIFICATION_THRESHOLD = CFG['classification_threshold']
RANDOM_SEED = CFG['seed']

def set_seed(seed: int=42) -> None:
    """Seed Python, NumPy and PyTorch where used; deterministic kernels do not guarantee cross-platform bitwise equality."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('=' * 60)
print(f'预测样品：{SAMPLE_NAME}')
print(f'数据文件：{DATA_PATH}')
print(f'加载模型：{TRAINED_MODEL_PATH}')
print(f'时间窗口：{TOTAL_WINDOW} d (前{LEFT_WINDOW}+目标日+后{RIGHT_WINDOW})')
print(f'计算设备：{device}')
print('=' * 60)
df = pd.read_excel(DATA_PATH)
if len(df.columns) < 8:
    raise ValueError('Excel列数不足，需包含日期、6个特征和标签列。')
date_col = 'date'
feature_cols = FEATURES
label_col = 'label'
print(f'日期列：{date_col}')
print(f'使用特征列：{feature_cols}')
print(f'标签列：{label_col}')
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
if df[date_col].isna().any():
    bad_rows = df.index[df[date_col].isna()].tolist()
    raise ValueError(f'日期解析失败行：{bad_rows[:10]}')
df = df.sort_values(date_col).reset_index(drop=True)
if df[feature_cols].isna().any().any():
    missing = df[feature_cols].isna().sum()
    raise ValueError(f'特征存在缺失值：\n{missing[missing > 0]}')
df[label_col] = df[label_col].astype(int)
unique_labels = set(df[label_col].unique())
if not unique_labels.issubset({0, 1}):
    raise ValueError(f'标签只能为0/1，当前值：{sorted(unique_labels)}')
print(f'总记录数：{len(df)}')
print('标签分布：', df[label_col].value_counts().sort_index().to_dict())
scaler = joblib.load(TRAINED_SCALER_PATH)
X_raw = scaler.transform(df[feature_cols])
y_raw = df[label_col].to_numpy(dtype=int)
all_dates = df[date_col].reset_index(drop=True)

def get_date_segments(dates: pd.Series) -> np.ndarray:
    """Assign a new segment whenever consecutive dates differ by anything other than one day."""
    dates = pd.to_datetime(dates).reset_index(drop=True)
    day_diff = dates.diff().dt.days
    new_segment = day_diff.isna() | (day_diff != 1)
    return new_segment.cumsum().to_numpy(dtype=int)

def build_sliding_window(X: np.ndarray, y: np.ndarray, dates: pd.Series, left_window: int=5, right_window: int=3) -> tuple[np.ndarray, np.ndarray]:
    """Build endpoint-padded sequences separately within each uninterrupted daily segment. Never bridge date gaps."""
    expected_window = left_window + right_window + 1
    segment_ids = get_date_segments(dates)
    X_seq, y_seq = ([], [])
    for segment_id in np.unique(segment_ids):
        seg_idx = np.where(segment_ids == segment_id)[0]
        X_seg = X[seg_idx]
        y_seg = y[seg_idx]
        seg_len = len(X_seg)
        for local_i in range(seg_len):
            start = local_i - left_window
            end = local_i + right_window + 1
            window_parts = []
            if start < 0:
                window_parts.append(np.repeat(X_seg[0:1], -start, axis=0))
                actual_start = 0
            else:
                actual_start = start
            actual_end = min(end, seg_len)
            window_parts.append(X_seg[actual_start:actual_end])
            if end > seg_len:
                window_parts.append(np.repeat(X_seg[-1:], end - seg_len, axis=0))
            window = np.vstack(window_parts)
            if len(window) != expected_window:
                raise RuntimeError(f'窗口长度错误：期望{expected_window}，实际{len(window)}')
            X_seq.append(window)
            y_seq.append(y_seg[local_i])
    return (np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.int64))
X_all, y_all = build_sliding_window(X=X_raw, y=y_raw, dates=all_dates, left_window=LEFT_WINDOW, right_window=RIGHT_WINDOW)
print(f'\n窗口构造完成，形状：{X_all.shape}')
print(f'窗口长度校验：{X_all.shape[1] == TOTAL_WINDOW}')

class TimeSeriesDataset(Dataset):
    """Timeseriesdataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """  init  ."""
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        """  len  ."""
        return len(self.X)

    def __getitem__(self, idx: int):
        """  getitem  ."""
        return (self.X[idx], self.y[idx])
dataset = TimeSeriesDataset(X_all, y_all)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

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
input_dim = len(feature_cols)
model = BiLSTMAttention(input_dim=input_dim, hidden_size=HIDDEN_SIZE).to(device)
model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
model.eval()
print('\n模型权重加载完成')
all_predictions = []
all_targets = []
all_tc_probabilities = []
with torch.no_grad():
    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        logits = model(batch_X)
        probabilities = torch.softmax(logits, dim=1)
        tc_prob = probabilities[:, 1]
        preds = (tc_prob >= CLASSIFICATION_THRESHOLD).long()
        all_tc_probabilities.extend(tc_prob.cpu().numpy().tolist())
        all_predictions.extend(preds.cpu().numpy().tolist())
        all_targets.extend(batch_y.numpy().tolist())
all_targets = np.asarray(all_targets, dtype=int)
all_predictions = np.asarray(all_predictions, dtype=int)
all_tc_probabilities = np.asarray(all_tc_probabilities, dtype=float)
if not len(all_dates) == len(all_targets) == len(all_predictions) == len(all_tc_probabilities):
    raise ValueError('日期与预测结果长度不匹配')
result_df = pd.DataFrame({'date': all_dates, 'reference_label': all_targets, 'tc_probability': all_tc_probabilities, 'predicted_label': all_predictions})
result_df['reference_class'] = np.where(result_df['reference_label'] == 1, 'TC', 'Non-TC')
result_df['predicted_class'] = np.where(result_df['predicted_label'] == 1, 'TC', 'Non-TC')
result_df.to_csv(RESULT_CSV_PATH, index=False, encoding='utf-8-sig')
result_df.to_excel(RESULT_EXCEL_PATH, index=False)
print(f'\n逐日预测结果已保存：{RESULT_CSV_PATH}')
