"""Train the legacy chronological 70/30 experiment; the held-out subset selects the checkpoint."""
# The subset named test is also used for early stopping and checkpoint selection.
# It is not an independent final test set; this naming change preserves the original experiment.
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
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
SAMPLE_NAME = SAMPLE
DATA_PATH = DATA_FILE
OUTPUT_DIR = STAGE_OUTPUT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sample_slug = SAMPLE_NAME.lower()
MODEL_PATH = OUTPUT_DIR / f'{sample_slug}_best.pth'
SCALER_PATH = OUTPUT_DIR / f'scaler_{sample_slug}.pkl'
RESULT_CSV_PATH = OUTPUT_DIR / f'{sample_slug}_test_daily_probability.csv'
RESULT_EXCEL_PATH = OUTPUT_DIR / f'{sample_slug}_test_daily_probability.xlsx'
REPORT_PATH = OUTPUT_DIR / f'{sample_slug}_classification_report.csv'
HISTORY_PATH = OUTPUT_DIR / f'{sample_slug}_training_history.csv'
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
MAX_EPOCHS = CFG['max_epochs']
LEARNING_RATE = CFG['learning_rate']
TEST_SIZE = CFG['heldout_fraction']
PATIENCE = CFG['patience']
MIN_DELTA = CFG['min_delta']
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
print(f'样品：{SAMPLE_NAME}')
print(f'数据文件：{DATA_PATH}')
print(f'时间窗口：前{LEFT_WINDOW} d + 目标日 + 后{RIGHT_WINDOW} d = {TOTAL_WINDOW} d')
print(f'计算设备：{device}')
print('=' * 60)
df = pd.read_excel(DATA_PATH)
if len(df.columns) < 8:
    raise ValueError('Excel列数不足。程序默认至少包含：日期、6个特征和标签列。')
date_col = 'date'
feature_cols = FEATURES
label_col = 'label'
print(f'日期列：{date_col}')
print(f'使用特征列：{feature_cols}')
print(f'标签列：{label_col}')
print(f'特征维度：{len(feature_cols)}')
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
if df[date_col].isna().any():
    bad_rows = df.index[df[date_col].isna()].tolist()
    raise ValueError(f'日期列存在无法解析的值，对应行号：{bad_rows[:10]}')
df = df.sort_values(date_col).reset_index(drop=True)
if df[feature_cols].isna().any().any():
    missing = df[feature_cols].isna().sum()
    missing = missing[missing > 0]
    raise ValueError(f'特征列存在缺失值：\n{missing}')
df[label_col] = df[label_col].astype(int)
unique_labels = set(df[label_col].unique())
if not unique_labels.issubset({0, 1}):
    raise ValueError(f'标签只能包含0和1，当前标签值为：{sorted(unique_labels)}')
print(f'总记录数：{len(df)}')
print('标签分布：', df[label_col].value_counts().sort_index().to_dict())
split_idx = int(len(df) * (1 - TEST_SIZE))
if split_idx <= 0 or split_idx >= len(df):
    raise ValueError('训练集和测试集划分失败。')
df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()
print('\n数据集划分：')
print(f'训练集：{len(df_train)}条，{df_train[date_col].min().date()} 至 {df_train[date_col].max().date()}')
print(f'测试集：{len(df_test)}条，{df_test[date_col].min().date()} 至 {df_test[date_col].max().date()}')
print('训练集标签分布：', df_train[label_col].value_counts().sort_index().to_dict())
print('测试集标签分布：', df_test[label_col].value_counts().sort_index().to_dict())
scaler = StandardScaler()
scaler.fit(df_train[feature_cols])
X_train_raw = scaler.transform(df_train[feature_cols])
X_test_raw = scaler.transform(df_test[feature_cols])
y_train_raw = df_train[label_col].to_numpy(dtype=int)
y_test_raw = df_test[label_col].to_numpy(dtype=int)
train_dates = df_train[date_col].reset_index(drop=True)
test_dates = df_test[date_col].reset_index(drop=True)

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
X_train, y_train = build_sliding_window(X=X_train_raw, y=y_train_raw, dates=train_dates, left_window=LEFT_WINDOW, right_window=RIGHT_WINDOW)
X_test, y_test = build_sliding_window(X=X_test_raw, y=y_test_raw, dates=test_dates, left_window=LEFT_WINDOW, right_window=RIGHT_WINDOW)
print('\n窗口构造结果：')
print(f'训练集窗口形状：{X_train.shape}')
print(f'测试集窗口形状：{X_test.shape}')
print('窗口长度是否正确：', X_train.shape[1] == TOTAL_WINDOW)

class TimeSeriesDataset(Dataset):
    """Timeseriesdataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """  init  ."""
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        """  len  ."""
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """  getitem  ."""
        return (self.X[index], self.y[index])
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
print('\n模型结构：')
print(model)
class_counts = np.bincount(y_train, minlength=2)
if np.any(class_counts == 0):
    raise ValueError(f'训练集中至少有一个类别没有样本：{class_counts}')
class_weights = len(y_train) / (2 * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
print('\n训练集类别数量：', class_counts)
print('类别权重：', class_weights)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
best_test_f1 = -1.0
best_epoch = 0
early_stop_counter = 0
history = {'epoch': [], 'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': [], 'test_tc_f1': []}
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item() * batch_X.size(0)
        predictions = logits.argmax(dim=1)
        train_correct += (predictions == batch_y).sum().item()
        train_total += batch_X.size(0)
    train_loss = train_loss_sum / train_total
    train_accuracy = train_correct / train_total
    model.eval()
    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            predictions = logits.argmax(dim=1)
            test_loss_sum += loss.item() * batch_X.size(0)
            test_correct += (predictions == batch_y).sum().item()
            test_total += batch_X.size(0)
            test_predictions.extend(predictions.cpu().numpy().tolist())
            test_targets.extend(batch_y.cpu().numpy().tolist())
    test_loss = test_loss_sum / test_total
    test_accuracy = test_correct / test_total
    test_tc_f1 = f1_score(test_targets, test_predictions, pos_label=1, zero_division=0)
    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['train_accuracy'].append(train_accuracy)
    history['test_loss'].append(test_loss)
    history['test_accuracy'].append(test_accuracy)
    history['test_tc_f1'].append(test_tc_f1)
    print(f'Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f} | TC F1: {test_tc_f1:.4f}')
    if test_tc_f1 > best_test_f1 + MIN_DELTA:
        best_test_f1 = test_tc_f1
        best_epoch = epoch
        early_stop_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print(f'  保存最佳模型：TC F1 = {best_test_f1:.4f}')
    else:
        early_stop_counter += 1
        print(f'  TC F1未提升，counter = {early_stop_counter}/{PATIENCE}')
        if early_stop_counter >= PATIENCE:
            print(f'\nEarly stopping at epoch {epoch}')
            break
print('\n训练完成')
print(f'最佳Epoch：{best_epoch}')
print(f'最佳测试集TC F1：{best_test_f1:.4f}')
history_df = pd.DataFrame(history)
history_df.to_csv(HISTORY_PATH, index=False, encoding='utf-8-sig')
joblib.dump(scaler, SCALER_PATH)
print(f'模型已保存：{MODEL_PATH}')
print(f'StandardScaler已保存：{SCALER_PATH}')
print(f'训练历史已保存：{HISTORY_PATH}')
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
all_predictions = []
all_targets = []
all_tc_probabilities = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        logits = model(batch_X)
        probabilities = torch.softmax(logits, dim=1)
        tc_probabilities = probabilities[:, 1]
        predictions = (tc_probabilities >= CLASSIFICATION_THRESHOLD).long()
        all_tc_probabilities.extend(tc_probabilities.cpu().numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_targets.extend(batch_y.numpy().tolist())
all_targets = np.asarray(all_targets, dtype=int)
all_predictions = np.asarray(all_predictions, dtype=int)
all_tc_probabilities = np.asarray(all_tc_probabilities, dtype=float)
print('\n--- Final Classification Report ---')
report_text = classification_report(all_targets, all_predictions, labels=[0, 1], target_names=['Non-TC', 'TC'], digits=4, zero_division=0)
print(report_text)
report_dict = classification_report(all_targets, all_predictions, labels=[0, 1], target_names=['Non-TC', 'TC'], output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(REPORT_PATH, encoding='utf-8-sig')
cm = confusion_matrix(all_targets, all_predictions, labels=[0, 1])
print('\n' + '=' * 50)
print(f"{'':<15}{'预测=Non-TC':<16}{'预测=TC':<12}")
print(f"{'真实=Non-TC':<15}{cm[0, 0]:<16}{cm[0, 1]:<12}")
print(f"{'真实=TC':<15}{cm[1, 0]:<16}{cm[1, 1]:<12}")
print('=' * 50)
if not len(test_dates) == len(all_targets) == len(all_predictions) == len(all_tc_probabilities):
    raise ValueError(f'日期、真实标签、预测类别和预测概率长度不一致：dates={len(test_dates)}, targets={len(all_targets)}, predictions={len(all_predictions)}, probabilities={len(all_tc_probabilities)}')
result_df = pd.DataFrame({'date': test_dates, 'true_label': all_targets, 'tc_probability': all_tc_probabilities, 'predicted_label': all_predictions})
result_df['true_class'] = np.where(result_df['true_label'] == 1, 'TC', 'Non-TC')
result_df['predicted_class'] = np.where(result_df['predicted_label'] == 1, 'TC', 'Non-TC')
result_df.to_csv(RESULT_CSV_PATH, index=False, encoding='utf-8-sig')
result_df.to_excel(RESULT_EXCEL_PATH, index=False)
print('\n逐日测试结果已保存：')
print(RESULT_CSV_PATH)
print(RESULT_EXCEL_PATH)
print(f'分类报告已保存：{REPORT_PATH}')
sns.set_theme(style='white')
fig, ax = plt.subplots(figsize=(5.2, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True, linewidths=0.8, linecolor='white', xticklabels=['Non-TC', 'TC'], yticklabels=['Non-TC', 'TC'], annot_kws={'fontsize': 13}, ax=ax)
ax.set_xlabel('Predicted class', fontsize=12)
ax.set_ylabel('Actual class', fontsize=12)
plt.tight_layout()
fig.savefig(CM_FIG_PATH, dpi=600, bbox_inches='tight')
pass
plt.close(fig)
print(f'混淆矩阵已保存：{CM_FIG_PATH}')
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def plot_daily_tc_probability(prediction_df: pd.DataFrame, sample_name: str, threshold: float, output_path: Path) -> None:
    """Plot daily tc probability."""
    plot_df = prediction_df.copy()
    plot_df['date'] = pd.to_datetime(plot_df['date'])
    plot_df = plot_df.sort_values('date').reset_index(drop=True)
    segment_ids = get_date_segments(plot_df['date'])
    plot_df['segment_id'] = segment_ids
    probability_color = '#1F5AA6'
    tc_shading_color = '#F28E2B'
    threshold_color = '#404040'
    grid_color = '#D9D9D9'
    fig, ax = plt.subplots(figsize=(13, 4.8), facecolor='white')
    ax.set_facecolor('white')
    for _, segment_df in plot_df.groupby('segment_id', sort=True):
        ax.plot(segment_df['date'], segment_df['tc_probability'], color=probability_color, linewidth=1.7, zorder=3)
    for _, segment_df in plot_df.groupby('segment_id', sort=True):
        tc_mask = segment_df['true_label'].to_numpy() == 1
        ax.fill_between(segment_df['date'], 0, 1, where=tc_mask, step='mid', color=tc_shading_color, alpha=0.3, linewidth=0, zorder=1)
    ax.axhline(y=threshold, color=threshold_color, linestyle='--', linewidth=1.4, zorder=2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Predicted TC probability', fontsize=12)
    ax.set_title(sample_name, fontsize=14)
    ax.grid(axis='y', color=grid_color, linestyle=':', linewidth=0.8, alpha=0.8, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend_handles = [Line2D([0], [0], color=probability_color, linewidth=1.7, label='Predicted TC probability'), Line2D([0], [0], color=threshold_color, linestyle='--', linewidth=1.4, label=f'Classification threshold = {threshold:.1f}'), Patch(facecolor=tc_shading_color, alpha=0.3, label='Observed TC-affected day')]
    ax.legend(handles=legend_handles, frameon=False, loc='upper right', fontsize=10)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    pass
    plt.close(fig)
plot_daily_tc_probability(prediction_df=result_df, sample_name=SAMPLE_NAME, threshold=CLASSIFICATION_THRESHOLD, output_path=PROB_FIG_PATH)
print(f'日尺度概率图已保存：{PROB_FIG_PATH}')
print('\n' + '=' * 60)
print('全部结果已生成：')
print(f'1. 最佳模型：{MODEL_PATH}')
print(f'2. 标准化器：{SCALER_PATH}')
print(f'3. 训练历史：{HISTORY_PATH}')
print(f'4. 分类报告：{REPORT_PATH}')
print(f'5. 逐日结果CSV：{RESULT_CSV_PATH}')
print(f'6. 逐日结果Excel：{RESULT_EXCEL_PATH}')
print(f'7. 混淆矩阵：{CM_FIG_PATH}')
print(f'8. 日尺度概率图：{PROB_FIG_PATH}')
print('=' * 60)
