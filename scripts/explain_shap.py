"""Explain TC-class probabilities using saved models and episode bootstrap."""
from __future__ import annotations
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
try:
    import shap
except ImportError as exc:
    raise ImportError('未安装 shap，请先运行：pip install shap') from exc
SAMPLE_NAME = SAMPLE
MODEL_RESULTS_DIR = MODEL_DIR
SAMPLE_CONFIGS = {name: {'data_path': PROCESSED_DIR / (name + '.xlsx')} for name in ['XB10', 'LHJ2']}
if SAMPLE_NAME not in SAMPLE_CONFIGS:
    raise ValueError(f'SAMPLE_NAME必须为{list(SAMPLE_CONFIGS)}之一，当前为：{SAMPLE_NAME}')
CONFIG = SAMPLE_CONFIGS[SAMPLE_NAME]
DATA_PATH = DATA_FILE
LEFT_WINDOW = CFG['left_window']
RIGHT_WINDOW = CFG['right_window']
TOTAL_WINDOW = LEFT_WINDOW + RIGHT_WINDOW + 1
SAMPLE_SLUG = SAMPLE_NAME.lower()
MODEL_PATH = MODEL_RESULTS_DIR / f'{SAMPLE_SLUG}_best.pth'
SCALER_PATH = MODEL_RESULTS_DIR / f'scaler_{SAMPLE_SLUG}.pkl'
OUTPUT_DIR = STAGE_OUTPUT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATE_COL_INDEX = 0
FEATURE_INDICES = FEATURE_INDICES_CONFIG
LABEL_COL_INDEX = -1
HIDDEN_SIZE = CFG['hidden_size']
TEST_SIZE = CFG['heldout_fraction']
RANDOM_SEED = CFG['seed']
BACKGROUND_SIZE = CFG['shap_background_size']
SHAP_NSAMPLES = CFG['shap_nsamples']
MODEL_PREDICT_BATCH_SIZE = 256
BOOTSTRAP_REPEATS = CFG['bootstrap_repeats']
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_UNIT = 'event'
CLASSIFICATION_THRESHOLD = CFG['classification_threshold']
RUN_TP_FN_ANALYSIS = False
SHOW_FIGURES = False
HEATMAP_VMIN = 0.0
HEATMAP_VMAX = 10.0
HEATMAP_TICKS = [0, 2, 4, 6, 8, 10]
TEMPORAL_CSV = OUTPUT_DIR / f'{SAMPLE_SLUG}_tc_temporal_contribution.csv'
TEMPORAL_XLSX = OUTPUT_DIR / f'{SAMPLE_SLUG}_tc_temporal_contribution.xlsx'
SIGNED_CSV = OUTPUT_DIR / f'{SAMPLE_SLUG}_tc_signed_temporal_shap.csv'
SIGNED_XLSX = OUTPUT_DIR / f'{SAMPLE_SLUG}_tc_signed_temporal_shap.xlsx'
HEATMAP_CSV = OUTPUT_DIR / f'{SAMPLE_SLUG}_tc_feature_time_abs_shap_matrix.csv'
HEATMAP_XLSX = OUTPUT_DIR / f'{SAMPLE_SLUG}_tc_feature_time_abs_shap_matrix.xlsx'
SUMMARY_CSV = OUTPUT_DIR / f'{SAMPLE_SLUG}_tc_temporal_organization_summary.csv'
SUMMARY_XLSX = OUTPUT_DIR / f'{SAMPLE_SLUG}_tc_temporal_organization_summary.xlsx'
ADDITIVITY_CSV = OUTPUT_DIR / f'{SAMPLE_SLUG}_tc_shap_additivity_check.csv'
EXPLAINED_SAMPLES_CSV = OUTPUT_DIR / f'{SAMPLE_SLUG}_tc_explained_samples.csv'
RAW_SHAP_NPZ = OUTPUT_DIR / f'{SAMPLE_SLUG}_tc_raw_shap_values.npz'
TEMPORAL_FIG = OUTPUT_DIR / f'{SAMPLE_SLUG}_tc_temporal_contribution.png'
SIGNED_FIG = OUTPUT_DIR / f'{SAMPLE_SLUG}_tc_signed_temporal_shap.png'
HEATMAP_FIG = OUTPUT_DIR / f'{SAMPLE_SLUG}_tc_feature_time_heatmap.png'
PHASE_FIG = OUTPUT_DIR / f'{SAMPLE_SLUG}_tc_pre_target_post_contribution.png'
PAPER_SUMMARY_FIG = OUTPUT_DIR / f'{SAMPLE_SLUG}_tc_temporal_organization_summary.png'
TP_FN_CSV = OUTPUT_DIR / f'{SAMPLE_SLUG}_tc_tp_fn_temporal_comparison.csv'
TP_FN_FIG = OUTPUT_DIR / f'{SAMPLE_SLUG}_tc_tp_fn_temporal_comparison.png'

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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_required_files() -> None:
    """Check required files."""
    missing = [p for p in (DATA_PATH, MODEL_PATH, SCALER_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError('以下文件不存在，请检查路径或先运行最终训练脚本：\n' + '\n'.join((str(p) for p in missing)))

def get_date_segments(dates: pd.Series) -> np.ndarray:
    """Assign a new segment whenever consecutive dates differ by anything other than one day."""
    dates = pd.to_datetime(dates).reset_index(drop=True)
    day_diff = dates.diff().dt.days
    new_segment = day_diff.isna() | (day_diff != 1)
    return new_segment.cumsum().to_numpy(dtype=int)

def build_sliding_window(X: np.ndarray, y: np.ndarray, dates: pd.Series, left_window: int, right_window: int) -> tuple[np.ndarray, np.ndarray]:
    """Build endpoint-padded sequences separately within each uninterrupted daily segment. Never bridge date gaps."""
    if len(X) != len(y) or len(X) != len(dates):
        raise ValueError(f'X、y和dates长度不一致：X={len(X)}, y={len(y)}, dates={len(dates)}')
    expected_window = left_window + right_window + 1
    segment_ids = get_date_segments(dates)
    X_seq: list[np.ndarray] = []
    y_seq: list[int] = []
    for segment_id in np.unique(segment_ids):
        segment_indices = np.where(segment_ids == segment_id)[0]
        X_segment = X[segment_indices]
        y_segment = y[segment_indices]
        segment_length = len(X_segment)
        for local_i in range(segment_length):
            start = local_i - left_window
            end = local_i + right_window + 1
            parts: list[np.ndarray] = []
            if start < 0:
                parts.append(np.repeat(X_segment[0:1], -start, axis=0))
                actual_start = 0
            else:
                actual_start = start
            actual_end = min(end, segment_length)
            parts.append(X_segment[actual_start:actual_end])
            if end > segment_length:
                parts.append(np.repeat(X_segment[-1:], end - segment_length, axis=0))
            window = np.vstack(parts)
            if len(window) != expected_window:
                raise RuntimeError(f'窗口长度错误：期望{expected_window}，实际{len(window)}')
            X_seq.append(window)
            y_seq.append(int(y_segment[local_i]))
    return (np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.int64))

def proportional_sample_indices(labels: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    """Proportional sample indices."""
    labels = np.asarray(labels, dtype=int)
    n_total = len(labels)
    if max_samples <= 0 or max_samples >= n_total:
        return np.arange(n_total, dtype=int)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    raw_alloc = max_samples * counts / counts.sum()
    alloc = np.floor(raw_alloc).astype(int)
    for i, count in enumerate(counts):
        if count > 0 and alloc[i] == 0 and (max_samples >= len(classes)):
            alloc[i] = 1
    while alloc.sum() < max_samples:
        remaining_capacity = counts - alloc
        valid = np.where(remaining_capacity > 0)[0]
        if len(valid) == 0:
            break
        fractional = raw_alloc - np.floor(raw_alloc)
        candidate = valid[np.argmax(fractional[valid] + 1e-12 * remaining_capacity[valid])]
        alloc[candidate] += 1
    while alloc.sum() > max_samples:
        valid = np.where(alloc > 1)[0]
        if len(valid) == 0:
            valid = np.where(alloc > 0)[0]
        candidate = valid[np.argmax(alloc[valid])]
        alloc[candidate] -= 1
    selected: list[int] = []
    for cls, n_select in zip(classes, alloc):
        cls_idx = np.where(labels == cls)[0]
        sampled = rng.choice(cls_idx, size=int(n_select), replace=False)
        selected.extend(sampled.tolist())
    selected = np.asarray(selected, dtype=int)
    rng.shuffle(selected)
    return np.sort(selected)

def normalize_shap_result(shap_result, expected_shape: tuple[int, int]) -> np.ndarray:
    """Normalize shap result."""
    if isinstance(shap_result, list):
        if len(shap_result) != 1:
            raise RuntimeError(f'模型预测函数只返回一个TC概率，但SHAP返回了{len(shap_result)}组结果。')
        arr = np.asarray(shap_result[0], dtype=float)
    else:
        arr = np.asarray(shap_result, dtype=float)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.shape != expected_shape:
        raise RuntimeError(f'SHAP结果形状与解释输入不一致：SHAP={arr.shape}, input={expected_shape}')
    return arr

def build_tc_episode_ids(tc_dates: pd.Series) -> np.ndarray:
    """Group consecutive TC-labelled dates into episodes for bootstrap resampling."""
    dates = pd.to_datetime(tc_dates).reset_index(drop=True)
    if len(dates) == 0:
        return np.asarray([], dtype=int)
    day_diff = dates.diff().dt.days
    new_event = day_diff.isna() | (day_diff != 1)
    return new_event.cumsum().to_numpy(dtype=int) - 1

def bootstrap_sample_indices(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap sample indices."""
    return rng.integers(0, n_samples, size=n_samples)

def bootstrap_event_indices(event_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap event indices."""
    unique_events = np.unique(event_ids)
    sampled_events = rng.choice(unique_events, size=len(unique_events), replace=True)
    sampled_indices: list[int] = []
    for event_id in sampled_events:
        sampled_indices.extend(np.where(event_ids == event_id)[0].tolist())
    return np.asarray(sampled_indices, dtype=int)

def get_bootstrap_indices(n_samples: int, event_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Get bootstrap indices."""
    if BOOTSTRAP_UNIT == 'sample':
        return bootstrap_sample_indices(n_samples, rng)
    if BOOTSTRAP_UNIT == 'event':
        n_events = len(np.unique(event_ids))
        if n_events >= 2:
            return bootstrap_event_indices(event_ids, rng)
        return bootstrap_sample_indices(n_samples, rng)
    raise ValueError("BOOTSTRAP_UNIT必须为'event'或'sample'")

def temporal_relative_contribution(abs_shap_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Temporal relative contribution."""
    time_values = abs_shap_3d.mean(axis=(0, 2))
    total = time_values.sum()
    relative = time_values / total * 100.0 if total > 0 else np.zeros_like(time_values)
    return (time_values, relative)

def signed_temporal_shap(shap_3d: np.ndarray) -> np.ndarray:
    """Signed temporal shap."""
    return shap_3d.mean(axis=(0, 2))

def temporal_summary(abs_shap_3d: np.ndarray, offsets: np.ndarray) -> dict[str, float]:
    """Temporal summary."""
    time_values, relative = temporal_relative_contribution(abs_shap_3d)
    total = time_values.sum()
    pre_mask = offsets < 0
    target_mask = offsets == 0
    post_mask = offsets > 0
    pre_share = float(relative[pre_mask].sum())
    target_share = float(relative[target_mask].sum())
    post_share = float(relative[post_mask].sum())
    pre_mean = float(time_values[pre_mask].mean())
    target_mean = float(time_values[target_mask].mean())
    post_mean = float(time_values[post_mask].mean())
    asymmetry_denom = pre_mean + post_mean
    asymmetry = float((post_mean - pre_mean) / asymmetry_denom) if asymmetry_denom > 0 else np.nan
    post_pre_ratio = float(post_mean / pre_mean) if pre_mean > 0 else np.nan
    centroid = float(np.sum(offsets * time_values) / total) if total > 0 else np.nan
    uniform_window_centroid = float(np.mean(offsets))
    adjusted_centroid = float(centroid - uniform_window_centroid) if np.isfinite(centroid) else np.nan
    peak_idx = int(np.argmax(time_values))
    return {'pre_target_share_percent': pre_share, 'target_day_share_percent': target_share, 'post_target_share_percent': post_share, 'pre_mean_abs_shap_per_position': pre_mean, 'target_mean_abs_shap': target_mean, 'post_mean_abs_shap_per_position': post_mean, 'post_to_pre_mean_ratio': post_pre_ratio, 'temporal_asymmetry_index': asymmetry, 'shap_weighted_centroid_day': centroid, 'uniform_window_centroid_day': uniform_window_centroid, 'window_adjusted_centroid_day': adjusted_centroid, 'peak_offset_day': float(offsets[peak_idx]), 'peak_relative_contribution_percent': float(relative[peak_idx])}

def bootstrap_temporal_results(shap_3d: np.ndarray, offsets: np.ndarray, event_ids: np.ndarray, repeats: int, confidence: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, tuple[float, float]]]:
    """Estimate temporal contribution intervals by the configured resampling unit."""
    rng = np.random.default_rng(seed)
    n_samples = shap_3d.shape[0]
    relative_dist: list[np.ndarray] = []
    signed_dist: list[np.ndarray] = []
    summary_metric_names = ['pre_target_share_percent', 'target_day_share_percent', 'post_target_share_percent', 'pre_mean_abs_shap_per_position', 'target_mean_abs_shap', 'post_mean_abs_shap_per_position', 'post_to_pre_mean_ratio', 'temporal_asymmetry_index', 'shap_weighted_centroid_day', 'window_adjusted_centroid_day']
    summary_dist: dict[str, list[float]] = {k: [] for k in summary_metric_names}
    abs_shap_3d = np.abs(shap_3d)
    for _ in range(repeats):
        sampled_idx = get_bootstrap_indices(n_samples, event_ids, rng)
        sampled_shap = shap_3d[sampled_idx]
        sampled_abs = abs_shap_3d[sampled_idx]
        _, relative = temporal_relative_contribution(sampled_abs)
        signed = signed_temporal_shap(sampled_shap)
        summary = temporal_summary(sampled_abs, offsets)
        relative_dist.append(relative)
        signed_dist.append(signed)
        for metric in summary_metric_names:
            summary_dist[metric].append(summary[metric])
    relative_arr = np.vstack(relative_dist)
    signed_arr = np.vstack(signed_dist)
    alpha = (1.0 - confidence) / 2.0
    relative_lower = np.quantile(relative_arr, alpha, axis=0)
    relative_upper = np.quantile(relative_arr, 1.0 - alpha, axis=0)
    signed_lower = np.quantile(signed_arr, alpha, axis=0)
    signed_upper = np.quantile(signed_arr, 1.0 - alpha, axis=0)
    summary_ci: dict[str, tuple[float, float]] = {}
    for metric, values in summary_dist.items():
        arr = np.asarray(values, dtype=float)
        summary_ci[metric] = (float(np.nanquantile(arr, alpha)), float(np.nanquantile(arr, 1.0 - alpha)))
    return (relative_lower, relative_upper, signed_lower, signed_upper, summary_ci)

def finish_figure(fig: plt.Figure, output_path: Path) -> None:
    """Finish figure."""
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    if SHOW_FIGURES:
        pass
    plt.close(fig)

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
        return self.classifier(context)
check_required_files()
print('=' * 80)
print(f'样品：{SAMPLE_NAME}')
print(f'数据：{DATA_PATH}')
print(f'模型：{MODEL_PATH}')
print(f'标准化器：{SCALER_PATH}')
print(f'时间窗口：前{LEFT_WINDOW} d + 目标日 + 后{RIGHT_WINDOW} d = {TOTAL_WINDOW} d')
print(f'Bootstrap unit：{BOOTSTRAP_UNIT}')
print(f'计算设备：{DEVICE}')
print('=' * 80)
DF = pd.read_excel(DATA_PATH)
if len(DF.columns) < 8:
    raise ValueError('Excel至少应包含日期、6个特征和标签列。')
DATE_COL = 'date'
FEATURE_COLS = FEATURES
LABEL_COL = 'label'
DF[DATE_COL] = pd.to_datetime(DF[DATE_COL], errors='coerce')
if DF[DATE_COL].isna().any():
    bad_rows = DF.index[DF[DATE_COL].isna()].tolist()
    raise ValueError(f'日期列存在无法解析的值，对应行号：{bad_rows[:10]}')
DF = DF.sort_values(DATE_COL).reset_index(drop=True)
if DF[FEATURE_COLS].isna().any().any():
    missing = DF[FEATURE_COLS].isna().sum()
    missing = missing[missing > 0]
    raise ValueError(f'特征列存在缺失值：\n{missing}')
DF[LABEL_COL] = DF[LABEL_COL].astype(int)
if not set(DF[LABEL_COL].unique()).issubset({0, 1}):
    raise ValueError(f'标签只能为0和1，当前标签值：{sorted(DF[LABEL_COL].unique())}')
split_idx = int(len(DF) * (1.0 - TEST_SIZE))
DF_TRAIN = DF.iloc[:split_idx].copy()
DF_TEST = DF.iloc[split_idx:].copy()
SCALER = joblib.load(SCALER_PATH)
X_TRAIN_RAW = SCALER.transform(DF_TRAIN[FEATURE_COLS])
X_TEST_RAW = SCALER.transform(DF_TEST[FEATURE_COLS])
Y_TRAIN_RAW = DF_TRAIN[LABEL_COL].to_numpy(dtype=int)
Y_TEST_RAW = DF_TEST[LABEL_COL].to_numpy(dtype=int)
TRAIN_DATES = DF_TRAIN[DATE_COL].reset_index(drop=True)
TEST_DATES = DF_TEST[DATE_COL].reset_index(drop=True)
X_TRAIN, Y_TRAIN = build_sliding_window(X_TRAIN_RAW, Y_TRAIN_RAW, TRAIN_DATES, LEFT_WINDOW, RIGHT_WINDOW)
X_TEST, Y_TEST = build_sliding_window(X_TEST_RAW, Y_TEST_RAW, TEST_DATES, LEFT_WINDOW, RIGHT_WINDOW)
if len(TEST_DATES) != len(X_TEST):
    raise RuntimeError('TEST_DATES与X_TEST长度不一致。')
print(f'使用特征：{FEATURE_COLS}')
print(f'训练窗口：{X_TRAIN.shape}')
print(f'测试窗口：{X_TEST.shape}')
print('测试标签分布：', dict(zip(*np.unique(Y_TEST, return_counts=True))))
MODEL = BiLSTMAttention(input_dim=len(FEATURE_COLS), hidden_size=HIDDEN_SIZE).to(DEVICE)
try:
    STATE_DICT = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
except TypeError:
    STATE_DICT = torch.load(MODEL_PATH, map_location=DEVICE)
MODEL.load_state_dict(STATE_DICT)
MODEL.eval()

def predict_tc_probability(flat_inputs: np.ndarray) -> np.ndarray:
    """Map flattened feature windows to the TC-class softmax probability for Kernel SHAP."""
    flat_inputs = np.asarray(flat_inputs, dtype=np.float32)
    if flat_inputs.ndim == 1:
        flat_inputs = flat_inputs.reshape(1, -1)
    expected_dim = TOTAL_WINDOW * len(FEATURE_COLS)
    if flat_inputs.shape[1] != expected_dim:
        raise ValueError(f'扁平输入维度错误：期望{expected_dim}，实际{flat_inputs.shape[1]}')
    sequence_inputs = flat_inputs.reshape(-1, TOTAL_WINDOW, len(FEATURE_COLS))
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sequence_inputs), MODEL_PREDICT_BATCH_SIZE):
            end = start + MODEL_PREDICT_BATCH_SIZE
            batch = torch.tensor(sequence_inputs[start:end], dtype=torch.float32, device=DEVICE)
            logits = MODEL(batch)
            tc_probability = torch.softmax(logits, dim=1)[:, 1]
            probabilities.append(tc_probability.cpu().numpy())
    return np.concatenate(probabilities, axis=0).astype(float)
background_indices = proportional_sample_indices(labels=Y_TRAIN, max_samples=min(BACKGROUND_SIZE, len(Y_TRAIN)), seed=RANDOM_SEED)
X_BACKGROUND = X_TRAIN[background_indices]
X_BACKGROUND_FLAT = X_BACKGROUND.reshape(len(X_BACKGROUND), -1)
TC_INDICES = np.where(Y_TEST == 1)[0]
if len(TC_INDICES) == 0:
    raise RuntimeError('test set中没有TC-labelled样本，无法进行TC-specific SHAP。')
X_TC = X_TEST[TC_INDICES]
Y_TC = Y_TEST[TC_INDICES]
DATES_TC = TEST_DATES.iloc[TC_INDICES].reset_index(drop=True)
X_TC_FLAT = X_TC.reshape(len(X_TC), -1)
TC_EPISODE_IDS = build_tc_episode_ids(DATES_TC)
N_TC_EPISODES = len(np.unique(TC_EPISODE_IDS))
TIME_LABELS = ['t' if offset == 0 else f't{offset:+d}' for offset in range(-LEFT_WINDOW, RIGHT_WINDOW + 1)]
OFFSETS = np.arange(-LEFT_WINDOW, RIGHT_WINDOW + 1, dtype=int)
print('\n========== TC-specific SHAP样本 ==========')
print(f'Background窗口数：{len(X_BACKGROUND)}')
print(f'全部true TC解释样本数：{len(X_TC)}')
print(f'连续TC episodes数：{N_TC_EPISODES}')
print(f'Kernel SHAP nsamples：{SHAP_NSAMPLES}')
print('\n开始计算TC-specific Kernel SHAP。该过程可能需要数分钟……')
EXPLAINER = shap.KernelExplainer(predict_tc_probability, X_BACKGROUND_FLAT)
EXPECTED_VALUE_ARRAY = np.asarray(EXPLAINER.expected_value, dtype=float).reshape(-1)
if EXPECTED_VALUE_ARRAY.size != 1:
    raise RuntimeError(f'预期一个SHAP基线值，实际得到：{EXPECTED_VALUE_ARRAY}')
BASE_VALUE = float(EXPECTED_VALUE_ARRAY[0])
TC_SHAP_RESULT = EXPLAINER.shap_values(X_TC_FLAT, nsamples=SHAP_NSAMPLES)
TC_SHAP_FLAT = normalize_shap_result(TC_SHAP_RESULT, X_TC_FLAT.shape)
TC_SHAP_3D = TC_SHAP_FLAT.reshape(len(X_TC), TOTAL_WINDOW, len(FEATURE_COLS))
TC_ABS_SHAP_3D = np.abs(TC_SHAP_3D)
print('TC-specific SHAP计算完成。')
print(f'SHAP背景基线TC概率：{BASE_VALUE:.6f}')
TC_MODEL_PROBABILITIES = predict_tc_probability(X_TC_FLAT)
TC_SHAP_RECONSTRUCTED = BASE_VALUE + TC_SHAP_FLAT.sum(axis=1)
TC_ADDITIVITY_DIFF = TC_MODEL_PROBABILITIES - TC_SHAP_RECONSTRUCTED
ADDITIVITY_DF = pd.DataFrame({'test_index': TC_INDICES, 'date': DATES_TC, 'true_label': Y_TC, 'model_tc_probability': TC_MODEL_PROBABILITIES, 'shap_base_value': BASE_VALUE, 'base_plus_shap_sum': TC_SHAP_RECONSTRUCTED, 'difference': TC_ADDITIVITY_DIFF, 'absolute_difference': np.abs(TC_ADDITIVITY_DIFF)})
ADDITIVITY_DF.to_csv(ADDITIVITY_CSV, index=False, encoding='utf-8-sig')
print('\n========== SHAP加和检查 ==========')
print(f'平均绝对差值：{np.mean(np.abs(TC_ADDITIVITY_DIFF)):.8f}')
print(f'最大绝对差值：{np.max(np.abs(TC_ADDITIVITY_DIFF)):.8f}')
TIME_MEAN_ABS_SHAP, TIME_RELATIVE_PERCENT = temporal_relative_contribution(TC_ABS_SHAP_3D)
SIGNED_TIME_MEAN_SHAP = signed_temporal_shap(TC_SHAP_3D)
TIME_CI_LOWER, TIME_CI_UPPER, SIGNED_CI_LOWER, SIGNED_CI_UPPER, SUMMARY_CI = bootstrap_temporal_results(shap_3d=TC_SHAP_3D, offsets=OFFSETS, event_ids=TC_EPISODE_IDS, repeats=BOOTSTRAP_REPEATS, confidence=BOOTSTRAP_CONFIDENCE, seed=RANDOM_SEED + 1)
TEMPORAL_DF = pd.DataFrame({'time_position': TIME_LABELS, 'offset_day': OFFSETS, 'mean_abs_shap': TIME_MEAN_ABS_SHAP, 'relative_contribution_percent': TIME_RELATIVE_PERCENT, 'ci_lower_percent': TIME_CI_LOWER, 'ci_upper_percent': TIME_CI_UPPER})
TEMPORAL_DF.to_csv(TEMPORAL_CSV, index=False, encoding='utf-8-sig')
TEMPORAL_DF.to_excel(TEMPORAL_XLSX, index=False)
SIGNED_DF = pd.DataFrame({'time_position': TIME_LABELS, 'offset_day': OFFSETS, 'mean_signed_shap': SIGNED_TIME_MEAN_SHAP, 'ci_lower': SIGNED_CI_LOWER, 'ci_upper': SIGNED_CI_UPPER})
SIGNED_DF.to_csv(SIGNED_CSV, index=False, encoding='utf-8-sig')
SIGNED_DF.to_excel(SIGNED_XLSX, index=False)
FEATURE_TIME_MEAN_ABS = TC_ABS_SHAP_3D.mean(axis=0)
FEATURE_TIME_TOTAL = FEATURE_TIME_MEAN_ABS.sum()
FEATURE_TIME_RELATIVE = FEATURE_TIME_MEAN_ABS / FEATURE_TIME_TOTAL * 100.0 if FEATURE_TIME_TOTAL > 0 else np.zeros_like(FEATURE_TIME_MEAN_ABS)
HEATMAP_DF = pd.DataFrame(FEATURE_TIME_RELATIVE.T, index=FEATURE_COLS, columns=TIME_LABELS)
HEATMAP_DF.index.name = 'feature'
HEATMAP_DF.to_csv(HEATMAP_CSV, encoding='utf-8-sig')
HEATMAP_DF.to_excel(HEATMAP_XLSX)
SUMMARY = temporal_summary(TC_ABS_SHAP_3D, OFFSETS)
summary_rows: list[dict[str, object]] = [{'metric': 'n_tc_labelled_test_samples', 'value': float(len(X_TC)), 'ci_lower': np.nan, 'ci_upper': np.nan, 'note': 'Number of all true TC-labelled samples used for TC-specific SHAP.'}, {'metric': 'n_contiguous_tc_episodes', 'value': float(N_TC_EPISODES), 'ci_lower': np.nan, 'ci_upper': np.nan, 'note': 'Consecutive TC-labelled days grouped by calendar continuity.'}]
for metric, value in SUMMARY.items():
    ci_lower, ci_upper = SUMMARY_CI.get(metric, (np.nan, np.nan))
    if metric == 'temporal_asymmetry_index':
        note = '(post mean |SHAP| per position - pre mean |SHAP| per position) / (post + pre); >0 indicates relatively stronger post-target information.'
    elif metric == 'shap_weighted_centroid_day':
        note = 'SHAP-weighted temporal centroid in raw offset-day coordinates.'
    elif metric == 'window_adjusted_centroid_day':
        note = 'Raw centroid minus the uniform-contribution centroid of the asymmetric window; 0 is the neutral baseline for this window.'
    elif metric == 'post_to_pre_mean_ratio':
        note = 'Post-target mean |SHAP| per position divided by pre-target mean |SHAP| per position.'
    elif metric == 'peak_offset_day':
        note = 'Temporal position with the largest mean |SHAP|; CI not estimated as a continuous metric.'
    elif metric == 'peak_relative_contribution_percent':
        note = 'Relative contribution of the single peak temporal position; CI omitted here.'
    else:
        note = 'TC-specific temporal organization metric.'
    summary_rows.append({'metric': metric, 'value': value, 'ci_lower': ci_lower, 'ci_upper': ci_upper, 'note': note})
SUMMARY_DF = pd.DataFrame(summary_rows)
SUMMARY_DF.to_csv(SUMMARY_CSV, index=False, encoding='utf-8-sig')
SUMMARY_DF.to_excel(SUMMARY_XLSX, index=False)
print('\n========== TC-specific temporal contribution ==========')
print(TEMPORAL_DF.to_string(index=False))
print('\n========== TC-specific signed temporal SHAP ==========')
print(SIGNED_DF.to_string(index=False))
print('\n========== TC-specific temporal organization summary ==========')
print(SUMMARY_DF.to_string(index=False))
TC_PREDICTED_CLASS = (TC_MODEL_PROBABILITIES >= CLASSIFICATION_THRESHOLD).astype(int)
EXPLAINED_SAMPLES_DF = pd.DataFrame({'test_index': TC_INDICES, 'date': DATES_TC, 'tc_episode_id': TC_EPISODE_IDS, 'true_label': Y_TC, 'model_tc_probability': TC_MODEL_PROBABILITIES, 'predicted_class': TC_PREDICTED_CLASS})
EXPLAINED_SAMPLES_DF.to_csv(EXPLAINED_SAMPLES_CSV, index=False, encoding='utf-8-sig')
np.savez_compressed(RAW_SHAP_NPZ, shap_values=TC_SHAP_3D, abs_shap_values=TC_ABS_SHAP_3D, explained_inputs=X_TC, explained_labels=Y_TC, explained_indices=TC_INDICES, explained_dates=DATES_TC.astype(str).to_numpy(), tc_episode_ids=TC_EPISODE_IDS, model_tc_probabilities=TC_MODEL_PROBABILITIES, predicted_class=TC_PREDICTED_CLASS, base_value=np.asarray([BASE_VALUE]), feature_names=np.asarray(FEATURE_COLS, dtype=str), time_labels=np.asarray(TIME_LABELS, dtype=str), offsets=OFFSETS)
TP_FN_DF = None
if RUN_TP_FN_ANALYSIS:
    rows: list[dict[str, object]] = []
    for group_name, mask in {'TP': TC_PREDICTED_CLASS == 1, 'FN': TC_PREDICTED_CLASS == 0}.items():
        group_abs = TC_ABS_SHAP_3D[mask]
        if len(group_abs) == 0:
            continue
        _, group_relative = temporal_relative_contribution(group_abs)
        for label, offset, value in zip(TIME_LABELS, OFFSETS, group_relative):
            rows.append({'group': group_name, 'n_samples': int(len(group_abs)), 'time_position': label, 'offset_day': int(offset), 'relative_contribution_percent': float(value)})
    if rows:
        TP_FN_DF = pd.DataFrame(rows)
        TP_FN_DF.to_csv(TP_FN_CSV, index=False, encoding='utf-8-sig')
sns.set_theme(style='whitegrid')
fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor='white')
ax.plot(TEMPORAL_DF['offset_day'], TEMPORAL_DF['relative_contribution_percent'], marker='o')
ax.fill_between(TEMPORAL_DF['offset_day'].to_numpy(), TEMPORAL_DF['ci_lower_percent'].to_numpy(), TEMPORAL_DF['ci_upper_percent'].to_numpy(), alpha=0.2)
ax.axvline(0, linestyle='--', linewidth=1)
ax.set_xticks(OFFSETS)
ax.set_xticklabels(TIME_LABELS)
ax.set_xlabel('Temporal position relative to target day')
ax.set_ylabel('Relative mean |SHAP| contribution (%)')
ax.set_title(f'{SAMPLE_NAME}: TC-specific temporal contribution')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
finish_figure(fig, TEMPORAL_FIG)
fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor='white')
yerr = np.vstack([SIGNED_DF['mean_signed_shap'] - SIGNED_DF['ci_lower'], SIGNED_DF['ci_upper'] - SIGNED_DF['mean_signed_shap']])
ax.errorbar(SIGNED_DF['offset_day'], SIGNED_DF['mean_signed_shap'], yerr=yerr, marker='o', capsize=3)
ax.axhline(0, linestyle='--', linewidth=1)
ax.axvline(0, linestyle='--', linewidth=1)
ax.set_xticks(OFFSETS)
ax.set_xticklabels(TIME_LABELS)
ax.set_xlabel('Temporal position relative to target day')
ax.set_ylabel('Mean signed SHAP value')
ax.set_title(f'{SAMPLE_NAME}: direction of TC-specific temporal contributions')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
finish_figure(fig, SIGNED_FIG)
fig, ax = plt.subplots(figsize=(10.0, 5.2), facecolor='white')
sns.heatmap(HEATMAP_DF, annot=True, fmt='.2f', cmap='YlGnBu', vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX, cbar_kws={'label': 'Relative mean |SHAP| contribution (%)', 'ticks': HEATMAP_TICKS}, linewidths=0.4, linecolor='white', ax=ax)
ax.set_xlabel('Temporal position relative to target day')
ax.set_ylabel('Feature')
ax.set_title(f'{SAMPLE_NAME}: TC-specific feature × time contribution')
finish_figure(fig, HEATMAP_FIG)
phase_names = ['Pre-target', 'Target day', 'Post-target']
phase_values = [SUMMARY['pre_target_share_percent'], SUMMARY['target_day_share_percent'], SUMMARY['post_target_share_percent']]
phase_ci_lower = [SUMMARY_CI['pre_target_share_percent'][0], SUMMARY_CI['target_day_share_percent'][0], SUMMARY_CI['post_target_share_percent'][0]]
phase_ci_upper = [SUMMARY_CI['pre_target_share_percent'][1], SUMMARY_CI['target_day_share_percent'][1], SUMMARY_CI['post_target_share_percent'][1]]
phase_yerr = np.vstack([np.asarray(phase_values) - np.asarray(phase_ci_lower), np.asarray(phase_ci_upper) - np.asarray(phase_values)])
fig, ax = plt.subplots(figsize=(6.5, 4.8), facecolor='white')
ax.bar(phase_names, phase_values, yerr=phase_yerr, capsize=4)
ax.set_ylabel('Relative mean |SHAP| contribution (%)')
ax.set_title(f'{SAMPLE_NAME}: pre-, target-, and post-target TC information')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
finish_figure(fig, PHASE_FIG)
fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2), facecolor='white')
axes[0].plot(TEMPORAL_DF['offset_day'], TEMPORAL_DF['relative_contribution_percent'], marker='o')
axes[0].fill_between(TEMPORAL_DF['offset_day'].to_numpy(), TEMPORAL_DF['ci_lower_percent'].to_numpy(), TEMPORAL_DF['ci_upper_percent'].to_numpy(), alpha=0.2)
axes[0].axvline(0, linestyle='--', linewidth=1)
axes[0].set_xticks(OFFSETS)
axes[0].set_xticklabels(TIME_LABELS)
axes[0].set_xlabel('Relative day')
axes[0].set_ylabel('Relative mean |SHAP| (%)')
axes[0].set_title('(a) Temporal contribution')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[1].errorbar(SIGNED_DF['offset_day'], SIGNED_DF['mean_signed_shap'], yerr=yerr, marker='o', capsize=3)
axes[1].axhline(0, linestyle='--', linewidth=1)
axes[1].axvline(0, linestyle='--', linewidth=1)
axes[1].set_xticks(OFFSETS)
axes[1].set_xticklabels(TIME_LABELS)
axes[1].set_xlabel('Relative day')
axes[1].set_ylabel('Mean signed SHAP')
axes[1].set_title('(b) Direction of contribution')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
sns.heatmap(HEATMAP_DF, annot=False, cmap='YlGnBu', vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX, cbar_kws={'label': 'Relative mean |SHAP| (%)', 'ticks': HEATMAP_TICKS}, linewidths=0.3, linecolor='white', ax=axes[2])
axes[2].set_xlabel('Relative day')
axes[2].set_ylabel('Feature')
axes[2].set_title('(c) Feature × time contribution')
fig.suptitle(f'{SAMPLE_NAME}: TC-specific temporal organization of model contributions', y=1.02)
finish_figure(fig, PAPER_SUMMARY_FIG)
if TP_FN_DF is not None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor='white')
    for group_name, group_df in TP_FN_DF.groupby('group'):
        ax.plot(group_df['offset_day'], group_df['relative_contribution_percent'], marker='o', label=f"{group_name} (n={int(group_df['n_samples'].iloc[0])})")
    ax.axvline(0, linestyle='--', linewidth=1)
    ax.set_xticks(OFFSETS)
    ax.set_xticklabels(TIME_LABELS)
    ax.set_xlabel('Temporal position relative to target day')
    ax.set_ylabel('Relative mean |SHAP| contribution (%)')
    ax.set_title(f'{SAMPLE_NAME}: TP vs FN temporal contribution')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    finish_figure(fig, TP_FN_FIG)
print('\n' + '=' * 80)
print('TC-specific temporal SHAP分析完成。')
print('本脚本未计算、未输出任何Global SHAP结果。')
print('\n核心输出：')
print(f'1. TC时间贡献表：{TEMPORAL_XLSX}')
print(f'2. TC signed时间贡献表：{SIGNED_XLSX}')
print(f'3. Feature×time矩阵：{HEATMAP_XLSX}')
print(f'4. 时间组织汇总指标：{SUMMARY_XLSX}')
print(f'5. SHAP加和检查：{ADDITIVITY_CSV}')
print(f'6. 解释样本：{EXPLAINED_SAMPLES_CSV}')
print(f'7. Raw SHAP数组：{RAW_SHAP_NPZ}')
print(f'8. 时间贡献图：{TEMPORAL_FIG}')
print(f'9. Signed SHAP图：{SIGNED_FIG}')
print(f'10. Feature×time热图：{HEATMAP_FIG}')
print(f'11. Pre/target/post图：{PHASE_FIG}')
print(f'12. 论文三联汇总图：{PAPER_SUMMARY_FIG}')
if RUN_TP_FN_ANALYSIS:
    print(f'13. TP/FN对比表：{TP_FN_CSV}')
    print(f'14. TP/FN对比图：{TP_FN_FIG}')
print('=' * 80)
