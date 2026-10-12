"""Train the flattened-window Random Forest baseline."""
from project_config import *
import random
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
SAMPLE_NAME = SAMPLE
DATA_PATH = DATA_FILE
OUTPUT_DIR = STAGE_OUTPUT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sample_slug = SAMPLE_NAME.lower()
MODEL_PATH = OUTPUT_DIR / f'{sample_slug}_random_forest_best.pkl'
FEATURE_PATH = OUTPUT_DIR / f'{sample_slug}_random_forest_features.csv'
RESULT_CSV_PATH = OUTPUT_DIR / f'{sample_slug}_random_forest_test_daily_probability.csv'
RESULT_EXCEL_PATH = OUTPUT_DIR / f'{sample_slug}_random_forest_test_daily_probability.xlsx'
REPORT_PATH = OUTPUT_DIR / f'{sample_slug}_random_forest_classification_report.csv'
METRICS_PATH = OUTPUT_DIR / f'{sample_slug}_random_forest_metrics.csv'
CM_FIG_PATH = OUTPUT_DIR / f'{sample_slug}_random_forest_confusion_matrix.png'
PROB_FIG_PATH = OUTPUT_DIR / f'{sample_slug}_random_forest_daily_tc_probability.png'
IMPORTANCE_PATH = OUTPUT_DIR / f'{sample_slug}_random_forest_feature_importance.csv'
IMPORTANCE_FIG_PATH = OUTPUT_DIR / f'{sample_slug}_random_forest_feature_importance.png'
DATE_COL_INDEX = 0
FEATURE_INDICES = FEATURE_INDICES_CONFIG
LABEL_COL_INDEX = -1
LEFT_WINDOW = CFG['left_window']
RIGHT_WINDOW = CFG['right_window']
TOTAL_WINDOW = LEFT_WINDOW + RIGHT_WINDOW + 1
N_ESTIMATORS = 300
RANDOM_FOREST_MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
CLASS_WEIGHT = 'balanced'
VALIDATION_SIZE = CFG['heldout_fraction']
CLASSIFICATION_THRESHOLD = CFG['classification_threshold']
RANDOM_SEED = CFG['seed']

def set_seed(seed: int=42) -> None:
    """Seed Python, NumPy and PyTorch where used; deterministic kernels do not guarantee cross-platform bitwise equality."""
    random.seed(seed)
    np.random.seed(seed)
set_seed(RANDOM_SEED)
print('=' * 70)
print(f'样品：{SAMPLE_NAME}')
print(f'数据文件：{DATA_PATH}')
print(f'时间窗口：前{LEFT_WINDOW} d + 目标日 + 后{RIGHT_WINDOW} d = {TOTAL_WINDOW} d')
print(f'Random Forest：{N_ESTIMATORS} trees')
print('=' * 70)
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
split_idx = int(len(df) * (1 - VALIDATION_SIZE))
if split_idx <= 0 or split_idx >= len(df):
    raise ValueError('训练集和验证集划分失败。')
df_train = df.iloc[:split_idx].copy()
df_val = df.iloc[split_idx:].copy()
print('\n数据集划分：')
print(f'训练集：{len(df_train)}条，{df_train[date_col].min().date()} 至 {df_train[date_col].max().date()}')
print(f'验证集：{len(df_val)}条，{df_val[date_col].min().date()} 至 {df_val[date_col].max().date()}')
print('训练集标签分布：', df_train[label_col].value_counts().sort_index().to_dict())
print('验证集标签分布：', df_val[label_col].value_counts().sort_index().to_dict())
X_train_raw = df_train[feature_cols].to_numpy(dtype=np.float32)
X_val_raw = df_val[feature_cols].to_numpy(dtype=np.float32)
y_train_raw = df_train[label_col].to_numpy(dtype=int)
y_val_raw = df_val[label_col].to_numpy(dtype=int)
train_dates = df_train[date_col].reset_index(drop=True)
val_dates = df_val[date_col].reset_index(drop=True)

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
X_train_seq, y_train = build_sliding_window(X=X_train_raw, y=y_train_raw, dates=train_dates, left_window=LEFT_WINDOW, right_window=RIGHT_WINDOW)
X_val_seq, y_val = build_sliding_window(X=X_val_raw, y=y_val_raw, dates=val_dates, left_window=LEFT_WINDOW, right_window=RIGHT_WINDOW)
print('\n窗口构造结果：')
print(f'训练集窗口形状：{X_train_seq.shape}')
print(f'验证集窗口形状：{X_val_seq.shape}')
print('窗口长度是否正确：', X_train_seq.shape[1] == TOTAL_WINDOW)

def flatten_time_window(X_seq: np.ndarray, feature_cols: list[str], left_window: int, right_window: int) -> tuple[np.ndarray, list[str]]:
    """Flatten time window."""
    n_samples = X_seq.shape[0]
    n_time_steps = X_seq.shape[1]
    n_features = X_seq.shape[2]
    expected_time_steps = left_window + right_window + 1
    if n_time_steps != expected_time_steps:
        raise ValueError(f'时间窗口长度错误：实际={n_time_steps}，期望={expected_time_steps}')
    if n_features != len(feature_cols):
        raise ValueError('特征数量不一致。')
    X_flat = X_seq.reshape(n_samples, n_time_steps * n_features)
    feature_names = []
    time_offsets = list(range(-left_window, right_window + 1))
    for offset in time_offsets:
        if offset < 0:
            time_name = f't{offset}'
        elif offset == 0:
            time_name = 't'
        else:
            time_name = f't+{offset}'
        for feature in feature_cols:
            feature_names.append(f'{feature}_{time_name}')
    if len(feature_names) != X_flat.shape[1]:
        raise RuntimeError('特征名称数量与实际特征数量不一致。')
    return (X_flat.astype(np.float32), feature_names)
X_train_rf, feature_names = flatten_time_window(X_train_seq, feature_cols, LEFT_WINDOW, RIGHT_WINDOW)
X_val_rf, _ = flatten_time_window(X_val_seq, feature_cols, LEFT_WINDOW, RIGHT_WINDOW)
print('\nRandom Forest输入：')
print(f'训练集：{X_train_rf.shape}')
print(f'验证集：{X_val_rf.shape}')
print(f'RF输入特征数量：{len(feature_names)}')
feature_name_df = pd.DataFrame({'feature_index': np.arange(len(feature_names)), 'feature_name': feature_names})
feature_name_df.to_csv(FEATURE_PATH, index=False, encoding='utf-8-sig')
class_counts = np.bincount(y_train, minlength=2)
if np.any(class_counts == 0):
    raise ValueError(f'训练集中至少有一个类别没有样本：{class_counts}')
print('\nRandom Forest训练集类别数量：', class_counts)
rf_model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=RANDOM_FOREST_MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, min_samples_leaf=MIN_SAMPLES_LEAF, class_weight=CLASS_WEIGHT, random_state=RANDOM_SEED, n_jobs=CFG.get("rf_n_jobs", 1))
print('\n开始训练Random Forest...')
rf_model.fit(X_train_rf, y_train)
print('Random Forest训练完成。')
joblib.dump(rf_model, MODEL_PATH)
print(f'Random Forest模型已保存：{MODEL_PATH}')
all_predictions = rf_model.predict(X_val_rf)
probability_matrix = rf_model.predict_proba(X_val_rf)
if 1 not in rf_model.classes_:
    raise ValueError('Random Forest模型中不存在TC类别1。')
tc_class_index = np.where(rf_model.classes_ == 1)[0][0]
all_tc_probabilities = probability_matrix[:, tc_class_index]
all_predictions = (all_tc_probabilities >= CLASSIFICATION_THRESHOLD).astype(int)
all_targets = np.asarray(y_val, dtype=int)
all_predictions = np.asarray(all_predictions, dtype=int)
all_tc_probabilities = np.asarray(all_tc_probabilities, dtype=float)
accuracy = accuracy_score(all_targets, all_predictions)
precision = precision_score(all_targets, all_predictions, pos_label=1, zero_division=0)
recall = recall_score(all_targets, all_predictions, pos_label=1, zero_division=0)
f1 = f1_score(all_targets, all_predictions, pos_label=1, zero_division=0)
print('\n' + '=' * 60)
print('Random Forest验证集结果')
print('=' * 60)
print(f'Accuracy  = {accuracy:.4f}')
print(f'TC Precision = {precision:.4f}')
print(f'TC Recall    = {recall:.4f}')
print(f'TC F1-score  = {f1:.4f}')
print('=' * 60)
metrics_df = pd.DataFrame({'model': ['Random Forest'], 'sample': [SAMPLE_NAME], 'features': ['+'.join(feature_cols)], 'window': [f'{LEFT_WINDOW}+1+{RIGHT_WINDOW}'], 'n_estimators': [N_ESTIMATORS], 'accuracy': [accuracy], 'TC_precision': [precision], 'TC_recall': [recall], 'TC_F1': [f1]})
metrics_df.to_csv(METRICS_PATH, index=False, encoding='utf-8-sig')
print(f'评价指标已保存：{METRICS_PATH}')
print('\n--- Final Classification Report ---')
report_text = classification_report(all_targets, all_predictions, labels=[0, 1], target_names=['Non-TC', 'TC'], digits=4, zero_division=0)
print(report_text)
report_dict = classification_report(all_targets, all_predictions, labels=[0, 1], target_names=['Non-TC', 'TC'], output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(REPORT_PATH, encoding='utf-8-sig')
print(f'分类报告已保存：{REPORT_PATH}')
cm = confusion_matrix(all_targets, all_predictions, labels=[0, 1])
print('\n' + '=' * 50)
print(f"{'':<15}{'预测=Non-TC':<16}{'预测=TC':<12}")
print(f"{'真实=Non-TC':<15}{cm[0, 0]:<16}{cm[0, 1]:<12}")
print(f"{'真实=TC':<15}{cm[1, 0]:<16}{cm[1, 1]:<12}")
print('=' * 50)
if not len(val_dates) == len(all_targets) == len(all_predictions) == len(all_tc_probabilities):
    raise ValueError(f'日期、真实标签、预测类别和预测概率长度不一致：dates={len(val_dates)}, targets={len(all_targets)}, predictions={len(all_predictions)}, probabilities={len(all_tc_probabilities)}')
result_df = pd.DataFrame({'date': val_dates, 'true_label': all_targets, 'tc_probability': all_tc_probabilities, 'predicted_label': all_predictions})
result_df['true_class'] = np.where(result_df['true_label'] == 1, 'TC', 'Non-TC')
result_df['predicted_class'] = np.where(result_df['predicted_label'] == 1, 'TC', 'Non-TC')
result_df.to_csv(RESULT_CSV_PATH, index=False, encoding='utf-8-sig')
result_df.to_excel(RESULT_EXCEL_PATH, index=False)
print('\n逐日验证结果已保存：')
print(RESULT_CSV_PATH)
print(RESULT_EXCEL_PATH)
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
    ax.set_title(f'{sample_name} - Random Forest', fontsize=14)
    ax.grid(axis='y', color=grid_color, linestyle=':', linewidth=0.8, alpha=0.8, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend_handles = [Line2D([0], [0], color=probability_color, linewidth=1.7, label='Random Forest predicted TC probability'), Line2D([0], [0], color=threshold_color, linestyle='--', linewidth=1.4, label=f'Classification threshold = {threshold:.1f}'), Patch(facecolor=tc_shading_color, alpha=0.3, label='Observed TC-affected day')]
    ax.legend(handles=legend_handles, frameon=False, loc='upper right', fontsize=10)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    pass
    plt.close(fig)
plot_daily_tc_probability(prediction_df=result_df, sample_name=SAMPLE_NAME, threshold=CLASSIFICATION_THRESHOLD, output_path=PROB_FIG_PATH)
print(f'日尺度概率图已保存：{PROB_FIG_PATH}')
print('\n正在计算Random Forest特征重要性...')
importances = rf_model.feature_importances_
if len(importances) != len(feature_names):
    raise RuntimeError('特征重要性数量与特征名称数量不一致。')
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
importance_df['relative_importance_%'] = importance_df['importance'] / importance_df['importance'].sum() * 100
importance_df.to_csv(IMPORTANCE_PATH, index=False, encoding='utf-8-sig')
print(f'特征重要性已保存：{IMPORTANCE_PATH}')
print('\nTop 20 Random Forest特征：')
print(importance_df.head(20))
top_n = min(20, len(importance_df))
top_importance = importance_df.head(top_n).sort_values('importance', ascending=True)
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(top_importance['feature'], top_importance['importance'])
ax.set_xlabel('Random Forest feature importance', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title(f'{SAMPLE_NAME} - Random Forest', fontsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(IMPORTANCE_FIG_PATH, dpi=600, bbox_inches='tight')
pass
plt.close(fig)
print(f'特征重要性图已保存：{IMPORTANCE_FIG_PATH}')
print('\n' + '=' * 70)
print('Random Forest全部结果已生成：')
print(f'1. Random Forest模型：{MODEL_PATH}')
print(f'2. 54个输入特征名称：{FEATURE_PATH}')
print(f'3. 评价指标：{METRICS_PATH}')
print(f'4. 分类报告：{REPORT_PATH}')
print(f'5. 逐日结果CSV：{RESULT_CSV_PATH}')
print(f'6. 逐日结果Excel：{RESULT_EXCEL_PATH}')
print(f'7. 混淆矩阵：{CM_FIG_PATH}')
print(f'8. 日尺度TC概率图：{PROB_FIG_PATH}')
print(f'9. 特征重要性：{IMPORTANCE_PATH}')
print(f'10. 特征重要性图：{IMPORTANCE_FIG_PATH}')
print('=' * 70)
print('\n最终Random Forest结果：')
print(f'Accuracy     : {accuracy:.4f}')
print(f'TC Precision : {precision:.4f}')
print(f'TC Recall    : {recall:.4f}')
print(f'TC F1-score  : {f1:.4f}')
print('\n程序运行结束。')
