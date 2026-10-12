"""Extract potential disturbances and compare monthly patterns with manual references."""
from project_config import *
import calendar
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import pearsonr
INPUT_FILE = RUN_DIR / 'predict' / 'a114_daily_probability.xlsx'
OUTPUT_DIR = STAGE_OUTPUT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATE_COL = 'date'
PROB_COL = 'tc_probability'
REFERENCE_LABEL_COL = 'reference_label'
DAILY_OUTPUT_PATH = OUTPUT_DIR / 'a114_daily_event_assignment.xlsx'
PRED_EVENT_PATH = OUTPUT_DIR / 'a114_predicted_event_list.xlsx'
REF_EVENT_PATH = OUTPUT_DIR / 'a114_reference_event_list.xlsx'
MONTHLY_OUTPUT_PATH = OUTPUT_DIR / 'a114_monthly_event_counts.xlsx'
SEASONAL_OUTPUT_PATH = OUTPUT_DIR / 'a114_seasonal_event_summary.xlsx'
ANNUAL_OUTPUT_PATH = OUTPUT_DIR / 'a114_annual_event_counts.xlsx'
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / 'a114_event_analysis_summary.xlsx'
SENSITIVITY_OUTPUT_PATH = OUTPUT_DIR / 'a114_parameter_sensitivity.xlsx'
MONTHLY_FIG_PATH = OUTPUT_DIR / 'a114_monthly_event_comparison.png'
MONTHLY_FIG_PDF_PATH = OUTPUT_DIR / 'a114_monthly_event_comparison.pdf'
SEASON_FIG_PATH = OUTPUT_DIR / 'a114_seasonal_event_proportion.png'
SEASON_FIG_PDF_PATH = OUTPUT_DIR / 'a114_seasonal_event_proportion.pdf'
SMOOTH_WINDOW = CFG['smooth_window']
PROBABILITY_THRESHOLD = CFG['event_threshold']
MIN_EVENT_DAYS = CFG['min_event_days']
MAX_GAP_DAYS = CFG['max_gap_days']
SPLIT_MODE = 'peak'
MAX_EVENT_DAYS = CFG['max_event_days']
PEAK_MIN_DISTANCE = CFG['peak_min_distance']
PEAK_PROMINENCE = CFG['peak_prominence']
PREDICTED_EVENT_ASSIGNMENT = 'peak'
REFERENCE_EVENT_ASSIGNMENT = 'center'
TC_SEASON_MONTHS = [6, 7, 8, 9, 10, 11]
WINTER_SPRING_MONTHS = [1, 2, 3, 4, 5, 12]
RUN_SENSITIVITY = CFG['run_sensitivity']
SENSITIVITY_SMOOTH_WINDOWS = [1, 3, 5]
SENSITIVITY_THRESHOLDS = [0.45, 0.5, 0.55]
SENSITIVITY_SPLIT_MODES = ['none', 'peak', 'balanced']
MODEL_COLOR = '#628fc2'
REFERENCE_COLOR = '#f7a05f'
TC_SEASON_SHADE = '#cfe0ee'
GRID_COLOR = '#D9D9D9'
TEXT_COLOR = '#222222'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9.5, 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})

def safe_pearson(x, y):
    """Safe pearson."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return (np.nan, np.nan)
    if np.std(x) == 0 or np.std(y) == 0:
        return (np.nan, np.nan)
    pearson_r, pearson_p = pearsonr(x, y)
    return (float(pearson_r), float(pearson_p))

def get_date_segments(dates):
    """Assign a new segment whenever consecutive dates differ by anything other than one day."""
    dates = pd.to_datetime(dates).reset_index(drop=True)
    day_gap = dates.diff().dt.days
    new_segment = day_gap.isna() | (day_gap != 1)
    return new_segment.cumsum().to_numpy(dtype=int)

def validate_virtual_calendar(year_day_counts):
    """Check indexing-calendar year lengths; fossil indexing dates are not historical CE dates."""
    unexpected_years = {}
    for year, actual_count in year_day_counts.items():
        expected_count = 366 if calendar.isleap(int(year)) else 365
        if int(actual_count) != expected_count:
            unexpected_years[int(year)] = {'actual': int(actual_count), 'expected': int(expected_count)}
    if unexpected_years:
        warnings.warn(f'以下虚拟年份的记录数与公历预期不一致：{unexpected_years}。请检查是否存在缺失日或重复日。')
    else:
        print('各虚拟年份记录数均与连续公历日期一致。')

def validate_and_prepare_data(input_file):
    """Validate and prepare data."""
    if not input_file.exists():
        raise FileNotFoundError(f'找不到输入文件：{input_file}')
    df = pd.read_excel(input_file)
    required_cols = [DATE_COL, PROB_COL, REFERENCE_LABEL_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f'输入文件缺少必要列：{missing_cols}\n当前列名：{list(df.columns)}')
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    if df[DATE_COL].isna().any():
        bad_rows = df.index[df[DATE_COL].isna()].tolist()
        raise ValueError(f'日期列存在无法解析的值，行号示例：{bad_rows[:10]}')
    df[PROB_COL] = pd.to_numeric(df[PROB_COL], errors='coerce')
    if df[PROB_COL].isna().any():
        raise ValueError('预测概率列存在缺失值或非数值。')
    if ((df[PROB_COL] < 0) | (df[PROB_COL] > 1)).any():
        raise ValueError('预测概率必须位于[0, 1]范围内。')
    df[REFERENCE_LABEL_COL] = pd.to_numeric(df[REFERENCE_LABEL_COL], errors='coerce')
    if df[REFERENCE_LABEL_COL].isna().any():
        raise ValueError('人工参考标签列存在缺失值或非数值。')
    unique_reference_labels = set(df[REFERENCE_LABEL_COL].astype(int).unique().tolist())
    if not unique_reference_labels.issubset({0, 1}):
        raise ValueError(f'人工参考标签只能为0或1，当前值为：{sorted(unique_reference_labels)}')
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    duplicated_dates = df[DATE_COL].duplicated(keep=False)
    if duplicated_dates.any():
        duplicate_examples = df.loc[duplicated_dates, DATE_COL].dt.strftime('%Y-%m-%d').head(10).tolist()
        raise ValueError(f'日期列存在重复值，示例：{duplicate_examples}')
    df['segment_id'] = get_date_segments(df[DATE_COL])
    if 'relative_year' not in df.columns:
        first_year = int(df[DATE_COL].dt.year.min())
        df['relative_year'] = (df[DATE_COL].dt.year - first_year + 1).astype(int)
    else:
        df['relative_year'] = pd.to_numeric(df['relative_year'], errors='raise').astype(int)
    df['month'] = df[DATE_COL].dt.month.astype(int)
    gap_count = int((df[DATE_COL].diff().dt.days.fillna(1) != 1).sum())
    year_day_counts = df.groupby(df[DATE_COL].dt.year).size().rename('record_count')
    print('=' * 70)
    print(f'输入文件：{input_file}')
    print(f'数据总行数：{len(df)}')
    print(f'日期范围：{df[DATE_COL].min().date()} 至 {df[DATE_COL].max().date()}')
    print(f'日期断点数量：{gap_count}')
    print('\n各虚拟年份记录数：')
    print(year_day_counts.to_string())
    validate_virtual_calendar(year_day_counts)
    print('说明：上述日期仅作为连续时间索引，论文中应使用相对年份，不要将虚拟年份解释为A114的真实公元年份。')
    print('=' * 70)
    return df

def smooth_probability_by_segment(df, window):
    """Smooth probability by segment."""
    if window < 1:
        raise ValueError('平滑窗口必须大于等于1。')
    result = df.copy()
    if window == 1:
        result['smooth_probability'] = result[PROB_COL].astype(float)
    else:
        result['smooth_probability'] = result.groupby('segment_id', group_keys=False)[PROB_COL].transform(lambda series: series.rolling(window=window, center=True, min_periods=1).mean())
    return result

def extract_binary_events(df, binary_col, min_event_days=2, max_gap_days=0):
    """Retain runs meeting the minimum duration, then merge short gaps within each continuous date segment."""
    all_events = []
    for _, segment_df in df.groupby('segment_id', sort=True):
        segment_indices = segment_df.index.to_numpy()
        binary = segment_df[binary_col].to_numpy(dtype=int)
        raw_events = []
        start_local = None
        for local_i, value in enumerate(binary):
            if value == 1 and start_local is None:
                start_local = local_i
            elif value == 0 and start_local is not None:
                end_local = local_i - 1
                if end_local - start_local + 1 >= min_event_days:
                    raw_events.append((int(segment_indices[start_local]), int(segment_indices[end_local])))
                start_local = None
        if start_local is not None:
            end_local = len(binary) - 1
            if end_local - start_local + 1 >= min_event_days:
                raw_events.append((int(segment_indices[start_local]), int(segment_indices[end_local])))
        if not raw_events:
            continue
        merged_events = [raw_events[0]]
        for current_start, current_end in raw_events[1:]:
            last_start, last_end = merged_events[-1]
            gap_days = (df.loc[current_start, DATE_COL] - df.loc[last_end, DATE_COL]).days - 1
            if gap_days <= max_gap_days:
                merged_events[-1] = (last_start, current_end)
            else:
                merged_events.append((current_start, current_end))
        all_events.extend(merged_events)
    return all_events

def balanced_split_event(start_idx, end_idx, max_event_days):
    """Split a long candidate into approximately equal intervals no longer than max_event_days."""
    event_length = end_idx - start_idx + 1
    if max_event_days is None or event_length <= max_event_days:
        return [(start_idx, end_idx)]
    segment_number = int(np.ceil(event_length / max_event_days))
    base_length = event_length // segment_number
    remainder = event_length % segment_number
    segment_lengths = [base_length + (1 if i < remainder else 0) for i in range(segment_number)]
    segments = []
    current_start = start_idx
    for segment_length in segment_lengths:
        current_end = current_start + segment_length - 1
        segments.append((current_start, current_end))
        current_start = current_end + 1
    return segments

def peak_split_event(df, start_idx, end_idx, threshold, max_event_days, min_peak_distance, peak_prominence, min_event_days):
    """Partition a long candidate using probability peaks and intervening valleys."""
    event_length = end_idx - start_idx + 1
    if max_event_days is None or event_length <= max_event_days:
        return ([(start_idx, end_idx)], 'not_split')
    probability = df.loc[start_idx:end_idx, 'smooth_probability'].to_numpy(dtype=float)
    peaks, _ = find_peaks(probability, height=threshold, distance=min_peak_distance, prominence=peak_prominence)
    endpoint_candidates = []
    if len(probability) >= 2:
        if probability[0] >= threshold and probability[0] > probability[1]:
            endpoint_candidates.append(0)
        if probability[-1] >= threshold and probability[-1] > probability[-2]:
            endpoint_candidates.append(len(probability) - 1)
    if endpoint_candidates:
        peaks = np.unique(np.concatenate([peaks, np.asarray(endpoint_candidates, dtype=int)]))
    if len(peaks) < 2:
        return (balanced_split_event(start_idx, end_idx, max_event_days), 'balanced_fallback')
    peaks = np.sort(peaks)
    boundary_local_indices = []
    for left_peak, right_peak in zip(peaks[:-1], peaks[1:]):
        if right_peak <= left_peak + 1:
            continue
        valley_relative = int(np.argmin(probability[left_peak:right_peak + 1]))
        boundary_local_indices.append(left_peak + valley_relative)
    if not boundary_local_indices:
        return (balanced_split_event(start_idx, end_idx, max_event_days), 'balanced_fallback')
    candidate_segments = []
    current_start = start_idx
    for boundary_local in boundary_local_indices:
        boundary_global = start_idx + boundary_local
        if boundary_global >= current_start:
            candidate_segments.append((current_start, boundary_global))
            current_start = boundary_global + 1
    if current_start <= end_idx:
        candidate_segments.append((current_start, end_idx))
    if any((segment_end - segment_start + 1 < min_event_days for segment_start, segment_end in candidate_segments)):
        return (balanced_split_event(start_idx, end_idx, max_event_days), 'balanced_fallback')
    final_segments = []
    for segment_start, segment_end in candidate_segments:
        final_segments.extend(balanced_split_event(segment_start, segment_end, max_event_days))
    return (final_segments, 'peak')

def split_candidate_events(df, candidate_events, split_mode, threshold, max_event_days, min_peak_distance, peak_prominence, min_event_days):
    """Split candidate events."""
    if split_mode not in {'none', 'peak', 'balanced'}:
        raise ValueError("SPLIT_MODE只能是'none'、'peak'或'balanced'。")
    final_events = []
    split_records = []
    for candidate_id, (start_idx, end_idx) in enumerate(candidate_events, start=1):
        if split_mode == 'none':
            segments = [(start_idx, end_idx)]
            method = 'none'
        elif split_mode == 'balanced':
            segments = balanced_split_event(start_idx, end_idx, max_event_days)
            method = 'balanced' if len(segments) > 1 else 'not_split'
        else:
            segments, method = peak_split_event(df=df, start_idx=start_idx, end_idx=end_idx, threshold=threshold, max_event_days=max_event_days, min_peak_distance=min_peak_distance, peak_prominence=peak_prominence, min_event_days=min_event_days)
        for segment in segments:
            final_events.append(segment)
            split_records.append({'candidate_event_id': candidate_id, 'split_method': method})
    return (final_events, split_records)

def build_event_dataframe(df, index_events, event_type, split_records=None, assignment='center', probability_col=None):
    """Build event dataframe."""
    records = []
    if split_records is None:
        split_records = [{'candidate_event_id': event_id, 'split_method': 'reference'} for event_id in range(1, len(index_events) + 1)]
    for event_id, ((start_idx, end_idx), split_info) in enumerate(zip(index_events, split_records), start=1):
        start_date = pd.Timestamp(df.loc[start_idx, DATE_COL])
        end_date = pd.Timestamp(df.loc[end_idx, DATE_COL])
        center_idx = int((start_idx + end_idx) // 2)
        center_date = pd.Timestamp(df.loc[center_idx, DATE_COL])
        if probability_col is not None:
            event_probability = df.loc[start_idx:end_idx, probability_col]
            peak_idx = int(event_probability.idxmax())
            peak_date = pd.Timestamp(df.loc[peak_idx, DATE_COL])
            peak_probability = float(df.loc[peak_idx, probability_col])
            mean_probability = float(event_probability.mean())
        else:
            peak_idx = center_idx
            peak_date = center_date
            peak_probability = np.nan
            mean_probability = np.nan
        if assignment == 'start':
            assignment_idx = start_idx
            assignment_date = start_date
        elif assignment == 'peak' and probability_col is not None:
            assignment_idx = peak_idx
            assignment_date = peak_date
        else:
            assignment_idx = center_idx
            assignment_date = center_date
        duration_rows = end_idx - start_idx + 1
        duration_calendar_days = (end_date - start_date).days + 1
        records.append({'event_type': event_type, 'event_id': event_id, 'candidate_event_id': split_info['candidate_event_id'], 'split_method': split_info['split_method'], 'start_idx': start_idx, 'end_idx': end_idx, 'start_date': start_date, 'end_date': end_date, 'duration_rows': duration_rows, 'duration_calendar_days': duration_calendar_days, 'center_date': center_date, 'peak_date': peak_date, 'peak_probability': peak_probability, 'mean_probability': mean_probability, 'assignment_date': assignment_date, 'relative_year': int(df.loc[assignment_idx, 'relative_year']), 'month': int(assignment_date.month)})
    return pd.DataFrame(records)

def count_monthly(event_df):
    """Count monthly."""
    if event_df.empty:
        return pd.Series(0, index=range(1, 13), name='event_count', dtype=int)
    return event_df.groupby('month').size().reindex(range(1, 13), fill_value=0).astype(int).rename('event_count')

def count_annual(event_df, relative_years):
    """Count annual."""
    if event_df.empty:
        return pd.Series(0, index=relative_years, name='event_count', dtype=int)
    return event_df.groupby('relative_year').size().reindex(relative_years, fill_value=0).astype(int).rename('event_count')

def run_prediction_event_extraction(source_df, smooth_window, threshold, min_event_days, max_gap_days, split_mode, max_event_days):
    """Run prediction event extraction."""
    work_df = smooth_probability_by_segment(source_df, window=smooth_window)
    work_df['predicted_positive_day'] = (work_df['smooth_probability'] >= threshold).astype(int)
    candidate_events = extract_binary_events(work_df, binary_col='predicted_positive_day', min_event_days=min_event_days, max_gap_days=max_gap_days)
    final_events, split_records = split_candidate_events(df=work_df, candidate_events=candidate_events, split_mode=split_mode, threshold=threshold, max_event_days=max_event_days, min_peak_distance=PEAK_MIN_DISTANCE, peak_prominence=PEAK_PROMINENCE, min_event_days=min_event_days)
    event_df = build_event_dataframe(df=work_df, index_events=final_events, event_type='model_reconstruction', split_records=split_records, assignment=PREDICTED_EVENT_ASSIGNMENT, probability_col='smooth_probability')
    return (work_df, candidate_events, final_events, event_df)

def calculate_comparison_metrics(predicted_monthly, reference_monthly):
    """Calculate comparison metrics."""
    monthly_pearson_r, monthly_pearson_p = safe_pearson(predicted_monthly.values, reference_monthly.values)
    tc_season_predicted = predicted_monthly.loc[TC_SEASON_MONTHS]
    tc_season_reference = reference_monthly.loc[TC_SEASON_MONTHS]
    tc_season_pearson_r, tc_season_pearson_p = safe_pearson(tc_season_predicted.values, tc_season_reference.values)
    predicted_total = int(predicted_monthly.sum())
    reference_total = int(reference_monthly.sum())
    predicted_reference_count_ratio = predicted_total / reference_total if reference_total > 0 else np.nan
    predicted_tc_season_count = int(predicted_monthly.loc[TC_SEASON_MONTHS].sum())
    predicted_winter_spring_count = int(predicted_monthly.loc[WINTER_SPRING_MONTHS].sum())
    reference_tc_season_count = int(reference_monthly.loc[TC_SEASON_MONTHS].sum())
    reference_winter_spring_count = int(reference_monthly.loc[WINTER_SPRING_MONTHS].sum())
    predicted_tc_season_proportion = predicted_tc_season_count / predicted_total if predicted_total > 0 else np.nan
    predicted_winter_spring_proportion = predicted_winter_spring_count / predicted_total if predicted_total > 0 else np.nan
    reference_tc_season_proportion = reference_tc_season_count / reference_total if reference_total > 0 else np.nan
    reference_winter_spring_proportion = reference_winter_spring_count / reference_total if reference_total > 0 else np.nan
    predicted_seasonal_ratio = predicted_tc_season_count / predicted_winter_spring_count if predicted_winter_spring_count > 0 else np.nan
    reference_seasonal_ratio = reference_tc_season_count / reference_winter_spring_count if reference_winter_spring_count > 0 else np.nan
    monthly_difference = predicted_monthly.values - reference_monthly.values
    monthly_mae = float(np.mean(np.abs(monthly_difference)))
    monthly_rmse = float(np.sqrt(np.mean(monthly_difference ** 2)))
    tc_season_difference = tc_season_predicted.values - tc_season_reference.values
    tc_season_mae = float(np.mean(np.abs(tc_season_difference)))
    tc_season_rmse = float(np.sqrt(np.mean(tc_season_difference ** 2)))
    return {'predicted_event_count': predicted_total, 'reference_event_count': reference_total, 'predicted_reference_count_ratio': predicted_reference_count_ratio, 'predicted_tc_season_count': predicted_tc_season_count, 'predicted_winter_spring_count': predicted_winter_spring_count, 'reference_tc_season_count': reference_tc_season_count, 'reference_winter_spring_count': reference_winter_spring_count, 'predicted_tc_season_proportion': predicted_tc_season_proportion, 'predicted_winter_spring_proportion': predicted_winter_spring_proportion, 'reference_tc_season_proportion': reference_tc_season_proportion, 'reference_winter_spring_proportion': reference_winter_spring_proportion, 'predicted_tc_season_winter_spring_ratio': predicted_seasonal_ratio, 'reference_tc_season_winter_spring_ratio': reference_seasonal_ratio, 'monthly_pearson_r': monthly_pearson_r, 'monthly_pearson_p': monthly_pearson_p, 'tc_season_pearson_r': tc_season_pearson_r, 'tc_season_pearson_p': tc_season_pearson_p, 'monthly_mae': monthly_mae, 'monthly_rmse': monthly_rmse, 'tc_season_mae': tc_season_mae, 'tc_season_rmse': tc_season_rmse}

def build_monthly_comparison_df(predicted_monthly, reference_monthly):
    """Build monthly comparison df."""
    monthly_df = pd.DataFrame({'month': range(1, 13), 'predicted_event_count': predicted_monthly.values, 'reference_event_count': reference_monthly.values})
    monthly_df['seasonal_period'] = np.where(monthly_df['month'].isin(TC_SEASON_MONTHS), 'Jun-Nov', 'Dec-May')
    monthly_df['difference'] = monthly_df['predicted_event_count'] - monthly_df['reference_event_count']
    monthly_df['absolute_difference'] = monthly_df['difference'].abs()
    return monthly_df

def build_seasonal_summary_df(summary_metrics):
    """Build seasonal summary df."""
    return pd.DataFrame({'period': ['Dec-May', 'Jun-Nov'], 'model_event_count': [summary_metrics['predicted_winter_spring_count'], summary_metrics['predicted_tc_season_count']], 'model_proportion': [summary_metrics['predicted_winter_spring_proportion'], summary_metrics['predicted_tc_season_proportion']], 'reference_event_count': [summary_metrics['reference_winter_spring_count'], summary_metrics['reference_tc_season_count']], 'reference_proportion': [summary_metrics['reference_winter_spring_proportion'], summary_metrics['reference_tc_season_proportion']]})

def format_p_value(p_value):
    """Format p value."""
    if np.isnan(p_value):
        return 'NA'
    if p_value < 0.001:
        return '< 0.001'
    return f'= {p_value:.3f}'

def plot_monthly_event_comparison(monthly_comparison_df, summary_metrics, png_path, pdf_path):
    """Plot monthly event comparison."""
    months = np.arange(1, 13)
    bar_width = 0.36
    fig, ax = plt.subplots(figsize=(9.4, 5.2), facecolor='white')
    ax.set_facecolor('white')
    ax.axvspan(5.5, 11.5, facecolor=TC_SEASON_SHADE, alpha=0.65, zorder=0)
    predicted_bars = ax.bar(months - bar_width / 2, monthly_comparison_df['predicted_event_count'], width=bar_width, color=MODEL_COLOR, edgecolor='black', linewidth=0.5, label='Model reconstruction', zorder=3)
    reference_bars = ax.bar(months + bar_width / 2, monthly_comparison_df['reference_event_count'], width=bar_width, color=REFERENCE_COLOR, edgecolor='black', linewidth=0.5, label='Manual reference', zorder=3)
    ax.bar_label(predicted_bars, padding=2, fontsize=8.2)
    ax.bar_label(reference_bars, padding=2, fontsize=8.2)
    ax.set_xticks(months)
    ax.set_xlim(0.45, 12.55)
    ax.set_xlabel('Month')
    ax.set_ylabel('Event count')
    ax.grid(axis='y', linestyle='--', linewidth=0.7, color=GRID_COLOR, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    current_top = max(monthly_comparison_df['predicted_event_count'].max(), monthly_comparison_df['reference_event_count'].max())
    ax.set_ylim(0, current_top * 1.36)
    season_r = summary_metrics['tc_season_pearson_r']
    season_p = summary_metrics['tc_season_pearson_p']
    correlation_text = f'Jun-Nov monthly frequency\nPearson r = {season_r:.3f}, p {format_p_value(season_p)}'
    ax.text(0.018, 0.965, correlation_text, transform=ax.transAxes, ha='left', va='top', fontsize=9.3, bbox={'boxstyle': 'round,pad=0.35', 'facecolor': 'white', 'edgecolor': '#BFBFBF', 'linewidth': 0.7, 'alpha': 0.95})
    ax.legend(frameon=False, loc='upper right', ncol=1, columnspacing=1.2, handlelength=1.4)
    fig.tight_layout()
    fig.savefig(png_path, dpi=600, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

def plot_seasonal_event_proportion(seasonal_summary_df, png_path, pdf_path):
    """Plot seasonal event proportion."""
    x = np.arange(len(seasonal_summary_df))
    bar_width = 0.36
    model_proportions = seasonal_summary_df['model_proportion'].to_numpy(dtype=float) * 100
    reference_proportions = seasonal_summary_df['reference_proportion'].to_numpy(dtype=float) * 100
    model_counts = seasonal_summary_df['model_event_count'].to_numpy(dtype=int)
    reference_counts = seasonal_summary_df['reference_event_count'].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(6.8, 5.0), facecolor='white')
    ax.set_facecolor('white')
    model_bars = ax.bar(x - bar_width / 2, model_proportions, width=bar_width, color=MODEL_COLOR, edgecolor='black', linewidth=0.5, label='Model reconstruction', zorder=3)
    reference_bars = ax.bar(x + bar_width / 2, reference_proportions, width=bar_width, color=REFERENCE_COLOR, edgecolor='black', linewidth=0.5, label='Manual reference', zorder=3)
    model_labels = [f'n={count}' for count in model_counts]
    reference_labels = [f'n={count}' for count in reference_counts]
    ax.bar_label(model_bars, labels=model_labels, padding=3, fontsize=9)
    ax.bar_label(reference_bars, labels=reference_labels, padding=3, fontsize=9)
    ax.axhline(50, color='#666666', linestyle='--', linewidth=0.9, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(seasonal_summary_df['period'])
    ax.set_xlabel('Seasonal period')
    ax.set_ylabel('Proportion of events (%)')
    y_top = max(model_proportions.max(), reference_proportions.max()) * 1.25
    ax.set_ylim(0, max(72, y_top))
    ax.grid(axis='y', linestyle='--', linewidth=0.7, color=GRID_COLOR, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='upper left', ncol=1)
    fig.tight_layout()
    fig.savefig(png_path, dpi=600, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

def print_main_results(candidate_events, predicted_event_df, reference_event_df, predicted_monthly, reference_monthly, predicted_annual, reference_annual, summary_metrics):
    """Print main results."""
    print('\n' + '=' * 70)
    print('主分析参数')
    print(f'SMOOTH_WINDOW                 = {SMOOTH_WINDOW}')
    print(f'PROBABILITY_THRESHOLD         = {PROBABILITY_THRESHOLD}')
    print(f'MIN_EVENT_DAYS                = {MIN_EVENT_DAYS}')
    print(f'MAX_GAP_DAYS                  = {MAX_GAP_DAYS}')
    print(f'SPLIT_MODE                    = {SPLIT_MODE}')
    print(f'MAX_EVENT_DAYS                = {MAX_EVENT_DAYS}')
    print(f'PREDICTED_EVENT_ASSIGNMENT    = {PREDICTED_EVENT_ASSIGNMENT}')
    print(f'REFERENCE_EVENT_ASSIGNMENT    = {REFERENCE_EVENT_ASSIGNMENT}')
    print('\n' + '=' * 70)
    print(f'切分前候选事件数：{len(candidate_events)}')
    print(f'最终模型事件数：{len(predicted_event_df)}')
    print(f'人工参考事件数：{len(reference_event_df)}')
    print('\n预测事件月频次：')
    print(predicted_monthly.to_string())
    print('\n人工参考事件月频次：')
    print(reference_monthly.to_string())
    print('\n预测事件相对年频次（辅助结果）：')
    print(predicted_annual.to_string())
    print('\n人工参考事件相对年频次（辅助结果）：')
    print(reference_annual.to_string())
    print('\n' + '=' * 70)
    print('全年月频次相关（辅助结果）')
    print(f"Pearson r = {summary_metrics['monthly_pearson_r']:.3f}, p = {summary_metrics['monthly_pearson_p']:.3e}")
    print('\n6—11月月频次相关（主要结果）')
    print(f"Pearson r = {summary_metrics['tc_season_pearson_r']:.3f}, p = {summary_metrics['tc_season_pearson_p']:.3e}")
    if summary_metrics['tc_season_pearson_p'] < 0.05:
        print('解释：6—11月月频次呈显著正相关。')
    else:
        print('解释：6—11月月频次表现出较强正相关趋势，但未达到p<0.05的显著性水平。')
    print('\n主要TC季与冬春季事件分布')
    print(f"模型重建：6—11月 {summary_metrics['predicted_tc_season_count']} 个，占 {summary_metrics['predicted_tc_season_proportion']:.1%}；冬春季 {summary_metrics['predicted_winter_spring_count']} 个，占 {summary_metrics['predicted_winter_spring_proportion']:.1%}")
    print(f"人工参考：6—11月 {summary_metrics['reference_tc_season_count']} 个，占 {summary_metrics['reference_tc_season_proportion']:.1%}；冬春季 {summary_metrics['reference_winter_spring_count']} 个，占 {summary_metrics['reference_winter_spring_proportion']:.1%}")
    print(f"模型6—11月/冬春季事件数比 = {summary_metrics['predicted_tc_season_winter_spring_ratio']:.3f}")
    print(f"人工6—11月/冬春季事件数比 = {summary_metrics['reference_tc_season_winter_spring_ratio']:.3f}")
    print('\n月频次误差')
    print(f"全年 MAE = {summary_metrics['monthly_mae']:.3f}, RMSE = {summary_metrics['monthly_rmse']:.3f}")
    print(f"6—11月 MAE = {summary_metrics['tc_season_mae']:.3f}, RMSE = {summary_metrics['tc_season_rmse']:.3f}")
    print('=' * 70)

def save_main_outputs(analysis_df, predicted_event_df, reference_event_df, monthly_comparison_df, seasonal_summary_df, annual_comparison_df, summary_metrics):
    """Save main outputs."""
    parameter_df = pd.DataFrame({'parameter': ['smooth_window', 'probability_threshold', 'min_event_days', 'max_gap_days', 'split_mode', 'max_event_days', 'peak_min_distance', 'peak_prominence', 'predicted_event_assignment', 'reference_event_assignment'], 'value': [SMOOTH_WINDOW, PROBABILITY_THRESHOLD, MIN_EVENT_DAYS, MAX_GAP_DAYS, SPLIT_MODE, MAX_EVENT_DAYS, PEAK_MIN_DISTANCE, PEAK_PROMINENCE, PREDICTED_EVENT_ASSIGNMENT, REFERENCE_EVENT_ASSIGNMENT]})
    summary_df = pd.DataFrame({'metric': list(summary_metrics.keys()), 'value': list(summary_metrics.values())})
    analysis_df.to_excel(DAILY_OUTPUT_PATH, index=False)
    predicted_event_df.to_excel(PRED_EVENT_PATH, index=False)
    reference_event_df.to_excel(REF_EVENT_PATH, index=False)
    monthly_comparison_df.to_excel(MONTHLY_OUTPUT_PATH, index=False)
    seasonal_summary_df.to_excel(SEASONAL_OUTPUT_PATH, index=False)
    annual_comparison_df.to_excel(ANNUAL_OUTPUT_PATH, index=False)
    with pd.ExcelWriter(SUMMARY_OUTPUT_PATH) as writer:
        parameter_df.to_excel(writer, sheet_name='parameters', index=False)
        summary_df.to_excel(writer, sheet_name='metrics', index=False)
        monthly_comparison_df.to_excel(writer, sheet_name='monthly_counts', index=False)
        seasonal_summary_df.to_excel(writer, sheet_name='seasonal_summary', index=False)
        annual_comparison_df.to_excel(writer, sheet_name='annual_counts_auxiliary', index=False)

def run_sensitivity_analysis(df, reference_monthly):
    """Run sensitivity analysis."""
    sensitivity_records = []
    for smooth_window in SENSITIVITY_SMOOTH_WINDOWS:
        for threshold in SENSITIVITY_THRESHOLDS:
            for split_mode in SENSITIVITY_SPLIT_MODES:
                _, sensitivity_candidates, _, sensitivity_event_df = run_prediction_event_extraction(source_df=df, smooth_window=smooth_window, threshold=threshold, min_event_days=MIN_EVENT_DAYS, max_gap_days=MAX_GAP_DAYS, split_mode=split_mode, max_event_days=MAX_EVENT_DAYS)
                sensitivity_monthly = count_monthly(sensitivity_event_df)
                metrics = calculate_comparison_metrics(predicted_monthly=sensitivity_monthly, reference_monthly=reference_monthly)
                sensitivity_records.append({'smooth_window': smooth_window, 'threshold': threshold, 'min_event_days': MIN_EVENT_DAYS, 'max_gap_days': MAX_GAP_DAYS, 'split_mode': split_mode, 'max_event_days': MAX_EVENT_DAYS if split_mode != 'none' else np.nan, 'candidate_event_count': len(sensitivity_candidates), 'final_event_count': len(sensitivity_event_df), **metrics})
    sensitivity_df = pd.DataFrame(sensitivity_records).sort_values(by=['smooth_window', 'threshold', 'split_mode']).reset_index(drop=True)
    sensitivity_df.to_excel(SENSITIVITY_OUTPUT_PATH, index=False)
    return sensitivity_df

def main():
    """Main."""
    df = validate_and_prepare_data(INPUT_FILE)
    relative_years = list(range(int(df['relative_year'].min()), int(df['relative_year'].max()) + 1))
    df['reference_positive_day'] = (df[REFERENCE_LABEL_COL].astype(int) == 1).astype(int)
    reference_index_events = extract_binary_events(df, binary_col='reference_positive_day', min_event_days=1, max_gap_days=0)
    reference_event_df = build_event_dataframe(df=df, index_events=reference_index_events, event_type='manual_reference', assignment=REFERENCE_EVENT_ASSIGNMENT, probability_col=None)
    analysis_df, candidate_events, predicted_index_events, predicted_event_df = run_prediction_event_extraction(source_df=df, smooth_window=SMOOTH_WINDOW, threshold=PROBABILITY_THRESHOLD, min_event_days=MIN_EVENT_DAYS, max_gap_days=MAX_GAP_DAYS, split_mode=SPLIT_MODE, max_event_days=MAX_EVENT_DAYS)
    predicted_monthly = count_monthly(predicted_event_df)
    reference_monthly = count_monthly(reference_event_df)
    predicted_annual = count_annual(predicted_event_df, relative_years)
    reference_annual = count_annual(reference_event_df, relative_years)
    summary_metrics = calculate_comparison_metrics(predicted_monthly=predicted_monthly, reference_monthly=reference_monthly)
    monthly_comparison_df = build_monthly_comparison_df(predicted_monthly, reference_monthly)
    seasonal_summary_df = build_seasonal_summary_df(summary_metrics)
    annual_comparison_df = pd.DataFrame({'relative_year': relative_years, 'predicted_event_count': predicted_annual.values, 'reference_event_count': reference_annual.values})
    analysis_df['predicted_event_id'] = 0
    analysis_df['reference_event_id'] = 0
    for row_number, row in predicted_event_df.iterrows():
        analysis_df.loc[int(row['start_idx']):int(row['end_idx']), 'predicted_event_id'] = int(row_number) + 1
    for row_number, row in reference_event_df.iterrows():
        analysis_df.loc[int(row['start_idx']):int(row['end_idx']), 'reference_event_id'] = int(row_number) + 1
    print_main_results(candidate_events=candidate_events, predicted_event_df=predicted_event_df, reference_event_df=reference_event_df, predicted_monthly=predicted_monthly, reference_monthly=reference_monthly, predicted_annual=predicted_annual, reference_annual=reference_annual, summary_metrics=summary_metrics)
    save_main_outputs(analysis_df=analysis_df, predicted_event_df=predicted_event_df, reference_event_df=reference_event_df, monthly_comparison_df=monthly_comparison_df, seasonal_summary_df=seasonal_summary_df, annual_comparison_df=annual_comparison_df, summary_metrics=summary_metrics)
    if RUN_SENSITIVITY:
        run_sensitivity_analysis(df=df, reference_monthly=reference_monthly)
    plot_monthly_event_comparison(monthly_comparison_df=monthly_comparison_df, summary_metrics=summary_metrics, png_path=MONTHLY_FIG_PATH, pdf_path=MONTHLY_FIG_PDF_PATH)
    plot_seasonal_event_proportion(seasonal_summary_df=seasonal_summary_df, png_path=SEASON_FIG_PATH, pdf_path=SEASON_FIG_PDF_PATH)
    print('\n' + '=' * 70)
    print('分析完成，结果已保存：')
    print(f'1. 逐日事件标注：{DAILY_OUTPUT_PATH}')
    print(f'2. 模型事件列表：{PRED_EVENT_PATH}')
    print(f'3. 人工参考事件列表：{REF_EVENT_PATH}')
    print(f'4. 月频次对比：{MONTHLY_OUTPUT_PATH}')
    print(f'5. 季节事件汇总：{SEASONAL_OUTPUT_PATH}')
    print(f'6. 相对年度频次（辅助结果）：{ANNUAL_OUTPUT_PATH}')
    print(f'7. 参数与指标汇总：{SUMMARY_OUTPUT_PATH}')
    if RUN_SENSITIVITY:
        print(f'8. 参数敏感性分析：{SENSITIVITY_OUTPUT_PATH}')
    print(f'9. 月频次对比图PNG：{MONTHLY_FIG_PATH}')
    print(f'10. 月频次对比图PDF：{MONTHLY_FIG_PDF_PATH}')
    print(f'11. 季节事件占比图PNG：{SEASON_FIG_PATH}')
    print(f'12. 季节事件占比图PDF：{SEASON_FIG_PDF_PATH}')
    print('=' * 70)
if __name__ == '__main__':
    main()
