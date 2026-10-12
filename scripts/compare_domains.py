"""Compare fossil and modern feature distributions and source compatibility."""
from __future__ import annotations
from project_config import *
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist, pdist
FILE_PATHS: Dict[str, Path] = {s: PROCESSED_DIR / (s + '.xlsx') for s in ['A114', 'LHJ2', 'XB10']}
OUTPUT_DIR = STAGE_OUTPUT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_COLUMNS: List[str] = ['DGR', 'DGRV', 'TDI_A', 'TDI_V', 'TDI_F', 'TDI_E']
DATE_COLUMN = 'date'
LABEL_COLUMN_CANDIDATES = ['label', 'true_label', 'Label']
DRAW_OVERALL_KDE = True
DRAW_LABEL_BOXPLOT = True
CALCULATE_SOURCE_COMPATIBILITY = True
FIGSIZE = (14, 8)
DPI = 600
SAMPLE_COLORS = {'A114': '#4C78A8', 'LHJ2': '#F58518', 'XB10': '#54A24B'}
SUBFIGURE_LABELS = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
MODERN_SOURCE_SAMPLES = ['LHJ2', 'XB10']
MMD_MAX_SAMPLES = 2000
RANDOM_SEED = CFG['seed']

def configure_plot_style() -> None:
    """Configure plot style."""
    sns.set_theme(style='whitegrid', context='paper')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans'], 'axes.unicode_minus': False, 'axes.titlesize': 14, 'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11, 'savefig.dpi': DPI})

def read_table(path: Path) -> pd.DataFrame:
    """Read table."""
    if not path.exists():
        raise FileNotFoundError(f'文件不存在：{path}')
    suffix = path.suffix.lower()
    if suffix in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    if suffix == '.csv':
        return pd.read_csv(path)
    raise ValueError(f'不支持的文件格式：{path.suffix}')

def find_label_column(df: pd.DataFrame) -> str | None:
    """Find label column."""
    for column in LABEL_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    return None

def prepare_sample_data(sample_name: str, path: Path) -> Tuple[pd.DataFrame, str | None]:
    """Prepare sample data."""
    df = read_table(path)
    missing_features = [feature for feature in FEATURE_COLUMNS if feature not in df.columns]
    if missing_features:
        raise ValueError(f'{sample_name} 缺少特征列：{missing_features}\n当前列名：{list(df.columns)}')
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
        if df[DATE_COLUMN].isna().any():
            bad_rows = df.index[df[DATE_COLUMN].isna()].tolist()
            raise ValueError(f'{sample_name} 日期列存在无法解析值，行号示例：{bad_rows[:10]}')
        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    for feature in FEATURE_COLUMNS:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    missing_counts = df[FEATURE_COLUMNS].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        warnings.warn(f'{sample_name} 以下特征存在缺失值，分析时自动忽略：\n{missing_counts}')
    label_col = find_label_column(df)
    if label_col is not None:
        df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
    print('-' * 70)
    print(f'样品：{sample_name}')
    print(f'文件：{path}')
    print(f'总行数：{len(df)}')
    print('标签列：', label_col if label_col is not None else '未找到')
    if label_col is not None:
        label_counts = df[label_col].dropna().astype(int).value_counts().sort_index().to_dict()
        print(f'标签分布：{label_counts}')
    return (df, label_col)

def load_all_samples() -> Tuple[Dict[str, pd.DataFrame], Dict[str, str | None]]:
    """Load all samples."""
    data_dict: Dict[str, pd.DataFrame] = {}
    label_column_dict: Dict[str, str | None] = {}
    errors = []
    for sample_name, path in FILE_PATHS.items():
        try:
            df, label_col = prepare_sample_data(sample_name, path)
            data_dict[sample_name] = df
            label_column_dict[sample_name] = label_col
        except Exception as exc:
            errors.append(f'{sample_name}: {exc}')
    if errors:
        raise RuntimeError('部分样品读取失败：\n\n' + '\n\n'.join(errors))
    if len(data_dict) < 2:
        raise ValueError('有效样品不足2个，无法进行比较。')
    return (data_dict, label_column_dict)

def add_subfigure_label(ax: plt.Axes, label: str) -> None:
    """Add subfigure label."""
    ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontsize=15, fontweight='bold', va='top', ha='left')

def simplify_axes(ax: plt.Axes) -> None:
    """Simplify axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='both', linestyle='--', linewidth=0.6, alpha=0.35)

def create_axes() -> Tuple[plt.Figure, np.ndarray]:
    """Create axes."""
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE)
    return (fig, axes.flatten())

def plot_overall_kde(data_dict: Dict[str, pd.DataFrame]) -> None:
    """Plot overall kde."""
    sample_names = ['XB10', 'LHJ2', 'A114']
    sample_names = [sample for sample in sample_names if sample in data_dict]
    fig, axes = create_axes()
    legend_handles = []
    legend_labels = []
    for index, feature in enumerate(FEATURE_COLUMNS):
        ax = axes[index]
        for sample in sample_names:
            values = data_dict[sample][feature].dropna().to_numpy(dtype=float)
            if len(values) < 2 or np.isclose(np.nanstd(values), 0.0):
                warnings.warn(f'{sample} 的 {feature} 有效变化不足，跳过 KDE。')
                continue
            sns.kdeplot(x=values, ax=ax, label=sample, linewidth=2.2, fill=True, alpha=0.12, color=SAMPLE_COLORS[sample], common_norm=False, warn_singular=False)
        ax.set_title(feature)
        ax.set_xlabel('Feature value')
        ax.set_ylabel('Probability density')
        add_subfigure_label(ax, SUBFIGURE_LABELS[index])
        simplify_axes(ax)
        handles, labels = ax.get_legend_handles_labels()
        if handles and (not legend_handles):
            legend_handles = handles
            legend_labels = labels
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=len(legend_labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=2.0)
    output_path = OUTPUT_DIR / 'three_sample_feature_kde.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Fig. 3 KDE图已保存：{output_path}')

def plot_label_boxplots(data_dict: Dict[str, pd.DataFrame], label_column_dict: Dict[str, str | None]) -> None:
    """Plot label boxplots."""
    modern_samples = ['XB10', 'LHJ2']
    for sample_name in modern_samples:
        if sample_name not in data_dict:
            continue
        df = data_dict[sample_name]
        label_col = label_column_dict[sample_name]
        if label_col is None:
            print(f'{sample_name} 没有标签列，跳过分类箱线图。')
            continue
        plot_df = df[FEATURE_COLUMNS + [label_col]].copy()
        plot_df = plot_df[plot_df[label_col].isin([0, 1])].copy()
        if plot_df.empty:
            print(f'{sample_name} 没有有效0/1标签，跳过。')
            continue
        plot_df['Class'] = plot_df[label_col].map({0: 'Non-TC', 1: 'TC'})
        fig, axes = create_axes()
        for index, feature in enumerate(FEATURE_COLUMNS):
            ax = axes[index]
            sns.boxplot(data=plot_df, x='Class', y=feature, order=['Non-TC', 'TC'], hue='Class', palette={'Non-TC': '#9ECAE1', 'TC': '#F4A582'}, legend=False, showfliers=False, width=0.55, ax=ax)
            ax.set_title(feature)
            ax.set_xlabel('')
            ax.set_ylabel('Feature value')
            add_subfigure_label(ax, SUBFIGURE_LABELS[index])
            simplify_axes(ax)
        fig.tight_layout(pad=2.0)
        output_path = OUTPUT_DIR / f'{sample_name}_feature_boxplots_by_class.png'
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'{sample_name} TC/Non-TC箱线图已保存：{output_path}')

def fit_common_scaler(data_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.Series, pd.Series]:
    """Fit pooled modern-specimen means and sample standard deviations for compatibility analysis only."""
    modern_data = pd.concat([data_dict[sample][FEATURE_COLUMNS] for sample in MODERN_SOURCE_SAMPLES], axis=0, ignore_index=True)
    modern_data = modern_data.replace([np.inf, -np.inf], np.nan)
    feature_mean = modern_data.mean(axis=0)
    feature_std = modern_data.std(axis=0, ddof=1)
    feature_std = feature_std.replace(0, np.nan)
    return (feature_mean, feature_std)

def standardize_features(df: pd.DataFrame, feature_mean: pd.Series, feature_std: pd.Series) -> pd.DataFrame:
    """Standardize features."""
    standardized = (df[FEATURE_COLUMNS] - feature_mean) / feature_std
    standardized = standardized.replace([np.inf, -np.inf], np.nan)
    return standardized

def estimate_mmd_gamma(standardized_data: Dict[str, pd.DataFrame]) -> float:
    """Estimate mmd gamma."""
    modern_matrix = pd.concat([standardized_data[sample] for sample in MODERN_SOURCE_SAMPLES], axis=0, ignore_index=True)
    modern_matrix = modern_matrix.dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    if len(modern_matrix) > MMD_MAX_SAMPLES:
        index = rng.choice(len(modern_matrix), size=MMD_MAX_SAMPLES, replace=False)
        modern_matrix = modern_matrix[index]
    squared_distances = pdist(modern_matrix, metric='sqeuclidean')
    squared_distances = squared_distances[squared_distances > 0]
    if len(squared_distances) == 0:
        return 1.0
    median_squared_distance = np.median(squared_distances)
    if not np.isfinite(median_squared_distance) or median_squared_distance <= 0:
        return 1.0
    gamma = 1.0 / median_squared_distance
    return float(gamma)

def calculate_mmd_squared(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Calculate mmd squared."""
    rng = np.random.default_rng(RANDOM_SEED)
    if len(x) > MMD_MAX_SAMPLES:
        index = rng.choice(len(x), size=MMD_MAX_SAMPLES, replace=False)
        x = x[index]
    if len(y) > MMD_MAX_SAMPLES:
        index = rng.choice(len(y), size=MMD_MAX_SAMPLES, replace=False)
        y = y[index]
    k_xx = np.exp(-gamma * cdist(x, x, metric='sqeuclidean'))
    k_yy = np.exp(-gamma * cdist(y, y, metric='sqeuclidean'))
    k_xy = np.exp(-gamma * cdist(x, y, metric='sqeuclidean'))
    mmd_squared = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return float(max(mmd_squared, 0.0))

def calculate_source_compatibility(data_dict: Dict[str, pd.DataFrame]) -> None:
    """Calculate source compatibility."""
    required_samples = {'A114', 'LHJ2', 'XB10'}
    if not required_samples.issubset(data_dict):
        warnings.warn('缺少 A114、LHJ2 或 XB10，无法进行 source-domain compatibility analysis。')
        return
    feature_mean, feature_std = fit_common_scaler(data_dict)
    standardized_data = {}
    for sample_name, df in data_dict.items():
        standardized_data[sample_name] = standardize_features(df, feature_mean, feature_std)
    wasserstein_records = []
    for feature in FEATURE_COLUMNS:
        a114_values = standardized_data['A114'][feature].dropna().to_numpy(dtype=float)
        if len(a114_values) < 2:
            warnings.warn(f'A114 的 {feature} 有效数据不足。')
            continue
        for source in ['LHJ2', 'XB10']:
            source_values = standardized_data[source][feature].dropna().to_numpy(dtype=float)
            if len(source_values) < 2:
                warnings.warn(f'{source} 的 {feature} 有效数据不足。')
                continue
            distance = wasserstein_distance(a114_values, source_values)
            wasserstein_records.append({'feature': feature, 'source': source, 'wasserstein_distance': float(distance)})
    wasserstein_df = pd.DataFrame(wasserstein_records)
    if wasserstein_df.empty:
        raise RuntimeError('未能计算 Wasserstein distance。')
    feature_table = wasserstein_df.pivot(index='feature', columns='source', values='wasserstein_distance').reset_index()
    if {'LHJ2', 'XB10'}.issubset(feature_table.columns):
        feature_table['closer_source'] = np.where(feature_table['LHJ2'] < feature_table['XB10'], 'LHJ2', np.where(feature_table['LHJ2'] > feature_table['XB10'], 'XB10', 'Equal'))
    mean_wasserstein = wasserstein_df.groupby('source')['wasserstein_distance'].mean().to_dict()
    gamma = estimate_mmd_gamma(standardized_data)
    a114_matrix = standardized_data['A114'].dropna().to_numpy(dtype=float)
    if len(a114_matrix) < 2:
        raise RuntimeError('A114用于MMD计算的完整特征记录不足。')
    mmd_results = {}
    for source in ['LHJ2', 'XB10']:
        source_matrix = standardized_data[source].dropna().to_numpy(dtype=float)
        if len(source_matrix) < 2:
            raise RuntimeError(f'{source} 用于MMD计算的完整记录不足。')
        mmd_squared = calculate_mmd_squared(a114_matrix, source_matrix, gamma)
        mmd_results[source] = mmd_squared
    summary_df = pd.DataFrame({'source': ['LHJ2', 'XB10'], 'mean_standardized_wasserstein': [mean_wasserstein['LHJ2'], mean_wasserstein['XB10']], 'MMD_squared': [mmd_results['LHJ2'], mmd_results['XB10']]})
    wasserstein_best = summary_df.loc[summary_df['mean_standardized_wasserstein'].idxmin(), 'source']
    mmd_best = summary_df.loc[summary_df['MMD_squared'].idxmin(), 'source']
    output_path = OUTPUT_DIR / 'source_domain_compatibility.xlsx'
    with pd.ExcelWriter(output_path) as writer:
        feature_table.to_excel(writer, sheet_name='feature_wasserstein', index=False)
        summary_df.to_excel(writer, sheet_name='summary', index=False)
    print('\n' + '=' * 70)
    print('Source-domain compatibility')
    print('=' * 70)
    print('\nFeature-wise standardized Wasserstein:')
    print(feature_table.to_string(index=False))
    print('\nSummary:')
    print(summary_df.to_string(index=False))
    print(f'\nWasserstein判断：A114更接近 {wasserstein_best}')
    print(f'MMD²判断：A114更接近 {mmd_best}')
    if wasserstein_best == mmd_best:
        print(f'\n两个指标结果一致：{wasserstein_best} 更适合作为A114的source domain。')
    else:
        print('\n注意：Wasserstein和MMD²给出了不同的source排序，需要进一步检查。')
    print(f'\n结果已保存：{output_path}')

def main() -> None:
    """Main."""
    configure_plot_style()
    data_dict, label_column_dict = load_all_samples()
    print('\n已读取样品：', list(data_dict.keys()))
    print('特征列：', FEATURE_COLUMNS)
    if DRAW_OVERALL_KDE:
        plot_overall_kde(data_dict)
    if DRAW_LABEL_BOXPLOT:
        plot_label_boxplots(data_dict, label_column_dict)
    if CALCULATE_SOURCE_COMPATIBILITY:
        calculate_source_compatibility(data_dict)
    print('\n' + '=' * 70)
    print('全部分析完成。')
    print(f'结果目录：{OUTPUT_DIR.resolve()}')
    print('=' * 70)
if __name__ == '__main__':
    main()
