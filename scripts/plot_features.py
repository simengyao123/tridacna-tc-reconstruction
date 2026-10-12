"""Plot growth descriptors grouped by modern TC labels."""
from project_config import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
xb10_file = PROCESSED_DIR / 'XB10.xlsx'
lhj2_file = PROCESSED_DIR / 'LHJ2.xlsx'
df_xb10 = pd.read_excel(xb10_file)
df_lhj2 = pd.read_excel(lhj2_file)
basic_features = ['DGR', 'DGRV']
tdi_features = ['TDI_A', 'TDI_V', 'TDI_F', 'TDI_E']
required_cols = basic_features + tdi_features + ['label']
for name, df in [('XB10', df_xb10), ('LHJ2', df_lhj2)]:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f'{name} 缺少以下列：{missing}\n{name} 当前包含的列为：\n{df.columns.tolist()}')
df_xb10 = df_xb10[df_xb10['label'].isin([0, 1])].copy()
df_lhj2 = df_lhj2[df_lhj2['label'].isin([0, 1])].copy()
for feature in basic_features + tdi_features:
    df_xb10[feature] = pd.to_numeric(df_xb10[feature], errors='coerce')
    df_lhj2[feature] = pd.to_numeric(df_lhj2[feature], errors='coerce')
df_xb10 = df_xb10.dropna(subset=['label'])
df_lhj2 = df_lhj2.dropna(subset=['label'])

def get_group_data(df, feature):
    """Get group data."""
    feature_df = df.dropna(subset=[feature])
    non_tc = feature_df.loc[feature_df['label'] == 0, feature].to_numpy()
    tc = feature_df.loc[feature_df['label'] == 1, feature].to_numpy()
    if len(non_tc) == 0:
        raise ValueError(f'{feature} 的Non-TC组没有有效数据。')
    if len(tc) == 0:
        raise ValueError(f'{feature} 的TC组没有有效数据。')
    return (non_tc, tc)

def draw_boxplot(ax, data1, data2, title, ylabel, show_xlabel=True):
    """Draw boxplot."""
    bp = ax.boxplot([data1, data2], tick_labels=['Non-TC', 'TC'], patch_artist=True, widths=0.5, showfliers=True, medianprops=dict(linewidth=1.5, color='black'), boxprops=dict(linewidth=1.2, edgecolor='black'), whiskerprops=dict(linewidth=1.2, color='black'), capprops=dict(linewidth=1.2, color='black'), flierprops=dict(marker='o', markersize=3.5, alpha=0.5, markeredgecolor='black'))
    colors = ['#D9EAF7', '#F6D6AD']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_ylabel(ylabel, fontsize=11)
    if not show_xlabel:
        ax.set_xticklabels([])
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
xb10_dgr_non, xb10_dgr_tc = get_group_data(df_xb10, 'DGR')
xb10_dgrv_non, xb10_dgrv_tc = get_group_data(df_xb10, 'DGRV')
lhj2_dgr_non, lhj2_dgr_tc = get_group_data(df_lhj2, 'DGR')
lhj2_dgrv_non, lhj2_dgrv_tc = get_group_data(df_lhj2, 'DGRV')
fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
draw_boxplot(axes1[0, 0], xb10_dgr_non, xb10_dgr_tc, '(a) XB10: DGR', 'DGR')
draw_boxplot(axes1[0, 1], xb10_dgrv_non, xb10_dgrv_tc, '(b) XB10: DGRV', 'DGRV')
draw_boxplot(axes1[1, 0], lhj2_dgr_non, lhj2_dgr_tc, '(c) LHJ2: DGR', 'DGR')
draw_boxplot(axes1[1, 1], lhj2_dgrv_non, lhj2_dgrv_tc, '(d) LHJ2: DGRV', 'DGRV')
fig1.tight_layout()
tdi_display_names = {'TDI_A': 'TDI$_A$', 'TDI_V': 'TDI$_V$', 'TDI_F': 'TDI$_F$', 'TDI_E': 'TDI$_E$'}
fig2, axes2 = plt.subplots(nrows=2, ncols=4, figsize=(16, 7))
panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
panel_index = 0
for col_index, feature in enumerate(tdi_features):
    non_tc, tc = get_group_data(df_xb10, feature)
    display_name = tdi_display_names[feature]
    draw_boxplot(ax=axes2[0, col_index], data1=non_tc, data2=tc, title=f'{panel_labels[panel_index]} XB10: {display_name}', ylabel=display_name)
    panel_index += 1
for col_index, feature in enumerate(tdi_features):
    non_tc, tc = get_group_data(df_lhj2, feature)
    display_name = tdi_display_names[feature]
    draw_boxplot(ax=axes2[1, col_index], data1=non_tc, data2=tc, title=f'{panel_labels[panel_index]} LHJ2: {display_name}', ylabel=display_name)
    panel_index += 1
fig2.tight_layout(h_pad=2.0, w_pad=1.5)
print('\n========== XB10样本数量 ==========')
print(df_xb10['label'].value_counts().sort_index())
print('0 = Non-TC，1 = TC')
print('\n========== LHJ2样本数量 ==========')
print(df_lhj2['label'].value_counts().sort_index())
print('0 = Non-TC，1 = TC')
pass
fig1.savefig(STAGE_OUTPUT / 'basic_features.png', dpi=300, bbox_inches='tight')
fig2.savefig(STAGE_OUTPUT / 'tdi_features.png', dpi=300, bbox_inches='tight')
plt.close('all')

