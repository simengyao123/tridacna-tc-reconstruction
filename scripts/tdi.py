"""TDI numerical definitions preserved from the supplied feature script."""
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import entropy

def extract_tdi_features(window, eps=1e-08, entropy_bins=5):
    """Return anomaly intensity, DGR coefficient of variation, spectral ratio and histogram entropy."""
    x = window['DGR'].values.astype(float)
    xv = window['DGRV'].values.astype(float)
    TDI_A = np.max(np.abs(xv))
    TDI_V = np.std(x) / (np.mean(x) + eps)
    x_detrend = x - np.mean(x)
    X = fft(x_detrend)
    power = np.abs(X) ** 2
    half = len(power) // 2
    high_freq_start = int(half / 3)
    high_freq_energy = np.sum(power[high_freq_start:half])
    total_energy = np.sum(power[:half]) + eps
    TDI_F = high_freq_energy / total_energy
    hist, _ = np.histogram(x, bins=entropy_bins, density=False)
    p = hist / (np.sum(hist) + eps)
    TDI_E = entropy(p + eps)
    return [TDI_A, TDI_V, TDI_F, TDI_E]

def add_daily_tdi_features(df, window_size=7, include_edge_flag=True):
    """Compute local TDI descriptors with endpoint replication; window 14 uses seven preceding and six following rows."""
    required_cols = ['DGR', 'DGRV']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f'输入数据缺少必要列: {missing_cols}')
    left = window_size // 2
    right = window_size - left - 1
    all_features = []
    for i in range(len(df)):
        start = i - left
        end = i + right + 1
        edge_flag = int(start < 0 or end > len(df))
        if start < 0:
            pad_len = -start
            pad = pd.concat([df.iloc[[0]]] * pad_len, ignore_index=True)
            win = pd.concat([pad, df.iloc[0:end]], ignore_index=True)
        elif end > len(df):
            pad_len = end - len(df)
            pad = pd.concat([df.iloc[[-1]]] * pad_len, ignore_index=True)
            win = pd.concat([df.iloc[start:len(df)], pad], ignore_index=True)
        else:
            win = df.iloc[start:end].copy()
        win = win.iloc[:window_size]
        tdi = extract_tdi_features(win)
        if include_edge_flag:
            tdi.append(edge_flag)
        all_features.append(tdi)
    if include_edge_flag:
        tdi_names = ['TDI_A', 'TDI_V', 'TDI_F', 'TDI_E', 'edge_flag']
    else:
        tdi_names = ['TDI_A', 'TDI_V', 'TDI_F', 'TDI_E']
    tdi_df = pd.DataFrame(all_features, columns=tdi_names, index=df.index)
    df_out = pd.concat([df, tdi_df], axis=1)
    return df_out
