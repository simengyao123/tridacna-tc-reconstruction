# Data and numerical conventions

The `data/raw` name denotes the starting files for this repository. These files are already daily measured/derived series, not raw shell images or CMA storm tracks. The upstream DGR measurement, DGRV derivation and label-construction code were not supplied. Existing DGRV and labels are preserved, not reconstructed from undocumented assumptions.

| Column | Meaning |
| --- | --- |
| `date` | Modern observation date or A114 virtual date index |
| `DGR` | Supplied daily growth rate; authors must confirm physical units |
| `DGRV` | Supplied growth-rate variation relative to the local centered three-day DGR mean; endpoint preprocessing still requires confirmation |
| `label` | Modern TC presence (1) / absence (0); A114 manual-reference marker |
| `TDI_A` | Maximum absolute DGRV in the local window |
| `TDI_V` | Population standard deviation of DGR divided by mean DGR plus `1e-8` |
| `TDI_F` | Demeaned FFT energy ratio, preserving the supplied frequency-bin convention |
| `TDI_E` | Entropy of a five-bin local DGR histogram, using natural logarithms and epsilon handling from the supplied script |
| `edge_flag` | One when the TDI window required endpoint replication; never a classifier feature or label |
| `reference_label` | A114 manual-reference marker in the regenerated probability file |
| `tc_probability` | TC-class softmax probability; potential-disturbance probability for A114 |
| `predicted_label` | Probability at least 0.5 |

The prepared column order is date, DGR, DGRV, TDI_A, TDI_V, TDI_F, TDI_E, edge_flag, label. Models select columns by name. The empty `Unnamed: 0` XB10 column is omitted from prepared files; the original workbook is retained unchanged.

TDI uses a 14-row window: seven preceding days, target day and six following days. The first/last row is replicated at record edges. For frequency energy, `half = q // 2`, `start = int(half / 3)`, numerator is `power[start:half]`, denominator is `sum(power[:half]) + 1e-8`; this omits the Nyquist bin. Histogram edges are recalculated for each local window. Entropy is calculated using `scipy.stats.entropy(p + eps)` with `p = counts / (sum(counts) + eps)`.

LHJ2 TDI is calculated on all 2,047 consecutive daily records before selecting June–November (1,098 rows). XB10 has 587 rows, A114 4,383. Training/inference sequence windows are separately padded inside uninterrupted daily segments and, for training, inside each partition. Sequence windows cannot bridge winter gaps. The main sequence includes t−5 through t+3.

Each modern classifier uses its own training-fitted StandardScaler. Fossil inference loads the LHJ2 scaler without refitting. Domain compatibility uses a different pooled-modern scaler (sample standard deviation, ddof=1), used only for that comparison.

Event extraction uses centered three-day smoothing, threshold 0.5, minimum two-day runs, then merging gaps of at most three days. Candidates longer than seven days are partitioned using peaks at least four days apart with prominence at least 0.05; balanced splitting is the fallback. Model events are assigned by peak day. The manual reference code groups consecutive markers and uses each run's center; in the supplied A114 file all 96 reference events are single-day markers, so this equals their labelled date.

Authors must supply dataset source citations, ownership/reuse terms, DGR units and upstream preprocessing provenance before public release.

## Alignment with the 10 October 2026 manuscript

For every interior day in all three supplied raw workbooks, DGRV matches `(DGR[t] - mean(DGR[t-1:t+2])) / mean(DGR[t-1:t+2])` to within 9e-16. This supports a centered three-day mean, not the preceding three days. The first and last rows were excluded from this identity check; their supplied values remain unchanged because the upstream endpoint rule is not available. The pipeline continues to read DGRV from the supplied files.

Equation (7) writes `-sum(p * log(p + eps))`; the supplied implementation instead uses `scipy.stats.entropy(p + eps)`, which normalizes `p + eps` before calculating entropy. With the current 14-day windows and eps=1e-8, the maximum absolute difference across the supplied records is about 4.26e-7. The implementation is preserved to maintain the existing feature/model pairing. For exact mathematical correspondence to the implementation, define `q = (p + eps) / sum(p + eps)` and write `-sum(q * log(q))`.

The manuscript attributes the 96 A114 manual growth-anomaly reference events to Zhao et al. (2025), *Prehistoric shifts in tropical cyclone season in the South China Sea: Evidence from daily resolution records of giant clam shells*, npj Climate and Atmospheric Science 8, 218, doi:10.1038/s41612-025-01100-1. This attribution does not establish redistribution permission; data terms still require author confirmation.
