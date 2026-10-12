# Verification record

Validation completed on 8 October 2026 using the recorded local Python environment.

| Check | Observed result |
| --- | --- |
| Features regenerated from supplied daily files | All dates and labels matched reference files; maximum feature difference below 6e-15 |
| XB10 checkpoint on 177 held-out days | TC F1 0.8243243243; all labels matched archive; largest probability difference 1.20e-7 |
| LHJ2 checkpoint on 330 held-out days | TC F1 0.5882352941; all labels matched archive; largest probability difference 1.79e-7 |
| Fossil event reconstruction | 60 candidates, 81 final model events, 96 manual reference events |
| June–November model / reference proportions | 50/81 = 61.7%; 58/96 = 60.4% |
| Monthly correlations | June–November r=0.802, p=0.055; all months r=0.652, p=0.022 (rounded) |
| Source comparison | LHJ2 Wasserstein 0.286028 and MMD squared 0.063717; XB10 1.211292 and 0.603150 |
| Random Forest, XB10 / LHJ2 TC F1 | 0.6611570248 / 0.2, matching Table 3 after rounding |
| Full SHAP and 1,000 episode bootstraps | Recomputed for both samples; all numeric temporal-summary fields matched supplied summaries |
| SHAP post/pre mean contribution ratio, XB10 | 1.5788536314, 95% interval 1.1652087144–1.9969387480 |
| SHAP post/pre mean contribution ratio, LHJ2 | 2.8764871309, 95% interval 2.1625767475–3.8003091913 |
| Training entry point | One-epoch LHJ2 smoke run completed and saved weights, scaler, history, reports and figures in a separate scratch run |
| Boundary regression tests | Three tests passed: seasonal sequence gaps, TDI alignment/padding, event filtering/merging/splitting |
| Python syntax and comment audit | All Python files parsed; no Chinese comments or docstrings remained |

The workflow creates regenerated outputs and timestamped logs under `results/default`; these generated files are excluded from this distribution. Two initial Random Forest attempts failed because Windows denied creation of a multiprocessing pipe; reruns passed after defaulting to one worker. Their logs are retained in the local preparation backup, outside this distribution. A training smoke run is not a replication of the full optimization experiment. Full retraining and the Table 4–5 ablation grid were not rerun, and cross-platform clean installation was not tested.

These checks establish computational agreement with supplied artifacts, not independent scientific validation. In particular, the checkpoint selection/evaluation issue and upstream data provenance remain as documented in `SCIENTIFIC_NOTES.md`.
