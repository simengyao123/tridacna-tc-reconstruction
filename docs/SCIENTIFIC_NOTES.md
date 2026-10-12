# Scientific issues requiring manuscript reconciliation

## Selection on the reported evaluation subset

The supplied BiLSTM training loop evaluates the final 30% after every epoch, uses its TC F1 for early stopping, and saves the checkpoint with the highest score (minimum improvement `1e-4`, patience 10). Therefore this subset influenced model selection. The manuscript calls it an independent test set, including in the SHAP description. Either accurately describe the existing validation-based experiment, or establish an internal validation subset within the first 70%, reserve the final 30% for a single independent evaluation, and rerun all affected analyses. The organized project preserves existing behavior and results so this decision remains explicit.

## Temporal feature overlap

The centered 14-day TDI is computed before the chronological partition. Near that boundary, descriptors can contain raw growth observations from the other partition. Although labels are not used to calculate TDI and retrospective context is intentional, this is not strict raw-observation separation. An independent temporal evaluation needs a stated boundary policy and possibly an embargo determined from both TDI and sequence context. It must be addressed together with any new validation protocol, not by silently changing the reproducibility release.

## Starting data and transfer interpretation

Existing DGRV columns and labels are preserved. Upstream measurement, DGRV generation, and meteorological matching are outside the supplied code. Record their definitions, units, source citations and provenance. The two modern specimens use different label criteria, as described in the paper. Fossil manual markers are not independent ground truth; use reconstructed monthly/seasonal consistency language, not fossil classification accuracy.

## Numerical and boundary details

The repository documents 14-day TDI alignment, endpoint replication, ddof choices, FFT bin selection, entropy bins, smoothing and split/merge order. These details are needed to reproduce results even when not obvious from the paper's formulas. A114 virtual dates include Gregorian leap days and must not be interpreted as historical CE dates.

## Packaging changes

Hard-coded `lwtest` paths were replaced with portable project paths and a shared JSON configuration. Feature/date/label lookup now uses names, fixing the XB10 column-order issue and the risk of using `edge_flag` as the label. A114 output uses `reference_label` instead of `true_label`. Interactive display calls were removed and feature plots are explicitly saved. Script comments and docstrings are English; some original runtime diagnostic messages remain Chinese. Random Forest defaults to one worker because the original `n_jobs=-1` path failed under restricted Windows process/pipe permissions. This does not change its statistical parameters.

Function logic was preserved except for these explicit input/output and execution changes. The user's original files were not edited. Bundled data, reference models and reference results are hashed in `source_manifest.json`. Checkpoint inference and numerical-feature comparisons provide the principal regression evidence. New training results are not substituted for the supplied manuscript checkpoints.
