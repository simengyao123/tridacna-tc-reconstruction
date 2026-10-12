# Environment and compute requirements

Validation used an existing Python 3.11.15 environment on Windows. No packages were installed or changed. Main package versions are pinned in `requirements.txt`; the actual PyTorch distribution was 2.6.0+cu124 with CUDA available. Run manifests record the exact observed versions. The requirements file is an observed-version record, not a fully transitive or cross-platform lockfile; clean-environment installation was not tested.

CPU inference was explicitly verified for both modern reference models. The original PyTorch scripts select CUDA when available and otherwise CPU. Full SHAP explains 61 XB10 and 39 LHJ2 held-out TC-labelled days using the saved weights. CPU execution of the full SHAP/training workflow was not separately benchmarked. No GPU is required by the program interface.

The largest supplied daily dataset has 4,383 rows; principal model inputs are 9 days by 6 features. Domain comparison caps MMD samples at 2,000 per sample and allocates pairwise distance/kernel matrices. SHAP may temporarily evaluate many perturbed windows. Actual wall time and peak memory depend on hardware; no universal minimum RAM, GPU or runtime is asserted. Preserve sufficient disk space for the PNG/PDF figures and all result tables.

Use a compatible PyTorch build for the intended CPU/CUDA environment. Because fitted scikit-learn objects are serialized, matching the recorded scikit-learn version is particularly important. Only load model/scaler objects from a trusted source. Package-version changes can produce small probability or attribution differences even when random seeds are fixed.
