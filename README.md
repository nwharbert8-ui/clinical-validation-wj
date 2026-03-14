# Weighted Jaccard Similarity Detects Organ-System Deterioration Cascades Where Binary Jaccard Fails

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19019981.svg)](https://doi.org/10.5281/zenodo.19019981)

**Author:** Drake H. Harbert (D.H.H.)
**Affiliation:** Inner Architecture LLC, Canton, OH
**ORCID:** [0009-0007-7740-3616](https://orcid.org/0009-0007-7740-3616)
**Contact:** drake@innerarchitecturellc.com

## Summary

This repository contains the complete analytical pipeline for detecting organ-system correlation network reorganization during ICU sepsis using weighted Jaccard similarity. The study demonstrates that binary Jaccard similarity is structurally degenerate at clinical variable counts (6–33 variables) and that weighted Jaccard detects significant network reorganization (patient-level z = −4.95, p < 0.001) where binary Jaccard is blind (z = 0.28).

## Key Findings

- **Binary Jaccard degeneracy:** Null distributions collapse to 2–3 unique values at standard thresholds (τ ≤ 2%), rendering the statistic structurally unsuitable for clinical network comparison
- **Weighted Jaccard detection:** Significant patient-level network reorganization in PhysioNet Sepsis Challenge 2019 (z = −4.95, 95% CI [0.624, 0.674])
- **Cascade profiles:** Hematologic and renal interactions reorganize earliest; cardiovascular within-system coordination is preserved — consistent with sepsis compensatory physiology
- **Permutation-level inflation:** Observation-level permutation inflates z-scores 14-fold (z = −68.91 vs −4.95), demonstrating the necessity of patient-level inference for clustered ICU data

## Datasets

Both datasets are publicly available (no credentialing required):

- **PhysioNet Sepsis Challenge 2019, Training Set A** (20,321 patients, single center)
  https://physionet.org/content/challenge-2019/1.0.0/
- **eICU-CRD Demo v2.0.1** (2,520 stays, 186 hospitals)
  https://physionet.org/content/eicu-crd-demo/2.0.1/

## Repository Contents

| File | Description |
|------|-------------|
| `clinical_wj_validation_pipeline.py` | Main pipeline: data download, cohort construction, permutation testing, all analyses |
| `postprocess_results.py` | Post-processing: cascade analysis, figures, summary from pre-computed permutation results |
| `robustness_analyses.py` | Robustness battery: BJ threshold sweep, multi-threshold, alternative metrics, LOHO, cascade sensitivity |
| `download_physionet.py` | Parallel download utility for PhysioNet 20K .psv files |
| `generate_manuscript.py` | Manuscript generation (.docx) for Computers in Biology and Medicine |
| `requirements.txt` | Python dependencies |

## Reproduction

```bash
# Install dependencies
pip install -r requirements.txt

# Download data (PhysioNet: ~200MB, eICU: ~27MB)
python download_physionet.py
# eICU downloads automatically in main pipeline

# Run main pipeline (includes permutation testing — ~6 hours)
python clinical_wj_validation_pipeline.py

# Or run post-processing only (uses pre-computed permutation results — ~5 min)
python postprocess_results.py

# Run robustness analyses (~6 min)
python robustness_analyses.py
```

**Note:** Data is downloaded to `C:\Users\nwhar\repos\clinical-wj-data\` by default. Update the `DATA_DIR` variable in each script if your data location differs.

## Dependencies

- Python ≥ 3.10
- NumPy, pandas, SciPy, matplotlib
- python-docx (for manuscript generation only)

## Citation

If you use this code, please cite:

> Harbert, D.H. (2026). Weighted Jaccard Similarity Detects Organ-System Deterioration Cascades Where Binary Jaccard Fails: Evidence from Two ICU Databases. *Computers in Biology and Medicine* [submitted].

## License

MIT License. See [LICENSE](LICENSE).
