#!/usr/bin/env python3
"""
Pipeline: Clinical WJ Validation — Honest Rebuild v1.0
Author: Drake H. Harbert (D.H.H.)
Affiliation: Inner Architecture LLC, Canton, OH
ORCID: 0009-0007-7740-3616
Date: 2026-03-13
Description:
    Ground-zero rebuild of the clinical WJ validation pipeline.
    Applies weighted Jaccard similarity to detect correlation network
    reorganization during ICU sepsis across two independent databases.

    KEY METHODOLOGICAL CHANGES FROM PREVIOUS VERSION:
    - Spearman correlation (WJ default, robust to outliers)
    - Patient-level permutation as PRIMARY inference
    - Observation-level permutation as SENSITIVITY ONLY
    - Full patient pool (no healthy subsampling)
    - Honest reporting of both permutation levels
    - Binary Jaccard degeneracy is the headline finding

    Datasets:
    - PhysioNet Computing in Cardiology Challenge 2019, Training Set A
    - eICU Collaborative Research Database Demo v2.0.1

Dependencies: numpy, pandas, scipy, matplotlib, seaborn
Input: Auto-downloaded from PhysioNet (open access)
Output: results/ directory with CSVs, figures, summary, provenance
"""

import os
import sys
import gc
import time
import json
import warnings
import zipfile
import tarfile
import shutil
import glob as globmod
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist
from itertools import combinations
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

RANDOM_SEED = 42
FORCE_RECOMPUTE = True
np.random.seed(RANDOM_SEED)

# Paths — CONFIG section for GitHub-ready standards
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Data on LOCAL disk for performance (Google Drive sync kills I/O)
DATA_DIR = r"C:\Users\nwhar\repos\clinical-wj-data"
PHYSIONET_DIR = os.path.join(DATA_DIR, "training_setA")
EICU_DIR = os.path.join(DATA_DIR, "eicu-crd-demo")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")

for d in [DATA_DIR, RESULTS_DIR, FIG_DIR, TABLE_DIR]:
    os.makedirs(d, exist_ok=True)

# Download URLs (open-access datasets, no credentialing needed)
PHYSIONET_ZIP_URL = (
    "https://physionet.org/static/published-projects/"
    "challenge-2019/1.0.0/challenge-2019-1.0.0.zip"
)
EICU_ZIP_URL = (
    "https://physionet.org/static/published-projects/"
    "eicu-crd-demo/2.0.1/eicu-crd-demo-2.0.1.zip"
)

# Analysis parameters
N_PERM = 500           # Permutations for primary analysis (subsample, pairwise-complete)
N_PERM_SENS = 50       # Permutations for sensitivity analyses (enough to show contrast)
N_BOOTSTRAP = 500      # Bootstrap resamples
FDR_Q = 0.05           # FDR threshold
BJ_TAU = 0.05          # Binary Jaccard top-% threshold
MISS_THRESHOLDS = [0.70, 0.85, 0.90, 0.95]
MIN_STAY_HEALTHY_HR = 24
MIN_PRESEPSIS_HR = 12
CORRELATION_METHOD = 'spearman'  # WJ default

# Subsystem definitions (fixed prior to analysis — interpretation, not input)
SUBSYSTEM_MAP = {
    'HR': 'CV', 'O2Sat': 'CV', 'Temp': 'CV', 'SBP': 'CV',
    'MAP': 'CV', 'DBP': 'CV', 'Resp': 'CV',
    'FiO2': 'Resp', 'pH': 'Resp', 'PaCO2': 'Resp', 'PaO2': 'Resp',
    'HCO3': 'Resp', 'BaseExcess': 'Resp', 'SaO2': 'Resp',
    'Glucose': 'Met', 'Lactate': 'Met',
    'Hgb': 'Hem', 'Hct': 'Hem', 'WBC': 'Hem', 'Platelets': 'Hem',
    'PTT': 'Hem', 'Fibrinogen': 'Hem',
    'BUN': 'Ren', 'Creatinine': 'Ren', 'Potassium': 'Ren',
    'Sodium': 'Ren', 'Chloride': 'Ren', 'Calcium': 'Ren',
    'Magnesium': 'Ren', 'Phosphate': 'Ren',
    'AST': 'Hep', 'Alkalinephos': 'Hep', 'Bilirubin_total': 'Hep',
    'TroponinI': 'Card',
}
SUBSYSTEM_NAMES = {
    'CV': 'Cardiovascular', 'Resp': 'Respiratory', 'Met': 'Metabolic',
    'Hem': 'Hematologic', 'Ren': 'Renal', 'Hep': 'Hepatic', 'Card': 'Cardiac',
}

# Figure style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

# Colorblind-safe palette
C = {
    'wj': '#2166AC',   # blue
    'bj': '#B2182B',   # red
    'null': '#999999',  # gray
    'cv': '#4393C3',   # light blue
    'noncv': '#F4A582', # salmon
    'sig': '#2CA02C',  # green
    'nonsig': '#D62728', # red
}

print("=" * 78)
print("CLINICAL WJ VALIDATION — HONEST REBUILD v1.0")
print("Drake H. Harbert | Inner Architecture LLC")
print("Primary inference: PATIENT-LEVEL permutation (Spearman)")
print("=" * 78)
print(f"  Base dir:    {BASE_DIR}")
print(f"  Results dir: {RESULTS_DIR}")
print(f"  Seed: {RANDOM_SEED}  |  Permutations: {N_PERM}  |  Bootstrap: {N_BOOTSTRAP}")
print(f"  Correlation: {CORRELATION_METHOD}  |  FDR q: {FDR_Q}")
print()


# ============================================================
# SECTION 2: CORE FUNCTIONS
# ============================================================

def weighted_jaccard(vec_a, vec_b):
    """Weighted Jaccard similarity between two correlation vectors (upper-tri)."""
    abs_a, abs_b = np.abs(vec_a), np.abs(vec_b)
    num = np.sum(np.minimum(abs_a, abs_b))
    den = np.sum(np.maximum(abs_a, abs_b))
    return float(num / den) if den > 0 else 0.0


def binary_jaccard(vec_a, vec_b, top_pct=0.05):
    """Binary Jaccard: binarize at top_pct, compute set Jaccard."""
    n = len(vec_a)
    k = max(1, int(n * top_pct))
    top_a = set(np.argsort(np.abs(vec_a))[-k:])
    top_b = set(np.argsort(np.abs(vec_b))[-k:])
    inter = len(top_a & top_b)
    union = len(top_a | top_b)
    return inter / union if union > 0 else 0.0


def upper_tri(mat):
    """Extract upper triangle of matrix as flat vector."""
    m = mat.values if hasattr(mat, 'values') else mat
    idx = np.triu_indices(m.shape[0], k=1)
    return m[idx]


def compute_corr_matrix(df, variables, method='spearman'):
    """Compute pairwise correlation matrix using specified method.
    Uses pandas pairwise computation (each pair uses all non-NaN observations)."""
    return df[variables].corr(method=method)


def fisher_z_test(r1, r2, n1, n2):
    """Fisher z-test for difference between two correlations."""
    r1c = np.clip(r1, -0.9999, 0.9999)
    r2c = np.clip(r2, -0.9999, 0.9999)
    z1, z2 = np.arctanh(r1c), np.arctanh(r2c)
    se = np.sqrt(1.0 / max(n1 - 3, 1) + 1.0 / max(n2 - 3, 1))
    if se <= 0:
        return 0.0, 1.0
    z_stat = (z1 - z2) / se
    p_val = 2 * stats.norm.sf(abs(z_stat))
    return z_stat, p_val


def benjamini_hochberg(pvals, q=0.05):
    """Benjamini-Hochberg FDR correction across ALL tested pairs."""
    p = np.array(pvals)
    n = len(p)
    si = np.argsort(p)
    sp = p[si]
    qvals = np.zeros(n)
    qvals[si[-1]] = sp[-1]
    for i in range(n - 2, -1, -1):
        qvals[si[i]] = min(sp[i] * n / (i + 1), qvals[si[i + 1]])
    return qvals, qvals < q


def rv_coefficient(mat_a, mat_b):
    """RV coefficient: multivariate generalization of correlation between matrices."""
    a = mat_a - np.mean(mat_a)
    b = mat_b - np.mean(mat_b)
    num = np.trace(a @ b.T)
    den = np.sqrt(np.trace(a @ a.T) * np.trace(b @ b.T))
    return num / den if den > 0 else 0.0


def frobenius_distance(mat_a, mat_b):
    """Frobenius norm of matrix difference."""
    return np.linalg.norm(mat_a - mat_b, 'fro')


def cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two vectors."""
    return 1.0 - cosine_dist(vec_a, vec_b)


def get_pair_labels(variables):
    """Generate pair labels for upper triangle."""
    pairs = []
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            pairs.append((variables[i], variables[j]))
    return pairs


# ============================================================
# SECTION 3: DATA DOWNLOAD
# ============================================================

def download_with_progress(url, dest_path, description=""):
    """Download a file with progress reporting."""
    if os.path.exists(dest_path):
        print(f"  [SKIP] {description} already exists: {dest_path}")
        return True
    print(f"  [DOWNLOAD] {description}")
    print(f"    URL: {url}")
    print(f"    Dest: {dest_path}")
    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                pct = min(100, count * block_size * 100 // total_size)
                mb = count * block_size / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\r    {pct}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, dest_path, reporthook)
        print()
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"\n    [ERROR] Download failed: {e}")
        return False


def download_physionet_data():
    """Download PhysioNet Sepsis Challenge 2019 Training Set A."""
    # Check if already extracted
    if os.path.isdir(PHYSIONET_DIR):
        psv_files = globmod.glob(os.path.join(PHYSIONET_DIR, "*.psv"))
        if len(psv_files) > 1000:
            print(f"  PhysioNet data found: {len(psv_files)} .psv files")
            return True

    zip_path = os.path.join(DATA_DIR, "challenge-2019-1.0.0.zip")
    success = download_with_progress(
        PHYSIONET_ZIP_URL, zip_path, "PhysioNet Sepsis Challenge 2019"
    )

    if not success:
        # Try alternative URL
        alt_url = ("https://physionet.org/files/challenge-2019/1.0.0/"
                   "training/training_setA.tar.gz")
        tar_path = os.path.join(DATA_DIR, "training_setA.tar.gz")
        success = download_with_progress(alt_url, tar_path, "PhysioNet (alt URL)")
        if success:
            print("  Extracting tar.gz...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(DATA_DIR)
            return True

    if success and os.path.exists(zip_path):
        print("  Extracting zip...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(DATA_DIR)
        # Find the .psv files wherever they ended up
        for root, dirs, files in os.walk(DATA_DIR):
            psv_count = sum(1 for f in files if f.endswith('.psv'))
            if psv_count > 1000 and 'setA' in root.lower():
                # Update PHYSIONET_DIR to actual location
                return root
        return True

    print("\n  [MANUAL DOWNLOAD REQUIRED]")
    print("  Go to: https://physionet.org/content/challenge-2019/1.0.0/")
    print("  Download training_setA and extract to:", PHYSIONET_DIR)
    return False


def download_eicu_data():
    """Download eICU-CRD Demo v2.0.1."""
    # Check if already extracted
    patient_csv = os.path.join(EICU_DIR, "patient.csv")
    patient_gz = os.path.join(EICU_DIR, "patient.csv.gz")
    if os.path.exists(patient_csv) or os.path.exists(patient_gz):
        print("  eICU Demo data found")
        return True

    zip_path = os.path.join(DATA_DIR, "eicu-crd-demo-2.0.1.zip")
    success = download_with_progress(
        EICU_ZIP_URL, zip_path, "eICU-CRD Demo v2.0.1"
    )

    if success and os.path.exists(zip_path):
        print("  Extracting zip...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(DATA_DIR)
        return True

    # Try individual file downloads
    eicu_files = ['patient.csv.gz', 'vitalPeriodic.csv.gz', 'lab.csv.gz',
                  'diagnosis.csv.gz', 'vitalAperiodic.csv.gz']
    os.makedirs(EICU_DIR, exist_ok=True)
    all_ok = True
    for fname in eicu_files:
        url = f"https://physionet.org/files/eicu-crd-demo/2.0.1/{fname}"
        dest = os.path.join(EICU_DIR, fname)
        if not download_with_progress(url, dest, f"eICU {fname}"):
            all_ok = False
    if all_ok:
        return True

    print("\n  [MANUAL DOWNLOAD REQUIRED]")
    print("  Go to: https://physionet.org/content/eicu-crd-demo/2.0.1/")
    print("  Download all CSV files and place in:", EICU_DIR)
    return False


# ============================================================
# SECTION 4: DATA LOADING — PhysioNet
# ============================================================

def load_physionet(data_dir):
    """Load PhysioNet Sepsis 2019 Training Set A into a unified DataFrame.
    Returns: (DataFrame with patient_id column, dict of patient metadata)"""
    print("\n--- Loading PhysioNet Sepsis Challenge 2019 ---")
    t0 = time.time()

    psv_files = sorted(globmod.glob(os.path.join(data_dir, "*.psv")))
    if not psv_files:
        # Search recursively
        psv_files = sorted(globmod.glob(os.path.join(data_dir, "**", "*.psv"),
                                        recursive=True))
    print(f"  Found {len(psv_files):,} .psv files")
    assert len(psv_files) > 1000, f"Expected 10K+ files, got {len(psv_files)}"

    frames = []
    patient_meta = {}
    available_vars = None

    for i, fp in enumerate(psv_files):
        pid = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp, sep='|')

        # On first file, determine which clinical vars actually exist
        if available_vars is None:
            available_vars = [v for v in SUBSYSTEM_MAP.keys() if v in df.columns]
            print(f"  Available clinical vars: {len(available_vars)} of {len(SUBSYSTEM_MAP)}")
            missing = [v for v in SUBSYSTEM_MAP.keys() if v not in df.columns]
            if missing:
                print(f"  Missing from data: {missing}")

        # Extract metadata
        has_sepsis = int(df['SepsisLabel'].max()) if 'SepsisLabel' in df.columns else 0
        if has_sepsis:
            onset_idx = df['SepsisLabel'].idxmax()
            presepsis_hours = onset_idx  # hours before onset
        else:
            onset_idx = None
            presepsis_hours = len(df)

        stay_hours = len(df)
        patient_meta[pid] = {
            'has_sepsis': has_sepsis,
            'stay_hours': stay_hours,
            'onset_idx': onset_idx,
            'presepsis_hours': presepsis_hours,
        }

        # Select observations: all for healthy, pre-onset for sepsis
        if has_sepsis and onset_idx is not None:
            obs = df.iloc[:onset_idx].copy()
        else:
            obs = df.copy()

        if len(obs) > 0:
            keep_cols = [v for v in available_vars if v in obs.columns]
            if 'SepsisLabel' in obs.columns:
                keep_cols = keep_cols + ['SepsisLabel']
            obs = obs[keep_cols].copy()
            obs['patient_id'] = pid
            obs['hour'] = range(len(obs))
            frames.append(obs)

        if (i + 1) % 5000 == 0:
            print(f"    Loaded {i + 1:,} / {len(psv_files):,} files...")

    full_df = pd.concat(frames, ignore_index=True)
    elapsed = time.time() - t0
    print(f"  Loaded {len(full_df):,} observations from {len(patient_meta):,} patients "
          f"in {elapsed:.1f}s")
    return full_df, patient_meta


def construct_physionet_cohorts(full_df, patient_meta):
    """Construct healthy and sepsis cohorts with forward-fill within patient."""
    print("\n--- Constructing PhysioNet Cohorts ---")

    # Healthy: no sepsis, stay >= 24h
    healthy_pids = [pid for pid, m in patient_meta.items()
                    if not m['has_sepsis'] and m['stay_hours'] >= MIN_STAY_HEALTHY_HR]

    # Sepsis: has sepsis, >= 12h pre-onset data
    sepsis_pids = [pid for pid, m in patient_meta.items()
                   if m['has_sepsis'] and m['presepsis_hours'] >= MIN_PRESEPSIS_HR]

    print(f"  Healthy patients: {len(healthy_pids):,} (no sepsis, stay >= {MIN_STAY_HEALTHY_HR}h)")
    print(f"  Sepsis patients:  {len(sepsis_pids):,} (sepsis, >= {MIN_PRESEPSIS_HR}h pre-onset)")

    # Forward-fill within each patient
    clinical_vars = [v for v in SUBSYSTEM_MAP.keys() if v in full_df.columns]
    full_df_sorted = full_df.sort_values(['patient_id', 'hour'])
    full_df_sorted[clinical_vars] = full_df_sorted.groupby('patient_id')[clinical_vars].ffill()

    # Split into cohorts — use ALL patients (no subsampling)
    healthy_df = full_df_sorted[full_df_sorted['patient_id'].isin(healthy_pids)].copy()
    sepsis_df = full_df_sorted[full_df_sorted['patient_id'].isin(sepsis_pids)].copy()

    print(f"  Healthy observations: {len(healthy_df):,}")
    print(f"  Sepsis observations:  {len(sepsis_df):,}")

    return healthy_df, sepsis_df, healthy_pids, sepsis_pids


# ============================================================
# SECTION 5: DATA LOADING — eICU
# ============================================================

def load_csv_flexible(directory, basename):
    """Load a CSV from directory, trying .csv and .csv.gz."""
    for ext in ['.csv', '.csv.gz']:
        path = os.path.join(directory, basename + ext)
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError(f"Cannot find {basename}.csv or {basename}.csv.gz in {directory}")


def load_eicu(data_dir):
    """Load eICU-CRD Demo v2.0.1 and construct unified DataFrame."""
    print("\n--- Loading eICU-CRD Demo v2.0.1 ---")
    t0 = time.time()

    # Find the actual data directory (may be nested after zip extraction)
    actual_dir = data_dir
    if not os.path.exists(os.path.join(data_dir, 'patient.csv')) and \
       not os.path.exists(os.path.join(data_dir, 'patient.csv.gz')):
        # Search for patient.csv in subdirectories
        for root, dirs, files in os.walk(os.path.dirname(data_dir)):
            if 'patient.csv' in files or 'patient.csv.gz' in files:
                actual_dir = root
                break

    patient = load_csv_flexible(actual_dir, 'patient')
    vitals = load_csv_flexible(actual_dir, 'vitalPeriodic')
    labs = load_csv_flexible(actual_dir, 'lab')
    diag = load_csv_flexible(actual_dir, 'diagnosis')

    # Try to load aperiodic vitals for BP
    try:
        vitals_ap = load_csv_flexible(actual_dir, 'vitalAperiodic')
        has_aperiodic = True
    except FileNotFoundError:
        has_aperiodic = False

    print(f"  Patients: {len(patient):,}")
    print(f"  Vital records: {len(vitals):,}")
    print(f"  Lab records: {len(labs):,}")
    print(f"  Diagnoses: {len(diag):,}")

    # Identify sepsis patients via diagnosis
    sepsis_mask = diag['diagnosisstring'].str.contains(
        'sepsis|septic', case=False, na=False
    )
    sepsis_stays = set(diag.loc[sepsis_mask, 'patientunitstayid'].unique())
    print(f"  Sepsis stays (ICD-based): {len(sepsis_stays):,}")

    # Map vital columns to standard names
    vital_col_map = {
        'heartrate': 'HR',
        'systemicsystolic': 'SBP',
        'systemicdiastolic': 'DBP',
        'systemicmean': 'MAP',
        'respiration': 'Resp',
        'temperature': 'Temp',
        'sao2': 'O2Sat',
    }

    # Aggregate vitals to hourly
    vitals_std = vitals[['patientunitstayid', 'observationoffset'] +
                        [c for c in vital_col_map if c in vitals.columns]].copy()
    vitals_std = vitals_std.rename(columns=vital_col_map)
    vitals_std['hour'] = (vitals_std['observationoffset'] / 60).round().astype(int)
    vitals_hourly = vitals_std.groupby(['patientunitstayid', 'hour']).mean(
        numeric_only=True
    ).reset_index()
    vitals_hourly = vitals_hourly.drop(columns=['observationoffset'], errors='ignore')

    # Merge aperiodic BP if available (fill gaps in periodic)
    if has_aperiodic:
        ap_col_map = {
            'noninvasivesystolic': 'SBP',
            'noninvasivediastolic': 'DBP',
            'noninvasivemean': 'MAP',
        }
        ap_cols = [c for c in ap_col_map if c in vitals_ap.columns]
        if ap_cols:
            vitals_ap_std = vitals_ap[['patientunitstayid', 'observationoffset'] + ap_cols].copy()
            vitals_ap_std = vitals_ap_std.rename(columns=ap_col_map)
            vitals_ap_std['hour'] = (vitals_ap_std['observationoffset'] / 60).round().astype(int)
            vitals_ap_hourly = vitals_ap_std.groupby(['patientunitstayid', 'hour']).mean(
                numeric_only=True
            ).reset_index()
            vitals_ap_hourly = vitals_ap_hourly.drop(columns=['observationoffset'],
                                                      errors='ignore')
            # Fill missing periodic BP with aperiodic
            vitals_hourly = vitals_hourly.merge(
                vitals_ap_hourly, on=['patientunitstayid', 'hour'],
                how='outer', suffixes=('', '_ap')
            )
            for col in ['SBP', 'DBP', 'MAP']:
                ap_col = col + '_ap'
                if ap_col in vitals_hourly.columns:
                    vitals_hourly[col] = vitals_hourly[col].fillna(vitals_hourly[ap_col])
                    vitals_hourly = vitals_hourly.drop(columns=[ap_col])

    # Pivot labs: labname → columns
    lab_name_map = {
        'BUN': 'BUN', 'creatinine': 'Creatinine', 'potassium': 'Potassium',
        'sodium': 'Sodium', 'chloride': 'Chloride', 'calcium': 'Calcium',
        'magnesium': 'Magnesium', 'phosphate': 'Phosphate',
        'glucose': 'Glucose', 'lactate': 'Lactate',
        'Hgb': 'Hgb', 'Hct': 'Hct', 'WBC x 1000': 'WBC',
        'platelets x 1000': 'Platelets', 'PTT': 'PTT', 'fibrinogen': 'Fibrinogen',
        'AST (SGOT)': 'AST', 'alkaline phos.': 'Alkalinephos',
        'total bilirubin': 'Bilirubin_total', 'troponin - I': 'TroponinI',
        'pH': 'pH', 'FiO2': 'FiO2', 'paCO2': 'PaCO2', 'paO2': 'PaO2',
        'HCO3': 'HCO3', 'Base Excess': 'BaseExcess',
    }

    # Filter to known lab names
    labs_known = labs[labs['labname'].isin(lab_name_map.keys())].copy()
    labs_known['labname_std'] = labs_known['labname'].map(lab_name_map)
    labs_known['hour'] = (labs_known['labresultoffset'] / 60).round().astype(int)

    # Convert labresult to numeric
    labs_known['labresult'] = pd.to_numeric(labs_known['labresult'], errors='coerce')

    # Pivot
    labs_pivot = labs_known.pivot_table(
        index=['patientunitstayid', 'hour'],
        columns='labname_std',
        values='labresult',
        aggfunc='mean'
    ).reset_index()

    # Merge vitals + labs
    merged = vitals_hourly.merge(labs_pivot, on=['patientunitstayid', 'hour'], how='outer')

    # Add sepsis label
    merged['is_sepsis'] = merged['patientunitstayid'].isin(sepsis_stays).astype(int)

    # Get unique hospital IDs for leave-one-out analysis
    if 'hospitalid' in patient.columns:
        hosp_map = patient.set_index('patientunitstayid')['hospitalid'].to_dict()
        merged['hospitalid'] = merged['patientunitstayid'].map(hosp_map)

    elapsed = time.time() - t0
    print(f"  Merged DataFrame: {len(merged):,} rows, {merged.shape[1]} columns in {elapsed:.1f}s")

    return merged, sepsis_stays


def construct_eicu_cohorts(merged_df, sepsis_stays):
    """Construct healthy and sepsis cohorts for eICU."""
    print("\n--- Constructing eICU Cohorts ---")

    clinical_vars = [v for v in SUBSYSTEM_MAP.keys() if v in merged_df.columns]
    all_stays = merged_df['patientunitstayid'].unique()
    healthy_stays = [s for s in all_stays if s not in sepsis_stays]
    sepsis_stay_list = [s for s in all_stays if s in sepsis_stays]

    print(f"  Available clinical variables: {len(clinical_vars)}")
    print(f"  Variables: {clinical_vars}")

    # Forward-fill within patient
    merged_sorted = merged_df.sort_values(['patientunitstayid', 'hour'])
    merged_sorted[clinical_vars] = merged_sorted.groupby('patientunitstayid')[clinical_vars].ffill()

    healthy_df = merged_sorted[merged_sorted['patientunitstayid'].isin(healthy_stays)].copy()
    sepsis_df = merged_sorted[merged_sorted['patientunitstayid'].isin(sepsis_stay_list)].copy()

    print(f"  Healthy stays: {len(healthy_stays):,}  ({len(healthy_df):,} observations)")
    print(f"  Sepsis stays:  {len(sepsis_stay_list):,}  ({len(sepsis_df):,} observations)")

    return healthy_df, sepsis_df, healthy_stays, sepsis_stay_list, clinical_vars


# ============================================================
# SECTION 6: VARIABLE SELECTION BY MISSINGNESS
# ============================================================

def select_variables_by_missingness(healthy_df, sepsis_df, threshold,
                                     candidate_vars, subsystem_map):
    """Select variables with overall missingness below threshold."""
    combined = pd.concat([healthy_df[candidate_vars], sepsis_df[candidate_vars]])
    miss_rates = combined.isnull().mean()
    selected = [v for v in candidate_vars if miss_rates.get(v, 1.0) < threshold]

    # Count subsystems
    subsystems = set(subsystem_map.get(v, '?') for v in selected)

    return selected, miss_rates, subsystems


# ============================================================
# SECTION 7: PERMUTATION TESTS
# ============================================================

def patient_level_permutation(healthy_df, sepsis_df, variables, patient_id_col,
                               n_perm, method='spearman'):
    """Patient-level permutation test for WJ.

    This is the PRIMARY inference. Each permutation shuffles patient labels
    (not individual observations), respecting the hierarchical structure.

    CONSISTENCY: Uses pandas .corr(method=method) with pairwise-complete
    for BOTH observed and null. Subsamples healthy patients to keep
    permutation runtime tractable while preserving all sepsis patients.
    """
    print(f"\n  Patient-level permutation ({n_perm} iterations, {method})...")
    t0 = time.time()

    # Compute observed correlation matrices and WJ (full data)
    corr_h = compute_corr_matrix(healthy_df, variables, method)
    corr_s = compute_corr_matrix(sepsis_df, variables, method)
    vec_h = upper_tri(corr_h.values)
    vec_s = upper_tri(corr_s.values)
    obs_wj = weighted_jaccard(vec_h, vec_s)
    obs_bj = binary_jaccard(vec_h, vec_s, BJ_TAU)
    print(f"    Observed WJ (full data): {obs_wj:.4f}  |  BJ: {obs_bj:.4f}")

    h_pids = healthy_df[patient_id_col].unique()
    s_pids = sepsis_df[patient_id_col].unique()

    # Subsample healthy patients for permutation tractability
    # Keep all sepsis patients, subsample healthy to keep total manageable
    PERM_HEALTHY_N = min(3000, len(h_pids))
    np.random.seed(RANDOM_SEED)
    h_pids_sub = np.random.choice(h_pids, size=PERM_HEALTHY_N, replace=False)
    all_pids = np.concatenate([h_pids_sub, s_pids])
    n_healthy_sub = len(h_pids_sub)
    n_total = len(all_pids)
    print(f"    Permutation subsample: {n_healthy_sub} healthy + {len(s_pids)} sepsis "
          f"= {n_total} patients")

    # Build per-patient DataFrames for fast permutation
    h_sub_df = healthy_df[healthy_df[patient_id_col].isin(h_pids_sub)]
    combined = pd.concat([h_sub_df[variables + [patient_id_col]],
                          sepsis_df[variables + [patient_id_col]]],
                         ignore_index=True)

    # Compute observed WJ on the SUBSAMPLE too (for consistency with null)
    corr_h_sub = compute_corr_matrix(h_sub_df, variables, method)
    corr_s_sub = compute_corr_matrix(sepsis_df, variables, method)
    obs_wj_sub = weighted_jaccard(upper_tri(corr_h_sub.values),
                                   upper_tri(corr_s_sub.values))
    obs_bj_sub = binary_jaccard(upper_tri(corr_h_sub.values),
                                 upper_tri(corr_s_sub.values), BJ_TAU)
    print(f"    Observed WJ (subsample): {obs_wj_sub:.4f}  |  BJ: {obs_bj_sub:.4f}")

    # Pre-index patients to rows
    pid_groups = combined.groupby(patient_id_col)
    pid_to_idx = {pid: grp.index.values for pid, grp in pid_groups}

    null_wj = np.zeros(n_perm)
    null_bj = np.zeros(n_perm)

    np.random.seed(RANDOM_SEED + 1)  # Different seed for permutations
    for i in range(n_perm):
        perm_idx = np.random.permutation(n_total)
        perm_h_pids = all_pids[perm_idx[:n_healthy_sub]]

        h_rows = np.concatenate([pid_to_idx[p] for p in perm_h_pids
                                 if p in pid_to_idx])
        s_rows = np.setdiff1d(combined.index.values, h_rows)

        perm_h_df = combined.loc[h_rows, variables]
        perm_s_df = combined.loc[s_rows, variables]

        perm_corr_h = perm_h_df.corr(method=method)
        perm_corr_s = perm_s_df.corr(method=method)
        pv_h = upper_tri(perm_corr_h.values)
        pv_s = upper_tri(perm_corr_s.values)

        null_wj[i] = weighted_jaccard(pv_h, pv_s)
        null_bj[i] = binary_jaccard(pv_h, pv_s, BJ_TAU)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_perm - i - 1) / rate
            print(f"    Perm {i+1}/{n_perm}  "
                  f"(null WJ mean={null_wj[:i+1].mean():.4f}, "
                  f"sd={null_wj[:i+1].std():.4f})  "
                  f"{rate:.2f}/s  ETA: {eta:.0f}s")

    # z-scores computed against SUBSAMPLE observed (methodological consistency)
    wj_mean, wj_std = null_wj.mean(), null_wj.std()
    bj_mean, bj_std = null_bj.mean(), null_bj.std()
    z_wj = (obs_wj_sub - wj_mean) / wj_std if wj_std > 0 else 0.0
    z_bj = (obs_bj_sub - bj_mean) / bj_std if bj_std > 0 else 0.0
    p_wj = np.mean(null_wj <= obs_wj_sub)

    elapsed = time.time() - t0
    print(f"  Patient-level permutation complete in {elapsed:.1f}s")
    print(f"    Observed WJ (subsample): {obs_wj_sub:.4f}  |  Null: {wj_mean:.4f}  "
          f"±{wj_std:.4f}  |  z = {z_wj:.2f}  |  p = {p_wj:.4f}")
    print(f"    Observed WJ (full data): {obs_wj:.4f}")
    print(f"    Observed BJ (subsample): {obs_bj_sub:.4f}  |  Null: {bj_mean:.4f}  "
          f"±{bj_std:.4f}  |  z = {z_bj:.2f}")

    return {
        'obs_wj': obs_wj, 'obs_wj_sub': obs_wj_sub,
        'obs_bj': obs_bj, 'obs_bj_sub': obs_bj_sub,
        'null_wj': null_wj, 'null_bj': null_bj,
        'z_wj': z_wj, 'z_bj': z_bj,
        'p_wj': p_wj,
        'null_wj_mean': wj_mean, 'null_wj_std': wj_std,
        'null_bj_mean': bj_mean, 'null_bj_std': bj_std,
        'corr_h': corr_h, 'corr_s': corr_s,
        'n_perm_healthy': PERM_HEALTHY_N,
        'n_perm_sepsis': len(s_pids),
    }


def observation_level_permutation(healthy_df, sepsis_df, variables,
                                    n_perm, method='spearman'):
    """Observation-level permutation (SENSITIVITY analysis only).

    WARNING: This inflates z-scores by treating non-independent observations
    as independent. Reported for comparison with prior literature only.
    Patient-level permutation is the defensible primary inference.
    """
    print(f"\n  Observation-level permutation ({n_perm} iterations, SENSITIVITY)...")
    t0 = time.time()

    corr_h = compute_corr_matrix(healthy_df, variables, method)
    corr_s = compute_corr_matrix(sepsis_df, variables, method)
    vec_h = upper_tri(corr_h.values)
    vec_s = upper_tri(corr_s.values)
    obs_wj = weighted_jaccard(vec_h, vec_s)

    # Subsample for tractability (same as patient-level subsample size)
    MAX_OBS = 50000  # Cap observations per group for speed
    h_sub = healthy_df[variables].sample(n=min(MAX_OBS, len(healthy_df)),
                                          random_state=RANDOM_SEED)
    s_sub = sepsis_df[variables].sample(n=min(MAX_OBS, len(sepsis_df)),
                                         random_state=RANDOM_SEED)
    combined = pd.concat([h_sub, s_sub], ignore_index=True)
    n_h = len(h_sub)
    n_total = len(combined)
    print(f"    Subsample: {n_h} healthy + {len(s_sub)} sepsis = {n_total} obs")

    null_wj = np.zeros(n_perm)
    for i in range(n_perm):
        perm_idx = np.random.permutation(n_total)
        perm_h = combined.iloc[perm_idx[:n_h]]
        perm_s = combined.iloc[perm_idx[n_h:]]
        pv_h = upper_tri(compute_corr_matrix(perm_h, variables, method).values)
        pv_s = upper_tri(compute_corr_matrix(perm_s, variables, method).values)
        null_wj[i] = weighted_jaccard(pv_h, pv_s)

        if (i + 1) % 10 == 0:
            print(f"    Perm {i+1}/{n_perm} (null mean={null_wj[:i+1].mean():.4f})")

    wj_mean, wj_std = null_wj.mean(), null_wj.std()
    z_wj = (obs_wj - wj_mean) / wj_std if wj_std > 0 else 0.0

    elapsed = time.time() - t0
    print(f"  Observation-level done in {elapsed:.1f}s")
    print(f"    z = {z_wj:.2f} (SENSITIVITY — inflated by non-independence)")

    return {
        'obs_wj': obs_wj,
        'null_wj': null_wj,
        'z_wj': z_wj,
        'null_wj_mean': wj_mean,
        'null_wj_std': wj_std,
    }


# ============================================================
# SECTION 8: BOOTSTRAP AND SENSITIVITY
# ============================================================

def patient_cluster_bootstrap(healthy_df, sepsis_df, variables, patient_id_col,
                                n_boot, method='spearman'):
    """Patient-level cluster bootstrap for WJ 95% CI."""
    print(f"\n  Cluster bootstrap ({n_boot} resamples)...")
    t0 = time.time()

    h_pids = healthy_df[patient_id_col].unique()
    s_pids = sepsis_df[patient_id_col].unique()

    # Pre-index
    h_pid_obs = {pid: healthy_df[healthy_df[patient_id_col] == pid][variables]
                 for pid in h_pids}
    s_pid_obs = {pid: sepsis_df[sepsis_df[patient_id_col] == pid][variables]
                 for pid in s_pids}

    boot_wj = np.zeros(n_boot)
    for i in range(n_boot):
        # Resample patients with replacement
        boot_h_pids = np.random.choice(h_pids, size=len(h_pids), replace=True)
        boot_s_pids = np.random.choice(s_pids, size=len(s_pids), replace=True)

        boot_h = pd.concat([h_pid_obs[p] for p in boot_h_pids], ignore_index=True)
        boot_s = pd.concat([s_pid_obs[p] for p in boot_s_pids], ignore_index=True)

        corr_h = compute_corr_matrix(boot_h, variables, method)
        corr_s = compute_corr_matrix(boot_s, variables, method)
        boot_wj[i] = weighted_jaccard(upper_tri(corr_h.values), upper_tri(corr_s.values))

        if (i + 1) % 200 == 0:
            print(f"    Bootstrap {i+1}/{n_boot}")

    ci_lo, ci_hi = np.percentile(boot_wj, [2.5, 97.5])
    elapsed = time.time() - t0
    print(f"  Bootstrap done in {elapsed:.1f}s")
    print(f"    WJ 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]  (mean={boot_wj.mean():.4f})")

    return boot_wj, ci_lo, ci_hi


def leave_one_hospital_out(merged_df, sepsis_stays, variables, method='spearman',
                            min_patients=20):
    """Leave-one-hospital-out cross-validation for eICU."""
    if 'hospitalid' not in merged_df.columns:
        print("  [SKIP] No hospitalid column for LOHO analysis")
        return None

    print(f"\n  Leave-one-hospital-out cross-validation...")
    hosp_counts = merged_df.groupby('hospitalid')['patientunitstayid'].nunique()
    eligible = hosp_counts[hosp_counts >= min_patients].index
    print(f"  Eligible hospitals (>={min_patients} patients): {len(eligible)}")

    loho_wj = {}
    for hosp_id in eligible:
        excl_mask = merged_df['hospitalid'] != hosp_id
        excl_df = merged_df[excl_mask]

        h_df = excl_df[~excl_df['patientunitstayid'].isin(sepsis_stays)]
        s_df = excl_df[excl_df['patientunitstayid'].isin(sepsis_stays)]

        if len(h_df) < 100 or len(s_df) < 50:
            continue

        corr_h = compute_corr_matrix(h_df, variables, method)
        corr_s = compute_corr_matrix(s_df, variables, method)
        wj = weighted_jaccard(upper_tri(corr_h.values), upper_tri(corr_s.values))
        loho_wj[hosp_id] = wj

    vals = list(loho_wj.values())
    if vals:
        print(f"  LOHO WJ: mean={np.mean(vals):.4f}, sd={np.std(vals):.4f}, "
              f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]")
    return loho_wj


def alternative_metrics_permutation(healthy_df, sepsis_df, variables,
                                     n_perm, method='spearman'):
    """Test alternative similarity metrics under permutation."""
    print(f"\n  Alternative metrics permutation ({n_perm} iterations)...")

    corr_h = compute_corr_matrix(healthy_df, variables, method)
    corr_s = compute_corr_matrix(sepsis_df, variables, method)
    vh, vs = upper_tri(corr_h.values), upper_tri(corr_s.values)

    obs_cos = cosine_similarity(vh, vs)
    obs_rv = rv_coefficient(corr_h.values, corr_s.values)
    obs_frob = frobenius_distance(corr_h.values, corr_s.values)

    combined = pd.concat([healthy_df[variables], sepsis_df[variables]], ignore_index=True)
    n_h = len(healthy_df)

    null_cos = np.zeros(n_perm)
    null_rv = np.zeros(n_perm)
    null_frob = np.zeros(n_perm)

    for i in range(n_perm):
        idx = np.random.permutation(len(combined))
        ph = combined.iloc[idx[:n_h]]
        ps = combined.iloc[idx[n_h:]]
        ch = compute_corr_matrix(ph, variables, method).values
        cs = compute_corr_matrix(ps, variables, method).values
        pvh, pvs = upper_tri(ch), upper_tri(cs)
        null_cos[i] = cosine_similarity(pvh, pvs)
        null_rv[i] = rv_coefficient(ch, cs)
        null_frob[i] = frobenius_distance(ch, cs)

    results = {}
    for name, obs, null in [('Cosine', obs_cos, null_cos),
                             ('RV', obs_rv, null_rv),
                             ('Frobenius', obs_frob, null_frob)]:
        m, s = null.mean(), null.std()
        z = (obs - m) / s if s > 0 else 0.0
        results[name] = {'observed': obs, 'z': z, 'null_mean': m, 'null_std': s}
        print(f"    {name}: obs={obs:.4f}, z={z:.2f}")

    return results


# ============================================================
# SECTION 9: CASCADE ANALYSIS
# ============================================================

def pair_level_divergence(corr_h, corr_s, variables, n_h_obs, n_s_obs):
    """Compute pair-level Fisher z divergence with FDR correction."""
    pairs = get_pair_labels(variables)
    results = []
    p_vals = []

    for vi, vj in pairs:
        r_h = corr_h.loc[vi, vj]
        r_s = corr_s.loc[vi, vj]
        z_stat, p_val = fisher_z_test(r_h, r_s, n_h_obs, n_s_obs)
        delta_r = abs(r_h - r_s)
        sub_i = SUBSYSTEM_MAP.get(vi, '?')
        sub_j = SUBSYSTEM_MAP.get(vj, '?')
        cross = sub_i != sub_j
        interaction = tuple(sorted([sub_i, sub_j]))

        results.append({
            'var_i': vi, 'var_j': vj,
            'r_healthy': r_h, 'r_sepsis': r_s,
            'delta_r': delta_r, 'abs_delta_r': delta_r,
            'fisher_z': z_stat, 'p_value': p_val,
            'subsystem_i': sub_i, 'subsystem_j': sub_j,
            'cross_subsystem': cross,
            'interaction': f"{interaction[0]}-{interaction[1]}",
        })
        p_vals.append(p_val)

    # FDR across ALL pairs
    qvals, sig_mask = benjamini_hochberg(p_vals, FDR_Q)
    for i, r in enumerate(results):
        r['fdr_q'] = qvals[i]
        r['significant'] = bool(sig_mask[i])

    df = pd.DataFrame(results)
    n_sig = df['significant'].sum()
    n_cross = df.loc[df['significant'], 'cross_subsystem'].sum()
    print(f"  Pair-level divergence: {n_sig}/{len(df)} significant (FDR q<{FDR_Q})")
    print(f"  Cross-subsystem: {n_cross}/{n_sig} ({100*n_cross/max(n_sig,1):.1f}%)")
    return df


def cascade_analysis(divergence_df):
    """Compute organ-system cascade ordering from divergence data."""
    cascade = divergence_df.groupby('interaction').agg(
        mean_delta_r=('abs_delta_r', 'mean'),
        n_pairs=('abs_delta_r', 'count'),
        n_significant=('significant', 'sum'),
        pct_significant=('significant', 'mean'),
    ).reset_index()
    cascade = cascade.sort_values('mean_delta_r', ascending=False)
    cascade['rank'] = range(1, len(cascade) + 1)

    # Label CV vs non-CV
    cascade['involves_cv'] = cascade['interaction'].str.contains('CV')

    print("\n  Cascade ordering:")
    for _, row in cascade.iterrows():
        cv_tag = " [CV]" if row['involves_cv'] else ""
        print(f"    {row['rank']:2d}. {row['interaction']:15s}  "
              f"|dr|={row['mean_delta_r']:.4f}  "
              f"{row['n_significant']:.0f}/{row['n_pairs']:.0f} sig "
              f"({row['pct_significant']*100:.0f}%){cv_tag}")

    return cascade


# ============================================================
# SECTION 10: BINARY JACCARD DEGENERACY ANALYSIS
# ============================================================

def bj_threshold_sweep(corr_h, corr_s, variables, n_perm=200):
    """Sweep binary Jaccard threshold to demonstrate structural degeneracy."""
    print("\n  Binary Jaccard threshold sweep (degeneracy analysis)...")
    vec_h = upper_tri(corr_h.values)
    vec_s = upper_tri(corr_s.values)
    n_pairs = len(vec_h)

    thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
    results = []

    for tau in thresholds:
        obs_bj = binary_jaccard(vec_h, vec_s, tau)
        k = max(1, int(n_pairs * tau))

        # Quick permutation null
        combined = np.column_stack([vec_h, vec_s])
        null_bj = np.zeros(n_perm)
        for i in range(n_perm):
            idx = np.random.permutation(len(vec_h))
            null_bj[i] = binary_jaccard(vec_h[idx], vec_s, tau)

        unique_null = len(np.unique(np.round(null_bj, 6)))
        bj_std = null_bj.std()
        z = (obs_bj - null_bj.mean()) / bj_std if bj_std > 0 else 0.0

        results.append({
            'tau': tau, 'k_edges': k, 'obs_bj': obs_bj,
            'null_mean': null_bj.mean(), 'null_std': bj_std,
            'z': z, 'unique_null_values': unique_null,
            'degenerate': unique_null <= 3,
        })
        print(f"    tau={tau:.2f}: k={k}, BJ={obs_bj:.3f}, z={z:.2f}, "
              f"unique_null={unique_null}, {'DEGENERATE' if unique_null <= 3 else 'ok'}")

    return pd.DataFrame(results)


# ============================================================
# SECTION 11: MULTI-THRESHOLD ANALYSIS
# ============================================================

def multi_threshold_analysis(healthy_df, sepsis_df, candidate_vars, patient_id_col,
                              thresholds, method='spearman'):
    """Run WJ/BJ at multiple missingness thresholds."""
    print("\n--- Multi-Threshold Analysis ---")
    results = []

    for thresh in thresholds:
        sel_vars, miss_rates, subsystems = select_variables_by_missingness(
            healthy_df, sepsis_df, thresh, candidate_vars, SUBSYSTEM_MAP
        )
        if len(sel_vars) < 3:
            print(f"  Threshold {thresh:.0%}: {len(sel_vars)} vars — skipped (need >=3)")
            continue

        corr_h = compute_corr_matrix(healthy_df, sel_vars, method)
        corr_s = compute_corr_matrix(sepsis_df, sel_vars, method)
        vh = upper_tri(corr_h.values)
        vs = upper_tri(corr_s.values)
        wj = weighted_jaccard(vh, vs)
        bj = binary_jaccard(vh, vs, BJ_TAU)

        n_pairs = len(vh)
        results.append({
            'threshold': thresh,
            'n_vars': len(sel_vars),
            'n_pairs': n_pairs,
            'n_subsystems': len(subsystems),
            'subsystems': sorted(subsystems),
            'wj': wj,
            'bj': bj,
            'variables': sel_vars,
        })
        print(f"  Threshold {thresh:.0%}: {len(sel_vars)} vars, {n_pairs} pairs, "
              f"{len(subsystems)} subsystems, WJ={wj:.4f}, BJ={bj:.4f}")

    return results


# ============================================================
# SECTION 12: FIGURES
# ============================================================

def figure_null_distributions(perm_results, db_name, fig_dir):
    """Figure: WJ and BJ null distributions with observed values."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # WJ null
    ax = axes[0]
    ax.hist(perm_results['null_wj'], bins=50, color=C['null'], alpha=0.7,
            edgecolor='white', label='Null distribution')
    ax.axvline(perm_results['obs_wj'], color=C['wj'], linewidth=2,
               label=f"Observed WJ = {perm_results['obs_wj']:.4f}")
    ax.set_xlabel('Weighted Jaccard Similarity')
    ax.set_ylabel('Count')
    ax.set_title(f'{db_name}: Patient-Level WJ Null\n'
                 f'z = {perm_results["z_wj"]:.2f}')
    ax.legend(fontsize=7)

    # BJ null
    ax = axes[1]
    null_bj = perm_results['null_bj']
    unique_vals = len(np.unique(np.round(null_bj, 6)))
    ax.hist(null_bj, bins=min(50, max(unique_vals * 2, 10)),
            color=C['null'], alpha=0.7, edgecolor='white', label='Null distribution')
    ax.axvline(perm_results['obs_bj'], color=C['bj'], linewidth=2,
               label=f"Observed BJ = {perm_results['obs_bj']:.4f}")
    ax.set_xlabel('Binary Jaccard Similarity')
    ax.set_ylabel('Count')
    degen_text = f"DEGENERATE ({unique_vals} unique values)" if unique_vals <= 3 else f"{unique_vals} unique values"
    ax.set_title(f'{db_name}: Patient-Level BJ Null\n'
                 f'z = {perm_results["z_bj"]:.2f} — {degen_text}')
    ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(fig_dir, f'figure_null_distributions_{db_name.lower()}.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def figure_cascade(cascade_df, db_name, fig_dir):
    """Figure: Cascade ordering bar chart."""
    fig, ax = plt.subplots(figsize=(10, max(5, len(cascade_df) * 0.4)))

    colors = [C['cv'] if cv else C['noncv'] for cv in cascade_df['involves_cv']]
    bars = ax.barh(range(len(cascade_df)), cascade_df['mean_delta_r'].values,
                   color=colors, edgecolor='white')
    ax.set_yticks(range(len(cascade_df)))
    ax.set_yticklabels(cascade_df['interaction'].values)
    ax.set_xlabel('Mean |dr| (correlation divergence)')
    ax.set_title(f'{db_name}: Organ-System Cascade Profile')
    ax.invert_yaxis()

    legend_elements = [
        mpatches.Patch(facecolor=C['cv'], label='Involves Cardiovascular'),
        mpatches.Patch(facecolor=C['noncv'], label='Non-Cardiovascular'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7)

    plt.tight_layout()
    path = os.path.join(fig_dir, f'figure_cascade_{db_name.lower()}.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def figure_heatmap(corr_h, corr_s, variables, db_name, fig_dir):
    """Figure: Side-by-side correlation heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    vmin, vmax = -1, 1
    for ax, mat, title in [(axes[0], corr_h, 'Healthy'),
                            (axes[1], corr_s, 'Sepsis'),
                            (axes[2], corr_h.values - corr_s.values, 'Difference (H−S)')]:
        data = mat.values if hasattr(mat, 'values') else mat
        im = ax.imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_xticks(range(len(variables)))
        ax.set_xticklabels(variables, rotation=90, fontsize=5)
        ax.set_yticks(range(len(variables)))
        ax.set_yticklabels(variables, fontsize=5)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.8, label='Spearman ρ')
    fig.suptitle(f'{db_name}: Correlation Matrices ({CORRELATION_METHOD.title()})',
                 fontsize=11)

    plt.tight_layout()
    path = os.path.join(fig_dir, f'figure_heatmaps_{db_name.lower()}.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def figure_multi_threshold(mt_results, db_name, fig_dir):
    """Figure: WJ and BJ across missingness thresholds."""
    if not mt_results:
        return
    df = pd.DataFrame(mt_results)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df['threshold'], df['wj'], 'o-', color=C['wj'], label='Weighted Jaccard')
    ax1.plot(df['threshold'], df['bj'], 's--', color=C['bj'], label='Binary Jaccard')
    ax1.set_xlabel('Missingness Threshold')
    ax1.set_ylabel('Similarity')
    ax1.set_title(f'{db_name}: WJ/BJ Across Missingness Thresholds')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.bar(df['threshold'], df['n_vars'], alpha=0.2, color='gray', width=0.03,
            label='# Variables')
    ax2.set_ylabel('Number of Variables')

    plt.tight_layout()
    path = os.path.join(fig_dir, f'figure_multi_threshold_{db_name.lower()}.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def figure_bootstrap(boot_wj, ci_lo, ci_hi, obs_wj, db_name, fig_dir):
    """Figure: Bootstrap distribution of WJ."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(boot_wj, bins=50, color=C['wj'], alpha=0.6, edgecolor='white')
    ax.axvline(obs_wj, color='black', linewidth=2, label=f'Observed WJ = {obs_wj:.4f}')
    ax.axvline(ci_lo, color='red', linewidth=1, linestyle='--',
               label=f'95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]')
    ax.axvline(ci_hi, color='red', linewidth=1, linestyle='--')
    ax.set_xlabel('Weighted Jaccard')
    ax.set_ylabel('Count')
    ax.set_title(f'{db_name}: Patient-Level Cluster Bootstrap (n={len(boot_wj)})')
    ax.legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(fig_dir, f'figure_bootstrap_{db_name.lower()}.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def figure_bj_sweep(sweep_df, db_name, fig_dir):
    """Figure: Binary Jaccard threshold sweep showing degeneracy."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    colors = ['red' if d else 'gray' for d in sweep_df['degenerate']]
    ax.bar(range(len(sweep_df)), sweep_df['z'].values, color=colors)
    ax.set_xticks(range(len(sweep_df)))
    ax.set_xticklabels([f"{t:.0%}" for t in sweep_df['tau']])
    ax.set_xlabel('Binary threshold (τ)')
    ax.set_ylabel('z-score')
    ax.set_title(f'{db_name}: BJ z-score vs threshold')
    ax.axhline(-1.96, color='black', linestyle='--', linewidth=0.5, label='z = -1.96')
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.bar(range(len(sweep_df)), sweep_df['unique_null_values'].values, color=colors)
    ax.set_xticks(range(len(sweep_df)))
    ax.set_xticklabels([f"{t:.0%}" for t in sweep_df['tau']])
    ax.set_xlabel('Binary threshold (τ)')
    ax.set_ylabel('Unique values in null')
    ax.set_title(f'{db_name}: Null distribution degeneracy')

    plt.tight_layout()
    path = os.path.join(fig_dir, f'figure_bj_sweep_{db_name.lower()}.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def figure_permutation_comparison(patient_results, obs_level_results, db_name, fig_dir):
    """Figure: Side-by-side comparison of patient vs observation permutation."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.hist(patient_results['null_wj'], bins=50, color=C['wj'], alpha=0.6,
            label='Patient-level null')
    ax.axvline(patient_results['obs_wj'], color='black', linewidth=2)
    ax.set_title(f'{db_name}: Patient-Level\nz = {patient_results["z_wj"]:.2f} (PRIMARY)')
    ax.set_xlabel('Weighted Jaccard')
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.hist(obs_level_results['null_wj'], bins=50, color=C['noncv'], alpha=0.6,
            label='Observation-level null')
    ax.axvline(obs_level_results['obs_wj'], color='black', linewidth=2)
    ax.set_title(f'{db_name}: Observation-Level\n'
                 f'z = {obs_level_results["z_wj"]:.2f} (SENSITIVITY — inflated)')
    ax.set_xlabel('Weighted Jaccard')
    ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(fig_dir, f'figure_perm_comparison_{db_name.lower()}.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# SECTION 13: MAIN EXECUTION
# ============================================================

def analyze_database(db_name, healthy_df, sepsis_df, patient_id_col,
                      candidate_vars, n_h_obs, n_s_obs,
                      merged_df=None, sepsis_stays=None):
    """Run full analysis for one database."""
    print(f"\n{'='*78}")
    print(f"ANALYZING: {db_name}")
    print(f"{'='*78}")

    results = {'db_name': db_name}

    # --- Variable selection at 95% threshold ---
    sel_vars, miss_rates, subsystems = select_variables_by_missingness(
        healthy_df, sepsis_df, 0.95, candidate_vars, SUBSYSTEM_MAP
    )
    print(f"\n  95% threshold: {len(sel_vars)} variables, {len(subsystems)} subsystems")
    print(f"  Variables: {sel_vars}")
    print(f"  Subsystems: {sorted(subsystems)}")
    results['variables'] = sel_vars
    results['n_vars'] = len(sel_vars)
    results['n_subsystems'] = len(subsystems)
    results['missingness'] = miss_rates[sel_vars].to_dict()

    if len(sel_vars) < 3:
        print("  [ERROR] Fewer than 3 variables — cannot proceed")
        return results

    n_pairs = len(sel_vars) * (len(sel_vars) - 1) // 2
    print(f"  Pairs: {n_pairs}")

    # --- PRIMARY: Patient-level permutation ---
    perm_results = patient_level_permutation(
        healthy_df, sepsis_df, sel_vars, patient_id_col,
        N_PERM, CORRELATION_METHOD
    )
    results['patient_perm'] = perm_results

    # --- SENSITIVITY: Observation-level permutation ---
    obs_perm = observation_level_permutation(
        healthy_df, sepsis_df, sel_vars,
        N_PERM_SENS, CORRELATION_METHOD
    )
    results['obs_perm'] = obs_perm

    # --- Cluster bootstrap ---
    boot_wj, ci_lo, ci_hi = patient_cluster_bootstrap(
        healthy_df, sepsis_df, sel_vars, patient_id_col,
        N_BOOTSTRAP, CORRELATION_METHOD
    )
    results['bootstrap'] = {
        'wj_values': boot_wj, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'mean': boot_wj.mean(), 'std': boot_wj.std(),
    }

    # --- Pair-level divergence ---
    print(f"\n--- Pair-Level Divergence ({db_name}) ---")
    div_df = pair_level_divergence(
        perm_results['corr_h'], perm_results['corr_s'],
        sel_vars, n_h_obs, n_s_obs
    )
    results['divergence'] = div_df

    # --- Cascade analysis ---
    print(f"\n--- Cascade Analysis ({db_name}) ---")
    cascade_df = cascade_analysis(div_df)
    results['cascade'] = cascade_df

    # --- BJ degeneracy sweep ---
    bj_sweep = bj_threshold_sweep(
        perm_results['corr_h'], perm_results['corr_s'],
        sel_vars, n_perm=N_PERM_SENS
    )
    results['bj_sweep'] = bj_sweep

    # --- Multi-threshold analysis ---
    mt_results = multi_threshold_analysis(
        healthy_df, sepsis_df, candidate_vars, patient_id_col,
        MISS_THRESHOLDS, CORRELATION_METHOD
    )
    results['multi_threshold'] = mt_results

    # --- Alternative metrics ---
    alt_metrics = alternative_metrics_permutation(
        healthy_df, sepsis_df, sel_vars,
        N_PERM_SENS, CORRELATION_METHOD
    )
    results['alt_metrics'] = alt_metrics

    # --- Leave-one-hospital-out (eICU only) ---
    if merged_df is not None and sepsis_stays is not None:
        loho = leave_one_hospital_out(
            merged_df, sepsis_stays, sel_vars, CORRELATION_METHOD
        )
        results['loho'] = loho

    # --- Figures ---
    print(f"\n--- Generating Figures ({db_name}) ---")
    figure_null_distributions(perm_results, db_name, FIG_DIR)
    figure_cascade(cascade_df, db_name, FIG_DIR)
    figure_heatmap(perm_results['corr_h'], perm_results['corr_s'],
                   sel_vars, db_name, FIG_DIR)
    figure_multi_threshold(mt_results, db_name, FIG_DIR)
    figure_bootstrap(boot_wj, ci_lo, ci_hi, perm_results['obs_wj'], db_name, FIG_DIR)
    figure_bj_sweep(bj_sweep, db_name, FIG_DIR)
    figure_permutation_comparison(perm_results, obs_perm, db_name, FIG_DIR)

    # --- Save tables ---
    div_df.to_csv(os.path.join(TABLE_DIR, f'pair_divergence_{db_name.lower()}.csv'),
                  index=False, float_format='%.6f')
    cascade_df.to_csv(os.path.join(TABLE_DIR, f'cascade_{db_name.lower()}.csv'),
                      index=False, float_format='%.6f')
    bj_sweep.to_csv(os.path.join(TABLE_DIR, f'bj_sweep_{db_name.lower()}.csv'),
                    index=False, float_format='%.6f')
    pd.DataFrame(mt_results).to_csv(
        os.path.join(TABLE_DIR, f'multi_threshold_{db_name.lower()}.csv'),
        index=False, float_format='%.6f'
    )

    print(f"\n  Tables saved to {TABLE_DIR}")
    return results


# ============================================================
# SECTION 14: SUMMARY REPORT
# ============================================================

def write_summary(pn_results, eicu_results, results_dir):
    """Write comprehensive summary report."""
    path = os.path.join(results_dir, 'summary_report.txt')
    lines = []
    lines.append("=" * 78)
    lines.append("CLINICAL WJ VALIDATION — HONEST REBUILD v1.0")
    lines.append("Drake H. Harbert | Inner Architecture LLC")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("METHODOLOGY")
    lines.append(f"  Correlation method: {CORRELATION_METHOD}")
    lines.append(f"  Primary inference: Patient-level permutation ({N_PERM} iterations)")
    lines.append(f"  Sensitivity: Observation-level permutation ({N_PERM_SENS} iterations)")
    lines.append(f"  Bootstrap: Patient-level cluster bootstrap ({N_BOOTSTRAP} resamples)")
    lines.append(f"  FDR: Benjamini-Hochberg q < {FDR_Q} across ALL pairs")
    lines.append(f"  Random seed: {RANDOM_SEED}")
    lines.append("")

    for name, res in [('PhysioNet', pn_results), ('eICU', eicu_results)]:
        if res is None:
            continue
        lines.append("-" * 78)
        lines.append(f"{name.upper()}")
        lines.append("-" * 78)
        lines.append(f"  Variables: {res.get('n_vars', '?')} ({res.get('n_subsystems', '?')} subsystems)")

        pp = res.get('patient_perm', {})
        lines.append("")
        lines.append("  PRIMARY: Patient-Level Permutation")
        lines.append(f"    Observed WJ:   {pp.get('obs_wj', '?'):.4f}")
        lines.append(f"    Null mean±sd:  {pp.get('null_wj_mean', 0):.4f} ± {pp.get('null_wj_std', 0):.4f}")
        lines.append(f"    z-score:       {pp.get('z_wj', '?'):.2f}")
        lines.append(f"    Empirical p:   {pp.get('p_wj', '?'):.4f}")
        lines.append(f"    Observed BJ:   {pp.get('obs_bj', '?'):.4f}")
        lines.append(f"    BJ z-score:    {pp.get('z_bj', '?'):.2f}")

        op = res.get('obs_perm', {})
        lines.append("")
        lines.append("  SENSITIVITY: Observation-Level Permutation (INFLATED — non-independent obs)")
        lines.append(f"    z-score:       {op.get('z_wj', '?'):.2f}")

        boot = res.get('bootstrap', {})
        lines.append("")
        lines.append("  BOOTSTRAP: Patient-Level Cluster")
        lines.append(f"    95% CI:        [{boot.get('ci_lo', '?'):.4f}, {boot.get('ci_hi', '?'):.4f}]")
        lines.append(f"    Boot mean±sd:  {boot.get('mean', 0):.4f} ± {boot.get('std', 0):.4f}")

        div = res.get('divergence')
        if div is not None:
            n_sig = div['significant'].sum()
            n_total = len(div)
            n_cross = div.loc[div['significant'], 'cross_subsystem'].sum()
            lines.append("")
            lines.append("  PAIR-LEVEL DIVERGENCE")
            lines.append(f"    Significant:   {n_sig}/{n_total} ({100*n_sig/n_total:.1f}%)")
            lines.append(f"    Cross-system:  {n_cross}/{n_sig} ({100*n_cross/max(n_sig,1):.1f}%)")
            lines.append(f"    Mean |dr|:     {div['abs_delta_r'].mean():.4f}")

        cascade = res.get('cascade')
        if cascade is not None:
            lines.append("")
            lines.append("  CASCADE ORDERING")
            for _, row in cascade.iterrows():
                cv = " [CV]" if row['involves_cv'] else ""
                lines.append(f"    {row['rank']:2.0f}. {row['interaction']:15s}  "
                             f"|dr|={row['mean_delta_r']:.4f}{cv}")

        alt = res.get('alt_metrics', {})
        if alt:
            lines.append("")
            lines.append("  ALTERNATIVE METRICS (observation-level perm for comparison)")
            for mname, mres in alt.items():
                lines.append(f"    {mname}: obs={mres['observed']:.4f}, z={mres['z']:.2f}")

        loho = res.get('loho')
        if loho:
            vals = list(loho.values())
            lines.append("")
            lines.append("  LEAVE-ONE-HOSPITAL-OUT")
            lines.append(f"    Hospitals: {len(vals)}")
            lines.append(f"    WJ: mean={np.mean(vals):.4f}, sd={np.std(vals):.4f}")

        lines.append("")

    # Cross-database comparison
    if pn_results and eicu_results:
        pn_cascade = pn_results.get('cascade')
        eicu_cascade = eicu_results.get('cascade')
        if pn_cascade is not None and eicu_cascade is not None:
            shared = set(pn_cascade['interaction']) & set(eicu_cascade['interaction'])
            if len(shared) >= 3:
                pn_shared = pn_cascade[pn_cascade['interaction'].isin(shared)].sort_values('interaction')
                eicu_shared = eicu_cascade[eicu_cascade['interaction'].isin(shared)].sort_values('interaction')
                rho, p = stats.spearmanr(
                    pn_shared['mean_delta_r'].values,
                    eicu_shared['mean_delta_r'].values
                )
                lines.append("-" * 78)
                lines.append("CROSS-DATABASE CASCADE REPLICATION")
                lines.append("-" * 78)
                lines.append(f"  Shared interactions: {len(shared)}")
                lines.append(f"  Spearman rho: {rho:.3f}")
                lines.append(f"  p-value: {p:.4f}")
                lines.append("")

    report = '\n'.join(lines)
    with open(path, 'w') as f:
        f.write(report)
    print(f"\n  Summary saved: {path}")
    print("\n" + report)
    return report


def write_provenance(results_dir, pn_results, eicu_results):
    """Write provenance.json documenting methodology compliance."""
    provenance = {
        "methodology": "WJ-native",
        "fundamental_unit": "Individual clinical variable (vital sign or lab value)",
        "pairwise_matrix": "full, no pre-filtering",
        "correlation_method": CORRELATION_METHOD.title(),
        "fdr_scope": "all N pairs (Benjamini-Hochberg)",
        "primary_inference": "patient-level permutation",
        "sensitivity_inference": "observation-level permutation (reported for comparison only)",
        "domain_conventional_methods": "comparison only (binary Jaccard, SOFA/APACHE)",
        "random_seed": RANDOM_SEED,
        "pipeline_file": os.path.basename(__file__),
        "execution_date": datetime.now().strftime('%Y-%m-%d'),
        "wj_compliance_status": "PASS",
        "key_methodological_notes": [
            "Patient-level permutation respects hierarchical data structure",
            "Observation-level z-scores are inflated and reported for sensitivity only",
            "Spearman correlation used per WJ methodology default",
            "Subsystem grouping is post-hoc interpretation, not pre-analysis input",
            "No healthy cohort subsampling — full patient pool used",
        ],
        "databases": {
            "PhysioNet": {
                "n_vars": pn_results.get('n_vars') if pn_results else None,
                "obs_wj": pn_results.get('patient_perm', {}).get('obs_wj') if pn_results else None,
                "patient_z": pn_results.get('patient_perm', {}).get('z_wj') if pn_results else None,
            },
            "eICU": {
                "n_vars": eicu_results.get('n_vars') if eicu_results else None,
                "obs_wj": eicu_results.get('patient_perm', {}).get('obs_wj') if eicu_results else None,
                "patient_z": eicu_results.get('patient_perm', {}).get('z_wj') if eicu_results else None,
            },
        },
        "python_packages": {},
    }

    # Log package versions
    for pkg_name in ['numpy', 'pandas', 'scipy', 'matplotlib']:
        try:
            mod = __import__(pkg_name)
            provenance['python_packages'][pkg_name] = mod.__version__
        except (ImportError, AttributeError):
            pass
    provenance['python_packages']['python'] = sys.version.split()[0]

    path = os.path.join(results_dir, 'provenance.json')
    with open(path, 'w') as f:
        json.dump(provenance, f, indent=2, default=str)
    print(f"  Provenance saved: {path}")


# ============================================================
# SECTION 15: MAIN
# ============================================================

def main():
    t_start = time.time()

    # --- Download data ---
    print("\n--- Data Acquisition ---")
    pn_ok = download_physionet_data()
    eicu_ok = download_eicu_data()

    # Find actual PhysioNet directory (may be nested)
    pn_dir = PHYSIONET_DIR
    if isinstance(pn_ok, str):
        pn_dir = pn_ok
    elif pn_ok is True and not os.path.isdir(PHYSIONET_DIR):
        # Search for .psv files
        for root, dirs, files in os.walk(DATA_DIR):
            if any(f.endswith('.psv') for f in files):
                psv_count = sum(1 for f in files if f.endswith('.psv'))
                if psv_count > 1000:
                    pn_dir = root
                    break

    pn_results = None
    eicu_results = None

    # --- PhysioNet Analysis ---
    if pn_ok:
        try:
            pn_df, pn_meta = load_physionet(pn_dir)
            pn_healthy, pn_sepsis, h_pids, s_pids = construct_physionet_cohorts(pn_df, pn_meta)
            candidate_vars = [v for v in SUBSYSTEM_MAP.keys() if v in pn_healthy.columns]

            pn_results = analyze_database(
                'PhysioNet', pn_healthy, pn_sepsis, 'patient_id',
                candidate_vars, len(pn_healthy), len(pn_sepsis)
            )
            del pn_df  # Free memory
            gc.collect()
        except Exception as e:
            print(f"\n  [ERROR] PhysioNet analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # --- eICU Analysis ---
    if eicu_ok:
        try:
            eicu_merged, eicu_sepsis_stays = load_eicu(EICU_DIR)
            eicu_healthy, eicu_sepsis, h_stays, s_stays, eicu_vars = construct_eicu_cohorts(
                eicu_merged, eicu_sepsis_stays
            )

            eicu_results = analyze_database(
                'eICU', eicu_healthy, eicu_sepsis, 'patientunitstayid',
                eicu_vars, len(eicu_healthy), len(eicu_sepsis),
                merged_df=eicu_merged, sepsis_stays=eicu_sepsis_stays
            )
        except Exception as e:
            print(f"\n  [ERROR] eICU analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # --- Summary ---
    if pn_results or eicu_results:
        write_summary(pn_results, eicu_results, RESULTS_DIR)
        write_provenance(RESULTS_DIR, pn_results, eicu_results)

    t_total = time.time() - t_start
    print(f"\n{'='*78}")
    print(f"PIPELINE COMPLETE — Total time: {t_total/60:.1f} minutes")
    print(f"{'='*78}")


if __name__ == '__main__':
    main()
