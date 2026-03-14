#!/usr/bin/env python3
"""
Robustness analyses for Clinical WJ Validation.
Quick computations from observed correlation matrices — no permutation loops.

Author: Drake H. Harbert (D.H.H.)
Affiliation: Inner Architecture LLC, Canton, OH
ORCID: 0009-0007-7740-3616
"""

import os, sys, time, warnings, json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist
import glob as globmod

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore', category=RuntimeWarning)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r"C:\Users\nwhar\repos\clinical-wj-data"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")

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

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9, 'axes.titlesize': 10,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})
C = {'wj': '#2166AC', 'bj': '#B2182B', 'null': '#999999',
     'cv': '#4393C3', 'noncv': '#F4A582'}


def upper_tri(mat):
    m = mat.values if hasattr(mat, 'values') else mat
    return m[np.triu_indices(m.shape[0], k=1)]

def weighted_jaccard(a, b):
    aa, bb = np.abs(a), np.abs(b)
    d = np.sum(np.maximum(aa, bb))
    return float(np.sum(np.minimum(aa, bb)) / d) if d > 0 else 0.0

def binary_jaccard(a, b, top_pct=0.05):
    n = len(a)
    k = max(1, int(n * top_pct))
    ta = set(np.argsort(np.abs(a))[-k:])
    tb = set(np.argsort(np.abs(b))[-k:])
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union > 0 else 0.0

def rv_coefficient(a, b):
    ac, bc = a - np.mean(a), b - np.mean(b)
    d = np.sqrt(np.trace(ac @ ac.T) * np.trace(bc @ bc.T))
    return np.trace(ac @ bc.T) / d if d > 0 else 0.0

def cosine_similarity(a, b):
    return 1.0 - cosine_dist(a, b)

def frobenius_distance(a, b):
    return np.linalg.norm(a - b, 'fro')

def benjamini_hochberg(pvals, q=0.05):
    p = np.array(pvals); n = len(p)
    si = np.argsort(p); sp = p[si]; qv = np.zeros(n)
    qv[si[-1]] = sp[-1]
    for i in range(n - 2, -1, -1):
        qv[si[i]] = min(sp[i] * n / (i + 1), qv[si[i + 1]])
    return qv, qv < q

def fisher_z_test(r1, r2, n1, n2):
    r1c, r2c = np.clip(r1, -0.9999, 0.9999), np.clip(r2, -0.9999, 0.9999)
    z1, z2 = np.arctanh(r1c), np.arctanh(r2c)
    se = np.sqrt(1.0 / max(n1-3,1) + 1.0 / max(n2-3,1))
    if se <= 0: return 0.0, 1.0
    return (z1-z2)/se, 2*stats.norm.sf(abs((z1-z2)/se))

def load_csv_flexible(directory, basename):
    for ext in ['.csv', '.csv.gz']:
        path = os.path.join(directory, basename + ext)
        if os.path.exists(path): return pd.read_csv(path)
    raise FileNotFoundError(f"Cannot find {basename} in {directory}")


# ============================================================
# DATA LOADING (reused from postprocess)
# ============================================================

def load_physionet():
    print("Loading PhysioNet...")
    t0 = time.time()
    pn_dir = os.path.join(DATA_DIR, "training_setA")
    psv_files = sorted(globmod.glob(os.path.join(pn_dir, "*.psv")))

    frames = []
    available_vars = None
    patient_meta = {}

    for i, fp in enumerate(psv_files):
        pid = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp, sep='|')
        if available_vars is None:
            available_vars = [v for v in SUBSYSTEM_MAP.keys() if v in df.columns]

        has_sepsis = int(df['SepsisLabel'].max()) if 'SepsisLabel' in df.columns else 0
        onset_idx = df['SepsisLabel'].idxmax() if has_sepsis else None
        patient_meta[pid] = {'has_sepsis': has_sepsis, 'stay_hours': len(df),
                              'onset_idx': onset_idx}

        obs = df.iloc[:onset_idx].copy() if has_sepsis and onset_idx else df.copy()
        if len(obs) > 0:
            keep = [v for v in available_vars if v in obs.columns]
            obs = obs[keep].copy()
            obs['patient_id'] = pid
            frames.append(obs)
        if (i+1) % 5000 == 0:
            print(f"    {i+1:,}/{len(psv_files):,}...")

    full_df = pd.concat(frames, ignore_index=True)
    clinical_vars = [v for v in SUBSYSTEM_MAP.keys() if v in full_df.columns]

    h_pids = [p for p, m in patient_meta.items()
              if not m['has_sepsis'] and m['stay_hours'] >= 24]
    s_pids = [p for p, m in patient_meta.items()
              if m['has_sepsis'] and (m['onset_idx'] or 0) >= 12]

    full_df[clinical_vars] = full_df.groupby('patient_id')[clinical_vars].ffill()
    h_df = full_df[full_df['patient_id'].isin(h_pids)]
    s_df = full_df[full_df['patient_id'].isin(s_pids)]
    print(f"  Done in {time.time()-t0:.0f}s")
    return h_df, s_df, clinical_vars, patient_meta


def load_eicu():
    print("Loading eICU...")
    t0 = time.time()
    eicu_dir = os.path.join(DATA_DIR, "eicu-crd-demo")
    patient = load_csv_flexible(eicu_dir, 'patient')
    vitals = load_csv_flexible(eicu_dir, 'vitalPeriodic')
    labs = load_csv_flexible(eicu_dir, 'lab')
    diag = load_csv_flexible(eicu_dir, 'diagnosis')

    sepsis_mask = diag['diagnosisstring'].str.contains('sepsis|septic', case=False, na=False)
    sepsis_stays = set(diag.loc[sepsis_mask, 'patientunitstayid'].unique())

    vital_map = {'heartrate':'HR','systemicsystolic':'SBP','systemicdiastolic':'DBP',
                 'systemicmean':'MAP','respiration':'Resp','temperature':'Temp','sao2':'O2Sat'}
    vitals_std = vitals[['patientunitstayid','observationoffset'] +
                        [c for c in vital_map if c in vitals.columns]].rename(columns=vital_map)
    vitals_std['hour'] = (vitals_std['observationoffset']/60).round().astype(int)
    vitals_hourly = vitals_std.groupby(['patientunitstayid','hour']).mean(
        numeric_only=True).reset_index().drop(columns=['observationoffset'], errors='ignore')

    lab_map = {'BUN':'BUN','creatinine':'Creatinine','potassium':'Potassium',
               'sodium':'Sodium','chloride':'Chloride','calcium':'Calcium',
               'magnesium':'Magnesium','phosphate':'Phosphate','glucose':'Glucose',
               'lactate':'Lactate','Hgb':'Hgb','Hct':'Hct','WBC x 1000':'WBC',
               'platelets x 1000':'Platelets','PTT':'PTT','fibrinogen':'Fibrinogen',
               'AST (SGOT)':'AST','alkaline phos.':'Alkalinephos',
               'total bilirubin':'Bilirubin_total','troponin - I':'TroponinI',
               'pH':'pH','FiO2':'FiO2','paCO2':'PaCO2','paO2':'PaO2',
               'HCO3':'HCO3','Base Excess':'BaseExcess'}
    labs_known = labs[labs['labname'].isin(lab_map.keys())].copy()
    labs_known['labname_std'] = labs_known['labname'].map(lab_map)
    labs_known['hour'] = (labs_known['labresultoffset']/60).round().astype(int)
    labs_known['labresult'] = pd.to_numeric(labs_known['labresult'], errors='coerce')
    labs_pivot = labs_known.pivot_table(index=['patientunitstayid','hour'],
                                        columns='labname_std', values='labresult',
                                        aggfunc='mean').reset_index()

    merged = vitals_hourly.merge(labs_pivot, on=['patientunitstayid','hour'], how='outer')
    merged['is_sepsis'] = merged['patientunitstayid'].isin(sepsis_stays).astype(int)
    if 'hospitalid' in patient.columns:
        merged['hospitalid'] = merged['patientunitstayid'].map(
            patient.set_index('patientunitstayid')['hospitalid'].to_dict())

    clinical_vars = [v for v in SUBSYSTEM_MAP.keys() if v in merged.columns]
    all_stays = merged['patientunitstayid'].unique()
    h_stays = [s for s in all_stays if s not in sepsis_stays]
    s_stays = [s for s in all_stays if s in sepsis_stays]

    merged[clinical_vars] = merged.groupby('patientunitstayid')[clinical_vars].ffill()
    h_df = merged[merged['patientunitstayid'].isin(h_stays)]
    s_df = merged[merged['patientunitstayid'].isin(s_stays)]
    print(f"  Done in {time.time()-t0:.0f}s")
    return h_df, s_df, clinical_vars, merged, sepsis_stays


# ============================================================
# ROBUSTNESS ANALYSIS 1: BJ THRESHOLD SWEEP
# ============================================================

def bj_threshold_sweep(corr_h, corr_s, variables, db_name, n_perm=200):
    """Sweep binary Jaccard threshold to demonstrate structural degeneracy."""
    print(f"\n--- BJ Threshold Sweep ({db_name}) ---")
    vh = upper_tri(corr_h.values)
    vs = upper_tri(corr_s.values)
    n_pairs = len(vh)

    thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
    results = []

    for tau in thresholds:
        obs_bj = binary_jaccard(vh, vs, tau)
        k = max(1, int(n_pairs * tau))

        # Permutation null for BJ at this threshold
        null_bj = np.zeros(n_perm)
        for i in range(n_perm):
            idx = np.random.permutation(len(vh))
            null_bj[i] = binary_jaccard(vh[idx], vs, tau)

        unique_null = len(np.unique(np.round(null_bj, 6)))
        bj_std = null_bj.std()
        z = (obs_bj - null_bj.mean()) / bj_std if bj_std > 0 else 0.0

        results.append({
            'tau': tau, 'k_edges': k, 'n_pairs': n_pairs,
            'obs_bj': obs_bj, 'null_mean': null_bj.mean(),
            'null_std': bj_std, 'z': z,
            'unique_null_values': unique_null,
            'degenerate': unique_null <= 3,
        })
        status = 'DEGENERATE' if unique_null <= 3 else 'ok'
        print(f"  tau={tau:.2f}: k={k}, BJ={obs_bj:.3f}, z={z:.2f}, "
              f"unique_null={unique_null}, {status}")

    sweep_df = pd.DataFrame(results)
    sweep_df.to_csv(os.path.join(TABLE_DIR, f'bj_sweep_{db_name.lower()}.csv'),
                    index=False, float_format='%.6f')

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ['#B2182B' if d else '#999999' for d in sweep_df['degenerate']]

    ax = axes[0]
    ax.bar(range(len(sweep_df)), sweep_df['z'].values, color=colors)
    ax.set_xticks(range(len(sweep_df)))
    ax.set_xticklabels([f"{t:.0%}" for t in sweep_df['tau']])
    ax.set_xlabel('Binary threshold (tau)')
    ax.set_ylabel('z-score')
    ax.set_title(f'{db_name}: BJ z-score vs threshold')
    ax.axhline(-1.96, color='black', linestyle='--', linewidth=0.5, label='z = -1.96')
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.bar(range(len(sweep_df)), sweep_df['unique_null_values'].values, color=colors)
    ax.set_xticks(range(len(sweep_df)))
    ax.set_xticklabels([f"{t:.0%}" for t in sweep_df['tau']])
    ax.set_xlabel('Binary threshold (tau)')
    ax.set_ylabel('Unique values in null')
    ax.set_title(f'{db_name}: Null distribution degeneracy')
    ax.axhline(3, color='black', linestyle='--', linewidth=0.5, label='Degeneracy threshold')
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'figure_bj_sweep_{db_name.lower()}.png'))
    plt.close()
    print(f"  Figure saved")
    return sweep_df


# ============================================================
# ROBUSTNESS ANALYSIS 2: MULTI-THRESHOLD
# ============================================================

def multi_threshold_analysis(h_df, s_df, candidate_vars, db_name):
    """WJ/BJ at multiple missingness thresholds."""
    print(f"\n--- Multi-Threshold Analysis ({db_name}) ---")
    thresholds = [0.50, 0.70, 0.85, 0.90, 0.95]
    results = []

    combined = pd.concat([h_df[candidate_vars], s_df[candidate_vars]])
    miss_rates = combined.isnull().mean()

    for thresh in thresholds:
        sel = [v for v in candidate_vars if miss_rates.get(v, 1.0) < thresh]
        if len(sel) < 3:
            print(f"  {thresh:.0%}: {len(sel)} vars - skipped")
            continue

        subsystems = set(SUBSYSTEM_MAP.get(v, '?') for v in sel)
        corr_h = h_df[sel].corr(method='spearman')
        corr_s = s_df[sel].corr(method='spearman')
        vh, vs = upper_tri(corr_h.values), upper_tri(corr_s.values)
        wj = weighted_jaccard(vh, vs)
        bj = binary_jaccard(vh, vs, 0.05)
        n_pairs = len(vh)

        results.append({
            'threshold': thresh, 'n_vars': len(sel), 'n_pairs': n_pairs,
            'n_subsystems': len(subsystems), 'wj': wj, 'bj': bj,
        })
        print(f"  {thresh:.0%}: {len(sel)} vars, {n_pairs} pairs, "
              f"{len(subsystems)} subsys, WJ={wj:.4f}, BJ={bj:.4f}")

    if not results:
        return None

    mt_df = pd.DataFrame(results)
    mt_df.to_csv(os.path.join(TABLE_DIR, f'multi_threshold_{db_name.lower()}.csv'),
                 index=False, float_format='%.6f')

    # Figure
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(mt_df['threshold'], mt_df['wj'], 'o-', color=C['wj'],
             linewidth=2, markersize=8, label='Weighted Jaccard')
    ax1.plot(mt_df['threshold'], mt_df['bj'], 's--', color=C['bj'],
             linewidth=2, markersize=8, label='Binary Jaccard')
    ax1.set_xlabel('Missingness Threshold')
    ax1.set_ylabel('Similarity')
    ax1.set_title(f'{db_name}: WJ/BJ Across Missingness Thresholds')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.bar(mt_df['threshold'], mt_df['n_vars'], alpha=0.15, color='gray',
            width=0.04, label='# Variables')
    ax2.set_ylabel('Number of Variables', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'figure_multi_threshold_{db_name.lower()}.png'))
    plt.close()
    print(f"  Figure saved")
    return mt_df


# ============================================================
# ROBUSTNESS ANALYSIS 3: CASCADE WITHOUT TROPONIN
# ============================================================

def cascade_without_troponin(h_df, s_df, full_vars, db_name):
    """Re-run cascade excluding TroponinI to check if CV preservation holds
    without the single-variable Cardiac subsystem distortion."""
    print(f"\n--- Cascade Without TroponinI ({db_name}) ---")
    vars_no_trop = [v for v in full_vars if v != 'TroponinI']
    if len(vars_no_trop) == len(full_vars):
        print("  TroponinI not in variable set - skipping")
        return None

    corr_h = h_df[vars_no_trop].corr(method='spearman')
    corr_s = s_df[vars_no_trop].corr(method='spearman')
    vh, vs = upper_tri(corr_h.values), upper_tri(corr_s.values)
    wj = weighted_jaccard(vh, vs)
    print(f"  WJ without TroponinI: {wj:.4f}")

    # Pair-level divergence
    pairs = []
    p_vals = []
    n_h, n_s = len(h_df), len(s_df)
    for i in range(len(vars_no_trop)):
        for j in range(i+1, len(vars_no_trop)):
            vi, vj = vars_no_trop[i], vars_no_trop[j]
            rh, rs = corr_h.loc[vi, vj], corr_s.loc[vi, vj]
            z_stat, p_val = fisher_z_test(rh, rs, n_h, n_s)
            sub_i, sub_j = SUBSYSTEM_MAP.get(vi,'?'), SUBSYSTEM_MAP.get(vj,'?')
            pairs.append({
                'var_i': vi, 'var_j': vj,
                'abs_delta_r': abs(rh - rs),
                'subsystem_i': sub_i, 'subsystem_j': sub_j,
                'cross_subsystem': sub_i != sub_j,
                'interaction': '-'.join(sorted([sub_i, sub_j])),
                'p_value': p_val,
            })
            p_vals.append(p_val)

    qvals, sig_mask = benjamini_hochberg(p_vals, 0.05)
    for i, p in enumerate(pairs):
        p['significant'] = bool(sig_mask[i])

    div_df = pd.DataFrame(pairs)
    cascade = div_df.groupby('interaction').agg(
        mean_delta_r=('abs_delta_r', 'mean'),
        n_pairs=('abs_delta_r', 'count'),
        n_significant=('significant', 'sum'),
        pct_significant=('significant', 'mean'),
    ).reset_index().sort_values('mean_delta_r', ascending=False)
    cascade['rank'] = range(1, len(cascade) + 1)
    cascade['involves_cv'] = cascade['interaction'].str.contains('CV')

    print(f"\n  Cascade without TroponinI:")
    cv_ranks = []
    for _, row in cascade.iterrows():
        cv_tag = " [CV]" if row['involves_cv'] else ""
        if row['involves_cv']:
            cv_ranks.append(row['rank'])
        print(f"    {row['rank']:2.0f}. {row['interaction']:15s}  "
              f"|dr|={row['mean_delta_r']:.4f}  "
              f"{row['n_significant']:.0f}/{row['n_pairs']:.0f} sig{cv_tag}")

    if cv_ranks:
        print(f"\n  CV interaction ranks: {cv_ranks}")
        print(f"  CV mean rank: {np.mean(cv_ranks):.1f} / {len(cascade)}")

    cascade.to_csv(os.path.join(TABLE_DIR, f'cascade_no_troponin_{db_name.lower()}.csv'),
                   index=False, float_format='%.6f')

    # Figure
    fig, ax = plt.subplots(figsize=(10, max(5, len(cascade)*0.4)))
    colors = [C['cv'] if cv else C['noncv'] for cv in cascade['involves_cv']]
    ax.barh(range(len(cascade)), cascade['mean_delta_r'].values, color=colors, edgecolor='white')
    ax.set_yticks(range(len(cascade)))
    ax.set_yticklabels(cascade['interaction'].values)
    ax.set_xlabel('Mean |delta r|')
    ax.set_title(f'{db_name}: Cascade Without TroponinI')
    ax.invert_yaxis()
    legend_elements = [mpatches.Patch(facecolor=C['cv'], label='Involves CV'),
                       mpatches.Patch(facecolor=C['noncv'], label='Non-CV')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'figure_cascade_no_troponin_{db_name.lower()}.png'))
    plt.close()
    print(f"  Figure saved")
    return cascade


# ============================================================
# ROBUSTNESS ANALYSIS 4: ALTERNATIVE METRICS
# ============================================================

def alternative_metrics(corr_h, corr_s, variables, db_name, n_perm=200):
    """Test cosine, RV coefficient, Frobenius under permutation."""
    print(f"\n--- Alternative Metrics ({db_name}) ---")
    vh, vs = upper_tri(corr_h.values), upper_tri(corr_s.values)

    obs_wj = weighted_jaccard(vh, vs)
    obs_cos = cosine_similarity(vh, vs)
    obs_rv = rv_coefficient(corr_h.values, corr_s.values)
    obs_frob = frobenius_distance(corr_h.values, corr_s.values)

    # Permutation null (shuffle upper-tri vector)
    null_cos = np.zeros(n_perm)
    null_rv = np.zeros(n_perm)
    null_frob = np.zeros(n_perm)
    null_wj = np.zeros(n_perm)

    for i in range(n_perm):
        idx = np.random.permutation(len(vh))
        vh_perm = vh[idx]
        null_wj[i] = weighted_jaccard(vh_perm, vs)
        null_cos[i] = cosine_similarity(vh_perm, vs)
        # For RV and Frobenius, reconstruct matrix from permuted upper tri
        mat_perm = np.zeros_like(corr_h.values)
        mat_perm[np.triu_indices(len(variables), k=1)] = vh_perm
        mat_perm = mat_perm + mat_perm.T
        np.fill_diagonal(mat_perm, 1.0)
        null_rv[i] = rv_coefficient(mat_perm, corr_s.values)
        null_frob[i] = frobenius_distance(mat_perm, corr_s.values)

    results = {}
    print(f"  {'Metric':<12} {'Observed':>10} {'Null mean':>10} {'z':>8} {'Detects?':>10}")
    print(f"  {'-'*52}")
    for name, obs, null in [('WJ', obs_wj, null_wj), ('Cosine', obs_cos, null_cos),
                             ('RV', obs_rv, null_rv), ('Frobenius', obs_frob, null_frob)]:
        m, s = null.mean(), null.std()
        z = (obs - m) / s if s > 0 else 0.0
        detects = '|z|>1.96' if abs(z) > 1.96 else 'NO'
        results[name] = {'observed': obs, 'null_mean': m, 'null_std': s, 'z': z}
        print(f"  {name:<12} {obs:>10.4f} {m:>10.4f} {z:>8.2f} {detects:>10}")

    # Save
    pd.DataFrame(results).T.to_csv(
        os.path.join(TABLE_DIR, f'alt_metrics_{db_name.lower()}.csv'), float_format='%.6f')
    print(f"  Table saved")
    return results


# ============================================================
# ROBUSTNESS ANALYSIS 5: LEAVE-ONE-HOSPITAL-OUT (eICU)
# ============================================================

def leave_one_hospital_out(merged_df, sepsis_stays, variables, min_patients=20):
    """Leave-one-hospital-out for eICU."""
    print(f"\n--- Leave-One-Hospital-Out (eICU) ---")
    if 'hospitalid' not in merged_df.columns:
        print("  No hospitalid column - skipping")
        return None

    hosp_counts = merged_df.groupby('hospitalid')['patientunitstayid'].nunique()
    eligible = hosp_counts[hosp_counts >= min_patients].index
    print(f"  Eligible hospitals (>={min_patients} patients): {len(eligible)}")

    loho_wj = {}
    for hosp_id in eligible:
        excl_df = merged_df[merged_df['hospitalid'] != hosp_id]
        h_df = excl_df[~excl_df['patientunitstayid'].isin(sepsis_stays)]
        s_df = excl_df[excl_df['patientunitstayid'].isin(sepsis_stays)]

        if len(h_df) < 100 or len(s_df) < 50:
            continue

        corr_h = h_df[variables].corr(method='spearman')
        corr_s = s_df[variables].corr(method='spearman')
        wj = weighted_jaccard(upper_tri(corr_h.values), upper_tri(corr_s.values))
        loho_wj[hosp_id] = wj

    vals = list(loho_wj.values())
    if vals:
        print(f"  LOHO WJ: mean={np.mean(vals):.4f}, sd={np.std(vals):.4f}, "
              f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]")
        print(f"  Max deviation from full: {max(abs(v - 0.5718) for v in vals):.4f}")

        # Figure
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(vals)), sorted(vals), color=C['wj'], alpha=0.7)
        ax.axhline(0.5718, color='black', linewidth=2, linestyle='--',
                   label=f'Full dataset WJ = 0.5718')
        ax.set_xlabel('Hospital (excluded)')
        ax.set_ylabel('Weighted Jaccard')
        ax.set_title('eICU: Leave-One-Hospital-Out WJ Stability')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'figure_loho_eicu.png'))
        plt.close()
        print(f"  Figure saved")

    pd.DataFrame({'hospitalid': list(loho_wj.keys()),
                  'wj': list(loho_wj.values())}).to_csv(
        os.path.join(TABLE_DIR, 'loho_eicu.csv'), index=False, float_format='%.6f')
    return loho_wj


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()

    # Load data
    pn_h, pn_s, pn_vars, _ = load_physionet()
    eicu_h, eicu_s, eicu_vars, eicu_merged, eicu_sepsis = load_eicu()

    # Correlation matrices (needed for all analyses)
    print("\nComputing correlation matrices...")
    pn_corr_h = pn_h[pn_vars].corr(method='spearman')
    pn_corr_s = pn_s[pn_vars].corr(method='spearman')
    eicu_corr_h = eicu_h[eicu_vars].corr(method='spearman')
    eicu_corr_s = eicu_s[eicu_vars].corr(method='spearman')
    print("  Done")

    # 1. BJ Threshold Sweep
    pn_sweep = bj_threshold_sweep(pn_corr_h, pn_corr_s, pn_vars, 'PhysioNet')
    eicu_sweep = bj_threshold_sweep(eicu_corr_h, eicu_corr_s, eicu_vars, 'eICU')

    # 2. Multi-Threshold
    pn_mt = multi_threshold_analysis(pn_h, pn_s, pn_vars, 'PhysioNet')
    eicu_mt = multi_threshold_analysis(eicu_h, eicu_s, eicu_vars, 'eICU')

    # 3. Cascade Without TroponinI
    pn_notrop = cascade_without_troponin(pn_h, pn_s, pn_vars, 'PhysioNet')
    eicu_notrop = cascade_without_troponin(eicu_h, eicu_s, eicu_vars, 'eICU')

    # 4. Alternative Metrics
    pn_alt = alternative_metrics(pn_corr_h, pn_corr_s, pn_vars, 'PhysioNet')
    eicu_alt = alternative_metrics(eicu_corr_h, eicu_corr_s, eicu_vars, 'eICU')

    # 5. Leave-One-Hospital-Out
    loho = leave_one_hospital_out(eicu_merged, eicu_sepsis, eicu_vars)

    # Cross-database cascade comparison WITHOUT TroponinI
    if pn_notrop is not None and eicu_notrop is not None:
        shared = set(pn_notrop['interaction']) & set(eicu_notrop['interaction'])
        if len(shared) >= 3:
            pn_sh = pn_notrop[pn_notrop['interaction'].isin(shared)].sort_values('interaction')
            eicu_sh = eicu_notrop[eicu_notrop['interaction'].isin(shared)].sort_values('interaction')
            rho, p = stats.spearmanr(pn_sh['mean_delta_r'].values,
                                      eicu_sh['mean_delta_r'].values)
            print(f"\n--- Cross-Database Cascade (No TroponinI) ---")
            print(f"  Shared interactions: {len(shared)}")
            print(f"  Spearman rho: {rho:.3f}, p = {p:.4f}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ROBUSTNESS ANALYSES COMPLETE - {elapsed/60:.1f} minutes")
    print(f"{'='*70}")

    # List all outputs
    print(f"\nOutputs in {RESULTS_DIR}:")
    for root, dirs, files in os.walk(RESULTS_DIR):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            sz = os.path.getsize(fpath)
            print(f"  {os.path.relpath(fpath, RESULTS_DIR):50s} {sz:>8,} bytes")


if __name__ == '__main__':
    main()
