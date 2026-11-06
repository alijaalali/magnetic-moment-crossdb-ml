"""Variance-controlled subgroup diagnosis for the correlated-electron vs.
non-correlated split: MAE/target-std ratio alongside raw MAE and R2, and a
composition-grouped bootstrap confidence interval on the MAE difference.

Input: data/predictions_mp_to_jarvis.csv and data/predictions_jarvis_to_mp.csv.
Output: data/subgroup_variance_controlled.json.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os
ROOT = os.environ.get("MAGMOM_ROOT", os.path.dirname(os.path.abspath(__file__)))

FILES = {
    "mp_to_jarvis": f"{ROOT}/data/predictions_mp_to_jarvis.csv",
    "jarvis_to_mp": f"{ROOT}/data/predictions_jarvis_to_mp.csv",
}

out = {}

for direction, path in FILES.items():
    df = pd.read_csv(path)
    y_col, p_col = "mag_per_atom", "pred_mag_per_atom"
    df["correlated_electron"] = df["correlated_electron"].astype(bool)

    out[direction] = {}
    sub_stats = {}
    for label, mask in [("correlated_electron", df["correlated_electron"]),
                         ("non_correlated", ~df["correlated_electron"])]:
        sub = df[mask]
        y, p = sub[y_col].to_numpy(), sub[p_col].to_numpy()
        mae = float(mean_absolute_error(y, p))
        rmse = float(np.sqrt(mean_squared_error(y, p)))
        r2 = float(r2_score(y, p))
        target_std = float(y.std())
        target_mean = float(y.mean())
        target_median = float(np.median(y))
        frac_magnetic = float(sub["is_magnetic"].mean())
        sub_stats[label] = {
            "n": int(len(sub)),
            "target_mean": target_mean, "target_std": target_std, "target_median": target_median,
            "frac_magnetic": frac_magnetic,
            "mae": mae, "rmse": rmse, "r2": r2,
            "mae_over_target_std": mae / target_std if target_std > 0 else None,
        }
        print(f"[{direction}] {label}: n={sub_stats[label]['n']} "
              f"target_mean={target_mean:.3f} target_std={target_std:.3f} "
              f"frac_magnetic={frac_magnetic:.3f} MAE={mae:.3f} RMSE={rmse:.3f} "
              f"R2={r2:.3f} MAE/std={sub_stats[label]['mae_over_target_std']:.3f}")
    out[direction]["subgroups"] = sub_stats

    # Composition-grouped bootstrap CI on the MAE difference
    # (correlated_electron - non_correlated), resampling whole reduced
    # formulas together. Every reduced_formula appears exactly once per file
    # (n_rows == n_unique_formulas in both directions), so composition-grouped
    # resampling is identical to row-level resampling within each subgroup
    # and is vectorized.
    assert df["reduced_formula"].nunique() == len(df), \
        "expected one row per reduced_formula; grouped-vs-row bootstrap equivalence assumption violated"

    rng = np.random.RandomState(0)
    target_arr = df[y_col].to_numpy()
    pred_arr = df[p_col].to_numpy()
    corr_mask = df["correlated_electron"].to_numpy()
    corr_idx_pool = np.where(corr_mask)[0]
    noncorr_idx_pool = np.where(~corr_mask)[0]

    n_boot = 5000
    n_corr, n_noncorr = len(corr_idx_pool), len(noncorr_idx_pool)
    samp_corr = rng.choice(corr_idx_pool, size=(n_boot, n_corr), replace=True)
    samp_noncorr = rng.choice(noncorr_idx_pool, size=(n_boot, n_noncorr), replace=True)
    mae_corr_boot = np.abs(target_arr[samp_corr] - pred_arr[samp_corr]).mean(axis=1)
    mae_noncorr_boot = np.abs(target_arr[samp_noncorr] - pred_arr[samp_noncorr]).mean(axis=1)
    diffs = mae_corr_boot - mae_noncorr_boot
    ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    excludes_zero = bool(ci[0] > 0 or ci[1] < 0)
    out[direction]["mae_difference_correlated_minus_noncorrelated"] = {
        "mean": float(diffs.mean()), "ci95_composition_grouped_bootstrap": ci,
        "n_boot": int(len(diffs)), "excludes_zero": excludes_zero,
    }
    print(f"[{direction}] MAE difference (correlated - non-correlated): "
          f"mean={diffs.mean():.3f}, 95% CI {ci}, excludes zero={excludes_zero}")

    # Outlier-concentration diagnostic: is the correlated-electron
    # subgroup's excess error broad or driven by a small tail?
    sub = df[df["correlated_electron"]]
    abs_err = (sub[y_col] - sub[p_col]).abs().to_numpy()
    abs_err_sorted = np.sort(abs_err)[::-1]
    n_top10 = max(1, int(0.10 * len(abs_err)))
    frac_err_top10 = float(abs_err_sorted[:n_top10].sum() / abs_err.sum())
    median_ae = float(np.median(abs_err))
    mean_ae = float(abs_err.mean())
    out[direction]["correlated_electron_error_concentration"] = {
        "n": int(len(sub)),
        "median_abs_error": median_ae, "mean_abs_error": mean_ae,
        "frac_total_error_from_top10pct_worst": frac_err_top10,
    }
    print(f"[{direction}] correlated-electron error concentration: "
          f"median|err|={median_ae:.3f} mean|err|={mean_ae:.3f} "
          f"top-10%-worst share of total error={frac_err_top10:.3f}")

with open(f"{ROOT}/data/subgroup_variance_controlled.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved data/subgroup_variance_controlled.json")
