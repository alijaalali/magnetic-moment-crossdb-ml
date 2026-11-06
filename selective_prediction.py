"""Selective prediction (abstention) for the magnetic-moment cross-database
transfer study, MP to JARVIS direction (the larger collapse, R2~0.01-0.02).

A bootstrap ensemble of N_ENS GBMs is trained on bootstrap-resampled MP
data; the ensemble mean is the prediction and the ensemble standard
deviation is the uncertainty. The script tests (a) whether predicted
uncertainty ranks the actual error (Spearman), (b) whether uncertainty is
higher for the correlated-electron subgroup, and (c) whether an abstention
policy based on predicted uncertainty preferentially flags
correlated-electron compounds and improves accuracy on the retained set.

Ensemble size: N_ENS=20. With 74,461 MP training rows each GBM fit is
expensive, so the members are fitted in parallel with joblib (independent
fits).

Outputs: data/selective_prediction_results.json and
data/selective_prediction_curve.csv.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
RNG = 0
import os
ROOT = os.environ.get("MAGMOM_ROOT", os.path.dirname(os.path.abspath(__file__)))
N_ENS = 20
BEST_CFG = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8)

CORRELATED_ELECTRON_ELEMENTS = {"V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Mo", "W"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def has_correlated_electron_element(formula):
    try:
        els = set(str(e) for e in Composition(formula).elements)
    except Exception:
        return None
    return len(els & CORRELATED_ELECTRON_ELEMENTS) > 0


def fit_one(seed, Xmp_arr, ymp_arr, Xjv_arr):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(Xmp_arr), len(Xmp_arr))
    m = GradientBoostingRegressor(random_state=seed, **BEST_CFG)
    m.fit(Xmp_arr[idx], ymp_arr[idx])
    return m.predict(Xjv_arr)


def main():
    t0 = time.time()
    Xmp = pd.read_parquet(f"{ROOT}/data/featurized_mp.parquet")
    Xjv = pd.read_parquet(f"{ROOT}/data/featurized_jarvis.parquet")
    mp_meta = pd.read_csv(f"{ROOT}/data/mp_meta_agg.csv")
    jv_meta = pd.read_csv(f"{ROOT}/data/jarvis_meta_agg.csv")
    feature_cols = list(Xmp.columns)
    Xjv = Xjv[feature_cols]

    ymp = mp_meta["mag_per_atom"].to_numpy(dtype=float)
    yjv = jv_meta["mag_per_atom"].to_numpy(dtype=float)

    if "correlated_electron" not in jv_meta.columns:
        jv_meta["correlated_electron"] = jv_meta["formula"].apply(has_correlated_electron_element)
    corr_jv = jv_meta["correlated_electron"].to_numpy(dtype=bool)

    Xmp_arr = Xmp.to_numpy(dtype=float)
    Xjv_arr = Xjv.to_numpy(dtype=float)

    log(f"Training {N_ENS}-model bootstrap ensemble (MP train, JARVIS predict), parallelized...")
    preds_list = Parallel(n_jobs=4)(
        delayed(fit_one)(seed, Xmp_arr, ymp, Xjv_arr) for seed in range(N_ENS)
    )
    ens_preds = np.stack(preds_list, axis=0)
    mean_pred = ens_preds.mean(axis=0)
    std_pred = ens_preds.std(axis=0)
    log(f"Ensemble done ({time.time()-t0:.1f}s). Mean predicted std={std_pred.mean():.4f}")

    abs_err = np.abs(yjv - mean_pred)

    # (a) ranking signal
    rho, p = spearmanr(std_pred, abs_err)
    log(f"Spearman(predicted std, |error|) = {rho:.3f} (p={p:.2e})")

    # (b) uncertainty by subgroup
    std_corr = std_pred[corr_jv]
    std_noncorr = std_pred[~corr_jv]
    log(f"Median predicted std: correlated-electron={np.median(std_corr):.4f} "
        f"(n={corr_jv.sum()}), non-correlated={np.median(std_noncorr):.4f} (n={(~corr_jv).sum()})")

    # (c) selective-prediction curve
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    order = np.argsort(-std_pred)  # descending uncertainty
    n = len(std_pred)
    curve = []
    for frac in thresholds:
        n_abstain = int(frac * n)
        abstain_idx = set(order[:n_abstain].tolist())
        retain_mask = np.ones(n, dtype=bool)
        if n_abstain > 0:
            retain_mask[order[:n_abstain]] = False
        y_ret, p_ret = yjv[retain_mask], mean_pred[retain_mask]
        mae_ret = float(np.mean(np.abs(y_ret - p_ret)))
        ss_res = float(np.sum((y_ret - p_ret) ** 2))
        ss_tot = float(np.sum((y_ret - y_ret.mean()) ** 2))
        r2_ret = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
        frac_corr_abstained = float((~retain_mask & corr_jv).sum() / max(1, corr_jv.sum()))
        frac_noncorr_abstained = float((~retain_mask & ~corr_jv).sum() / max(1, (~corr_jv).sum()))
        curve.append({
            "abstain_fraction": frac, "n_retained": int(retain_mask.sum()),
            "mae_retained": mae_ret, "r2_retained": r2_ret,
            "frac_correlated_electron_abstained": frac_corr_abstained,
            "frac_non_correlated_abstained": frac_noncorr_abstained,
        })
        log(f"  abstain={frac:.1f}: n_retained={retain_mask.sum()} MAE={mae_ret:.4f} R2={r2_ret} "
            f"frac_corr_abstained={frac_corr_abstained:.3f} frac_noncorr_abstained={frac_noncorr_abstained:.3f}")

    out = {
        "n_ensemble": N_ENS,
        "spearman_std_vs_abs_error": {"rho": float(rho), "p": float(p)},
        "median_std_by_subgroup": {
            "correlated_electron": float(np.median(std_corr)),
            "non_correlated": float(np.median(std_noncorr)),
            "n_correlated_electron": int(corr_jv.sum()),
            "n_non_correlated": int((~corr_jv).sum()),
        },
        "selective_prediction_curve": curve,
    }
    with open(f"{ROOT}/data/selective_prediction_results.json", "w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame(curve).to_csv(f"{ROOT}/data/selective_prediction_curve.csv", index=False)
    log(f"\nSaved data/selective_prediction_results.json, data/selective_prediction_curve.csv. "
        f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
