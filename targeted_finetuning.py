"""Targeted vs. random domain-adaptation experiment.

Tests whether knowing the chemistry-based failure mode (correlated-
electron/transition-metal compounds carry most of the MP to JARVIS excess
transfer error) lets a small JARVIS labeling budget be used more
efficiently than random sampling. Protocol: a fixed held-out JARVIS test
set; k labeled JARVIS pool compounds are added to the MP training data
("joint") or used alone ("target-only"); repeated over seeds, mean and
standard deviation reported. Models are retrained from scratch at each k
(not gradient fine-tuning).

Two sampling strategies at each k:
  random   - k compounds drawn uniformly at random from the JARVIS pool.
  targeted - correlated-electron (oxide/fluoride + MP-U-list cation) pool
             compounds are drawn first; non-correlated compounds fill any
             remaining budget.

Evaluated on the full held-out JARVIS test set and on its
correlated-electron subset.

Output: data/targeted_finetuning_results.json.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
RNG = 0
import os
PROJ = Path(os.environ.get("MAGMOM_ROOT", str(Path(__file__).resolve().parent)))
DATA = PROJ / "data"
BEST_CFG = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8)

CORRELATED_ELECTRON_ELEMENTS = {"V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Mo", "W"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def has_correlated_electron_element(formula):
    try:
        els = set(str(e) for e in Composition(formula).elements)
    except Exception:
        return False
    return len(els & CORRELATED_ELECTRON_ELEMENTS) > 0


def main():
    t0 = time.time()
    mp_meta = pd.read_csv(DATA / "mp_meta_agg.csv")
    jv_meta = pd.read_csv(DATA / "jarvis_meta_agg.csv")
    Xmp = pd.read_parquet(DATA / "featurized_mp.parquet")
    Xjv = pd.read_parquet(DATA / "featurized_jarvis.parquet")
    assert len(mp_meta) == len(Xmp) and len(jv_meta) == len(Xjv)

    ymp = mp_meta["mag_per_atom"].to_numpy(dtype=float)
    jv_meta = jv_meta.copy()
    jv_meta["correlated_electron"] = jv_meta["formula"].apply(has_correlated_electron_element)
    log(f"JARVIS correlated-electron compounds: {jv_meta['correlated_electron'].sum()}/{len(jv_meta)}")

    # Fixed held-out JARVIS test set (20%, random; equivalent to
    # composition-grouped since every reduced_formula appears exactly once,
    # see subgroup_variance_controlled.py).
    idx_all = np.arange(len(Xjv))
    idx_pool, idx_test = train_test_split(idx_all, test_size=0.2, random_state=RNG)
    Xjv_pool, Xjv_test = Xjv.iloc[idx_pool].reset_index(drop=True), Xjv.iloc[idx_test].reset_index(drop=True)
    meta_pool = jv_meta.iloc[idx_pool].reset_index(drop=True)
    meta_test = jv_meta.iloc[idx_test].reset_index(drop=True)
    yjv_pool = meta_pool["mag_per_atom"].to_numpy(dtype=float)
    yjv_test = meta_test["mag_per_atom"].to_numpy(dtype=float)
    test_corr_mask = meta_test["correlated_electron"].to_numpy()
    log(f"JARVIS pool={len(Xjv_pool)} (correlated={meta_pool['correlated_electron'].sum()}), "
        f"test={len(Xjv_test)} (correlated={test_corr_mask.sum()})")

    pool_corr_idx = np.where(meta_pool["correlated_electron"].to_numpy())[0]
    pool_noncorr_idx = np.where(~meta_pool["correlated_electron"].to_numpy())[0]
    log(f"Pool composition: {len(pool_corr_idx)} correlated-electron, {len(pool_noncorr_idx)} non-correlated available")

    k_list = [10, 100]
    n_seeds = 3  # each 74k-row joint GBM refit takes ~5-6 minutes, so the
    # number of seeds and k values is kept small to bound total runtime
    strategies = ["random", "targeted"]
    modes = ["joint", "target_only"]

    def sample_indices(strategy, k, seed):
        rng = np.random.RandomState(seed)
        if strategy == "random":
            return rng.choice(len(Xjv_pool), size=min(k, len(Xjv_pool)), replace=False)
        else:  # targeted: correlated-electron pool compounds first
            corr_shuffled = rng.permutation(pool_corr_idx)
            if len(corr_shuffled) >= k:
                return corr_shuffled[:k]
            remaining = k - len(corr_shuffled)
            noncorr_shuffled = rng.permutation(pool_noncorr_idx)
            return np.concatenate([corr_shuffled, noncorr_shuffled[:remaining]])

    def eval_metrics(y_true, y_pred, mask=None):
        if mask is not None:
            y_true, y_pred = y_true[mask], y_pred[mask]
        if len(y_true) < 2:
            return {"n": int(len(y_true)), "mae": None, "r2": None}
        return {"n": int(len(y_true)), "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred))}

    results = {s: {m: {k: {"overall": [], "correlated": []} for k in k_list} for m in modes} for s in strategies}

    for k in k_list:
        for strategy in strategies:
            for seed in range(n_seeds):
                idx = sample_indices(strategy, k, seed)
                Xk, yk = Xjv_pool.iloc[idx], yjv_pool[idx]

                # joint: MP full + k JARVIS rows, retrained from scratch
                Xtr_joint = pd.concat([Xmp, Xk], ignore_index=True)
                ytr_joint = np.concatenate([ymp, yk])
                m_joint = GradientBoostingRegressor(random_state=seed, **BEST_CFG)
                m_joint.fit(Xtr_joint, ytr_joint)
                pred_joint = m_joint.predict(Xjv_test)

                # target-only: k JARVIS rows alone
                m_tonly = GradientBoostingRegressor(random_state=seed, **BEST_CFG)
                m_tonly.fit(Xk, yk)
                pred_tonly = m_tonly.predict(Xjv_test)

                for mode, pred in [("joint", pred_joint), ("target_only", pred_tonly)]:
                    results[strategy][mode][k]["overall"].append(eval_metrics(yjv_test, pred))
                    results[strategy][mode][k]["correlated"].append(eval_metrics(yjv_test, pred, test_corr_mask))
            log(f"  k={k} strategy={strategy} done ({time.time()-t0:.1f}s)")
            # checkpoint raw results after every (k, strategy) batch so an
            # interrupted run keeps completed work
            with open(DATA / "targeted_finetuning_raw_checkpoint.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

    def summarize(entries, field):
        vals = [e[field] for e in entries if e[field] is not None]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_seeds": len(vals)} if vals else None

    summary = {}
    for strategy in strategies:
        summary[strategy] = {}
        for mode in modes:
            summary[strategy][mode] = {}
            for k in k_list:
                ov = results[strategy][mode][k]["overall"]
                co = results[strategy][mode][k]["correlated"]
                summary[strategy][mode][str(k)] = {
                    "overall_mae": summarize(ov, "mae"), "overall_r2": summarize(ov, "r2"),
                    "correlated_mae": summarize(co, "mae"), "correlated_r2": summarize(co, "r2"),
                }

    # Paired comparison (same seed = same sample draw within a strategy):
    # does targeted beat random, on the joint model, for the correlated
    # subgroup.
    paired = {}
    for k in k_list:
        rand_mae = np.array([e["mae"] for e in results["random"]["joint"][k]["correlated"] if e["mae"] is not None])
        targ_mae = np.array([e["mae"] for e in results["targeted"]["joint"][k]["correlated"] if e["mae"] is not None])
        n = min(len(rand_mae), len(targ_mae))
        diffs = targ_mae[:n] - rand_mae[:n]  # negative = targeted better (lower MAE)
        boot_rng = np.random.RandomState(42)
        boot_means = [boot_rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(5000)] if n > 1 else [0.0]
        paired[str(k)] = {
            "mean_mae_diff_targeted_minus_random": float(diffs.mean()) if n else None,
            "ci95_bootstrap_over_seeds": [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))],
            "n_seeds": int(n),
            "targeted_better": bool(diffs.mean() < 0) if n else None,
        }
        log(f"  k={k} paired (targeted-random) correlated-subgroup MAE diff: "
            f"{paired[str(k)]['mean_mae_diff_targeted_minus_random']}, CI {paired[str(k)]['ci95_bootstrap_over_seeds']}")

    out = {"summary": summary, "paired_targeted_vs_random_correlated_subgroup": paired,
           "k_list": k_list, "n_seeds": n_seeds,
           "pool_size": len(Xjv_pool), "pool_correlated": int(len(pool_corr_idx)),
           "test_size": len(Xjv_test), "test_correlated": int(test_corr_mask.sum())}
    with open(DATA / "targeted_finetuning_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"\nSaved data/targeted_finetuning_results.json ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
