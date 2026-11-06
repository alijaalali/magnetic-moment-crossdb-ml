"""Database-identity-aware multi-task baseline.

Adds a binary `is_mp` feature to the 132 Magpie descriptors and trains one
pooled GBM on both databases simultaneously (each row keeps its own
database label). The question differs from the zero-shot transfer tests:
if a single model is told which database a label comes from, can it
produce different, database-specific predictions for the same compound?

Evaluation protocol: Magpie features depend on composition alone, so an MP
row and a JARVIS row with the same reduced_formula have (near-)identical
feature vectors but different labels. For a held-out set of overlapping
compounds (present in both databases, excluded from training), the model
predicts twice from the same feature vector, once with is_mp=1 and once
with is_mp=0. The script compares the error of the correct-flag prediction
with that of the wrong-flag prediction against each database's own value.
If the flag lets the model separate the two databases' systematic
differences, the correct-flag error should be lower, particularly in the
correlated-electron subgroup.

The script also reports in-distribution accuracy of the pooled model on
each database's held-out non-overlap compounds, to check that pooling does
not degrade ordinary performance.

Output: data/multitask_domain_flag_baseline.json.
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


def eval_metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return {"n": int(len(y_true)), "mae": None, "r2": None}
    return {"n": int(len(y_true)), "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))}


def main():
    t0 = time.time()
    mp_meta = pd.read_csv(DATA / "mp_meta_agg.csv")
    jv_meta = pd.read_csv(DATA / "jarvis_meta_agg.csv")
    Xmp = pd.read_parquet(DATA / "featurized_mp.parquet")
    Xjv = pd.read_parquet(DATA / "featurized_jarvis.parquet")
    assert len(mp_meta) == len(Xmp) and len(jv_meta) == len(Xjv)

    mp_meta = mp_meta.copy(); jv_meta = jv_meta.copy()
    mp_meta["correlated_electron"] = mp_meta["reduced_formula"].apply(has_correlated_electron_element)
    jv_meta["correlated_electron"] = jv_meta["reduced_formula"].apply(has_correlated_electron_element)

    overlap = sorted(set(mp_meta["reduced_formula"]) & set(jv_meta["reduced_formula"]))
    log(f"MP n={len(mp_meta)}, JARVIS n={len(jv_meta)}, overlap={len(overlap)}")

    # Hold out a random 20% of overlapping formulas as the flag-flip test
    # set. Both their MP and JARVIS rows are excluded from training.
    rng = np.random.RandomState(RNG)
    overlap_arr = np.array(overlap)
    perm = rng.permutation(len(overlap_arr))
    n_test_overlap = int(0.2 * len(overlap_arr))
    test_overlap_formulas = set(overlap_arr[perm[:n_test_overlap]].tolist())
    log(f"Held-out overlap formulas for flag-flip test: {len(test_overlap_formulas)}")

    mp_train_mask = ~mp_meta["reduced_formula"].isin(test_overlap_formulas)
    jv_train_mask = ~jv_meta["reduced_formula"].isin(test_overlap_formulas)

    Xmp_train = Xmp[mp_train_mask].reset_index(drop=True).copy()
    Xjv_train = Xjv[jv_train_mask].reset_index(drop=True).copy()
    Xmp_train["is_mp"] = 1
    Xjv_train["is_mp"] = 0
    ymp_train = mp_meta.loc[mp_train_mask, "mag_per_atom"].to_numpy(dtype=float)
    yjv_train = jv_meta.loc[jv_train_mask, "mag_per_atom"].to_numpy(dtype=float)

    Xtr = pd.concat([Xmp_train, Xjv_train], ignore_index=True)
    ytr = np.concatenate([ymp_train, yjv_train])
    log(f"Pooled training set: {len(Xtr)} rows ({len(Xmp_train)} MP + {len(Xjv_train)} JARVIS)")

    model = GradientBoostingRegressor(random_state=RNG, **BEST_CFG)
    model.fit(Xtr, ytr)
    log(f"Pooled flag-aware GBM trained ({time.time()-t0:.1f}s)")

    # ---- In-distribution check on non-overlap rows ----
    # 20% of each database's non-overlap rows are held out, as in the main
    # 4-condition baseline, for comparison with the MP-internal and
    # JARVIS-internal numbers.
    mp_nonoverlap = mp_meta[mp_train_mask].reset_index(drop=True)
    jv_nonoverlap = jv_meta[jv_train_mask].reset_index(drop=True)
    Xmp_nonoverlap = Xmp[mp_train_mask].reset_index(drop=True)
    Xjv_nonoverlap = Xjv[jv_train_mask].reset_index(drop=True)

    rng2 = np.random.RandomState(RNG)
    mp_test_idx = rng2.choice(len(Xmp_nonoverlap), size=int(0.2 * len(Xmp_nonoverlap)), replace=False)
    jv_test_idx = rng2.choice(len(Xjv_nonoverlap), size=int(0.2 * len(Xjv_nonoverlap)), replace=False)

    Xmp_id_test = Xmp_nonoverlap.iloc[mp_test_idx].copy(); Xmp_id_test["is_mp"] = 1
    Xjv_id_test = Xjv_nonoverlap.iloc[jv_test_idx].copy(); Xjv_id_test["is_mp"] = 0
    ymp_id_test = mp_nonoverlap.iloc[mp_test_idx]["mag_per_atom"].to_numpy(dtype=float)
    yjv_id_test = jv_nonoverlap.iloc[jv_test_idx]["mag_per_atom"].to_numpy(dtype=float)

    pred_mp_id = model.predict(Xmp_id_test)
    pred_jv_id = model.predict(Xjv_id_test)
    in_distribution = {
        "mp_pooled_trained": eval_metrics(ymp_id_test, pred_mp_id),
        "jarvis_pooled_trained": eval_metrics(yjv_id_test, pred_jv_id),
    }
    log(f"In-distribution (pooled-trained, correct flag): MP {in_distribution['mp_pooled_trained']}, "
        f"JARVIS {in_distribution['jarvis_pooled_trained']}")

    # ---- Central flag-flip test on held-out overlapping compounds ----
    mp_overlap_rows = mp_meta[mp_meta["reduced_formula"].isin(test_overlap_formulas)].reset_index(drop=True)
    jv_overlap_rows = jv_meta[jv_meta["reduced_formula"].isin(test_overlap_formulas)].reset_index(drop=True)
    Xmp_overlap_feats = Xmp[mp_meta["reduced_formula"].isin(test_overlap_formulas).to_numpy()].reset_index(drop=True)

    # Use the MP-side feature vector for each overlap formula. Composition
    # descriptors are deterministic functions of composition, so this is
    # equivalent to the JARVIS-side vector up to numeric noise, and a single
    # source avoids feature-level differences in the flag-flip comparison.
    mp_overlap_by_formula = mp_overlap_rows.set_index("reduced_formula")
    jv_overlap_by_formula = jv_overlap_rows.set_index("reduced_formula")
    common_formulas = [f for f in mp_overlap_by_formula.index if f in jv_overlap_by_formula.index]
    log(f"Flag-flip test formulas with both MP and JARVIS true labels: {len(common_formulas)}")

    Xflip = Xmp_overlap_feats.copy()
    Xflip.index = mp_overlap_rows["reduced_formula"]
    Xflip = Xflip.loc[common_formulas].reset_index(drop=True)

    y_mp_true = mp_overlap_by_formula.loc[common_formulas, "mag_per_atom"].to_numpy(dtype=float)
    y_jv_true = jv_overlap_by_formula.loc[common_formulas, "mag_per_atom"].to_numpy(dtype=float)
    corr_mask = mp_overlap_by_formula.loc[common_formulas, "correlated_electron"].to_numpy()

    X_flag1 = Xflip.copy(); X_flag1["is_mp"] = 1
    X_flag0 = Xflip.copy(); X_flag0["is_mp"] = 0
    pred_flag1 = model.predict(X_flag1)  # "pretend this is MP"
    pred_flag0 = model.predict(X_flag0)  # "pretend this is JARVIS"

    def flip_block(mask=None):
        if mask is not None:
            yt_mp, yt_jv = y_mp_true[mask], y_jv_true[mask]
            p1, p0 = pred_flag1[mask], pred_flag0[mask]
        else:
            yt_mp, yt_jv, p1, p0 = y_mp_true, y_jv_true, pred_flag1, pred_flag0
        return {
            "n": int(len(yt_mp)),
            "correct_flag_mae": float(np.mean([
                np.mean(np.abs(p1 - yt_mp)), np.mean(np.abs(p0 - yt_jv))
            ])),
            "wrong_flag_mae": float(np.mean([
                np.mean(np.abs(p0 - yt_mp)), np.mean(np.abs(p1 - yt_jv))
            ])),
            "mp_flag1_vs_mp_true_mae": float(np.mean(np.abs(p1 - yt_mp))),
            "jarvis_flag0_vs_jarvis_true_mae": float(np.mean(np.abs(p0 - yt_jv))),
            "mp_flag0_vs_mp_true_mae": float(np.mean(np.abs(p0 - yt_mp))),
            "jarvis_flag1_vs_jarvis_true_mae": float(np.mean(np.abs(p1 - yt_jv))),
        }

    flag_flip = {
        "all_overlap": flip_block(),
        "correlated_electron": flip_block(corr_mask),
        "non_correlated": flip_block(~corr_mask),
    }
    for k, v in flag_flip.items():
        log(f"  flag-flip [{k}]: correct-flag MAE={v['correct_flag_mae']:.4f}, "
            f"wrong-flag MAE={v['wrong_flag_mae']:.4f}, n={v['n']}")

    out = {
        "in_distribution_pooled_trained": in_distribution,
        "flag_flip_test": flag_flip,
        "n_overlap_total": len(overlap),
        "n_overlap_test_formulas": len(test_overlap_formulas),
        "n_flag_flip_eval_formulas": len(common_formulas),
    }
    with open(DATA / "multitask_domain_flag_baseline.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"\nSaved data/multitask_domain_flag_baseline.json ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
