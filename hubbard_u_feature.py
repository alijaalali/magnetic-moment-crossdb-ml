"""Hubbard-U sensitivity feature experiment.

Builds a feature from the Materials Project GGA+U correction scheme (as
implemented in pymatgen's MPRelaxSet): the per-element U values MP applies
to a fixed list of transition-metal elements when the anion is O or F.
JARVIS's standard dft_3d workflow does not apply +U. Values are read from
the installed pymatgen io/vasp/MPRelaxSet.yaml and are identical for the
F and O anion blocks:
  Co=3.32, Cr=3.7, Fe=5.3, Mn=3.9, Mo=4.38, Ni=6.2, V=3.25, W=6.2 eV.
This INCAR LDAUU list is narrower than the CORRELATED_ELECTRON_ELEMENTS set
used for the subgroup diagnosis, which also includes Cu; the feature is
restricted to elements that receive a Hubbard correction in the standard
MP workflow.

The 132 Magpie descriptors are augmented (not replaced) with two candidate
U-sensitivity features. The GBM configuration selected by the
hyperparameter sweep is retrained, and the 4-condition evaluation and the
variance-controlled correlated-electron subgroup comparison are repeated
to test whether the feature improves cross-database transfer.

Output: data/hubbard_u_feature_results.json.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Composition
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
RNG = 0
import os
PROJ = Path(os.environ.get("MAGMOM_ROOT", str(Path(__file__).resolve().parent)))
DATA = PROJ / "data"
BEST_CFG = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- Read the U-value table from the installed pymatgen ----
import pymatgen
mprelax_path = Path(pymatgen.__path__[0]) / "io" / "vasp" / "MPRelaxSet.yaml"
mprelax = yaml.safe_load(open(mprelax_path))
ldauu = mprelax["INCAR"]["LDAUU"]
assert ldauu["O"] == ldauu["F"], "expected identical U values for O and F anion blocks"
U_TABLE = {k: float(v) for k, v in ldauu["O"].items()}
log(f"Loaded LDAUU table from {mprelax_path}: {U_TABLE}")

U_ANIONS = {"O", "F"}


def u_features(formula):
    """Returns (u_max, u_sum_frac). Both 0 if the compound has no O/F anion
    present (MP only applies the +U correction for oxides/fluorides) or no
    qualifying transition-metal cation."""
    try:
        comp = Composition(formula)
    except Exception:
        return 0.0, 0.0
    el_frac = comp.get_el_amt_dict()
    total = sum(el_frac.values())
    els = set(el_frac.keys())
    if not (els & U_ANIONS):
        return 0.0, 0.0
    qualifying = els & set(U_TABLE.keys())
    if not qualifying:
        return 0.0, 0.0
    u_max = max(U_TABLE[e] for e in qualifying)
    u_sum_frac = sum(U_TABLE[e] * (el_frac[e] / total) for e in qualifying)
    return u_max, u_sum_frac


def load_and_augment(meta_path, formula_col, feat_path):
    meta = pd.read_csv(meta_path)
    X = pd.read_parquet(feat_path)
    assert len(meta) == len(X), f"row mismatch: {meta_path} vs {feat_path}"
    u_max, u_sum = [], []
    for f in meta[formula_col]:
        um, us = u_features(f)
        u_max.append(um)
        u_sum.append(us)
    meta = meta.copy()
    meta["u_max"] = u_max
    meta["u_sum_frac"] = u_sum
    return meta, X


def main():
    t0 = time.time()
    mp_meta, Xmp = load_and_augment(DATA / "mp_meta_agg.csv", "formula_pretty", DATA / "featurized_mp.parquet")
    jv_meta, Xjv = load_and_augment(DATA / "jarvis_meta_agg.csv", "formula", DATA / "featurized_jarvis.parquet")

    n_u_mp = int((mp_meta["u_max"] > 0).sum())
    n_u_jv = int((jv_meta["u_max"] > 0).sum())
    log(f"Compounds with u_max>0 (oxide/fluoride + qualifying TM cation): "
        f"MP {n_u_mp}/{len(mp_meta)} ({100*n_u_mp/len(mp_meta):.1f}%), "
        f"JARVIS {n_u_jv}/{len(jv_meta)} ({100*n_u_jv/len(jv_meta):.1f}%)")

    results = {"u_table": U_TABLE, "n_u_positive_mp": n_u_mp, "n_u_positive_jarvis": n_u_jv}

    for feat_name in ["u_max", "u_sum_frac"]:
        log(f"\n=== Feature formulation: {feat_name} ===")
        Xmp_aug = Xmp.copy()
        Xmp_aug[feat_name] = mp_meta[feat_name].to_numpy()
        Xjv_aug = Xjv.copy()
        Xjv_aug[feat_name] = jv_meta[feat_name].to_numpy()
        feature_cols = list(Xmp_aug.columns)
        Xjv_aug = Xjv_aug[feature_cols]

        ymp = mp_meta["mag_per_atom"].to_numpy(dtype=float)
        yjv = jv_meta["mag_per_atom"].to_numpy(dtype=float)

        # 4-condition table (same protocol as phase1_pipeline.py: full-DB
        # train, zero-shot eval on the other full DB; internal = 80/20 split)
        from sklearn.model_selection import train_test_split
        Xmp_tr, Xmp_te, ymp_tr, ymp_te = train_test_split(Xmp_aug, ymp, test_size=0.2, random_state=RNG)
        Xjv_tr, Xjv_te, yjv_tr, yjv_te = train_test_split(Xjv_aug, yjv, test_size=0.2, random_state=RNG)

        reg_mp_int = GradientBoostingRegressor(random_state=RNG, **BEST_CFG).fit(Xmp_tr, ymp_tr)
        reg_jv_int = GradientBoostingRegressor(random_state=RNG, **BEST_CFG).fit(Xjv_tr, yjv_tr)
        reg_mp_full = GradientBoostingRegressor(random_state=RNG, **BEST_CFG).fit(Xmp_aug, ymp)
        reg_jv_full = GradientBoostingRegressor(random_state=RNG, **BEST_CFG).fit(Xjv_aug, yjv)

        pred_mp_int = reg_mp_int.predict(Xmp_te)
        pred_jv_int = reg_jv_int.predict(Xjv_te)
        pred_mp2jv = reg_mp_full.predict(Xjv_aug)
        pred_jv2mp = reg_jv_full.predict(Xmp_aug)

        cond = {
            "mp_internal": {"n": len(ymp_te), "mae": float(mean_absolute_error(ymp_te, pred_mp_int)), "r2": float(r2_score(ymp_te, pred_mp_int))},
            "jarvis_internal": {"n": len(yjv_te), "mae": float(mean_absolute_error(yjv_te, pred_jv_int)), "r2": float(r2_score(yjv_te, pred_jv_int))},
            "mp_to_jarvis": {"n": len(yjv), "mae": float(mean_absolute_error(yjv, pred_mp2jv)), "r2": float(r2_score(yjv, pred_mp2jv))},
            "jarvis_to_mp": {"n": len(ymp), "mae": float(mean_absolute_error(ymp, pred_jv2mp)), "r2": float(r2_score(ymp, pred_jv2mp))},
        }
        for k, v in cond.items():
            log(f"  {k}: n={v['n']} MAE={v['mae']:.4f} R2={v['r2']:.3f}")

        # Subgroup diagnosis on the zero-shot predictions, same methodology
        # as subgroup_variance_controlled.py
        jv_sub = jv_meta.copy()
        jv_sub["pred_mag_per_atom"] = pred_mp2jv
        jv_sub["correlated_electron"] = jv_sub["u_max"] > 0  # oxide/fluoride + qualifying TM cation (MP U-list definition)
        mp_sub = mp_meta.copy()
        mp_sub["pred_mag_per_atom"] = pred_jv2mp
        mp_sub["correlated_electron"] = mp_sub["u_max"] > 0

        def subgroup_stats(df, y_col="mag_per_atom", p_col="pred_mag_per_atom"):
            out = {}
            for label, mask in [("correlated_electron", df["correlated_electron"]),
                                 ("non_correlated", ~df["correlated_electron"])]:
                sub = df[mask]
                y, p = sub[y_col].to_numpy(), sub[p_col].to_numpy()
                mae = float(mean_absolute_error(y, p))
                target_std = float(y.std())
                out[label] = {
                    "n": int(len(sub)), "mae": mae, "target_std": target_std,
                    "mae_over_target_std": mae / target_std if target_std > 0 else None,
                }
            return out

        sg_mp2jv = subgroup_stats(jv_sub)
        sg_jv2mp = subgroup_stats(mp_sub)
        log(f"  [mp_to_jarvis subgroup] correlated MAE/std={sg_mp2jv['correlated_electron']['mae_over_target_std']:.3f} "
            f"vs non-correlated MAE/std={sg_mp2jv['non_correlated']['mae_over_target_std']:.3f}")
        log(f"  [jarvis_to_mp subgroup] correlated MAE/std={sg_jv2mp['correlated_electron']['mae_over_target_std']:.3f} "
            f"vs non-correlated MAE/std={sg_jv2mp['non_correlated']['mae_over_target_std']:.3f}")

        results[feat_name] = {
            "baseline_4condition": cond,
            "subgroup_mp_to_jarvis": sg_mp2jv,
            "subgroup_jarvis_to_mp": sg_jv2mp,
        }

    with open(DATA / "hubbard_u_feature_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nSaved data/hubbard_u_feature_results.json ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
