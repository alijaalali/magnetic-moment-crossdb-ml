"""Multi-model robustness panel for the magnetic-moment cross-database
transfer study: repeats the 4-condition zero-shot transfer evaluation and
the correlated-electron subgroup diagnosis across 5 regressors from
different algorithmic families, to test whether the central finding is a
chemistry effect or specific to gradient boosting.

Model substitution: k-NN (instance-based) is used in place of SVR (kernel
method) because the MP training set has 74,461 rows, where RBF-kernel SVR
scales poorly (roughly O(n^2) to O(n^3)) and is impractical. k-NN still
covers the kernel/instance-based model family.

Output: data/multimodel_panel_results.json.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
RNG = 0
import os
ROOT = os.environ.get("MAGMOM_ROOT", os.path.dirname(os.path.abspath(__file__)))

CORRELATED_ELECTRON_ELEMENTS = {"V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Mo", "W"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def has_correlated_electron_element(formula):
    try:
        els = set(str(e) for e in Composition(formula).elements)
    except Exception:
        return None
    return len(els & CORRELATED_ELECTRON_ELEMENTS) > 0


def main():
    t0 = time.time()
    Xmp = pd.read_parquet(f"{ROOT}/data/featurized_mp.parquet")
    Xjv = pd.read_parquet(f"{ROOT}/data/featurized_jarvis.parquet")
    mp_meta = pd.read_csv(f"{ROOT}/data/mp_meta_agg.csv")
    jv_meta = pd.read_csv(f"{ROOT}/data/jarvis_meta_agg.csv")
    log(f"Loaded: MP={len(Xmp)} rows, JARVIS={len(Xjv)} rows")

    feature_cols = list(Xmp.columns)
    Xjv = Xjv[feature_cols]

    ymp = mp_meta["mag_per_atom"].to_numpy(dtype=float)
    yjv = jv_meta["mag_per_atom"].to_numpy(dtype=float)

    if "correlated_electron" not in mp_meta.columns:
        mp_meta["correlated_electron"] = mp_meta["formula_pretty"].apply(has_correlated_electron_element)
    if "correlated_electron" not in jv_meta.columns:
        jv_meta["correlated_electron"] = jv_meta["formula"].apply(has_correlated_electron_element)
    corr_mp = mp_meta["correlated_electron"].to_numpy(dtype=bool)
    corr_jv = jv_meta["correlated_electron"].to_numpy(dtype=bool)

    Xmp_tr, Xmp_te, ymp_tr, ymp_te = train_test_split(Xmp, ymp, test_size=0.2, random_state=RNG)
    Xjv_tr, Xjv_te, yjv_tr, yjv_te = train_test_split(Xjv, yjv, test_size=0.2, random_state=RNG)

    models = {
        "XGBoost": lambda: XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8, random_state=RNG,
                                         n_jobs=4, verbosity=0),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=300, max_depth=10,
                                                        random_state=RNG, n_jobs=4),
        "Ridge": lambda: Ridge(alpha=1.0),
        "Lasso": lambda: Lasso(alpha=0.001, max_iter=5000),
        "kNN": lambda: KNeighborsRegressor(n_neighbors=8, weights="distance", n_jobs=4),
    }

    def metrics(y, p):
        return {"n": int(len(y)), "mae": float(mean_absolute_error(y, p)),
                "r2": float(r2_score(y, p))}

    def subgroup_metrics(y, p, corr_mask):
        out = {}
        for label, mask in [("correlated_electron", corr_mask), ("non_correlated", ~corr_mask)]:
            yy, pp = y[mask], p[mask]
            mae = float(mean_absolute_error(yy, pp))
            std = float(yy.std())
            out[label] = {"n": int(mask.sum()), "mae": mae, "target_std": std,
                          "mae_over_target_std": mae / std if std > 0 else None}
        return out

    panel = {}
    for name, ctor in models.items():
        log(f"--- {name} ---")
        m_int_mp = ctor(); m_int_mp.fit(Xmp_tr, ymp_tr)
        mp_internal = metrics(ymp_te, m_int_mp.predict(Xmp_te))

        m_int_jv = ctor(); m_int_jv.fit(Xjv_tr, yjv_tr)
        jarvis_internal = metrics(yjv_te, m_int_jv.predict(Xjv_te))

        m_full_mp = ctor(); m_full_mp.fit(Xmp, ymp)
        pred_mp2jv = m_full_mp.predict(Xjv)
        mp_to_jarvis = metrics(yjv, pred_mp2jv)
        sub_mp2jv = subgroup_metrics(yjv, pred_mp2jv, corr_jv)

        m_full_jv = ctor(); m_full_jv.fit(Xjv, yjv)
        pred_jv2mp = m_full_jv.predict(Xmp)
        jarvis_to_mp = metrics(ymp, pred_jv2mp)
        sub_jv2mp = subgroup_metrics(ymp, pred_jv2mp, corr_mp)

        panel[name] = {
            "mp_internal": mp_internal, "jarvis_internal": jarvis_internal,
            "mp_to_jarvis": mp_to_jarvis, "jarvis_to_mp": jarvis_to_mp,
            "subgroup_mp_to_jarvis": sub_mp2jv, "subgroup_jarvis_to_mp": sub_jv2mp,
        }
        log(f"  mp_internal R2={mp_internal['r2']:.3f} jarvis_internal R2={jarvis_internal['r2']:.3f} "
            f"mp_to_jarvis R2={mp_to_jarvis['r2']:.3f} jarvis_to_mp R2={jarvis_to_mp['r2']:.3f}")
        log(f"  mp_to_jarvis MAE/std: correlated={sub_mp2jv['correlated_electron']['mae_over_target_std']:.3f} "
            f"non_correlated={sub_mp2jv['non_correlated']['mae_over_target_std']:.3f}")
        log(f"  jarvis_to_mp MAE/std: correlated={sub_jv2mp['correlated_electron']['mae_over_target_std']:.3f} "
            f"non_correlated={sub_jv2mp['non_correlated']['mae_over_target_std']:.3f}  ({time.time()-t0:.1f}s elapsed)")

    with open(f"{ROOT}/data/multimodel_panel_results.json", "w") as f:
        json.dump(panel, f, indent=2)
    log(f"\nSaved data/multimodel_panel_results.json. DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
