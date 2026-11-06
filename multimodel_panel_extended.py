"""Extended multi-model robustness panel: adds 9 regressors to the 5-model
panel (multimodel_panel.py), repeating the same 4-condition zero-shot
transfer evaluation and correlated-electron subgroup diagnosis.

Added models: GaussianProcess, SVR, AdaBoost, MLP, LightGBM, CatBoost,
ExtraTrees, BayesianRidge, ElasticNet.

Subsampling: GaussianProcess (O(n^3)) and RBF-kernel SVR (roughly O(n^2)
to O(n^3)) cannot be run on the full ~75k-100k-row training sets in
reasonable time. Both are trained on a fixed random subsample of size
CAP=1500. All other models (tree ensembles, linear models) use the full
training data.

Output: data/multimodel_panel_extended.json.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
RNG = 0
import os
ROOT = os.environ.get("MAGMOM_ROOT", os.path.dirname(os.path.abspath(__file__)))
CAP = 1500  # subsample size cap for GP/SVR

CORRELATED_ELECTRON_ELEMENTS = {"V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Mo", "W"}

NEEDS_SCALING = {"GaussianProcess", "SVR", "BayesianRidge", "ElasticNet", "MLP"}
NEEDS_CAP = {"GaussianProcess", "SVR"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def has_correlated_electron_element(formula):
    try:
        els = set(str(e) for e in Composition(formula).elements)
    except Exception:
        return None
    return len(els & CORRELATED_ELECTRON_ELEMENTS) > 0


def maybe_cap(name, X, y, seed=RNG):
    if name in NEEDS_CAP and len(y) > CAP:
        rr = np.random.RandomState(seed)
        idx = rr.choice(len(y), CAP, replace=False)
        if hasattr(X, "iloc"):
            return X.iloc[idx], y[idx]
        return X[idx], y[idx]
    return X, y


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
        "GaussianProcess": lambda: GaussianProcessRegressor(
            kernel=RBF(length_scale=10.0) + WhiteKernel(noise_level=1.0),
            alpha=1e-6, normalize_y=True, random_state=RNG),
        "SVR": lambda: SVR(kernel="rbf", C=10.0, epsilon=0.01),
        "AdaBoost": lambda: AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=RNG),
        "MLP": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                     random_state=RNG, early_stopping=True),
        "LightGBM": lambda: LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8,
                                           random_state=RNG, verbosity=-1, n_jobs=4),
        "CatBoost": lambda: CatBoostRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                               random_state=RNG, verbose=False, thread_count=4),
        "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=300, max_depth=10,
                                                   random_state=RNG, n_jobs=4),
        "BayesianRidge": lambda: BayesianRidge(),
        "ElasticNet": lambda: ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=RNG),
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
        scale = name in NEEDS_SCALING

        # MP-internal
        Xtr_c, ytr_c = maybe_cap(name, Xmp_tr, ymp_tr, seed=RNG)
        if scale:
            sc = StandardScaler().fit(Xtr_c)
            Xtr_s, Xte_s = sc.transform(Xtr_c), sc.transform(Xmp_te)
        else:
            Xtr_s, Xte_s = Xtr_c, Xmp_te
        m_int_mp = ctor(); m_int_mp.fit(Xtr_s, ytr_c)
        mp_internal = metrics(ymp_te, m_int_mp.predict(Xte_s))

        # JARVIS-internal
        Xtr_c, ytr_c = maybe_cap(name, Xjv_tr, yjv_tr, seed=RNG)
        if scale:
            sc = StandardScaler().fit(Xtr_c)
            Xtr_s, Xte_s = sc.transform(Xtr_c), sc.transform(Xjv_te)
        else:
            Xtr_s, Xte_s = Xtr_c, Xjv_te
        m_int_jv = ctor(); m_int_jv.fit(Xtr_s, ytr_c)
        jarvis_internal = metrics(yjv_te, m_int_jv.predict(Xte_s))

        # MP -> JARVIS zero-shot (train on FULL MP)
        Xtr_c, ytr_c = maybe_cap(name, Xmp, ymp, seed=RNG)
        if scale:
            sc = StandardScaler().fit(Xtr_c)
            Xtr_s, Xte_s = sc.transform(Xtr_c), sc.transform(Xjv)
        else:
            Xtr_s, Xte_s = Xtr_c, Xjv
        m_full_mp = ctor(); m_full_mp.fit(Xtr_s, ytr_c)
        pred_mp2jv = m_full_mp.predict(Xte_s)
        mp_to_jarvis = metrics(yjv, pred_mp2jv)
        sub_mp2jv = subgroup_metrics(yjv, pred_mp2jv, corr_jv)

        # JARVIS -> MP zero-shot (train on FULL JARVIS)
        Xtr_c, ytr_c = maybe_cap(name, Xjv, yjv, seed=RNG)
        if scale:
            sc = StandardScaler().fit(Xtr_c)
            Xtr_s, Xte_s = sc.transform(Xtr_c), sc.transform(Xmp)
        else:
            Xtr_s, Xte_s = Xtr_c, Xmp
        m_full_jv = ctor(); m_full_jv.fit(Xtr_s, ytr_c)
        pred_jv2mp = m_full_jv.predict(Xte_s)
        jarvis_to_mp = metrics(ymp, pred_jv2mp)
        sub_jv2mp = subgroup_metrics(ymp, pred_jv2mp, corr_mp)

        panel[name] = {
            "mp_internal": mp_internal, "jarvis_internal": jarvis_internal,
            "mp_to_jarvis": mp_to_jarvis, "jarvis_to_mp": jarvis_to_mp,
            "subgroup_mp_to_jarvis": sub_mp2jv, "subgroup_jarvis_to_mp": sub_jv2mp,
            "capped": name in NEEDS_CAP,
        }
        log(f"  mp_internal R2={mp_internal['r2']:.3f} jarvis_internal R2={jarvis_internal['r2']:.3f} "
            f"mp_to_jarvis R2={mp_to_jarvis['r2']:.3f} jarvis_to_mp R2={jarvis_to_mp['r2']:.3f}")
        log(f"  mp_to_jarvis MAE/std: correlated={sub_mp2jv['correlated_electron']['mae_over_target_std']:.3f} "
            f"non_correlated={sub_mp2jv['non_correlated']['mae_over_target_std']:.3f}")
        log(f"  jarvis_to_mp MAE/std: correlated={sub_jv2mp['correlated_electron']['mae_over_target_std']:.3f} "
            f"non_correlated={sub_jv2mp['non_correlated']['mae_over_target_std']:.3f}  ({time.time()-t0:.1f}s elapsed)")

    with open(f"{ROOT}/data/multimodel_panel_extended.json", "w") as f:
        json.dump(panel, f, indent=2)
    log(f"\nSaved data/multimodel_panel_extended.json. DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
