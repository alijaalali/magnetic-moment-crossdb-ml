"""
Main pipeline for the magnetic-moment cross-database transfer study
(Materials Project vs. JARVIS-DFT): reads the cleaned MP and JARVIS
tables from data/, computes composition-only Magpie features (132
descriptors), and runs the baseline 4-condition transfer test, the
subgroup diagnosis and the composition-overlap/leakage check.

Target handling: mag_per_atom is strongly zero-inflated (57% of MP and 79%
of JARVIS rows are ~0 mu_B/atom), with a continuous distribution among the
nonzero ("magnetic") compounds. A regression on the raw value is dominated
by the zero mass and says little about whether a database labels a
compound magnetic or not, or about the moment size among magnetic
compounds. Two outputs are therefore reported: (1) a binary
magnetic/non-magnetic classifier (threshold 0.1 mu_B/atom, the
pymatgen CollinearMagneticStructureAnalyzer default for magnetically
ordered structures), and (2) a gradient-boosting regression on
mag_per_atom (all rows) as the standard headline metric.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")
RNG = 0
import os
PROJ = Path(os.environ.get("MAGMOM_ROOT", str(Path(__file__).resolve().parent)))
DATA = PROJ / "data"
MODELS = PROJ / "models"
MODELS.mkdir(exist_ok=True)

MAG_THRESHOLD = 0.1  # mu_B/atom, pymatgen CollinearMagneticStructureAnalyzer default

# MP's own documented GGA+U correction element list (Wang, Jain, Ong, Ceder
# mixing scheme; pymatgen MaterialsProjectCompatibility/MP2020Compatibility),
# i.e. the exact elements for which MP applies a Hubbard +U correction and
# JARVIS's standard dft_3d workflow does not. This is a direct, citable
# definition of the "correlated-electron" subgroup, tighter than a generic
# redox-active-element list.
CORRELATED_ELECTRON_ELEMENTS = {"V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Mo", "W"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def has_correlated_electron_element(formula):
    try:
        els = set(str(e) for e in Composition(formula).elements)
    except Exception:
        return None
    return len(els & CORRELATED_ELECTRON_ELEMENTS) > 0


def metrics(y_true, y_pred, y_true_bin=None, y_pred_bin=None, p_mag=None):
    out = {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
        "target_mean": float(np.mean(y_true)),
        "target_std": float(np.std(y_true, ddof=1)) if len(y_true) > 1 else None,
    }
    if y_true_bin is not None:
        out["clf_accuracy"] = float(accuracy_score(y_true_bin, y_pred_bin))
        out["clf_f1"] = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
        if p_mag is not None and len(set(y_true_bin)) > 1:
            out["clf_auc"] = float(roc_auc_score(y_true_bin, p_mag))
        out["frac_magnetic_true"] = float(np.mean(y_true_bin))
    return out


def main():
    t0 = time.time()

    # ---------- 1. Load ----------
    mp = pd.read_csv(DATA / "mp_magmom_clean.csv")
    jv = pd.read_csv(DATA / "jarvis_magmom_clean.csv")
    log(f"Loaded: MP={len(mp)} rows, JARVIS={len(jv)} rows")

    # One row per reduced formula. A lowest-energy polymorph is not
    # identifiable in this table, so the median mag_per_atom per reduced
    # formula is used. The median is insensitive to polymorphs with
    # inflated moments (e.g. FeO, CoO and MnO have mixed polymorphs).
    mp_agg = mp.groupby("reduced_formula").agg(
        mag_per_atom=("mag_per_atom", "median"),
        formula_pretty=("formula_pretty", "first"),
    ).reset_index()
    jv_agg = jv.groupby("reduced_formula").agg(
        mag_per_atom=("mag_per_atom", "median"),
        formula=("formula", "first"),
    ).reset_index()
    log(f"Aggregated to one row per reduced formula (median mag_per_atom): "
        f"MP={len(mp_agg)}, JARVIS={len(jv_agg)}")

    mp_agg["mag_abs"] = mp_agg["mag_per_atom"].abs()
    jv_agg["mag_abs"] = jv_agg["mag_per_atom"].abs()
    mp_agg["is_magnetic"] = (mp_agg["mag_abs"] > MAG_THRESHOLD).astype(int)
    jv_agg["is_magnetic"] = (jv_agg["mag_abs"] > MAG_THRESHOLD).astype(int)
    log(f"Magnetic fraction (|m|>{MAG_THRESHOLD} mu_B/atom): "
        f"MP={mp_agg['is_magnetic'].mean():.3f}, JARVIS={jv_agg['is_magnetic'].mean():.3f}")

    # ---------- 2. Featurize (Magpie, composition-only) ----------
    feat = ElementProperty.from_preset("magpie")
    feat.set_n_jobs(8)

    def featurize(df, formula_col):
        comps, ok = [], []
        for f in df[formula_col]:
            try:
                comps.append(Composition(f))
                ok.append(True)
            except Exception:
                ok.append(False)
        d = df[ok].reset_index(drop=True)
        X = feat.featurize_dataframe(pd.DataFrame({"composition": comps}), "composition",
                                      ignore_errors=True)
        Xf = X.drop(columns=["composition"])
        mask = Xf.notna().all(axis=1)
        return Xf[mask].reset_index(drop=True), d.loc[mask].reset_index(drop=True)

    log("Featurizing MP (Magpie)...")
    Xmp, mp_meta = featurize(mp_agg, "formula_pretty")
    log(f"  MP featurized: {len(Xmp)} rows, {Xmp.shape[1]} features ({time.time()-t0:.1f}s)")

    log("Featurizing JARVIS (Magpie)...")
    Xjv, jv_meta = featurize(jv_agg, "formula")
    log(f"  JARVIS featurized: {len(Xjv)} rows ({time.time()-t0:.1f}s)")

    feature_cols = list(Xmp.columns)
    Xjv = Xjv[feature_cols]

    ymp = mp_meta["mag_per_atom"].to_numpy(dtype=float)
    yjv = jv_meta["mag_per_atom"].to_numpy(dtype=float)
    ymp_bin = mp_meta["is_magnetic"].to_numpy(dtype=int)
    yjv_bin = jv_meta["is_magnetic"].to_numpy(dtype=int)

    Xmp.to_parquet(DATA / "featurized_mp.parquet")
    Xjv.to_parquet(DATA / "featurized_jarvis.parquet")
    mp_meta.to_csv(DATA / "mp_meta_agg.csv", index=False)
    jv_meta.to_csv(DATA / "jarvis_meta_agg.csv", index=False)
    log("Saved featurized tables.")

    # ---------- 3. Light hyperparameter sweep (MP 5-fold CV) ----------
    log("Hyperparameter sweep (5-fold CV on MP, regression target)...")
    configs = [
        dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8),
        dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8),
        dict(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8),
        dict(n_estimators=500, max_depth=2, learning_rate=0.05, subsample=0.8),
    ]
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    Xmp_arr = Xmp.to_numpy(dtype=float)
    sweep_results = []
    for cfg in configs:
        r2s = []
        for tr_idx, te_idx in kf.split(Xmp_arr):
            m = GradientBoostingRegressor(random_state=RNG, **cfg)
            m.fit(Xmp_arr[tr_idx], ymp[tr_idx])
            r2s.append(r2_score(ymp[te_idx], m.predict(Xmp_arr[te_idx])))
        cfg_r2 = float(np.mean(r2s))
        sweep_results.append({"config": cfg, "cv_r2_mean": cfg_r2, "cv_r2_std": float(np.std(r2s))})
        log(f"  {cfg}: CV R2={cfg_r2:.4f} +- {np.std(r2s):.4f}")
    best_cfg = max(sweep_results, key=lambda r: r["cv_r2_mean"])["config"]
    log(f"Selected config: {best_cfg}")

    with open(DATA / "hyperparam_sweep.json", "w") as f:
        json.dump(sweep_results, f, indent=2, default=str)

    # ---------- 4. Baseline 4-condition table ----------
    Xmp_tr, Xmp_te, ymp_tr, ymp_te, ymp_bin_tr, ymp_bin_te = train_test_split(
        Xmp, ymp, ymp_bin, test_size=0.2, random_state=RNG)
    Xjv_tr, Xjv_te, yjv_tr, yjv_te, yjv_bin_tr, yjv_bin_te = train_test_split(
        Xjv, yjv, yjv_bin, test_size=0.2, random_state=RNG)

    def fit_reg_clf(Xtr, ytr, ytr_bin):
        reg = GradientBoostingRegressor(random_state=RNG, **best_cfg)
        reg.fit(Xtr, ytr)
        clf = GradientBoostingClassifier(random_state=RNG, **best_cfg)
        clf.fit(Xtr, ytr_bin)
        return reg, clf

    def eval_reg_clf(reg, clf, Xte, yte, yte_bin):
        pred = reg.predict(Xte)
        pred_bin = clf.predict(Xte)
        p_mag = clf.predict_proba(Xte)[:, 1]
        return metrics(yte, pred, yte_bin, pred_bin, p_mag), pred

    log("Training MP-internal and JARVIS-internal models...")
    reg_mp_int, clf_mp_int = fit_reg_clf(Xmp_tr, ymp_tr, ymp_bin_tr)
    m_mp_internal, _ = eval_reg_clf(reg_mp_int, clf_mp_int, Xmp_te, ymp_te, ymp_bin_te)

    reg_jv_int, clf_jv_int = fit_reg_clf(Xjv_tr, yjv_tr, yjv_bin_tr)
    m_jv_internal, _ = eval_reg_clf(reg_jv_int, clf_jv_int, Xjv_te, yjv_te, yjv_bin_te)

    log("Training full-MP and full-JARVIS models for zero-shot transfer...")
    reg_mp_full, clf_mp_full = fit_reg_clf(Xmp, ymp, ymp_bin)
    reg_jv_full, clf_jv_full = fit_reg_clf(Xjv, yjv, yjv_bin)

    m_mp2jv, pred_mp2jv = eval_reg_clf(reg_mp_full, clf_mp_full, Xjv, yjv, yjv_bin)
    m_jv2mp, pred_jv2mp = eval_reg_clf(reg_jv_full, clf_jv_full, Xmp, ymp, ymp_bin)

    baseline = {
        "mp_internal": m_mp_internal,
        "jarvis_internal": m_jv_internal,
        "mp_to_jarvis": m_mp2jv,
        "jarvis_to_mp": m_jv2mp,
        "best_config": best_cfg,
        "mag_threshold": MAG_THRESHOLD,
    }
    log("\n--- 4-condition baseline table ---")
    for k, v in baseline.items():
        if isinstance(v, dict) and "mae" in v:
            log(f"  {k}: n={v['n']} MAE={v['mae']:.4f} R2={v['r2']:.3f} "
                f"clf_acc={v.get('clf_accuracy')} clf_f1={v.get('clf_f1')} clf_auc={v.get('clf_auc')}")

    with open(DATA / "baseline_4condition.json", "w") as f:
        json.dump(baseline, f, indent=2, default=str)

    # save predictions for downstream subgroup/overlap analysis
    jv_meta["pred_mag_per_atom"] = pred_mp2jv
    jv_meta["pred_is_magnetic"] = clf_mp_full.predict(Xjv)
    jv_meta["pred_proba_magnetic"] = clf_mp_full.predict_proba(Xjv)[:, 1]
    jv_meta["correlated_electron"] = jv_meta["formula"].apply(has_correlated_electron_element)
    jv_meta.to_csv(DATA / "predictions_mp_to_jarvis.csv", index=False)

    mp_meta["pred_mag_per_atom"] = pred_jv2mp
    mp_meta["pred_is_magnetic"] = clf_jv_full.predict(Xmp)
    mp_meta["pred_proba_magnetic"] = clf_jv_full.predict_proba(Xmp)[:, 1]
    mp_meta["correlated_electron"] = mp_meta["formula_pretty"].apply(has_correlated_electron_element)
    mp_meta.to_csv(DATA / "predictions_jarvis_to_mp.csv", index=False)

    # ---------- 5. Subgroup diagnosis: correlated-electron vs. not ----------
    log("\n--- Subgroup diagnosis (correlated-electron vs. non-correlated) ---")

    def subgroup_metrics(meta, y_true_col, pred_col, bin_true_col, bin_pred_col, proba_col):
        out = {}
        for label, mask in [("correlated_electron", meta["correlated_electron"] == True),
                             ("non_correlated", meta["correlated_electron"] == False)]:
            sub = meta[mask]
            if len(sub) < 2:
                out[label] = None
                continue
            out[label] = metrics(
                sub[y_true_col].to_numpy(dtype=float), sub[pred_col].to_numpy(dtype=float),
                sub[bin_true_col].to_numpy(dtype=int), sub[bin_pred_col].to_numpy(dtype=int),
                sub[proba_col].to_numpy(dtype=float),
            )
        return out

    subgroup_mp2jv = subgroup_metrics(jv_meta, "mag_per_atom", "pred_mag_per_atom",
                                       "is_magnetic", "pred_is_magnetic", "pred_proba_magnetic")
    subgroup_jv2mp = subgroup_metrics(mp_meta, "mag_per_atom", "pred_mag_per_atom",
                                       "is_magnetic", "pred_is_magnetic", "pred_proba_magnetic")

    for direction, sg in [("MP->JARVIS", subgroup_mp2jv), ("JARVIS->MP", subgroup_jv2mp)]:
        log(f"  {direction}:")
        for label, m in sg.items():
            if m is None:
                log(f"    {label}: too few samples")
                continue
            log(f"    {label}: n={m['n']} MAE={m['mae']:.4f} R2={m['r2']:.3f} "
                f"target_mean={m['target_mean']:.3f} target_std={m['target_std']:.3f} "
                f"clf_acc={m['clf_accuracy']:.3f} clf_f1={m['clf_f1']:.3f}")

    with open(DATA / "subgroup_diagnosis.json", "w") as f:
        json.dump({"mp_to_jarvis": subgroup_mp2jv, "jarvis_to_mp": subgroup_jv2mp}, f, indent=2, default=str)

    # ---------- 6. Composition-overlap / leakage check ----------
    log("\n--- Overlap/leakage check ---")
    overlap_formulas = set(json.load(open(DATA / "overlap_formulas.json")))

    def overlap_split_metrics(meta, formula_col, y_true_col, pred_col, bin_true_col, bin_pred_col, proba_col):
        is_overlap = meta[formula_col].isin(overlap_formulas)
        out = {}
        for label, mask in [("overlap_with_train_db", is_overlap), ("novel_to_train_db", ~is_overlap)]:
            sub = meta[mask]
            if len(sub) < 2:
                out[label] = None
                continue
            out[label] = metrics(
                sub[y_true_col].to_numpy(dtype=float), sub[pred_col].to_numpy(dtype=float),
                sub[bin_true_col].to_numpy(dtype=int), sub[bin_pred_col].to_numpy(dtype=int),
                sub[proba_col].to_numpy(dtype=float),
            )
        return out

    overlap_mp2jv = overlap_split_metrics(jv_meta, "reduced_formula", "mag_per_atom", "pred_mag_per_atom",
                                           "is_magnetic", "pred_is_magnetic", "pred_proba_magnetic")
    overlap_jv2mp = overlap_split_metrics(mp_meta, "reduced_formula", "mag_per_atom", "pred_mag_per_atom",
                                           "is_magnetic", "pred_is_magnetic", "pred_proba_magnetic")

    for direction, ov in [("MP->JARVIS", overlap_mp2jv), ("JARVIS->MP", overlap_jv2mp)]:
        log(f"  {direction}:")
        for label, m in ov.items():
            if m is None:
                log(f"    {label}: too few samples")
                continue
            log(f"    {label}: n={m['n']} MAE={m['mae']:.4f} R2={m['r2']:.3f} clf_acc={m['clf_accuracy']:.3f}")

    with open(DATA / "overlap_leakage_check.json", "w") as f:
        json.dump({"mp_to_jarvis": overlap_mp2jv, "jarvis_to_mp": overlap_jv2mp}, f, indent=2, default=str)

    log(f"\nDONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
