"""Validation of JARVIS magnetic moments against NEMAD experimental FM/AFM/NM
ordering labels, over all overlapping compounds. Uses the correlated_electron
flag from predictions_mp_to_jarvis.csv rather than redefining the subgroup.

Outputs: data/nemad_overlap_check.json and data/jarvis_ordering_validation.json.
"""
import json
import pandas as pd
import numpy as np
from pymatgen.core import Composition

import os
DATA = os.path.join(os.environ.get("MAGMOM_ROOT", os.path.dirname(os.path.abspath(__file__))), "data")

def to_reduced(formula):
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return None

# --- Load NEMAD classification (Type: 0=FM, 1=AFM, 2=NM; checked against
# membership in FM_with_curie.csv / AFM_with_Neel.csv) ---
nemad = pd.read_csv(f"{DATA}/nemad_classification_raw.csv", usecols=["Normalized_Composition", "Type"])
nemad["ordering"] = nemad["Type"].map({0.0: "FM", 1.0: "AFM", 2.0: "NM"})
nemad["reduced_formula"] = nemad["Normalized_Composition"].apply(to_reduced)
nemad = nemad.dropna(subset=["reduced_formula"])
# Some reduced formulas repeat (stoichiometric variants normalizing to the
# same reduced formula, possibly with different labels): keep the first
# and count the conflicts
dup_check = nemad.groupby("reduced_formula")["ordering"].nunique()
n_conflicting = int((dup_check > 1).sum())
nemad_dedup = nemad.drop_duplicates(subset="reduced_formula", keep="first")
print(f"NEMAD: {len(nemad)} raw rows -> {nemad['reduced_formula'].nunique()} unique reduced formulas "
      f"({n_conflicting} reduced formulas with conflicting ordering labels across variants, "
      f"first occurrence kept)")

# --- Load JARVIS predictions (columns: reduced_formula, true mag_per_atom,
# correlated_electron flag) ---
jarvis = pd.read_csv(f"{DATA}/predictions_mp_to_jarvis.csv")
print(f"JARVIS (MP->JARVIS prediction set): {len(jarvis)} rows, "
      f"{jarvis['reduced_formula'].nunique()} unique reduced formulas")

# --- MP side, if available ---
try:
    mp_pred = pd.read_csv(f"{DATA}/predictions_jarvis_to_mp.csv")
    has_mp = True
except Exception:
    has_mp = False

# --- Overlap ---
nemad_formulas = set(nemad_dedup["reduced_formula"])
jarvis_formulas = set(jarvis["reduced_formula"])
overlap_all = nemad_formulas & jarvis_formulas
print(f"NEMAD ^ JARVIS overlap (all): {len(overlap_all)} formulas")

corr_formulas = set(jarvis.loc[jarvis["correlated_electron"], "reduced_formula"])
overlap_corr = nemad_formulas & corr_formulas
print(f"NEMAD ^ JARVIS correlated-electron subgroup overlap: {len(overlap_corr)} formulas")

if has_mp:
    mp_formulas = set(mp_pred["reduced_formula"])
    overlap_mp = nemad_formulas & mp_formulas
    print(f"NEMAD ^ MP overlap (all): {len(overlap_mp)} formulas")

out = {
    "n_nemad_raw_rows": int(len(nemad)),
    "n_nemad_unique_formulas": int(nemad["reduced_formula"].nunique()),
    "n_nemad_conflicting_formulas": n_conflicting,
    "n_jarvis_unique_formulas": int(jarvis["reduced_formula"].nunique()),
    "n_overlap_all": len(overlap_all),
    "n_overlap_correlated_electron": len(overlap_corr),
}
if has_mp:
    out["n_mp_unique_formulas"] = int(mp_pred["reduced_formula"].nunique())
    out["n_overlap_mp_all"] = len(overlap_mp)

json.dump(out, open(f"{DATA}/nemad_overlap_check.json", "w"), indent=2)
print("Saved nemad_overlap_check.json")

# =====================================================================
# Core test: does JARVIS report a large |moment| for NEMAD-confirmed AFM
# compounds (net moment expected near zero), compared with NEMAD-confirmed
# FM compounds (large nonzero moment expected)?
# =====================================================================
merged = jarvis.merge(nemad_dedup[["reduced_formula", "ordering"]], on="reduced_formula", how="inner")
merged["abs_mag_jarvis"] = merged["mag_per_atom"].abs()

# Threshold for a "spuriously large" moment: the 0.1 muB/atom
# magnetic/non-magnetic boundary used in the baseline. For AFM compounds
# (net moment expected ~0) exceeding it is spurious; for FM compounds it is
# expected.
THRESH = 0.1

results = {}
for subgroup_name, subset in [("all_overlap", merged), ("correlated_electron_overlap", merged[merged["correlated_electron"]])]:
    row = {}
    for ordering in ["FM", "AFM", "NM"]:
        sub = subset[subset["ordering"] == ordering]
        if len(sub) == 0:
            continue
        frac_large = float((sub["abs_mag_jarvis"] > THRESH).mean())
        row[ordering] = {
            "n": int(len(sub)),
            "median_abs_mag_jarvis": float(sub["abs_mag_jarvis"].median()),
            "mean_abs_mag_jarvis": float(sub["abs_mag_jarvis"].mean()),
            "frac_exceeding_0.1_muB": frac_large,
        }
    results[subgroup_name] = row
    print(f"\n--- {subgroup_name} ---")
    for k, v in row.items():
        print(f"  {k}: n={v['n']}, median|m|={v['median_abs_mag_jarvis']:.4f}, "
              f"frac>0.1muB={v['frac_exceeding_0.1_muB']:.3f}")

# Key comparison: whether the fraction of AFM compounds exceeding the
# threshold is closer to that of FM compounds (spurious) or of NM compounds
# (near zero, as expected).
for subgroup_name in results:
    r = results[subgroup_name]
    if "AFM" in r and "FM" in r and "NM" in r:
        afm_rate = r["AFM"]["frac_exceeding_0.1_muB"]
        fm_rate = r["FM"]["frac_exceeding_0.1_muB"]
        nm_rate = r["NM"]["frac_exceeding_0.1_muB"]
        print(f"\n{subgroup_name}: AFM frac-large={afm_rate:.3f} "
              f"(compare: FM={fm_rate:.3f} [expected large], NM={nm_rate:.3f} [expected small])")
        results[subgroup_name]["interpretation"] = {
            "afm_rate": afm_rate, "fm_rate": fm_rate, "nm_rate": nm_rate,
            "afm_closer_to_fm_than_nm": abs(afm_rate - fm_rate) < abs(afm_rate - nm_rate),
        }

json.dump(results, open(f"{DATA}/jarvis_ordering_validation.json", "w"), indent=2)
print("\nSaved jarvis_ordering_validation.json")
