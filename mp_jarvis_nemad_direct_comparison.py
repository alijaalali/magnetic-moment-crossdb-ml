"""Direct MP vs. JARVIS vs. NEMAD comparison of reported moments.

Runs the MP-side NEMAD ordering validation (analogous to the JARVIS-side
one) and a three-way overlap (NEMAD, MP and JARVIS: compounds present in
all three), so that MP and JARVIS are compared on an identical compound
set.

NEMAD provides experimental FM/AFM/NM ordering labels only. Its
'avg_magnetic_moment' column is a composition-derived input feature, not a
measurement, and is not used. The test is therefore restricted to ordering
consistency: for NEMAD-confirmed AFM compounds (net moment expected to be
~0), which database's reported |moment| stays below the 0.1 muB/atom
magnetic/non-magnetic threshold more often?

Outputs: data/mp_ordering_validation.json and
data/mp_vs_jarvis_vs_nemad_direct.json.
"""
import json
import pandas as pd
from pymatgen.core import Composition

import os
DATA = os.path.join(os.environ.get("MAGMOM_ROOT", os.path.dirname(os.path.abspath(__file__))), "data")
THRESH = 0.1


def to_reduced(formula):
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return None


nemad = pd.read_csv(f"{DATA}/nemad_classification_raw.csv", usecols=["Normalized_Composition", "Type"])
nemad["ordering"] = nemad["Type"].map({0.0: "FM", 1.0: "AFM", 2.0: "NM"})
nemad["reduced_formula"] = nemad["Normalized_Composition"].apply(to_reduced)
nemad = nemad.dropna(subset=["reduced_formula"])
nemad_dedup = nemad.drop_duplicates(subset="reduced_formula", keep="first")
print(f"NEMAD: {len(nemad_dedup)} unique reduced formulas (deduplicated)")

jarvis = pd.read_csv(f"{DATA}/predictions_mp_to_jarvis.csv")  # JARVIS's own true mag_per_atom + correlated_electron flag
mp = pd.read_csv(f"{DATA}/predictions_jarvis_to_mp.csv")      # MP's own true mag_per_atom + correlated_electron flag
print(f"JARVIS: {jarvis['reduced_formula'].nunique()} unique formulas; "
      f"MP: {mp['reduced_formula'].nunique()} unique formulas")

# --- MP-side ordering validation (mirrors the existing JARVIS-side analysis) ---
merged_mp = mp.merge(nemad_dedup[["reduced_formula", "ordering"]], on="reduced_formula", how="inner")
merged_mp["abs_mag_mp"] = merged_mp["mag_per_atom"].abs()

mp_results = {}
for subgroup_name, subset in [("all_overlap", merged_mp), ("correlated_electron_overlap", merged_mp[merged_mp["correlated_electron"]])]:
    row = {}
    for ordering in ["FM", "AFM", "NM"]:
        sub = subset[subset["ordering"] == ordering]
        if len(sub) == 0:
            continue
        row[ordering] = {
            "n": int(len(sub)),
            "median_abs_mag_mp": float(sub["abs_mag_mp"].median()),
            "mean_abs_mag_mp": float(sub["abs_mag_mp"].mean()),
            "frac_exceeding_0.1_muB": float((sub["abs_mag_mp"] > THRESH).mean()),
        }
    mp_results[subgroup_name] = row
    if all(k in row for k in ("FM", "AFM", "NM")):
        afm, fm, nm = row["AFM"]["frac_exceeding_0.1_muB"], row["FM"]["frac_exceeding_0.1_muB"], row["NM"]["frac_exceeding_0.1_muB"]
        mp_results[subgroup_name]["interpretation"] = {
            "afm_rate": afm, "fm_rate": fm, "nm_rate": nm,
            "afm_closer_to_fm_than_nm": abs(afm - fm) < abs(afm - nm),
        }
json.dump(mp_results, open(f"{DATA}/mp_ordering_validation.json", "w"), indent=2)
print("Saved data/mp_ordering_validation.json")
for k, v in mp_results.items():
    print(f"  {k}: {v}")

# ============================================================
# THE DIRECT TEST: three-way overlap (NEMAD ^ MP ^ JARVIS), same
# compounds scored by both databases simultaneously.
# ============================================================
nemad_formulas = set(nemad_dedup["reduced_formula"])
jarvis_formulas = set(jarvis["reduced_formula"])
mp_formulas = set(mp["reduced_formula"])
three_way = nemad_formulas & jarvis_formulas & mp_formulas
print(f"\nThree-way overlap (NEMAD ^ MP ^ JARVIS): {len(three_way)} formulas")

jarvis_3w = jarvis[jarvis["reduced_formula"].isin(three_way)][["reduced_formula", "mag_per_atom", "correlated_electron"]].rename(
    columns={"mag_per_atom": "mag_per_atom_jarvis"})
mp_3w = mp[mp["reduced_formula"].isin(three_way)][["reduced_formula", "mag_per_atom"]].rename(
    columns={"mag_per_atom": "mag_per_atom_mp"})
nemad_3w = nemad_dedup[nemad_dedup["reduced_formula"].isin(three_way)][["reduced_formula", "ordering"]]

combo = jarvis_3w.merge(mp_3w, on="reduced_formula").merge(nemad_3w, on="reduced_formula")
combo = combo.drop_duplicates(subset="reduced_formula")
print(f"Combined three-way table: {len(combo)} rows (post-dedup)")
combo["abs_mag_jarvis"] = combo["mag_per_atom_jarvis"].abs()
combo["abs_mag_mp"] = combo["mag_per_atom_mp"].abs()

direct = {"n_three_way_overlap": int(len(combo))}
for subgroup_name, subset in [("all_three_way", combo), ("correlated_electron_three_way", combo[combo["correlated_electron"]])]:
    row = {}
    for ordering in ["FM", "AFM", "NM"]:
        sub = subset[subset["ordering"] == ordering]
        if len(sub) == 0:
            row[ordering] = None
            continue
        row[ordering] = {
            "n": int(len(sub)),
            "jarvis_frac_exceeding_0.1": float((sub["abs_mag_jarvis"] > THRESH).mean()),
            "mp_frac_exceeding_0.1": float((sub["abs_mag_mp"] > THRESH).mean()),
            "jarvis_median_abs_mag": float(sub["abs_mag_jarvis"].median()),
            "mp_median_abs_mag": float(sub["abs_mag_mp"].median()),
        }
    direct[subgroup_name] = row

# Key comparison: for NEMAD-confirmed AFM compounds (net moment expected to
# be ~0), which database reports a spuriously large |moment| less often,
# scored on the same compound set for both.
afm_subset_corr = combo[(combo["ordering"] == "AFM") & (combo["correlated_electron"])]
afm_subset_all = combo[combo["ordering"] == "AFM"]
direct["head_to_head_AFM_spurious_rate"] = {
    "correlated_electron_subgroup": {
        "n": int(len(afm_subset_corr)),
        "jarvis_spurious_rate": float((afm_subset_corr["abs_mag_jarvis"] > THRESH).mean()) if len(afm_subset_corr) else None,
        "mp_spurious_rate": float((afm_subset_corr["abs_mag_mp"] > THRESH).mean()) if len(afm_subset_corr) else None,
        "mp_better_than_jarvis": (float((afm_subset_corr["abs_mag_mp"] > THRESH).mean()) <
                                   float((afm_subset_corr["abs_mag_jarvis"] > THRESH).mean())) if len(afm_subset_corr) else None,
    },
    "all_compounds": {
        "n": int(len(afm_subset_all)),
        "jarvis_spurious_rate": float((afm_subset_all["abs_mag_jarvis"] > THRESH).mean()) if len(afm_subset_all) else None,
        "mp_spurious_rate": float((afm_subset_all["abs_mag_mp"] > THRESH).mean()) if len(afm_subset_all) else None,
        "mp_better_than_jarvis": (float((afm_subset_all["abs_mag_mp"] > THRESH).mean()) <
                                   float((afm_subset_all["abs_mag_jarvis"] > THRESH).mean())) if len(afm_subset_all) else None,
    },
}

mp_better_corr = direct["head_to_head_AFM_spurious_rate"]["correlated_electron_subgroup"]["mp_better_than_jarvis"]
mp_better_all = direct["head_to_head_AFM_spurious_rate"]["all_compounds"]["mp_better_than_jarvis"]
direct["verdict"] = (
    f"On the same three-way-overlap compound set, within the "
    f"correlated-electron subgroup (n={direct['head_to_head_AFM_spurious_rate']['correlated_electron_subgroup']['n']}), "
    f"MP's AFM spurious-large-moment rate is "
    f"{'lower (better)' if mp_better_corr else 'not lower (not better, or equal/worse)'} "
    f"than JARVIS's. Across all NEMAD-confirmed AFM compounds in the "
    f"three-way overlap regardless of subgroup (n={direct['head_to_head_AFM_spurious_rate']['all_compounds']['n']}), "
    f"MP is "
    f"{'also lower (consistent, broader support)' if mp_better_all else 'not lower there (the effect may be specific to the correlated-electron subgroup, or may not hold directly)'}. "
    f"This is the direct, same-compound-set test of the manuscript's indirect "
    f"argument."
)
print("\n" + direct["verdict"])

json.dump(direct, open(f"{DATA}/mp_vs_jarvis_vs_nemad_direct.json", "w"), indent=2)
print("Saved data/mp_vs_jarvis_vs_nemad_direct.json")
