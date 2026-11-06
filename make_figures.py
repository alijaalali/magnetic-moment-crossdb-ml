# Figure generation for the magnetic-moment cross-database transfer study.
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
PROJ = Path(os.environ.get("MAGMOM_ROOT", str(Path(__file__).resolve().parent)))
FIG = PROJ / "figures"
FIG.mkdir(exist_ok=True, parents=True)

import textwrap

W_FULL = 5.9   # 15 cm double-column width (inches)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "lines.linewidth": 1.2, "lines.markersize": 4,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "figure.dpi": 150, "pdf.fonttype": 42, "savefig.dpi": 220,
})


def ptitle(ax, label, text, width=44):
    """Panel title with bold panel label, wrapped to fit the panel."""
    wrapped = "\n".join(textwrap.wrap(text, width))
    lines = wrapped.split("\n")
    lines[0] = r"$\mathbf{(" + label + r")}$ " + lines[0]
    ax.set_title("\n".join(lines), loc="left", fontsize=8.5)


def check_fonts(fig, minsize=8.0):
    fig.canvas.draw()
    sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text)
             if t.get_text().strip() and t.get_visible()]
    assert min(sizes) >= minsize, min(sizes)
    return min(sizes)

BLUE = "#2166ac"
RED = "#b2182b"
GRAY = "#666666"
GRAY2 = "#aaaaaa"
ORANGE = "#e08214"
GREEN = "#1a9850"


def save(fig, name):
    m = check_fonts(fig)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=220)
    plt.close(fig)
    print(f"saved {name}  min font {m}  size {fig.get_size_inches()}")


mp = pd.read_csv(PROJ / "data" / "mp_magmom_clean.csv")
jv = pd.read_csv(PROJ / "data" / "jarvis_magmom_clean.csv")
baseline = json.load(open(PROJ / "data" / "baseline_4condition.json"))
subgroup = json.load(open(PROJ / "data" / "subgroup_variance_controlled.json"))
pred_mp_internal = None

pred_mp2jv = pd.read_csv(PROJ / "data" / "predictions_mp_to_jarvis.csv")
pred_jv2mp = pd.read_csv(PROJ / "data" / "predictions_jarvis_to_mp.csv")


# ---------- Figure 1: dataset overview ----------
def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.7), constrained_layout=True)

    ax = axes[0]
    ax.bar(["Materials\nProject", "JARVIS-\nDFT"], [len(mp), len(jv)], color=[BLUE, RED],
           hatch=None)
    ax.patches[1].set_hatch("//")
    ax.patches[1].set_edgecolor("white")
    ax.set_ylabel("Number of entries")
    ptitle(ax, "a", "Dataset size", 20)
    ax.set_ylim(0, max(len(mp), len(jv)) * 1.12)
    for i, v in enumerate([len(mp), len(jv)]):
        ax.text(i, v + 1000, str(v), ha="center", fontsize=8)

    ax = axes[1]
    mp_col = "mag_per_atom" if "mag_per_atom" in mp.columns else "total_magnetization"
    jv_col = "mag_per_atom"
    bins = np.linspace(0, 3, 50)
    ax.hist(mp[mp_col].abs().clip(upper=3), bins=bins, alpha=0.6, color=BLUE, label="Materials Project", density=True)
    ax.hist(jv[jv_col].abs().clip(upper=3), bins=bins, alpha=0.6, color=RED, label="JARVIS-DFT", density=True,
            hatch="//", edgecolor="white", linewidth=0.3)
    ax.set_xlabel(r"$|m|$ ($\mu_B$/atom)")
    ax.set_ylabel("Probability density")
    ptitle(ax, "b", "Magnetic moment magnitude", 20)
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.02, 0.85), handlelength=1.0)

    ax = axes[2]
    frac_mag_mp = (mp[mp_col].abs() > 0.1).mean() * 100
    frac_mag_jv = (jv[jv_col].abs() > 0.1).mean() * 100
    ax.bar(["Materials\nProject", "JARVIS-\nDFT"], [frac_mag_mp, frac_mag_jv], color=[BLUE, RED])
    ax.patches[1].set_hatch("//")
    ax.patches[1].set_edgecolor("white")
    ax.set_ylabel("Magnetic (%)\n($|m| > 0.1\\ \\mu_B$/atom)")
    ptitle(ax, "c", "Fraction magnetic", 20)
    ax.set_ylim(0, max(frac_mag_mp, frac_mag_jv) * 1.15)
    for i, v in enumerate([frac_mag_mp, frac_mag_jv]):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8)

    save(fig, "fig1_dataset_overview")


# ---------- Figure 2: 4-condition summary ----------
def fig2():
    conds = ["mp_internal", "jarvis_internal", "mp_to_jarvis", "jarvis_to_mp"]
    labels = ["MP-internal", "JARVIS-internal", "MP→JARVIS", "JARVIS→MP"]
    colors = [GRAY, GRAY2, BLUE, RED]
    hatches = ["", "..", "//", "\\\\"]

    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.0), constrained_layout=True)

    for ax, key, ylab, lab, ttl in [
            (axes[0], "r2", "$R^2$", "a", "Transfer accuracy"),
            (axes[1], "mae", r"MAE ($\mu_B$/atom)", "b", "Transfer error")]:
        vals = [baseline[c][key] for c in conds]
        bars = ax.bar(labels, vals, color=colors)
        for b, h in zip(bars, hatches):
            b.set_hatch(h); b.set_edgecolor("white")
        if key == "r2":
            ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(ylab)
        ptitle(ax, lab, ttl, 30)
        ax.tick_params(axis="x", rotation=35)
        for t in ax.get_xticklabels():
            t.set_ha("right")

    save(fig, "fig2_crossdb_transfer_summary")


# ---------- Figure 3: parity plots (MP-internal, JARVIS-internal) ----------
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.2), constrained_layout=True)

    ax = axes[0]
    t, p = pred_mp2jv["mag_per_atom"], pred_mp2jv["pred_mag_per_atom"]
    # Per-row internal-split predictions are not saved, so the two
    # cross-database parity scatters are shown instead.
    ax.scatter(t, p, s=2, alpha=0.15, color=BLUE, rasterized=True)
    lims = [-0.2, 3.5]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal")
    m = baseline["mp_to_jarvis"]
    ax.text(0.05, 0.95, f"$R^2$ = {m['r2']:.2f}\nMAE = {m['mae']:.3f}",
            transform=ax.transAxes, fontsize=8.5, va="top")
    ax.set_xlabel(r"True $|m|$ ($\mu_B$/atom, JARVIS)")
    ax.set_ylabel(r"Predicted $|m|$ ($\mu_B$/atom)")
    ptitle(ax, "a", "MP $\\to$ JARVIS zero-shot transfer", 34)

    ax = axes[1]
    t, p = pred_jv2mp["mag_per_atom"], pred_jv2mp["pred_mag_per_atom"]
    ax.scatter(t, p, s=2, alpha=0.15, color=RED, rasterized=True)
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal")
    m = baseline["jarvis_to_mp"]
    ax.text(0.05, 0.95, f"$R^2$ = {m['r2']:.2f}\nMAE = {m['mae']:.3f}",
            transform=ax.transAxes, fontsize=8.5, va="top")
    ax.set_xlabel(r"True $|m|$ ($\mu_B$/atom, MP)")
    ax.set_ylabel(r"Predicted $|m|$ ($\mu_B$/atom)")
    ptitle(ax, "b", "JARVIS $\\to$ MP zero-shot transfer", 34)

    save(fig, "fig3_crossdb_parity")


# ---------- Figure 4: central subgroup-diagnosis figure ----------
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.4), constrained_layout=True)

    directions = ["mp_to_jarvis", "jarvis_to_mp"]
    dir_labels = ["MP → JARVIS", "JARVIS → MP"]

    ax = axes[0]
    x = np.arange(2)
    w = 0.35
    corr_vals = [subgroup[d]["subgroups"]["correlated_electron"]["mae_over_target_std"] for d in directions]
    noncorr_vals = [subgroup[d]["subgroups"]["non_correlated"]["mae_over_target_std"] for d in directions]
    ax.bar(x - w/2, corr_vals, w, color=ORANGE, label="Correlated-electron")
    ax.bar(x + w/2, noncorr_vals, w, color=GRAY2, label="Non-correlated",
           hatch="//", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(dir_labels)
    ax.set_ylabel("MAE / target std.\n(variance-normalized)")
    ptitle(ax, "a", "Correlated-electron compounds transfer worse", 34)
    ax.set_ylim(0, max(corr_vals + noncorr_vals) * 1.35)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[1]
    diffs = [subgroup[d]["mae_difference_correlated_minus_noncorrelated"]["mean"] for d in directions]
    cis = [subgroup[d]["mae_difference_correlated_minus_noncorrelated"]["ci95_composition_grouped_bootstrap"] for d in directions]
    errs = [[diffs[i] - cis[i][0] for i in range(2)], [cis[i][1] - diffs[i] for i in range(2)]]
    bars = ax.bar(dir_labels, diffs, color=[BLUE, RED])
    bars[1].set_hatch("\\\\"); bars[1].set_edgecolor("white")
    ax.errorbar(dir_labels, diffs, yerr=errs, fmt="none", ecolor="black", capsize=4, linewidth=1.1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("MAE excess, corr. $-$ non-corr.\n($\\mu_B$/atom, 95% bootstrap CI)")
    ptitle(ax, "b", "Excess error excludes zero in both directions", 34)

    save(fig, "fig4_subgroup_diagnosis")


# ---------- Figure 5: selective-prediction / abstention ----------
def fig5():
    sp = json.load(open(PROJ / "data" / "selective_prediction_results.json"))
    curve = sp["selective_prediction_curve"]
    frac = [c["abstain_fraction"] * 100 for c in curve]
    mae = [c["mae_retained"] for c in curve]
    f_corr = [c["frac_correlated_electron_abstained"] * 100 for c in curve]
    f_non = [c["frac_non_correlated_abstained"] * 100 for c in curve]

    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.3), constrained_layout=True)

    ax = axes[0]
    ax.plot(frac, mae, "-o", color=BLUE, markersize=4)
    ax.set_xlabel("Abstention fraction (%)")
    ax.set_ylabel(r"Retained-set MAE ($\mu_B$/atom)")
    ptitle(ax, "a", "Abstaining on uncertain predictions improves retained-set accuracy", 34)

    ax = axes[1]
    ax.plot(frac, f_corr, "-o", color=ORANGE, markersize=4, label="Correlated-electron")
    ax.plot(frac, f_non, "-s", color=GRAY, markersize=4, label="Non-correlated")
    ax.plot([0, 50], [0, 50], "k--", lw=0.8, label="Equal abstention\n(reference)")
    ax.set_xlabel("Overall abstention fraction (%)")
    ax.set_ylabel("Subgroup abstained on (%)")
    ptitle(ax, "b", "Abstention preferentially targets correlated-electron compounds", 34)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    save(fig, "fig5_selective_prediction")


# ---------- Figure 6: extended-scale targeted vs. random fine-tuning ----------
def fig6():
    ft = json.load(open(PROJ / "data" / "finetuning_extended_scale.json"))
    k_list = ft["k_list"]
    # k=2000 and k=5000 use the 10-seed merged results; k=10/100/500 use
    # the original 3 seeds.
    more = json.load(open(PROJ / "data" / "finetuning_more_seeds_merged.json"))
    for strat in ["random", "targeted"]:
        for protocol in ["joint", "target_only"]:
            for k in ["2000", "5000"]:
                ft["summary"][strat][protocol][k] = more["summary"][strat][protocol][k]
    for k in ["2000", "5000"]:
        ft["paired_targeted_vs_random_correlated_subgroup"][k] = more["paired_targeted_vs_random_correlated_subgroup"][k]

    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.4), constrained_layout=True)

    ax = axes[0]
    for strat, color, marker, ls in [("random", GRAY, "o", "--"), ("targeted", ORANGE, "s", "-")]:
        vals = [ft["summary"][strat]["joint"][str(k)]["correlated_mae"]["mean"] for k in k_list]
        errs = [ft["summary"][strat]["joint"][str(k)]["correlated_mae"]["std"] for k in k_list]
        ax.errorbar(k_list, vals, yerr=errs, fmt=f"{ls}{marker}", color=color, capsize=3,
                    markersize=5, label=strat.capitalize())
    ax.set_xlabel("Labeled JARVIS compounds\nadded, $k$")
    ax.set_ylabel("Correlated-electron\nsubgroup MAE ($\\mu_B$/atom)")
    ptitle(ax, "a", "Joint training: targeted sampling helps as $k$ grows", 34)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    diffs = [ft["paired_targeted_vs_random_correlated_subgroup"][str(k)]["mean_mae_diff_targeted_minus_random"] for k in k_list]
    cis = [ft["paired_targeted_vs_random_correlated_subgroup"][str(k)]["ci95_bootstrap_over_seeds"] for k in k_list]
    errs = [[diffs[i] - cis[i][0] for i in range(len(k_list))],
            [cis[i][1] - diffs[i] for i in range(len(k_list))]]
    flags = [d > cis[i][0] and 0 >= cis[i][0] and 0 <= cis[i][1] for i, d in enumerate(diffs)]
    bars = ax.bar([str(k) for k in k_list], diffs, color=[GRAY2 if f else BLUE for f in flags])
    for b, f in zip(bars, flags):
        if f:
            b.set_hatch("//"); b.set_edgecolor("white")
    ax.errorbar([str(k) for k in k_list], diffs, yerr=errs, fmt="none", ecolor="black", capsize=4, linewidth=1.1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Labeled JARVIS compounds\nadded, $k$")
    ax.set_ylabel("MAE difference, targeted $-$ random\n(95% bootstrap CI over seeds)")
    ptitle(ax, "b", "Targeted advantage grows with $k$ (excludes zero at $k\\geq$2000)", 26)

    save(fig, "fig6_extended_finetuning")


def fig7():
    """NEMAD experimental magnetic-ordering validation."""
    nemad = json.load(open(PROJ / "data" / "jarvis_ordering_validation.json"))
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.5), constrained_layout=True)

    order = ["FM", "AFM", "NM"]
    colors_order = [RED, ORANGE, GRAY2]
    labels_order = ["FM", "AFM", "NM"]
    hatches = ["\\\\", "//", ""]

    ax = axes[0]
    vals = [nemad["all_overlap"][o]["frac_exceeding_0.1_muB"] * 100 for o in order]
    ns = [nemad["all_overlap"][o]["n"] for o in order]
    bars = ax.bar(labels_order, vals, color=colors_order)
    for b, h in zip(bars, hatches):
        b.set_hatch(h); b.set_edgecolor("white")
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(i, v + 2.0, f"n={n}", ha="center", fontsize=8)
    ax.set_ylabel("% with spurious/expected large moment\n(JARVIS $|m|>0.1\\ \\mu_B$/atom)")
    ptitle(ax, "a", "General population (n=10,160)", 26)
    ax.set_ylim(0, 72)
    ax.set_xlabel("FM: large expected;\nAFM/NM: near-zero expected")

    ax2 = axes[1]
    vals2 = [nemad["correlated_electron_overlap"][o]["frac_exceeding_0.1_muB"] * 100 for o in order]
    ns2 = [nemad["correlated_electron_overlap"][o]["n"] for o in order]
    bars = ax2.bar(labels_order, vals2, color=colors_order)
    for b, h in zip(bars, hatches):
        b.set_hatch(h); b.set_edgecolor("white")
    for i, (v, n) in enumerate(zip(vals2, ns2)):
        ax2.text(i, v + 2.0, f"n={n}", ha="center", fontsize=8)
    ax2.set_ylabel("% with spurious/expected large moment")
    ptitle(ax2, "b", "Correlated-electron subgroup (n=4,368)", 26)
    ax2.set_ylim(0, 72)
    ax2.set_xlabel("FM: large expected;\nAFM/NM: near-zero expected")

    fig.suptitle("JARVIS moments vs. real experimental magnetic ordering (NEMAD)", fontsize=9)
    save(fig, "fig7_nemad_validation")


def fig8():
    """Multi-model panel: correlated-electron subgroup effect vs. in-distribution fit quality, all 15 models."""
    combined = json.load(open(PROJ / "data" / "multimodel_panel_combined_all15.json"))
    rows = []
    for name, d in combined.items():
        mp_r2 = d["mp_internal"]["r2"]
        s1 = d["subgroup_mp_to_jarvis"]
        c1 = s1["correlated_electron"]["mae_over_target_std"]
        n1 = s1["non_correlated"]["mae_over_target_std"]
        rows.append((name, mp_r2, c1, n1))
    rows.sort(key=lambda r: -r[1])

    names = [r[0] for r in rows]
    mp_r2s = [r[1] for r in rows]
    corr = [r[2] for r in rows]
    noncorr = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(W_FULL, 3.9), constrained_layout=True)
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, corr, w, label="Correlated-electron", color=BLUE, alpha=0.85)
    ax.bar(x + w / 2, noncorr, w, label="Non-correlated", color=ORANGE, alpha=0.85,
           hatch="//", edgecolor="white")
    ax.axvline(6.5, color="k", ls="--", lw=1)
    ax.set_ylim(0, 1.8)
    ax.text(3, 1.75, "well-fitting\n(MP-internal $R^2>0.8$)", ha="center", va="top", fontsize=8)
    ax.text(11, 1.75, "poorly-fitting\n($R^2<0.5$)", ha="center", va="top", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE/$\\sigma_{\\mathrm{target}}$, MP$\\to$JARVIS")
    ax.set_title("Correlated-electron effect across 15 independent regressors,\nsorted by in-distribution fit quality", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.8))
    save(fig, "fig8_multimodel_panel")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
    print("ALL FIGURES DONE")
