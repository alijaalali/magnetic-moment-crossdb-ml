"""
Graphical abstract for the magnetic-moment cross-database transfer study
(Materials Project vs. JARVIS-DFT): missing +U in JARVIS as the isolated
driver, NEMAD experimental validation, and the remedies that work.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "axes.grid": False,
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
})

WF, HF = 5.9, 4.7
fig = plt.figure(figsize=(WF, HF))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, WF)
ax.set_ylim(0, HF)
ax.axis("off")

border = Rectangle((0.04, 0.04), WF - 0.08, HF - 0.08, linewidth=1.2,
                    edgecolor="#333333", facecolor="none", zorder=10)
ax.add_patch(border)


def box(cx, cy, w, h, text, fc="#eef3fb", ec="#2b5a8c", fontsize=8, fontweight="normal"):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        linewidth=1.0, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color="#1a1a1a", linespacing=1.25)
    return b


def arrow(x1, y1, x2, y2, color="#2b5a8c"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                         mutation_scale=10, linewidth=1.1, color=color)
    ax.add_patch(a)


ax.text(WF / 2, 4.3,
        "Chemistry-Dependent Cross-Database ML Transfer Failure\nfor Magnetic-Moment Prediction",
        ha="center", va="center", fontsize=10, fontweight="bold", color="#15314d")

# Top row: the two source databases and their U treatment
box(1.55, 3.5, 2.5, 0.85,
    "Materials Project\n101,021 compounds\n$+U$ applied (documented list)",
    fc="#eef3fb", ec="#2b5a8c")
box(4.35, 3.5, 2.5, 0.85,
    "JARVIS-DFT\n88,811 compounds\nldau=False (confirmed default)",
    fc="#fdf1ea", ec="#c0622b")

# Composition-only ML model in the middle
box(2.95, 2.55, 2.9, 0.7,
    "Composition-only (Magpie)\ngradient-boosted model",
    fc="#f3eefc", ec="#6a4c93")
arrow(1.55, 3.06, 2.2, 2.92)
arrow(4.35, 3.06, 3.7, 2.92)

# Transfer collapse result
box(2.95, 1.6, 5.3, 0.85,
    "Zero-shot transfer collapses for correlated-electron,\ntransition-metal compounds\n"
    "(MP$\\to$JARVIS $R^2$=0.01; excess error excludes zero,\nCI [0.107,0.114])",
    fc="#fdeaea", ec="#b23b3b")
arrow(2.95, 2.2, 2.95, 2.04, color="#b23b3b")

# NEMAD validation box
box(1.5, 0.55, 2.6, 0.8,
    "NEMAD experimental validation\n1,031 real antiferromagnets:\nJARVIS artifact confirmed, n=large",
    fc="#eaf6ea", ec="#3f7a4a")
# Remedies box
box(4.4, 0.55, 2.8, 0.8,
    "What works: database-identity flag\n(+16-23%), targeted labeling ($k{\\geq}2000$),\n"
    "uncertainty abstention ($-73$% MAE)",
    fc="#eaf6ea", ec="#3f7a4a")
arrow(2.2, 1.15, 1.9, 0.97, color="#3f7a4a")
arrow(3.7, 1.15, 4.0, 0.97, color="#3f7a4a")

texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()]
assert min(t.get_fontsize() for t in texts) >= 8
fig.savefig(FIG_DIR / "graphical_abstract.pdf")
fig.savefig(FIG_DIR / "graphical_abstract.png", dpi=220)
plt.close()
print("Saved graphical_abstract.pdf/.png")
