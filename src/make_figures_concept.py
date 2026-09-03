"""
make_figures_concept.py
=======================
Explanatory figures for the Q-SOLVE presentation. These illustrate the METHOD,
not the results; results figures are produced by make_figures_results.py.

  fig1_tunneling.png    thermal hopping vs quantum tunnelling on a rugged
                        landscape, plus the annealing schedules that drive them
  fig2_qubo_mapping.png how docking becomes a QUBO / Ising problem
  fig3_pipeline.png     the end-to-end pipeline with the validation checkpoints
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

INK = "#1c1b22"
MUTED = "#6c6a78"
CLASSICAL = "#C1553B"
QUANTUM = "#3E6FB0"
ACCENT = "#4E9A7A"
GRID = "#d8d6de"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# ---------------------------------------------------------------------------
GX, TX, BX = -1.55, 1.85, 0.15        # global min, local trap, barrier centre


def landscape(x):
    """A 1-D docking-like energy landscape, built so the geometry is explicit.

    A deep global minimum and a shallower local trap are separated by a barrier
    that is deliberately TALL and NARROW -- the regime where tunnelling beats
    thermal activation, and the regime rugged docking landscapes actually
    produce.
    """
    y = -3.05 * np.exp(-((x - GX) ** 2) / 0.62)      # global minimum
    y = y - 2.05 * np.exp(-((x - TX) ** 2) / 0.55)   # local trap
    y = y + 2.35 * np.exp(-((x - BX) ** 2) / 0.055)  # tall, narrow barrier
    y = y - 1.05 * np.exp(-((x + 4.30) ** 2) / 0.75)
    y = y - 0.95 * np.exp(-((x - 4.35) ** 2) / 0.70)
    y = y + 0.16 * np.sin(3.9 * x) + 0.09 * np.sin(8.3 * x + 1.1)
    return y + 0.028 * x ** 2


def fig_tunneling():
    x = np.linspace(-6.4, 6.4, 4000)
    y = landscape(x)

    gy = landscape(np.array([GX]))[0]
    ty = landscape(np.array([TX]))[0]
    by = landscape(np.array([BX]))[0]

    # Half-width of the barrier measured at the trapped particle's energy.
    seg = (x > GX) & (x < TX)
    above = x[seg][landscape(x[seg]) > ty]
    w_lo, w_hi = above.min(), above.max()

    fig = plt.figure(figsize=(13.6, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.25, 1.0], hspace=0.78,
                          wspace=0.16, left=0.055, right=0.975,
                          top=0.760, bottom=0.075)

    for col, (title, mode) in enumerate([
        ("Simulated annealing  ·  CLASSICAL", "sa"),
        ("Quantum annealing  ·  TUNNELLING", "qa"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        colour = CLASSICAL if mode == "sa" else QUANTUM
        ax.plot(x, y, color=INK, lw=2.0, zorder=3)
        ax.fill_between(x, y, y.min() - 1.3, color=INK, alpha=0.05, zorder=1)

        ax.scatter([TX], [ty], s=140, color=colour, zorder=8,
                   edgecolor="white", linewidth=1.8)
        ax.scatter([GX], [gy], s=200, marker="*", color=ACCENT, zorder=8,
                   edgecolor="white", linewidth=1.0)
        ax.annotate("global minimum\n= correct binding pose",
                    xy=(GX, gy), xytext=(GX - 2.7, gy + 1.05),
                    fontsize=9, color=ACCENT, ha="center",
                    arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.0))
        ax.annotate("start: trapped in\na local minimum", xy=(TX, ty),
                    xytext=(TX + 2.55, ty + 2.35), fontsize=9, color=colour,
                    ha="center",
                    arrowprops=dict(arrowstyle="-", color=colour, lw=1.0))

        if mode == "sa":
            path_x = np.linspace(TX, GX, 400)
            ax.plot(path_x, landscape(path_x) + 0.13, color=CLASSICAL,
                    lw=2.8, ls="--", zorder=6)
            ax.add_patch(FancyArrowPatch((TX - 0.25, ty + 0.30),
                                         (BX + 0.30, by + 0.22),
                                         arrowstyle="-|>", mutation_scale=18,
                                         color=CLASSICAL, lw=2.4, zorder=7,
                                         connectionstyle="arc3,rad=-0.18"))
            ax.annotate("", xy=(BX - 0.95, by), xytext=(BX - 0.95, ty),
                        arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.3))
            ax.text(BX - 1.12, (by + ty) / 2, r"$\Delta E$", color=MUTED,
                    fontsize=12, va="center", ha="right")
            ax.text(0.5, 1.055,
                    r"must CLIMB over:   $P \sim e^{-\Delta E / k_{B}T}$",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color=CLASSICAL,
                    bbox=dict(boxstyle="round,pad=0.45", fc="#FBEDE9",
                              ec=CLASSICAL, lw=1.0))
            ax.text(0.5, -0.150,
                    "barrier height is all that matters — a tall barrier is\n"
                    "exponentially unlikely to cross, however thin it is",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=8.8, color=MUTED, linespacing=1.5)
        else:
            ax.add_patch(Rectangle((w_lo, y.min() - 1.3), w_hi - w_lo,
                                   by - y.min() + 1.3,
                                   facecolor=QUANTUM, alpha=0.15,
                                   edgecolor="none", zorder=2))
            ax.add_patch(FancyArrowPatch((TX - 0.18, ty), (GX + 0.18, gy + 0.72),
                                         arrowstyle="-|>", mutation_scale=18,
                                         color=QUANTUM, lw=3.0, zorder=7,
                                         linestyle=(0, (5, 2.2))))
            ax.annotate("", xy=(w_lo, ty - 1.02), xytext=(w_hi, ty - 1.02),
                        arrowprops=dict(arrowstyle="<->", color=QUANTUM, lw=1.4))
            ax.text(BX, ty - 1.30, r"width $w$", color=QUANTUM, fontsize=10.5,
                    ha="center", va="top")
            ax.text(0.5, 1.055,
                    r"passes THROUGH:   $P \sim e^{-w\sqrt{\Delta E}}$",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color=QUANTUM,
                    bbox=dict(boxstyle="round,pad=0.45", fc="#E9F0F9",
                              ec=QUANTUM, lw=1.0))
            ax.text(0.5, -0.150,
                    "a THIN barrier is cheap to tunnel through even when it is tall —\n"
                    "the wavefunction has amplitude on both sides at once",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=8.8, color=MUTED, linespacing=1.5)

        ax.set_title(title, fontsize=12.5, fontweight="bold", pad=32)
        ax.set_xlabel("configuration space  (ligand position / orientation)",
                      labelpad=6)
        ax.set_ylabel("binding energy")
        ax.set_ylim(y.min() - 1.35, y.max() + 1.15)
        ax.set_yticks([]); ax.set_xticks([])

    # --- schedules -----------------------------------------------------------
    t = np.linspace(0, 1, 400)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, np.exp(-3.1 * t), color=CLASSICAL, lw=2.6)
    ax3.set_title("Cooling schedule:  temperature $T \\rightarrow 0$",
                  fontsize=11, color=CLASSICAL, pad=8)
    ax3.set_xlabel("annealing time"); ax3.set_ylabel("$T$")
    ax3.set_yticks([]); ax3.set_xticks([])
    ax3.grid(alpha=0.25, color=GRID)
    ax3.text(0.52, 0.62, "exploration comes from HEAT",
             transform=ax3.transAxes, fontsize=9.5, color=MUTED, ha="center")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, np.exp(-3.1 * t), color=QUANTUM, lw=2.6,
             label="$\\Gamma(t)$  transverse field")
    ax4.plot(t, 1 - np.exp(-3.1 * t), color=ACCENT, lw=2.2, ls="--",
             label="problem Hamiltonian weight")
    ax4.set_title("Adiabatic schedule:  transverse field $\\Gamma \\rightarrow 0$",
                  fontsize=11, color=QUANTUM, pad=8)
    ax4.set_xlabel("annealing time"); ax4.set_ylabel("weight")
    ax4.set_yticks([]); ax4.set_xticks([])
    ax4.grid(alpha=0.25, color=GRID)
    ax4.legend(fontsize=8.5, frameon=False, loc="center right")

    fig.suptitle("Escaping local minima: the barrier SHAPE decides the winner",
                 fontsize=17, fontweight="bold", y=0.977)
    fig.text(0.5, 0.930,
             "Thermal hopping cares only about barrier HEIGHT.  Tunnelling cares about height AND WIDTH,",
             ha="center", fontsize=10.5, color=MUTED)
    fig.text(0.5, 0.902,
             "so it wins on the tall, narrow barriers that rugged docking landscapes are made of.",
             ha="center", fontsize=10.5, color=MUTED)

    path = os.path.join(OUT, "fig1_tunneling.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
def fig_qubo_mapping():
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.3))
    fig.subplots_adjust(left=0.03, right=0.98, top=0.80, bottom=0.08, wspace=0.16)

    rng = np.random.default_rng(4)

    # --- panel 1: the two graphs --------------------------------------------
    ax = axes[0]
    ax.set_title("1.  Two graphs", fontsize=12.5, fontweight="bold", pad=12)
    lig = np.array([[0.20, 0.78], [0.36, 0.90], [0.50, 0.74], [0.34, 0.62]])
    for i in range(len(lig)):
        for j in range(i + 1, len(lig)):
            ax.plot(*zip(lig[i], lig[j]), color=ACCENT, lw=1.0, alpha=0.55, zorder=1)
    ax.scatter(lig[:, 0], lig[:, 1], s=230, color=ACCENT, zorder=3,
               edgecolor="white", linewidth=1.8)
    for k, (px, py) in enumerate(lig):
        ax.text(px, py, f"$i_{k+1}$", ha="center", va="center", color="white",
                fontsize=10, fontweight="bold", zorder=4)
    ax.text(0.35, 1.00, "LIGAND graph\natoms + interatomic distances $d_{ij}$",
            ha="center", fontsize=9.5, color=ACCENT)

    pocket = np.array([[0.16, 0.30], [0.34, 0.40], [0.52, 0.32],
                       [0.66, 0.18], [0.30, 0.14], [0.50, 0.12]])
    for i in range(len(pocket)):
        for j in range(i + 1, len(pocket)):
            if np.linalg.norm(pocket[i] - pocket[j]) < 0.26:
                ax.plot(*zip(pocket[i], pocket[j]), color=QUANTUM, lw=0.9,
                        alpha=0.40, zorder=1)
    ax.scatter(pocket[:, 0], pocket[:, 1], s=210, color=QUANTUM, zorder=3,
               edgecolor="white", linewidth=1.8)
    for k, (px, py) in enumerate(pocket):
        ax.text(px, py, f"$a_{k+1}$", ha="center", va="center", color="white",
                fontsize=9.5, fontweight="bold", zorder=4)
    ax.text(0.42, 0.02, "POCKET graph\ngrid probe nodes + distances $D_{ab}$",
            ha="center", fontsize=9.5, color=QUANTUM)
    ax.set_xlim(0.02, 0.82); ax.set_ylim(-0.05, 1.12)
    ax.axis("off")

    # --- panel 2: binary variables ------------------------------------------
    ax = axes[1]
    ax.set_title("2.  One binary variable per (atom, node) pair",
                 fontsize=12.5, fontweight="bold", pad=12)
    n_i, n_a = 4, 6
    grid_vals = np.zeros((n_i, n_a), dtype=int)
    for r, c in enumerate([2, 4, 0, 5]):
        grid_vals[r, c] = 1
    for r in range(n_i):
        for c in range(n_a):
            on = grid_vals[r, c] == 1
            ax.add_patch(FancyBboxPatch(
                (c, n_i - 1 - r), 0.86, 0.86,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                facecolor=(QUANTUM if on else "#f2f1f5"),
                edgecolor=(QUANTUM if on else "#cfcdd8"), lw=1.2))
            ax.text(c + 0.43, n_i - 1 - r + 0.43, str(grid_vals[r, c]),
                    ha="center", va="center", fontsize=12,
                    color=("white" if on else MUTED),
                    fontweight=("bold" if on else "normal"))
    for r in range(n_i):
        ax.text(-0.30, n_i - 1 - r + 0.43, f"$i_{r+1}$", ha="right",
                va="center", fontsize=11, color=ACCENT, fontweight="bold")
    for c in range(n_a):
        ax.text(c + 0.43, n_i + 0.10, f"$a_{c+1}$", ha="center", va="bottom",
                fontsize=11, color=QUANTUM, fontweight="bold")
    ax.text(n_a / 2, -0.95,
            "$x_{i,a}=1$  means  atom $i$ sits on node $a$\n"
            "exactly one 1 per row  ·  at most one 1 per column",
            ha="center", fontsize=10, color=INK)
    ax.text(n_a / 2, -1.75, "4 atoms x 6 nodes = 24 qubits",
            ha="center", fontsize=10, color=MUTED, style="italic")
    ax.set_xlim(-0.75, n_a + 0.1); ax.set_ylim(-2.1, n_i + 0.75)
    ax.axis("off")

    # --- panel 3: the Hamiltonian -------------------------------------------
    ax = axes[2]
    ax.set_title("3.  Energy function to minimise", fontsize=12.5,
                 fontweight="bold", pad=12)
    ax.axis("off")

    # Sums are written without limits: mathtext stacks \sum_{i,a} underneath the
    # symbol, which collides with anything placed below it.
    blocks = [
        ("linear terms  ·  the diagonal",
         r"$\mathrm{FIT}\;=\;\Sigma\; E_{ia}\, x_{i,a}$",
         "grid interaction energy of atom $i$'s\natom type sitting at node $a$",
         ACCENT),
        ("COUPLINGS  ·  the real physics",
         r"$\mathrm{SHAPE}\;=\;\lambda_g\, \Sigma\, (d_{ij}-D_{ab})^2\, x_{i,a} x_{j,b}$",
         "punishes any assignment that stretches\nor squashes the molecule",
         QUANTUM),
        ("constraint penalties",
         r"$\mathrm{RULES}\;=\;\lambda_p\, \Sigma\, (\Sigma_a x_{i,a}-1)^2$",
         "one node per atom, one atom per node",
         "#8A6BAF"),
    ]

    top, block_h, gap = 0.98, 0.190, 0.030
    for k, (tag, expr, desc, colour) in enumerate(blocks):
        y_hi = top - k * (block_h + gap)
        ax.add_patch(FancyBboxPatch((0.02, y_hi - block_h), 0.96, block_h,
                                    boxstyle="round,pad=0.010,rounding_size=0.02",
                                    transform=ax.transAxes,
                                    facecolor=colour, alpha=0.10,
                                    edgecolor=colour, lw=1.3))
        ax.text(0.055, y_hi - 0.038, tag, transform=ax.transAxes, fontsize=8.4,
                color=colour, va="center", fontweight="bold")
        ax.text(0.055, y_hi - 0.095, expr, transform=ax.transAxes,
                fontsize=(10.5 if k == 1 else 12), color=INK, va="center")
        ax.text(0.055, y_hi - 0.156, desc, transform=ax.transAxes, fontsize=8.4,
                color=MUTED, va="center", linespacing=1.45)

    y_sum = top - 3 * (block_h + gap) - 0.040
    ax.text(0.5, y_sum, r"$H \;=\; \mathrm{FIT} \;+\; \mathrm{SHAPE} \;+\; \mathrm{RULES}$",
            transform=ax.transAxes, ha="center", fontsize=14.5, color=INK,
            fontweight="bold")
    ax.text(0.5, y_sum - 0.082,
            r"substitute  $x=(1+s)/2$,   $s \in \{-1,+1\}$",
            transform=ax.transAxes, ha="center", fontsize=10.5, color=MUTED)
    ax.text(0.5, y_sum - 0.150,
            r"$H \;=\; \Sigma\, h_i s_i \;+\; \Sigma\, J_{ij} s_i s_j$",
            transform=ax.transAxes, ha="center", fontsize=13, color=QUANTUM)
    ax.text(0.5, y_sum - 0.235,
            "QUBO and Ising are the same object in two costumes —\n"
            "which is why one formulation runs on an annealer AND on a gate QPU.",
            transform=ax.transAxes, ha="center", fontsize=8.8, color=MUTED,
            style="italic", linespacing=1.5)

    fig.suptitle("Mapping molecular docking onto a QUBO / Ising Hamiltonian",
                 fontsize=16, fontweight="bold", y=0.965)
    fig.text(0.5, 0.895,
             "Docking is posed as weighted subgraph isomorphism: place the ligand's atom graph onto the pocket's node graph "
             "so that energy is lowest and geometry is preserved.",
             ha="center", fontsize=10, color=MUTED)

    path = os.path.join(OUT, "fig2_qubo_mapping.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(14.6, 6.4))
    ax.axis("off")
    ax.set_xlim(0, 15.2); ax.set_ylim(0, 6.6)

    ax.set_xlim(0, 15.7); ax.set_ylim(0, 6.6)

    stages = [
        ("PDB 1HCK", "CDK2 + ATP\ncrystal structure", "#8A6BAF"),
        ("Receptor prep", "pdb2pqr AMBER\ncharges + AD types", "#8A6BAF"),
        ("Grid maps", "AutoDock 4.2 field\n13 types + e + d", "#8A6BAF"),
        ("SANCDB", "1002 natural\nproducts (RDKit)", ACCENT),
        ("Pose sampling", "quaternion poses\n8 confs x 64", ACCENT),
        ("QUBO / Ising", "atom-to-node map\n54 vars, ~1000 J", QUANTUM),
        ("Annealing", "simulated annealing\n+ QAOA (gate QPU)", QUANTUM),
        ("Ranked hits", "scores + contact\nprofiles", "#C08A2E"),
    ]

    w, h, gap = 1.68, 1.32, 0.22
    y = 3.70
    xs = []
    for k, (title, sub, colour) in enumerate(stages):
        x = 0.30 + k * (w + gap)
        xs.append(x)
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.04,rounding_size=0.12",
                                    facecolor=colour, alpha=0.13,
                                    edgecolor=colour, lw=1.6))
        ax.text(x + w / 2, y + h - 0.36, title, ha="center", va="center",
                fontsize=9.8, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + 0.40, sub, ha="center", va="center",
                fontsize=7.2, color=MUTED, linespacing=1.4)
        if k < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.01, y + h / 2),
                                         (x + w + gap - 0.01, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=12,
                                         color=MUTED, lw=1.4))

    # Validation checkpoints, spaced so their boxes cannot collide.
    checks = [
        (2, "ATP redocking control\n0.49 A RMSD   (bar 2.0 A)", ACCENT),
        (5, "Annealing beats greedy\ndescent on the coupled QUBO", QUANTUM),
        (7, "Vina scoring function\nas classical baseline", "#C08A2E"),
    ]
    cw = 2.30
    for k, label, colour in checks:
        x = xs[k] + w / 2
        bx = min(max(x - cw / 2, 0.15), 15.55 - cw)
        ax.add_patch(FancyArrowPatch((x, y - 0.04), (x, y - 0.70),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=colour, lw=1.6))
        ax.add_patch(FancyBboxPatch((bx, y - 1.68), cw, 0.94,
                                    boxstyle="round,pad=0.04,rounding_size=0.10",
                                    facecolor="white", edgecolor=colour, lw=1.5))
        ax.text(bx + cw / 2, y - 1.21, label, ha="center", va="center",
                fontsize=7.8, color=INK, linespacing=1.5)

    ax.text(0.30, y - 2.22, "VALIDATION CHECKPOINTS", fontsize=9,
            color=MUTED, fontweight="bold")
    ax.text(0.30, y - 2.66, "Every one is a number a judge can ask for.",
            fontsize=9.5, color=MUTED, style="italic")

    fig.suptitle("CDK2 quantum-annealing docking pipeline",
                 fontsize=16.5, fontweight="bold", y=0.95)
    fig.text(0.5, 0.875,
             "Reproducible end to end from a single PDB accession code.",
             ha="center", fontsize=10.5, color=MUTED)

    path = os.path.join(OUT, "fig3_pipeline.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    fig_tunneling()
    fig_qubo_mapping()
    fig_pipeline()
    print("\nAll concept figures in", os.path.abspath(OUT))
