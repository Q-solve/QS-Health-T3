"""
make_figures_results.py
=======================
Figures built from the actual run outputs.

  fig4_redock_funnel.png  energy-vs-RMSD funnel for the ATP redocking control
  fig5_screen_scores.png  score distribution over the SANCDB library
  fig6_solver_gap.png     simulated annealing vs greedy descent on the
                          coupled assignment QUBO
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "figures"
RES = "results"
os.makedirs(OUT, exist_ok=True)

INK = "#1c1b22"
MUTED = "#6c6a78"
CLASSICAL = "#C1553B"
QUANTUM = "#3E6FB0"
ACCENT = "#4E9A7A"
GOLD = "#C08A2E"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

PROJ = "C:/Users/EngageAI/Downloads/CDK2_Project"


# ---------------------------------------------------------------------------
def fig_redock_funnel(n_poses=3000, n_refine=40, seed=2026):
    """The canonical docking validation plot: score against RMSD.

    A scoring function that works produces a FUNNEL -- low-energy poses cluster
    at low RMSD. A flat cloud means the score carries no structural signal, no
    matter how good the top-1 number looks.
    """
    from qdock_core import (GridSet, PreparedLigand, generate_poses,
                            refine_pose, rmsd, score_pose)
    from run_01_redock_control import read_pdbqt_ligand

    grids = GridSet(PROJ, "1HCK_atp")
    crystal, ad_types, charges, elements = read_pdbqt_ligand(f"{PROJ}/ATP_ref.pdbqt")
    heavy = np.array([t != "HD" for t in ad_types])
    lig = PreparedLigand("ATP", elements, ad_types, charges, [crystal])

    rng = np.random.default_rng(seed)
    poses = generate_poses(crystal, grids.center, n_poses, rng, radius=3.5)
    e_raw = np.array([score_pose(p, lig, grids) for p in poses])
    r_raw = np.array([rmsd(p[heavy], crystal[heavy]) for p in poses])

    order = np.argsort(e_raw)[:n_refine]
    e_ref, r_ref = [], []
    for i in order:
        c, e = refine_pose(poses[i], lig, grids)
        e_ref.append(e)
        r_ref.append(rmsd(c[heavy], crystal[heavy]))
    e_ref, r_ref = np.array(e_ref), np.array(r_ref)
    e_crystal = score_pose(crystal, lig, grids)

    best = int(np.argmin(e_ref))

    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.scatter(r_raw, e_raw, s=11, alpha=0.30, color=MUTED, edgecolor="none",
               label=f"{n_poses} random rigid poses", zorder=2)
    ax.scatter(r_ref, e_ref, s=52, alpha=0.90, color=QUANTUM,
               edgecolor="white", linewidth=0.8,
               label=f"top {n_refine} after local refinement", zorder=4)
    ax.scatter([r_ref[best]], [e_ref[best]], s=250, marker="*", color=ACCENT,
               edgecolor="white", linewidth=1.2, zorder=6,
               label=f"best-scoring pose: {r_ref[best]:.2f} $\\AA$")
    ax.axhline(e_crystal, color=CLASSICAL, ls=":", lw=1.8, zorder=3)
    ax.text(ax.get_xlim()[1] * 0.985, e_crystal, " crystal pose energy ",
            color=CLASSICAL, fontsize=8.6, ha="right", va="bottom")
    ax.axvline(2.0, color=ACCENT, ls="--", lw=1.8, zorder=3)

    ax.set_xlabel("heavy-atom RMSD from the crystal pose  ($\\AA$)")
    ax.set_ylabel("AutoDock4 interaction energy  (kcal/mol)")
    ax.set_title("ATP redocking control: the scoring function funnels to the\n"
                 "crystallographic pose", fontsize=13, fontweight="bold", pad=14)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(alpha=0.22, linewidth=0.6)
    # Clip the top: a few hundred severely clashing poses would otherwise
    # squash the whole funnel into the bottom centimetre of the plot.
    y_lo = min(e_ref.min(), e_crystal) - 2.0
    y_hi = 30.0
    ax.set_ylim(y_lo, y_hi)
    ax.text(1.90, y_lo + 0.06 * (y_hi - y_lo), "2.0 $\\AA$ success threshold ",
            color=ACCENT, fontsize=9.5, va="bottom", ha="right", rotation=90)
    n_clip = int((e_raw > y_hi).sum())
    ax.text(0.015, 0.975,
            f"{n_clip} of {n_poses} random poses clash badly and score above "
            f"{y_hi:.0f} kcal/mol (off-scale)",
            transform=ax.transAxes, fontsize=8.2, color=MUTED, va="top")
    fig.tight_layout()
    path = os.path.join(OUT, "fig4_redock_funnel.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path, f"(best {r_ref[best]:.2f} A at {e_ref[best]:.2f} kcal/mol)")
    return {"best_rmsd": float(r_ref[best]), "best_energy": float(e_ref[best]),
            "crystal_energy": float(e_crystal)}


# ---------------------------------------------------------------------------
def fig_screen_scores():
    csv = os.path.join(RES, "screen_ranked.csv")
    if not os.path.exists(csv):
        print("skip fig5: no screen_ranked.csv yet")
        return
    df = pd.read_csv(csv)
    ctrl_path = os.path.join(RES, "redock_control.json")
    atp_e = None
    if os.path.exists(ctrl_path):
        atp_e = json.load(open(ctrl_path))["crystal_pose_energy"]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    ax.hist(df["Score_kcal_mol"], bins=45, color=QUANTUM, alpha=0.75,
            edgecolor="white", linewidth=0.6)
    if atp_e is not None:
        ax.axvline(atp_e, color=CLASSICAL, ls="--", lw=2.0)
        ax.text(atp_e, ax.get_ylim()[1] * 0.94,
                f"  ATP (native ligand)\n  {atp_e:.2f} kcal/mol",
                color=CLASSICAL, fontsize=9, va="top")
    top = df.nsmallest(10, "Score_kcal_mol")["Score_kcal_mol"].max()
    ax.axvspan(df["Score_kcal_mol"].min() - 0.2, top, color=ACCENT, alpha=0.16)
    ax.text(df["Score_kcal_mol"].min(), ax.get_ylim()[1] * 0.55,
            " top 10\n hits", color=ACCENT, fontsize=9, fontweight="bold",
            va="top")
    ax.set_xlabel("docking score  (kcal/mol)")
    ax.set_ylabel("number of compounds")
    ax.set_title(f"SANCDB library screened against CDK2  (n = {len(df)})",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.22, linewidth=0.6)

    ax = axes[1]
    sc = ax.scatter(df["Min_dist_to_hinge_A"], df["Score_kcal_mol"],
                    c=df["Hydrophobic_contacts"], cmap="viridis",
                    s=20, alpha=0.75, edgecolor="none")
    ax.axvspan(2.5, 3.8, color=ACCENT, alpha=0.14)
    ax.text(3.15, ax.get_ylim()[1] * 0.98, "hinge H-bond\nrange",
            color=ACCENT, fontsize=8.6, ha="center", va="top")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("hydrophobic contact residues", fontsize=9)
    cb.outline.set_visible(False)
    ax.set_xlabel("closest approach to the Leu83 hinge  ($\\AA$)")
    ax.set_ylabel("docking score  (kcal/mol)")
    ax.set_title("Score vs hinge engagement", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.22, linewidth=0.6)

    fig.tight_layout()
    path = os.path.join(OUT, "fig5_screen_scores.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
def fig_solver_gap():
    csv = os.path.join(RES, "assignment_qubo.csv")
    if not os.path.exists(csv):
        print("skip fig6: no assignment_qubo.csv yet")
        return
    df = pd.read_csv(csv)
    if "sa_energy" not in df or "steepest_energy" not in df:
        print("skip fig6: solver columns missing")
        return

    n = len(df)
    idx = np.arange(n)
    best = df[["sa_energy", "tabu_energy", "steepest_energy"]].min(axis=1)
    gap = df["steepest_energy"] - df["sa_energy"]        # > 0 => SA found lower
    sa_wins = int((gap > 1e-6).sum())
    ties = int((gap.abs() <= 1e-6).sum())
    greedy_wins = int((gap < -1e-6).sum())

    methods = [("tabu_energy", "tabu search", ACCENT),
               ("steepest_energy", "greedy steepest descent", CLASSICAL),
               ("sa_energy", "simulated annealing", QUANTUM)]

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    # -- how often each solver reaches the best answer anyone found ----------
    ax = axes[0]
    hits = [int(np.isclose(df[m], best, atol=1e-6).sum()) for m, _, _ in methods]
    bars = ax.barh([lbl for _, lbl, _ in methods], hits,
                   color=[c for _, _, c in methods], alpha=0.9)
    for b, h in zip(bars, hits):
        ax.text(h + 0.35, b.get_y() + b.get_height() / 2, f"{h}/{n}",
                va="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlim(0, n * 1.18)
    ax.set_xlabel(f"instances where the solver reached the best answer found (of {n})")
    ax.set_title("Classical local search wins at this size",
                 fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.22, axis="x", linewidth=0.6)
    ax.invert_yaxis()

    # -- per-instance signed gap --------------------------------------------
    ax = axes[1]
    colours = [QUANTUM if g > 1e-6 else (CLASSICAL if g < -1e-6 else MUTED)
               for g in gap]
    ax.barh(idx, gap, color=colours, alpha=0.9)
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_yticks(idx)
    ax.set_yticklabels(df["Ligand_ID"], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlabel("steepest descent energy  −  annealing energy")
    ax.set_title(f"Annealing better in {sa_wins}, greedy better in {greedy_wins}, "
                 f"tied in {ties}", fontsize=11.5, fontweight="bold")
    ax.grid(alpha=0.22, axis="x", linewidth=0.6)
    xl = ax.get_xlim()
    ax.text(xl[1], n + 0.4, "annealing better →", fontsize=8.4, color=QUANTUM,
            ha="right", va="top")
    ax.text(xl[0], n + 0.4, "← greedy better", fontsize=8.4, color=CLASSICAL,
            ha="left", va="top")
    ax.text(0.99, 0.015,
            f"{df['n_vars'].iloc[0]} variables · "
            f"{int(df['n_geometry_couplings'].median())} geometry couplings · "
            f"density {df['coupling_density'].iloc[0]:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.4,
            color=MUTED)

    fig.suptitle("Same Hamiltonian, three solvers — the honest comparison",
                 fontsize=14.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(OUT, "fig6_solver_gap.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path,
          f"(SA better {sa_wins}, greedy better {greedy_wins}, ties {ties})")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "funnel"):
        fig_redock_funnel()
    if which in ("all", "screen"):
        fig_screen_scores()
    if which in ("all", "solver"):
        fig_solver_gap()
