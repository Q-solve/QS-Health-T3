"""
run_03_vina_baseline.py
=======================
CLASSICAL BASELINE for criterion C4.

Re-docks the same SANCDB library with the AutoDock Vina scoring function and
compares the two rankings.

Experimental design
-------------------
The search is held CONSTANT and only the scoring function is swapped. Every
ligand gets the identical conformer ensemble and the identical set of rigid
poses as in run_02, because both scripts seed their random generator from the
ligand's index. What differs is the energy model used to choose among those
poses and to drive the local refinement:

    run_02  -> AutoDock 4.2   (12-6 LJ, directional 12-10 H-bond,
                               Mehler-Solmajer electrostatics, desolvation)
    run_03  -> AutoDock Vina  (steric gaussians, flat repulsion,
                               hydrophobic contact, piecewise H-bond,
                               no electrostatics at all)

So a rank correlation here measures agreement between two independently
parameterised, functionally different energy models -- not an artefact of one
search being luckier than the other.

The Vina bindings do not build on every platform, so the scoring function is
reimplemented in vina_score.py directly from the published functional form.
Its sanity check is that crystallographic ATP in 1HCK scores about
-6.5 kcal/mol, which is where the literature puts it.

Usage:
    python run_03_vina_baseline.py --sdf ... --receptor ... --our-csv results/screen_ranked.csv
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qdock_core import generate_poses, prepare_ligand, refine_pose
from run_02_screen import clean_name, read_molblocks
from vina_score import N_ROT_COEFF, VinaGrids, ligand_probes

_W: dict = {}


def _init_worker(receptor, center, npts, n_confs, n_poses, n_refine, seed):
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _W["grids"] = VinaGrids(receptor, center, npts, verbose=False)
    _W.update(n_confs=n_confs, n_poses=n_poses, n_refine=n_refine, seed=seed)


def _process_one(task):
    idx, molblock = task
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    mol = Chem.MolFromMolBlock(molblock, sanitize=True, removeHs=False)
    if mol is None:
        return ("fail", idx)
    name = clean_name(mol.GetProp("_Name") if mol.HasProp("_Name") else "", idx)

    lig = prepare_ligand(mol, name, n_confs=_W["n_confs"], seed=_W["seed"] + idx)
    if lig is None:
        return ("fail", idx)

    try:
        n_rot = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    except Exception:
        n_rot = 0

    grids = _W["grids"]
    probes = ligand_probes(lig.elements, lig.ad_types, lig.conformers[0])
    grids.set_ligand_classes(probes)

    # Same seed as run_02 -> same conformers, same poses. Only the score differs.
    rng = np.random.default_rng(_W["seed"] * 1_000_003 + idx)

    all_poses, all_energies = [], []
    for conf in lig.conformers:
        poses = generate_poses(conf, grids.center, _W["n_poses"], rng, radius=3.0)
        for p in poses:
            all_poses.append(p)
            all_energies.append(grids.score(p)[0])
    all_energies = np.array(all_energies)

    order = np.argsort(all_energies)[: _W["n_refine"]]
    best_e = np.inf
    for i in order:
        _, e = refine_pose(all_poses[i], lig, grids)
        best_e = min(best_e, e)

    affinity = best_e / (1.0 + N_ROT_COEFF * n_rot)
    return ("ok", {
        "Ligand_ID": name,
        "Vina_raw": round(float(best_e), 3),
        "Vina_kcal_mol": round(float(affinity), 3),
        "N_rotatable": n_rot,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf", required=True)
    ap.add_argument("--receptor", required=True)
    ap.add_argument("--our-csv", default="results/screen_ranked.csv")
    ap.add_argument("--center", nargs=3, type=float,
                    default=[100.541, 97.891, 81.717])
    ap.add_argument("--npts", nargs=3, type=int, default=[66, 66, 66])
    ap.add_argument("--n-confs", type=int, default=8)
    ap.add_argument("--n-poses", type=int, default=64)
    ap.add_argument("--n-refine", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top-n", type=int, default=0,
                    help="include this many of our best-scoring ligands")
    ap.add_argument("--random-n", type=int, default=0,
                    help="plus this many drawn at random from the whole library")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results/vina_comparison.csv")
    args = ap.parse_args()

    from scipy.stats import kendalltau, spearmanr

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ours = pd.read_csv(args.our_csv)
    blocks = read_molblocks(args.sdf, limit=args.limit)
    tasks = list(enumerate(blocks))

    # Optional subsetting. Running the baseline on OUR top hits alone would
    # restrict the score range and deflate the rank correlation, so the top-N
    # set is combined with a random sample drawn from the whole library. The
    # correlation is then reported on the unbiased random sample, and the top-N
    # set answers the separate question of whether the best hits survive an
    # independent scoring function.
    if args.top_n or args.random_n:
        names = [clean_name(b.splitlines()[0] if b.splitlines() else "", i)
                 for i, b in tasks]
        name_to_idx = {n: i for i, n in enumerate(names)}
        chosen, top_names = {}, []
        if args.top_n:
            top_names = list(ours.nsmallest(args.top_n, "Score_kcal_mol")["Ligand_ID"])
            for n in top_names:
                if n in name_to_idx:
                    chosen[name_to_idx[n]] = "top"
        if args.random_n:
            rng = np.random.default_rng(args.seed)
            pool = [i for i in range(len(tasks)) if i not in chosen]
            pick = rng.choice(pool, size=min(args.random_n, len(pool)),
                              replace=False)
            for i in pick:
                chosen[int(i)] = "random"
        tasks = [(i, blocks[i]) for i in sorted(chosen)]
        subset_kind = {names[i]: chosen[i] for i in chosen}
        print(f"Subset: {sum(1 for v in chosen.values() if v == 'top')} top hits + "
              f"{sum(1 for v in chosen.values() if v == 'random')} random "
              f"= {len(tasks)} ligands")
    else:
        subset_kind = {}

    print(f"Re-docking {len(tasks)} ligands with the Vina scoring function...")

    jobs = max(1, args.jobs if args.jobs > 0 else (os.cpu_count() or 1) - 1)
    worker_args = (args.receptor, args.center, args.npts,
                   args.n_confs, args.n_poses, args.n_refine, args.seed)
    print(f"  {jobs} worker process(es)")

    records, failures = [], 0
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(jobs, initializer=_init_worker, initargs=worker_args) as pool:
        for status, payload in pool.imap_unordered(_process_one, tasks, chunksize=4):
            if status == "ok":
                records.append(payload)
            else:
                failures += 1
            done = len(records) + failures
            if done % 100 == 0:
                rate = done / (time.perf_counter() - t0)
                print(f"  {done}/{len(tasks)} ({rate:.1f}/s, "
                      f"ETA {(len(tasks)-done)/max(rate,1e-9)/60:.1f} min)")

    vina = pd.DataFrame(records)
    print(f"\nVina baseline finished: {len(vina)} scored, {failures} failed "
          f"({time.perf_counter()-t0:.0f}s)")

    merged = ours.merge(vina, on="Ligand_ID", how="inner")
    if len(merged) < 5:
        sys.exit("Fewer than 5 ligands matched by name between the two runs.")
    print(f"Matched {len(merged)} ligands by name.")
    if subset_kind:
        merged["subset"] = merged["Ligand_ID"].map(subset_kind).fillna("random")

    rho, p_rho = spearmanr(merged["Score_kcal_mol"], merged["Vina_kcal_mol"])
    tau, p_tau = kendalltau(merged["Score_kcal_mol"], merged["Vina_kcal_mol"])

    # The unbiased estimate comes from the randomly drawn ligands only.
    rho_rand = None
    if subset_kind and (merged["subset"] == "random").sum() >= 10:
        rnd = merged[merged["subset"] == "random"]
        rr, pr = spearmanr(rnd["Score_kcal_mol"], rnd["Vina_kcal_mol"])
        rho_rand = {"n": int(len(rnd)), "spearman_rho": round(float(rr), 3),
                    "p": float(pr)}
        print(f"  unbiased random sample: n={len(rnd)}, rho={rr:+.3f}")

    n = len(merged)
    overlap = {}
    for frac in (0.05, 0.10, 0.25):
        k = max(1, int(round(frac * n)))
        a = set(merged.nsmallest(k, "Score_kcal_mol")["Ligand_ID"])
        b = set(merged.nsmallest(k, "Vina_kcal_mol")["Ligand_ID"])
        hits = len(a & b)
        overlap[f"top_{int(frac*100)}pct"] = {
            "k": k, "shared": hits,
            "enrichment_over_random": round(hits / max(1e-9, k * k / n), 2),
        }

    merged.to_csv(args.out, index=False)
    summary = {
        "n_compared": int(n),
        "spearman_rho": round(float(rho), 3),
        "spearman_p": float(p_rho),
        "kendall_tau": round(float(tau), 3),
        "kendall_p": float(p_tau),
        "overlap": overlap,
        "vina_failures": failures,
        "design": "identical conformers and poses; only the scoring function differs",
        "unbiased_random_subset": rho_rand,
    }
    with open(args.out.replace(".csv", "_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 62)
    print("AutoDock4-QUBO PIPELINE  vs  VINA SCORING FUNCTION")
    print("=" * 62)
    print(f"  Spearman rho : {rho:+.3f}  (p = {p_rho:.2e})")
    print(f"  Kendall tau  : {tau:+.3f}  (p = {p_tau:.2e})")
    for k, v in overlap.items():
        print(f"  {k:>12} overlap: {v['shared']}/{v['k']} "
              f"({v['enrichment_over_random']}x random)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.4, 5.0))
        ax.scatter(merged["Vina_kcal_mol"], merged["Score_kcal_mol"],
                   s=16, alpha=0.55, edgecolor="none", color="#5B4B8A")
        ax.set_xlabel("Vina scoring function (kcal/mol)")
        ax.set_ylabel("AutoDock4 QUBO pipeline (kcal/mol)")
        ax.set_title(f"Two scoring functions, identical search\n"
                     f"Spearman $\\rho$ = {rho:.2f}   (n = {n})")
        ax.grid(alpha=0.25, linewidth=0.6)
        fig.tight_layout()
        png = args.out.replace(".csv", "_scatter.png")
        fig.savefig(png, dpi=200)
        print(f"\nPlot -> {png}")
    except Exception as exc:
        print(f"(plot skipped: {exc})")

    print(f"-> {args.out}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
