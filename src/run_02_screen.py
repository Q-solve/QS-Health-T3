"""
run_02_screen.py
================
Main virtual screen of the SANCDB library against CDK2 (1HCK).

For every ligand:
  1. Prepare a conformer ensemble (RDKit ETKDGv3 + MMFF).
  2. Sample rigid poses per conformer with quaternion rotations.
  3. Score with the full AutoDock4 form (type map + q*elec + |q|*desolv).
  4. Solve TWO QUBOs and record both:
       A. pose selection      -- trivial couplings, kept as an honest baseline
       B. atom-to-node assignment -- geometry in the couplings, the real thing
  5. Profile contacts on the winning pose.

The solver comparison table this writes out is what answers criterion C2's
"what is lost if the quantum component is removed?" -- with numbers instead of
an assertion.

Usage:
    python run_02_screen.py --sdf CDK2_Project/sancdb.sdf --limit 200
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qdock_core import (GridSet, Receptor, generate_poses, prepare_ligand,
                        refine_pose, score_pose)
from qdock_qubo import (build_assignment_qubo, build_pose_selection_qubo,
                        decode_assignment, select_anchor_atoms,
                        select_pocket_nodes, solve_qubo)

# SANCDB ships molecule titles as absolute paths from the curators' own server,
# e.g. "/srv/my_project/.../SANC00107.pdb". Keep only the accession.
_SANC_RE = re.compile(r"(SANC\d+)", re.IGNORECASE)


def clean_name(raw: str, idx: int) -> str:
    m = _SANC_RE.search(raw or "")
    if m:
        return m.group(1).upper()
    base = os.path.basename((raw or "").strip())
    base = os.path.splitext(base)[0]
    return base or f"SANC_{idx:05d}"


# ---------------------------------------------------------------------------
# Worker process: one ligand from molblock to scored, profiled pose.
# ---------------------------------------------------------------------------

_W: dict = {}


def _init_worker(maps, prefix, receptor, hinge_weight, n_confs, n_poses,
                 n_refine, seed):
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _W["grids"] = GridSet(maps, prefix)
    _W["receptor"] = Receptor(receptor)
    _W.update(hinge_weight=hinge_weight, n_confs=n_confs, n_poses=n_poses,
              n_refine=n_refine, seed=seed)


def _process_one(task):
    idx, molblock = task
    from rdkit import Chem

    mol = Chem.MolFromMolBlock(molblock, sanitize=True, removeHs=False)
    if mol is None:
        return ("fail", idx)

    raw = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
    name = clean_name(raw, idx)

    grids, receptor = _W["grids"], _W["receptor"]
    lig = prepare_ligand(mol, name, n_confs=_W["n_confs"], seed=_W["seed"] + idx)
    if lig is None:
        return ("fail", idx)

    # Each ligand gets its own generator keyed on its index, so results do not
    # depend on how work happens to be distributed across processes.
    rng = np.random.default_rng(_W["seed"] * 1_000_003 + idx)

    all_poses, all_energies, conf_of_pose = [], [], []
    for ci, conf in enumerate(lig.conformers):
        poses = generate_poses(conf, grids.center, _W["n_poses"], rng, radius=3.0)
        for p in poses:
            all_poses.append(p)
            all_energies.append(score_pose(p, lig, grids, receptor.hinge_coord,
                                           _W["hinge_weight"]))
            conf_of_pose.append(ci)
    all_energies = np.array(all_energies)

    order = np.argsort(all_energies)[: _W["n_refine"]]
    refined = []
    for i in order:
        c, e = refine_pose(all_poses[i], lig, grids)
        refined.append((e, c, conf_of_pose[i]))
    refined.sort(key=lambda t: t[0])
    cand_energies = np.array([t[0] for t in refined])

    # QUBO A: pose selection. Trivial couplings, kept as an honest baseline.
    Qa, _ = build_pose_selection_qubo(cand_energies)
    sample_a, _, _, _ = solve_qubo(Qa, method="sa", num_reads=100, num_sweeps=500)
    picked = [k for k, v in sample_a.items() if v == 1]
    sel = picked[0] if len(picked) == 1 else int(np.argmin(cand_energies))
    best_e, best_coords, best_conf = refined[sel]

    contacts = receptor.profile_contacts(best_coords, lig.elements)

    return ("ok", {
        "Ligand_ID": name,
        "Score_kcal_mol": round(best_e, 3),
        "Conformer_used": best_conf,
        "N_conformers": len(lig.conformers),
        "N_poses_sampled": len(all_poses),
        "Polar_contacts": contacts["polar_contact_count"],
        "Polar_residues": contacts["polar_contact_residues"],
        "Hydrophobic_contacts": contacts["hydrophobic_count"],
        "Hydrophobic_residues": contacts["hydrophobic_residues"],
        "Min_dist_to_hinge_A": round(contacts["min_dist_to_hinge"], 2),
        "QUBO_A_agrees_with_argmin": bool(sel == int(np.argmin(cand_energies))),
        "_best_coords": best_coords,
        "_ligand": lig,
    })


def read_molblocks(path, limit=0):
    """Split an SDF into individual mol blocks without holding RDKit state."""
    blocks, current = [], []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("$$$$"):
                if current:
                    blocks.append("".join(current))
                    if limit and len(blocks) >= limit:
                        return blocks
                current = []
            else:
                current.append(line)
    if current and "".join(current).strip():
        blocks.append("".join(current))
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf", default="CDK2_Project/sancdb.sdf")
    ap.add_argument("--maps", default="CDK2_Project")
    ap.add_argument("--prefix", default="1HCK_atp")
    ap.add_argument("--receptor", default="CDK2_Project/1HCK.pdbqt")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N molecules in file order (0 = all)")
    ap.add_argument("--sample", type=int, default=0,
                    help="uniform random sample of N from the whole library; "
                         "unbiased, unlike --limit")
    ap.add_argument("--n-confs", type=int, default=8)
    ap.add_argument("--n-poses", type=int, default=64,
                    help="rigid poses per conformer")
    ap.add_argument("--n-refine", type=int, default=6)
    ap.add_argument("--hinge-weight", type=float, default=0.0,
                    help="0.0 = off (recommended). The maps already model H-bonds.")
    ap.add_argument("--assignment-top-n", type=int, default=20,
                    help="run the coupled QUBO on this many best-scoring ligands")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes; 0 = cpu_count() - 1")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    print("Loading grids...")
    grids = GridSet(args.maps, args.prefix)
    receptor = Receptor(args.receptor)
    print(f"  maps: {grids.available_types()} + elec + desolv")
    print(f"  hinge (Leu83 N/O midpoint): {np.round(receptor.hinge_coord, 3).tolist()}")

    if args.sample:
        # Uniform random subset of the WHOLE library. Prefer this over --limit:
        # --limit takes the first N in file order, and SANCDB accession order is
        # not independent of molecular size, so it silently biases the screen
        # toward whichever end of the size range happens to come first.
        all_blocks = read_molblocks(args.sdf, limit=0)
        rs = np.random.default_rng(args.seed)
        pick = sorted(rs.choice(len(all_blocks),
                                size=min(args.sample, len(all_blocks)),
                                replace=False).tolist())
        tasks = [(i, all_blocks[i]) for i in pick]
        n_seen = len(tasks)
        print(f"Read {len(all_blocks)} molecules from {os.path.basename(args.sdf)}; "
              f"screening a uniform random sample of {n_seen}")
    else:
        blocks = read_molblocks(args.sdf, limit=args.limit)
        n_seen = len(blocks)
        tasks = list(enumerate(blocks))
        print(f"Read {n_seen} molecules from {os.path.basename(args.sdf)}")

    rows = []
    prep_failures = 0
    t_start = time.perf_counter()

    worker_args = (args.maps, args.prefix, args.receptor, args.hinge_weight,
                   args.n_confs, args.n_poses, args.n_refine, args.seed)
    jobs = max(1, args.jobs if args.jobs > 0 else (os.cpu_count() or 1) - 1)
    print(f"Screening on {jobs} worker process(es)...")

    if jobs == 1:
        _init_worker(*worker_args)
        results = (_process_one(t) for t in tasks)
        for status, payload in results:
            if status == "ok":
                rows.append(payload)
            else:
                prep_failures += 1
            done = len(rows) + prep_failures
            if done % 25 == 0:
                rate = done / (time.perf_counter() - t_start)
                print(f"  {done}/{n_seen} processed ({rate:.1f}/s)")
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(jobs, initializer=_init_worker, initargs=worker_args) as pool:
            for status, payload in pool.imap_unordered(_process_one, tasks,
                                                       chunksize=4):
                if status == "ok":
                    rows.append(payload)
                else:
                    prep_failures += 1
                done = len(rows) + prep_failures
                if done % 50 == 0:
                    rate = done / (time.perf_counter() - t_start)
                    eta = (n_seen - done) / max(rate, 1e-9)
                    print(f"  {done}/{n_seen} processed "
                          f"({rate:.1f}/s, ETA {eta / 60:.1f} min)")

    print(f"\nPrepared {len(rows)} ligands, {prep_failures} failures out of {n_seen} read.")

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in rows])
    df = df.sort_values("Score_kcal_mol").reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    path = os.path.join(args.outdir, "screen_ranked.csv")
    df.to_csv(path, index=False)
    print(f"Ranked table -> {path}")
    print(df.head(15).to_string(index=False, max_colwidth=28))

    # ------------------------------------------------------------------
    # QUBO B on the top hits: physics in the couplings
    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("COUPLED QUBO (atom-to-node assignment) ON TOP HITS")
    print("=" * 62)

    nodes = select_pocket_nodes(grids, n_nodes=9, min_sep=2.5)
    print(f"Selected {len(nodes)} pocket nodes as the target graph.")

    by_name = {r["Ligand_ID"]: r for r in rows}
    assignment_rows = []

    for name in df["Ligand_ID"].head(args.assignment_top_n):
        r = by_name[name]
        lig, coords = r["_ligand"], r["_best_coords"]
        anchors = select_anchor_atoms(lig, coords, n_anchors=6)

        Qb, var_index, meta_b = build_assignment_qubo(
            lig, coords, grids, nodes, anchors, lambda_geom=1.0
        )

        res = {}
        for method in ("steepest", "tabu", "sa"):
            try:
                s, e, sec, m = solve_qubo(Qb, method=method,
                                          num_reads=200, num_sweeps=2000)
                res[method] = (e, sec, s, m)
            except Exception as exc:
                print(f"    {method} unavailable: {exc}")

        if not res:
            continue
        best_method = min(res, key=lambda k: res[k][0])
        e_best, sec_best, s_best, m_best = res[best_method]
        assign, viol = decode_assignment(s_best, var_index,
                                         len(anchors), len(nodes))

        assignment_rows.append({
            "Ligand_ID": name,
            "n_vars": meta_b["n_vars"],
            "n_geometry_couplings": meta_b["n_geometry_couplings"],
            "coupling_density": round(meta_b["coupling_density"], 3),
            "best_method": best_method,
            "best_energy": round(e_best, 3),
            "sa_energy": round(res["sa"][0], 3) if "sa" in res else None,
            "tabu_energy": round(res["tabu"][0], 3) if "tabu" in res else None,
            "steepest_energy": round(res["steepest"][0], 3) if "steepest" in res else None,
            "sa_seconds": round(res["sa"][1], 4) if "sa" in res else None,
            "valid_assignment": len(viol) == 0,
            "assignment": json.dumps({str(k): int(v) for k, v in assign.items()}),
        })

    if assignment_rows:
        adf = pd.DataFrame(assignment_rows)
        apath = os.path.join(args.outdir, "assignment_qubo.csv")
        adf.to_csv(apath, index=False)
        print(adf.to_string(index=False))
        print(f"\n-> {apath}")

        # Direction matters. A bare != comparison counts disagreements in BOTH
        # directions, so it reports a loss as though it were a win.
        gap = adf["steepest_energy"] - adf["sa_energy"]   # > 0 => SA found lower
        sa_wins = int((gap > 1e-6).sum())
        ties = int((gap.abs() <= 1e-6).sum())
        greedy_wins = int((gap < -1e-6).sum())
        best = adf[["sa_energy", "tabu_energy", "steepest_energy"]].min(axis=1)

        print("\nSOLVER COMPARISON:")
        print(f"  Variables per instance          : {adf['n_vars'].iloc[0]}")
        print(f"  Geometry couplings per instance : {adf['n_geometry_couplings'].iloc[0]}")
        print(f"  Coupling density                : {adf['coupling_density'].iloc[0]}")
        print(f"  Simulated annealing strictly better : {sa_wins}/{len(adf)}")
        print(f"  Exact ties                          : {ties}/{len(adf)}")
        print(f"  Greedy descent strictly better      : {greedy_wins}/{len(adf)}")
        for m in ("sa_energy", "tabu_energy", "steepest_energy"):
            hits = int(np.isclose(adf[m], best, atol=1e-6).sum())
            print(f"  reached best-known, {m:<16}: {hits}/{len(adf)}")
        if greedy_wins >= sa_wins:
            print("\n  Classical local search is winning at this problem size.")
            print("  Report that honestly. The quantum argument is about SCALING,")
            print("  not about beating classical solvers on 54 variables today.")

    np.save(os.path.join(args.outdir, "pocket_nodes.npy"), nodes)
    with open(os.path.join(args.outdir, "run_meta.json"), "w") as fh:
        json.dump({
            "n_ligands_scored": len(rows),
            "n_preparation_failures": prep_failures,
            "n_molecules_read": n_seen,
            "conformers_per_ligand": args.n_confs,
            "poses_per_conformer": args.n_poses,
            "hinge_weight": args.hinge_weight,
            "seed": args.seed,
            "wall_seconds": round(time.perf_counter() - t_start, 1),
        }, fh, indent=2)
    print(f"\nMetadata -> {os.path.join(args.outdir, 'run_meta.json')}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
