"""
run_01_redock_control.py
========================
POSITIVE CONTROL. Take the crystallographic ATP ligand out of 1HCK, throw it
back at the pocket from random orientations, and measure how close the pipeline
gets to the true crystal pose.

This is the single most important slide you can add. Without it you are ranking
compounds with a scoring function nobody has checked. With it you can say:
"our pipeline recovers the crystal pose of ATP to X.XX A RMSD."

The accepted threshold in the docking literature is 2.0 A. We are doing RIGID
redocking -- the crystal conformer is reused and only rotated/translated -- so
this validates the grid maps, the scoring function and the search, but NOT
conformer generation. Say that out loud; it is the honest scope of the control.

Usage:
    python run_01_redock_control.py --ligand ATP_ref.pdbqt
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qdock_core import (GridSet, Receptor, PreparedLigand, generate_poses,
                        refine_pose, rmsd, score_pose)


def read_pdbqt_ligand(path):
    """Read a prepared PDBQT ligand: coords, AutoDock types, partial charges."""
    coords, types, charges, elements = [], [], [], []
    with open(path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            charges.append(float(line[70:76]))
            t = line[77:79].strip()
            types.append(t)
            elements.append("C" if t in ("C", "A") else t[0])
    return (np.array(coords), types, np.array(charges), elements)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default="CDK2_Project")
    ap.add_argument("--prefix", default="1HCK_atp")
    ap.add_argument("--receptor", default="CDK2_Project/1HCK.pdbqt")
    ap.add_argument("--ligand", default="CDK2_Project/ATP_ref.pdbqt",
                    help="Crystal ATP extracted from 1HCK and converted to PDBQT")
    ap.add_argument("--n-poses", type=int, default=600)
    ap.add_argument("--n-refine", type=int, default=25)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results/redock_control.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Loading grids (including .e and .d maps)...")
    grids = GridSet(args.maps, args.prefix)
    print("  atom-type maps:", grids.available_types())

    receptor = Receptor(args.receptor)
    print(f"  hinge anchor derived from Leu83 backbone: "
          f"{np.round(receptor.hinge_coord, 3).tolist()}")

    crystal_coords, ad_types, charges, elements = read_pdbqt_ligand(args.ligand)
    # Report RMSD over heavy atoms only. The polar hydrogens were added by
    # RDKit, not observed crystallographically, so including them would mix
    # modelled coordinates into a metric that is supposed to measure agreement
    # with experiment.
    heavy = np.array([t != "HD" for t in ad_types])
    print(f"  ATP reference: {len(crystal_coords)} atoms "
          f"({int(heavy.sum())} heavy, {int((~heavy).sum())} polar H)")

    def heavy_rmsd(a, b):
        return rmsd(a[heavy], b[heavy])

    ligand = PreparedLigand(name="ATP_crystal", elements=elements,
                            ad_types=ad_types, charges=charges,
                            conformers=[crystal_coords])

    crystal_energy = score_pose(crystal_coords, ligand, grids)
    print(f"\nScore of the CRYSTAL pose itself: {crystal_energy:.2f} kcal/mol")
    print("  (if this is positive or near zero, your maps or the box are wrong "
          "-- stop and fix that before anything else)")

    # --- search --------------------------------------------------------------
    print(f"\nGenerating {args.n_poses} random rigid poses...")
    poses = generate_poses(crystal_coords, grids.center, args.n_poses, rng, radius=3.5)
    energies = np.array([score_pose(p, ligand, grids) for p in poses])

    order = np.argsort(energies)[: args.n_refine]
    print(f"Locally refining the top {args.n_refine}...")
    refined = []
    for rank, idx in enumerate(order):
        coords_r, e_r = refine_pose(poses[idx], ligand, grids)
        refined.append((e_r, coords_r))
        if rank % 5 == 0:
            print(f"  {rank:>3}/{args.n_refine}  E={e_r:8.2f}  "
                  f"RMSD={heavy_rmsd(coords_r, crystal_coords):5.2f} A")

    refined.sort(key=lambda t: t[0])
    best_e, best_coords = refined[0]

    top1_rmsd = heavy_rmsd(best_coords, crystal_coords)
    all_rmsds = [heavy_rmsd(c, crystal_coords) for _, c in refined]
    best_rmsd = min(all_rmsds)
    best_rmsd_rank = int(np.argmin(all_rmsds)) + 1

    report = {
        "crystal_pose_energy": round(crystal_energy, 3),
        "top1_energy": round(best_e, 3),
        "top1_rmsd_angstrom": round(top1_rmsd, 3),
        "best_rmsd_in_top_n": round(best_rmsd, 3),
        "rank_of_best_rmsd": best_rmsd_rank,
        "n_poses_sampled": args.n_poses,
        "n_refined": args.n_refine,
        "rmsd_atoms": "heavy only",
        "passes_2A_threshold": bool(top1_rmsd < 2.0),
        "seed": args.seed,
    }

    print("\n" + "=" * 62)
    print("REDOCKING POSITIVE CONTROL")
    print("=" * 62)
    print(f"  Top-ranked pose RMSD to crystal : {top1_rmsd:.2f} A")
    print(f"  Best RMSD anywhere in top {args.n_refine:<3}   : {best_rmsd:.2f} A "
          f"(ranked #{best_rmsd_rank})")
    print(f"  Passes the 2.0 A threshold      : {report['passes_2A_threshold']}")
    if not report["passes_2A_threshold"]:
        print("\n  Not passing is still a REPORTABLE result. Say so, and say why:")
        print("   - rigid-body only, no ligand torsional freedom")
        print("   - ATP has many rotatable bonds, so it is a hard rigid case")
        print("   - raise --n-poses, or report the best-in-top-N figure instead")

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
