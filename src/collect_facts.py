"""
collect_facts.py
================
Gather every number the presentation might need into one JSON file, so nothing
quoted on a slide is retyped by hand from a terminal scrollback.

    python collect_facts.py            -> results/presentation_facts.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd

RES = "results"
PROJ = "C:/Users/EngageAI/Downloads/CDK2_Project"


def safe_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def main():
    facts = {}

    # --- structure preparation ---------------------------------------------
    prep = {}
    rec = os.path.join(PROJ, "1HCK_rec.pdbqt")
    if os.path.exists(rec):
        types, charges = [], []
        for line in open(rec):
            if line.startswith("ATOM"):
                types.append(line[77:79].strip())
                charges.append(float(line[70:76]))
        from collections import Counter
        prep["receptor_atoms_united_atom"] = len(types)
        prep["receptor_type_counts"] = dict(Counter(types))
        prep["receptor_net_charge"] = round(float(sum(charges)), 3)
    lig = os.path.join(PROJ, "ATP_ref.pdbqt")
    if os.path.exists(lig):
        t = [l[77:79].strip() for l in open(lig) if l.startswith("ATOM")]
        prep["atp_atoms_total"] = len(t)
        prep["atp_heavy_atoms"] = sum(1 for x in t if x != "HD")
    prep["grid_box_angstrom"] = 66 * 0.375
    prep["grid_spacing"] = 0.375
    prep["grid_points_per_map"] = 67 ** 3
    prep["grid_centre"] = [100.541, 97.891, 81.717]
    prep["maps_written"] = sorted(
        os.path.basename(f).split(".")[1]
        for f in os.listdir(PROJ) if f.endswith(".map")
    ) if os.path.isdir(PROJ) else []
    facts["preparation"] = prep

    # --- stage 1: redocking control ----------------------------------------
    facts["redocking_control"] = safe_json(os.path.join(RES, "redock_control.json"))

    # --- stage 2: the screen -----------------------------------------------
    csv = os.path.join(RES, "screen_ranked.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        ctrl = facts.get("redocking_control") or {}
        atp_e = ctrl.get("crystal_pose_energy")
        s = df["Score_kcal_mol"]
        screen = {
            "n_ligands_ranked": int(len(df)),
            "score_min": float(s.min()),
            "score_median": float(s.median()),
            "score_max": float(s.max()),
            "n_better_than_ATP": (int((s < atp_e).sum()) if atp_e is not None else None),
            "pct_better_than_ATP": (round(100.0 * (s < atp_e).mean(), 1)
                                    if atp_e is not None else None),
            "top10": df.nsmallest(10, "Score_kcal_mol")[
                ["Ligand_ID", "Score_kcal_mol", "Polar_contacts",
                 "Hydrophobic_contacts", "Min_dist_to_hinge_A"]
            ].to_dict(orient="records"),
        }
        if "Min_dist_to_hinge_A" in df:
            hinge = df["Min_dist_to_hinge_A"]
            screen["n_in_hinge_hbond_range"] = int(((hinge >= 2.5) & (hinge <= 3.8)).sum())
        # Which pocket residues are contacted most often by the top hits?
        top = df.nsmallest(50, "Score_kcal_mol")
        from collections import Counter
        c = Counter()
        for cell in top["Hydrophobic_residues"].fillna(""):
            for r in str(cell).split(";"):
                if r.strip() and r.strip() != "None":
                    c[r.strip()] += 1
        screen["top50_most_contacted_hydrophobic"] = c.most_common(10)
        c2 = Counter()
        for cell in top["Polar_residues"].fillna(""):
            for r in str(cell).split(";"):
                if r.strip() and r.strip() != "None":
                    c2[r.strip()] += 1
        screen["top50_most_contacted_polar"] = c2.most_common(10)
        facts["screen"] = screen
    facts["run_meta"] = safe_json(os.path.join(RES, "run_meta.json"))

    # --- the solver evidence table -----------------------------------------
    acsv = os.path.join(RES, "assignment_qubo.csv")
    if os.path.exists(acsv):
        adf = pd.read_csv(acsv)
        gap = adf["steepest_energy"] - adf["sa_energy"]
        facts["coupled_qubo"] = {
            "n_instances": int(len(adf)),
            "n_vars": int(adf["n_vars"].iloc[0]),
            "n_geometry_couplings_median": int(adf["n_geometry_couplings"].median()),
            "coupling_density": float(adf["coupling_density"].iloc[0]),
            "sa_strictly_better_than_greedy": int((gap > 1e-6).sum()),
            "ties": int((gap.abs() <= 1e-6).sum()),
            "greedy_better": int((gap < -1e-6).sum()),
            "mean_energy_left_on_table": float(gap[gap > 1e-6].mean()) if (gap > 1e-6).any() else 0.0,
            "all_assignments_valid": bool(adf["valid_assignment"].all()),
            "median_sa_seconds": float(adf["sa_seconds"].median()),
        }

    # --- stage 3: classical baseline ---------------------------------------
    facts["vina_baseline"] = safe_json(
        os.path.join(RES, "vina_comparison_summary.json"))

    # --- stage 4: QAOA ------------------------------------------------------
    facts["qaoa"] = safe_json(os.path.join(RES, "qaoa_hardware.json"))

    out = os.path.join(RES, "presentation_facts.json")
    os.makedirs(RES, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(facts, fh, indent=2)
    print(json.dumps(facts, indent=2)[:4000])
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
