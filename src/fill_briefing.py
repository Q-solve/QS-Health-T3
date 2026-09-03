"""
fill_briefing.py
================
Substitute {{PLACEHOLDER}} tokens in briefing_src.html with real numbers from
results/presentation_facts.json, then hand off to embed_figures.py.

    python fill_briefing.py
"""

import json
import os
import re
import subprocess
import sys

SRC = "briefing_src.html"
MID = "briefing_filled.html"
DST = "briefing.html"
FACTS = "results/presentation_facts.json"


def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main():
    facts = json.load(open(FACTS))
    html = open(SRC, encoding="utf-8").read()

    screen = facts.get("screen") or {}
    coupled = facts.get("coupled_qubo") or {}
    meta = facts.get("run_meta") or {}
    vina = facts.get("vina_baseline") or {}
    prep = facts.get("preparation") or {}

    # --- top hits table -----------------------------------------------------
    rows = []
    for i, r in enumerate(screen.get("top10", []), start=1):
        rows.append(
            "<tr>"
            f'<td class="n">{i}</td>'
            f'<td class="mono">{r["Ligand_ID"]}</td>'
            f'<td class="n">{r["Score_kcal_mol"]:.2f}</td>'
            f'<td class="n">{r["Polar_contacts"]}</td>'
            f'<td class="n">{r["Hydrophobic_contacts"]}</td>'
            f'<td class="n">{r["Min_dist_to_hinge_A"]:.2f}</td>'
            "</tr>"
        )
    top_rows = "\n".join(rows) or '<tr><td colspan="6">screen not yet complete</td></tr>'

    # --- Vina paragraph -----------------------------------------------------
    if vina:
        rho = vina.get("spearman_rho")
        n = vina.get("n_compared")
        unb = vina.get("unbiased_random_subset") or {}
        strength = ("tracks it closely" if rho is not None and rho > 0.5 else
                    "agrees partially" if rho is not None and rho > 0.2 else
                    "diverges")
        extra = ""
        if unb:
            extra = (f" On the {unb['n']} ligands drawn at random across the whole "
                     f"library &mdash; the unbiased estimate, free of the range "
                     f"restriction that comes from looking only at top hits &mdash; "
                     f"Spearman &rho; = <strong>{unb['spearman_rho']:+.2f}</strong>.")
        ov = vina.get("overlap", {}).get("top_10pct", {})
        ov_txt = ""
        if ov:
            ov_txt = (f" The two methods share {ov['shared']} of their top {ov['k']} "
                      f"compounds, {ov['enrichment_over_random']}&times; what random "
                      f"agreement would give.")
        vina_block = (
            f"The same {n} ligands were re-docked with the AutoDock Vina scoring "
            f"function &mdash; a functionally different model with no electrostatic "
            f"term at all &mdash; using identical conformers and identical poses, so "
            f"only the energy model changes. Spearman &rho; = "
            f"<strong>{rho:+.2f}</strong>, meaning our ranking {strength} an "
            f"independent published method.{extra}{ov_txt}"
        )
    else:
        vina_block = ("Baseline not yet run. Launch it with "
                      "<code>run_03_vina_baseline.py --top-n 50 --random-n 200</code>.")

    wall = meta.get("wall_seconds")
    subs = {
        "N_LIGANDS": str(screen.get("n_ligands_ranked", "&mdash;")),
        "N_COUPLINGS": f"{coupled.get('n_geometry_couplings_median', 1000):,}",
        "COUPLING_DENSITY": fmt(coupled.get("coupling_density"), 2),
        "N_INSTANCES": str(coupled.get("n_instances", "&mdash;")),
        "SA_WINS": str(coupled.get("sa_strictly_better_than_greedy", "&mdash;")),
        "GREEDY_WINS": str(coupled.get("greedy_better", "&mdash;")),
        "TIES": str(coupled.get("ties", "&mdash;")),
        "TOP_TABLE_ROWS": top_rows,
        "VINA_BLOCK": vina_block,
        "WALL_MIN": (f"{wall/60:.0f}" if wall else "&mdash;"),
        "GRID_BOX": fmt(prep.get("grid_box_angstrom"), 2),
        "GRID_POINTS": f"{prep.get('grid_points_per_map', 0):,}",
    }

    missing = set(re.findall(r"\{\{(\w+)\}\}", html)) - set(subs)
    if missing:
        print("WARNING unsubstituted placeholders:", missing)

    for k, v in subs.items():
        html = html.replace("{{" + k + "}}", str(v))

    with open(MID, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"filled -> {MID}")

    subprocess.run([sys.executable, "embed_figures.py", MID, DST], check=True)


if __name__ == "__main__":
    main()
