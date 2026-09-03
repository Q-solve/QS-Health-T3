# Classical Molecular Docking — CDK2 (SANCDB vs. PDB 1HCK)

Classical docking stage of our **Quantum-Driven Platform for Discovery of Novel CDK2 Inhibitors** (OQI Hackathon). This notebook screens the SANCDB natural-product library against the prepared CDK2 receptor using AutoDock Vina, producing ranked binding affinities that feed into the downstream quantum-refinement stages of the pipeline.

## What this notebook does

1. **Ligand preparation** — converts the SANCDB SDF library (`sancdb_prepared_3d.sdf`) into individual `.pdbqt` files via OpenBabel, adding explicit hydrogens and sanitizing headers for Vina compatibility.
2. **Receptor** — docks against `1HCK_receptor.pdbqt`, the prepared CDK2 structure (ATP-pocket grid box centered on the hinge region).
3. **Batch virtual screening** — runs AutoDock Vina across the full ligand set (up to ~1,002 compounds), in parallel across CPU cores, with resume-on-restart and per-ligand timeout/error handling so a single bad structure doesn't kill the run.
4. **Pose validation** — heavy-atom RMSD check between reference and docked poses.
5. **Results export** — ranked binding affinities (kcal/mol) written to CSV/Excel (`binding_affinities.csv`, `1HCK_Detailed_Classical_Hits.xlsx`).

## Docking parameters

| Parameter | Value |
|---|---|
| Receptor | `1HCK_receptor.pdbqt` (CDK2, ATP-binding site) |
| Grid center | `(11.5, 20.1, 55.4)` |
| Grid size | `20 x 20 x 20` Å |
| Exhaustiveness | 8 |
| Poses per ligand | 1 |

## Requirements

```bash
pip install vina openbabel pandas openpyxl
apt-get install -y swig libboost-all-dev autodock-vina openbabel
```

## Inputs expected in the working directory

- `sancdb_prepared_3d.sdf` — curated SANCDB ligand library (3D, protonated)
- `1HCK_receptor.pdbqt` (or `1hck_receptor.pdbqt`) — prepared CDK2 receptor

## Outputs

- `sancdb_pdbqt/` / `prepared_ligands/` — per-ligand `.pdbqt` files
- `docked_poses/` — best docked pose per ligand
- `results/binding_affinities.csv` — ligand ID + binding affinity (kcal/mol) + status
- `1HCK_Detailed_Classical_Hits.xlsx` / `.csv` — full ranked hit list

## Notes

- Designed to resume cleanly if interrupted (skips ligands already present in `docked_poses/`).
- Failed ligands are logged separately rather than stopping the batch.
- Output of this stage (ranked classical binding affinities) is the input to the pipeline's quantum-informed ranking stage.

## Part of

**A Quantum-Driven Platform for Discovery of Novel CDK2 Inhibitors** — OQI Hackathon, integrating African biodiversity (SANCDB), experimental structural biology (PDB 1HCK + AlphaFold gap-filling), classical docking, and quantum-enhanced interaction descriptors.
