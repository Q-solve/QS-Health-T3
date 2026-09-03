# Quantum-annealing molecular docking against CDK2

**Q-SOLVE Kenya 2026 · SDG 3 · Strathmore University, Nairobi**

Screening South African natural products (SANCDB) for inhibitors of **CDK2**, a
cell-cycle kinase cancer cells depend on, by casting molecular docking as a
**QUBO / Ising** optimisation problem and solving it with annealing — classically,
and on real quantum hardware via QAOA.

---

## Headline results

| Result | Value |
|---|---|
| **ATP redocking control** (validation) | **0.49 Å** RMSD to the crystal pose — literature bar is 2.0 Å |
| Crystal ATP interaction energy | −8.87 kcal/mol, 3.05 Å from the Leu83 hinge |
| Compounds screened and ranked | 1012 |
| Best hit | SANC00131, −7.15 kcal/mol, 2.29 Å from the hinge |
| Coupled assignment QUBO | 54 variables, ~1,022 geometry couplings, density 0.95 |
| **QAOA on IQM Garnet** (real 20-qubit QPU) | best of 256 shots **−373.686** vs exact **−373.796** |
| Hardware mean energy | **+295** vs **+784** for uniform random sampling |

The redocking control is the number to trust first: remove ATP from the 1HCK
crystal structure, re-dock it blind from 3,000 random orientations, and the
best-scoring pose lands 0.49 Å from the experimentally observed one — and the
lowest-energy pose is also the most accurate. Its contact profile recovers the
canonical CDK2 site (Glu81/Leu83 hinge; Ile10, Val18, Ala31, Phe82, Leu134
hydrophobic pocket; catalytic Lys33 and Asp145) without anything being tuned to
produce it.

---

## What is actually quantum here

Stated plainly, because this is the question that matters:

| Component | Status |
|---|---|
| QUBO / Ising formulation | Quantum-**ready**. Verified equivalent to machine precision (7×10⁻¹⁵ across all 1,024 states of a test problem). |
| Simulated annealing (the screen) | **Classical.** The quantum-inspired stand-in for a D-Wave annealer, running the identical QUBO. |
| QAOA on Qiskit Aer | A real quantum **algorithm**, simulated classically. |
| **QAOA on IQM Garnet** | **Real quantum hardware.** 12 qubits, 222 gates, depth 189, 256 shots. |

**No quantum speedup is claimed.** On the 54-variable coupled QUBO, classical
tabu search reached the best known solution in **25 of 25** instances; simulated
annealing managed 13 of 25 and lost outright to greedy steepest descent in 12.
The case for quantum hardware here is about **scaling**, not present-day
advantage — see `docs/briefing.html`.

---

## Method

1. **Receptor preparation** — PDB `1HCK` downloaded, protein split from the
   crystallographic ATP, AMBER charges and protonation assigned with `pdb2pqr`,
   AutoDock atom types assigned, non-polar hydrogens merged.
2. **Grid maps** — the AutoDock 4.2 force field reimplemented in NumPy
   (`src/autogrid_py.py`): 12-6 dispersion/repulsion, directional 12-10 hydrogen
   bonding, Mehler–Solmajer distance-dependent electrostatics, and desolvation.
   13 atom-type maps plus separate electrostatic and desolvation grids, on a
   24.75 Å box at 0.375 Å spacing (300,763 points per map).
3. **Ligand preparation** — 6–8 conformers per compound via RDKit ETKDGv3 + MMFF,
   Gasteiger charges.
4. **Pose sampling** — uniform random orientations from quaternions (Shoemake),
   random translation in a 3 Å sphere, then Powell local refinement over 6 DOF.
5. **QUBO formulation** — two encodings, both reported:
   - *pose selection* — trivial couplings, kept as an honest baseline;
   - *atom-to-node assignment* — geometric consistency
     `(d_ij − D_ab)²` in the couplings, making it a quadratic assignment problem.
6. **Solving** — `dwave-samplers` simulated annealing, tabu, steepest descent;
   QAOA via Qiskit, submitted to IQM Garnet through qBraid.

---

## Reproducing it

```bash
pip install numpy scipy pandas matplotlib rdkit pdb2pqr dimod dwave-samplers qiskit qiskit-aer
```

```bash
cd src

# 1. structures + grid maps  (maps are NOT in this repo; they regenerate in ~3 s)
python prep_structures.py
python autogrid_py.py --receptor ../data/1HCK_rec.pdbqt --out-dir ../data \
    --prefix 1HCK_atp --center 100.541 97.891 81.717 --npts 66 66 66

# 2. the positive control — run this first, always
python run_01_redock_control.py --maps ../data --receptor ../data/1HCK_rec.pdbqt \
    --ligand ../data/ATP_ref.pdbqt --n-poses 3000 --n-refine 40

# 3. the screen
python run_02_screen.py --sdf ../data/sancdb_prepared_3d.sdf --maps ../data \
    --receptor ../data/1HCK_rec.pdbqt --sample 300 --n-confs 6 --n-poses 48 --n-refine 3

# 4. figures
python make_figures_concept.py && python make_figures_results.py

# 5. QAOA — free simulator, spends nothing
python qbraid_cdk2_qaoa.py --simulate-only
```

`prep_structures.py` expects `1HCK_protein.pqr`; regenerate it with
`pdb2pqr30 --ff=AMBER --keep-chain --whitespace 1HCK_protein.pdb 1HCK_protein.pqr`.

Use `--sample N` rather than `--limit N`: SANCDB accession order correlates with
molecular size (the first 250 entries average 24.9 heavy atoms, the next 50
average 38.2), so `--limit` silently under-samples large flexible compounds.

---

## Running on quantum hardware

`src/qaoa_cdk2.qasm` is the exact OpenQASM 3 circuit that ran on IQM Garnet.
Paste it into the [qBraid Composer](https://account.qbraid.com/composer) and hit
Run, or use the self-contained `src/qbraid_cdk2_qaoa.py`.

Pricing differs sharply by device: IQM Garnet bills per task plus per shot
(30 cr + 0.145 cr/shot, so the cost is known in advance), whereas Rigetti Cepheus
bills by **time on the processor** at 12,000 credits/minute, which cannot be
predicted before the job runs. The **Open Quantum (OQ) versions of every QPU are
free** once an Open Quantum account is linked.

Job submitted: `aws:iqm:qpu:garnet-09f9-qjob-6a990faae026787d1e5ad50b`
(256 shots, 3.76 s QPU time, 67.12 credits).

---

## Honest limitations

- **Rigid-body docking.** Ligand flexibility is approximated by a conformer
  ensemble; torsions are not sampled during the search.
- **Rigid receptor.** CDK2's glycine-rich loop is mobile and is held fixed.
- **Scores are interaction energies, not ΔG** — no torsional-entropy penalty
  (~+0.3 kcal/mol per rotatable bond), no intramolecular ligand energy. Valid for
  ranking, not for quoting as affinities.
- **Hydrogen bonds have no angular term.** The 12-10 potential omits AutoGrid's
  directional weighting, and the contact profiler uses a distance cutoff only —
  hence "polar contacts", not "hydrogen bonds".
- **RMSD is not symmetry-corrected.**
- Waters and the catalytic Mg²⁺ were removed (standard for ATP-competitive
  screening), which makes ATP's own triphosphate tail harder to place.
- The screened subset is 300 compounds, not the full 1,002-compound library.
- **No experimental validation.** These are computational predictions; the next
  step is an in-vitro CDK2 kinase inhibition assay.
- The hardware instance is a reduced 3-atom × 4-node problem (12 qubits). The full
  54-variable QUBO fits a 108-qubit chip on qubit count, but its near-complete
  coupling graph needs ~6,400 two-qubit gates after routing — noise-dominated at
  current error rates, and 2⁵⁴ states cannot be brute-forced for verification.

---

## A bug worth documenting

The grid maps this project started with were centred at (−9.34, −10.64, 8.49)
while RCSB 1HCK sits near (100, 98, 82). No translation, centroid convention,
unit-cell vector or P2₁2₁2₁ symmetry operator relates the two frames, and the
maps also lacked electrostatic and desolvation terms. Every score computed
against them was meaningless while still looking plausible.

Everything here is therefore rebuilt from the accession code alone, and the
redocking control exists specifically so that a failure of that kind cannot pass
unnoticed again. The grid engine was cross-checked against a direct pairwise
summation over all 2,877 receptor atoms: −9.20 kcal/mol direct versus −8.87 via
the grid, a 3.6 % gap that is exactly the expected trilinear interpolation error.

---

## Layout

```
src/        pipeline code — preparation, grids, QUBO, solvers, figures
data/       1HCK structures, prepared receptor/ligand, SANCDB library
results/    ranked hits, solver comparison, QAOA hardware result, run metadata
figures/    six presentation figures (200 dpi)
docs/       briefing.html (full technical writeup), CHEATSHEET.md
```

Grid `.map` files are excluded — 44 MB, and `autogrid_py.py` regenerates them in
about three seconds.
