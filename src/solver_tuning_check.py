"""
solver_tuning_check.py
======================
Is simulated annealing's poor showing on the coupled assignment QUBO a real
property of the problem, or just an under-tuned sampler?

run_02 reported that greedy steepest descent found a strictly better solution
than simulated annealing on 12 of 25 real instances, with tabu search reaching
the best-known answer on all 25. Before writing that up as "classical methods
win at this size", it is worth separating two explanations:

  (a) the instances are easy enough that any decent local search solves them, or
  (b) our SA call was simply run with too few sweeps.

This builds fresh assignment QUBOs of the same shape (6 anchor atoms x 9 pocket
nodes = 54 variables) from the REAL CDK2 grid maps and the REAL pocket nodes
selected in run_02, then sweeps SA effort from cheap to expensive and compares
against tabu, steepest descent, and the best answer any method found.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qdock_core import GridSet
from qdock_qubo import build_assignment_qubo, solve_qubo

PROJ = "C:/Users/EngageAI/Downloads/CDK2_Project"
NODES = "results/pocket_nodes.npy"


class _Lig:
    """Minimal ligand stand-in: real atom types and charges, synthetic geometry."""

    def __init__(self, rng, n):
        types = ["OA", "NA", "A", "C", "N", "OA", "C", "A"][:n]
        self.ad_types = types
        self.elements = ["O" if t == "OA" else "N" if t in ("N", "NA") else "C"
                         for t in types]
        self.charges = rng.uniform(-0.5, 0.3, size=n)


def main():
    grids = GridSet(PROJ, "1HCK_atp")
    nodes = np.load(NODES)
    print(f"pocket nodes from run_02: {len(nodes)}")

    n_inst = 12
    n_anchors, n_nodes = 6, 9
    configs = [
        ("SA  100 reads x 1k sweeps", dict(method="sa", num_reads=100, num_sweeps=1000)),
        ("SA  200 reads x 2k sweeps", dict(method="sa", num_reads=200, num_sweeps=2000)),
        ("SA  500 reads x 10k sweeps", dict(method="sa", num_reads=500, num_sweeps=10000)),
        ("SA 1000 reads x 50k sweeps", dict(method="sa", num_reads=1000, num_sweeps=50000)),
        ("tabu 200 reads", dict(method="tabu", num_reads=200)),
        ("steepest 200 reads", dict(method="steepest", num_reads=200)),
    ]

    results = {k: [] for k, _ in configs}
    times = {k: 0.0 for k, _ in configs}

    for inst in range(n_inst):
        rng = np.random.default_rng(1000 + inst)
        lig = _Lig(rng, n_anchors)
        conf = rng.normal(scale=2.6, size=(n_anchors, 3))
        Q, _, meta = build_assignment_qubo(
            lig, conf, grids, nodes[:n_nodes], anchors=list(range(n_anchors)),
            lambda_geom=1.0)
        if inst == 0:
            print(f"instance shape: {meta['n_vars']} vars, "
                  f"{meta['n_geometry_couplings']} geometry couplings, "
                  f"density {meta['coupling_density']:.3f}\n")
        for name, kw in configs:
            t0 = time.perf_counter()
            _, e, _, _ = solve_qubo(Q, seed=7 + inst, **kw)
            times[name] += time.perf_counter() - t0
            results[name].append(e)

    best = np.min(np.array([results[k] for k, _ in configs]), axis=0)

    print(f"{'method':<28} {'best-known hits':>16} {'mean gap':>10} {'total s':>9}")
    print("-" * 66)
    for name, _ in configs:
        arr = np.array(results[name])
        hits = int(np.isclose(arr, best, atol=1e-6).sum())
        gap = float(np.mean(arr - best))
        print(f"{name:<28} {hits:>10}/{n_inst:<5} {gap:>10.3f} {times[name]:>9.2f}")

    print("\nInterpretation:")
    print("  If more SA effort closes the gap, the earlier result was a TUNING")
    print("  artefact and should be reported as such. If it does not, these")
    print("  instances are genuinely easy for classical local search, which is")
    print("  the honest finding to present.")


if __name__ == "__main__":
    main()
