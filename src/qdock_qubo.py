"""
qdock_qubo.py
=============
Two QUBO formulations of docking, plus solvers and an Ising export for QAOA.

FORMULATION A -- pose selection (the one in the original submission)
    Variables : x_p = 1 if candidate pose p is chosen
    Objective : min  sum_p E_p x_p  +  lambda (sum_p x_p - 1)^2
    Couplings : constant 2*lambda, carrying NO physics
    Honest status: mathematically a valid QUBO, computationally trivial. Its
    global minimum is argmin(E_p). Kept as a BASELINE so we can show the
    contrast, not as the headline result.

FORMULATION B -- atom-to-node assignment  <-- physics lives in the couplings
    This is the weighted subgraph isomorphism encoding used in the quantum
    docking literature.

    Variables : x_{i,a} = 1 if ligand anchor atom i is placed on pocket node a
    Linear    : the AutoGrid interaction energy of atom i's type at node a
    Quadratic : lambda_geom * (d_ij - D_ab)^2  on x_{i,a} x_{j,b}
                where d_ij is the intramolecular ligand atom-atom distance and
                D_ab is the pocket node-node distance. This term penalises
                assignments that stretch or compress the ligand -- it is a real
                geometric consistency constraint, and it is what makes the
                problem a quadratic assignment problem (NP-hard).
    Penalties : one node per atom, at most one atom per node.

    Because the couplings encode geometry, the minimum is NOT the smallest
    diagonal entry and cannot be found by inspection. This is the formulation
    that justifies an annealer.
"""

from __future__ import annotations

import itertools
import time

import numpy as np


# ----------------------------------------------------------------------------
# Formulation A: pose selection (baseline, trivial)
# ----------------------------------------------------------------------------

def build_pose_selection_qubo(energies, lam=None):
    energies = np.asarray(energies, dtype=float)
    n = len(energies)
    if lam is None:
        lam = max(50.0, 2.0 * float(np.abs(energies).max()))
    Q = {}
    for i in range(n):
        Q[(i, i)] = float(energies[i] - lam)
        for j in range(i + 1, n):
            Q[(i, j)] = float(2.0 * lam)
    return Q, lam


# ----------------------------------------------------------------------------
# Pocket node selection
# ----------------------------------------------------------------------------

def select_pocket_nodes(grids, n_nodes=8, min_sep=2.5, probe_type="C",
                        search_radius=8.0, energy_ceiling=0.0):
    """Pick favourable, well-separated grid points as the target-graph nodes.

    Greedy farthest-point-style selection over grid points whose carbon-probe
    energy is below `energy_ceiling`, keeping nodes at least `min_sep` apart so
    the target graph is not degenerate.
    """
    gm = grids.maps.get(probe_type) or next(iter(grids.maps.values()))
    nx, ny, nz = gm.data.shape
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    coords = gm.origin + np.stack([ii, jj, kk], axis=-1).reshape(-1, 3) * gm.spacing
    vals = gm.data.reshape(-1)

    keep = np.linalg.norm(coords - grids.center, axis=1) <= search_radius
    keep &= vals < energy_ceiling
    coords, vals = coords[keep], vals[keep]
    if len(coords) == 0:
        raise ValueError("No favourable pocket nodes found. Widen search_radius "
                         "or raise energy_ceiling.")

    order = np.argsort(vals)
    chosen = []
    for idx in order:
        c = coords[idx]
        if all(np.linalg.norm(c - other) >= min_sep for other in chosen):
            chosen.append(c)
        if len(chosen) == n_nodes:
            break
    return np.array(chosen)


def select_anchor_atoms(ligand, conf_coords, n_anchors=5):
    """Farthest-point sampling over heavy atoms, seeded with polar atoms.

    Polar atoms are seeded first because they carry the hinge pharmacophore;
    farthest-point sampling then spreads the rest so the ligand graph spans the
    whole molecule rather than clustering on one ring.
    """
    heavy = [i for i, e in enumerate(ligand.elements) if e != "H"]
    polar = [i for i in heavy if ligand.elements[i] in ("N", "O")]
    chosen = polar[:2] if polar else [heavy[0]]

    while len(chosen) < min(n_anchors, len(heavy)):
        best, best_d = None, -1.0
        for i in heavy:
            if i in chosen:
                continue
            d = min(np.linalg.norm(conf_coords[i] - conf_coords[c]) for c in chosen)
            if d > best_d:
                best, best_d = i, d
        chosen.append(best)
    return chosen


# ----------------------------------------------------------------------------
# Formulation B: atom-to-node assignment (physics in couplings)
# ----------------------------------------------------------------------------

def build_assignment_qubo(ligand, conf_coords, grids, nodes, anchors,
                          lambda_geom=1.0, lambda_onehot=None,
                          geom_tolerance=0.3, geom_cap=25.0):
    """Return (Q, var_index, meta).

    var_index maps (atom_local_index, node_index) -> variable id.
    """
    n_a, n_n = len(anchors), len(nodes)
    var_index = {(i, a): i * n_n + a for i in range(n_a) for a in range(n_n)}
    n_vars = n_a * n_n

    # --- linear terms: grid energy of this atom type at this node -------------
    linear = np.zeros(n_vars)
    for i, atom_idx in enumerate(anchors):
        t = ligand.ad_types[atom_idx]
        q = ligand.charges[atom_idx]
        for a in range(n_n):
            ea, _, _ = grids.score(nodes[a:a + 1], [t], np.array([q]))
            linear[var_index[(i, a)]] = ea

    # --- pairwise geometric consistency: THE physics coupling -----------------
    d_lig = np.zeros((n_a, n_a))
    for i, j in itertools.combinations(range(n_a), 2):
        d_lig[i, j] = d_lig[j, i] = np.linalg.norm(
            conf_coords[anchors[i]] - conf_coords[anchors[j]]
        )
    d_node = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=2)

    quad: dict[tuple[int, int], float] = {}

    def add_quad(u, v, w):
        if u == v:
            linear[u] += w
            return
        key = (min(u, v), max(u, v))
        quad[key] = quad.get(key, 0.0) + w

    n_geom_terms = 0
    for i, j in itertools.combinations(range(n_a), 2):
        for a in range(n_n):
            for b in range(n_n):
                if a == b:
                    continue
                mismatch = abs(d_lig[i, j] - d_node[a, b])
                if mismatch <= geom_tolerance:
                    continue
                w = min(lambda_geom * mismatch ** 2, geom_cap)
                add_quad(var_index[(i, a)], var_index[(j, b)], w)
                n_geom_terms += 1

    # --- constraint penalties -------------------------------------------------
    scale = max(np.abs(linear).max(), 1.0)
    if lambda_onehot is None:
        lambda_onehot = 5.0 * max(scale, geom_cap)

    # each atom on exactly one node: lam*(sum_a x_ia - 1)^2
    for i in range(n_a):
        for a in range(n_n):
            linear[var_index[(i, a)]] -= lambda_onehot
            for b in range(a + 1, n_n):
                add_quad(var_index[(i, a)], var_index[(i, b)], 2.0 * lambda_onehot)

    # at most one atom per node: lam*sum_{i<j} x_ia x_ja
    for a in range(n_n):
        for i in range(n_a):
            for j in range(i + 1, n_a):
                add_quad(var_index[(i, a)], var_index[(j, a)], lambda_onehot)

    Q = {(u, u): float(linear[u]) for u in range(n_vars)}
    for (u, v), w in quad.items():
        Q[(u, v)] = float(Q.get((u, v), 0.0) + w)

    meta = {
        "n_vars": n_vars,
        "n_anchors": n_a,
        "n_nodes": n_n,
        "n_geometry_couplings": n_geom_terms,
        "n_nonzero_couplings": len(quad),
        "coupling_density": len(quad) / max(1, n_vars * (n_vars - 1) / 2),
        "lambda_geom": lambda_geom,
        "lambda_onehot": lambda_onehot,
    }
    return Q, var_index, meta


def decode_assignment(sample, var_index, n_anchors, n_nodes):
    """Turn a QUBO sample into {atom_local_index: node_index} and report validity."""
    assignment, violations = {}, []
    for i in range(n_anchors):
        chosen = [a for a in range(n_nodes) if sample.get(var_index[(i, a)], 0) == 1]
        if len(chosen) == 1:
            assignment[i] = chosen[0]
        else:
            violations.append(f"atom {i} assigned to {len(chosen)} nodes")
    used = list(assignment.values())
    if len(set(used)) != len(used):
        violations.append("two atoms share a node")
    return assignment, violations


# ----------------------------------------------------------------------------
# Solvers
# ----------------------------------------------------------------------------

def _sampler(kind: str):
    if kind == "sa":
        try:
            from dwave.samplers import SimulatedAnnealingSampler
        except ImportError:
            from neal import SimulatedAnnealingSampler
        return SimulatedAnnealingSampler()
    if kind == "tabu":
        from dwave.samplers import TabuSampler
        return TabuSampler()
    if kind == "steepest":
        from dwave.samplers import SteepestDescentSampler
        return SteepestDescentSampler()
    raise ValueError(f"unknown sampler {kind}")


def solve_qubo(Q, method="sa", num_reads=200, num_sweeps=2000, seed=7):
    """Solve a QUBO and return (best_sample, best_energy, wall_seconds, meta)."""
    import dimod

    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    t0 = time.perf_counter()

    if method == "exact":
        if bqm.num_variables > 22:
            raise ValueError("exact solver refuses > 22 variables")
        ss = dimod.ExactSolver().sample(bqm)
    elif method == "sa":
        ss = _sampler("sa").sample(bqm, num_reads=num_reads,
                                   num_sweeps=num_sweeps, seed=seed)
    elif method == "tabu":
        ss = _sampler("tabu").sample(bqm, num_reads=num_reads, seed=seed)
    elif method == "steepest":
        ss = _sampler("steepest").sample(bqm, num_reads=num_reads, seed=seed)
    else:
        raise ValueError(method)

    elapsed = time.perf_counter() - t0
    best = ss.first
    energies = np.array([r.energy for r in ss.data(["energy"])])
    meta = {
        "method": method,
        "n_vars": bqm.num_variables,
        "n_interactions": len(bqm.quadratic),
        "seconds": elapsed,
        "best_energy": float(best.energy),
        "mean_energy": float(energies.mean()),
        "frac_at_best": float(np.mean(np.isclose(energies, best.energy, atol=1e-6))),
    }
    return dict(best.sample), float(best.energy), elapsed, meta


# ----------------------------------------------------------------------------
# Ising export for QAOA on gate-based hardware
# ----------------------------------------------------------------------------

def qubo_to_ising(Q, n_vars):
    """QUBO (x in {0,1}) -> Ising (s in {-1,+1}) via x = (1 + s)/2.

    Returns (h, J, offset). This is the mapping that makes the SAME objective
    runnable on an annealer and, through QAOA, on a gate-based QPU.
    """
    h = np.zeros(n_vars)
    J: dict[tuple[int, int], float] = {}
    offset = 0.0
    for (u, v), w in Q.items():
        if u == v:
            h[u] += w / 2.0
            offset += w / 2.0
        else:
            J[(u, v)] = J.get((u, v), 0.0) + w / 4.0
            h[u] += w / 4.0
            h[v] += w / 4.0
            offset += w / 4.0
    return h, J, offset
