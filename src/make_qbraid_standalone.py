"""
make_qbraid_standalone.py
=========================
Emit ONE self-contained Python file to paste into qBraid Lab.

The generated script carries the real 12-qubit CDK2 docking Hamiltonian baked in
as literals, so it needs nothing from this project: no grid maps, no qdock_core,
no SANCDB. It only needs qiskit, qiskit-aer, dimod and qbraid, all of which are
present in a stock qBraid Lab environment.

    python make_qbraid_standalone.py  ->  qbraid_cdk2_qaoa.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qdock_core import GridSet
from qdock_qubo import build_assignment_qubo, qubo_to_ising, solve_qubo

PROJ = "C:/Users/EngageAI/Downloads/CDK2_Project"
NODES = "results/pocket_nodes.npy"
OUT = "qbraid_cdk2_qaoa.py"

N_ANCHORS, N_NODES = 3, 4


class _Lig:
    """Three real anchor-atom types with representative charges."""
    elements = ["O", "N", "C"]
    ad_types = ["OA", "NA", "A"]
    charges = np.array([-0.45, -0.35, 0.05])


def main():
    grids = GridSet(PROJ, "1HCK_atp")
    nodes = np.load(NODES)[:N_NODES]
    print(f"pocket nodes (real, from run_02): {nodes.shape}")

    rng = np.random.default_rng(2026)
    conf = rng.normal(scale=2.0, size=(N_ANCHORS, 3))

    Q, var_index, meta = build_assignment_qubo(
        _Lig(), conf, grids, nodes, anchors=list(range(N_ANCHORS)),
        lambda_geom=0.5)
    n = meta["n_vars"]
    print(f"QUBO: {n} vars, {meta['n_geometry_couplings']} geometry couplings")

    s_ex, e_ex, _, _ = solve_qubo(Q, method="exact")
    print(f"exact ground state: {e_ex:.6f}")
    h, J, offset = qubo_to_ising(Q, n)

    q_items = sorted(((int(u), int(v), float(w)) for (u, v), w in Q.items()))
    body = f'''"""
qbraid_cdk2_qaoa.py  --  RUN THIS IN qBraid Lab
================================================
QAOA on the CDK2 molecular-docking Ising Hamiltonian.

Self-contained: the Hamiltonian below was generated from the real AutoDock 4.2
grid maps of CDK2 (PDB 1HCK) and the real pocket nodes selected by the docking
pipeline. {N_ANCHORS} ligand anchor atoms x {N_NODES} pocket sites = {n} binary variables,
so {n} qubits.

Why only {n} qubits when the full docking QUBO is 54 variables? The full problem
fits a 108-qubit chip on qubit count alone, but its coupling graph is nearly
complete (density ~0.95). After routing onto fixed hardware connectivity that
needs roughly 6,400 two-qubit gates, and at ~1% two-qubit error the output is
indistinguishable from noise. At {n} qubits the state space is 2^{n} = {2**n}, small
enough that dimod.ExactSolver certifies the true ground state -- so we can state
exactly how close the hardware got, which is the whole point of the experiment.

Usage
-----
    python qbraid_cdk2_qaoa.py --list-devices        # free
    python qbraid_cdk2_qaoa.py --simulate-only       # free
    python qbraid_cdk2_qaoa.py --device cepheus      # dry run, spends NOTHING
    python qbraid_cdk2_qaoa.py --device cepheus --confirm    # SPENDS CREDITS
"""

import argparse, json, sys
import numpy as np

# ---------------------------------------------------------------------------
# The docking Hamiltonian, generated from the real CDK2 grids.
# ---------------------------------------------------------------------------
N_QUBITS = {n}
EXACT_GROUND_STATE = {e_ex!r}
QUBO = {{(u, v): w for u, v, w in {q_items!r}}}

PRICING = {{
    "cepheus": {{"per_task": 30, "per_shot": 0.0425}},
    "garnet":  {{"per_task": 30, "per_shot": 0.145}},
    "emerald": {{"per_task": 30, "per_shot": 0.16}},
    "forte":   {{"per_task": 30, "per_shot": 8.0}},
}}


def qubo_to_ising(Q, n_vars):
    h = np.zeros(n_vars); J = {{}}; offset = 0.0
    for (u, v), w in Q.items():
        if u == v:
            h[u] += w / 2.0; offset += w / 2.0
        else:
            J[(u, v)] = J.get((u, v), 0.0) + w / 4.0
            h[u] += w / 4.0; h[v] += w / 4.0; offset += w / 4.0
    return h, J, offset


def qubo_energy(bits):
    return sum(w * bits[u] * bits[v] if u != v else w * bits[u]
               for (u, v), w in QUBO.items())


def evaluate_counts(counts):
    best_bits, best_e, total, exp_e = None, np.inf, 0, 0.0
    for bitstr, c in counts.items():
        bits = [int(b) for b in bitstr[::-1]][:N_QUBITS]
        while len(bits) < N_QUBITS:
            bits.append(0)
        e = qubo_energy(bits)
        exp_e += e * c; total += c
        if e < best_e:
            best_e, best_bits = e, bits
    return best_bits, best_e, exp_e / max(1, total), total


def qaoa_circuit(h, J, gammas, betas):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))
    for gamma, beta in zip(gammas, betas):
        for i, hi in enumerate(h):
            if abs(hi) > 1e-9:
                qc.rz(2 * gamma * hi, i)
        for (i, j), Jij in J.items():
            if abs(Jij) > 1e-9:
                qc.cx(i, j); qc.rz(2 * gamma * Jij, j); qc.cx(i, j)
        qc.rx(2 * beta, range(N_QUBITS))
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--simulate-only", action="store_true")
    ap.add_argument("--device", default="cepheus")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--max-credits", type=float, default=150.0)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--out", default="qaoa_result.json")
    args = ap.parse_args()

    if args.list_devices:
        from qbraid.runtime import QbraidProvider
        for d in QbraidProvider().get_devices():
            try:
                print(f"{{str(d.id):<46}} {{str(d.status()):<12}} "
                      f"{{d.metadata().get('num_qubits','?')}} qubits")
            except Exception:
                print(f"{{str(d.id):<46}} (metadata unavailable)")
        return

    print(f"CDK2 docking Hamiltonian: {{N_QUBITS}} qubits, "
          f"{{len([1 for k in QUBO if k[0]!=k[1]])}} couplings")
    print(f"Exact ground state (brute force, 2^{{N_QUBITS}} states): "
          f"{{EXACT_GROUND_STATE:.6f}}")

    h, J, offset = qubo_to_ising(QUBO, N_QUBITS)
    scale = max(np.abs(h).max(), max((abs(v) for v in J.values()), default=1.0), 1.0)
    h_n = h / scale
    J_n = {{k: v / scale for k, v in J.items()}}

    from qiskit import transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator()

    print("\\nTuning QAOA angles on the free local simulator...")
    rng = np.random.default_rng(11)
    best_params, best_exp = None, np.inf
    for _ in range(40):
        g = rng.uniform(0, np.pi, args.layers)
        b = rng.uniform(0, np.pi / 2, args.layers)
        qc = qaoa_circuit(h_n, J_n, g, b)
        counts = sim.run(transpile(qc, sim), shots=512).result().get_counts()
        _, _, exp_e, _ = evaluate_counts(counts)
        if exp_e < best_exp:
            best_exp, best_params = exp_e, (g, b)
    gammas, betas = best_params

    qc = qaoa_circuit(h_n, J_n, gammas, betas)
    two_q = sum(1 for inst in qc.data if len(inst.qubits) == 2)
    print(f"  circuit depth {{qc.depth()}}, two-qubit gates {{two_q}}")

    counts_sim = sim.run(transpile(qc, sim), shots=args.shots).result().get_counts()
    _, e_sim, exp_sim, _ = evaluate_counts(counts_sim)
    print(f"  simulator best {{e_sim:.6f}} | mean {{exp_sim:.4f}}")
    print(f"  simulator found the exact ground state: "
          f"{{abs(e_sim - EXACT_GROUND_STATE) < 1e-6}}")

    report = {{
        "n_qubits": N_QUBITS,
        "circuit_depth": qc.depth(),
        "two_qubit_gates": two_q,
        "shots": args.shots,
        "energy_exact": EXACT_GROUND_STATE,
        "energy_qaoa_simulator_best": float(e_sim),
        "energy_qaoa_simulator_mean": float(exp_sim),
        "simulator_found_ground_state": bool(abs(e_sim - EXACT_GROUND_STATE) < 1e-6),
    }}

    if args.simulate_only:
        json.dump(report, open(args.out, "w"), indent=2)
        print("\\nSimulation only. 0 credits spent.")
        return

    key = next((k for k in PRICING if k in args.device.lower()), None)
    if key is None:
        sys.exit(f"No pricing entry for '{{args.device}}'.")
    est = PRICING[key]["per_task"] + PRICING[key]["per_shot"] * args.shots
    print("\\n" + "=" * 58)
    print(f"  Device         : {{args.device}}")
    print(f"  Shots          : {{args.shots}}")
    print(f"  Estimated cost : {{est:.1f}} credits  (${{est/100:.2f}})")
    print("=" * 58)
    if est > args.max_credits:
        sys.exit(f"Refusing: {{est:.1f}} exceeds --max-credits {{args.max_credits}}.")
    if not args.confirm:
        print("DRY RUN -- nothing submitted, nothing spent.")
        print("Add --confirm to actually submit.")
        return

    from qbraid.runtime import QbraidProvider
    provider = QbraidProvider()
    matches = [d for d in provider.get_devices()
               if args.device.lower() in str(d.id).lower()]
    if not matches:
        sys.exit(f"No device matching '{{args.device}}'. Try --list-devices.")
    device = matches[0]
    print(f"Submitting to {{device.id}} ...")
    job = device.run(qc, shots=args.shots)
    print(f"  job id: {{job.id}}")
    counts_hw = job.result().data.get_counts()
    _, e_hw, exp_hw, total = evaluate_counts(counts_hw)

    report.update({{
        "hardware_device": str(device.id),
        "hardware_job_id": str(job.id),
        "energy_qaoa_hardware_best": float(e_hw),
        "energy_qaoa_hardware_mean": float(exp_hw),
        "hardware_found_ground_state": bool(abs(e_hw - EXACT_GROUND_STATE) < 1e-6),
        "estimated_credits_spent": round(est, 1),
        "shots_returned": int(total),
    }})

    print("\\n" + "=" * 58)
    print("SAME DOCKING HAMILTONIAN, THREE WAYS")
    print("=" * 58)
    print(f"  Exact ground state        : {{EXACT_GROUND_STATE:10.4f}}")
    print(f"  QAOA, noiseless simulator : {{e_sim:10.4f}}  (mean {{exp_sim:.3f}})")
    print(f"  QAOA, {{str(device.id):<18}}: {{e_hw:10.4f}}  (mean {{exp_hw:.3f}})")
    print("\\n  The simulator-to-hardware gap IS your measured noise result.")
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\\n-> {{args.out}}")


if __name__ == "__main__":
    main()
'''

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(body)
    print(f"\nwrote {OUT}  ({os.path.getsize(OUT)/1024:.1f} KB, "
          f"{len(body.splitlines())} lines)")


if __name__ == "__main__":
    main()
