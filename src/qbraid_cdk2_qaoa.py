"""
qbraid_cdk2_qaoa.py  --  RUN THIS IN qBraid Lab
================================================
QAOA on the CDK2 molecular-docking Ising Hamiltonian.

Self-contained: the Hamiltonian below was generated from the real AutoDock 4.2
grid maps of CDK2 (PDB 1HCK) and the real pocket nodes selected by the docking
pipeline. 3 ligand anchor atoms x 4 pocket sites = 12 binary variables,
so 12 qubits.

Why only 12 qubits when the full docking QUBO is 54 variables? The full problem
fits a 108-qubit chip on qubit count alone, but its coupling graph is nearly
complete (density ~0.95). After routing onto fixed hardware connectivity that
needs roughly 6,400 two-qubit gates, and at ~1% two-qubit error the output is
indistinguishable from noise. At 12 qubits the state space is 2^12 = 4096, small
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
N_QUBITS = 12
EXACT_GROUND_STATE = -373.79570164273036
QUBO = {(u, v): w for u, v, w in [(0, 0, -125.336255), (0, 1, 250.0), (0, 2, 250.0), (0, 3, 250.0), (0, 4, 125.0), (0, 6, 0.5171774317507108), (0, 7, 5.447831014534022), (0, 8, 125.0), (0, 9, 2.6181227388955466), (0, 10, 0.5722023710793672), (0, 11, 14.513433125945554), (1, 1, -125.66858), (1, 2, 250.0), (1, 3, 250.0), (1, 5, 125.0), (1, 7, 1.2333264871404879), (1, 8, 2.6181227388955466), (1, 9, 125.0), (1, 10, 2.2312853141282036), (1, 11, 0.13325329051876708), (2, 2, -125.14556), (2, 3, 250.0), (2, 4, 0.5171774317507108), (2, 6, 125.0), (2, 7, 3.564551662085104), (2, 8, 0.5722023710793672), (2, 9, 2.2312853141282036), (2, 10, 125.0), (2, 11, 11.313758646608838), (3, 3, -125.064105), (3, 4, 5.447831014534022), (3, 5, 1.2333264871404879), (3, 6, 3.564551662085104), (3, 7, 125.0), (3, 8, 14.513433125945554), (3, 9, 0.13325329051876708), (3, 10, 11.313758646608838), (3, 11, 125.0), (4, 4, -125.311165), (4, 5, 250.0), (4, 6, 250.0), (4, 7, 250.0), (4, 8, 125.0), (4, 9, 2.4342876670508318), (4, 10, 0.4880413591243661), (4, 11, 14.076070601170192), (5, 5, -125.66774), (5, 6, 250.0), (5, 7, 250.0), (5, 8, 2.4342876670508318), (5, 9, 125.0), (5, 10, 2.061830986190299), (5, 11, 0.09437044457711377), (6, 6, -125.16158), (6, 7, 250.0), (6, 8, 0.4880413591243661), (6, 9, 2.061830986190299), (6, 10, 125.0), (6, 11, 10.927996895169802), (7, 7, -125.093815), (7, 8, 14.076070601170192), (7, 9, 0.09437044457711377), (7, 10, 10.927996895169802), (7, 11, 125.0), (8, 8, -125.461975), (8, 9, 250.0), (8, 10, 250.0), (8, 11, 250.0), (9, 9, -125.47283), (9, 10, 250.0), (9, 11, 250.0), (10, 10, -125.42574), (10, 11, 250.0), (11, 11, -125.423265)]}

PRICING = {
    "cepheus": {"per_task": 30, "per_shot": 0.0425},
    "garnet":  {"per_task": 30, "per_shot": 0.145},
    "emerald": {"per_task": 30, "per_shot": 0.16},
    "forte":   {"per_task": 30, "per_shot": 8.0},
}


def qubo_to_ising(Q, n_vars):
    h = np.zeros(n_vars); J = {}; offset = 0.0
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
                print(f"{str(d.id):<46} {str(d.status()):<12} "
                      f"{d.metadata().get('num_qubits','?')} qubits")
            except Exception:
                print(f"{str(d.id):<46} (metadata unavailable)")
        return

    print(f"CDK2 docking Hamiltonian: {N_QUBITS} qubits, "
          f"{len([1 for k in QUBO if k[0]!=k[1]])} couplings")
    print(f"Exact ground state (brute force, 2^{N_QUBITS} states): "
          f"{EXACT_GROUND_STATE:.6f}")

    h, J, offset = qubo_to_ising(QUBO, N_QUBITS)
    scale = max(np.abs(h).max(), max((abs(v) for v in J.values()), default=1.0), 1.0)
    h_n = h / scale
    J_n = {k: v / scale for k, v in J.items()}

    from qiskit import transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator()

    print("\nTuning QAOA angles on the free local simulator...")
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
    print(f"  circuit depth {qc.depth()}, two-qubit gates {two_q}")

    counts_sim = sim.run(transpile(qc, sim), shots=args.shots).result().get_counts()
    _, e_sim, exp_sim, _ = evaluate_counts(counts_sim)
    print(f"  simulator best {e_sim:.6f} | mean {exp_sim:.4f}")
    print(f"  simulator found the exact ground state: "
          f"{abs(e_sim - EXACT_GROUND_STATE) < 1e-6}")

    report = {
        "n_qubits": N_QUBITS,
        "circuit_depth": qc.depth(),
        "two_qubit_gates": two_q,
        "shots": args.shots,
        "energy_exact": EXACT_GROUND_STATE,
        "energy_qaoa_simulator_best": float(e_sim),
        "energy_qaoa_simulator_mean": float(exp_sim),
        "simulator_found_ground_state": bool(abs(e_sim - EXACT_GROUND_STATE) < 1e-6),
    }

    if args.simulate_only:
        json.dump(report, open(args.out, "w"), indent=2)
        print("\nSimulation only. 0 credits spent.")
        return

    key = next((k for k in PRICING if k in args.device.lower()), None)
    if key is None:
        sys.exit(f"No pricing entry for '{args.device}'.")
    est = PRICING[key]["per_task"] + PRICING[key]["per_shot"] * args.shots
    print("\n" + "=" * 58)
    print(f"  Device         : {args.device}")
    print(f"  Shots          : {args.shots}")
    print(f"  Estimated cost : {est:.1f} credits  (${est/100:.2f})")
    print("=" * 58)
    if est > args.max_credits:
        sys.exit(f"Refusing: {est:.1f} exceeds --max-credits {args.max_credits}.")
    if not args.confirm:
        print("DRY RUN -- nothing submitted, nothing spent.")
        print("Add --confirm to actually submit.")
        return

    from qbraid.runtime import QbraidProvider
    provider = QbraidProvider()
    matches = [d for d in provider.get_devices()
               if args.device.lower() in str(d.id).lower()]
    if not matches:
        sys.exit(f"No device matching '{args.device}'. Try --list-devices.")
    device = matches[0]
    print(f"Submitting to {device.id} ...")
    job = device.run(qc, shots=args.shots)
    print(f"  job id: {job.id}")
    counts_hw = job.result().data.get_counts()
    _, e_hw, exp_hw, total = evaluate_counts(counts_hw)

    report.update({
        "hardware_device": str(device.id),
        "hardware_job_id": str(job.id),
        "energy_qaoa_hardware_best": float(e_hw),
        "energy_qaoa_hardware_mean": float(exp_hw),
        "hardware_found_ground_state": bool(abs(e_hw - EXACT_GROUND_STATE) < 1e-6),
        "estimated_credits_spent": round(est, 1),
        "shots_returned": int(total),
    })

    print("\n" + "=" * 58)
    print("SAME DOCKING HAMILTONIAN, THREE WAYS")
    print("=" * 58)
    print(f"  Exact ground state        : {EXACT_GROUND_STATE:10.4f}")
    print(f"  QAOA, noiseless simulator : {e_sim:10.4f}  (mean {exp_sim:.3f})")
    print(f"  QAOA, {str(device.id):<18}: {e_hw:10.4f}  (mean {exp_hw:.3f})")
    print("\n  The simulator-to-hardware gap IS your measured noise result.")
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
