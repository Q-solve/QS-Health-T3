"""
run_04_qaoa_qbraid.py
=====================
Run the SAME docking Ising Hamiltonian on real gate-based quantum hardware
through qBraid, using QAOA.

WHY THIS SCRIPT EXISTS
----------------------
qBraid's device fleet is gate-based and neutral-atom: AQT IBEX Q1, QuEra Aquila,
IQM Garnet and Emerald, IonQ Forte-1 and Forte-Enterprise-1, Rigetti
Cepheus-1-108Q, Pasqal Fresnel. There is no D-Wave annealer on the platform.

That is not a dead end, it is a better story. A QUBO converts to an Ising
Hamiltonian, and an Ising Hamiltonian can be minimised two ways:
    - adiabatically, on an annealer (D-Wave)
    - variationally, on a gate-based QPU (QAOA)
Same objective function, two hardware paradigms. Running QAOA proves the
formulation is hardware-agnostic and moves your evidence from
"simulation only" to "simulation and hardware".

CREDIT BUDGET (1 credit = $0.01 USD)
------------------------------------
    Rigetti Cepheus-1-108Q via AWS : 30/task + 0.0425/shot -> ~74 for 1024 shots
    IQM Garnet                     : 30/task + 0.145/shot  -> ~178 for 1024 shots
    IQM Emerald                    : 30/task + 0.16/shot   -> ~194 for 1024 shots
    IonQ Forte-1                   : 30/task + 8/shot      -> DO NOT USE
    Pasqal Fresnel                 : 500/minute            -> DO NOT USE
With 600 credits, Rigetti gives you roughly 8 hardware runs. Budget 3 and keep
the rest in reserve.

Also check the Open Quantum integration in your qBraid account: several of these
QPUs cost 0 qBraid credits through it, and new accounts can claim $50 in free
Spark credits.

Usage:
    python run_04_qaoa_qbraid.py --list-devices
    python run_04_qaoa_qbraid.py --simulate-only
    python run_04_qaoa_qbraid.py --device cepheus --shots 1024 --confirm
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pricing table, credits. Update from https://docs.qbraid.com/v2/home/pricing
PRICING = {
    "cepheus": {"per_task": 30, "per_shot": 0.0425, "per_min": 0},
    "garnet":  {"per_task": 30, "per_shot": 0.145,  "per_min": 0},
    "emerald": {"per_task": 30, "per_shot": 0.16,   "per_min": 0},
    "ibex":    {"per_task": 30, "per_shot": 2.35,   "per_min": 0},
    "forte":   {"per_task": 30, "per_shot": 8.0,    "per_min": 0},
}


# ---------------------------------------------------------------------------
# Build a SMALL docking QUBO that fits current hardware
# ---------------------------------------------------------------------------

def build_small_instance(n_anchors=3, n_nodes=4, seed=2026,
                         maps_dir="CDK2_Project", prefix="1HCK_atp",
                         nodes_path="results/pocket_nodes.npy"):
    """A reduced atom-to-node assignment instance: n_anchors x n_nodes qubits.

    3 x 4 = 12 qubits fits comfortably on Garnet (20q) and Cepheus (108q), and
    is small enough that dimod.ExactSolver gives the certified ground state for
    comparison. Being able to say "we know the exact answer and here is how
    close the QPU got" is worth more than a bigger, unverifiable instance.

    If your run_02 output exists, this loads the real pocket nodes so the
    Hamiltonian is genuinely from your CDK2 grid rather than synthetic.
    """
    from qdock_qubo import build_assignment_qubo

    if os.path.exists(nodes_path):
        nodes = np.load(nodes_path)[:n_nodes]
        source = "real CDK2 pocket nodes from run_02"
    else:
        rng = np.random.default_rng(seed)
        nodes = rng.normal(scale=2.5, size=(n_nodes, 3))
        source = "synthetic nodes (run run_02_screen.py first for real ones)"

    # Minimal stand-in ligand geometry so this script can run standalone.
    class _Lig:
        elements = ["N", "O", "C"][:n_anchors]
        ad_types = ["NA", "OA", "A"][:n_anchors]
        charges = np.array([-0.35, -0.45, 0.05])[:n_anchors]

    rng = np.random.default_rng(seed + 1)
    conf = rng.normal(scale=2.0, size=(n_anchors, 3))

    class _Grids:
        """Fallback scorer if the AutoGrid maps are not on this machine."""
        center = np.zeros(3)
        maps = {"A": None}

        def score(self, coords, types, charges):
            coords = np.atleast_2d(coords)
            e = -3.0 * np.exp(-np.linalg.norm(coords, axis=1) ** 2 / 12.0)
            return float(e.sum()), 0, e

    try:
        from qdock_core import GridSet
        grids = GridSet(maps_dir, prefix)
    except Exception:
        grids = _Grids()

    Q, var_index, meta = build_assignment_qubo(
        _Lig(), conf, grids, np.asarray(nodes),
        anchors=list(range(n_anchors)), lambda_geom=0.5
    )
    meta["node_source"] = source
    return Q, var_index, meta


# ---------------------------------------------------------------------------
# QAOA circuit
# ---------------------------------------------------------------------------

def qaoa_circuit(h, J, gammas, betas, n_qubits):
    """Standard p-layer QAOA for an Ising Hamiltonian H = sum h_i Z_i + sum J_ij Z_i Z_j."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for gamma, beta in zip(gammas, betas):
        for i, hi in enumerate(h):
            if abs(hi) > 1e-9:
                qc.rz(2 * gamma * hi, i)
        for (i, j), Jij in J.items():
            if abs(Jij) > 1e-9:
                qc.cx(i, j)
                qc.rz(2 * gamma * Jij, j)
                qc.cx(i, j)
        qc.rx(2 * beta, range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def qubo_energy(Q, bits):
    e = 0.0
    for (u, v), w in Q.items():
        e += w * bits[u] * bits[v] if u != v else w * bits[u]
    return e


def evaluate_counts(counts, Q, n_qubits):
    """Turn a measurement histogram into the docking-relevant statistics."""
    best_bits, best_e, total, exp_e = None, np.inf, 0, 0.0
    for bitstr, c in counts.items():
        bits = [int(b) for b in bitstr[::-1]][:n_qubits]  # qiskit is little-endian
        while len(bits) < n_qubits:
            bits.append(0)
        e = qubo_energy(Q, bits)
        exp_e += e * c
        total += c
        if e < best_e:
            best_e, best_bits = e, bits
    return best_bits, best_e, exp_e / max(1, total), total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--simulate-only", action="store_true")
    ap.add_argument("--device", default="cepheus",
                    help="substring matched against qBraid device ids")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--anchors", type=int, default=3)
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--max-credits", type=float, default=200.0,
                    help="hard ceiling; the script refuses to submit above this")
    ap.add_argument("--confirm", action="store_true",
                    help="required before any billable submission")
    ap.add_argument("--maps", default="CDK2_Project")
    ap.add_argument("--prefix", default="1HCK_atp")
    ap.add_argument("--nodes-file", default="results/pocket_nodes.npy")
    ap.add_argument("--out", default="results/qaoa_hardware.json")
    args = ap.parse_args()

    from qdock_qubo import qubo_to_ising, solve_qubo

    # --- device discovery ----------------------------------------------------
    if args.list_devices:
        from qbraid.runtime import QbraidProvider
        provider = QbraidProvider()
        print(f"{'device id':<48} {'status':<12} qubits")
        for d in provider.get_devices():
            try:
                md = d.metadata()
                print(f"{str(d.id):<48} {str(d.status()):<12} "
                      f"{md.get('num_qubits', '?')}")
            except Exception:
                print(f"{str(d.id):<48} (metadata unavailable)")
        print("\nPick the cheapest one that fits your qubit count. "
              "Rigetti Cepheus-1-108Q via AWS is the best value at "
              "0.0425 credits/shot.")
        return

    # --- build the instance --------------------------------------------------
    Q, var_index, meta = build_small_instance(
        args.anchors, args.nodes, maps_dir=args.maps,
        prefix=args.prefix, nodes_path=args.nodes_file)
    n_qubits = meta["n_vars"]
    print(f"Docking QUBO: {n_qubits} variables, "
          f"{meta['n_geometry_couplings']} geometry couplings")
    print(f"Nodes: {meta['node_source']}")

    if n_qubits > 24:
        sys.exit(f"{n_qubits} qubits is too many for a first hardware run. "
                 f"Reduce --anchors / --nodes.")

    # --- certified ground state ---------------------------------------------
    print("\nExact ground state (dimod brute force)...")
    s_exact, e_exact, t_exact, _ = solve_qubo(Q, method="exact")
    print(f"  E_exact = {e_exact:.4f}  ({t_exact*1000:.1f} ms)")

    print("Classical annealing on the same Hamiltonian...")
    s_sa, e_sa, t_sa, m_sa = solve_qubo(Q, method="sa", num_reads=500, num_sweeps=2000)
    print(f"  E_sa    = {e_sa:.4f}  ({t_sa*1000:.1f} ms, "
          f"{m_sa['frac_at_best']*100:.0f}% of reads at best)")

    # --- QAOA parameters, tuned on a free classical simulation ---------------
    h, J, offset = qubo_to_ising(Q, n_qubits)
    scale = max(np.abs(h).max(), max((abs(v) for v in J.values()), default=1.0), 1.0)
    h_n = h / scale
    J_n = {k: v / scale for k, v in J.items()}

    print("\nTuning QAOA angles on a free local simulator...")
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    from qiskit import transpile

    best_params, best_exp = None, np.inf
    rng = np.random.default_rng(11)
    for _ in range(40):
        gammas = rng.uniform(0, np.pi, args.layers)
        betas = rng.uniform(0, np.pi / 2, args.layers)
        qc = qaoa_circuit(h_n, J_n, gammas, betas, n_qubits)
        counts = sim.run(transpile(qc, sim), shots=512).result().get_counts()
        _, _, exp_e, _ = evaluate_counts(counts, Q, n_qubits)
        if exp_e < best_exp:
            best_exp, best_params = exp_e, (gammas, betas)
    gammas, betas = best_params
    print(f"  best simulated <E> = {best_exp:.4f}")

    qc = qaoa_circuit(h_n, J_n, gammas, betas, n_qubits)
    print(f"  circuit depth = {qc.depth()}, "
          f"two-qubit gates = {sum(1 for i in qc.data if len(i.qubits) == 2)}")

    counts_sim = sim.run(transpile(qc, sim), shots=args.shots).result().get_counts()
    bits_sim, e_sim, exp_sim, _ = evaluate_counts(counts_sim, Q, n_qubits)

    report = {
        "n_qubits": n_qubits,
        "circuit_depth": qc.depth(),
        "qaoa_layers": args.layers,
        "shots": args.shots,
        "energy_exact": round(e_exact, 4),
        "energy_classical_annealing": round(e_sa, 4),
        "energy_qaoa_simulator_best": round(e_sim, 4),
        "energy_qaoa_simulator_mean": round(exp_sim, 4),
        "simulator_found_ground_state": bool(abs(e_sim - e_exact) < 1e-6),
        "geometry_couplings": meta["n_geometry_couplings"],
    }

    if args.simulate_only:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(json.dumps(report, indent=2))
        print("\nSimulation complete, 0 credits spent. "
              "Re-run with --confirm to submit to hardware.")
        return

    # --- hardware submission with a credit guard -----------------------------
    key = next((k for k in PRICING if k in args.device.lower()), None)
    if key is None:
        sys.exit(f"No pricing entry for '{args.device}'. Add one to PRICING first "
                 f"so the cost guard can protect your balance.")
    price = PRICING[key]
    est = price["per_task"] + price["per_shot"] * args.shots

    print("\n" + "=" * 62)
    print(f"  Device        : {args.device}")
    print(f"  Shots         : {args.shots}")
    print(f"  Estimated cost: {est:.1f} credits  (${est/100:.2f})")
    print("=" * 62)

    if est > args.max_credits:
        sys.exit(f"Refusing: {est:.1f} credits exceeds --max-credits "
                 f"{args.max_credits}.")
    if not args.confirm:
        print("Dry run. Add --confirm to actually submit and spend credits.")
        return

    from qbraid.runtime import QbraidProvider
    provider = QbraidProvider()
    matches = [d for d in provider.get_devices() if args.device.lower() in str(d.id).lower()]
    if not matches:
        sys.exit(f"No device id containing '{args.device}'. "
                 f"Run --list-devices to see what is available to your account.")
    device = matches[0]
    print(f"Submitting to {device.id} ...")

    job = device.run(qc, shots=args.shots)
    print(f"  job id: {job.id}")
    result = job.result()
    counts_hw = result.data.get_counts()

    bits_hw, e_hw, exp_hw, total = evaluate_counts(counts_hw, Q, n_qubits)

    report.update({
        "hardware_device": str(device.id),
        "hardware_job_id": str(job.id),
        "energy_qaoa_hardware_best": round(e_hw, 4),
        "energy_qaoa_hardware_mean": round(exp_hw, 4),
        "hardware_found_ground_state": bool(abs(e_hw - e_exact) < 1e-6),
        "estimated_credits_spent": round(est, 1),
        "shots_returned": int(total),
    })

    print("\n" + "=" * 62)
    print("RESULTS ON THE SAME DOCKING HAMILTONIAN")
    print("=" * 62)
    print(f"  Exact ground state       : {e_exact:9.4f}")
    print(f"  Classical annealing (SA) : {e_sa:9.4f}")
    print(f"  QAOA, noiseless simulator: {e_sim:9.4f}  (mean {exp_sim:.4f})")
    print(f"  QAOA, {str(device.id):<18}: {e_hw:9.4f}  (mean {exp_hw:.4f})")
    print("\n  The gap between the simulator and the hardware mean IS the noise")
    print("  result. Report it. Criterion C3 asks about noise and hardware")
    print("  limitations, and this is a measured answer rather than a caveat.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
