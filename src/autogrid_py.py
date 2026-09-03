"""
autogrid_py.py
==============
A NumPy reimplementation of AutoGrid 4 that writes AutoDock-format .map files.

Why not just run autogrid4? Because the binary is not available in this
environment, and the maps that were shipped with the project could not be
traced back to any coordinate frame derived from RCSB 1HCK. Rebuilding the
maps from the published AutoDock 4.2 force field means the entire pipeline is
reproducible from the PDB accession code, which is exactly what a reviewer
needs in order to check the work.

Energy model (Huey, Morris, Olson & Goodsell, J. Comput. Chem. 2007)
--------------------------------------------------------------------
For a probe atom of ligand type L placed at grid point g, summed over every
receptor atom j inside an 8 A non-bonded cutoff:

  dispersion / repulsion   W_vdw   * eps_ij [ (Rij/r)^12 - 2 (Rij/r)^6 ]
  directional H-bond       W_hbond * eps_hb [ 5(Rij/r)^12 - 6(Rij/r)^10 ]
  desolvation              W_sol   * (S_L V_j + S_j V_L) exp(-r^2 / 2 sigma^2)
  electrostatics           W_elec  * 332.06 q_j / (epsilon(r) r)

with S_i = ASP_i + q_sol |q_i|, sigma = 3.6 A, and epsilon(r) the
Mehler-Solmajer distance-dependent dielectric.

Following AutoGrid, the three charge-independent contributions are folded into
one map per ligand atom type, the |q_L|-dependent part of desolvation goes into
the .d map, and electrostatics goes into the .e map. The docking code then
evaluates

    E = sum_L [ map_L(r) + q_L * e(r) + |q_L| * d(r) ]

which is what qdock_core.GridSet.score does.

Documented approximations
-------------------------
* The hydrogen-bond term is computed without AutoGrid's angular weighting.
  AutoGrid modulates the 12-10 well depth by the cosine of the donor-H-acceptor
  and acceptor-lone-pair angles; here every H-bond pair is treated as if
  ideally oriented, so H-bonding is somewhat over-rewarded. Hydrogen positions
  come from pdb2pqr's hydrogen-bond optimiser, so donor geometry is at least
  physically reasonable.
* Receptor atoms are rigid, and the desolvation and electrostatic sums use the
  same 8 A cutoff as the van der Waals term.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from scipy.ndimage import minimum_filter1d

# --- AutoDock 4.2 free-energy weights ---------------------------------------
W_VDW = 0.1662
W_HBOND = 0.1209
W_ELEC = 0.1406
W_DSOLV = 0.1322

Q_SOLV = 0.01097          # charge-dependent solvation coefficient
SIGMA = 3.6               # A, desolvation gaussian width
NB_CUTOFF = 8.0           # A, non-bonded cutoff
E_CLAMP = 1.0e5           # ceiling on any tabulated energy
COULOMB_K = 332.06363     # kcal*A/(mol*e^2)

# Mehler-Solmajer distance-dependent dielectric
_MS_A = -8.5525
_MS_EPS0 = 78.4
_MS_LAMBDA = 0.003627
_MS_K = 7.7839
_MS_B = _MS_EPS0 - _MS_A

# --- AutoDock 4.2 atom parameters -------------------------------------------
# type: (Rii, epsii, volume, solpar, Rij_hb, eps_hb, hbond_code)
#   hbond_code 0 = none, 1/2 = donor hydrogen, 3/4/5 = acceptor
AD4_PARAMS: dict[str, tuple] = {
    "H":  (2.00, 0.020,  0.0000,  0.00051, 0.0, 0.0, 0),
    "HD": (2.00, 0.020,  0.0000,  0.00051, 0.0, 0.0, 2),
    "C":  (4.00, 0.150, 33.5103, -0.00143, 0.0, 0.0, 0),
    "A":  (4.00, 0.150, 33.5103, -0.00052, 0.0, 0.0, 0),
    "N":  (3.50, 0.160, 22.4493, -0.00162, 0.0, 0.0, 0),
    "NA": (3.50, 0.160, 22.4493, -0.00162, 1.9, 5.0, 4),
    "OA": (3.20, 0.200, 17.1573, -0.00251, 1.9, 5.0, 5),
    "SA": (4.00, 0.200, 33.5103, -0.00214, 2.5, 1.0, 5),
    "S":  (4.00, 0.200, 33.5103, -0.00214, 0.0, 0.0, 0),
    "P":  (4.20, 0.200, 38.7924, -0.00110, 0.0, 0.0, 0),
    "F":  (3.09, 0.080, 15.4480, -0.00110, 0.0, 0.0, 0),
    "Cl": (4.09, 0.276, 35.8235, -0.00110, 0.0, 0.0, 0),
    "Br": (4.33, 0.389, 42.5661, -0.00110, 0.0, 0.0, 0),
    "I":  (4.72, 0.550, 55.0585, -0.00110, 0.0, 0.0, 0),
}

DEFAULT_LIGAND_TYPES = ["A", "C", "HD", "N", "NA", "OA", "SA",
                        "S", "P", "F", "Cl", "Br", "I"]

DR = 0.005                                    # A, radial table resolution
_R = np.arange(DR, NB_CUTOFF + DR, DR)        # never evaluate at r = 0
N_BINS = len(_R)


def _is_hbond_pair(code_a: int, code_b: int) -> bool:
    return (code_a in (1, 2) and code_b >= 3) or (code_b in (1, 2) and code_a >= 3)


def pair_table(type_l: str, type_r: str, smooth: float = 0.5) -> np.ndarray:
    """Radial energy table for one ligand-type / receptor-type pair.

    Includes the van der Waals or hydrogen-bond term (smoothed the way AutoGrid
    smooths it) plus the two charge-independent halves of the desolvation term.
    """
    rii_l, eps_l, vol_l, sol_l, rhb_l, ehb_l, hb_l = AD4_PARAMS[type_l]
    rii_r, eps_r, vol_r, sol_r, rhb_r, ehb_r, hb_r = AD4_PARAMS[type_r]

    if _is_hbond_pair(hb_l, hb_r):
        # Well parameters come from whichever partner is the heavy H-bonder.
        if hb_l >= 3:
            rij, eps = rhb_l, ehb_l
        else:
            rij, eps = rhb_r, ehb_r
        x_a, x_b, weight = 12, 10, W_HBOND
    else:
        rij = (rii_l + rii_r) / 2.0
        eps = np.sqrt(eps_l * eps_r)
        x_a, x_b, weight = 12, 6, W_VDW

    if rij <= 0 or eps <= 0:                  # degenerate pair, no vdW well
        vdw = np.zeros(N_BINS)
    else:
        c_a = (eps * x_b / (x_a - x_b)) * rij ** x_a
        c_b = (eps * x_a / (x_a - x_b)) * rij ** x_b
        with np.errstate(over="ignore", divide="ignore"):
            vdw = weight * (c_a / _R ** x_a - c_b / _R ** x_b)
        vdw = np.clip(np.nan_to_num(vdw, nan=E_CLAMP, posinf=E_CLAMP),
                      -E_CLAMP, E_CLAMP)
        if smooth > 0:
            # AutoGrid flattens the well bottom by taking the minimum energy
            # within +/- smooth/2 of each distance.
            window = max(1, int(round(smooth / DR)))
            vdw = minimum_filter1d(vdw, size=window, mode="nearest")

    gauss = np.exp(-(_R ** 2) / (2.0 * SIGMA ** 2))
    desolv = W_DSOLV * (sol_l * vol_r + sol_r * vol_l) * gauss
    return vdw + desolv


def coulomb_table() -> np.ndarray:
    eps_r = _MS_A + _MS_B / (1.0 + _MS_K * np.exp(-_MS_LAMBDA * _MS_B * _R))
    val = COULOMB_K / (eps_r * _R)
    return np.clip(val, -E_CLAMP, E_CLAMP)


def gaussian_table() -> np.ndarray:
    return np.exp(-(_R ** 2) / (2.0 * SIGMA ** 2))


def read_receptor(path: str):
    coords, types, charges = [], [], []
    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            charges.append(float(line[70:76]))
            types.append(line[77:79].strip())
    return np.array(coords), types, np.array(charges)


def build_maps(rec_coords, rec_types, rec_charges, center, npts, spacing,
               ligand_types, verbose=True):
    """Accumulate every map in a single pass over the receptor atoms."""
    n = np.asarray(npts, dtype=int)
    shape = tuple(n + 1)
    origin = np.asarray(center, dtype=float) - (n * spacing) / 2.0

    maps = {t: np.zeros(shape) for t in ligand_types}
    e_map = np.zeros(shape)          # sum_j q_j * coulomb(r)
    dq_map = np.zeros(shape)         # sum_j |q_j| * gauss(r)
    dv_map = np.zeros(shape)         # sum_j vol_j * gauss(r)

    rec_kinds = sorted(set(rec_types))
    unknown = [t for t in rec_kinds if t not in AD4_PARAMS]
    if unknown:
        raise ValueError(f"Receptor contains untyped atoms: {unknown}")

    tables = {(l, r): pair_table(l, r) for l in ligand_types for r in rec_kinds}
    coul = coulomb_table()
    gauss = gaussian_table()

    axes = [origin[k] + np.arange(shape[k]) * spacing for k in range(3)]

    # Skip receptor atoms that cannot reach the box.
    box_lo = origin - NB_CUTOFF
    box_hi = origin + n * spacing + NB_CUTOFF
    relevant = np.all((rec_coords >= box_lo) & (rec_coords <= box_hi), axis=1)
    idx_relevant = np.where(relevant)[0]
    if verbose:
        print(f"  receptor atoms within cutoff of the box: "
              f"{len(idx_relevant)} of {len(rec_coords)}")

    t0 = time.time()
    for count, j in enumerate(idx_relevant):
        pos = rec_coords[j]
        rtype = rec_types[j]
        qj = rec_charges[j]
        volj = AD4_PARAMS[rtype][2]

        # Index window of grid points within the cutoff of this atom.
        lo = np.maximum(np.ceil((pos - NB_CUTOFF - origin) / spacing), 0).astype(int)
        hi = np.minimum(np.floor((pos + NB_CUTOFF - origin) / spacing),
                        n).astype(int)
        if np.any(lo > hi):
            continue
        sl = tuple(slice(lo[k], hi[k] + 1) for k in range(3))

        dx2 = (axes[0][sl[0]] - pos[0]) ** 2
        dy2 = (axes[1][sl[1]] - pos[1]) ** 2
        dz2 = (axes[2][sl[2]] - pos[2]) ** 2
        r2 = dx2[:, None, None] + dy2[None, :, None] + dz2[None, None, :]
        r = np.sqrt(r2)

        inside = r < NB_CUTOFF
        if not inside.any():
            continue
        bins = np.clip((r / DR).astype(np.intp) - 1, 0, N_BINS - 1)

        for lt in ligand_types:
            contrib = tables[(lt, rtype)][bins]
            maps[lt][sl] += np.where(inside, contrib, 0.0)

        g = np.where(inside, gauss[bins], 0.0)
        e_map[sl] += qj * np.where(inside, coul[bins], 0.0)
        dq_map[sl] += abs(qj) * g
        dv_map[sl] += volj * g

        if verbose and (count + 1) % 400 == 0:
            print(f"    {count + 1}/{len(idx_relevant)} atoms "
                  f"({time.time() - t0:.0f}s)")

    # Fold the |q_receptor|-dependent half of desolvation into the type maps,
    # and build the two charge-coupled maps.
    for lt in ligand_types:
        vol_l = AD4_PARAMS[lt][2]
        maps[lt] += W_DSOLV * Q_SOLV * vol_l * dq_map
        np.clip(maps[lt], -E_CLAMP, E_CLAMP, out=maps[lt])

    maps["e"] = np.clip(W_ELEC * e_map, -E_CLAMP, E_CLAMP)
    maps["d"] = np.clip(W_DSOLV * Q_SOLV * dv_map, -E_CLAMP, E_CLAMP)
    return maps, origin


def write_map(path, data, center, npts, spacing, prefix, receptor_name):
    """Write one AutoDock .map file (x varies fastest, then y, then z)."""
    n = np.asarray(npts, dtype=int)
    with open(path, "w") as fh:
        fh.write(f"GRID_PARAMETER_FILE {prefix}.gpf\n")
        fh.write(f"GRID_DATA_FILE {prefix}.maps.fld\n")
        fh.write(f"MACROMOLECULE {receptor_name}\n")
        fh.write(f"SPACING {spacing:.3f}\n")
        fh.write(f"NELEMENTS {n[0]:d} {n[1]:d} {n[2]:d}\n")
        fh.write(f"CENTER {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n")
        flat = np.ascontiguousarray(data.transpose(2, 1, 0)).ravel()
        fh.write("\n".join(f"{v:.4f}" for v in flat))
        fh.write("\n")


def main():
    ap = argparse.ArgumentParser(description="AutoGrid 4 work-alike")
    ap.add_argument("--receptor", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="1HCK_atp")
    ap.add_argument("--center", nargs=3, type=float, required=True)
    ap.add_argument("--npts", nargs=3, type=int, default=[66, 66, 66])
    ap.add_argument("--spacing", type=float, default=0.375)
    ap.add_argument("--types", nargs="*", default=DEFAULT_LIGAND_TYPES)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    coords, types, charges = read_receptor(args.receptor)
    print(f"Receptor: {len(coords)} atoms from {args.receptor}")
    print(f"  net charge {charges.sum():+.3f}")

    n = np.array(args.npts)
    print(f"Box: centre {args.center}, {n.tolist()} intervals at "
          f"{args.spacing} A = {(n * args.spacing).tolist()} A")
    print(f"  grid points per map: {int(np.prod(n + 1)):,}")
    print(f"Generating {len(args.types)} atom-type maps plus .e and .d ...")

    t0 = time.time()
    maps, origin = build_maps(coords, types, charges, args.center, n,
                              args.spacing, args.types)
    print(f"  accumulation finished in {time.time() - t0:.0f}s")

    rec_name = os.path.basename(args.receptor)
    for key, data in maps.items():
        path = os.path.join(args.out_dir, f"{args.prefix}.{key}.map")
        write_map(path, data, args.center, n, args.spacing, args.prefix, rec_name)
        print(f"  {os.path.basename(path):<24s} "
              f"min {data.min():10.3f}  max {data.max():10.1f}")

    fld = os.path.join(args.out_dir, f"{args.prefix}.maps.fld")
    with open(fld, "w") as fh:
        fh.write(f"# AVS field file written by autogrid_py\n")
        fh.write(f"# receptor {rec_name}\n")
        fh.write(f"# centre {args.center}\n")
        fh.write(f"# npts {n.tolist()} spacing {args.spacing}\n")
    print(f"\nWrote {len(maps)} maps to {args.out_dir}")


if __name__ == "__main__":
    main()
