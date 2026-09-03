"""
vina_score.py
=============
The AutoDock Vina scoring function (Trott & Olson, J. Comput. Chem. 2010),
implemented directly and precomputed onto the same grid geometry the AutoDock4
maps use.

Why reimplement it? The `vina` Python bindings do not build on this platform,
and criterion C4 needs a classical baseline that is genuinely independent of
the scoring function being tested. Vina's functional form is completely
different from AutoDock4's: it has no electrostatic term, no explicit
desolvation term and no 12-6 Lennard-Jones well. It is steric-plus-hydrogen-
bond only, fitted to a different training set. Agreement between the two is
therefore real evidence rather than a restatement of the same physics.

Functional form, in terms of the SURFACE distance d = r_ij - (R_i + R_j):

    gauss1       exp(-(d/0.5)^2)                      w = -0.035579
    gauss2       exp(-((d-3)/2)^2)                    w = -0.005156
    repulsion    d^2 if d < 0 else 0                  w = +0.840245
    hydrophobic  1 (d<0.5), 0 (d>1.5), linear between w = -0.035069
                 -- only between two hydrophobic atoms
    hbond        1 (d<-0.7), 0 (d>0), linear between  w = -0.587439
                 -- only between a donor and an acceptor

The reported affinity divides the intermolecular sum by (1 + 0.05846 * N_rot).
That normalisation is applied in run_03, where the rotatable-bond count is
known from RDKit.
"""

from __future__ import annotations

import numpy as np

W_GAUSS1 = -0.035579
W_GAUSS2 = -0.005156
W_REPULSION = 0.840245
W_HYDROPHOBIC = -0.035069
W_HBOND = -0.587439
N_ROT_COEFF = 0.05846

VINA_CUTOFF = 8.0

# X-S van der Waals radii used by Vina.
VINA_RADII = {
    "C": 1.9, "N": 1.8, "O": 1.7, "S": 2.0, "P": 2.1,
    "F": 1.5, "Cl": 1.8, "Br": 2.0, "I": 2.2,
}

DR = 0.005
_D = None  # filled lazily; surface-distance axis


def _radial_axis(r_sum_max: float):
    """Radial axis in true interatomic distance, 0 .. VINA_CUTOFF."""
    return np.arange(DR, VINA_CUTOFF + DR, DR)


def classify(elements, ad_types, coords):
    """Assign Vina atom classes.

    Returns (radii, hydrophobic, donor, acceptor, heavy_mask).

    A heavy atom is a donor when a polar hydrogen (AutoDock type HD) sits within
    1.3 A of it. Oxygen is always an acceptor in Vina; nitrogen accepts only
    when it was typed NA; sulfur accepts only when typed SA. Carbon is
    hydrophobic unless it is bonded to nitrogen or oxygen, and the halogens are
    always hydrophobic.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(elements)
    heavy = np.array([t != "HD" for t in ad_types])

    radii = np.zeros(n)
    hydrophobic = np.zeros(n, dtype=bool)
    donor = np.zeros(n, dtype=bool)
    acceptor = np.zeros(n, dtype=bool)

    h_idx = np.where(~heavy)[0]
    h_coords = coords[h_idx] if len(h_idx) else np.zeros((0, 3))

    for i in range(n):
        if not heavy[i]:
            continue
        el = elements[i]
        el = "Cl" if el in ("Cl", "CL") else ("Br" if el in ("Br", "BR") else el)
        radii[i] = VINA_RADII.get(el, 1.9)

        if el in ("F", "Cl", "Br", "I"):
            hydrophobic[i] = True
        elif el == "C":
            # bonded to N or O?
            polar_nbr = False
            for j in range(n):
                if j == i or not heavy[j]:
                    continue
                if elements[j] in ("N", "O") and \
                        np.linalg.norm(coords[i] - coords[j]) < 1.8:
                    polar_nbr = True
                    break
            hydrophobic[i] = not polar_nbr

        if el == "O":
            acceptor[i] = True
        elif el == "N" and ad_types[i] == "NA":
            acceptor[i] = True
        elif el == "S" and ad_types[i] == "SA":
            acceptor[i] = True

        if el in ("N", "O", "S") and len(h_coords):
            if np.min(np.linalg.norm(h_coords - coords[i], axis=1)) < 1.3:
                donor[i] = True

    return radii, hydrophobic, donor, acceptor, heavy


def pair_table(r_i, r_j, both_hydrophobic, donor_acceptor):
    """Radial Vina energy table for one pair class, in true distance."""
    r = np.arange(DR, VINA_CUTOFF + DR, DR)
    d = r - (r_i + r_j)

    e = W_GAUSS1 * np.exp(-((d / 0.5) ** 2))
    e = e + W_GAUSS2 * np.exp(-(((d - 3.0) / 2.0) ** 2))
    e = e + W_REPULSION * np.where(d < 0, d ** 2, 0.0)

    if both_hydrophobic:
        hyd = np.clip((1.5 - d) / 1.0, 0.0, 1.0)
        e = e + W_HYDROPHOBIC * hyd
    if donor_acceptor:
        hb = np.clip(-d / 0.7, 0.0, 1.0)
        e = e + W_HBOND * hb
    return e


# ---------------------------------------------------------------------------
# Grid representation, so the existing search code can be reused unchanged
# ---------------------------------------------------------------------------

class VinaGrids:
    """Drop-in replacement for qdock_core.GridSet that scores with Vina.

    Exposes .center and .score(coords, ad_types, charges) so that
    generate_poses() and refine_pose() work without modification. Charges are
    accepted and ignored: Vina has no electrostatic term.
    """

    # Probe classes the ligand side can take.
    PROBES = [
        ("C_H", "C", True, False, False),
        ("C_P", "C", False, False, False),
        ("N_P", "N", False, False, False),
        ("N_D", "N", False, True, False),
        ("N_A", "N", False, False, True),
        ("N_DA", "N", False, True, True),
        ("O_A", "O", False, False, True),
        ("O_DA", "O", False, True, True),
        ("S_P", "S", False, False, False),
        ("P_P", "P", False, False, False),
        ("F_H", "F", True, False, False),
        ("Cl_H", "Cl", True, False, False),
        ("Br_H", "Br", True, False, False),
        ("I_H", "I", True, False, False),
    ]

    def __init__(self, receptor_pdbqt, center, npts, spacing=0.375,
                 out_of_box_penalty=50.0, verbose=True):
        self.center = np.asarray(center, dtype=float)
        self.npts = np.asarray(npts, dtype=int)
        self.spacing = float(spacing)
        self.origin = self.center - (self.npts * self.spacing) / 2.0
        self.shape = tuple(self.npts + 1)
        self.out_of_box_penalty = out_of_box_penalty
        self._build(receptor_pdbqt, verbose)

    # -- receptor ----------------------------------------------------------
    def _read_receptor(self, path):
        coords, types, elements = [], [], []
        with open(path) as fh:
            for line in fh:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                coords.append([float(line[30:38]), float(line[38:46]),
                               float(line[46:54])])
                t = line[77:79].strip()
                types.append(t)
                elements.append("C" if t in ("C", "A") else t[0])
        return np.array(coords), types, elements

    def _build(self, receptor_pdbqt, verbose):
        coords, types, elements = self._read_receptor(receptor_pdbqt)
        radii, phob, don, acc, heavy = classify(elements, types, coords)

        self.maps = {name: np.zeros(self.shape) for name, *_ in self.PROBES}
        axes = [self.origin[k] + np.arange(self.shape[k]) * self.spacing
                for k in range(3)]

        lo_box = self.origin - VINA_CUTOFF
        hi_box = self.origin + self.npts * self.spacing + VINA_CUTOFF
        near = np.all((coords >= lo_box) & (coords <= hi_box), axis=1) & heavy
        idx = np.where(near)[0]
        if verbose:
            print(f"  Vina maps: {len(idx)} receptor heavy atoms in range")

        # Cache tables keyed on the receptor atom's discrete class.
        table_cache: dict[tuple, np.ndarray] = {}

        for j in idx:
            pos = coords[j]
            rj, pj, dj, aj = radii[j], phob[j], don[j], acc[j]

            lo = np.maximum(np.ceil((pos - VINA_CUTOFF - self.origin) / self.spacing), 0).astype(int)
            hi = np.minimum(np.floor((pos + VINA_CUTOFF - self.origin) / self.spacing), self.npts).astype(int)
            if np.any(lo > hi):
                continue
            sl = tuple(slice(lo[k], hi[k] + 1) for k in range(3))

            dx2 = (axes[0][sl[0]] - pos[0]) ** 2
            dy2 = (axes[1][sl[1]] - pos[1]) ** 2
            dz2 = (axes[2][sl[2]] - pos[2]) ** 2
            r = np.sqrt(dx2[:, None, None] + dy2[None, :, None] + dz2[None, None, :])
            inside = r < VINA_CUTOFF
            if not inside.any():
                continue
            bins = np.clip((r / DR).astype(np.intp) - 1, 0,
                           int(VINA_CUTOFF / DR) - 1)

            for name, el, p_phob, p_don, p_acc in self.PROBES:
                ri = VINA_RADII.get(el, 1.9)
                key = (round(ri, 3), round(float(rj), 3),
                       bool(p_phob and pj),
                       bool((p_don and aj) or (p_acc and dj)))
                tbl = table_cache.get(key)
                if tbl is None:
                    tbl = pair_table(ri, rj, key[2], key[3])
                    table_cache[key] = tbl
                self.maps[name][sl] += np.where(inside, tbl[bins], 0.0)

    # -- scoring -----------------------------------------------------------
    def probe_for(self, element, ad_type, is_donor, is_acceptor):
        el = element
        if el == "C":
            return "C_H", "C_P"
        return None, None

    def _interp(self, grid, pts):
        f = (pts - self.origin) / self.spacing
        i0 = np.floor(f).astype(int)
        frac = f - i0
        shape = np.array(grid.shape)
        oob = ((i0 < 0).any(axis=1) | (i0[:, 0] >= shape[0] - 1)
               | (i0[:, 1] >= shape[1] - 1) | (i0[:, 2] >= shape[2] - 1))
        i0c = np.clip(i0, 0, shape - 2)
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
        ix, iy, iz = i0c[:, 0], i0c[:, 1], i0c[:, 2]
        g = grid
        c00 = g[ix, iy, iz] * (1 - fx) + g[ix + 1, iy, iz] * fx
        c01 = g[ix, iy, iz + 1] * (1 - fx) + g[ix + 1, iy, iz + 1] * fx
        c10 = g[ix, iy + 1, iz] * (1 - fx) + g[ix + 1, iy + 1, iz] * fx
        c11 = g[ix, iy + 1, iz + 1] * (1 - fx) + g[ix + 1, iy + 1, iz + 1] * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        return c0 * (1 - fz) + c1 * fz, oob

    def set_ligand_classes(self, probes):
        """Cache the per-atom probe names for the ligand being docked."""
        self._probes = probes

    def score(self, coords, ad_types=None, charges=None):
        """Vina intermolecular energy. Signature matches GridSet.score()."""
        coords = np.atleast_2d(np.asarray(coords, dtype=float))
        probes = self._probes
        per_atom = np.zeros(len(coords))
        oob_any = np.zeros(len(coords), dtype=bool)

        by_probe: dict[str, list[int]] = {}
        for i, p in enumerate(probes):
            if p is None:
                continue
            by_probe.setdefault(p, []).append(i)

        for p, idxs in by_probe.items():
            vals, oob = self._interp(self.maps[p], coords[idxs])
            per_atom[idxs] = vals
            oob_any[idxs] = oob

        per_atom = np.where(oob_any, self.out_of_box_penalty, per_atom)
        return float(per_atom.sum()), int(oob_any.sum()), per_atom


def ligand_probes(elements, ad_types, coords):
    """Map ligand atoms onto VinaGrids probe names (None for polar hydrogens)."""
    radii, phob, don, acc, heavy = classify(elements, ad_types, coords)
    probes = []
    for i, el in enumerate(elements):
        if not heavy[i]:
            probes.append(None)
            continue
        e = "Cl" if el in ("Cl", "CL") else ("Br" if el in ("Br", "BR") else el)
        if e == "C":
            probes.append("C_H" if phob[i] else "C_P")
        elif e == "N":
            if don[i] and acc[i]:
                probes.append("N_DA")
            elif don[i]:
                probes.append("N_D")
            elif acc[i]:
                probes.append("N_A")
            else:
                probes.append("N_P")
        elif e == "O":
            probes.append("O_DA" if don[i] else "O_A")
        elif e == "S":
            probes.append("S_P")
        elif e == "P":
            probes.append("P_P")
        elif e in ("F", "Cl", "Br", "I"):
            probes.append(f"{e}_H")
        else:
            probes.append("C_P")
    return probes
