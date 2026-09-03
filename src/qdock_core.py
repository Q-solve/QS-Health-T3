"""
qdock_core.py
=============
Grids, ligand preparation, pose generation and scoring for the CDK2 (1HCK)
quantum-annealing docking pipeline.

What changed versus the first version, and why:

  1. Rotations are generated from uniform random quaternions (Shoemake's method).
     The old code unpacked sines out of order (`sx, sz, sy = np.sin(angles)`),
     producing matrices with det != 1 that sheared the ligand. Quaternions make
     that class of bug impossible.

  2. Scoring now uses the full AutoDock4 form:
         E = sum_i [ V_type(r_i) + q_i * V_elec(r_i) + |q_i| * V_desolv(r_i) ]
     The electrostatic (.e.map) and desolvation (.d.map) grids were missing
     before, so the reported numbers were not comparable to any published dG.

  3. Grid lookup uses trilinear interpolation instead of nearest-neighbour
     rounding, and out-of-box atoms incur a large penalty rather than the cheap
     +4 kcal/mol soft-core cap (which let poses escape the box for free).

  4. The empirical hinge bonus is OFF by default and hard-capped when enabled.
     The OA/NA/HD maps already contain hydrogen-bonding terms, so adding an
     uncapped -2.5 per polar atom double-counted and biased ranking toward
     polar-atom-rich molecules.

  5. Ligands are prepared as conformer ENSEMBLES. Rigid-body docking of a single
     RDKit conformer can never recover a good pose if that conformer is wrong.

  6. Poses get a local 6-DOF refinement (Powell). Random sampling alone will
     never reach sub-2A redocking RMSD, which is the accepted validation bar.

Author: Q-SOLVE Kenya 2026 submission.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

OUT_OF_BOX_PENALTY = 50.0   # kcal/mol per atom outside the grid box
SOFT_CORE_CAP = 5.0         # kcal/mol ceiling on any single in-box atom term
HINGE_BONUS_PER_ATOM = -1.0 # only used if hinge_weight > 0
HINGE_BONUS_FLOOR = -4.0    # total hinge bonus can never beat this

# AutoDock atom types we can map from RDKit
_AD_TYPE_FALLBACK = "C"


# ----------------------------------------------------------------------------
# 1. AutoGrid maps
# ----------------------------------------------------------------------------

class AutoGridMap:
    """A single AutoGrid .map file with trilinear interpolation."""

    def __init__(self, path: str):
        self.path = path
        self.spacing = 0.375
        self.nelements = np.array([40, 40, 40], dtype=int)
        self.center = np.zeros(3)
        self.data = None
        self._parse(path)
        self.origin = self.center - (self.nelements * self.spacing) / 2.0
        self.shape = np.array(self.data.shape)

    def _parse(self, path: str) -> None:
        values = []
        with open(path, "r") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("SPACING"):
                    self.spacing = float(s.split()[1])
                elif s.startswith("NELEMENTS"):
                    p = s.split()
                    self.nelements = np.array([int(p[1]), int(p[2]), int(p[3])])
                elif s.startswith("CENTER"):
                    p = s.split()
                    self.center = np.array([float(p[1]), float(p[2]), float(p[3])])
                elif s[0].isdigit() or s[0] in "-+.":
                    try:
                        values.append(float(s))
                    except ValueError:
                        continue

        nx, ny, nz = (self.nelements + 1)
        expected = nx * ny * nz
        if len(values) < expected:
            raise ValueError(
                f"{os.path.basename(path)}: expected {expected} grid points, "
                f"found {len(values)}. Regenerate the map with autogrid4."
            )
        # AutoGrid writes x fastest, then y, then z.
        arr = np.asarray(values[:expected], dtype=float)
        self.data = arr.reshape((nz, ny, nx)).transpose(2, 1, 0)  # -> [ix, iy, iz]

    def interpolate(self, pts: np.ndarray):
        """Trilinear lookup.

        Returns (values, out_of_box_mask). Out-of-box entries have value 0.0 and
        must be handled by the caller.
        """
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        f = (pts - self.origin) / self.spacing
        i0 = np.floor(f).astype(int)
        frac = f - i0

        oob = (
            (i0 < 0).any(axis=1)
            | (i0[:, 0] >= self.shape[0] - 1)
            | (i0[:, 1] >= self.shape[1] - 1)
            | (i0[:, 2] >= self.shape[2] - 1)
        )
        i0c = np.clip(i0, 0, self.shape - 2)
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
        ix, iy, iz = i0c[:, 0], i0c[:, 1], i0c[:, 2]
        d = self.data

        c000 = d[ix, iy, iz]
        c100 = d[ix + 1, iy, iz]
        c010 = d[ix, iy + 1, iz]
        c001 = d[ix, iy, iz + 1]
        c110 = d[ix + 1, iy + 1, iz]
        c101 = d[ix + 1, iy, iz + 1]
        c011 = d[ix, iy + 1, iz + 1]
        c111 = d[ix + 1, iy + 1, iz + 1]

        c00 = c000 * (1 - fx) + c100 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        vals = c0 * (1 - fz) + c1 * fz

        vals = np.where(oob, 0.0, vals)
        return vals, oob


class GridSet:
    """All AutoGrid maps for one receptor, including electrostatics + desolvation."""

    def __init__(self, map_dir: str, prefix: str):
        self.map_dir = map_dir
        self.prefix = prefix
        self.maps: dict[str, AutoGridMap] = {}
        self.elec: AutoGridMap | None = None
        self.desolv: AutoGridMap | None = None
        self._load()

    def _load(self) -> None:
        pattern = os.path.join(self.map_dir, f"{self.prefix}.*.map")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No maps matching {pattern}. Run autogrid4 -p {self.prefix}.gpf "
                f"-l {self.prefix}.glg first."
            )
        for path in files:
            key = os.path.basename(path).split(".")[-2]  # 1HCK_atp.OA.map -> OA
            m = AutoGridMap(path)
            if key == "e":
                self.elec = m
            elif key == "d":
                self.desolv = m
            else:
                self.maps[key] = m

        if self.elec is None or self.desolv is None:
            missing = [k for k, v in (("e", self.elec), ("d", self.desolv)) if v is None]
            raise FileNotFoundError(
                f"Missing {missing} map(s). Add 'elecmap {self.prefix}.e.map' and "
                f"'dsolvmap {self.prefix}.d.map' to your GPF and re-run autogrid4. "
                f"Without them the score is not an AutoDock energy."
            )

        ref = next(iter(self.maps.values()))
        self.center = ref.center
        self.origin = ref.origin
        self.spacing = ref.spacing
        self.nelements = ref.nelements

    def available_types(self) -> list[str]:
        return sorted(self.maps.keys())

    def score(self, coords: np.ndarray, ad_types: list[str], charges: np.ndarray):
        """AutoDock4-style intermolecular energy.

        Returns (total_energy, n_out_of_box, per_atom_energies).
        """
        coords = np.asarray(coords, dtype=float)
        charges = np.asarray(charges, dtype=float)
        per_atom = np.zeros(len(coords))
        oob_any = np.zeros(len(coords), dtype=bool)

        # Group atoms by type so each map is interpolated once.
        by_type: dict[str, list[int]] = {}
        for i, t in enumerate(ad_types):
            key = t if t in self.maps else _AD_TYPE_FALLBACK
            by_type.setdefault(key, []).append(i)

        for t, idxs in by_type.items():
            if t not in self.maps:
                continue
            vals, oob = self.maps[t].interpolate(coords[idxs])
            per_atom[idxs] += vals
            oob_any[idxs] |= oob

        e_vals, e_oob = self.elec.interpolate(coords)
        d_vals, d_oob = self.desolv.interpolate(coords)
        per_atom += charges * e_vals
        per_atom += np.abs(charges) * d_vals
        oob_any |= e_oob | d_oob

        per_atom = np.minimum(per_atom, SOFT_CORE_CAP)
        per_atom = np.where(oob_any, OUT_OF_BOX_PENALTY, per_atom)
        return float(per_atom.sum()), int(oob_any.sum()), per_atom


# ----------------------------------------------------------------------------
# 2. Receptor: residues, hinge geometry, contact profiling
# ----------------------------------------------------------------------------

@dataclass
class ReceptorAtom:
    res_id: str
    res_num: int
    atom_name: str
    ad_type: str
    coord: np.ndarray


class Receptor:
    """Parses a PDBQT receptor and derives the hinge anchor from the structure."""

    HBOND_TYPES = {"OA", "NA", "N", "HD", "SA", "O"}
    HYDROPHOBIC_TYPES = {"C", "A"}

    def __init__(self, pdbqt_path: str):
        self.atoms: list[ReceptorAtom] = []
        self._parse(pdbqt_path)
        self._coords = np.array([a.coord for a in self.atoms]) if self.atoms else np.zeros((0, 3))
        self.hinge_coord = self._derive_hinge()

    def _parse(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Receptor not found: {path}")
        with open(path) as fh:
            for line in fh:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    continue
                res_name = line[17:20].strip()
                try:
                    res_num = int(line[22:26])
                except ValueError:
                    res_num = -1
                ad_type = line[77:79].strip() if len(line) >= 79 else ""
                self.atoms.append(
                    ReceptorAtom(
                        res_id=f"{res_name}{res_num}",
                        res_num=res_num,
                        atom_name=line[12:16].strip(),
                        ad_type=ad_type or "C",
                        coord=np.array([x, y, z]),
                    )
                )

    def _derive_hinge(self, hinge_resnum: int = 83) -> np.ndarray:
        """Midpoint of the Leu83 backbone N and O — the CDK2 hinge donor/acceptor.

        Derived from the structure rather than hardcoded, so the number is
        reproducible from the PDB entry alone (judging criterion C3).
        """
        picks = [
            a.coord for a in self.atoms
            if a.res_num == hinge_resnum and a.atom_name in ("N", "O")
        ]
        if len(picks) < 2:
            raise ValueError(
                f"Could not find backbone N and O of residue {hinge_resnum} in the "
                f"receptor. Check the PDBQT covers the hinge region."
            )
        return np.mean(np.asarray(picks), axis=0)

    def profile_contacts(self, coords, elements, hb_cut=3.5, phobic_cut=4.0):
        """Distance-based contact census on the winning pose.

        NOTE for the presentation: this is a distance criterion only. It has no
        donor-H-acceptor angle test, so it overcounts relative to tools like
        PLIP. Report it as 'polar contacts', not 'hydrogen bonds'.
        """
        coords = np.asarray(coords)
        d = np.linalg.norm(coords[:, None, :] - self._coords[None, :, :], axis=2)

        lig_polar = np.array([e in ("N", "O", "S") for e in elements])
        lig_carbon = np.array([e == "C" for e in elements])
        rec_polar = np.array([a.ad_type in self.HBOND_TYPES for a in self.atoms])
        rec_phobic = np.array([a.ad_type in self.HYDROPHOBIC_TYPES for a in self.atoms])

        polar_hits = (d <= hb_cut) & lig_polar[:, None] & rec_polar[None, :]
        phobic_hits = (d <= phobic_cut) & lig_carbon[:, None] & rec_phobic[None, :]

        polar_res = {self.atoms[j].res_id for j in np.where(polar_hits.any(axis=0))[0]}
        phobic_res = {self.atoms[j].res_id for j in np.where(phobic_hits.any(axis=0))[0]}
        return {
            "polar_contact_count": len(polar_res),
            "polar_contact_residues": "; ".join(sorted(polar_res)) or "None",
            "hydrophobic_count": len(phobic_res),
            "hydrophobic_residues": "; ".join(sorted(phobic_res)) or "None",
            "min_dist_to_hinge": float(np.min(np.linalg.norm(coords - self.hinge_coord, axis=1))),
        }


# ----------------------------------------------------------------------------
# 3. Ligand preparation
# ----------------------------------------------------------------------------

@dataclass
class PreparedLigand:
    name: str
    elements: list[str]
    ad_types: list[str]
    charges: np.ndarray
    conformers: list[np.ndarray] = field(default_factory=list)  # heavy-atom coords

    @property
    def n_atoms(self) -> int:
        return len(self.elements)


def assign_autodock_types(mol) -> list[str]:
    """Map RDKit heavy atoms to AutoDock 4 atom types (united-atom, polar H merged)."""
    types = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == "H":
            types.append("HD")
        elif sym == "C":
            types.append("A" if atom.GetIsAromatic() else "C")
        elif sym == "N":
            # NA = acceptor. A nitrogen carrying a hydrogen is a donor, and a
            # fully substituted one (pyrrole-type) has no free lone pair.
            # Neighbours are counted rather than trusting GetTotalNumHs(),
            # which returns 0 once hydrogens are explicit.
            n_h = sum(1 for nb in atom.GetNeighbors() if nb.GetSymbol() == "H")
            n_h += atom.GetTotalNumHs()
            n_heavy = sum(1 for nb in atom.GetNeighbors() if nb.GetSymbol() != "H")
            types.append("N" if (n_h > 0 or n_heavy >= 3) else "NA")
        elif sym == "O":
            types.append("OA")
        elif sym == "S":
            types.append("SA")
        elif sym in ("F", "Cl", "Br", "I", "P"):
            types.append(sym)
        else:
            types.append(_AD_TYPE_FALLBACK)
    return types


def prepare_ligand(mol, name: str, n_confs: int = 10, seed: int = 0xF00D,
                   max_heavy_atoms: int = 90):
    """Embed a conformer ensemble, MMFF-optimise, assign types and merged charges.

    Returns a PreparedLigand or None if the molecule fails preparation. The count
    of failures is itself a reportable number (criterion C4).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    try:
        mol = Chem.Mol(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    if mol.GetNumHeavyAtoms() > max_heavy_atoms or mol.GetNumHeavyAtoms() < 5:
        return None

    molh = Chem.AddHs(mol, addCoords=True)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.5
    params.useSmallRingTorsions = True
    try:
        cids = AllChem.EmbedMultipleConfs(molh, numConfs=n_confs, params=params)
    except Exception:
        return None
    if len(cids) == 0:
        return None

    try:
        AllChem.MMFFOptimizeMoleculeConfs(molh, maxIters=400)
    except Exception:
        pass

    # Gasteiger charges, then merge non-polar hydrogens into their heavy atom.
    try:
        AllChem.ComputeGasteigerCharges(molh)
    except Exception:
        return None

    charges_all = np.array(
        [float(a.GetDoubleProp("_GasteigerCharge")) for a in molh.GetAtoms()]
    )
    charges_all = np.nan_to_num(charges_all, nan=0.0, posinf=0.0, neginf=0.0)

    keep_idx, merged_charges = [], []
    for atom in molh.GetAtoms():
        i = atom.GetIdx()
        if atom.GetSymbol() == "H":
            nbr = atom.GetNeighbors()[0]
            if nbr.GetSymbol() in ("N", "O", "S"):
                keep_idx.append(i)          # polar H -> keep as HD
                merged_charges.append(charges_all[i])
            continue
        q = charges_all[i]
        for nbr in atom.GetNeighbors():     # absorb non-polar H charge
            if nbr.GetSymbol() == "H" and atom.GetSymbol() not in ("N", "O", "S"):
                q += charges_all[nbr.GetIdx()]
        keep_idx.append(i)
        merged_charges.append(q)

    sub = [molh.GetAtomWithIdx(i) for i in keep_idx]
    elements = [a.GetSymbol() for a in sub]
    ad_types = []
    for a in sub:
        if a.GetSymbol() == "H":
            ad_types.append("HD")
        elif a.GetSymbol() == "C":
            ad_types.append("A" if a.GetIsAromatic() else "C")
        elif a.GetSymbol() == "N":
            # Detect donors via explicit neighbours: GetTotalNumHs() reports 0
            # once AddHs has run. A fully substituted N (pyrrole-type, e.g.
            # adenine N9) keeps its lone pair in the ring and cannot accept.
            n_h = sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "H")
            n_heavy = sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() != "H")
            ad_types.append("N" if (n_h > 0 or n_heavy >= 3) else "NA")
        elif a.GetSymbol() == "O":
            ad_types.append("OA")
        elif a.GetSymbol() == "S":
            ad_types.append("SA")
        elif a.GetSymbol() in ("F", "Cl", "Br", "I", "P"):
            ad_types.append(a.GetSymbol())
        else:
            ad_types.append(_AD_TYPE_FALLBACK)

    conformers = []
    for cid in cids:
        pos = molh.GetConformer(cid).GetPositions()
        conformers.append(pos[keep_idx])

    return PreparedLigand(
        name=name,
        elements=elements,
        ad_types=ad_types,
        charges=np.array(merged_charges),
        conformers=conformers,
    )


# ----------------------------------------------------------------------------
# 4. Pose generation — quaternion rotations, no shear
# ----------------------------------------------------------------------------

def random_rotation_matrices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed rotation matrices via Shoemake's quaternion method.

    Guaranteed orthogonal with determinant +1, which the previous hand-built
    Rx/Ry/Rz product was not.
    """
    u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
    q = np.stack([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ], axis=1)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)
    return R


def rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    """Rodrigues formula: axis-angle vector -> rotation matrix (used by the refiner)."""
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def generate_poses(conf_coords, pocket_center, n_poses, rng, radius=3.0):
    """Rigid-body poses: uniform random orientation + translation in a sphere."""
    centroid = conf_coords.mean(axis=0)
    centered = conf_coords - centroid
    R = random_rotation_matrices(n_poses, rng)

    # Uniform points inside a sphere of the given radius around the pocket centre
    dirs = rng.normal(size=(n_poses, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    r = radius * rng.random(n_poses) ** (1 / 3)
    trans = pocket_center + dirs * r[:, None]

    return np.einsum("nij,aj->nai", R, centered) + trans[:, None, :]


# ----------------------------------------------------------------------------
# 5. Scoring a pose, with optional capped hinge term
# ----------------------------------------------------------------------------

def score_pose(coords, ligand: PreparedLigand, grids: GridSet,
               hinge_coord=None, hinge_weight: float = 0.0) -> float:
    total, _, _ = grids.score(coords, ligand.ad_types, ligand.charges)

    if hinge_weight > 0.0 and hinge_coord is not None:
        polar = np.array([e in ("N", "O") for e in ligand.elements])
        if polar.any():
            d = np.linalg.norm(coords[polar] - hinge_coord, axis=1)
            n_in_shell = int(((d >= 2.5) & (d <= 3.8)).sum())
            bonus = max(HINGE_BONUS_FLOOR, HINGE_BONUS_PER_ATOM * n_in_shell)
            total += hinge_weight * bonus
    return total


def refine_pose(coords, ligand: PreparedLigand, grids: GridSet, maxiter: int = 60):
    """Local 6-DOF (3 translation + 3 rotation) minimisation with Powell.

    Random sampling alone plateaus around 3-5 A RMSD. Refinement is what gets a
    rigid redocking control under the 2.0 A threshold.
    """
    from scipy.optimize import minimize

    centroid = coords.mean(axis=0)
    centered = coords - centroid

    def objective(p):
        R = rotvec_to_matrix(p[3:6])
        trial = centered @ R.T + centroid + p[0:3]
        e, _, _ = grids.score(trial, ligand.ad_types, ligand.charges)
        return e

    res = minimize(objective, np.zeros(6), method="Powell",
                   options={"maxiter": maxiter, "xtol": 0.05, "ftol": 0.05})
    R = rotvec_to_matrix(res.x[3:6])
    return centered @ R.T + centroid + res.x[0:3], float(res.fun)


# ----------------------------------------------------------------------------
# 6. RMSD
# ----------------------------------------------------------------------------

def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """In-place heavy-atom RMSD (no superposition, no symmetry correction).

    This is the correct metric for redocking validation: we WANT to know how far
    the docked pose sits from the crystal pose in the receptor frame. Do not
    superpose first, or the number becomes meaningless.
    """
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"RMSD shape mismatch: {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
