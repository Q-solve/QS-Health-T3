"""
prep_structures.py
==================
Turn the raw RCSB entry 1HCK into the two inputs the docking pipeline needs,
*in the native crystallographic coordinate frame*:

  1HCK_rec.pdbqt  -- CDK2 receptor, AutoDock 4 atom types, AMBER charges from
                     pdb2pqr, non-polar hydrogens merged into their heavy atom.
  ATP_ref.pdbqt   -- the crystallographic ATP ligand, bond orders restored from
                     a SMILES template, Gasteiger charges, non-polar H merged.

Why this script exists
----------------------
The grid maps shipped with the first version of this project were centred at
(-9.344, -10.640, 8.487) while RCSB 1HCK sits near (100, 98, 82). No
translation, centroid convention or P2(1)2(1)2(1) symmetry operation relates
the two frames, so the provenance of those maps could not be reconstructed and
every score computed against them was of unknown meaning. Everything is
therefore rebuilt from the PDB entry, which makes the whole pipeline
reproducible from the accession code alone.

Waters and the Mg(2+) ion are removed. That is the usual convention for
screening ATP-competitive kinase inhibitors, and it keeps the redocking control
honest: the control validates exactly the maps the screen later uses.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

ATP_SMILES = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O"

# Aromatic carbons by residue, typed "A" in AutoDock 4.
AROMATIC_C = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "CD2", "CE1"},
    "HIE": {"CG", "CD2", "CE1"},
    "HID": {"CG", "CD2", "CE1"},
    "HIP": {"CG", "CD2", "CE1"},
}
# Ring nitrogens that carry no hydrogen are acceptors (NA); everything else
# nitrogenous in a protein is a donor or fully substituted (N).
HIS_RING_N = {"ND1", "NE2"}


def read_pqr(path):
    """Parse a pdb2pqr PQR file into a list of atom dictionaries."""
    atoms = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            parts = line.split()
            # whitespace-delimited: ATOM serial name resName chain resSeq x y z q r
            try:
                name = parts[2]
                res_name = parts[3]
                res_num = int(parts[5])
                x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
                q = float(parts[9])
            except (IndexError, ValueError):
                continue
            atoms.append(
                {
                    "name": name,
                    "res_name": res_name,
                    "res_num": res_num,
                    "coord": np.array([x, y, z]),
                    "q": q,
                }
            )
    return atoms


def element_of(name: str) -> str:
    """Infer the element from a PDB atom name."""
    n = name.strip()
    if n[0].isdigit():
        n = n[1:]
    if n[:2].upper() in ("CL", "BR", "MG", "ZN", "FE", "MN", "NA", "CA"):
        # In protein atom names these two-letter forms essentially never occur
        # for carbon/nitrogen, so only trust them for genuine hetero atoms.
        pass
    return n[0].upper()


def receptor_ad_type(atom) -> str:
    """AutoDock 4 atom type for a protein atom (polar H already identified)."""
    el = element_of(atom["name"])
    res = atom["res_name"]
    name = atom["name"]
    if el == "C":
        return "A" if name in AROMATIC_C.get(res, ()) else "C"
    if el == "N":
        if res.startswith("HI") and name in HIS_RING_N:
            return "NA"
        return "N"
    if el == "O":
        return "OA"
    if el == "S":
        return "SA"
    if el == "H":
        return "HD"
    return "C"


def merge_nonpolar_hydrogens(atoms):
    """United-atom conversion.

    Hydrogens bonded to carbon are deleted and their charge is added to that
    carbon. Hydrogens bonded to N/O/S are kept and typed HD, because AutoDock's
    hydrogen-bond term is defined on the polar hydrogen itself.
    """
    coords = np.array([a["coord"] for a in atoms])
    elements = [element_of(a["name"]) for a in atoms]
    heavy_idx = [i for i, e in enumerate(elements) if e != "H"]
    h_idx = [i for i, e in enumerate(elements) if e == "H"]

    heavy_coords = coords[heavy_idx]
    keep = {i: True for i in range(len(atoms))}

    for i in h_idx:
        d = np.linalg.norm(heavy_coords - coords[i], axis=1)
        j_local = int(np.argmin(d))
        if d[j_local] > 1.4:          # orphan hydrogen, drop it
            keep[i] = False
            continue
        j = heavy_idx[j_local]
        if elements[j] == "C":
            atoms[j]["q"] += atoms[i]["q"]
            keep[i] = False           # non-polar H: merged away
    return [a for i, a in enumerate(atoms) if keep[i]]


def write_pdbqt(atoms, path, header_lines=()):
    with open(path, "w") as fh:
        for h in header_lines:
            fh.write(f"REMARK {h}\n")
        for serial, a in enumerate(atoms, start=1):
            x, y, z = a["coord"]
            # Strict PDB column layout. Getting this wrong by even one column
            # silently corrupts the charge (cols 71-76) and the AutoDock type
            # (cols 78-79) that every downstream reader depends on.
            fh.write(
                "ATOM  {:>5d} {:<4s} {:>3s} A{:>4d}    "
                "{:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}    "
                "{:>6.3f} {:<2s}\n".format(
                    serial,
                    a["name"][:4],
                    a["res_name"][:3],
                    a["res_num"],
                    x, y, z,
                    1.00, 0.00,
                    a["q"],
                    a["ad_type"],
                )
            )


# ---------------------------------------------------------------------------
# Ligand
# ---------------------------------------------------------------------------

def nitrogen_type(atom) -> str:
    """AutoDock type for a ligand nitrogen: NA if it can accept, otherwise N.

    Two traps live here. `GetTotalNumHs()` returns 0 once hydrogens are
    explicit, so donors must be detected by looking at the neighbours. And a
    fully substituted nitrogen -- pyrrole-type ring N such as adenine N9 -- has
    its lone pair in the aromatic system and is not an acceptor, so counting
    heavy neighbours matters too.
    """
    n_h = sum(1 for nb in atom.GetNeighbors() if nb.GetSymbol() == "H")
    n_heavy = sum(1 for nb in atom.GetNeighbors() if nb.GetSymbol() != "H")
    if n_h > 0:
        return "N"          # donor
    if n_heavy >= 3:
        return "N"          # fully substituted, lone pair unavailable
    return "NA"             # pyridine-type acceptor


def build_atp(pdb_path: str):
    """Restore bond orders on the crystal ATP and compute Gasteiger charges."""
    block = open(pdb_path).read()
    raw = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False,
                               proximityBonding=True)
    if raw is None:
        raise RuntimeError("RDKit could not read ATP_crystal.pdb")

    template = Chem.MolFromSmiles(ATP_SMILES)
    try:
        mol = AllChem.AssignBondOrdersFromTemplate(template, raw)
    except Exception as exc:
        raise RuntimeError(f"Bond-order assignment against the ATP template failed: {exc}")

    Chem.SanitizeMol(mol)
    molh = Chem.AddHs(mol, addCoords=True)
    AllChem.ComputeGasteigerCharges(molh)

    q = np.array([float(a.GetDoubleProp("_GasteigerCharge")) for a in molh.GetAtoms()])
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

    out = []
    for atom in molh.GetAtoms():
        i = atom.GetIdx()
        sym = atom.GetSymbol()
        if sym == "H":
            nbr = atom.GetNeighbors()[0]
            if nbr.GetSymbol() in ("N", "O", "S"):
                pos = molh.GetConformer().GetAtomPosition(i)
                out.append({"name": "HD", "res_name": "ATP", "res_num": 400,
                            "coord": np.array([pos.x, pos.y, pos.z]),
                            "q": q[i], "ad_type": "HD"})
            continue
        charge = q[i]
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() == "H" and sym not in ("N", "O", "S"):
                charge += q[nbr.GetIdx()]
        if sym == "C":
            ad = "A" if atom.GetIsAromatic() else "C"
        elif sym == "N":
            ad = nitrogen_type(atom)
        elif sym == "O":
            ad = "OA"
        elif sym == "S":
            ad = "SA"
        elif sym in ("P", "F", "Cl", "Br", "I"):
            ad = sym
        else:
            ad = "C"
        pos = molh.GetConformer().GetAtomPosition(i)
        info = atom.GetPDBResidueInfo()
        name = info.GetName().strip() if info else sym
        out.append({"name": name or sym, "res_name": "ATP", "res_num": 400,
                    "coord": np.array([pos.x, pos.y, pos.z]),
                    "q": charge, "ad_type": ad})
    return out


def main():
    d = os.path.dirname(os.path.abspath(__file__))
    proj = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(d), "CDK2_Project")

    pqr = os.path.join(proj, "1HCK_protein.pqr")
    atoms = read_pqr(pqr)
    print(f"PQR atoms (with hydrogens): {len(atoms)}")
    print(f"  total charge before merge: {sum(a['q'] for a in atoms):+.3f}")

    atoms = merge_nonpolar_hydrogens(atoms)
    for a in atoms:
        a["ad_type"] = receptor_ad_type(a)
    print(f"United-atom receptor: {len(atoms)} atoms")
    print(f"  total charge after merge : {sum(a['q'] for a in atoms):+.3f}")
    from collections import Counter
    print("  AutoDock types:", dict(Counter(a["ad_type"] for a in atoms)))

    rec_path = os.path.join(proj, "1HCK_rec.pdbqt")
    write_pdbqt(atoms, rec_path, header_lines=[
        "CDK2 from RCSB 1HCK, native crystallographic frame.",
        "Charges: AMBER via pdb2pqr 3.7.1. Non-polar hydrogens merged.",
        "Waters and the Mg2+ ion removed.",
    ])
    print(f"  -> {rec_path}")

    lig = build_atp(os.path.join(proj, "ATP_crystal.pdb"))
    lig_path = os.path.join(proj, "ATP_ref.pdbqt")
    write_pdbqt(lig, lig_path, header_lines=[
        "Crystallographic ATP from 1HCK, native frame.",
        "Bond orders from SMILES template; Gasteiger charges (neutral form).",
    ])
    heavy = [a for a in lig if a["ad_type"] != "HD"]
    print(f"ATP ligand: {len(lig)} atoms ({len(heavy)} heavy, "
          f"{len(lig) - len(heavy)} polar H)")
    print(f"  total charge: {sum(a['q'] for a in lig):+.3f}")
    print(f"  -> {lig_path}")

    # Geometry that determines the grid box.
    hv = np.array([a["coord"] for a in heavy])
    centroid = hv.mean(axis=0)
    rad = np.linalg.norm(hv - centroid, axis=1).max()
    print("\nGrid box guidance")
    print(f"  ATP heavy-atom centroid : {np.round(centroid, 3).tolist()}")
    print(f"  max atom-centroid radius: {rad:.2f} A")
    print(f"  ATP bounding box        : {np.round(hv.max(0) - hv.min(0), 2).tolist()}")
    print(f"  half-width needed for a +/-3.5 A translation search: {rad + 3.5:.2f} A")


if __name__ == "__main__":
    main()
