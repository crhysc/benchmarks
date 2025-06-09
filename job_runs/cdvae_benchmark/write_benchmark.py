#!/usr/bin/env python
"""
make_cdvae_submission.py (debug version)

Same as the original, but with additional logging in rms_check()
to report when and why RMS values aren’t computed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jarvis.core.atoms import Atoms, pmg_to_atoms
from jarvis.core.lattice import Lattice
from jarvis.core.specie import atomic_numbers_to_symbols
from jarvis.io.vasp.inputs import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from tqdm import tqdm


def batched_tensors_to_atoms(batch_dict):
    """Convert a single-batch dict written by CDVAE into a list of Atoms."""
    num_atoms = batch_dict["num_atoms"][0]
    atom_types = batch_dict["atom_types"][0]
    frac_coords = batch_dict["frac_coords"][0]
    lengths = batch_dict["lengths"][0]
    angles = batch_dict["angles"][0]

    idx_cumsum = torch.cumsum(num_atoms, dim=0).tolist()
    boundaries = [(0, idx_cumsum[0])]
    for i in range(1, len(idx_cumsum)):
        boundaries.append((idx_cumsum[i - 1], idx_cumsum[i]))

    structures = []
    for idx, (s, e) in enumerate(boundaries):
        lat = Lattice.from_parameters(
            *lengths[idx],
            *angles[idx]
        ).matrix
        atoms = Atoms(
            lattice_mat=lat,
            elements=atomic_numbers_to_symbols(atom_types[s:e].tolist()),
            coords=frac_coords[s:e].tolist(),
            cartesian=False,
        )
        structures.append(atoms)

    return structures


def read_split(csv_path):
    """Read a JARVIS CSV split → (list[Atoms], list[JID])."""
    df = pd.read_csv(csv_path)
    structs, jids = [], []
    for _, row in df.iterrows():
        atoms = pmg_to_atoms(Structure.from_str(row["cif"], fmt="cif"))
        structs.append(atoms)
        jids.append(row["material_id"])
    return structs, jids


def rms_check(target_atoms, pred_atoms):
    """Anonymous RMS distance averaged over all structures, with debug prints."""
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    rms_vals = []

    for i, (tgt, pred) in enumerate(tqdm(zip(target_atoms, pred_atoms),
                                         total=len(target_atoms))):
        try:
            score, mapping = matcher.get_rms_anonymous(
                pred.pymatgen_converter(),
                tgt.pymatgen_converter(),
            )
            if score is None:
                print(f"  [#{i}] No match found within tolerances "
                      f"(stol=0.5Å, ltol=0.3, angle_tol=10°).")
            else:
                rms_vals.append(score)
        except Exception as e:
            print(f"  [#{i}] Error during matching: {e}")

    if rms_vals:
        mean_rms = float(np.mean(rms_vals))
        return round(mean_rms, 4)
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",    required=True,
                        help="Folder containing eval_*.pt")
    parser.add_argument("--data_dir",   required=True,
                        help="Folder with train/val/test CSVs")
    parser.add_argument("--split",      default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--output_csv", required=True,
                        help="Submission CSV name")
    parser.add_argument("--dump_json",  help="Optional JSON dump of all splits")
    parser.add_argument("--use_tensor", default="recon",
                        choices=["recon", "gen", "opt"],
                        help="Which eval_*.pt to use (default: recon)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    tensor_map = {"recon": "eval_recon.pt",
                  "gen":   "eval_gen.pt",
                  "opt":   "eval_opt.pt"}
    tensor_path = run_dir / tensor_map[args.use_tensor]
    if not tensor_path.exists():
        raise FileNotFoundError(f"Cannot find {tensor_path}")

    # Load predictions
    batch = torch.load(tensor_path, map_location="cpu")
    pred_structs = batched_tensors_to_atoms(batch)

    # Load targets
    target_structs, jids = read_split(Path(args.data_dir) /
                                       f"{args.split}.csv")
    if len(target_structs) != len(pred_structs):
        raise RuntimeError("Mismatch between dataset length and CDVAE output length")

    # Write submission CSV
    with open(args.output_csv, "w") as f:
        f.write("id,target,prediction\n")
        for jid, tgt, pred in zip(jids, target_structs, pred_structs):
            tgt_poscar  = Poscar(tgt).to_string().replace("\n", r"\n")
            pred_poscar = Poscar(pred).to_string().replace("\n", r"\n")
            f.write(f"{jid},{tgt_poscar},{pred_poscar}\n")
    print(f"[✓] Submission written to {args.output_csv}")

    # Optional JSON dump
    if args.dump_json:
        info = {}
        for split_name in ["train", "val", "test"]:
            structs, ids = read_split(Path(args.data_dir) / f"{split_name}.csv")
            info[split_name] = {
                jid: Poscar(at).to_string().replace("\n", r"\n")
                for jid, at in zip(ids, structs)
            }
        with open(args.dump_json, "w") as jf:
            json.dump(info, jf)
        print(f"[✓] JSON dump saved to {args.dump_json}")

    # RMS sanity check with debug
    print("\nRunning RMS debug check:")
    rms = rms_check(target_structs, pred_structs)
    if rms is not None:
        print(f"[✓] Anonymous RMS (mean over {len(target_structs)} structures): {rms}")
    else:
        print("[!] RMS could not be computed for any structure")


if __name__ == "__main__":
    main()

