#!/usr/bin/env python
"""
make_cdvae_submission.py

Combine the scattered Colab cells for CDVAE inference into one
stand-alone script that

1.  Loads CDVAE tensors (recon / gen / opt) from a run directory.
2.  Re-assembles them into individual `jarvis.core.atoms.Atoms` objects.
3.  Reads the JARVIS dataset CSV split (train / val / test).
4.  Writes a leaderboard-ready submission CSV (newline-escaped POSCARs).
5.  Optionally dumps a helper JSON with POSCARs for every split.
6.  Computes a local RMS check with `pymatgen.analysis.StructureMatcher`.

Usage
-----
python make_cdvae_submission.py \
    --run_dir  /path/to/HYDRA_JOBS/singlerun/<RUN_ID>/supercon_test02 \
    --data_dir /path/to/cdvae/data/supercon \
    --split    test \
    --output_csv AI-AtomGen-Tc_supercon-dft_3d-test-rmse.csv \
    [--dump_json dft_3d_Tc_supercon.json]

Author: Your Name
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


# -----------------------------------------------------------------------------#
#                               Helper functions                               #
# -----------------------------------------------------------------------------#
def batched_tensors_to_atoms(batch_dict):
    """Convert a single-batch dict written by CDVAE into a list of Atoms."""
    num_atoms = batch_dict["num_atoms"][0]  # (n_structs,)
    atom_types = batch_dict["atom_types"][0]
    frac_coords = batch_dict["frac_coords"][0]
    lengths = batch_dict["lengths"][0]
    angles = batch_dict["angles"][0]

    # Boundaries of each structure in the flattened atom list
    idx_cumsum = torch.cumsum(num_atoms, dim=0).tolist()
    boundaries = [(0, idx_cumsum[0])]
    for i in range(1, len(idx_cumsum)):
        boundaries.append((idx_cumsum[i - 1], idx_cumsum[i]))

    structures = []
    for s, e in boundaries:
        lat = Lattice.from_parameters(
            *lengths[len(structures)],
            *angles[len(structures)]
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
    """Anonymous RMS distance averaged over all structures."""
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    rms_vals = []
    for tgt, pred in tqdm(zip(target_atoms, pred_atoms),
                          total=len(target_atoms)):
        try:
            score, _ = matcher.get_rms_anonymous(
                pred.pymatgen_converter(),
                tgt.pymatgen_converter(),
            )
            if score is not None:
                rms_vals.append(score)
        except Exception:
            pass

    if rms_vals:
        return round(float(np.mean(rms_vals)), 4)
    return None


# -----------------------------------------------------------------------------#
#                                      main                                    #
# -----------------------------------------------------------------------------#
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Folder containing eval_*.pt",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Folder with train/val/test CSVs",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Submission CSV name",
    )
    parser.add_argument(
        "--dump_json",
        help="Optional JSON dump of all splits",
    )
    parser.add_argument(
        "--use_tensor",
        default="recon",
        choices=["recon", "gen", "opt"],
        help="Which eval_*.pt to use (default: recon)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    tensor_map = {
        "recon": "eval_recon.pt",
        "gen": "eval_gen.pt",
        "opt": "eval_opt.pt",
    }
    tensor_name = tensor_map[args.use_tensor]
    tensor_path = run_dir / tensor_name
    if not tensor_path.exists():
        raise FileNotFoundError(f"Cannot find {tensor_path}")

    batch = torch.load(tensor_path, map_location="cpu")
    pred_structs = batched_tensors_to_atoms(batch)

    # Read chosen split
    target_structs, jids = read_split(Path(args.data_dir) /
                                       f"{args.split}.csv")
    if len(target_structs) != len(pred_structs):
        raise RuntimeError(
            "Mismatch between dataset length and CDVAE output length"
        )

    # Write submission CSV
    with open(args.output_csv, "w") as f:
        f.write("id,target,prediction\n")
        for jid, tgt, pred in zip(jids, target_structs, pred_structs):
            tgt_poscar = Poscar(tgt).to_string().replace("\n", r"\n")
            pred_poscar = Poscar(pred).to_string().replace("\n", r"\n")
            f.write(f"{jid},{tgt_poscar},{pred_poscar}\n")
    print(f"[✓] Submission written to {args.output_csv}")

    # Optional JSON dump
    if args.dump_json:
        info = {}
        for split_name in ["train", "val", "test"]:
            structs, ids = read_split(Path(args.data_dir) /
                                      f"{split_name}.csv")
            info[split_name] = {
                jid: Poscar(at).to_string().replace("\n", r"\n")
                for jid, at in zip(ids, structs)
            }
        with open(args.dump_json, "w") as jf:
            json.dump(info, jf)
        print(f"[✓] JSON dump saved to {args.dump_json}")

    # RMS sanity check
    rms = rms_check(target_structs, pred_structs)
    if rms is not None:
        print(
            f"[✓] Anonymous RMS (mean over "
            f"{len(target_structs)} structures): {rms}"
        )
    else:
        print("[!] RMS could not be computed for any structure")


if __name__ == "__main__":
    main()
