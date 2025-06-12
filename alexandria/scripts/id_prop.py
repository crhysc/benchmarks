#!/usr/bin/env python
"""
id_prop.py – Build POSCAR + id_prop.csv from one or more Alexandria-style CSVs.

Example
-------
python id_prop.py \
    --csv-files dataset1.csv dataset2.csv \
    --id-key mat_id --target Tc \
    --seed 123 --max-size 1000 \
    --output data/supercon_csv
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
import random
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from pymatgen.core import Structure
from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar


# ────────────────────── debug helpers ────────────────────────────
#   • set DEBUG=true … to enable
#   • use dprint() instead of print() for conditional logging

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

def dprint(*args, **kwargs):
    """Debug-print if DEBUG=true."""
    if DEBUG:
        print(*args, **kwargs)


# ───────────────────────── helpers ────────────────────────────────────────────
def make_dataframe_from_csv(
    csv_paths: list[Path],
    id_key: str,
    target_key: str,
    max_size: Optional[int],
) -> pd.DataFrame:
    """Read one or many CSV files and keep rows with a valid target value."""
    dfs = [pd.read_csv(p) for p in csv_paths]
    combined = pd.concat(dfs, ignore_index=True)

    records: list[dict] = []

    # counters for post-mortem
    counts = {
        "invalid_target": 0,
        "struct_parse_fail": 0,
        "convert_fail": 0,
        "poscar_fail": 0,
        "kept": 0,
    }

    iter_rows = tqdm(
        combined.itertuples(index=False),
        total=len(combined),
        desc="Parsing CSV rows",
    )

    for idx, row in enumerate(iter_rows):
        target_val = getattr(row, target_key, "na")
        if (
            target_val in ("na", None)
            or (isinstance(target_val, float) and np.isnan(target_val))
        ):
            counts["invalid_target"] += 1
            dprint(f"[skip #{idx}] invalid target ({target_val})")
            continue  # skip invalid target

        # 1) Parse the pymatgen Structure dict that is stored as a string
        try:
            struct_dict = ast.literal_eval(getattr(row, "structure"))
            pmg_struct = Structure.from_dict(struct_dict)
        except Exception as e:
            counts["struct_parse_fail"] += 1
            dprint(f"[skip #{idx}] structure parse failed: {e}")
            if DEBUG and counts["struct_parse_fail"] <= 3:
                traceback.print_exc()
            continue  # malformed structure string → skip

        # 2) Convert to a JARVIS Atoms object for Poscar writing
        try:
            atoms = Atoms.from_pymatgen(pmg_struct)
        except Exception as e:
            counts["convert_fail"] += 1
            dprint(f"[skip #{idx}] Atoms conversion failed: {e}")
            if DEBUG and counts["convert_fail"] <= 3:
                traceback.print_exc()
            continue  # conversion failed → skip

        # 3) Sanity-check: can Poscar serialize it?
        try:
            _ = Poscar(atoms)
        except Exception as e:
            counts["poscar_fail"] += 1
            dprint(f"[skip #{idx}] Poscar serialisation failed: {e}")
            if DEBUG and counts["poscar_fail"] <= 3:
                traceback.print_exc()
            continue  # malformed lattice / species → skip

        records.append(
            {
                id_key: getattr(row, id_key),
                "atoms": atoms,
                target_key: target_val,
            }
        )
        counts["kept"] += 1

        if max_size is not None and len(records) == max_size:
            break  # hard-cap just like the original script

    # final tallies
    if DEBUG:
        dprint("\n─── row filtering summary ───")
        for k, v in counts.items():
            dprint(f"{k:<20}: {v}")

    return pd.DataFrame(records)


def sha(lst) -> str:
    h = hashlib.sha256()
    for x in lst:
        h.update(str(x).encode())
        h.update(b",")
    return h.hexdigest()[:10]


# ──────────────────────────── CLI ─────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv-files",
        nargs="+",
        required=True,
        metavar="CSV",
        help="One or more CSV files produced from the Alexandria dataset.",
    )
    ap.add_argument("--id-key", default="mat_id")
    ap.add_argument("--target", dest="target_key", default="Tc")
    ap.add_argument(
        "--output",
        required=True,
        help="Directory in which POSCAR files + id_prop.csv are written",
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Cap on the number of *valid* structures",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    csv_paths = [Path(p).expanduser().resolve() for p in args.csv_files]
    df = make_dataframe_from_csv(
        csv_paths, args.id_key, args.target_key, args.max_size
    )
    print(f"Collected {len(df)} valid records (max-size={args.max_size})")

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    structure_paths: List[str] = []
    target_values: List[float] = []

    print("Writing POSCAR files …")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        jid = row[args.id_key]
        atoms = row["atoms"]
        target_val = row[args.target_key]

        fname = f"{jid}.vasp"
        fpath = out_dir / fname

        Poscar(atoms).write_file(str(fpath))

        structure_paths.append(fname)
        target_values.append(target_val)

    id_prop = pd.DataFrame(
        {"structure_path": structure_paths, args.target_key: target_values}
    )
    csv_path = out_dir / "id_prop.csv"
    id_prop.to_csv(csv_path, index=False, header=False)

    print(f"✓ Wrote {len(structure_paths)} POSCAR files to {out_dir}")
    print(f"✓ Wrote {csv_path.name}")
    print(f"SHA-10 of ids: {sha(structure_paths)}")

    if DEBUG and not structure_paths:
        dprint(
            "\nNo structures survived the filters. "
            "Check the summary above for the dominant failure mode."
        )


if __name__ == "__main__":
    main()
