#!/usr/bin/env python
"""
supercon_preprocess.py  –  Python 3.9 compatible

Example
-------
python supercon_preprocess.py \
    --dataset dft_3d --id-key jid --target Tc_supercon \
    --seed 123 --max-size 1000 \
    --output data/supercon
"""
from __future__ import annotations

import argparse, random, json, hashlib
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from jarvis.db.figshare import data as jarvis_data
from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# ───────────────────────── helpers ────────────────────────────────────────────
def make_dataframe(
    dataset_name: str,
    id_key: str,
    target_key: str,
    max_size: Optional[int],
) -> pd.DataFrame:
    """Download JARVIS records and keep those with a defined target.

    If --max-size is given, we **stop as soon as we have exactly that many
    valid structures** (i.e. after filtering out `"na"` / `None` targets).
    The returned DataFrame has columns:
      - id_key        (e.g. "jid")
      - "atoms"       (a jarvis.core.Atoms instance)
      - target_key    (e.g. "Tc_supercon")
    """
    records: List[dict] = []
    for item in tqdm(jarvis_data(dataset_name), desc="Downloading/JARVIS"):
        target_val = item.get(target_key, "na")
        if target_val in ("na", None):
            continue  # skip invalid target

        # Build a JARVIS Atoms object from the raw dictionary
        try:
            atoms = Atoms.from_dict(item["atoms"])
        except Exception:
            continue  # skip if Atoms.from_dict fails

        # (Optionally) double‐check that we can convert to POSCAR via Poscar(...)
        try:
            _ = Poscar(atoms)  # will raise if the Atoms object is malformed
        except Exception:
            continue  # skip any structure that Poscar() can't handle

        records.append(
            {
                id_key: item[id_key],       # keep original identifier
                "atoms": atoms,             # store the Atoms instance
                target_key: target_val,
            }
        )

        # --- stop exactly at the requested cap -------------------------------
        if max_size is not None and len(records) == max_size:
            break

    return pd.DataFrame(records)


def sha(lst) -> str:
    m = hashlib.sha256()
    for x in lst:
        m.update(str(x).encode())
        m.update(b",")
    return m.hexdigest()[:10]


# ──────────────────────────── CLI ─────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--id-key", default="jid")
    ap.add_argument("--target", dest="target_key", default="Tc_supercon")
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
    np.random.seed(args.seed)  # deterministic ordering if numpy is used later

    # Build a DataFrame that holds `jid`, the Atoms object, and target
    df = make_dataframe(args.dataset, args.id_key, args.target_key, args.max_size)
    print(f"Collected {len(df)} records (max-size={args.max_size})")

    # ── write outputs ────────────────────────────────────────────────────────
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    structure_paths: List[str] = []
    target_values: List[float] = []

    print("Writing POSCAR files …")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        jid = row[args.id_key]
        atoms = row["atoms"]           # jarvis.core.Atoms instance
        target_val = row[args.target_key]

        # Build the filename, e.g. "JVASP-12345.vasp"
        fname = f"{jid}.vasp"
        fpath = out_dir / fname

        # Use the Poscar API to write directly
        p = Poscar(atoms)              # wrap the JARVIS Atoms in a Poscar
        p.write_file(str(fpath))       # write to "<output>/<jid>.vasp"

        structure_paths.append(fname)  # CSV should reference just the basename
        target_values.append(target_val)

    # Build id_prop.csv
    id_prop = pd.DataFrame(
        {
            "structure_path": structure_paths,
            args.target_key: target_values,
        }
    )
    id_prop = id_prop.iloc[1:]
    csv_path = out_dir / "id_prop.csv"
    id_prop.to_csv(csv_path, index=False)

    print(f"✓ Wrote {len(structure_paths)} POSCAR files to {out_dir}")
    print(f"✓ Wrote {csv_path.name}")
    print(f"SHA-10 of ids: {sha(structure_paths)}")


if __name__ == "__main__":
    main()
