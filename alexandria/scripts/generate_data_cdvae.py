#!/usr/bin/env python
"""
generate_data_cdvae_csv.py – Build train/val/test CSV splits from Alexandria
superconductivity CSV files.

Example
-------
python scripts/generate_data_cdvae_csv.py \
    --csv-files dataset1.csv dataset2.csv \
    --id-key mat_id \
    --target Tc \
    --max-size 1000 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --split-seed 123 \
    --output-dir .
"""

from __future__ import annotations

import argparse
import ast
import random
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import Structure
from tqdm import tqdm

# ────────────────────────── CrystalNN set-up ────────────────────────────────
CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None,
    x_diff_weight=-1,
    porous_adjustment=False,
)


# ───────────────────── splitting helper functions ───────────────────────────
def compute_split_counts(
    total_size: int, val_ratio: float, test_ratio: float
) -> Tuple[int, int, int]:
    """Return train/val/test counts that sum to total_size."""
    n_val = int(val_ratio * total_size)
    n_test = int(test_ratio * total_size)
    n_train = total_size - n_val - n_test
    return n_train, n_val, n_test


def get_id_splits(
    total_size: int,
    split_seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    keep_data_order: bool = False,
) -> Tuple[List[int], List[int], List[int]]:
    """Return shuffled (or ordered) index splits for train/val/test."""
    indices = list(range(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(indices)
    train = indices[:n_train]
    val = indices[n_train : n_train + n_val]
    test = indices[n_train + n_val : n_train + n_val + n_test]
    return train, val, test


# ──────────────────────────────── main ───────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split Alexandria CSVs into train/val/test for CDVAE."
    )
    parser.add_argument(
        "--csv-files",
        required=True,
        nargs="+",
        metavar="CSV",
        help="Paths to one or more Alexandria CSV files.",
    )
    parser.add_argument(
        "--id-key",
        default="mat_id",
        help="Column name for the material identifier.",
    )
    parser.add_argument(
        "--target",
        dest="target_key",
        default="Tc",
        help="Column name for the target property.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1000,
        help="Maximum number of structures to collect.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of total to use for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of total to use for testing.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=123,
        help="Random seed for train/val/test splitting.",
    )
    parser.add_argument(
        "--check-graph",
        action="store_true",
        help="If set, attempt building a StructureGraph for each entry.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory in which to write train/val/test CSVs.",
    )
    args = parser.parse_args()

    # Gather all rows from CSVs
    csv_paths = [Path(p).expanduser().resolve() for p in args.csv_files]
    df_all = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)

    collected: List[dict[str, Any]] = []
    for row in tqdm(
        df_all.itertuples(index=False), total=len(df_all), desc="Reading"
    ):
        if len(collected) >= args.max_size:
            break

        tgt_val = getattr(row, args.target_key, "na")
        if tgt_val in ("na", None) or (
            isinstance(tgt_val, float) and np.isnan(tgt_val)
        ):
            continue

        try:
            struct_dict = ast.literal_eval(getattr(row, "structure"))
            pmg_struct = Structure.from_dict(struct_dict)

            if args.check_graph:
                _ = StructureGraph.with_local_env_strategy(pmg_struct, CrystalNN)

            collected.append(
                {
                    "material_id": getattr(row, args.id_key),
                    "cif": pmg_struct.to(fmt="cif"),
                    args.target_key: tgt_val,
                }
            )
        except Exception:
            # Skip parse/graph/CIF errors
            continue

    actual_size = len(collected)
    if actual_size == 0:
        raise RuntimeError(
            f"No valid entries found for target '{args.target_key}'."
        )
    if actual_size < args.max_size:
        print(f"Warning: only collected {actual_size}/{args.max_size} structures.")

    # Compute splits
    n_train, n_val, n_test = compute_split_counts(
        actual_size, args.val_ratio, args.test_ratio
    )
    train_idx, val_idx, test_idx = get_id_splits(
        total_size=actual_size,
        split_seed=args.split_seed,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
    )

    df_train = pd.DataFrame([collected[i] for i in train_idx])
    df_val = pd.DataFrame([collected[i] for i in val_idx])
    df_test = pd.DataFrame([collected[i] for i in test_idx])

    # Write outputs
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(out_dir / "train.csv", index=False)
    df_val.to_csv(out_dir / "val.csv", index=False)
    df_test.to_csv(out_dir / "test.csv", index=False)

    print(
        f"Exported {len(df_train):d} train / {len(df_val):d} val / "
        f"{len(df_test):d} test entries → {out_dir}"
    )


if __name__ == "__main__":
    main()

