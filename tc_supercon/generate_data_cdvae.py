import pandas as pd
import random
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph

# --------------- USER-CONFIGURABLE PARAMETERS ---------------
MAX_SIZE    = 25      # Maximum number of structures to gather
SPLIT_SEED  = 123     # Random seed for reproducibility
VAL_RATIO   = 0.1     # Fraction of MAX_SIZE to assign to validation
TEST_RATIO  = 0.1     # Fraction of MAX_SIZE to assign to testing
CHECK_GRAPH = False   # If True, attempt to build a StructureGraph for each entry
TAG         = "jid"   # Field name for the material ID in the dataset
DATASET     = "dft_3d"
PROPERTY    = "Tc_supercon"
# --------------------------------------------------------------

# Initialize the local‐environment-based neighbor‐finder (CrystalNN)
CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None,
    x_diff_weight=-1,
    porous_adjustment=False,
)


def compute_split_counts(total_size, val_ratio, test_ratio):
    """
    Compute integer counts for train/val/test so that:
      n_val   = floor(val_ratio * total_size)
      n_test  = floor(test_ratio * total_size)
      n_train = total_size - (n_val + n_test)
    This guarantees n_train + n_val + n_test == total_size.
    """
    n_val = int(val_ratio * total_size)
    n_test = int(test_ratio * total_size)
    n_train = total_size - n_val - n_test
    return n_train, n_val, n_test


def get_id_splits(
    total_size,
    split_seed,
    n_train,
    n_val,
    n_test,
    keep_data_order=False,
):
    """
    Return three lists of indices (id_train, id_val, id_test) corresponding
    to a random split of range(total_size). If keep_data_order is True,
    no shuffling is performed; otherwise, indices are shuffled with split_seed.
    """
    indices = list(range(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(indices)

    # Train = first n_train, Val = next n_val, Test = next n_test
    id_train = indices[:n_train]
    id_val = indices[n_train : n_train + n_val]
    id_test = indices[
        n_train + n_val : n_train + n_val + n_test
    ]
    return id_train, id_val, id_test


def main():
    # 1) Gather up to MAX_SIZE entries with a valid PROPERTY
    dat = data(DATASET)
    collected = []
    for entry in dat:
        if len(collected) == MAX_SIZE:
            break

        if entry[PROPERTY] != "na":
            info = {}
            info["material_id"] = entry[TAG]
            pmg = Atoms.from_dict(entry["atoms"]).pymatgen_converter()
            try:
                if CHECK_GRAPH:
                    _ = StructureGraph.with_local_env_strategy(
                        pmg,
                        CrystalNN,
                    )
                info["cif"] = pmg.to(fmt="cif")
                info["prop"] = entry[PROPERTY]
                collected.append(info)
            except Exception:
                # Skip any entries that fail graph construction or CIF conversion
                continue

    actual_size = len(collected)
    if actual_size == 0:
        raise RuntimeError(
            "No valid entries found in dataset "
            f"'{DATASET}' with property '{PROPERTY}'."
        )
    if actual_size < MAX_SIZE:
        print(
            "Warning: Only "
            f"{actual_size} entries were collected (MAX_SIZE={MAX_SIZE})."
        )

    # 2) Compute how many samples go into each split
    n_train, n_val, n_test = compute_split_counts(
        actual_size,
        VAL_RATIO,
        TEST_RATIO,
    )
    if n_train + n_val + n_test != actual_size:
        # Sanity check (should not fail)
        raise AssertionError(
            f"Split counts do not add up: {n_train}+{n_val}+"
            f"{n_test} != {actual_size}"
        )

    # 3) Generate train/val/test index lists
    id_train, id_val, id_test = get_id_splits(
        total_size=actual_size,
        split_seed=SPLIT_SEED,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        keep_data_order=False,
    )

    # 4) Build DataFrames for each split
    df_train = pd.DataFrame([collected[i] for i in id_train])
    df_val = pd.DataFrame([collected[i] for i in id_val])
    df_test = pd.DataFrame([collected[i] for i in id_test])

    # 5) Write to CSV (no index column)
    df_train.to_csv("train.csv", index=False)
    df_val.to_csv("val.csv", index=False)
    df_test.to_csv("test.csv", index=False)

    print(
        "Exported "
        f"{len(df_train)} train / {len(df_val)} val / "
        f"{len(df_test)} test entries."
    )
    print("Files written: train.csv, val.csv, test.csv")


if __name__ == "__main__":
    main()

