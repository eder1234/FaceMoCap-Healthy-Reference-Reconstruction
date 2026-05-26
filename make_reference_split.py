#!/usr/bin/env python3
"""
Add a participant-level reference/evaluation split to FaceMoCap metadata.

The split is intended for healthy-reference construction:
  - usable healthy participants are split into reference/evaluation cohorts
  - usable pathological participants are assigned to evaluation
  - unusable rows are assigned to excluded

Usable rows are single-movement, valid-for-processing samples.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = {
    "participant_id",
    "facial_movement",
    "single_movement",
    "condition",
    "valid_for_processing",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Create a reference_split column in FaceMoCap metadata."
    )
    ap.add_argument(
        "--metadata",
        default="facemocap_metadata.csv",
        help="Input metadata CSV.",
    )
    ap.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV. Default: <metadata stem>_reference_split.csv. "
            "Ignored when --in_place is used."
        ),
    )
    ap.add_argument(
        "--eval_frac",
        type=float,
        default=0.20,
        help="Fraction of usable healthy participants assigned to evaluation.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the healthy participant split.",
    )
    ap.add_argument(
        "--in_place",
        action="store_true",
        help="Overwrite the input metadata CSV instead of creating a new file.",
    )
    return ap.parse_args()


def validate_columns(fieldnames: list[str]) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(fieldnames))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")


def is_one(value) -> bool:
    try:
        return int(float(str(value).strip())) == 1
    except ValueError:
        return False


def movement_label(value) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if text.upper().startswith("M"):
        digits = "".join(ch for ch in text[1:] if ch.isdigit())
        return f"M{int(digits)}" if digits else text.upper()
    try:
        return f"M{int(float(text))}"
    except ValueError:
        return text.upper()


def choose_eval_participants(
    healthy_participants: list[str], eval_frac: float, seed: int
) -> set[str]:
    if not 0.0 < eval_frac < 1.0:
        raise ValueError("--eval_frac must be between 0 and 1.")
    if not healthy_participants:
        return set()

    rng = np.random.default_rng(seed)
    participants = np.array(sorted(healthy_participants), dtype=object)
    rng.shuffle(participants)

    n_eval = int(round(len(participants) * eval_frac))
    n_eval = max(1, min(len(participants) - 1, n_eval)) if len(participants) > 1 else 1
    return set(str(x) for x in participants[:n_eval])


def print_participant_list(title: str, participants: list[str]) -> None:
    print(f"\n{title} ({len(participants)} participants)")
    if participants:
        print("  " + ", ".join(participants))
    else:
        print("  <none>")


def print_counts(df: list[dict[str, str]]) -> None:
    print("\nRow counts by split and condition")
    counts = Counter(
        (row["reference_split"], row["condition"])
        for row in df
        if row["reference_split"] in {"reference", "evaluation"}
    )
    print_table(["reference_split", "condition", "n_rows"], counts)

    print("\nUsable row counts by split, condition, and movement")
    movement_counts = Counter(
        (row["reference_split"], row["condition"], movement_label(row["facial_movement"]))
        for row in df
        if row["reference_split"] in {"reference", "evaluation"}
    )
    print_table(["reference_split", "condition", "movement", "n_rows"], movement_counts)

    excluded = sum(1 for row in df if row["reference_split"] == "excluded")
    print(f"\nExcluded rows: {excluded}")


def print_table(headers: list[str], counts: Counter) -> None:
    rows = [tuple([*key, value]) for key, value in sorted(counts.items())]
    if not rows:
        print("  <none>")
        return

    widths = [
        max(len(str(header)), *(len(str(row[idx])) for row in rows))
        for idx, header in enumerate(headers)
    ]
    print(" ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)))
    for row in rows:
        print(" ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)))


def read_metadata(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata has no header: {path}")
        rows = list(reader)
        return rows, list(reader.fieldnames)


def write_metadata(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    rows, fieldnames = read_metadata(metadata_path)
    validate_columns(fieldnames)

    healthy_participants = sorted(
        {
            row["participant_id"].strip()
            for row in rows
            if is_usable(row)
            and row["condition"].strip().lower() == "healthy"
            and row["participant_id"].strip()
        }
    )
    patho_participants = sorted(
        {
            row["participant_id"].strip()
            for row in rows
            if is_usable(row)
            and row["condition"].strip().lower() == "pathological"
            and row["participant_id"].strip()
        }
    )
    eval_healthy = choose_eval_participants(
        healthy_participants, eval_frac=args.eval_frac, seed=args.seed
    )
    ref_healthy = set(healthy_participants) - eval_healthy

    for row in rows:
        row["reference_split"] = split_for_row(row, ref_healthy, eval_healthy)

    if "reference_split" not in fieldnames:
        fieldnames.append("reference_split")

    print("Proposed participant-level reference_split")
    print(f"Metadata: {metadata_path}")
    print(f"Seed: {args.seed}")
    print(f"Healthy evaluation fraction: {args.eval_frac:.2f}")

    print_participant_list("Healthy reference", sorted(ref_healthy))
    print_participant_list("Healthy evaluation", sorted(eval_healthy))
    print_participant_list("Pathological evaluation", patho_participants)
    print_counts(rows)

    if args.in_place:
        output_path = metadata_path
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = metadata_path.with_name(f"{metadata_path.stem}_reference_split.csv")

    write_metadata(output_path, rows, fieldnames)
    print(f"\nWrote: {output_path}")


def is_usable(row: dict[str, str]) -> bool:
    return is_one(row["single_movement"]) and is_one(row["valid_for_processing"])


def split_for_row(
    row: dict[str, str], ref_healthy: set[str], eval_healthy: set[str]
) -> str:
    if not is_usable(row):
        return "excluded"

    condition = row["condition"].strip().lower()
    participant = row["participant_id"].strip()

    if condition == "healthy" and participant in ref_healthy:
        return "reference"
    if condition == "healthy" and participant in eval_healthy:
        return "evaluation"
    if condition == "pathological" and participant:
        return "evaluation"
    return "excluded"


if __name__ == "__main__":
    main()
