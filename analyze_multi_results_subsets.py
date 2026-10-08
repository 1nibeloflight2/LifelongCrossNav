#!/usr/bin/env python3
"""Summarize all and selected multi-object episodes from an existing CSV.

The metric definitions and formatting match read_results_multi_floor_3d.py.
No geodesic paths are recomputed; this script only aggregates stored metrics.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


LIST_COLUMNS = ("SR_per", "SPL_per")
DEFAULT_EPISODE_IDS_FILE = "config/mon/cross_floor_episode_ids.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print overall and cross-floor subset metrics from results.csv"
    )
    parser.add_argument(
        "--csv",
        default="results_multi_floor_onemap/results.csv",
        help="Input results.csv",
    )
    parser.add_argument(
        "--episode_ids_file",
        default=None,
        help=(
            "Comma/whitespace-separated subset IDs; # comments are ignored. "
            "When explicitly supplied, write results_corss.csv beside --csv."
        ),
    )
    return parser.parse_args()


def parse_list_cell(value: object) -> list[Optional[float]]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    # CSV list strings may contain the bare token nan, which literal_eval does
    # not recognize. It is equivalent to a missing per-target value here.
    text = re.sub(r"(?<![\w.])(?:nan|NaN)(?![\w.])", "None", text)
    parsed = ast.literal_eval(text)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Expected a list cell, got: {value!r}")
    return list(parsed)


def load_episode_ids(path: Path) -> list[int]:
    ids: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0]
        ids.extend(int(token) for token in re.findall(r"\d+", line))
    if not ids:
        raise ValueError(f"No episode IDs found in {path}")
    duplicates = sorted({episode_id for episode_id in ids if ids.count(episode_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate episode IDs in {path}: {duplicates}")
    return ids


def finite_values(values: Iterable[object]) -> list[float]:
    result = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def value_at(values: list[object], index: int) -> object:
    return values[index] if index < len(values) else None


def print_metrics(title: str, frame: pd.DataFrame, sequence_count: int) -> None:
    print(f"\n========== {title}: Overall Metrics ==========")
    print(f"Episodes: {len(frame)}")
    for column in ("SR", "SPL", "PR", "PPL"):
        values = pd.to_numeric(frame[column], errors="coerce")
        print(f"{column}_all:   {values.mean():.4f}")

    n_per_valid = []
    sr_per_condition = []
    spl_per_condition = []
    spl_per_global = []
    sr_per_global = []

    for seq_num in range(sequence_count):
        sr_values = [value_at(values, seq_num) for values in frame["SR_per"]]
        spl_values = [value_at(values, seq_num) for values in frame["SPL_per"]]
        valid_sr = finite_values(sr_values)
        valid_spl = finite_values(spl_values)

        n_per_valid.append(len(valid_sr))
        sr_per_condition.append(
            sum(1 for value in valid_sr if value == 1) / len(valid_sr)
            if valid_sr
            else 0.0
        )
        spl_per_condition.append(
            sum(valid_spl) / len(valid_spl) if valid_spl else 0.0
        )
        sr_per_global.append(sum(valid_sr) / len(frame) if len(frame) else 0.0)
        spl_per_global.append(sum(valid_spl) / len(frame) if len(frame) else 0.0)

    quoted = lambda values: [f"{value:.4f}" for value in values]
    print(f"N_per_valid:            {n_per_valid}")
    print(f"SR_per_all_condition:   {quoted(sr_per_condition)}")
    print(f"SPL_per_all_condition:  {quoted(spl_per_condition)}")
    print(f"SPL_per_all_global:     {quoted(spl_per_global)}")
    print(f"SR_per_all_global:      {quoted(sr_per_global)}")

    print(f"\n========== {title}: Progress Distribution ==========")
    progress = pd.to_numeric(frame["PR"], errors="coerce").dropna()
    for value, count in progress.value_counts().sort_index().items():
        print(f"PR = {value:.3f}: {count} episodes")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    ids_path = Path(
        args.episode_ids_file or DEFAULT_EPISODE_IDS_FILE
    ).expanduser().resolve()

    source_frame = pd.read_csv(csv_path)
    frame = source_frame.copy()
    required = {"experiment", "SR_per", "SPL_per", "SR", "SPL", "PR", "PPL"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {missing_columns}")

    for column in LIST_COLUMNS:
        frame[column] = frame[column].map(parse_list_cell)
    frame["experiment"] = pd.to_numeric(frame["experiment"], errors="raise").astype(int)

    duplicate_rows = frame.loc[frame["experiment"].duplicated(), "experiment"].tolist()
    if duplicate_rows:
        raise ValueError(f"Duplicate experiment rows in CSV: {duplicate_rows}")

    sequence_count = max((len(values) for values in frame["SR_per"]), default=0)
    subset_ids = load_episode_ids(ids_path)
    available_ids = set(frame["experiment"])
    missing_ids = [episode_id for episode_id in subset_ids if episode_id not in available_ids]
    subset_mask = frame["experiment"].isin(subset_ids)
    subset = frame.loc[subset_mask].copy()
    if args.episode_ids_file is not None and missing_ids:
        raise ValueError(
            f"Cannot export an incomplete subset; missing episode IDs: {missing_ids}"
        )

    print(f"Input CSV: {csv_path}")
    print(
        f"CSV rows: {len(frame)}; unique experiment IDs: {frame['experiment'].nunique()}; "
        f"ID range: {frame['experiment'].min()}..{frame['experiment'].max()}"
    )
    print(
        f"Subset ID file: {ids_path}; requested: {len(subset_ids)}; "
        f"matched: {len(subset)}; missing: {len(missing_ids)}"
    )
    if missing_ids:
        print(f"Missing subset IDs: {missing_ids}")

    for column in ("SR", "SPL", "PR", "PPL"):
        invalid_count = pd.to_numeric(frame[column], errors="coerce").isna().sum()
        if invalid_count:
            print(
                f"Warning: {column} contains {invalid_count} non-finite values; "
                "pandas-compatible means exclude them."
            )

    print_metrics("All episodes", frame, sequence_count)
    print_metrics("Cross-floor subset", subset, sequence_count)
    if args.episode_ids_file is not None:
        output_path = csv_path.parent / "results_corss.csv"
        source_frame.loc[subset_mask].to_csv(output_path, index=False)
        print(f"Saved {len(subset)} selected episodes to {output_path}")


if __name__ == "__main__":
    main()
