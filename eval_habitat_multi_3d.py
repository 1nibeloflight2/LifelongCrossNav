"""
Multi-episode evaluation with 3D VoxelMap navigation.

This entry point is for dataset-scale evaluation, not live visualization:
rerun is not initialized, but files keep the "_rerun" suffix so the saved
trajectory/voxel CSVs can still be replayed later.
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# Parse local args before Spock intercepts sys.argv.
parser = argparse.ArgumentParser(description="Batch 3D ObjectNav evaluation.")
parser.add_argument("--config", default="config/mon/eval_multi_conf_floor.yaml")
parser.add_argument("--out_dir", default="")
parser.add_argument("--suffix", default="rerun")
parser.add_argument(
    "--start", type=int, default=0,
    help="Start index after optional episode-ID filtering.",
)
parser.add_argument(
    "--limit", type=int, default=0,
    help="Number of selected episodes to run; 0 means all.",
)
episode_selector = parser.add_mutually_exclusive_group()
episode_selector.add_argument(
    "--episode_ids",
    nargs="+",
    default=None,
    metavar="ID",
    help="Episode IDs to evaluate; accepts spaces and/or commas.",
)
episode_selector.add_argument(
    "--episode_ids_file",
    type=Path,
    default=None,
    help="Text file containing episode IDs separated by commas or whitespace.",
)
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], "--config", args.config] + remaining

from config import load_eval_config
from eval.actor_3d import MONActor3D
from eval.habitat_multi_evaluator import HabitatMultiEvaluator
from eval_single_episode_3d import TeeLogger, log_line, run_single_episode


def _episode_label(episode_id: Any) -> str:
    try:
        return f"{int(episode_id):03d}"
    except Exception:
        return str(episode_id).replace("/", "_")


def _episode_key(episode_id: Any) -> str:
    """Normalize numeric IDs while retaining support for string IDs."""
    value = str(episode_id).strip()
    try:
        return str(int(value))
    except ValueError:
        return value


def _parse_episode_id_tokens(tokens: Iterable[str]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        for value in token.replace(",", " ").split():
            key = _episode_key(value)
            if key and key not in seen:
                selected.append(key)
                seen.add(key)
    return selected


def _requested_episode_ids() -> list[str] | None:
    if args.episode_ids is not None:
        return _parse_episode_id_tokens(args.episode_ids)
    if args.episode_ids_file is None:
        return None
    if not args.episode_ids_file.is_file():
        raise SystemExit(f"Episode ID file not found: {args.episode_ids_file}")
    tokens: list[str] = []
    with args.episode_ids_file.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if line:
                tokens.append(line)
    return _parse_episode_id_tokens(tokens)


def main() -> None:
    eval_config = load_eval_config()

    # Batch evaluation should not start rerun logging/visualization.
    try:
        eval_config.EvalConf.log_rerun = False
    except Exception:
        pass

    actor = MONActor3D(eval_config.EvalConf)
    evaluator = HabitatMultiEvaluator(eval_config.EvalConf, actor)
    try:
        evaluator.log_rerun = False
        evaluator.logger = None
    except Exception:
        pass

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    root_dir = args.out_dir or os.path.join(os.getcwd(), f"result_3d_{timestamp}")
    if (
        eval_config.EvalConf.reset_map_between_targets and
        os.path.basename(os.path.normpath(root_dir)) == "result_3d_history_poi_off"
    ):
        raise SystemExit(
            "The History-POI-OFF output directory cannot be reused for "
            "the no-cross-target-memory experiment."
        )
    os.makedirs(root_dir, exist_ok=True)

    all_episodes = list(evaluator.episodes)
    requested_ids = _requested_episode_ids()
    if requested_ids is None:
        episodes = all_episodes
        selector_description = "all episodes"
    else:
        episodes_by_id = {_episode_key(ep.episode_id): ep for ep in all_episodes}
        missing_ids = [episode_id for episode_id in requested_ids if episode_id not in episodes_by_id]
        if missing_ids:
            preview = ", ".join(missing_ids[:20])
            more = f" ... (+{len(missing_ids) - 20})" if len(missing_ids) > 20 else ""
            raise SystemExit(f"Episode IDs not found in dataset: {preview}{more}")
        episodes = [episodes_by_id[episode_id] for episode_id in requested_ids]
        selector_description = f"explicit ID list ({len(requested_ids)})"

    # start/limit are applied after explicit-ID filtering, which also permits
    # deterministic splitting of one ID file across multiple workers.
    if args.start:
        episodes = episodes[args.start:]
    if args.limit and args.limit > 0:
        episodes = episodes[:args.limit]

    completed_labels: set[str] = set()
    if args.out_dir:
        for episode in episodes:
            label = _episode_label(episode.episode_id)
            suffix_text = f"_{args.suffix}" if args.suffix else ""
            state_path = os.path.join(
                root_dir, f"episode_{label}", f"state_{label}{suffix_text}.txt"
            )
            if os.path.isfile(state_path) and os.path.getsize(state_path) > 0:
                completed_labels.add(label)

    pending_episodes = [
        episode for episode in episodes
        if _episode_label(episode.episode_id) not in completed_labels
    ]

    print("=== Multi-episode 3D VoxelMap Evaluation ===")
    print(
        f"Loaded episodes: {len(evaluator.episodes)}; selector: {selector_description}; "
        f"selected after start/limit: {len(episodes)}; "
        f"completed: {len(completed_labels)}; pending: {len(pending_episodes)}"
    )
    print(
        f"VoxelMap: voxel_xy={actor.mapper.one_map.voxel_size:.3f}m, "
        f"voxel_z={actor.mapper.one_map.voxel_size_z:.3f}m, "
        f"n_cells={actor.mapper.one_map.n_cells}"
    )
    print("Rerun visualization: disabled")
    print(f"Results root: {root_dir}")

    original_stdout = sys.stdout
    for run_idx, episode in enumerate(pending_episodes, start=1):
        label = _episode_label(episode.episode_id)
        episode_dir = os.path.join(root_dir, f"episode_{label}")
        os.makedirs(episode_dir, exist_ok=True)
        log_path = os.path.join(episode_dir, f"ep{label}_{args.suffix}.log")

        tee = TeeLogger(log_path, original_stdout)
        sys.stdout = tee
        try:
            log_line(
                f"\n=== Batch episode {episode.episode_id} "
                f"({run_idx}/{len(pending_episodes)} pending; "
                f"{len(completed_labels)} previously completed) ===",
                log_path,
            )
            run_single_episode(
                evaluator=evaluator,
                episode_id=episode.episode_id,
                suffix=args.suffix,
                log_path=log_path,
                out_dir=episode_dir,
                use_rerun=False,
                episode_label=label,
                save_stats_file=False,
            )
        finally:
            sys.stdout = original_stdout
            tee.close()

    print(f"Evaluation complete. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
