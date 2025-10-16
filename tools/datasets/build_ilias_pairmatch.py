#!/usr/bin/env python
"""Generate the 4k yes/no pair-matching dataset from vrg-prague/ilias core_db."""

import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

QUESTION = (
    "Do these images contain the same or identical products? For two products to be considered identical, "
    "minor changes such as those that can be explained context, backgrounds or photography conditions are "
    "allowed, but characteristic features of the product (color, shape, size, etc.) should remain consistent. "
    "Explain your reasoning and then conclude with a yes or no answer in <answer> tags as "
    "<answer>yes</answer> or <answer>no</answer>"
)
YES_LABEL = "yes"
NO_LABEL = "no"


def parse_instance_id(key: str) -> str:
    if "P" not in key:
        raise ValueError(f"Unexpected __key__ pattern: {key}")
    after_p = key.split("P", 1)[1]
    digits = "".join(ch for ch in after_p if ch.isdigit())
    if len(digits) < 4:
        raise ValueError(f"Unexpected __key__ pattern: {key}")
    return digits[:4]


def index_by_instance(entries: Iterable[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for sample in tqdm(entries, desc="Indexing instances", unit="sample"):
        instance_id = parse_instance_id(sample["__key__"])
        grouped[instance_id].append(
            {
                "__key__": sample["__key__"],
                "instance_id": instance_id,
                "image": sample["jpg"].convert("RGB"),
            }
        )
    return grouped


def sample_same_pairs(grouped: Dict[str, List[Dict]], limit: int, rng: random.Random) -> List[Tuple[Dict, Dict]]:
    pool: List[Tuple[Dict, Dict]] = []
    for entries in grouped.values():
        if len(entries) < 2:
            continue
        combos = itertools.combinations(range(len(entries)), 2)
        pool.extend((entries[i], entries[j]) for i, j in combos)
    if len(pool) < limit:
        raise RuntimeError(f"Only {len(pool)} same-instance combos available; need {limit}.")
    rng.shuffle(pool)
    seen = set()
    picked: List[Tuple[Dict, Dict]] = []
    with tqdm(total=limit, desc="Sampling same-instance pairs", unit="pair") as progress:
        for left, right in pool:
            key = tuple(sorted((left["__key__"], right["__key__"])))
            if key in seen:
                continue
            seen.add(key)
            picked.append((left, right))
            progress.update(1)
            if len(picked) == limit:
                break
    if len(picked) < limit:
        raise RuntimeError(f"Could not collect {limit} unique same-instance pairs (got {len(picked)}).")
    return picked


def sample_diff_pairs(grouped: Dict[str, List[Dict]], limit: int, rng: random.Random) -> List[Tuple[Dict, Dict]]:
    inst_ids = [inst for inst, entries in grouped.items() if entries]
    if len(inst_ids) < 2:
        raise RuntimeError("Need at least two distinct instance IDs to form negative pairs.")
    seen = set()
    picked: List[Tuple[Dict, Dict]] = []
    with tqdm(total=limit, desc="Sampling different-instance pairs", unit="pair") as progress:
        while len(picked) < limit:
            id_a, id_b = rng.sample(inst_ids, 2)
            left = rng.choice(grouped[id_a])
            right = rng.choice(grouped[id_b])
            key = tuple(sorted((left["__key__"], right["__key__"])))
            if key in seen:
                continue
            seen.add(key)
            picked.append((left, right))
            progress.update(1)
    return picked


def write_dataset(
    pairs: Sequence[Tuple[Dict, Dict]],
    labels: Sequence[str],
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    json_path = output_dir / "ilias_pairmatch.jsonl"
    stats_path = output_dir / "stats.json"

    if json_path.exists() and not overwrite:
        raise FileExistsError(f"{json_path} already exists. Use --overwrite to replace it.")

    with json_path.open("w", encoding="utf-8") as handle, tqdm(total=len(labels), desc="Writing dataset", unit="pair") as progress:
        for idx in range(len(labels)):
            left, right = pairs[idx]
            label = labels[idx]
            pair_id = f"{idx:04d}"
            left_path = images_dir / f"{pair_id}_a.jpg"
            right_path = images_dir / f"{pair_id}_b.jpg"
            left["image"].save(left_path, quality=95)
            right["image"].save(right_path, quality=95)

            record = {
                "pair_id": pair_id,
                "question": QUESTION,
                "answer": label,
                "image_paths": [
                    str(left_path.relative_to(output_dir)),
                    str(right_path.relative_to(output_dir)),
                ],
                "instance_ids": [left["instance_id"], right["instance_id"]],
                "source_keys": [left["__key__"], right["__key__"]],
                "same_instance": label == YES_LABEL,
            }
            handle.write(json.dumps(record) + "\n")
            progress.update(1)

    stats = {
        "total": len(labels),
        "label_distribution": {YES_LABEL: labels.count(YES_LABEL), NO_LABEL: labels.count(NO_LABEL)},
    }
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(f"Wrote {len(labels)} samples -> {json_path}")
    print(f"Label breakdown: {stats['label_distribution']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the ilias_pairmatch dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lmms_eval/tasks/ilias_pairmatch/data"),
        help="Destination directory for images + jsonl.",
    )
    parser.add_argument("--count-per-label", type=int, default=2000, help="Samples per class.")
    parser.add_argument("--seed", type=int, default=13, help="RNG seed.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset = load_dataset("vrg-prague/ilias", "core_db", split="core_db", trust_remote_code=True)
    grouped = index_by_instance(dataset.with_format("python"))

    positives = sample_same_pairs(grouped, args.count_per_label, rng)
    negatives = sample_diff_pairs(grouped, args.count_per_label, rng)

    combined = list(zip(positives, [YES_LABEL] * len(positives))) + list(zip(negatives, [NO_LABEL] * len(negatives)))
    rng.shuffle(combined)
    pairs, labels = zip(*combined)
    write_dataset(pairs, labels, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
