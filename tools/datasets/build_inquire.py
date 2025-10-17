#!/usr/bin/env python
"""Generate the inquire dataset (pairwise + multi-choice) from evendrow/INQUIRE-Rerank."""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from datasets import load_dataset
from tqdm import tqdm

PAIR_QUESTION = "Do these two images show an animal of the same scientific species?"

MULTI_QUESTION = (
    "The first image is a reference image. How many of the other images show an animal of the same scientific "
    "species as the reference image? Answer briefly with a number from 0 to 3 in <answer> tags as <answer>n</answer>."
)

YES_LABEL = "yes"
NO_LABEL = "no"
PAIR_SAMPLES_PER_LABEL = 500
MULTI_SAMPLES_PER_CLASS = 250


def index_by_species(entries: Iterable[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for sample in tqdm(entries, desc="Indexing species", unit="sample"):
        species = sample["inat24_species_name"]
        if not species:
            continue
        grouped[species].append(
            {
                "species": species,
                "image_id": sample["inat24_image_id"],
                "file_name": sample["inat24_file_name"],
                "image": sample["image"].convert("RGB"),
            }
        )
    return grouped


def make_image_entry(entry: Dict) -> Dict:
    return {
        "image": entry["image"],
        "species": entry["species"],
        "image_id": entry["image_id"],
        "file_name": entry["file_name"],
    }


def sample_same_pairs(grouped: Dict[str, List[Dict]], limit: int, rng: random.Random) -> List[Tuple[Dict, Dict]]:
    pool: List[Tuple[Dict, Dict]] = []
    for entries in grouped.values():
        if len(entries) < 2:
            continue
        indices = list(range(len(entries)))
        rng.shuffle(indices)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pool.append((entries[indices[i]], entries[indices[j]]))
    if len(pool) < limit:
        raise RuntimeError(f"Need {limit} same-species pairs, only found {len(pool)}.")
    rng.shuffle(pool)
    seen = set()
    selected: List[Tuple[Dict, Dict]] = []
    for left, right in pool:
        key = tuple(sorted((left["image_id"], right["image_id"])))
        if key in seen:
            continue
        seen.add(key)
        selected.append((left, right))
        if len(selected) == limit:
            break
    if len(selected) < limit:
        raise RuntimeError(f"Could not sample enough same-species pairs ({len(selected)}).")
    return selected


def sample_diff_pairs(grouped: Dict[str, List[Dict]], limit: int, rng: random.Random) -> List[Tuple[Dict, Dict]]:
    species_list = [sp for sp, entries in grouped.items() if entries]
    if len(species_list) < 2:
        raise RuntimeError("Need at least two distinct species to form different-species pairs.")
    seen = set()
    selected: List[Tuple[Dict, Dict]] = []
    while len(selected) < limit:
        sp_a, sp_b = rng.sample(species_list, 2)
        left = rng.choice(grouped[sp_a])
        right = rng.choice(grouped[sp_b])
        key = tuple(sorted((left["image_id"], right["image_id"])))
        if key in seen:
            continue
        seen.add(key)
        selected.append((left, right))
    return selected


def build_pair_records(
    positives: Sequence[Tuple[Dict, Dict]],
    negatives: Sequence[Tuple[Dict, Dict]],
) -> Tuple[List[Dict], Dict[str, int]]:
    records: List[Dict] = []
    counts = {YES_LABEL: 0, NO_LABEL: 0}
    counter = 0

    for left, right in positives:
        record_id = f"pair_{counter:04d}"
        records.append(
            {
                "id": record_id,
                "question_type": "pairwise",
                "question": PAIR_QUESTION,
                "answer": YES_LABEL,
                "species": [left["species"], right["species"]],
                "images": [make_image_entry(left), make_image_entry(right)],
            }
        )
        counts[YES_LABEL] += 1
        counter += 1

    for left, right in negatives:
        record_id = f"pair_{counter:04d}"
        records.append(
            {
                "id": record_id,
                "question_type": "pairwise",
                "question": PAIR_QUESTION,
                "answer": NO_LABEL,
                "species": [left["species"], right["species"]],
                "images": [make_image_entry(left), make_image_entry(right)],
            }
        )
        counts[NO_LABEL] += 1
        counter += 1

    return records, counts


def sample_multi_images(
    grouped: Dict[str, List[Dict]],
    count_value: int,
    rng: random.Random,
) -> List[Dict]:
    assert 0 <= count_value <= 3, "count_value must be between 0 and 3."
    species_list = [sp for sp, entries in grouped.items() if entries]
    if not species_list:
        raise RuntimeError("Dataset is empty.")

    max_attempts = 100
    for _ in range(max_attempts):
        eligible_species = [sp for sp, entries in grouped.items() if len(entries) >= count_value + 1]
        if not eligible_species:
            raise RuntimeError(f"No species with at least {count_value + 1} images.")
        ref_species = rng.choice(eligible_species)
        ref_entries = grouped[ref_species]

        ref_entry = rng.choice(ref_entries)
        positives_pool = [entry for entry in ref_entries if entry is not ref_entry]
        if len(positives_pool) < count_value:
            continue
        positives = rng.sample(positives_pool, count_value)

        negatives_needed = 3 - count_value
        negatives: List[Dict] = []
        if negatives_needed > 0:
            other_species = [sp for sp in species_list if sp != ref_species]
            if len(other_species) < negatives_needed:
                continue
            chosen_species = rng.sample(other_species, negatives_needed)
            negatives = [rng.choice(grouped[sp]) for sp in chosen_species]

        reference = make_image_entry(ref_entry)
        test_entries = [make_image_entry(entry) for entry in positives]
        test_entries.extend(make_image_entry(entry) for entry in negatives)
        rng.shuffle(test_entries)
        return [reference] + test_entries

    raise RuntimeError("Failed to sample multi-choice example after multiple attempts.")


def build_multi_records(grouped: Dict[str, List[Dict]], rng: random.Random) -> Tuple[List[Dict], Dict[str, int]]:
    records: List[Dict] = []
    distribution = {str(i): 0 for i in range(4)}
    counter = 0

    for count_value in range(4):
        for _ in range(MULTI_SAMPLES_PER_CLASS):
            record_id = f"multi_{counter:04d}"
            images = sample_multi_images(grouped, count_value, rng)
            records.append(
                {
                    "id": record_id,
                    "question_type": "multiple_choice",
                    "question": MULTI_QUESTION,
                    "answer": str(count_value),
                    "target_count": count_value,
                    "species": [img["species"] for img in images],
                    "images": images,
                }
            )
            distribution[str(count_value)] += 1
            counter += 1

    return records, distribution


def write_dataset(records: Sequence[Dict], output_dir: Path, overwrite: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    if images_dir.exists() and overwrite:
        shutil.rmtree(images_dir)
    images_dir.mkdir(exist_ok=True)

    json_path = output_dir / "inquire.jsonl"
    stats_path = output_dir / "stats.json"

    if json_path.exists() and not overwrite:
        raise FileExistsError(f"{json_path} already exists. Use --overwrite to replace it.")

    with json_path.open("w", encoding="utf-8") as handle, tqdm(total=len(records), desc="Writing dataset", unit="sample") as progress:
        for record in records:
            record_id = record["id"]
            image_paths: List[str] = []
            for idx, image_entry in enumerate(record["images"]):
                suffix = chr(ord("a") + idx)
                img_path = images_dir / f"{record_id}_{suffix}.jpg"
                image_entry["image"].save(img_path, quality=95)
                image_paths.append(str(img_path.relative_to(output_dir)))

            doc = {
                "pair_id": record_id,
                "sample_id": record_id,
                "question_type": record["question_type"],
                "question": record["question"],
                "answer": record["answer"],
                "image_paths": image_paths,
                "species": record.get("species"),
                "instance_ids": [image_entry["image_id"] for image_entry in record["images"]],
                "source_keys": [image_entry["file_name"] for image_entry in record["images"]],
            }
            if record["question_type"] == "multiple_choice":
                doc["target_count"] = record["target_count"]

            handle.write(json.dumps(doc) + "\n")
            progress.update(1)

    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the inquire dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lmms_eval/tasks/inquire/data"),
        help="Destination directory for images + jsonl.",
    )
    parser.add_argument("--seed", type=int, default=17, help="RNG seed.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    parser.add_argument("--cache-dir", type=Path, default=Path("hf_cache"), help="HuggingFace cache directory.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset = load_dataset(
        "evendrow/INQUIRE-Rerank",
        split="validation",
        trust_remote_code=True,
        cache_dir=str(args.cache_dir),
    )
    grouped = index_by_species(dataset.with_format("python"))

    positives = sample_same_pairs(grouped, PAIR_SAMPLES_PER_LABEL, rng)
    negatives = sample_diff_pairs(grouped, PAIR_SAMPLES_PER_LABEL, rng)
    pair_records, pair_counts = build_pair_records(positives, negatives)

    multi_records, multi_distribution = build_multi_records(grouped, rng)

    all_records = pair_records + multi_records
    json_path = write_dataset(all_records, args.output_dir, overwrite=args.overwrite)

    stats = {
        "total": len(all_records),
        "pairwise_total": len(pair_records),
        "pairwise_label_distribution": pair_counts,
        "multiple_choice_total": len(multi_records),
        "multiple_choice_distribution": multi_distribution,
    }
    stats_path = args.output_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(f"Wrote {len(all_records)} samples -> {json_path}")
    print(f"Pairwise label distribution: {pair_counts}")
    print(f"Multiple choice distribution: {multi_distribution}")


if __name__ == "__main__":
    main()
