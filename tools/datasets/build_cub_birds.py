#!/usr/bin/env python
"""Generate the cub_birds dataset (pairwise + multi-choice) from cassiekang/cub200_dataset."""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

PAIR_QUESTION = "Do these two images show a bird of the same species?"

MULTI_QUESTION = (
    "The first image is a reference image. How many of the other images show a bird of the same species as the "
    "reference image? Answer briefly with a number from 0 to 3 in <answer> tags as <answer>n</answer>."
)

YES_LABEL = "yes"
NO_LABEL = "no"
PAIR_SAMPLES_PER_LABEL = 500
MULTI_SAMPLES_PER_CLASS = 250


def _make_image_entry(image, species: str, image_id: int, source_path: str) -> Dict:
    return {
        "image": image.convert("RGB"),
        "species": species,
        "image_id": image_id,
        "source_path": source_path,
    }


def _group_by_species(dataset) -> Dict[str, List[Tuple[int, Dict]]]:
    grouped: Dict[str, List[Tuple[int, Dict]]] = defaultdict(list)
    for idx, sample in enumerate(tqdm(dataset, desc="Indexing species", unit="sample")):
        species = sample["text"].strip()
        if not species:
            continue
        grouped[species].append((idx, sample))
    return grouped


def _sample_positive_pairs(grouped, limit: int, rng: random.Random):
    eligible = [species for species, entries in grouped.items() if len(entries) >= 2]
    if not eligible:
        raise RuntimeError("No species with at least two images available.")
    seen = set()
    pairs = []
    attempts = 0
    max_attempts = limit * 20
    while len(pairs) < limit and attempts < max_attempts:
        species = rng.choice(eligible)
        entries = grouped[species]
        if len(entries) < 2:
            attempts += 1
            continue
        (idx_a, sample_a), (idx_b, sample_b) = rng.sample(entries, 2)
        key = tuple(sorted((idx_a, idx_b)))
        if key in seen:
            attempts += 1
            continue
        seen.add(key)
        pairs.append((species, sample_a, sample_b))
        attempts += 1
    if len(pairs) < limit:
        raise RuntimeError(f"Could only sample {len(pairs)} positive pairs (requested {limit}).")
    return pairs


def _sample_negative_pairs(grouped, limit: int, rng: random.Random):
    species_list = [species for species, entries in grouped.items() if entries]
    if len(species_list) < 2:
        raise RuntimeError("Need at least two species to form negative pairs.")
    seen = set()
    pairs = []
    attempts = 0
    max_attempts = limit * 10
    while len(pairs) < limit and attempts < max_attempts:
        sp_a, sp_b = rng.sample(species_list, 2)
        sample_a = rng.choice(grouped[sp_a])[1]
        sample_b = rng.choice(grouped[sp_b])[1]
        key = tuple(sorted((id(sample_a["image"]), id(sample_b["image"]))))
        if key in seen:
            attempts += 1
            continue
        seen.add(key)
        pairs.append((sp_a, sample_a, sp_b, sample_b))
        attempts += 1
    if len(pairs) < limit:
        raise RuntimeError(f"Could only sample {len(pairs)} negative pairs (requested {limit}).")
    return pairs


def _sample_multi_images(grouped, count_value: int, rng: random.Random):
    assert 0 <= count_value <= 3
    species_list = [species for species, entries in grouped.items() if entries]
    if not species_list:
        raise RuntimeError("Dataset is empty.")

    max_attempts = 100
    for _ in range(max_attempts):
        eligible = [species for species, entries in grouped.items() if len(entries) >= count_value + 1]
        if not eligible:
            raise RuntimeError(f"No species with at least {count_value + 1} images available.")
        ref_species = rng.choice(eligible)
        entries = grouped[ref_species]
        ref_idx, ref_sample = rng.choice(entries)
        positives_pool = [(idx, sample) for idx, sample in entries if idx != ref_idx]
        if len(positives_pool) < count_value:
            continue
        positives = rng.sample(positives_pool, count_value)

        negatives_needed = 3 - count_value
        negatives: List[Tuple[int, Dict]] = []
        if negatives_needed > 0:
            other_species = [species for species in species_list if species != ref_species]
            if len(other_species) < negatives_needed:
                continue
            chosen_species = rng.sample(other_species, negatives_needed)
            negatives = [rng.choice(grouped[species]) for species in chosen_species]

        entries_ordered = [(ref_idx, ref_sample)] + positives + negatives
        tests = entries_ordered[1:]
        rng.shuffle(tests)
        return [entries_ordered[0]] + tests

    raise RuntimeError("Failed to sample multi-choice example after multiple attempts.")


def build_pair_records(positives, negatives, rng: random.Random):
    records: List[Dict] = []
    counts = {YES_LABEL: 0, NO_LABEL: 0}
    counter = 0
    for species, sample_a, sample_b in positives:
        record_id = f"pair_{counter:04d}"
        entry_a = _make_image_entry(sample_a["image"], species, counter * 2, sample_a.get("id", ""))
        entry_b = _make_image_entry(sample_b["image"], species, counter * 2 + 1, sample_b.get("id", ""))
        records.append(
            {
                "id": record_id,
                "question_type": "pairwise",
                "question": PAIR_QUESTION,
                "answer": YES_LABEL,
                "species": [species, species],
                "images": [entry_a, entry_b],
            }
        )
        counts[YES_LABEL] += 1
        counter += 1

    for sp_a, sample_a, sp_b, sample_b in negatives:
        record_id = f"pair_{counter:04d}"
        entry_a = _make_image_entry(sample_a["image"], sp_a, counter * 2, sample_a.get("id", ""))
        entry_b = _make_image_entry(sample_b["image"], sp_b, counter * 2 + 1, sample_b.get("id", ""))
        records.append(
            {
                "id": record_id,
                "question_type": "pairwise",
                "question": PAIR_QUESTION,
                "answer": NO_LABEL,
                "species": [sp_a, sp_b],
                "images": [entry_a, entry_b],
            }
        )
        counts[NO_LABEL] += 1
        counter += 1

    rng.shuffle(records)
    return records, counts


def build_multi_records(grouped, rng: random.Random):
    records: List[Dict] = []
    distribution = {str(i): 0 for i in range(4)}
    counter = 0
    for count_value in range(4):
        for _ in range(MULTI_SAMPLES_PER_CLASS):
            entries = _sample_multi_images(grouped, count_value, rng)
            record_id = f"multi_{counter:04d}"
            images = []
            for idx, (image_idx, sample) in enumerate(entries):
                images.append(_make_image_entry(sample["image"], sample["text"].strip(), counter * 4 + idx, sample.get("id", "")))
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
    rng.shuffle(records)
    return records, distribution


def write_dataset(records: Sequence[Dict], output_dir: Path, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    if images_dir.exists() and overwrite:
        shutil.rmtree(images_dir)
    images_dir.mkdir(exist_ok=True)

    json_path = output_dir / "cub_birds.jsonl"
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
                "source_paths": [image_entry["source_path"] for image_entry in record["images"]],
            }
            if record["question_type"] == "multiple_choice":
                doc["target_count"] = record["target_count"]

            handle.write(json.dumps(doc) + "\n")
            progress.update(1)

    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the cub_birds dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lmms_eval/tasks/cub_birds/data"),
        help="Destination directory for images + jsonl.",
    )
    parser.add_argument("--seed", type=int, default=31, help="RNG seed.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    parser.add_argument("--cache-dir", type=Path, default=Path("hf_cache_cub"), help="HF cache directory.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset = load_dataset(
        "cassiekang/cub200_dataset",
        split="test",
        cache_dir=str(args.cache_dir),
        trust_remote_code=True,
    )

    grouped = _group_by_species(dataset)
    positives = _sample_positive_pairs(grouped, PAIR_SAMPLES_PER_LABEL, rng)
    negatives = _sample_negative_pairs(grouped, PAIR_SAMPLES_PER_LABEL, rng)
    pair_records, pair_counts = build_pair_records(positives, negatives, rng)

    multi_records, multi_distribution = build_multi_records(grouped, rng)

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[2] / output_dir

    all_records = pair_records + multi_records
    json_path = write_dataset(all_records, output_dir, overwrite=args.overwrite)

    stats = {
        "total": len(all_records),
        "pairwise_total": len(pair_records),
        "pairwise_label_distribution": pair_counts,
        "multiple_choice_total": len(multi_records),
        "multiple_choice_distribution": multi_distribution,
    }
    stats_path = output_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(f"Wrote {len(all_records)} samples -> {json_path}")
    print(f"Pairwise label distribution: {pair_counts}")
    print(f"Multiple choice distribution: {multi_distribution}")


if __name__ == "__main__":
    main()
