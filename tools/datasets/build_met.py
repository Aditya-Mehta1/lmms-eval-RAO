#!/usr/bin/env python
"""Generate the met dataset (pairwise + multi-choice) from the local MET mini exhibit."""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

PAIR_QUESTION = "Do these two images contain the same piece of art?"

MULTI_QUESTION = (
    "The first image is a reference image. How many of the other images contain the same piece of art as the "
    "reference image? Answer briefly with a number from 0 to 3 in <answer> tags as <answer>n</answer>."
)

YES_LABEL = "yes"
NO_LABEL = "no"
PAIR_SAMPLES_PER_LABEL = 500
MULTI_SAMPLES_PER_CLASS = 250
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _list_images(root: Path) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = defaultdict(list)
    for class_dir in tqdm(list(root.iterdir()), desc="Scanning classes"):
        if not class_dir.is_dir():
            continue
        files = [p for p in class_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if files:
            grouped[class_dir.name].extend(files)
    return grouped


def _make_image_entry(image_path: Path, art_id: str, image_index: int) -> Dict:
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    return {
        "image": image,
        "art_id": art_id,
        "image_index": image_index,
        "source_path": str(image_path),
    }


def _sample_positive_pairs(grouped: Dict[str, List[Path]], limit: int, rng: random.Random) -> List[Tuple[str, Path, Path]]:
    eligible = [art_id for art_id, files in grouped.items() if len(files) >= 2]
    if len(eligible) == 0:
        raise RuntimeError("No art IDs with at least two images available.")
    seen = set()
    pairs = []
    attempts = 0
    max_attempts = limit * 20
    while len(pairs) < limit and attempts < max_attempts:
        art_id = rng.choice(eligible)
        files = grouped[art_id]
        if len(files) < 2:
            attempts += 1
            continue
        left, right = rng.sample(files, 2)
        key = tuple(sorted((str(left), str(right))))
        if key in seen:
            attempts += 1
            continue
        seen.add(key)
        pairs.append((art_id, left, right))
        attempts += 1
    if len(pairs) < limit:
        raise RuntimeError(f"Could only sample {len(pairs)} positive pairs (requested {limit}).")
    return pairs


def _sample_negative_pairs(grouped: Dict[str, List[Path]], limit: int, rng: random.Random) -> List[Tuple[str, Path, str, Path]]:
    art_ids = [art_id for art_id, files in grouped.items() if files]
    if len(art_ids) < 2:
        raise RuntimeError("Need at least two distinct art IDs to form negative pairs.")
    seen = set()
    pairs = []
    attempts = 0
    max_attempts = limit * 10
    while len(pairs) < limit and attempts < max_attempts:
        art_a, art_b = rng.sample(art_ids, 2)
        left = rng.choice(grouped[art_a])
        right = rng.choice(grouped[art_b])
        key = tuple(sorted((str(left), str(right))))
        if key in seen:
            attempts += 1
            continue
        seen.add(key)
        pairs.append((art_a, left, art_b, right))
        attempts += 1
    if len(pairs) < limit:
        raise RuntimeError(f"Could only sample {len(pairs)} negative pairs (requested {limit}).")
    return pairs


def _sample_multi_images(grouped: Dict[str, List[Path]], count_value: int, rng: random.Random) -> List[Tuple[str, Path]]:
    assert 0 <= count_value <= 3
    art_ids = [art_id for art_id, files in grouped.items() if files]
    if not art_ids:
        raise RuntimeError("Dataset is empty.")

    max_attempts = 100
    for _ in range(max_attempts):
        eligible = [art_id for art_id, files in grouped.items() if len(files) >= count_value + 1]
        if not eligible:
            raise RuntimeError(f"No art IDs with at least {count_value + 1} images available.")
        ref_id = rng.choice(eligible)
        ref_files = grouped[ref_id]
        ref_image = rng.choice(ref_files)
        positives_pool = [p for p in ref_files if p != ref_image]
        if len(positives_pool) < count_value:
            continue
        positives = rng.sample(positives_pool, count_value)

        negatives_needed = 3 - count_value
        negatives: List[Tuple[str, Path]] = []
        if negatives_needed > 0:
            other_ids = [art_id for art_id in art_ids if art_id != ref_id]
            if len(other_ids) < negatives_needed:
                continue
            chosen = rng.sample(other_ids, negatives_needed)
            negatives = [(art_id, rng.choice(grouped[art_id])) for art_id in chosen]

        test_entries = [(ref_id, ref_image)]
        test_entries.extend((ref_id, p) for p in positives)
        test_entries.extend(negatives)
        reference = test_entries[0]
        tests = test_entries[1:]
        rng.shuffle(tests)
        return [reference] + tests

    raise RuntimeError("Failed to sample multi-choice example after multiple attempts.")


def build_pair_records(
    positives: Sequence[Tuple[str, Path, Path]],
    negatives: Sequence[Tuple[str, Path, str, Path]],
    rng: random.Random,
) -> Tuple[List[Dict], Dict[str, int]]:
    records: List[Dict] = []
    counts = {YES_LABEL: 0, NO_LABEL: 0}
    counter = 0

    for art_id, left_path, right_path in positives:
        record_id = f"pair_{counter:04d}"
        left_entry = _make_image_entry(left_path, art_id, counter * 2)
        right_entry = _make_image_entry(right_path, art_id, counter * 2 + 1)
        records.append(
            {
                "id": record_id,
                "question_type": "pairwise",
                "question": PAIR_QUESTION,
                "answer": YES_LABEL,
                "art_ids": [art_id, art_id],
                "images": [left_entry, right_entry],
            }
        )
        counts[YES_LABEL] += 1
        counter += 1

    for art_a, left_path, art_b, right_path in negatives:
        record_id = f"pair_{counter:04d}"
        left_entry = _make_image_entry(left_path, art_a, counter * 2)
        right_entry = _make_image_entry(right_path, art_b, counter * 2 + 1)
        records.append(
            {
                "id": record_id,
                "question_type": "pairwise",
                "question": PAIR_QUESTION,
                "answer": NO_LABEL,
                "art_ids": [art_a, art_b],
                "images": [left_entry, right_entry],
            }
        )
        counts[NO_LABEL] += 1
        counter += 1

    rng.shuffle(records)
    return records, counts


def build_multi_records(grouped: Dict[str, List[Path]], rng: random.Random) -> Tuple[List[Dict], Dict[str, int]]:
    records: List[Dict] = []
    distribution = {str(i): 0 for i in range(4)}
    counter = 0
    for count_value in range(4):
        for _ in range(MULTI_SAMPLES_PER_CLASS):
            entries = _sample_multi_images(grouped, count_value, rng)
            record_id = f"multi_{counter:04d}"
            images = []
            for idx, (art_id, path) in enumerate(entries):
                images.append(_make_image_entry(path, art_id, counter * 4 + idx))
            records.append(
                {
                    "id": record_id,
                    "question_type": "multiple_choice",
                    "question": MULTI_QUESTION,
                    "answer": str(count_value),
                    "target_count": count_value,
                    "art_ids": [img["art_id"] for img in images],
                    "images": images,
                }
            )
            distribution[str(count_value)] += 1
            counter += 1
    rng.shuffle(records)
    return records, distribution


def write_dataset(records: Sequence[Dict], output_dir: Path, overwrite: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    if images_dir.exists() and overwrite:
        shutil.rmtree(images_dir)
    images_dir.mkdir(exist_ok=True)

    json_path = output_dir / "met.jsonl"
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
                "art_ids": record.get("art_ids"),
                "instance_ids": [image_entry["image_index"] for image_entry in record["images"]],
                "source_paths": [image_entry["source_path"] for image_entry in record["images"]],
            }
            if record["question_type"] == "multiple_choice":
                doc["target_count"] = record["target_count"]

            handle.write(json.dumps(doc) + "\n")
            progress.update(1)

    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the met dataset.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/met/MET"),
        help="Root directory containing class-organized MET images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lmms_eval/tasks/met/data"),
        help="Destination directory for images + jsonl.",
    )
    parser.add_argument("--seed", type=int, default=29, help="RNG seed.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    script_root = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[3]

    input_dir = args.input_dir
    if not input_dir.exists():
        candidate = repo_root / input_dir
        if candidate.exists():
            input_dir = candidate
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = script_root / output_dir

    grouped = _list_images(input_dir)
    if len(grouped) == 0:
        raise RuntimeError(f"No images found under {args.input_dir}.")

    positives = _sample_positive_pairs(grouped, PAIR_SAMPLES_PER_LABEL, rng)
    negatives = _sample_negative_pairs(grouped, PAIR_SAMPLES_PER_LABEL, rng)
    pair_records, pair_counts = build_pair_records(positives, negatives, rng)

    multi_records, multi_distribution = build_multi_records(grouped, rng)

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
