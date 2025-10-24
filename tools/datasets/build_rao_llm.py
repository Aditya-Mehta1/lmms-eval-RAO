#!/usr/bin/env python
"""Generate the rao_llm dataset from local Rao test data."""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

PAIR_QUESTION = (
    "Are these images of the same or identical products? For two products to be considered identical, minor changes "
    "such as those that can be explained context, backgrounds or photography conditions are allowed, but "
    "characteristic features of the product (color, shape, size, etc.) should remain consistent. For images with "
    "multiple products, compare only the primary product. Explain your reasoning and then conclude with a yes or no "
    "answer in <answer> tags as <answer>yes</answer> or <answer>no</answer>."
)

MULTI_QUESTION = (
    "The first image is a reference image. How many of the other images depict the same or identical products as the "
    "reference image? For two products to be considered identical, minor changes such as those that can be explained "
    "by context, backgrounds or photography conditions are allowed, but characteristic features of the product "
    "(color, shape, size, etc.) should remain consistent. For images with multiple products, compare only the primary "
    "product. Explain your reasoning and then answer with a number from 0 to 3 in <answer> tags as <answer>n</answer>."
)

YES_LABEL = "yes"
NO_LABEL = "no"
PAIR_POSITIVE_SAMPLES = 500
PAIR_NEGATIVE_REAL_SAMPLES = 250
PAIR_NEGATIVE_GEN_SAMPLES = 250
MULTI_SAMPLES_PER_CLASS = 250


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _make_image_entry(image_path: Path, product_id: str, image_id: int) -> Dict:
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    return {
        "image": image,
        "product_id": product_id,
        "image_id": image_id,
        "source_path": str(image_path),
    }


def _ensure_product(product_map: Dict[str, Dict], product_id: str, ref_path: Path) -> Dict:
    entry = product_map.get(product_id)
    if entry is None:
        entry = {
            "product_id": product_id,
            "ref_path": ref_path,
            "positives": set(),
            "real_negs": set(),
            "gen_negs": set(),
        }
        product_map[product_id] = entry
    return entry


def build_pair_records(
    positives: List[Tuple[str, Path, Path]],
    real_negs: List[Tuple[str, Path, Path]],
    gen_negs: List[Tuple[str, Path, Path]],
    rng: random.Random,
) -> Tuple[List[Dict], Dict[str, int]]:
    records: List[Dict] = []
    counts = {YES_LABEL: 0, NO_LABEL: 0}
    counter = 0

    for product_id, ref_path, eval_path in positives:
        record_id = f"pair_{counter:04d}"
        ref_entry = _make_image_entry(ref_path, product_id, counter * 2)
        eval_entry = _make_image_entry(eval_path, product_id, counter * 2 + 1)
        records.append(
            {
                "id": record_id,
                "question_type": "pairwise",
                "question": PAIR_QUESTION,
                "answer": YES_LABEL,
                "product_ids": [product_id, product_id],
                "images": [ref_entry, eval_entry],
            }
        )
        counts[YES_LABEL] += 1
        counter += 1

    for product_id, ref_path, eval_path in real_negs + gen_negs:
        record_id = f"pair_{counter:04d}"
        ref_entry = _make_image_entry(ref_path, product_id, counter * 2)
        eval_entry = _make_image_entry(eval_path, product_id + "#neg", counter * 2 + 1)
        records.append(
            {
                "id": record_id,
                "question_type": "pairwise",
                "question": PAIR_QUESTION,
                "answer": NO_LABEL,
                "product_ids": [product_id, eval_entry["product_id"]],
                "images": [ref_entry, eval_entry],
            }
        )
        counts[NO_LABEL] += 1
        counter += 1

    rng.shuffle(records)
    return records, counts


def build_multi_records(product_map: Dict[str, Dict], images_dir: Path, rng: random.Random) -> Tuple[List[Dict], Dict[str, int]]:
    records: List[Dict] = []
    distribution = {str(i): 0 for i in range(4)}
    global_negative_pool: List[Tuple[str, Path]] = []
    for product_id, info in product_map.items():
        for neg in info["real_negs"]:
            global_negative_pool.append((product_id + "#real", neg))
        for neg in info["gen_negs"]:
            global_negative_pool.append((product_id + "#gen", neg))
        # positives from other products are also valid negatives
        for pos in info["positives"]:
            global_negative_pool.append((product_id, pos))

    if len(global_negative_pool) < MULTI_SAMPLES_PER_CLASS * 3:
        raise RuntimeError("Insufficient negatives to build multi-choice examples.")

    product_ids = list(product_map.keys())

    def pick_negatives(exclude_product: str, needed: int) -> List[Tuple[str, Path]]:
        choices = [item for item in global_negative_pool if item[0] != exclude_product]
        if len(choices) < needed:
            raise RuntimeError("Not enough negative images available.")
        return rng.sample(choices, needed)

    counter = 0
    for count_value in range(4):
        candidates = [pid for pid, info in product_map.items() if len(info["positives"]) >= max(1, count_value)]
        if not candidates:
            raise RuntimeError(f"No products with enough positives for count {count_value}.")
        while distribution[str(count_value)] < MULTI_SAMPLES_PER_CLASS:
            product_id = rng.choice(candidates)
            info = product_map[product_id]
            ref_path = info["ref_path"]

            positives = []
            if count_value > 0:
                positives = rng.sample(list(info["positives"]), count_value)

            negatives_needed = 3 - count_value
            negatives = []
            if negatives_needed > 0:
                # prefer product-specific negatives first
                candidate_negatives = list(info["real_negs"]) + list(info["gen_negs"])
                rng.shuffle(candidate_negatives)
                while candidate_negatives and len(negatives) < negatives_needed:
                    neg_path = candidate_negatives.pop()
                    negatives.append((product_id + "#neg", neg_path))
                if len(negatives) < negatives_needed:
                    remaining = negatives_needed - len(negatives)
                    negatives.extend(pick_negatives(product_id, remaining))

            entries = []
            image_counter_base = counter * 4
            entries.append(_make_image_entry(ref_path, product_id, image_counter_base))
            current_id = image_counter_base + 1
            for pos_path in positives:
                entries.append(_make_image_entry(pos_path, product_id, current_id))
                current_id += 1
            for neg_owner, neg_path in negatives:
                entries.append(_make_image_entry(neg_path, neg_owner, current_id))
                current_id += 1

            record_id = f"multi_{counter:04d}"
            records.append(
                {
                    "id": record_id,
                    "question_type": "multiple_choice",
                    "question": MULTI_QUESTION,
                    "answer": str(count_value),
                    "target_count": count_value,
                    "product_ids": [entry["product_id"] for entry in entries],
                    "images": entries,
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

    json_path = output_dir / "rao_llm.jsonl"
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
                "product_ids": record.get("product_ids"),
                "instance_ids": [image_entry["image_id"] for image_entry in record["images"]],
                "source_paths": [image_entry["source_path"] for image_entry in record["images"]],
            }
            if record["question_type"] == "multiple_choice":
                doc["target_count"] = record["target_count"]

            handle.write(json.dumps(doc) + "\n")
            progress.update(1)

    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the rao_llm dataset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/rao-test/test"),
        help="Root directory containing Rao test data (with data/ and images/ subfolders).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lmms_eval/tasks/rao_llm/data"),
        help="Destination directory for images + jsonl.",
    )
    parser.add_argument("--seed", type=int, default=41, help="RNG seed.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    script_root = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[3]

    data_root = args.data_root
    if not data_root.is_absolute():
        candidate = script_root / data_root
        if candidate.exists():
            data_root = candidate
        else:
            data_root = repo_root / data_root
    images_dir = data_root / "images"
    data_dir = data_root / "data"

    if not data_dir.exists() or not images_dir.exists():
        raise FileNotFoundError("Expected data and images directories under the provided data root.")

    pos_json = _load_json(data_dir / "pos_pos.json")
    real_neg_json = _load_json(data_dir / "pos_real_neg.json")
    gen_neg_json = _load_json(data_dir / "pos_gen_neg.json")

    positives_all = [
        (key, images_dir / value["ref"], images_dir / value["eval"])
        for key, value in pos_json.items()
    ]
    real_neg_all = [
        (key, images_dir / value["ref"], images_dir / value["eval"])
        for key, value in real_neg_json.items()
    ]
    gen_neg_all = [
        (key, images_dir / value["ref"], images_dir / value["eval"])
        for key, value in gen_neg_json.items()
    ]

    positives = rng.sample(positives_all, PAIR_POSITIVE_SAMPLES)
    real_neg = rng.sample(real_neg_all, PAIR_NEGATIVE_REAL_SAMPLES)
    gen_neg = rng.sample(gen_neg_all, PAIR_NEGATIVE_GEN_SAMPLES)

    product_map: Dict[str, Dict] = {}
    for key, ref_path, eval_path in positives_all:
        product_id = ref_path.name
        entry = _ensure_product(product_map, product_id, ref_path)
        entry["positives"].add(eval_path)

    for key, ref_path, eval_path in real_neg_all:
        product_id = ref_path.name
        entry = _ensure_product(product_map, product_id, ref_path)
        entry["real_negs"].add(eval_path)

    for key, ref_path, eval_path in gen_neg_all:
        product_id = ref_path.name
        entry = _ensure_product(product_map, product_id, ref_path)
        entry["gen_negs"].add(eval_path)

    pair_records, pair_counts = build_pair_records(positives, real_neg, gen_neg, rng)
    multi_records, multi_distribution = build_multi_records(product_map, images_dir, rng)

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
