#!/usr/bin/env python
"""Preview utility for the cub_birds task."""

import argparse
import json
import random
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont


def load_records(jsonl_path: Path) -> List[dict]:
    records = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _load_font(size: int = 18) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def create_preview_image(record: dict, data_dir: Path, save_dir: Path) -> Path:
    question = record["question"]
    answer = record["answer"]
    sample_id = record.get("sample_id", record.get("pair_id"))
    question_type = record.get("question_type")
    species = record.get("species")

    images = []
    for rel_path in record["image_paths"]:
        img_path = (data_dir / rel_path).resolve()
        images.append(Image.open(img_path).convert("RGB"))

    margin = 20
    text_height = 140
    panel_width = sum(img.width for img in images) + margin * (len(images) + 1)
    panel_height = max(img.height for img in images) + text_height + margin * 2
    canvas = Image.new("RGB", (panel_width, panel_height), color=(245, 245, 245))

    draw = ImageDraw.Draw(canvas)
    font_label = _load_font(28)
    font_meta = _load_font(18)

    x_offset = margin
    for idx, img in enumerate(images):
        canvas.paste(img, (x_offset, margin))
        draw.rectangle([x_offset, margin, x_offset + img.width - 1, margin + img.height - 1], outline=(0, 0, 0), width=2)
        label = "Ref" if (question_type == "multiple_choice" and idx == 0) else chr(ord("A") + idx)
        draw.text((x_offset + 10, margin + 10), label, fill=(255, 0, 0), font=font_label)
        x_offset += img.width + margin

    text_y = max(img.height for img in images) + margin + 10
    meta_lines = [
        f"sample_id={sample_id} | type={question_type} | answer={answer}",
        f"species={species}",
        f"question: {question}",
    ]
    for line in meta_lines:
        draw.text((margin, text_y), line, fill=(20, 20, 20), font=font_meta)
        text_y += font_meta.getbbox(line)[3] - font_meta.getbbox(line)[1] + 4

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{sample_id}.jpg"
    canvas.save(out_path, quality=95)
    return out_path


def sample_records(records: List[dict], k: int, seed: int, shuffle: bool) -> List[dict]:
    rng = random.Random(seed)
    if shuffle:
        records = records.copy()
        rng.shuffle(records)
    return records[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview items from cub_birds.")
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=Path("lmms_eval/tasks/cub_birds/data/cub_birds.jsonl"),
        help="Path to cub_birds.jsonl.",
    )
    parser.add_argument("--num-samples", type=int, default=5, help="How many samples to preview.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed used with --shuffle.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before selecting samples.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("preview_cub_birds"),
        help="Directory where preview composites will be written.",
    )
    args = parser.parse_args()

    jsonl_path = args.dataset_json
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Could not find {jsonl_path}. Run the dataset builder first.")

    data_dir = jsonl_path.parent
    records = load_records(jsonl_path)
    subset = sample_records(records, args.num_samples, args.seed, args.shuffle)

    for idx, record in enumerate(subset, 1):
        print(
            f"[{idx}/{len(subset)}] sample_id={record.get('sample_id')} "
            f"type={record.get('question_type')} answer={record['answer']} species={record.get('species')}"
        )
        out_path = create_preview_image(record, data_dir, args.save_dir)
        print(f"    saved preview -> {out_path}")


if __name__ == "__main__":
    main()
