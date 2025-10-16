import re
from pathlib import Path
from typing import Dict, List

from PIL import Image

TASK_DIR = Path(__file__).resolve().parent
DATA_DIR = TASK_DIR / "data"

YES_SET = {"yes", "y", "yeah", "yep", "true", "1"}
NO_SET = {"no", "n", "nope", "false", "0"}


def _resolve_image_path(rel_path: str) -> Path:
    candidate = DATA_DIR / rel_path
    if candidate.exists():
        return candidate
    return Path(rel_path).expanduser().resolve()


def ilias_pairmatch_doc_to_visual(doc: Dict) -> List[Image.Image]:
    visuals: List[Image.Image] = []
    for rel_path in doc["image_paths"]:
        path = _resolve_image_path(rel_path)
        visuals.append(Image.open(path).convert("RGB"))
    return visuals


def ilias_pairmatch_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc.get("question", "Do these two images depict the same instance? Answer yes or no.")
    return f"{pre_prompt}{question}{post_prompt}"


def ilias_pairmatch_doc_to_target(doc: Dict) -> str:
    return doc["answer"]


def _normalize_yes_no(text: str) -> str:
    cleaned = text.strip().lower()
    if not cleaned:
        return ""
    match = re.findall(r"\b(yes|no)\b", cleaned)
    if match:
        return match[-1]
    token = re.sub(r"[^a-z0-9]+", "", cleaned)
    if token in YES_SET:
        return "yes"
    if token in NO_SET:
        return "no"
    parts = cleaned.split()
    return parts[0] if parts else ""


def _extract_tag_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    inner = match.group(1).strip()
    return _normalize_yes_no(inner)


def ilias_pairmatch_process_results(doc: Dict, results) -> Dict:
    prediction = results[0] if results else ""
    normalized = _extract_tag_answer(prediction)
    if not normalized:
        no_tags = re.sub(r"</?answer>", "", prediction, flags=re.IGNORECASE)
        normalized = _normalize_yes_no(no_tags)
    score = 1.0 if normalized == doc["answer"] else 0.0
    return {
        "accuracy": score,
        "model_answer": normalized or prediction.strip().lower(),
        "target": doc["answer"],
        "same_instance": doc.get("same_instance"),
        "instance_ids": doc.get("instance_ids"),
        "source_keys": doc.get("source_keys"),
    }


def ilias_pairmatch_aggregate_accuracy(results: List[float]) -> float:
    return float(sum(results) / len(results)) if results else 0.0
