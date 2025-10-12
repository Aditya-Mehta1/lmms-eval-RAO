from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

ImageSequence = Sequence[Image.Image]
_DTYPE_ALIASES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "single": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
    "double": torch.float64,
}


def resolve_torch_dtype(
    value: Optional[Union[str, torch.dtype]]
) -> Optional[torch.dtype]:
    """Normalize CLI-provided dtype values into torch.dtype objects."""
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value

    normalized = value.strip().lower()
    if not normalized:
        return None

    if normalized in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[normalized]

    raise ValueError(f"Unsupported dtype value: {value!r}")


@register_model("moondream")
class Moondream(lmms):
    def __init__(
        self,
        pretrained: str = "vikhyatk/moondream2",
        device: str = "cuda",
        max_new_tokens: int = 256,
        dtype: Optional[Union[str, torch.dtype]] = None,
        batch_size: int = 1,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        
        self.pretrained = pretrained
        self._device = torch.device(device)
        self.batch_size_per_gpu = int(batch_size)
        if self.batch_size_per_gpu != 1:
            raise ValueError("Moondream currently supports batch_size == 1 only.")
        dtype_override = resolve_torch_dtype(dtype)
        if dtype_override is None:
            dtype_override = torch.float16 if self._device.type == "cuda" else torch.float32
        self._dtype = dtype_override
        self.max_new_tokens = max_new_tokens
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype_override,
            "trust_remote_code": trust_remote_code,
        }
        if self._device.type != "cpu":
            model_kwargs["device_map"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(pretrained, **model_kwargs)
        if self._device.type == "cpu":
            self._model.to(device=self._device, dtype=dtype_override)
        self._model.eval()
        
    @property
    def model(self) -> AutoModelForCausalLM:
        return self._model
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def batch_size(self) -> int:
        return self.batch_size_per_gpu
    
    @property
    def rank(self) -> int:
        return self._rank
    
    @property
    def world_size(self) -> int:
        return self._world_size
    
    def loglikelihood(self, requests: List[Instance]) -> List[Any]:
        raise NotImplementedError("loglikelihood.")
    
    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("generate_until_multi_round.")
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        responses: List[str] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            doc = self.task_dict[task][split][doc_id]
            visuals = doc_to_visual(doc)

            images: List[Image.Image] = []

            def _collect(visual: Any) -> None:
                if visual is None:
                    return
                if isinstance(visual, Image.Image):
                    images.append(visual)
                elif isinstance(visual, str):
                    if not os.path.exists(visual):
                        raise FileNotFoundError(f"Image path not found: {visual}")
                    with Image.open(visual) as img:
                        images.append(img.convert("RGB"))
                elif isinstance(visual, (list, tuple)):
                    for item in visual:
                        _collect(item)
                else:
                    raise TypeError(f"Unsupported visual type: {type(visual)}")

            _collect(visuals)

            if not images:
                raise ValueError("Moondream forward requires at least one image input")

            payload_list = [img.convert("RGB") for img in images]
            if len(payload_list) == 1:
                payload = payload_list[0]
            else:
                widths = [img.width for img in payload_list]
                heights = [img.height for img in payload_list]
                total_width = sum(widths)
                max_height = max(heights)
                stitched = Image.new("RGB", (total_width, max_height), color=(0, 0, 0))
                offset = 0
                for img in payload_list:
                    stitched.paste(img, (offset, 0))
                    offset += img.width
                payload = stitched

            gen_kwargs = gen_kwargs or {}
            settings = {
                "max_tokens": gen_kwargs.get("max_new_tokens", self.max_new_tokens),
                "temperature": gen_kwargs.get("temperature", 0.0),
            }

            with torch.inference_mode():
                result = self.model.query(payload, contexts, settings=settings)

            if isinstance(result, dict):
                text_output: Optional[str] = None
                for key in ("answer", "caption", "text"):
                    value = result.get(key)
                    if isinstance(value, str):
                        text_output = value.strip()
                        break
                if text_output is None:
                    text_output = str(result).strip()
            else:
                text_output = str(result).strip()

            responses.append(text_output)
            self.cache_hook.add_partial("generate_until", (contexts, gen_kwargs), text_output)
            pbar.update(1)

        pbar.close()
        return responses
