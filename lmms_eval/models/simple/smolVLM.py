from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

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


@register_model("smolVLM")
class SmolVLM(lmms):
    def __init__(
        self,
        pretrained: str,
        device: str = "cuda",
        max_new_tokens: int = 256,
        dtype: Optional[Union[str, torch.dtype]] = None,
        batch_size: int = 1,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            unexpected = ", ".join(kwargs.keys())
            raise ValueError(f"Unexpected kwargs received: {unexpected}")

        self.pretrained = pretrained
        self._device = torch.device(device)
        self.batch_size_per_gpu = int(batch_size)
        if self.batch_size_per_gpu != 1:
            raise ValueError("SmolVLM currently supports batch_size == 1 only.")

        dtype_override = resolve_torch_dtype(dtype)
        if dtype_override is None:
            dtype_override = torch.float16 if self._device.type == "cuda" else torch.float32
        self._dtype = dtype_override

        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt.strip() if system_prompt else None

        processor_kwargs: Dict[str, Any] = {}
        processor_kwargs["trust_remote_code"] = True
        self.processor = AutoProcessor.from_pretrained(pretrained, **processor_kwargs)

        attn_impl = "flash_attention_2" if self._device.type == "cuda" else None
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype_override,
            "trust_remote_code": True,
        }
        if attn_impl is not None:
            model_kwargs["_attn_implementation"] = attn_impl

        self._model = AutoModelForImageTextToText.from_pretrained(pretrained, **model_kwargs)
        self._model.to(device=self._device, dtype=dtype_override)
        self._model.eval()

        tokenizer = getattr(self.processor, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            eos_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
            pad_token_id = eos_id if eos_id is not None else self._model.generation_config.eos_token_id
            if tokenizer is not None and hasattr(tokenizer, "pad_token_id"):
                tokenizer.pad_token_id = pad_token_id
        self._model.generation_config.pad_token_id = pad_token_id
        self._pad_token_id = pad_token_id

    @property
    def model(self) -> AutoModelForImageTextToText:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def batch_size(self) -> int:
        return self.batch_size_per_gpu

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

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
                    images.append(visual.convert("RGB"))
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
                raise ValueError("SmolVLM forward requires at least one image input")

            gen_kwargs = gen_kwargs or {}
            max_new_tokens = int(gen_kwargs.get("max_new_tokens", self.max_new_tokens))
            temperature = float(gen_kwargs.get("temperature", 0.0))
            do_sample = temperature > 0.0

            messages: List[Dict[str, Any]] = []
            if self.system_prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": self.system_prompt}],
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        *({"type": "image", "image": image} for image in images),
                        {"type": "text", "text": contexts},
                    ],
                }
            )

            chat = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            inputs = self.processor(
                text=[chat],
                images=list(images),
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            generation_kwargs: Dict[str, Any] = {
                "do_sample": do_sample,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self._pad_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature

            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, **generation_kwargs)

            prompt_length = inputs["input_ids"].shape[-1]
            trimmed = generated_ids[:, prompt_length:]

            output = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            text_output = output.strip()

            responses.append(text_output)
            self.cache_hook.add_partial("generate_until", (contexts, gen_kwargs), text_output)
            pbar.update(1)

        pbar.close()
        return responses
