from __future__ import annotations

import json
from typing import Any

import numpy as np
from PIL import Image


MODEL_DEFAULT = "v0.9"
THRESHOLD_MODES = ["model default", "custom"]


def _load_get_pixai_tags():
    try:
        from imgutils.tagging import get_pixai_tags
    except ImportError as exc:
        raise ImportError(
            "PixAI Tagger requires dghs-imgutils and its runtime dependencies. "
            "Install them in the ComfyUI Python environment with: "
            "pip install \"dghs-imgutils>=0.19.0\" torch huggingface_hub timm pillow pandas"
        ) from exc

    return get_pixai_tags


def _image_tensor_to_pil(image: Any, batch_index: int) -> Image.Image:
    if not hasattr(image, "detach"):
        raise TypeError("Expected a ComfyUI IMAGE tensor.")

    image = image.detach().cpu()
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    if len(image.shape) != 4:
        raise ValueError(f"Expected IMAGE shape [B,H,W,C], got {tuple(image.shape)}.")

    batch_size = int(image.shape[0])
    if batch_index < 0 or batch_index >= batch_size:
        raise ValueError(f"batch_index {batch_index} is out of range for batch size {batch_size}.")

    pixels = image[batch_index].numpy()
    pixels = np.clip(pixels, 0.0, 1.0)

    if pixels.shape[-1] == 1:
        pixels = np.repeat(pixels, 3, axis=-1)
    elif pixels.shape[-1] > 4:
        pixels = pixels[..., :3]

    array = (pixels * 255.0).round().astype(np.uint8)

    if array.shape[-1] == 4:
        rgba = Image.fromarray(array, mode="RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")

    if array.shape[-1] != 3:
        raise ValueError(f"Expected IMAGE channel count 1, 3, or 4, got {array.shape[-1]}.")

    return Image.fromarray(array, mode="RGB")


def _thresholds(
    threshold_mode: str,
    general_threshold: float,
    character_threshold: float,
) -> dict[str, float] | None:
    if threshold_mode == "model default":
        return None

    return {
        "general": float(general_threshold),
        "character": float(character_threshold),
    }


def _limit_tags(tags: dict[str, float], limit: int) -> dict[str, float]:
    if limit <= 0:
        return tags

    return dict(list(tags.items())[:limit])


def _decode_separator(separator: str) -> str:
    return separator.replace("\\n", "\n").replace("\\t", "\t")


def _format_tags(
    tags: dict[str, float],
    separator: str,
    replace_underscores: bool,
    include_scores: bool,
) -> str:
    output = []
    for tag, score in tags.items():
        display_tag = tag.replace("_", " ") if replace_underscores else tag
        if include_scores:
            output.append(f"{display_tag}:{score:.3f}")
        else:
            output.append(display_tag)

    return _decode_separator(separator).join(output)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _run_pixai_tagger(
    image: Any,
    model_name: str,
    threshold_mode: str,
    general_threshold: float,
    character_threshold: float,
    max_general_tags: int,
    max_character_tags: int,
    tag_separator: str,
    replace_underscores: bool,
    include_scores: bool,
    character_first: bool,
) -> tuple[str, str, str, str, str, str, str]:
    get_pixai_tags = _load_get_pixai_tags()

    general, character, ips, ips_mapping = get_pixai_tags(
        image,
        model_name=model_name.strip() or MODEL_DEFAULT,
        thresholds=_thresholds(threshold_mode, general_threshold, character_threshold),
        fmt=("general", "character", "ips", "ips_mapping"),
    )

    general = _limit_tags(dict(general), int(max_general_tags))
    character = _limit_tags(dict(character), int(max_character_tags))

    general_tags = _format_tags(general, tag_separator, replace_underscores, include_scores)
    character_tags = _format_tags(character, tag_separator, replace_underscores, include_scores)

    combined_parts = [character_tags, general_tags] if character_first else [general_tags, character_tags]
    tags = _decode_separator(tag_separator).join(part for part in combined_parts if part)
    ips_text = _decode_separator(tag_separator).join(ips)

    return (
        tags,
        general_tags,
        character_tags,
        ips_text,
        _json_dumps(general),
        _json_dumps(character),
        _json_dumps(ips_mapping),
    )


class _PixAITaggerInputs:
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "tags",
        "general_tags",
        "character_tags",
        "ips",
        "general_json",
        "character_json",
        "ips_mapping_json",
    )
    FUNCTION = "tag"
    CATEGORY = "image/tagging"

    @classmethod
    def _settings_inputs(cls) -> dict[str, tuple[Any, dict[str, Any]]]:
        return {
            "model_name": (
                "STRING",
                {
                    "default": MODEL_DEFAULT,
                    "tooltip": "PixAI model name or Hugging Face repository ID.",
                },
            ),
            "threshold_mode": (THRESHOLD_MODES, {"default": "model default"}),
            "general_threshold": (
                "FLOAT",
                {
                    "default": 0.35,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Used only when threshold_mode is custom.",
                },
            ),
            "character_threshold": (
                "FLOAT",
                {
                    "default": 0.50,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Used only when threshold_mode is custom.",
                },
            ),
            "max_general_tags": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "tooltip": "0 keeps every general tag returned by the model.",
                },
            ),
            "max_character_tags": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 1024,
                    "tooltip": "0 keeps every character tag returned by the model.",
                },
            ),
            "tag_separator": (
                "STRING",
                {
                    "default": ", ",
                    "tooltip": "Separator for string outputs. Supports escaped \\n and \\t.",
                },
            ),
            "replace_underscores": ("BOOLEAN", {"default": False}),
            "include_scores": ("BOOLEAN", {"default": False}),
            "character_first": ("BOOLEAN", {"default": True}),
        }


class PixAITagger(_PixAITaggerInputs):
    DESCRIPTION = "Tags a ComfyUI IMAGE with PixAI and returns prompt-friendly strings plus JSON scores."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "batch_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "tooltip": "Image index to tag when the input is a batch.",
                    },
                ),
                **cls._settings_inputs(),
            }
        }

    def tag(
        self,
        image,
        batch_index,
        model_name,
        threshold_mode,
        general_threshold,
        character_threshold,
        max_general_tags,
        max_character_tags,
        tag_separator,
        replace_underscores,
        include_scores,
        character_first,
    ):
        pil_image = _image_tensor_to_pil(image, int(batch_index))
        return _run_pixai_tagger(
            pil_image,
            model_name,
            threshold_mode,
            general_threshold,
            character_threshold,
            max_general_tags,
            max_character_tags,
            tag_separator,
            replace_underscores,
            include_scores,
            character_first,
        )


class PixAITaggerFromPath(_PixAITaggerInputs):
    DESCRIPTION = "Tags an image from a local path or URL with PixAI."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path_or_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "D:/images/sample.webp or https://...",
                    },
                ),
                **cls._settings_inputs(),
            }
        }

    def tag(
        self,
        image_path_or_url,
        model_name,
        threshold_mode,
        general_threshold,
        character_threshold,
        max_general_tags,
        max_character_tags,
        tag_separator,
        replace_underscores,
        include_scores,
        character_first,
    ):
        image_path_or_url = image_path_or_url.strip()
        if not image_path_or_url:
            raise ValueError("image_path_or_url is required.")

        return _run_pixai_tagger(
            image_path_or_url,
            model_name,
            threshold_mode,
            general_threshold,
            character_threshold,
            max_general_tags,
            max_character_tags,
            tag_separator,
            replace_underscores,
            include_scores,
            character_first,
        )
