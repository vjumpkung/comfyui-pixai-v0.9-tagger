# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A ComfyUI custom node pack that wraps `imgutils.tagging.get_pixai_tags` (from `dghs-imgutils`) to tag anime-style images with PixAI. It lives inside a ComfyUI installation at `custom_nodes/comfyui-pixai-tagger` and is loaded by ComfyUI at startup.

## Commands

- Install dependencies into the ComfyUI Python environment: `python install.py` (ComfyUI-Manager runs it automatically). It installs `requirements.txt`, then `dghs-imgutils>=0.19.0` with `--no-deps` — `dghs-imgutils` pins `numpy<2` in its metadata, so it must be installed without its declared dependencies; `requirements.txt` instead lists the packages the `get_pixai_tags` code path actually imports (with `numpy<2.3`).
- There are no tests, linters, or build steps. Verifying changes requires restarting ComfyUI so it reloads the node pack, then running the node in a workflow. The PixAI ONNX model downloads from Hugging Face on first use.

## Architecture

Two source files:

- `__init__.py` — ComfyUI's entry point. Exports `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`; a new node class must be registered in both.
- `pixai_tagger.py` — everything else.

Structure of `pixai_tagger.py`:

- `_PixAITaggerInputs` is the shared base class. It defines the 7 STRING return outputs (`RETURN_TYPES`/`RETURN_NAMES`), `FUNCTION = "tag"`, `CATEGORY`, and `_settings_inputs()` — the dict of settings widgets (model name, thresholds, tag limits, formatting flags) shared by both nodes.
- Two concrete nodes differ only in image source: `PixAITagger` takes a ComfyUI `IMAGE` tensor plus `batch_index`; `PixAITaggerFromPath` takes a local path or URL string. Both funnel into the module-level `_run_pixai_tagger`, which calls `get_pixai_tags` and formats all 7 outputs.
- Adding a shared setting means touching four places: `_settings_inputs()`, both nodes' `tag()` signatures, and `_run_pixai_tagger`'s parameters.

Key constraints:

- The `dghs-imgutils` import is deferred inside `_load_get_pixai_tags()` so ComfyUI can load the node pack (and show a helpful error at runtime) even when dependencies are missing. Keep any new `imgutils` imports lazy in the same way; only `numpy` and `PIL` are imported at module level.
- `_image_tensor_to_pil` converts ComfyUI's `[B,H,W,C]` float-0..1 tensors to PIL RGB, handling grayscale (channel repeat) and RGBA (alpha-composite onto white). `get_pixai_tags` itself accepts either a PIL image or a path/URL string, which is why both nodes can share `_run_pixai_tagger`.
- `threshold_mode = "model default"` passes `thresholds=None` to `get_pixai_tags` so the library's per-model defaults apply; the `general_threshold`/`character_threshold` widgets are only used in `custom` mode. A tag limit of `0` means unlimited.
- `tag_separator` supports escaped `\n` and `\t`, decoded by `_decode_separator` before joining outputs.
