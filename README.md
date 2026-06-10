# ComfyUI PixAI Tagger

ComfyUI custom nodes for `imgutils.tagging.get_pixai_tags`.

## Nodes

- `PixAI Tagger`: tags a connected ComfyUI `IMAGE`.
- `PixAI Tagger From Path/URL`: tags a local file path or URL directly.

Both nodes return:

- `tags`: character tags and general tags as one prompt-friendly string.
- `general_tags`: general tag string.
- `character_tags`: character tag string.
- `ips`: detected IP names such as `genshin_impact`.
- `general_json`: general tags with confidence scores.
- `character_json`: character tags with confidence scores.
- `ips_mapping_json`: detected characters mapped to IP names.

## Install

Installing through ComfyUI-Manager handles dependencies automatically (it runs
`install.py`). For a manual install, run this in the same Python environment
used by ComfyUI:

```powershell
python install.py
```

`dghs-imgutils` pins `numpy<2` in its package metadata, which would downgrade
ComfyUI's numpy — so `install.py` installs it with `--no-deps`, while
`requirements.txt` lists the dependencies its PixAI code path actually needs.
Equivalent manual commands:

```powershell
pip install -r requirements.txt
pip install --no-deps "dghs-imgutils>=0.19.0"
```

The PixAI ONNX model is downloaded from Hugging Face on first use.

## Usage Notes

Keep `threshold_mode` set to `model default` to use the thresholds provided by `dghs-imgutils`. Switch it to `custom` to use `general_threshold` and `character_threshold`.

Set `max_general_tags` or `max_character_tags` to `0` to keep every tag returned by the model.
