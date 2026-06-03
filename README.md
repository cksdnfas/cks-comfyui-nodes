# CKS ComfyUI Nodes

Custom ComfyUI nodes for deterministic prompt utilities, artist-style prompt blending, and workflow-aware image saving.

Nodes are grouped in ComfyUI under `CKS ComfyUI Nodes`.

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cksdnfas/cks-comfyui-nodes.git cks_comfyui_nodes
```

Install Python dependencies:

```bash
pip install -r cks_comfyui_nodes/requirements.txt
```

Restart ComfyUI after installation or updates.

## Nodes

### Seed Random 0-100

Deterministically maps a seed to a random value from `0` to `100`.

Inputs:

- `seed`

Outputs:

- `value_int`
- `value_float`

Use this node when a workflow needs a stable random scalar derived from a seed.

### Artist Prompt Weighted Blend

Builds CLIP conditioning from top prompt text, weighted artist tags, and bottom prompt text.

Inputs:

- `clip`
- `top_prompt`
- `bottom_prompt`
- `artists`
- `blend_mode`
- `separator`

Example artist list:

```text
(@koku:1.4), (@mataro \(matarou\):1.5), (@starraisins:0.6), (@nanashi \(nlo\):0.6)
```

Blend modes:

- `average`: encode each artist separately, then return one weighted-average conditioning.
- `exact`: return one conditioning entry per artist with normalized `strength`.
- `prompt`: encode one weighted prompt, closest to manually typing every weighted tag.

Outputs:

- `conditioning`
- `summary`
- `composite_prompt`

If `artists` is empty, the node encodes only `top_prompt + bottom_prompt`.

### Artist Style Delta Blend

Extracts an artist-style delta from a styled prompt and adds it back onto a base prompt.

Inputs:

- `clip`
- `top_prompt`
- `bottom_prompt`
- `artists`
- `blend_mode`
- `separator`
- `style_strength`

Behavior:

- Builds a base prompt without artist text.
- Builds styled artist conditioning with the selected blend mode.
- Applies `styled_conditioning - base_conditioning` at `style_strength`.

Use this node when the base prompt should stay stable while the artist style is blended in.

Outputs:

- `conditioning`
- `summary`
- `composite_prompt`

### Save Image w/Workflow Name

Saves images with generation metadata and a workflow name.

Inputs include:

- `images`
- `workflow_name`
- `filename`
- `path`
- `extension`
- `steps`
- `cfg`
- `modelname`
- `sampler_name`
- `scheduler`
- optional prompt, seed, size, quality, compression, and format settings

Metadata written:

- PNG text key `workflow_name`
- PNG `parameters` text containing `Workflow: ...`
- JPEG/WebP EXIF user comment containing `Workflow: ...`
- ComfyUI hidden `prompt` and `extra_pnginfo` metadata for PNG output

The node uses ComfyUI's safe output path handling and counter-based filenames to avoid saving outside the output directory or overwriting files with the same prefix.

## Requirements

- ComfyUI
- Pillow
- NumPy
- PyTorch
- piexif

ComfyUI normally provides Pillow, NumPy, and PyTorch. `piexif` is listed in `requirements.txt`.

## Development Notes

- Python node entrypoint: `__init__.py`
- Browser extension directory: `js`
- The current JavaScript extension is intentionally empty so stale browser caches stop rebuilding old dynamic rows.

Basic validation:

```bash
python -m py_compile __init__.py
node --check js/artist_prompt_blend.js
```

## License

MIT
