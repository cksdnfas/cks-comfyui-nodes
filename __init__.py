import hashlib
import json
import math
import os
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import folder_paths
import numpy as np
import piexif
import piexif.helper
import torch
from nodes import MAX_RESOLUTION
from PIL import Image
from PIL.PngImagePlugin import PngInfo


MAX_SEED = 0xFFFFFFFFFFFFFFFF
WEIGHTED_TAG_RE = re.compile(r"^\(\s*(?P<tag>.*?)\s*:\s*(?P<weight>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)$")
PROMPT_CATEGORY = "CKS ComfyUI Nodes/Prompt"
IMAGE_CATEGORY = "CKS ComfyUI Nodes/Image"
ARTIFACT_CATEGORY = "CKS ComfyUI Nodes/Artifacts"
SAFE_ARTIFACT_NAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1F]')


@dataclass(frozen=True)
class ArtistEntry:
    tag: str
    raw_tag: str
    weight: float


def _string_input(default="", multiline=True, tooltip=None):
    data = {"default": default, "multiline": multiline}
    if tooltip:
        data["tooltip"] = tooltip
    return ("STRING", data)


def _string_socket(tooltip=None):
    data = {"forceInput": True, "multiline": True}
    if tooltip:
        data["tooltip"] = tooltip
    return ("STRING", data)


def _safe_artifact_file_name(value, fallback="artifact.bin"):
    name = SAFE_ARTIFACT_NAME_RE.sub("_", os.path.basename(str(value or "").strip()))
    return name or fallback


def _safe_artifact_subfolder(value):
    normalized = str(value or "").strip().replace("\\", "/").strip("/")
    parts = []
    for part in normalized.split("/"):
        if not part or part in {".", ".."}:
            continue
        parts.append(SAFE_ARTIFACT_NAME_RE.sub("_", part))
    return "/".join(parts)


def _available_artifact_path(directory, file_name, overwrite):
    target = directory / file_name
    if overwrite or not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    counter = 2
    while True:
        candidate = directory / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _available_artifact_directory(target, overwrite):
    if overwrite:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        return target

    if not target.exists():
        return target

    counter = 2
    while True:
        candidate = target.parent / f"{target.name}-{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def _iter_artifact_files(root):
    for child in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if child.is_dir():
            yield from _iter_artifact_files(child)
        elif child.is_file():
            yield child


def _is_within_directory(root, candidate):
    return os.path.commonpath([str(root), str(candidate)]) == str(root)


def _artifact_timestamp_prefix():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]


def _history_file_entries(output_root, root):
    entries = []
    for file_path in _iter_artifact_files(root):
        entries.append(
            {
                "filename": file_path.name,
                "subfolder": file_path.parent.relative_to(output_root).as_posix(),
                "type": "output",
            }
        )
    return entries


def _parse_model_name(model_name):
    filename = str(model_name).replace("\\", "/").split("/")[-1]
    parts = filename.split(".")[:-1]
    return ".".join(parts) if parts else filename


def _find_checkpoint_path(model_name):
    model_name = str(model_name or "").strip()
    if not model_name:
        return None

    direct_path = folder_paths.get_full_path("checkpoints", model_name)
    if direct_path:
        return direct_path

    target_name = os.path.basename(model_name.replace("\\", "/")).lower()
    target_stem = os.path.splitext(target_name)[0]
    for checkpoint_name in folder_paths.get_filename_list("checkpoints"):
        checkpoint_base = os.path.basename(str(checkpoint_name).replace("\\", "/")).lower()
        checkpoint_stem = os.path.splitext(checkpoint_base)[0]
        if target_name in {checkpoint_base, checkpoint_stem} or target_stem == checkpoint_stem:
            return folder_paths.get_full_path("checkpoints", checkpoint_name)

    return None


def _calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as file:
        for byte_block in iter(lambda: file.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _clean_metadata_text(value):
    return str(value).strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")


def _get_timestamp(time_format):
    now = datetime.now()
    try:
        return now.strftime(time_format)
    except Exception:
        return now.strftime("%Y-%m-%d-%H%M%S")


def _make_pathname(filename, seed, model_name, counter, time_format):
    filename = str(filename)
    filename = filename.replace("%date", _get_timestamp("%Y-%m-%d"))
    filename = filename.replace("%time", _get_timestamp(time_format))
    filename = filename.replace("%model", _parse_model_name(model_name))
    filename = filename.replace("%seed", str(seed))
    filename = filename.replace("%counter", str(counter))
    return filename


def _make_filename(filename, seed, model_name, counter, time_format):
    filename = _make_pathname(filename, seed, model_name, counter, time_format)
    return _get_timestamp(time_format) if filename == "" else filename


def _make_filename_prefix(path, filename):
    path = os.path.normpath(str(path or "").strip())
    filename = str(filename or "").strip()
    if path in {"", "."}:
        return filename
    return os.path.join(path, filename)


def _join_prompt(top_prompt, artist_prompt, bottom_prompt, separator):
    parts = [top_prompt.strip(), artist_prompt.strip(), bottom_prompt.strip()]
    return separator.join(part for part in parts if part)


def _split_artist_tags(artists):
    tags = []
    current = []
    escaped = False
    depth = 0

    for char in str(artists or ""):
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char == "(":
            depth += 1
            current.append(char)
            continue
        if char == ")" and depth > 0:
            depth -= 1
            current.append(char)
            continue
        if char == "," and depth == 0:
            tag = "".join(current).strip()
            if tag:
                tags.append(tag)
            current = []
            continue
        current.append(char)

    tag = "".join(current).strip()
    if tag:
        tags.append(tag)
    return tags


def _normalize_artist_prompt(artists, separator):
    return separator.join(_split_artist_tags(artists))


def _parse_artist_entries(artists):
    entries = []

    for raw_tag in _split_artist_tags(artists):
        match = WEIGHTED_TAG_RE.match(raw_tag.strip())
        if match:
            tag = match.group("tag").strip()
            try:
                weight = float(match.group("weight"))
            except ValueError:
                continue
        elif raw_tag.startswith("(") and raw_tag.endswith(")"):
            tag = raw_tag[1:-1].strip()
            weight = 1.0
        else:
            tag = raw_tag.strip()
            weight = 1.0

        if tag and math.isfinite(weight) and weight > 0:
            entries.append(ArtistEntry(tag=tag, raw_tag=raw_tag.strip(), weight=weight))

    return entries


def _format_weighted_artist_tag(artist):
    return artist.tag if math.isclose(artist.weight, 1.0) else f"({artist.tag}:{artist.weight:g})"


def _join_artist_entries(artists, separator, weighted=False):
    if weighted:
        return separator.join(_format_weighted_artist_tag(artist) for artist in artists)
    return separator.join(artist.tag for artist in artists)


def _encode_prompt(clip, text):
    if clip is None:
        raise RuntimeError("CLIP input is invalid: None")
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)


def _pad_to_length(tensor, target_length):
    if tensor.shape[1] >= target_length:
        return tensor[:, :target_length]

    pad = torch.zeros(
        (tensor.shape[0], target_length - tensor.shape[1], tensor.shape[2]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat((tensor, pad), dim=1)


def _copy_conditioning(conditioning):
    return [[item[0], item[1].copy()] for item in conditioning]


def _blend_artist_conditionings(clip, top_prompt, bottom_prompt, artist_entries, separator, blend_mode):
    if clip is None:
        raise RuntimeError("CLIP input is invalid: None")

    base_prompt = _join_prompt(top_prompt, "", bottom_prompt, separator)
    if not artist_entries:
        return _encode_prompt(clip, base_prompt), base_prompt

    weighted_artist_prompt = _join_artist_entries(artist_entries, separator, weighted=True)
    composite_prompt = _join_prompt(top_prompt, weighted_artist_prompt, bottom_prompt, separator)

    if blend_mode == "prompt":
        return _encode_prompt(clip, composite_prompt), composite_prompt

    total_weight = sum(artist.weight for artist in artist_entries)

    if blend_mode == "exact":
        encoded = []
        for artist in artist_entries:
            prompt = _join_prompt(top_prompt, artist.tag, bottom_prompt, separator)
            strength = artist.weight / total_weight
            for cond_tensor, meta in _encode_prompt(clip, prompt):
                output_meta = meta.copy()
                output_meta["strength"] = strength
                encoded.append([cond_tensor, output_meta])
        return encoded, composite_prompt

    encoded_artists = []
    for artist in artist_entries:
        prompt = _join_prompt(top_prompt, artist.tag, bottom_prompt, separator)
        conditioning = _encode_prompt(clip, prompt)
        if len(conditioning) != 1:
            return _blend_artist_conditionings(clip, top_prompt, bottom_prompt, artist_entries, separator, "exact")
        encoded_artists.append((artist, conditioning[0]))

    composite_conditioning = _encode_prompt(clip, composite_prompt)
    if len(composite_conditioning) != 1:
        return _blend_artist_conditionings(clip, top_prompt, bottom_prompt, artist_entries, separator, "exact")

    max_length = max(cond_tensor.shape[1] for _, (cond_tensor, _) in encoded_artists)
    mixed_cond = None
    mixed_pooled = None

    for artist, (cond_tensor, meta) in encoded_artists:
        normalized_weight = artist.weight / total_weight
        weighted_cond = _pad_to_length(cond_tensor, max_length) * normalized_weight
        mixed_cond = weighted_cond if mixed_cond is None else mixed_cond + weighted_cond

        pooled_output = meta.get("pooled_output")
        if pooled_output is not None:
            weighted_pooled = pooled_output * normalized_weight
            mixed_pooled = weighted_pooled if mixed_pooled is None else mixed_pooled + weighted_pooled

    _, composite_meta = composite_conditioning[0]
    output_meta = composite_meta.copy()
    output_meta.pop("strength", None)
    if mixed_pooled is not None:
        output_meta["pooled_output"] = mixed_pooled
    else:
        output_meta.pop("pooled_output", None)

    return [[mixed_cond, output_meta]], composite_prompt


def _artist_style_delta_conditioning(base_conditioning, styled_conditioning, style_strength):
    strength = float(style_strength)
    if math.isclose(strength, 0.0):
        return _copy_conditioning(base_conditioning)
    if math.isclose(strength, 1.0):
        return _copy_conditioning(styled_conditioning)

    output = []

    for cond_index, styled_item in enumerate(styled_conditioning):
        base_item = base_conditioning[min(cond_index, len(base_conditioning) - 1)]
        target_length = max(
            base_item[0].shape[1],
            styled_item[0].shape[1],
        )
        base_tensor = _pad_to_length(base_item[0], target_length)
        styled_tensor = _pad_to_length(styled_item[0], target_length).to(
            dtype=base_tensor.dtype,
            device=base_tensor.device,
        )
        target_metadata = (base_item[1] if math.isclose(strength, 0.0) else styled_item[1]).copy()
        blended = base_tensor + (styled_tensor - base_tensor) * strength

        base_pooled = base_item[1].get("pooled_output", None)
        styled_pooled = styled_item[1].get("pooled_output", None)
        if base_pooled is not None and styled_pooled is not None:
            styled_pooled = styled_pooled.to(dtype=base_pooled.dtype, device=base_pooled.device)
            target_metadata["pooled_output"] = base_pooled + (styled_pooled - base_pooled) * strength

        output.append([blended, target_metadata])

    return output


class CKSSeedRandom0100:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_SEED,
                        "control_after_generate": True,
                        "tooltip": "Deterministically maps a seed to a random value from 0 to 100.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("value_int", "value_float")
    FUNCTION = "generate"
    CATEGORY = PROMPT_CATEGORY

    def generate(self, seed):
        value = random.Random(int(seed)).randint(0, 100)
        return (value, float(value))


class CKSArtistPromptWeightedBlend:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "clip": ("CLIP",),
            "top_prompt": _string_input(
                multiline=True,
                tooltip="Fixed prompt text placed before every artist tag.",
            ),
            "bottom_prompt": _string_input(
                multiline=True,
                tooltip="Fixed prompt text placed after every artist tag.",
            ),
            "artists": _string_input(
                default="",
                multiline=True,
                tooltip="Comma-separated artist tags. Tag weights like (@artist:1.4) are preserved.",
            ),
            "blend_mode": (
                ["average", "exact", "prompt"],
                {"default": "average", "tooltip": "average mixes weighted conditioning tensors, exact outputs one conditioning entry per artist, prompt encodes one weighted prompt."},
            ),
            "separator": (
                "STRING",
                {
                    "default": ", ",
                    "multiline": False,
                    "tooltip": "Text inserted between top prompt, artist tag, and bottom prompt.",
                },
            ),
        }

        return {"required": required}

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "summary", "composite_prompt")
    FUNCTION = "blend"
    CATEGORY = PROMPT_CATEGORY

    def blend(self, clip, top_prompt, bottom_prompt, artists, blend_mode, separator):
        artist_entries = _parse_artist_entries(artists)
        conditioning, composite_prompt = _blend_artist_conditionings(
            clip,
            top_prompt,
            bottom_prompt,
            artist_entries,
            separator,
            blend_mode,
        )
        summary = "Artist weighted blend\n"
        summary += f"blend_mode: {blend_mode}\n"
        summary += f"composite_prompt: {composite_prompt}\n"
        summary += "\n".join(
            f"{index}: {_format_weighted_artist_tag(artist)}"
            for index, artist in enumerate(artist_entries, 1)
        )

        return (conditioning, summary, composite_prompt)


class CKSArtistStyleDeltaBlend:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "clip": ("CLIP",),
            "top_prompt": _string_input(
                multiline=True,
                tooltip="Fixed prompt text placed before every artist tag.",
            ),
            "bottom_prompt": _string_input(
                multiline=True,
                tooltip="Fixed prompt text placed after every artist tag.",
            ),
            "artists": _string_input(
                default="",
                multiline=True,
                tooltip="Comma-separated artist tags. Tag weights like (@artist:1.4) are preserved.",
            ),
            "blend_mode": (
                ["average", "exact", "prompt"],
                {"default": "average", "tooltip": "Blend mode used to build the styled artist conditioning before delta extraction."},
            ),
            "separator": (
                "STRING",
                {
                    "default": ", ",
                    "multiline": False,
                    "tooltip": "Text inserted between top prompt, artist tag, and bottom prompt.",
                },
            ),
            "style_strength": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 4.0,
                    "step": 0.01,
                    "tooltip": "Global strength for the blended artist-style delta.",
                },
            ),
        }

        return {"required": required}

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "summary", "composite_prompt")
    FUNCTION = "blend"
    CATEGORY = PROMPT_CATEGORY

    def blend(self, clip, top_prompt, bottom_prompt, artists, blend_mode, separator, style_strength):
        artist_entries = _parse_artist_entries(artists)
        base_prompt = _join_prompt(top_prompt, "", bottom_prompt, separator)
        base_conditioning = _encode_prompt(clip, base_prompt)
        styled_conditioning, composite_prompt = _blend_artist_conditionings(
            clip,
            top_prompt,
            bottom_prompt,
            artist_entries,
            separator,
            blend_mode,
        )
        conditioning = _artist_style_delta_conditioning(
            base_conditioning,
            styled_conditioning,
            style_strength,
        )
        summary = "Artist style delta blend\n"
        summary += f"blend_mode: {blend_mode}\n"
        summary += f"style_strength: {style_strength:g}\n"
        summary += f"base_prompt: {base_prompt}\n"
        summary += f"composite_prompt: {composite_prompt}\n"
        summary += "\n".join(
            f"{index}: {_format_weighted_artist_tag(artist)}"
            for index, artist in enumerate(artist_entries, 1)
        )

        return (conditioning, summary, composite_prompt)


class CKSImageSaveWithWorkflowName:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "workflow_name": ("STRING", {"default": "unknown", "multiline": False}),
                "filename": ("STRING", {"default": "%time_%seed", "multiline": False}),
                "path": ("STRING", {"default": "", "multiline": False}),
                "extension": (["png", "jpeg", "webp"],),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "modelname": ("STRING", {"default": "", "multiline": False, "tooltip": "Model/checkpoint name string for metadata."}),
                "sampler_name": ("STRING", {"default": "", "multiline": False, "tooltip": "Sampler name string for metadata."}),
                "scheduler": ("STRING", {"default": "normal", "multiline": False, "tooltip": "Scheduler name string for metadata."}),
            },
            "optional": {
                "positive": ("STRING", {"default": "unknown", "multiline": True}),
                "negative": ("STRING", {"default": "unknown", "multiline": True}),
                "seed_value": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                "lossless_webp": ("BOOLEAN", {"default": True}),
                "quality_jpeg_or_webp": ("INT", {"default": 100, "min": 1, "max": 100}),
                "png_compress_level": ("INT", {"default": 1, "min": 0, "max": 9}),
                "counter": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S", "multiline": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_files"
    OUTPUT_NODE = True
    CATEGORY = IMAGE_CATEGORY

    def save_files(
        self,
        images,
        workflow_name="unknown",
        seed_value=0,
        steps=20,
        cfg=8.0,
        sampler_name="",
        scheduler="normal",
        positive="unknown",
        negative="unknown",
        modelname="",
        quality_jpeg_or_webp=100,
        png_compress_level=1,
        lossless_webp=True,
        width=512,
        height=512,
        counter=0,
        filename="%time_%seed",
        path="",
        extension="png",
        time_format="%Y-%m-%d-%H%M%S",
        prompt=None,
        extra_pnginfo=None,
    ):
        filename = _make_filename(filename, seed_value, modelname, counter, time_format)
        path = _make_pathname(path, seed_value, modelname, counter, time_format)
        checkpoint_path = _find_checkpoint_path(modelname)
        base_model_name = _parse_model_name(modelname) if str(modelname).strip() else "unknown"
        model_hash = _calculate_sha256(checkpoint_path)[:10] if checkpoint_path else "unknown"
        workflow_name = _clean_metadata_text(workflow_name)
        sampler_name = _clean_metadata_text(sampler_name)
        scheduler = _clean_metadata_text(scheduler)
        sampler = f"{sampler_name}_{scheduler}" if scheduler and scheduler != "normal" else sampler_name
        comment = (
            f"{_clean_metadata_text(positive)}\n"
            f"Negative prompt: {_clean_metadata_text(negative)}\n"
            f"Steps: {steps}, Sampler: {sampler}, CFG Scale: {cfg}, Seed: {seed_value}, "
            f"Size: {width}x{height}, Model hash: {model_hash}, Model: {base_model_name}, "
            f"Workflow: {workflow_name}, Version: ComfyUI"
        )
        filename_prefix = _make_filename_prefix(path, filename)
        output_path, safe_filename, save_counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            width,
            height,
        )

        filenames = self.save_images(
            images,
            output_path,
            safe_filename,
            save_counter,
            comment,
            workflow_name,
            extension,
            quality_jpeg_or_webp,
            png_compress_level,
            lossless_webp,
            prompt,
            extra_pnginfo,
        )

        return {
            "ui": {
                "images": [
                    {
                        "filename": saved_filename,
                        "subfolder": subfolder if subfolder != "." else "",
                        "type": "output",
                    }
                    for saved_filename in filenames
                ]
            }
        }

    def save_images(
        self,
        images,
        output_path,
        filename_prefix,
        save_counter,
        comment,
        workflow_name,
        extension,
        quality_jpeg_or_webp,
        png_compress_level,
        lossless_webp,
        prompt=None,
        extra_pnginfo=None,
    ):
        image_count = 1
        paths = []
        for image in images:
            array = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            current_prefix = filename_prefix
            if images.size()[0] > 1:
                current_prefix += "_{:02d}".format(image_count)
            current_counter = save_counter + image_count - 1

            if extension == "png":
                metadata = PngInfo()
                metadata.add_text("parameters", comment)
                metadata.add_text("workflow_name", workflow_name)

                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for key, value in extra_pnginfo.items():
                        metadata.add_text(key, json.dumps(value))

                saved_filename = f"{current_prefix}_{current_counter:05d}_.png"
                img.save(
                    os.path.join(output_path, saved_filename),
                    pnginfo=metadata,
                    compress_level=int(png_compress_level),
                )
            else:
                if extension not in {"jpeg", "webp"}:
                    raise RuntimeError(f"Unsupported image extension: {extension}")
                saved_filename = f"{current_prefix}_{current_counter:05d}_.{extension}"
                file_path = os.path.join(output_path, saved_filename)
                exif_bytes = piexif.dump(
                    {
                        "0th": {
                            piexif.ImageIFD.ImageDescription: f"workflow_name={workflow_name}",
                        },
                        "Exif": {
                            piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(
                                comment,
                                encoding="unicode",
                            )
                        },
                    }
                )
                img.save(
                    file_path,
                    quality=quality_jpeg_or_webp,
                    lossless=lossless_webp,
                    exif=exif_bytes,
                )

            paths.append(saved_filename)
            image_count += 1
        return paths


class CoNAIArtifactFileOutput:
    """Copy a file or its parent folder into ComfyUI output for CoNAI artifact collection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "subfolder": ("STRING", {"default": "conai_artifacts", "multiline": False}),
                "filename_override": ("STRING", {"default": "", "multiline": False}),
                "copy_parent_folder": ("BOOLEAN", {"default": True}),
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "copy_file"
    OUTPUT_NODE = True
    CATEGORY = ARTIFACT_CATEGORY
    DESCRIPTION = "Copies a file or parent folder into ComfyUI output so CoNAI can collect it as an artifact."

    def copy_file(
        self,
        file_path,
        subfolder="conai_artifacts",
        filename_override="",
        copy_parent_folder=True,
        overwrite=False,
    ):
        raw_path = str(file_path or "").strip().strip('"')
        if not raw_path:
            raise ValueError("CoNAI Artifact File Output requires file_path.")

        source_path = Path(raw_path).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"Artifact source file not found: {source_path}")

        output_root = Path(folder_paths.get_output_directory()).resolve()
        safe_subfolder = _safe_artifact_subfolder(subfolder or "conai_artifacts")
        target_parent = (output_root / safe_subfolder).resolve()
        if not _is_within_directory(output_root, target_parent):
            raise ValueError("Artifact subfolder escapes ComfyUI output directory.")
        target_parent.mkdir(parents=True, exist_ok=True)

        if copy_parent_folder:
            source_root = source_path.parent
            base_name = _safe_artifact_file_name(
                str(filename_override).strip() if filename_override else source_root.name,
                source_root.name,
            )
            target_name = f"{_artifact_timestamp_prefix()}_{base_name}"
            target_root = _available_artifact_directory((target_parent / target_name).resolve(), bool(overwrite))
            if not _is_within_directory(output_root, target_root):
                raise ValueError("Artifact target escapes ComfyUI output directory.")
            shutil.copytree(str(source_root), str(target_root))
            return {"ui": {"files": _history_file_entries(output_root, target_root)}}

        target_name = _safe_artifact_file_name(
            str(filename_override).strip() if filename_override else source_path.name,
            source_path.name,
        )
        target_path = _available_artifact_path(target_parent, target_name, bool(overwrite)).resolve()
        if not _is_within_directory(output_root, target_path):
            raise ValueError("Artifact target escapes ComfyUI output directory.")
        shutil.copy2(str(source_path), str(target_path))
        return {
            "ui": {
                "files": [
                    {
                        "filename": target_path.name,
                        "subfolder": target_path.parent.relative_to(output_root).as_posix(),
                        "type": "output",
                    }
                ]
            }
        }


NODE_CLASS_MAPPINGS = {
    "CKSSeedRandom0100": CKSSeedRandom0100,
    "CKSArtistPromptWeightedBlend": CKSArtistPromptWeightedBlend,
    "CKSArtistStyleDeltaBlend": CKSArtistStyleDeltaBlend,
    "CKSImageSaveWithWorkflowName": CKSImageSaveWithWorkflowName,
    "CoNAIArtifactFileOutput": CoNAIArtifactFileOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CKSSeedRandom0100": "Seed Random 0-100",
    "CKSArtistPromptWeightedBlend": "Artist Prompt Weighted Blend",
    "CKSArtistStyleDeltaBlend": "Artist Style Delta Blend",
    "CKSImageSaveWithWorkflowName": "Save Image w/Workflow Name",
    "CoNAIArtifactFileOutput": "CoNAI Artifact File Output",
}

WEB_DIRECTORY = "js"
