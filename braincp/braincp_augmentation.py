"""
braincp_augmentation.py

BRAIN-CP augmentation module for BraTS2024 Glio.

Design goal
-----------
This file reuses the TumorCP-BraTS transformation/paste core, while changing
the paste-center sampler to BRAIN-CP occurrence-based sampling:

1. BraTS 4-channel MRI: (C, D, H, W), C=4
2. BraTS labels: 1/2/3 are tumor subregions and must be preserved
3. nnU-Net v2 trainer call style
4. BraTS has no explicit organ label, so target_organ_positions is replaced by
   target_brain_positions / valid_positions
5. Tumor paste is constrained to the target valid brain region
6. Paste center is sampled from a patch-level occurrence probability map
7. Inter-case intensity matching is performed channel-wise when do_match=True

Main original TumorCP concepts preserved
----------------------------------------
- cp_configs
- aug_one_pair()
- do_spatial_augment()
- do_blurring()
- do_flipping()
- do_gamma()
- paste_to()
- cp_times
- do_inter_cp / do_match
- online stochastic random tumor patch selection
- BRAIN-CP occurrence-based paste-center sampling
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from braincp.occurrence_sampler import (
    make_mixed_probability_map,
    sample_paste_center,
)


ArrayLike = np.ndarray
PathLike = Union[str, os.PathLike]


# -------------------------------------------------------------------------
# TumorCP-like configuration
# -------------------------------------------------------------------------

default_braincp_configs: Dict[str, Any] = {
    # Copy-Paste control
    "do_cp": True,
    "p_cp": 0.8,
    "cp_times": 1,

    # BRAIN-CP paste-center sampling control
    # P_paste = alpha * P_occurrence + (1 - alpha) * P_uniform
    "alpha": 0.8,
    "uniform_region": "positive",
    "require_probability_map": False,


    # Inter-patient Copy-Paste control.
    # In this BraTS adapter, patch_pool sampling is effectively inter-case.
    "do_inter_cp": True,
    "p_inter_cp": 1.0,
    "do_match": False,

    # Object-level spatial transformations
    "do_elastic": True,
    "elastic_deform_alpha": (0.0, 900.0),
    "elastic_deform_sigma": (9.0, 13.0),
    "p_eldef": 0.5,

    "do_scaling": True,
    "scale_range": (0.75, 1.25),
    "p_scale": 0.5,

    "do_rotation": True,
    # Original TumorCP allows broad rotation.
    # Unit: degree. Internally converted to radians.
    "degree": 180.0,
    "angle_range": (-180.0, 180.0),
    "p_rot": 0.5,

    # Object-level intensity / texture transformations
    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.5,

    "do_mirror": True,
    "p_mirror": 0.5,

    "do_blurring": True,
    "blur_sigma": (0.5, 1.0),
    "p_blur": 0.5,

    # BraTS adapter-specific safety controls
    "min_inside_fraction": 1.0,
    "max_existing_tumor_overlap_fraction": 0.10,
    "max_location_attempts": 50,
    "brain_mask_mode": "seg_valid",  # "seg_valid", "data_nonzero", or "both"

    # BRAIN-CP safety control. If True, the transformed source patch box must
    # fit fully inside the target training patch before paste. This prevents
    # partial tumors being clipped at the target boundary.
    "require_full_patch_inside": True,
}


# Backward-compatible aliases used by older scripts/tests.
default_cp_configs = default_braincp_configs
DEFAULT_TRANSFORM_CFG = default_braincp_configs


def _merge_cfg(user_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(default_braincp_configs)
    if user_cfg is not None:
        cfg.update(user_cfg)

    # Compatibility aliases from the previous TumorCP-style implementation.
    if "blur_sigma_range" in cfg and "blur_sigma" not in cfg:
        cfg["blur_sigma"] = cfg["blur_sigma_range"]
    if "elastic_alpha_range" in cfg and "elastic_deform_alpha" not in cfg:
        cfg["elastic_deform_alpha"] = cfg["elastic_alpha_range"]
    if "elastic_sigma_range" in cfg and "elastic_deform_sigma" not in cfg:
        cfg["elastic_deform_sigma"] = cfg["elastic_sigma_range"]

    return cfg


def _as_rng(rng: Optional[Union[np.random.Generator, np.random.RandomState, int]]) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, np.random.RandomState):
        return np.random.default_rng(int(rng.randint(0, 2**31 - 1)))
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    return np.random.default_rng()


def _ensure_tuple3(x: Sequence[int]) -> Tuple[int, int, int]:
    if len(x) != 3:
        raise ValueError(f"Expected length 3 shape, got {x}")
    return int(x[0]), int(x[1]), int(x[2])


def _get_highres_target(target: Any) -> Tuple[np.ndarray, Any, Optional[int]]:
    """
    nnU-Net v2 may pass deep-supervision target as a list/tuple.
    Use target[0] as high-resolution target and put it back after augmentation.
    """
    if isinstance(target, list):
        if len(target) == 0:
            raise ValueError("target is an empty list.")
        return target[0], target, 0
    if isinstance(target, tuple):
        if len(target) == 0:
            raise ValueError("target is an empty tuple.")
        return target[0], target, 0
    return target, target, None


def _set_highres_target(original_target: Any, updated_highres: np.ndarray, list_index: Optional[int]) -> Any:
    if list_index is None:
        return updated_highres

    if isinstance(original_target, list):
        original_target[list_index] = updated_highres
        return original_target

    if isinstance(original_target, tuple):
        tmp = list(original_target)
        tmp[list_index] = updated_highres
        return tuple(tmp)

    return updated_highres


def _seg_view_3d(seg: np.ndarray, batch_index: Optional[int] = None) -> np.ndarray:
    """
    Convert target/seg to a 3D view without copying when possible.
    Supported:
        batched:     (B, 1, D, H, W) or (B, D, H, W)
        non-batched: (1, D, H, W) or (D, H, W)
    """
    if batch_index is not None:
        if seg.ndim == 5:
            return seg[batch_index, 0]
        if seg.ndim == 4:
            return seg[batch_index]
        raise ValueError(f"Unsupported batched target shape: {seg.shape}")

    if seg.ndim == 4:
        if seg.shape[0] != 1:
            raise ValueError(f"Unsupported non-batched target shape: {seg.shape}")
        return seg[0]

    if seg.ndim == 3:
        return seg

    raise ValueError(f"Unsupported target shape: {seg.shape}")


def _image_view_4d(data: np.ndarray, batch_index: Optional[int] = None) -> np.ndarray:
    """
    Convert data to a 4D image view: (C, D, H, W).
    """
    if batch_index is not None:
        if data.ndim != 5:
            raise ValueError(f"Expected batched data shape (B,C,D,H,W), got {data.shape}")
        return data[batch_index]

    if data.ndim == 4:
        return data

    raise ValueError(f"Expected image shape (C,D,H,W), got {data.shape}")


# -------------------------------------------------------------------------
# Patch pool loading
# -------------------------------------------------------------------------

def _discover_patch_files(pool_dir: PathLike) -> List[Path]:
    pool_path = Path(pool_dir)
    if not pool_path.exists():
        raise FileNotFoundError(f"Tumor patch pool directory does not exist: {pool_path}")

    summary_path = pool_path / "summary.json"
    candidate_files: List[Path] = []

    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            possible_lists = []
            if isinstance(summary, dict):
                for key in ["patches", "patch_files", "files", "items"]:
                    if key in summary and isinstance(summary[key], list):
                        possible_lists.append(summary[key])
            elif isinstance(summary, list):
                possible_lists.append(summary)

            for item_list in possible_lists:
                for item in item_list:
                    if isinstance(item, str):
                        p = pool_path / item
                    elif isinstance(item, dict):
                        raw = (
                            item.get("file")
                            or item.get("path")
                            or item.get("npz")
                            or item.get("patch_file")
                            or item.get("image_file")
                        )
                        if raw is None:
                            continue
                        p = pool_path / str(raw)
                    else:
                        continue

                    if p.exists() and p.suffix.lower() in [".npz", ".npy"]:
                        candidate_files.append(p)
        except Exception:
            candidate_files = []

    if not candidate_files:
        candidate_files = sorted(
            list(pool_path.rglob("*.npz")) + list(pool_path.rglob("*.npy"))
        )

    filtered = []
    for p in candidate_files:
        name = p.name.lower()
        if "summary" in name or "stat" in name:
            continue
        filtered.append(p)

    if len(filtered) == 0:
        raise FileNotFoundError(f"No .npz/.npy tumor patch files found under: {pool_path}")

    return sorted(set(filtered))


def _first_existing_key(obj: Any, keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        try:
            if k in obj:
                return k
        except Exception:
            pass
    return None


def _load_patch_from_file(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one tumor patch.

    Returns:
        img_patch: (C, d, h, w)
        seg_patch: (d, h, w)
    """
    p = Path(path)
    suffix = p.suffix.lower()

    image_keys = ["image", "img", "data", "x", "patch_img", "patch_image"]
    seg_keys = ["seg", "label", "labels", "mask", "y", "patch_seg", "patch_label"]

    if suffix == ".npz":
        with np.load(p, allow_pickle=True) as z:
            img_key = _first_existing_key(z, image_keys)
            seg_key = _first_existing_key(z, seg_keys)
            if img_key is None or seg_key is None:
                raise KeyError(
                    f"Patch file {p} does not contain image/seg keys. "
                    f"Available keys: {list(z.keys())}"
                )
            img_patch = z[img_key]
            seg_patch = z[seg_key]

    elif suffix == ".npy":
        obj = np.load(p, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            obj = obj.item()

        if not isinstance(obj, dict):
            raise ValueError(
                f"Unsupported .npy patch format in {p}. "
                "Use .npz or dict-style .npy with image/data and seg/label keys."
            )

        img_key = _first_existing_key(obj, image_keys)
        seg_key = _first_existing_key(obj, seg_keys)
        if img_key is None or seg_key is None:
            raise KeyError(
                f"Patch dict {p} does not contain image/seg keys. "
                f"Available keys: {list(obj.keys())}"
            )
        img_patch = obj[img_key]
        seg_patch = obj[seg_key]

    else:
        raise ValueError(f"Unsupported patch file extension: {p}")

    img_patch = np.asarray(img_patch)
    seg_patch = np.asarray(seg_patch)

    # Normalize image shape to (C, D, H, W)
    if img_patch.ndim == 5 and img_patch.shape[0] == 1:
        img_patch = img_patch[0]
    if img_patch.ndim != 4:
        raise ValueError(f"Expected image patch shape (C,D,H,W), got {img_patch.shape} from {p}")

    # Normalize seg shape to (D, H, W)
    if seg_patch.ndim == 5 and seg_patch.shape[0] == 1 and seg_patch.shape[1] == 1:
        seg_patch = seg_patch[0, 0]
    elif seg_patch.ndim == 4 and seg_patch.shape[0] == 1:
        seg_patch = seg_patch[0]
    elif seg_patch.ndim != 3:
        raise ValueError(f"Expected seg patch shape (D,H,W), got {seg_patch.shape} from {p}")

    if img_patch.shape[-3:] != seg_patch.shape:
        raise ValueError(
            f"Image/seg patch spatial shape mismatch in {p}: "
            f"{img_patch.shape[-3:]} vs {seg_patch.shape}"
        )

    return img_patch, seg_patch


# -------------------------------------------------------------------------
# Original TumorCP-like transform core
# -------------------------------------------------------------------------

def create_zero_centered_coordinate_mesh(shape: Sequence[int]) -> np.ndarray:
    """
    Original TumorCP-style coordinate mesh.
    shape: (D, H, W)
    returns: (3, D, H, W)
    """
    shape = _ensure_tuple3(shape)
    coords = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    coords = np.asarray(coords, dtype=np.float32)
    for d in range(3):
        coords[d] -= (shape[d] - 1) / 2.0
    return coords


def rotate_coords_3d(coords: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate coordinate mesh around the D-axis.
    For patch shape (D, H, W), this rotates the H-W plane.
    angle: radians.
    """
    c = float(np.cos(angle))
    s = float(np.sin(angle))

    out = coords.copy()
    y = coords[1].copy()
    x = coords[2].copy()

    out[1] = c * y - s * x
    out[2] = s * y + c * x
    return out


def scale_coords(coords: np.ndarray, scale: float) -> np.ndarray:
    return coords * float(scale)


def _elastic_deform_coords(
    coords: np.ndarray,
    alpha: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = coords.copy()
    spatial_shape = coords.shape[1:]

    for d in range(3):
        displacement = gaussian_filter(
            (rng.random(spatial_shape).astype(np.float32) * 2.0 - 1.0),
            sigma=float(sigma),
            mode="constant",
            cval=0,
        ) * float(alpha)
        out[d] += displacement

    return out


def interpolate_img(
    img: np.ndarray,
    coords: np.ndarray,
    order: int,
    mode: str = "constant",
    cval: float = 0.0,
    is_seg: bool = False,
) -> np.ndarray:
    """
    Original TumorCP-like interpolation wrapper.
    img can be (D,H,W) or (C,D,H,W).
    """
    if img.ndim == 3:
        return map_coordinates(
            img,
            coords,
            order=int(order),
            mode=mode,
            cval=cval,
        )

    if img.ndim == 4:
        out = np.empty_like(img)
        for c in range(img.shape[0]):
            out[c] = map_coordinates(
                img[c],
                coords,
                order=int(order),
                mode=mode,
                cval=cval,
            )
        return out

    raise ValueError(f"interpolate_img expects 3D or 4D input, got {img.shape}")


def do_scaling(
    data: np.ndarray,
    seg: np.ndarray,
    scale: Tuple[float, float] = (0.75, 1.25),
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Original TumorCP helper name retained.
    Applies scaling via coordinate mesh to all MRI channels and segmentation.
    """
    rng = _as_rng(rng)
    coords = create_zero_centered_coordinate_mesh(seg.shape)
    sc = float(rng.uniform(scale[0], scale[1]))
    coords = scale_coords(coords, sc)

    for d in range(3):
        coords[d] += (seg.shape[d] - 1) / 2.0

    data = interpolate_img(data, coords, order=1, mode="nearest", cval=0)
    seg = interpolate_img(seg, coords, order=0, mode="constant", cval=0, is_seg=True)
    return data, seg


def do_spatial_augment(
    data: np.ndarray,
    seg: np.ndarray,
    do_scaling: bool = False,
    p_scale: float = 0.5,
    scale: Tuple[float, float] = (0.75, 1.25),
    do_rotation: bool = False,
    p_rot: float = 0.5,
    angle: Tuple[float, float] = (-np.pi, np.pi),
    do_elastic: bool = False,
    p_elas: float = 0.5,
    alpha: Tuple[float, float] = (0.0, 900.0),
    sigma: Tuple[float, float] = (9.0, 13.0),
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Original TumorCP-style spatial augmentation.

    Adaptation:
        data: (C,D,H,W), all channels share the same coordinate transform.
        seg:  (D,H,W), nearest interpolation.
    """
    rng = _as_rng(rng)

    coords = create_zero_centered_coordinate_mesh(seg.shape)
    modified_coords = False

    if do_rotation and rng.random() <= float(p_rot):
        a = float(rng.uniform(angle[0], angle[1]))
        coords = rotate_coords_3d(coords, a)
        modified_coords = True

    if do_scaling and rng.random() <= float(p_scale):
        sc = float(rng.uniform(scale[0], scale[1]))
        coords = scale_coords(coords, sc)
        modified_coords = True

    if do_elastic and rng.random() <= float(p_elas):
        a = float(rng.uniform(alpha[0], alpha[1]))
        s = float(rng.uniform(sigma[0], sigma[1]))
        coords = _elastic_deform_coords(coords, a, s, rng)
        modified_coords = True

    if modified_coords:
        for d in range(3):
            coords[d] += (seg.shape[d] - 1) / 2.0

        data = interpolate_img(data, coords, order=1, mode="nearest", cval=0)
        seg = interpolate_img(seg, coords, order=0, mode="constant", cval=0, is_seg=True)

    return data, seg


def do_flipping(
    data: np.ndarray,
    seg: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Original TumorCP helper name retained.
    Randomly flips along one of the same axis combinations as the original style.
    """
    rng = _as_rng(rng)

    # seg axes: (D,H,W) -> 0,1,2
    axis_options = [None, (2,), (1,), (2, 1), (0,), (2, 0), (1, 0), (2, 1, 0)]
    axis = axis_options[int(rng.integers(0, len(axis_options)))]

    if axis is None:
        return data, seg

    # data axes are shifted by +1 because data is (C,D,H,W).
    data_axes = tuple(a + 1 for a in axis)
    data = np.flip(data, data_axes).copy()
    seg = np.flip(seg, axis).copy()

    return data, seg


def do_gamma(
    data: np.ndarray,
    seg: np.ndarray,
    retain_stats: bool = True,
    gamma_range: Tuple[float, float] = (0.7, 1.5),
    epsilon: float = 1e-7,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Original TumorCP helper name retained.
    MRI adaptation: apply channel-wise to full patch volume (not tumor-only).
    """
    rng = _as_rng(rng)
    out = data.copy()

    for c in range(out.shape[0]):
        ch = out[c].astype(np.float32)

        if retain_stats:
            mn = float(np.mean(ch))
            sd = float(np.std(ch))

        if rng.random() < 0.5 and gamma_range[0] < 1:
            gamma = float(rng.uniform(gamma_range[0], 1.0))
        else:
            gamma = float(rng.uniform(max(gamma_range[0], 1.0), gamma_range[1]))

        minm = float(np.min(ch))
        rnge = float(np.max(ch)) - minm

        if rnge <= epsilon:
            continue

        ch = np.power((ch - minm) / float(rnge + epsilon), gamma) * float(rnge + epsilon) + minm

        if retain_stats:
            ch = ch - np.mean(ch) + mn
            ch = ch / (np.std(ch) + 1e-8) * sd

        out[c] = ch.astype(data.dtype)

    return out, seg


def do_blurring(
    data: np.ndarray,
    seg: np.ndarray,
    blur_sigma: Tuple[float, float] = (0.5, 1.0),
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Original TumorCP helper name retained.
    MRI adaptation: apply channel-wise to full patch volume (not tumor-only).
    """
    rng = _as_rng(rng)
    sigma = float(rng.uniform(blur_sigma[0], blur_sigma[1]))
    out = data.copy()

    for c in range(out.shape[0]):
        out[c] = gaussian_filter(out[c], sigma=sigma, order=0)

    return out, seg


# -------------------------------------------------------------------------
# BraTS-specific patch preparation and target positions
# -------------------------------------------------------------------------

def _remove_empty_border(
    img_patch: np.ndarray,
    seg_patch: np.ndarray,
    margin: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = seg_patch > 0
    if not np.any(mask):
        return img_patch, seg_patch

    coords = np.where(mask)
    starts = [max(0, int(c.min()) - margin) for c in coords]
    ends = [min(seg_patch.shape[i], int(coords[i].max()) + margin + 1) for i in range(3)]
    sl = tuple(slice(starts[i], ends[i]) for i in range(3))

    return img_patch[(slice(None),) + sl], seg_patch[sl]


def _center_crop_patch_to_max_shape(
    img_patch: np.ndarray,
    seg_patch: np.ndarray,
    max_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    max_shape = _ensure_tuple3(max_shape)
    slices = []

    for dim, max_s in enumerate(max_shape):
        cur_s = seg_patch.shape[dim]
        if cur_s <= max_s:
            slices.append(slice(0, cur_s))
        else:
            start = (cur_s - max_s) // 2
            end = start + max_s
            slices.append(slice(start, end))

    sl = tuple(slices)
    return img_patch[(slice(None),) + sl], seg_patch[sl]


def _prepare_source_tumor_patch(
    img_patch: np.ndarray,
    seg_patch: np.ndarray,
    target_max_shape: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare source tumor patch before TumorCP transforms.
    BraTS labels 1/2/3 are preserved.
    """
    original_img_dtype = img_patch.dtype
    original_seg_dtype = seg_patch.dtype

    img_patch = np.asarray(img_patch)
    seg_patch = np.asarray(seg_patch).copy()

    # Ignore labels should not become tumor.
    seg_patch[seg_patch < 0] = 0

    if img_patch.ndim != 4:
        raise ValueError(f"Expected source image patch (C,D,H,W), got {img_patch.shape}")
    if seg_patch.ndim != 3:
        raise ValueError(f"Expected source seg patch (D,H,W), got {seg_patch.shape}")
    if img_patch.shape[-3:] != seg_patch.shape:
        raise ValueError(f"Patch image/seg shape mismatch: {img_patch.shape[-3:]} vs {seg_patch.shape}")

    if not np.any(seg_patch > 0):
        return img_patch.astype(original_img_dtype), seg_patch.astype(original_seg_dtype)

    img_patch, seg_patch = _remove_empty_border(img_patch, seg_patch, margin=2)

    if target_max_shape is not None:
        img_patch, seg_patch = _center_crop_patch_to_max_shape(img_patch, seg_patch, target_max_shape)

    return img_patch.astype(original_img_dtype), seg_patch.astype(original_seg_dtype)


def get_target_brain_positions(
    target_data: np.ndarray,
    target_seg: np.ndarray,
    mode: str = "seg_valid",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    BraTS replacement for original TumorCP target_organ_positions.

    Original TumorCP:
        target_organ_positions = properties["class_locations"][ORGAN_LABEL]

    BraTS adapter:
        target_brain_positions = np.argwhere(valid brain mask)

    mode:
        "seg_valid":    target_seg >= 0
        "data_nonzero": any nonzero voxel across 4 MRI channels
        "both":         intersection of the above
    """
    if target_data.ndim != 4:
        raise ValueError(f"target_data must be (C,D,H,W), got {target_data.shape}")
    if target_seg.ndim != 3:
        raise ValueError(f"target_seg must be (D,H,W), got {target_seg.shape}")

    valid_seg = target_seg >= 0

    # Robust but optional image-based brain mask.
    # nnU-Net preprocessed MRI can contain normalized values, so seg_valid is default.
    data_nonzero = np.any(np.abs(target_data) > 1e-8, axis=0)

    if mode == "seg_valid":
        brain_mask = valid_seg
    elif mode == "data_nonzero":
        brain_mask = data_nonzero
    elif mode == "both":
        brain_mask = valid_seg & data_nonzero
    else:
        raise ValueError(f"Unknown brain_mask_mode: {mode}")

    positions = np.argwhere(brain_mask)
    return positions, brain_mask


def get_valid_center(
    center: int,
    patch_length: int,
    volume_length: int,
) -> Tuple[int, int, Tuple[int, int]]:
    """
    Original TumorCP helper name retained.
    Returns target start/end and source patch start/end for one dimension.
    """
    center = int(center)
    patch_length = int(patch_length)
    volume_length = int(volume_length)

    start = center - patch_length // 2
    end = start + patch_length

    ps = 0
    pe = patch_length

    if start < 0:
        ps = -start
        start = 0

    if end > volume_length:
        pe = patch_length - (end - volume_length)
        end = volume_length

    return int(start), int(end), (int(ps), int(pe))


def _build_slices_from_center(
    center: Sequence[int],
    patch_shape: Sequence[int],
    target_shape: Sequence[int],
) -> Tuple[Tuple[slice, slice, slice], Tuple[slice, slice, slice]]:
    center = _ensure_tuple3(center)
    patch_shape = _ensure_tuple3(patch_shape)
    target_shape = _ensure_tuple3(target_shape)

    target_slices = []
    patch_slices = []

    for dim in range(3):
        start, end, (ps, pe) = get_valid_center(center[dim], patch_shape[dim], target_shape[dim])
        target_slices.append(slice(start, end))
        patch_slices.append(slice(ps, pe))

    return tuple(target_slices), tuple(patch_slices)


# -------------------------------------------------------------------------
# TumorCP Copy-Paste core: paste_to() and aug_one_pair()
# -------------------------------------------------------------------------

def paste_to(
    center: Sequence[int],
    data_patch: np.ndarray,
    seg_patch: np.ndarray,
    tgt_data: np.ndarray,
    tgt_seg: np.ndarray,
    cp_configs: Optional[Dict[str, Any]] = None,
    brain_mask: Optional[np.ndarray] = None,
    return_info: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    BraTS-adapted version of original TumorCP paste_to().

    Original TumorCP:
        paste only seg_patch == TUMOR_LABEL.
        optionally do mean matching for inter-CP.

    BraTS adapter:
        paste only seg_patch > 0.
        preserve label values 1/2/3.
        never paste into target ignore label -1.
        optionally do channel-wise mean matching.
        reject paste if tumor is outside brain/valid region more than allowed.
    """
    cfg = _merge_cfg(cp_configs)

    if tgt_data.ndim != 4:
        raise ValueError(f"Expected tgt_data (C,D,H,W), got {tgt_data.shape}")
    if tgt_seg.ndim != 3:
        raise ValueError(f"Expected tgt_seg (D,H,W), got {tgt_seg.shape}")
    if data_patch.ndim != 4:
        raise ValueError(f"Expected data_patch (C,d,h,w), got {data_patch.shape}")
    if seg_patch.ndim != 3:
        raise ValueError(f"Expected seg_patch (d,h,w), got {seg_patch.shape}")
    if data_patch.shape[0] != tgt_data.shape[0]:
        raise ValueError(f"Channel mismatch: patch {data_patch.shape[0]}, target {tgt_data.shape[0]}")

    seg_patch = seg_patch.copy()
    seg_patch[seg_patch < 0] = 0

    target_slices, patch_slices = _build_slices_from_center(
        center=center,
        patch_shape=seg_patch.shape,
        target_shape=tgt_seg.shape,
    )

    boundary_cropped = any(
        patch_slices[dim].start != 0 or patch_slices[dim].stop != seg_patch.shape[dim]
        for dim in range(3)
    )

    src_seg = seg_patch[patch_slices]
    tumor_mask = src_seg > 0

    info: Dict[str, Any] = {
        "applied": False,
        "reason": None,
        "center": tuple(int(x) for x in center),
        "target_slices": tuple((s.start, s.stop) for s in target_slices),
        "patch_slices": tuple((s.start, s.stop) for s in patch_slices),
        "inside_fraction": 0.0,
        "overlap_fraction": 0.0,
        "pasted_voxels": 0,
        "boundary_cropped": bool(boundary_cropped),
    }

    if bool(cfg.get("require_full_patch_inside", True)) and boundary_cropped:
        info["reason"] = "patch_box_outside_target_volume"
        if return_info:
            return tgt_data, tgt_seg, info
        return tgt_data, tgt_seg

    if not np.any(tumor_mask):
        info["reason"] = "empty_tumor_after_boundary_crop"
        if return_info:
            return tgt_data, tgt_seg, info
        return tgt_data, tgt_seg

    valid_target = tgt_seg[target_slices] >= 0
    if brain_mask is not None:
        valid_target = valid_target & brain_mask[target_slices]

    inside_count = int(np.logical_and(tumor_mask, valid_target).sum())
    total_count = int(tumor_mask.sum())
    inside_fraction = inside_count / max(total_count, 1)
    info["inside_fraction"] = float(inside_fraction)

    min_inside_fraction = float(cfg.get("min_inside_fraction", 1.0))
    if inside_fraction < min_inside_fraction:
        info["reason"] = "tumor_outside_brain"
        if return_info:
            return tgt_data, tgt_seg, info
        return tgt_data, tgt_seg

    existing_tumor = tgt_seg[target_slices] > 0
    overlap_count = int(np.logical_and(tumor_mask, existing_tumor).sum())
    overlap_fraction = overlap_count / max(total_count, 1)
    info["overlap_fraction"] = float(overlap_fraction)

    max_overlap = float(cfg.get("max_existing_tumor_overlap_fraction", 0.10))
    if overlap_fraction > max_overlap:
        info["reason"] = "too_much_existing_tumor_overlap"
        if return_info:
            return tgt_data, tgt_seg, info
        return tgt_data, tgt_seg

    # If min_inside_fraction < 1.0, still do not paste into ignore/outside region.
    paste_mask = tumor_mask & valid_target
    if not np.any(paste_mask):
        info["reason"] = "empty_paste_mask"
        if return_info:
            return tgt_data, tgt_seg, info
        return tgt_data, tgt_seg

    # Channel-wise image paste.
    for c in range(tgt_data.shape[0]):
        dst_view = tgt_data[(c,) + target_slices]
        src_view = data_patch[(c,) + patch_slices]

        if bool(cfg.get("do_inter_cp", True)) and bool(cfg.get("do_match", False)):
            src_vals = src_view[paste_mask]
            tgt_vals = dst_view[paste_mask]

            if src_vals.size > 0 and tgt_vals.size > 0:
                src_mean = float(np.mean(src_vals))
                tgt_mean = float(np.mean(tgt_vals))
                dst_view[paste_mask] = src_vals + (tgt_mean - src_mean)
            else:
                dst_view[paste_mask] = src_view[paste_mask]
        else:
            dst_view[paste_mask] = src_view[paste_mask]

    # Preserve BraTS labels 1/2/3.
    dst_seg_view = tgt_seg[target_slices]
    dst_seg_view[paste_mask] = src_seg[paste_mask]

    info["applied"] = True
    info["reason"] = "success"
    info["pasted_voxels"] = int(np.sum(paste_mask))

    if return_info:
        return tgt_data, tgt_seg, info

    return tgt_data, tgt_seg


def _apply_tumorcp_transforms(
    cropped_data: np.ndarray,
    cropped_seg: np.ndarray,
    cp_configs: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Keep original TumorCP aug_one_pair transform order:
        spatial -> blurring -> mirroring -> gamma
    """
    transformed: List[str] = []

    # Spatial augmentation: rotation/scaling/elastic in one coordinate-mesh pass.
    if (
        bool(cp_configs.get("do_rotation", False))
        or bool(cp_configs.get("do_scaling", False))
        or bool(cp_configs.get("do_elastic", False))
    ):
        angle_deg_range = cp_configs.get("angle_range", None)
        if angle_deg_range is None:
            degree = float(cp_configs.get("degree", 180.0))
            angle_deg_range = (-degree, degree)
        angle_rad_range = (np.deg2rad(float(angle_deg_range[0])), np.deg2rad(float(angle_deg_range[1])))

        before_seg_sum = int(np.sum(cropped_seg > 0))
        cropped_data, cropped_seg = do_spatial_augment(
            cropped_data,
            cropped_seg,
            do_scaling=bool(cp_configs.get("do_scaling", False)),
            p_scale=float(cp_configs.get("p_scale", 0.5)),
            scale=tuple(cp_configs.get("scale_range", (0.75, 1.25))),
            do_rotation=bool(cp_configs.get("do_rotation", False)),
            p_rot=float(cp_configs.get("p_rot", 0.5)),
            angle=angle_rad_range,
            do_elastic=bool(cp_configs.get("do_elastic", False)),
            p_elas=float(cp_configs.get("p_eldef", 0.5)),
            alpha=tuple(cp_configs.get("elastic_deform_alpha", (0.0, 900.0))),
            sigma=tuple(cp_configs.get("elastic_deform_sigma", (9.0, 13.0))),
            rng=rng,
        )
        cropped_seg = np.rint(cropped_seg).astype(np.int16)
        cropped_seg[cropped_seg < 0] = 0
        if int(np.sum(cropped_seg > 0)) != before_seg_sum or before_seg_sum > 0:
            transformed.append("spatial")

    if bool(cp_configs.get("do_blurring", False)) and rng.random() <= float(cp_configs.get("p_blur", 0.5)):
        cropped_data, cropped_seg = do_blurring(
            cropped_data,
            cropped_seg,
            blur_sigma=tuple(cp_configs.get("blur_sigma", (0.5, 1.0))),
            rng=rng,
        )
        transformed.append("blurring")

    if bool(cp_configs.get("do_mirror", False)) and rng.random() <= float(cp_configs.get("p_mirror", 0.5)):
        cropped_data, cropped_seg = do_flipping(cropped_data, cropped_seg, rng=rng)
        transformed.append("mirroring")

    if bool(cp_configs.get("do_gamma", False)) and rng.random() <= float(cp_configs.get("p_gamma", 0.5)):
        cropped_data, cropped_seg = do_gamma(
            cropped_data,
            cropped_seg,
            retain_stats=bool(cp_configs.get("gamma_retain_stats", True)),
            gamma_range=tuple(cp_configs.get("gamma_range", (0.7, 1.5))),
            rng=rng,
        )
        transformed.append("gamma")

    return cropped_data, cropped_seg, {"transformed": transformed}




def _squeeze_patch_probability_map(prob_map: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """
    Normalize patch probability map shape to (D, H, W).
    """
    arr = np.asarray(prob_map)
    target_shape = tuple(int(x) for x in target_shape)

    if arr.ndim == 3:
        out = arr
    elif arr.ndim == 4 and arr.shape[0] == 1:
        out = arr[0]
    else:
        raise ValueError(
            f"patch_probability_map must be 3D or (1,D,H,W), got {arr.shape}"
        )

    if out.shape != target_shape:
        raise ValueError(
            f"patch_probability_map shape mismatch. Expected {target_shape}, got {out.shape}"
        )

    return out.astype(np.float64, copy=False)


def _select_patch_probability_map(
    patch_probability_maps: Optional[Any],
    batch_index: int,
    is_batched: bool,
    batch_size: int,
    target_shape: Sequence[int],
) -> Optional[np.ndarray]:
    """
    Select per-sample patch probability map from several possible formats.

    Supported:
        None
        (D,H,W) for one sample or shared map
        (1,D,H,W) for one sample
        (B,D,H,W) for batched maps
        list/tuple of length B with each item (D,H,W)
    """
    if patch_probability_maps is None:
        return None

    if isinstance(patch_probability_maps, (list, tuple)):
        if len(patch_probability_maps) != batch_size:
            raise ValueError(
                f"patch_probability_maps list length must match batch_size. "
                f"Expected {batch_size}, got {len(patch_probability_maps)}"
            )
        return _squeeze_patch_probability_map(patch_probability_maps[batch_index], target_shape)

    arr = np.asarray(patch_probability_maps)

    if arr.ndim == 3:
        return _squeeze_patch_probability_map(arr, target_shape)

    if arr.ndim == 4:
        # Case 1: batched map, (B,D,H,W)
        if is_batched and arr.shape[0] == batch_size:
            return _squeeze_patch_probability_map(arr[batch_index], target_shape)

        # Case 2: single map with channel-like first dim, (1,D,H,W)
        if arr.shape[0] == 1:
            return _squeeze_patch_probability_map(arr, target_shape)

    raise ValueError(
        f"Unsupported patch_probability_maps shape: {arr.shape}. "
        "Use (D,H,W), (1,D,H,W), (B,D,H,W), or list of 3D maps."
    )


def _make_braincp_center_probability_map(
    patch_probability_map: Optional[np.ndarray],
    target_brain_mask: Optional[np.ndarray],
    alpha: float,
    cfg: Dict[str, Any],
) -> Optional[np.ndarray]:
    """
    Build BRAIN-CP mixed probability map for one already-cropped training patch.

    Formula:
        P_paste = alpha * P_occurrence + (1 - alpha) * P_uniform
    """
    if patch_probability_map is None:
        return None

    if target_brain_mask is not None and target_brain_mask.shape != patch_probability_map.shape:
        raise ValueError(
            f"target_brain_mask shape mismatch. "
            f"mask={target_brain_mask.shape}, prob={patch_probability_map.shape}"
        )

    valid_mask = target_brain_mask if target_brain_mask is not None else (patch_probability_map > 0)

    return make_mixed_probability_map(
        occurrence_prob_map=patch_probability_map,
        alpha=float(alpha),
        valid_mask=valid_mask,
        uniform_region=str(cfg.get("uniform_region", "positive")),
    )


def aug_one_pair_braincp(
    source_tumors: Sequence[PathLike],
    target_data: np.ndarray,
    target_seg: np.ndarray,
    patch_probability_map: Optional[np.ndarray] = None,
    target_brain_positions: Optional[np.ndarray] = None,
    cp_configs: Optional[Dict[str, Any]] = None,
    rng: Optional[Union[np.random.Generator, int]] = None,
    target_brain_mask: Optional[np.ndarray] = None,
    target_max_shape: Optional[Tuple[int, int, int]] = None,
    alpha: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    BRAIN-CP version of TumorCP aug_one_pair().

    Difference from TumorCP-BraTS:
        TumorCP-BraTS chooses paste center uniformly/randomly from brain positions.
        BRAIN-CP chooses paste center from patch_probability_map using alpha mixing.

    Transformation order is kept the same as TumorCP-BraTS:
        spatial -> blurring -> mirroring -> gamma
    """
    rng = _as_rng(rng)
    cfg = _merge_cfg(cp_configs)
    alpha_value = float(cfg.get("alpha", 0.8) if alpha is None else alpha)

    info: Dict[str, Any] = {
        "applied": False,
        "reason": "not_attempted",
        "patch_file": None,
        "center": None,
        "paste_pos": None,
        "patch_shape_before": None,
        "patch_shape_after": None,
        "transformed": [],
        "attempts": [],
        "sampler": "braincp_occurrence" if patch_probability_map is not None else "random_fallback",
        "alpha": alpha_value,
    }

    if target_data.ndim != 4:
        raise ValueError(f"target_data must be (C,D,H,W), got {target_data.shape}")
    if target_seg.ndim != 3:
        raise ValueError(f"target_seg must be (D,H,W), got {target_seg.shape}")

    if len(source_tumors) == 0:
        info["reason"] = "empty_source_tumors"
        return target_data, target_seg, info

    if patch_probability_map is None and bool(cfg.get("require_probability_map", False)):
        info["reason"] = "missing_patch_probability_map"
        return target_data, target_seg, info

    if patch_probability_map is None:
        if target_brain_positions is None or len(target_brain_positions) == 0:
            info["reason"] = "empty_target_brain_positions_and_no_probability_map"
            return target_data, target_seg, info
        center_probability_map = None
    else:
        patch_probability_map = _squeeze_patch_probability_map(
            patch_probability_map,
            target_shape=target_seg.shape,
        )
        center_probability_map = _make_braincp_center_probability_map(
            patch_probability_map=patch_probability_map,
            target_brain_mask=target_brain_mask,
            alpha=alpha_value,
            cfg=cfg,
        )

    if target_max_shape is None:
        target_max_shape = target_data.shape[-3:]

    cp_times = int(cfg.get("cp_times", 1))
    max_location_attempts = int(cfg.get("max_location_attempts", 50))

    for _cp_idx in range(cp_times):
        patch_path = Path(source_tumors[int(rng.integers(0, len(source_tumors)))])
        try:
            cropped_data, cropped_seg = _load_patch_from_file(patch_path)
        except Exception as e:
            info["attempts"].append({
                "applied": False,
                "reason": f"load_failed: {repr(e)}",
                "patch_file": str(patch_path),
            })
            continue

        if cropped_data.shape[0] != target_data.shape[0]:
            info["attempts"].append({
                "applied": False,
                "reason": "channel_mismatch",
                "patch_file": str(patch_path),
                "patch_channels": int(cropped_data.shape[0]),
                "target_channels": int(target_data.shape[0]),
            })
            continue

        patch_shape_before = tuple(int(x) for x in cropped_seg.shape)
        cropped_data, cropped_seg = _prepare_source_tumor_patch(
            cropped_data,
            cropped_seg,
            target_max_shape=target_max_shape,
        )

        if not np.any(cropped_seg > 0):
            info["attempts"].append({
                "applied": False,
                "reason": "empty_source_tumor",
                "patch_file": str(patch_path),
                "patch_shape_before": patch_shape_before,
                "patch_shape_after": tuple(int(x) for x in cropped_seg.shape),
            })
            continue

        cropped_data, cropped_seg, trans_info = _apply_tumorcp_transforms(
            cropped_data,
            cropped_seg,
            cp_configs=cfg,
            rng=rng,
        )

        cropped_seg = np.rint(cropped_seg).astype(target_seg.dtype, copy=False)
        cropped_seg[cropped_seg < 0] = 0

        if not np.any(cropped_seg > 0):
            info["attempts"].append({
                "applied": False,
                "reason": "empty_after_transform",
                "patch_file": str(patch_path),
                "patch_shape_before": patch_shape_before,
                "patch_shape_after": tuple(int(x) for x in cropped_seg.shape),
                "transformed": trans_info.get("transformed", []),
            })
            continue

        patch_shape_after = tuple(int(x) for x in cropped_seg.shape)

        for _loc_idx in range(max_location_attempts):
            if center_probability_map is not None:
                center = sample_paste_center(center_probability_map, rng=rng)
                sampler = "braincp_occurrence"
            else:
                center = target_brain_positions[int(rng.integers(0, len(target_brain_positions)))]
                sampler = "random_fallback"

            target_data, target_seg, paste_info = paste_to(
                center=center,
                data_patch=cropped_data,
                seg_patch=cropped_seg,
                tgt_data=target_data,
                tgt_seg=target_seg,
                cp_configs=cfg,
                brain_mask=target_brain_mask,
                return_info=True,
            )

            attempt_info = {
                **paste_info,
                "patch_file": str(patch_path),
                "patch_shape_before": patch_shape_before,
                "patch_shape_after": patch_shape_after,
                "transformed": trans_info.get("transformed", []),
                "sampler": sampler,
                "alpha": alpha_value,
            }
            info["attempts"].append(attempt_info)

            if paste_info.get("applied", False):
                info.update({
                    "applied": True,
                    "reason": "success",
                    "patch_file": str(patch_path),
                    "center": paste_info.get("center"),
                    "paste_pos": paste_info.get("center"),
                    "patch_shape_before": patch_shape_before,
                    "patch_shape_after": patch_shape_after,
                    "transformed": trans_info.get("transformed", []),
                    "inside_fraction": paste_info.get("inside_fraction"),
                    "overlap_fraction": paste_info.get("overlap_fraction"),
                    "pasted_voxels": paste_info.get("pasted_voxels"),
                    "sampler": sampler,
                    "alpha": alpha_value,
                })
                return target_data, target_seg, info

    if len(info["attempts"]) > 0:
        info["reason"] = info["attempts"][-1].get("reason", "all_attempts_failed")
    else:
        info["reason"] = "all_attempts_failed"

    return target_data, target_seg, info


# Backward-compatible alias. This lets older tests call aug_one_pair().
aug_one_pair = aug_one_pair_braincp


# -------------------------------------------------------------------------
# Public nnU-Net v2 entry points
# -------------------------------------------------------------------------

def apply_braincp(
    data: np.ndarray,
    target: Optional[Any] = None,
    tumor_pool_dir: Optional[PathLike] = None,
    patch_pool_dir: Optional[PathLike] = None,
    pool_dir: Optional[PathLike] = None,
    patch_pool: Optional[PathLike] = None,
    tumor_patch_pool: Optional[PathLike] = None,
    patch_files: Optional[Sequence[PathLike]] = None,
    patch_probability_maps: Optional[Any] = None,
    patch_probability_map: Optional[Any] = None,
    paste_probability_maps: Optional[Any] = None,
    p: Optional[float] = None,
    alpha: Optional[float] = None,
    rng: Optional[Union[np.random.Generator, np.random.RandomState, int]] = None,
    cp_configs: Optional[Dict[str, Any]] = None,
    transform_cfg: Optional[Dict[str, Any]] = None,
    max_paste_per_sample: Optional[int] = None,
    min_inside_fraction: Optional[float] = None,
    min_brain_fraction: Optional[float] = None,
    max_existing_tumor_overlap_fraction: Optional[float] = None,
    max_location_attempts: Optional[int] = None,
    brain_mask_mode: Optional[str] = None,
    require_probability_map: Optional[bool] = None,
    require_full_patch_inside: Optional[bool] = None,
    verbose: bool = False,
    return_info: bool = False,
    return_stats: Optional[bool] = None,
    **kwargs: Any,
) -> Union[Tuple[np.ndarray, Any], Tuple[np.ndarray, Any, Dict[str, Any]]]:
    """
    Apply BRAIN-CP augmentation to BraTS data.

    Expected dataloader usage:
        aug_data, aug_seg, info = apply_braincp(
            data=data_cropped,
            target=seg_cropped,
            patch_files=tumor_patch_files,
            patch_probability_map=patch_prob,
            alpha=0.8,
            return_info=True,
        )

    patch_probability_map is already aligned to the current nnU-Net training patch.
    Coordinate-space crop is handled outside this file by nnunet_dataloader_braincp.py.
    """
    if target is None:
        for key in ["seg", "label", "labels", "mask", "segmentation", "target_high"]:
            if key in kwargs:
                target = kwargs[key]
                break

    if target is None:
        raise TypeError(
            "apply_braincp() missing target/seg argument. "
            "Pass segmentation as target=... or seg=..."
        )

    if patch_probability_maps is None:
        patch_probability_maps = patch_probability_map
    if patch_probability_maps is None:
        patch_probability_maps = paste_probability_maps
    if patch_probability_maps is None:
        for key in ["patch_prob", "patch_probs", "patch_probability", "paste_probability_map"]:
            if key in kwargs:
                patch_probability_maps = kwargs[key]
                break

    rng = _as_rng(rng)

    cfg = _merge_cfg(cp_configs or transform_cfg)
    if p is not None:
        cfg["p_cp"] = float(p)
    if alpha is not None:
        cfg["alpha"] = float(alpha)
    if max_paste_per_sample is not None:
        cfg["cp_times"] = int(max_paste_per_sample)
    if min_inside_fraction is not None:
        cfg["min_inside_fraction"] = float(min_inside_fraction)
    elif min_brain_fraction is not None:
        cfg["min_inside_fraction"] = float(min_brain_fraction)
    if max_existing_tumor_overlap_fraction is not None:
        cfg["max_existing_tumor_overlap_fraction"] = float(max_existing_tumor_overlap_fraction)
    if max_location_attempts is not None:
        cfg["max_location_attempts"] = int(max_location_attempts)
    if brain_mask_mode is not None:
        cfg["brain_mask_mode"] = str(brain_mask_mode)
    if require_probability_map is not None:
        cfg["require_probability_map"] = bool(require_probability_map)
    if require_full_patch_inside is not None:
        cfg["require_full_patch_inside"] = bool(require_full_patch_inside)

    pool = (
        tumor_pool_dir
        or patch_pool_dir
        or pool_dir
        or patch_pool
        or tumor_patch_pool
        or kwargs.get("tumor_patch_pool")
        or kwargs.get("patch_pool")
    )

    if patch_files is None:
        if pool is None:
            raise ValueError(
                "tumor_pool_dir/patch_pool_dir/pool_dir/patch_pool or patch_files must be provided."
            )
        patch_files = _discover_patch_files(pool)
    else:
        patch_files = [Path(x) for x in patch_files]

    if len(patch_files) == 0:
        raise ValueError("patch_files is empty.")

    data_arr = np.asarray(data)
    high_target, original_target_container, target_index = _get_highres_target(target)
    target_arr = np.asarray(high_target)

    is_batched = data_arr.ndim == 5
    if is_batched:
        batch_size = data_arr.shape[0]
    elif data_arr.ndim == 4:
        batch_size = 1
    else:
        raise ValueError(f"Unsupported data shape: {data_arr.shape}")

    stats: Dict[str, Any] = {
        "num_samples": int(batch_size),
        "num_attempted": 0,
        "num_applied": 0,
        "applied": False,
        "reason": "not_attempted",
        "patch_pool_size": int(len(patch_files)),
        "used_patch_files": [],
        "patch_file": None,
        "paste_pos": None,
        "center": None,
        "patch_shape_before": None,
        "patch_shape_after": None,
        "transformed": [],
        "sample_infos": [],
        "alpha": float(cfg.get("alpha", 0.8)),
        "sampler": "braincp_occurrence",
    }

    for b in range(batch_size):
        if not bool(cfg.get("do_cp", True)):
            sample_info = {"sample_index": int(b), "applied": False, "reason": "do_cp_false"}
            stats["sample_infos"].append(sample_info)
            continue

        if rng.random() > float(cfg.get("p_cp", 0.8)):
            sample_info = {"sample_index": int(b), "applied": False, "reason": "p_cp_skip"}
            stats["sample_infos"].append(sample_info)
            continue

        image_4d = _image_view_4d(data_arr, batch_index=b if is_batched else None)
        seg_3d = _seg_view_3d(target_arr, batch_index=b if is_batched else None)

        try:
            target_brain_positions, target_brain_mask = get_target_brain_positions(
                target_data=image_4d,
                target_seg=seg_3d,
                mode=str(cfg.get("brain_mask_mode", "seg_valid")),
            )
        except Exception as e:
            sample_info = {
                "sample_index": int(b),
                "applied": False,
                "reason": f"target_brain_positions_failed: {repr(e)}",
            }
            stats["sample_infos"].append(sample_info)
            continue

        try:
            sample_patch_prob = _select_patch_probability_map(
                patch_probability_maps=patch_probability_maps,
                batch_index=b,
                is_batched=is_batched,
                batch_size=batch_size,
                target_shape=seg_3d.shape,
            )
        except Exception as e:
            sample_info = {
                "sample_index": int(b),
                "applied": False,
                "reason": f"patch_probability_map_failed: {repr(e)}",
            }
            stats["sample_infos"].append(sample_info)
            continue

        stats["num_attempted"] += 1

        aug_img, aug_seg, sample_info = aug_one_pair_braincp(
            source_tumors=patch_files,
            target_data=image_4d,
            target_seg=seg_3d,
            patch_probability_map=sample_patch_prob,
            target_brain_positions=target_brain_positions,
            cp_configs=cfg,
            rng=rng,
            target_brain_mask=target_brain_mask,
            target_max_shape=image_4d.shape[-3:],
            alpha=float(cfg.get("alpha", 0.8)),
        )

        if is_batched:
            data_arr[b] = aug_img
            if target_arr.ndim == 5:
                target_arr[b, 0] = aug_seg
            elif target_arr.ndim == 4:
                target_arr[b] = aug_seg
        else:
            data_arr[...] = aug_img
            if target_arr.ndim == 4:
                target_arr[0] = aug_seg
            elif target_arr.ndim == 3:
                target_arr[...] = aug_seg

        sample_info["sample_index"] = int(b)
        stats["sample_infos"].append(sample_info)

        if sample_info.get("applied", False):
            stats["num_applied"] += 1
            stats["applied"] = True
            stats["reason"] = "success"
            stats["used_patch_files"].append(str(sample_info.get("patch_file")))
            stats["patch_file"] = sample_info.get("patch_file")
            stats["paste_pos"] = sample_info.get("paste_pos")
            stats["center"] = sample_info.get("center")
            stats["patch_shape_before"] = sample_info.get("patch_shape_before")
            stats["patch_shape_after"] = sample_info.get("patch_shape_after")
            stats["transformed"] = sample_info.get("transformed", [])
            stats["sampler"] = sample_info.get("sampler", "braincp_occurrence")

            if verbose:
                print(
                    f"[BRAIN-CP] pasted patch: {Path(str(stats['patch_file'])).name}, "
                    f"center={stats['center']}, alpha={stats['alpha']}, "
                    f"transformed={stats['transformed']}"
                )
        else:
            stats["reason"] = sample_info.get("reason", "not_applied")

    updated_target = _set_highres_target(original_target_container, target_arr, target_index)

    if return_stats is not None:
        return_info = bool(return_stats)

    if return_info:
        return data_arr, updated_target, stats

    return data_arr, updated_target


# Backward-compatible alias for experiments/scripts that still call apply_tumorcp().
apply_tumorcp = apply_braincp


class BraTSBrainCPAugmenter:
    """
    Callable wrapper for nnU-Net v2 trainer/dataloader usage.

    Supported call patterns:
        augmenter(data, target, patch_probability_map=patch_prob, return_info=True)
        augmenter(data_dict)
        augmenter(**data_dict)
    """

    def __init__(
        self,
        tumor_pool_dir: Optional[PathLike] = None,
        patch_pool_dir: Optional[PathLike] = None,
        pool_dir: Optional[PathLike] = None,
        patch_pool: Optional[PathLike] = None,
        tumor_patch_pool: Optional[PathLike] = None,
        p: float = 0.8,
        alpha: float = 0.8,
        cp_configs: Optional[Dict[str, Any]] = None,
        transform_cfg: Optional[Dict[str, Any]] = None,
        max_paste_per_sample: int = 1,
        min_inside_fraction: Optional[float] = None,
        min_brain_fraction: Optional[float] = None,
        max_existing_tumor_overlap_fraction: float = 0.10,
        max_location_attempts: int = 50,
        brain_mask_mode: str = "seg_valid",
        require_probability_map: bool = False,
        require_full_patch_inside: bool = True,
        seed: Optional[int] = None,
        verbose: bool = False,
        debug: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        self.pool_dir = (
            tumor_pool_dir
            or patch_pool_dir
            or pool_dir
            or patch_pool
            or tumor_patch_pool
            or kwargs.get("tumor_patch_pool")
            or kwargs.get("patch_pool")
        )

        if self.pool_dir is None:
            raise ValueError(
                "BraTSBrainCPAugmenter requires tumor_pool_dir, patch_pool_dir, pool_dir, or patch_pool."
            )

        self.cp_configs = _merge_cfg(cp_configs or transform_cfg)
        self.cp_configs["p_cp"] = float(p)
        self.cp_configs["alpha"] = float(alpha)
        self.cp_configs["cp_times"] = int(max_paste_per_sample)

        if min_inside_fraction is not None:
            self.cp_configs["min_inside_fraction"] = float(min_inside_fraction)
        elif min_brain_fraction is not None:
            self.cp_configs["min_inside_fraction"] = float(min_brain_fraction)
        else:
            self.cp_configs["min_inside_fraction"] = float(self.cp_configs.get("min_inside_fraction", 1.0))

        self.cp_configs["max_existing_tumor_overlap_fraction"] = float(max_existing_tumor_overlap_fraction)
        self.cp_configs["max_location_attempts"] = int(max_location_attempts)
        self.cp_configs["brain_mask_mode"] = str(brain_mask_mode)
        self.cp_configs["require_probability_map"] = bool(require_probability_map)
        self.cp_configs["require_full_patch_inside"] = bool(require_full_patch_inside)

        self.rng = _as_rng(seed)
        self.verbose = bool(verbose if debug is None else debug)

        self.patch_files = _discover_patch_files(self.pool_dir)

        self.num_calls = 0
        self.num_attempted = 0
        self.num_applied = 0

        if self.verbose:
            print(
                f"BRAIN-CP initialized: pool_dir={self.pool_dir}, "
                f"patches={len(self.patch_files)}, p={self.cp_configs['p_cp']}, "
                f"alpha={self.cp_configs['alpha']}, "
                f"cp_times={self.cp_configs['cp_times']}"
            )
        else:
            print("BRAIN-CP initialized")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.num_calls += 1

        return_info = bool(kwargs.pop("return_info", kwargs.pop("return_stats", True)))

        # Pattern 1: augmenter(data_dict)
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            data_dict = args[0]
            patch_probability_maps = (
                data_dict.get("patch_probability_maps")
                or data_dict.get("patch_probability_map")
                or data_dict.get("patch_prob")
                or data_dict.get("paste_probability_map")
            )
            data, target, info = apply_braincp(
                data=data_dict["data"],
                target=data_dict["target"],
                patch_files=self.patch_files,
                patch_probability_maps=patch_probability_maps,
                rng=self.rng,
                cp_configs=self.cp_configs,
                verbose=self.verbose,
                return_info=True,
            )
            data_dict["data"] = data
            data_dict["target"] = target
            self._update_stats(info)
            if return_info:
                data_dict["braincp_info"] = info
            return data_dict

        # Pattern 2: augmenter(**data_dict)
        if "data" in kwargs and "target" in kwargs:
            patch_probability_maps = (
                kwargs.get("patch_probability_maps")
                or kwargs.get("patch_probability_map")
                or kwargs.get("patch_prob")
                or kwargs.get("paste_probability_map")
            )
            data, target, info = apply_braincp(
                data=kwargs["data"],
                target=kwargs["target"],
                patch_files=self.patch_files,
                patch_probability_maps=patch_probability_maps,
                rng=self.rng,
                cp_configs=self.cp_configs,
                verbose=self.verbose,
                return_info=True,
            )
            kwargs["data"] = data
            kwargs["target"] = target
            self._update_stats(info)
            if return_info:
                kwargs["braincp_info"] = info
            return kwargs

        # Pattern 3: augmenter(data, target)
        if len(args) >= 2:
            patch_probability_maps = kwargs.pop(
                "patch_probability_maps",
                kwargs.pop("patch_probability_map", kwargs.pop("patch_prob", None)),
            )
            data, target, info = apply_braincp(
                data=args[0],
                target=args[1],
                patch_files=self.patch_files,
                patch_probability_maps=patch_probability_maps,
                rng=self.rng,
                cp_configs=self.cp_configs,
                verbose=self.verbose,
                return_info=True,
            )
            self._update_stats(info)
            if return_info:
                return data, target, info
            return data, target

        raise TypeError(
            "Unsupported BraTSBrainCPAugmenter call pattern. "
            "Use augmenter(data, target), augmenter(data_dict), or augmenter(**data_dict)."
        )

    def _update_stats(self, info: Dict[str, Any]) -> None:
        self.num_attempted += int(info.get("num_attempted", 0))
        self.num_applied += int(info.get("num_applied", 0))

    def get_stats(self) -> Dict[str, int]:
        return {
            "num_calls": int(self.num_calls),
            "num_attempted": int(self.num_attempted),
            "num_applied": int(self.num_applied),
            "patch_pool_size": int(len(self.patch_files)),
        }

    def reset_stats(self) -> None:
        self.num_calls = 0
        self.num_attempted = 0
        self.num_applied = 0


# Compatibility alias.
BraTSTumorCPAugmenter = BraTSBrainCPAugmenter
