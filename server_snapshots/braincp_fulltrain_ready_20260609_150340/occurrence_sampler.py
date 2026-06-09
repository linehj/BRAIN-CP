"""
occurrence_sampler.py

BRAIN-CP paste center sampler.

Responsibilities:
- Load raw-space tumor occurrence probability map
- Crop it to each target case using nnU-Net bbox_used_for_cropping
- Build alpha-mixed paste probability map
- Sample paste center in NumPy index order: (i, j, k)

Formula:
    P_paste = alpha * P_occurrence + (1 - alpha) * P_uniform

Notes:
- P_uniform means random sampling over a valid brain/foreground region, not over
  the entire invalid outside-brain volume.
- This file does not transform or paste tumor patches. Transformation and label
  update belong in braincp_augmentation.py.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

Center = Tuple[int, int, int]
BBox = Sequence[Sequence[int]]
DEFAULT_ALPHA = 0.8


# -----------------------------------------------------------------------------
# Basic probability utilities
# -----------------------------------------------------------------------------


def load_probability_map(prob_map_path: str | Path) -> np.ndarray:
    prob_map_path = Path(prob_map_path)
    if not prob_map_path.exists():
        raise FileNotFoundError(f"Probability map not found: {prob_map_path}")
    prob_map = np.load(prob_map_path)
    if prob_map.ndim != 3:
        raise ValueError(f"Probability map must be 3D, got {prob_map.shape}")
    return prob_map


def validate_probability_map(
    prob_map: np.ndarray,
    expected_shape: Optional[Sequence[int]] = None,
    require_sum_one: bool = True,
    sum_atol: float = 1e-4,
) -> None:
    if prob_map.ndim != 3:
        raise ValueError(f"prob_map must be 3D, got {prob_map.ndim}D")
    if expected_shape is not None and tuple(prob_map.shape) != tuple(expected_shape):
        raise ValueError(
            f"Unexpected probability map shape. Expected {tuple(expected_shape)}, "
            f"got {prob_map.shape}"
        )
    if not np.all(np.isfinite(prob_map)):
        raise ValueError("Probability map contains NaN or Inf values.")
    if np.any(prob_map < 0):
        raise ValueError("Probability map contains negative values.")
    total = float(prob_map.sum())
    if total <= 0:
        raise ValueError(f"Probability map sum must be > 0, got {total}")
    if require_sum_one and not np.isclose(total, 1.0, atol=sum_atol):
        raise ValueError(f"Probability map sum must be approximately 1. Got {total:.8f}")


def validate_alpha(alpha: float) -> None:
    if not np.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")


def normalize_probability(prob: np.ndarray) -> np.ndarray:
    prob = prob.astype(np.float64, copy=False)
    if not np.all(np.isfinite(prob)):
        raise ValueError("Probability array contains NaN or Inf values.")
    if np.any(prob < 0):
        raise ValueError("Probability array contains negative values.")
    total = float(prob.sum())
    if total <= 0:
        raise ValueError("Probability sum must be > 0.")
    return prob / total


def summarize_probability_map(prob_map: np.ndarray) -> dict:
    validate_probability_map(prob_map, require_sum_one=False)
    positive = prob_map[prob_map > 0]
    max_index = np.unravel_index(np.argmax(prob_map), prob_map.shape)
    return {
        "shape": tuple(prob_map.shape),
        "dtype": str(prob_map.dtype),
        "sum": float(prob_map.sum()),
        "min": float(prob_map.min()),
        "max": float(prob_map.max()),
        "positive_voxels": int(positive.size),
        "max_index": tuple(int(x) for x in max_index),
        "max_probability": float(prob_map[max_index]),
    }


def _get_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


# -----------------------------------------------------------------------------
# nnU-Net preprocessed case utilities
# -----------------------------------------------------------------------------


def load_case_properties(pkl_path: str | Path) -> dict:
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Case .pkl not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_bbox_used_for_cropping(props: dict) -> BBox:
    if "bbox_used_for_cropping" not in props:
        raise KeyError("props does not contain 'bbox_used_for_cropping'.")
    bbox = props["bbox_used_for_cropping"]
    if len(bbox) != 3:
        raise ValueError(f"bbox must have 3 axes, got {bbox}")
    for axis_bbox in bbox:
        if len(axis_bbox) != 2:
            raise ValueError(f"Each bbox axis must be [start, end], got {bbox}")
        if int(axis_bbox[0]) < 0:
            raise ValueError(f"bbox start must be >= 0, got {bbox}")
        if int(axis_bbox[1]) <= int(axis_bbox[0]):
            raise ValueError(f"bbox end must be > start, got {bbox}")
    return bbox


def crop_probability_map_to_bbox(raw_prob_map: np.ndarray, bbox_used_for_cropping: BBox) -> np.ndarray:
    validate_probability_map(raw_prob_map, require_sum_one=False)
    b = bbox_used_for_cropping
    cropped = raw_prob_map[
        int(b[0][0]): int(b[0][1]),
        int(b[1][0]): int(b[1][1]),
        int(b[2][0]): int(b[2][1]),
    ]
    if cropped.ndim != 3 or cropped.size == 0:
        raise RuntimeError(f"Invalid cropped probability map shape: {cropped.shape}")
    return cropped.astype(np.float64, copy=False)


def make_case_occurrence_probability_map(
    raw_prob_map: np.ndarray,
    pkl_path: str | Path,
    expected_shape: Optional[Sequence[int]] = None,
) -> np.ndarray:
    props = load_case_properties(pkl_path)
    bbox = get_bbox_used_for_cropping(props)
    cropped = crop_probability_map_to_bbox(raw_prob_map, bbox)
    if expected_shape is not None and tuple(cropped.shape) != tuple(expected_shape):
        raise ValueError(
            f"Cropped probability shape mismatch. Expected {tuple(expected_shape)}, "
            f"got {cropped.shape}"
        )
    return normalize_probability(cropped)


def load_b2nd(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f".b2nd file not found: {path}")
    try:
        import blosc2
    except Exception as e:
        raise ImportError("blosc2 is required. Activate the nnunet conda environment.") from e
    obj = blosc2.open(urlpath=str(path), mode="r")
    return np.asarray(obj[:])


def squeeze_seg(seg: np.ndarray) -> np.ndarray:
    if seg.ndim == 4 and seg.shape[0] == 1:
        return seg[0]
    if seg.ndim == 3:
        return seg
    raise ValueError(f"Unsupported seg shape: {seg.shape}")


def make_brain_mask_from_data_and_seg(
    data: Optional[np.ndarray] = None,
    seg: Optional[np.ndarray] = None,
    use_image_nonzero: bool = True,
) -> np.ndarray:
    """
    Recommended first implementation:
        brain_mask = (seg != -1) & np.any(data != 0, axis=0)

    If data is unavailable, use seg != -1.
    If seg is unavailable, use image nonzero mask.
    """
    if data is None and seg is None:
        raise ValueError("At least one of data or seg must be provided.")

    brain_mask = None

    if seg is not None:
        seg3d = squeeze_seg(seg)
        brain_mask = seg3d != -1

    if data is not None and use_image_nonzero:
        if data.ndim != 4:
            raise ValueError(f"data must have shape (C, I, J, K), got {data.shape}")
        image_mask = np.any(data != 0, axis=0)
        if brain_mask is None:
            brain_mask = image_mask
        else:
            if brain_mask.shape != image_mask.shape:
                raise ValueError(
                    f"seg/data shape mismatch: seg mask {brain_mask.shape}, "
                    f"image mask {image_mask.shape}"
                )
            brain_mask = brain_mask & image_mask

    if brain_mask is None or int(brain_mask.sum()) == 0:
        raise ValueError("Brain mask has no valid voxels.")
    return brain_mask.astype(bool)


# -----------------------------------------------------------------------------
# Mixed probability construction
# -----------------------------------------------------------------------------


def make_uniform_probability_map(shape: Sequence[int], valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    shape = tuple(shape)
    if valid_mask is None:
        uniform = np.ones(shape, dtype=np.float64)
    else:
        if valid_mask.shape != shape:
            raise ValueError(f"valid_mask shape must be {shape}, got {valid_mask.shape}")
        uniform = (valid_mask > 0).astype(np.float64)
    return normalize_probability(uniform)


def make_default_valid_mask(occurrence_prob_map: np.ndarray, uniform_region: str = "positive") -> np.ndarray:
    if uniform_region in ("positive", "nonzero"):
        valid_mask = occurrence_prob_map > 0
    elif uniform_region == "all":
        valid_mask = np.ones_like(occurrence_prob_map, dtype=bool)
    else:
        raise ValueError(f"Unknown uniform_region: {uniform_region}")
    if int(valid_mask.sum()) == 0:
        raise ValueError("Default valid mask has no valid voxels.")
    return valid_mask


def make_mixed_probability_map(
    occurrence_prob_map: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    valid_mask: Optional[np.ndarray] = None,
    uniform_region: str = "positive",
) -> np.ndarray:
    validate_alpha(alpha)
    validate_probability_map(occurrence_prob_map, require_sum_one=False)

    occurrence_prob = normalize_probability(occurrence_prob_map)

    if valid_mask is None:
        valid_mask = make_default_valid_mask(occurrence_prob, uniform_region=uniform_region)
    else:
        if valid_mask.shape != occurrence_prob.shape:
            raise ValueError(
                f"valid_mask shape must match probability map shape. "
                f"Expected {occurrence_prob.shape}, got {valid_mask.shape}"
            )
        valid_mask = valid_mask > 0

    if int(valid_mask.sum()) == 0:
        raise ValueError("valid_mask has no valid voxels.")

    occurrence_restricted = occurrence_prob * valid_mask.astype(np.float64)
    if float(occurrence_restricted.sum()) > 0:
        occurrence_restricted = normalize_probability(occurrence_restricted)
    else:
        occurrence_restricted = make_uniform_probability_map(occurrence_prob.shape, valid_mask)

    uniform_prob = make_uniform_probability_map(occurrence_prob.shape, valid_mask)

    mixed = alpha * occurrence_restricted + (1.0 - alpha) * uniform_prob
    return normalize_probability(mixed)


# -----------------------------------------------------------------------------

def crop_and_pad_probability_map_to_patch(
    case_prob_map: np.ndarray,
    patch_bbox: BBox,
    patch_shape: Optional[Sequence[int]] = None,
    pad_value: float = 0.0,
    normalize: bool = True,
    fallback_to_uniform: bool = True,
) -> np.ndarray:
    """
    Crop a case-level probability map to the current nnU-Net training patch.

    case_prob_map:
        Probability map already matched to one preprocessed case.

    patch_bbox:
        Current nnU-Net training patch bbox in case coordinates.
        Format: [[i0, i1], [j0, j1], [k0, k1]]

    patch_shape:
        Output patch shape. If None, inferred from patch_bbox.

    Why crop + pad:
        nnU-Net can sample a patch bbox that slightly goes outside the case.
        data/seg are padded in that situation, so probability map must be
        padded in the same way.
    """
    validate_probability_map(case_prob_map, require_sum_one=False)

    if len(patch_bbox) != 3:
        raise ValueError(f"patch_bbox must have 3 axes, got {patch_bbox}")

    bbox = [[int(v[0]), int(v[1])] for v in patch_bbox]

    if patch_shape is None:
        patch_shape = tuple(int(b[1] - b[0]) for b in bbox)
    else:
        patch_shape = tuple(int(x) for x in patch_shape)

    if len(patch_shape) != 3:
        raise ValueError(f"patch_shape must have length 3, got {patch_shape}")

    if any(s <= 0 for s in patch_shape):
        raise ValueError(f"patch_shape must be positive, got {patch_shape}")

    patch_prob = np.full(
        patch_shape,
        fill_value=float(pad_value),
        dtype=np.float64,
    )

    src_slices = []
    dst_slices = []

    for axis in range(3):
        lb, ub = bbox[axis]
        dim = case_prob_map.shape[axis]

        src_start = max(lb, 0)
        src_end = min(ub, dim)

        dst_start = max(0, -lb)
        dst_end = dst_start + max(0, src_end - src_start)

        if src_end <= src_start:
            if normalize and fallback_to_uniform:
                uniform = np.ones(patch_shape, dtype=np.float64)
                return normalize_probability(uniform)

            raise ValueError(
                f"patch_bbox has no overlap with case_prob_map on axis {axis}. "
                f"bbox={patch_bbox}, case_shape={case_prob_map.shape}"
            )

        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    patch_prob[tuple(dst_slices)] = case_prob_map[tuple(src_slices)]

    if normalize:
        total = float(patch_prob.sum())

        if total <= 0:
            if fallback_to_uniform:
                uniform = np.ones(patch_shape, dtype=np.float64)
                return normalize_probability(uniform)

            raise ValueError("Cropped patch probability map has zero mass.")

        patch_prob = patch_prob / total

    return patch_prob


def make_patch_occurrence_probability_map(
    raw_prob_map: np.ndarray,
    case_pkl_path: str | Path,
    patch_bbox: BBox,
    patch_shape: Optional[Sequence[int]] = None,
    expected_case_shape: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Create probability map aligned with the current nnU-Net training patch.

    Flow:
        raw probability map
        -> crop by case bbox_used_for_cropping
        -> case probability map
        -> crop + pad by current training patch bbox
        -> patch probability map
    """
    case_prob = make_case_occurrence_probability_map(
        raw_prob_map=raw_prob_map,
        pkl_path=case_pkl_path,
        expected_shape=expected_case_shape,
    )

    patch_prob = crop_and_pad_probability_map_to_patch(
        case_prob_map=case_prob,
        patch_bbox=patch_bbox,
        patch_shape=patch_shape,
        pad_value=0.0,
        normalize=True,
        fallback_to_uniform=True,
    )

    return patch_prob



# Sampling
# -----------------------------------------------------------------------------


def _get_flat_probability(prob_map: np.ndarray) -> np.ndarray:
    validate_probability_map(prob_map, require_sum_one=False)
    return normalize_probability(prob_map.reshape(-1).astype(np.float64))


def sample_paste_center(prob_map: np.ndarray, rng: Optional[np.random.Generator] = None) -> Center:
    rng = _get_rng(rng)
    flat_prob = _get_flat_probability(prob_map)
    flat_index = rng.choice(flat_prob.size, p=flat_prob)
    center = np.unravel_index(flat_index, prob_map.shape)
    return int(center[0]), int(center[1]), int(center[2])


def sample_paste_centers(
    prob_map: np.ndarray,
    num_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")
    rng = _get_rng(rng)
    flat_prob = _get_flat_probability(prob_map)
    flat_indices = rng.choice(flat_prob.size, size=num_samples, p=flat_prob)
    centers = np.array(np.unravel_index(flat_indices, prob_map.shape)).T
    return centers.astype(int)


def sample_paste_center_for_case(
    raw_occurrence_prob_map: np.ndarray,
    case_pkl_path: str | Path,
    alpha: float = DEFAULT_ALPHA,
    valid_mask: Optional[np.ndarray] = None,
    uniform_region: str = "positive",
    expected_shape: Optional[Sequence[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Center:
    case_prob = make_case_occurrence_probability_map(
        raw_prob_map=raw_occurrence_prob_map,
        pkl_path=case_pkl_path,
        expected_shape=expected_shape,
    )
    mixed_prob = make_mixed_probability_map(
        occurrence_prob_map=case_prob,
        alpha=alpha,
        valid_mask=valid_mask,
        uniform_region=uniform_region,
    )
    return sample_paste_center(mixed_prob, rng=rng)


def sample_paste_centers_for_case(
    raw_occurrence_prob_map: np.ndarray,
    case_pkl_path: str | Path,
    num_samples: int,
    alpha: float = DEFAULT_ALPHA,
    valid_mask: Optional[np.ndarray] = None,
    uniform_region: str = "positive",
    expected_shape: Optional[Sequence[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    case_prob = make_case_occurrence_probability_map(
        raw_prob_map=raw_occurrence_prob_map,
        pkl_path=case_pkl_path,
        expected_shape=expected_shape,
    )
    mixed_prob = make_mixed_probability_map(
        occurrence_prob_map=case_prob,
        alpha=alpha,
        valid_mask=valid_mask,
        uniform_region=uniform_region,
    )
    return sample_paste_centers(mixed_prob, num_samples=num_samples, rng=rng)


# -----------------------------------------------------------------------------
# Center and patch validity helpers
# -----------------------------------------------------------------------------


def is_valid_center(
    center: Sequence[int],
    brain_mask: Optional[np.ndarray] = None,
    volume_shape: Optional[Sequence[int]] = None,
) -> bool:
    if len(center) != 3:
        return False
    i, j, k = int(center[0]), int(center[1]), int(center[2])

    if volume_shape is not None:
        si, sj, sk = tuple(volume_shape)
        if not (0 <= i < si and 0 <= j < sj and 0 <= k < sk):
            return False

    if brain_mask is not None:
        if brain_mask.ndim != 3:
            raise ValueError("brain_mask must be 3D.")
        if not (0 <= i < brain_mask.shape[0]):
            return False
        if not (0 <= j < brain_mask.shape[1]):
            return False
        if not (0 <= k < brain_mask.shape[2]):
            return False
        return bool(brain_mask[i, j, k] > 0)

    return True


def get_patch_bounds_from_center(
    center: Sequence[int],
    patch_shape: Sequence[int],
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    if len(center) != 3:
        raise ValueError(f"center must have length 3, got {center}")
    if len(patch_shape) != 3:
        raise ValueError(f"patch_shape must have length 3, got {patch_shape}")

    c = np.asarray(center, dtype=int)
    s = np.asarray(patch_shape, dtype=int)
    if np.any(s <= 0):
        raise ValueError(f"patch_shape must be positive, got {patch_shape}")

    start = c - (s // 2)
    end = start + s
    return tuple(int(x) for x in start), tuple(int(x) for x in end)


def is_patch_box_inside_volume(
    center: Sequence[int],
    patch_shape: Sequence[int],
    volume_shape: Sequence[int],
) -> bool:
    start, end = get_patch_bounds_from_center(center=center, patch_shape=patch_shape)
    for axis in range(3):
        if start[axis] < 0:
            return False
        if end[axis] > int(volume_shape[axis]):
            return False
    return True


def is_patch_tumor_inside_brain(
    center: Sequence[int],
    patch_seg: np.ndarray,
    brain_mask: np.ndarray,
) -> bool:
    """
    Check whether actual tumor voxels of a transformed patch are inside brain.

    This helper should be called from braincp_augmentation.py after patch
    transformation, because only then is the final patch_seg known.
    """
    if patch_seg.ndim != 3:
        raise ValueError(f"patch_seg must be 3D, got {patch_seg.shape}")
    if brain_mask.ndim != 3:
        raise ValueError(f"brain_mask must be 3D, got {brain_mask.shape}")

    if not is_patch_box_inside_volume(center, patch_seg.shape, brain_mask.shape):
        return False

    start, end = get_patch_bounds_from_center(center, patch_seg.shape)
    i0, j0, k0 = start
    i1, j1, k1 = end

    brain_region = brain_mask[i0:i1, j0:j1, k0:k1]
    patch_tumor_mask = patch_seg > 0

    if int(patch_tumor_mask.sum()) == 0:
        return False
    if brain_region.shape != patch_tumor_mask.shape:
        return False

    return bool(np.all(brain_region[patch_tumor_mask] > 0))


def sample_center_with_patch_rejection(
    prob_map: np.ndarray,
    patch_seg: np.ndarray,
    brain_mask: np.ndarray,
    max_tries: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Center:
    """
    Sample center and reject invalid transformed patch locations.

    patch_seg should already be transformed before calling this function.
    """
    if max_tries <= 0:
        raise ValueError(f"max_tries must be > 0, got {max_tries}")
    rng = _get_rng(rng)
    for _ in range(max_tries):
        center = sample_paste_center(prob_map, rng=rng)
        if is_patch_tumor_inside_brain(center=center, patch_seg=patch_seg, brain_mask=brain_mask):
            return center
    raise RuntimeError(f"Failed to sample a patch-valid center after {max_tries} tries.")


# -----------------------------------------------------------------------------
# CLI test
# -----------------------------------------------------------------------------


def quick_case_sampling_check(
    raw_occurrence_prob_map: np.ndarray,
    prep_dir: str | Path,
    case_id: str,
    alpha: float = DEFAULT_ALPHA,
    num_samples: int = 10,
    seed: int = 2026,
    use_brain_mask: bool = True,
    uniform_region: str = "positive",
) -> None:
    validate_alpha(alpha)
    validate_probability_map(raw_occurrence_prob_map, require_sum_one=False)

    prep_dir = Path(prep_dir)
    pkl_path = prep_dir / f"{case_id}.pkl"
    data_path = prep_dir / f"{case_id}.b2nd"
    seg_path = prep_dir / f"{case_id}_seg.b2nd"

    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing pkl file: {pkl_path}")
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing seg file: {seg_path}")

    seg_raw = load_b2nd(seg_path)
    seg = squeeze_seg(seg_raw)

    data = None
    if use_brain_mask:
        if not data_path.exists():
            raise FileNotFoundError(f"Missing data file: {data_path}")
        data = load_b2nd(data_path)

    brain_mask = None
    if use_brain_mask:
        brain_mask = make_brain_mask_from_data_and_seg(data=data, seg=seg, use_image_nonzero=True)

    case_prob = make_case_occurrence_probability_map(
        raw_prob_map=raw_occurrence_prob_map,
        pkl_path=pkl_path,
        expected_shape=seg.shape,
    )

    mixed_prob = make_mixed_probability_map(
        occurrence_prob_map=case_prob,
        alpha=alpha,
        valid_mask=brain_mask,
        uniform_region=uniform_region,
    )

    rng = np.random.default_rng(seed)
    centers = sample_paste_centers(mixed_prob, num_samples=num_samples, rng=rng)

    print("\n[Case sampling check]")
    print(f"case_id: {case_id}")
    print(f"alpha: {alpha}")
    print(f"use_brain_mask: {use_brain_mask}")
    print("formula: P_paste = alpha * P_occurrence + (1 - alpha) * P_uniform")

    print("\n[Shapes]")
    print(f"raw_occurrence_prob_map shape: {raw_occurrence_prob_map.shape}")
    print(f"seg raw shape: {seg_raw.shape}")
    print(f"seg squeezed shape: {seg.shape}")
    print(f"case_prob shape: {case_prob.shape}")
    if brain_mask is not None:
        print(f"brain_mask shape: {brain_mask.shape}")
        print(f"brain_mask voxels: {int(brain_mask.sum())}")

    print("\n[Case probability summary]")
    for key, value in summarize_probability_map(case_prob).items():
        print(f"case_prob {key}: {value}")

    print("\n[Mixed probability summary]")
    for key, value in summarize_probability_map(mixed_prob).items():
        print(f"mixed_prob {key}: {value}")

    print("\n[Sampled centers]")
    invalid_center_count = 0
    for idx, (i, j, k) in enumerate(centers, start=1):
        center = (int(i), int(j), int(k))
        center_valid = is_valid_center(center, brain_mask=brain_mask, volume_shape=seg.shape)
        if not center_valid:
            invalid_center_count += 1
        print(
            f"{idx:02d}: center=({i:3d}, {j:3d}, {k:3d}), "
            f"case_occ_prob={float(case_prob[i, j, k]):.10e}, "
            f"mixed_prob={float(mixed_prob[i, j, k]):.10e}, "
            f"center_valid={center_valid}"
        )

    print("\n[Summary]")
    print(f"invalid_center_count: {invalid_center_count}")
    if invalid_center_count == 0:
        print("[OK] all sampled centers are valid.")
    else:
        print("[WARN] some sampled centers are invalid.")


def main() -> None:
    parser = argparse.ArgumentParser(description="BRAIN-CP occurrence probability map sampler.")
    parser.add_argument("--prob-map", required=True, type=str)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expected-shape", type=int, nargs=3, default=None)
    parser.add_argument(
        "--uniform-region",
        type=str,
        default="positive",
        choices=["positive", "nonzero", "all"],
    )
    parser.add_argument("--prep-dir", type=str, default=None)
    parser.add_argument("--case-id", type=str, default=None)
    parser.add_argument("--no-brain-mask", action="store_true")

    args = parser.parse_args()

    raw_prob_map = load_probability_map(args.prob_map)
    validate_probability_map(
        raw_prob_map,
        expected_shape=args.expected_shape,
        require_sum_one=True,
    )

    print("[Raw occurrence probability map summary]")
    for key, value in summarize_probability_map(raw_prob_map).items():
        print(f"{key}: {value}")

    if args.prep_dir is not None or args.case_id is not None:
        if args.prep_dir is None or args.case_id is None:
            raise ValueError("For case-specific sampling, provide both --prep-dir and --case-id.")
        quick_case_sampling_check(
            raw_occurrence_prob_map=raw_prob_map,
            prep_dir=args.prep_dir,
            case_id=args.case_id,
            alpha=args.alpha,
            num_samples=args.num_samples,
            seed=args.seed,
            use_brain_mask=not args.no_brain_mask,
            uniform_region=args.uniform_region,
        )
    else:
        rng = np.random.default_rng(args.seed)
        mixed_prob = make_mixed_probability_map(
            occurrence_prob_map=raw_prob_map,
            alpha=args.alpha,
            valid_mask=None,
            uniform_region=args.uniform_region,
        )
        centers = sample_paste_centers(mixed_prob, num_samples=args.num_samples, rng=rng)

        print("\n[Global raw-space sampling check]")
        print(f"alpha: {args.alpha}")
        print(f"uniform_region: {args.uniform_region}")

        print("\n[Mixed probability summary]")
        for key, value in summarize_probability_map(mixed_prob).items():
            print(f"{key}: {value}")

        print("\n[Sampled centers]")
        for idx, (i, j, k) in enumerate(centers, start=1):
            print(
                f"{idx:02d}: center=({i:3d}, {j:3d}, {k:3d}), "
                f"raw_occ_prob={float(raw_prob_map[i, j, k]):.10e}, "
                f"mixed_prob={float(mixed_prob[i, j, k]):.10e}"
            )


if __name__ == "__main__":
    main()
