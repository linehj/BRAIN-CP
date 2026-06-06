"""
occurrence_sampler.py

BRAIN-CP paste center sampler.

This module loads a tumor occurrence probability map and samples paste centers
using a mixed probability distribution:

    P_paste = alpha * P_occurrence + (1 - alpha) * P_uniform

Why mixing?
-----------
If alpha = 1.0:
    The sampler fully follows the tumor occurrence probability map.
    This may over-focus on high-occurrence anatomical regions.

If alpha < 1.0:
    The sampler still prefers high-occurrence regions, but sometimes samples
    from other valid regions. This preserves diversity and helps avoid ignoring
    rare tumor locations.

Coordinate convention
---------------------
center = (i, j, k)

This is the same order as a NumPy 3D array:

    prob_map[i, j, k]

For BraTS labels with shape (182, 218, 182):

    i: axis 0
    j: axis 1
    k: axis 2

We intentionally avoid calling these x/y/z because medical image orientation
can be confusing. For implementation, NumPy index order is safer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


Center = Tuple[int, int, int]


def load_probability_map(prob_map_path: str | Path) -> np.ndarray:
    """
    Load probability map from .npy file.

    Parameters
    ----------
    prob_map_path:
        Path to tumor_sampling_probability_WT.npy.

    Returns
    -------
    np.ndarray
        3D probability map.
    """
    prob_map_path = Path(prob_map_path)

    if not prob_map_path.exists():
        raise FileNotFoundError(f"Probability map not found: {prob_map_path}")

    prob_map = np.load(prob_map_path)

    if prob_map.ndim != 3:
        raise ValueError(
            f"Probability map must be 3D, but got shape {prob_map.shape}"
        )

    return prob_map


def validate_probability_map(
    prob_map: np.ndarray,
    expected_shape: Optional[Sequence[int]] = None,
    sum_atol: float = 1e-4,
    require_sum_one: bool = True,
) -> None:
    """
    Validate probability map.

    Checks:
    - 3D array
    - finite values only
    - no negative values
    - optional expected shape
    - optional sum approximately 1

    Parameters
    ----------
    prob_map:
        3D probability map.
    expected_shape:
        Expected shape, for example (182, 218, 182).
    sum_atol:
        Tolerance for checking whether probability sum is 1.
    require_sum_one:
        If True, require probability map sum to be approximately 1.
    """
    if prob_map.ndim != 3:
        raise ValueError(f"prob_map must be 3D, but got {prob_map.ndim}D")

    if expected_shape is not None:
        expected_shape = tuple(expected_shape)
        if prob_map.shape != expected_shape:
            raise ValueError(
                f"Unexpected probability map shape. "
                f"Expected {expected_shape}, got {prob_map.shape}"
            )

    if not np.all(np.isfinite(prob_map)):
        raise ValueError("Probability map contains NaN or Inf values.")

    if np.any(prob_map < 0):
        raise ValueError("Probability map contains negative values.")

    prob_sum = float(prob_map.sum())

    if prob_sum <= 0:
        raise ValueError(f"Probability map sum must be > 0, but got {prob_sum}")

    if require_sum_one and not np.isclose(prob_sum, 1.0, atol=sum_atol):
        raise ValueError(
            f"Probability map sum must be approximately 1. "
            f"Got {prob_sum:.8f}"
        )


def validate_alpha(alpha: float) -> None:
    """
    Validate alpha value.

    alpha controls how much we trust the occurrence map.

    alpha = 1.0:
        pure occurrence-based sampling

    alpha = 0.0:
        pure uniform sampling over valid region
    """
    if not np.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}")

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")


def _get_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    """
    Return NumPy random generator.
    """
    if rng is None:
        return np.random.default_rng()
    return rng


def normalize_probability(prob: np.ndarray) -> np.ndarray:
    """
    Normalize non-negative array so that its sum becomes 1.
    """
    prob = prob.astype(np.float64, copy=False)

    if not np.all(np.isfinite(prob)):
        raise ValueError("Probability array contains NaN or Inf values.")

    if np.any(prob < 0):
        raise ValueError("Probability array contains negative values.")

    total = float(prob.sum())

    if total <= 0:
        raise ValueError("Probability sum must be > 0.")

    return prob / total


def make_uniform_probability_map(
    shape: Sequence[int],
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create uniform probability map over valid region.

    Parameters
    ----------
    shape:
        Shape of the output probability map.
    valid_mask:
        Optional 3D mask.
        If given, uniform probability is assigned only where valid_mask > 0.
        If None, uniform probability is assigned over the entire volume.

    Returns
    -------
    uniform_prob:
        3D probability map whose sum is 1.
    """
    shape = tuple(shape)

    if valid_mask is None:
        uniform_prob = np.ones(shape, dtype=np.float64)
    else:
        if valid_mask.shape != shape:
            raise ValueError(
                f"valid_mask shape must match probability map shape. "
                f"Expected {shape}, got {valid_mask.shape}"
            )

        uniform_prob = (valid_mask > 0).astype(np.float64)

    return normalize_probability(uniform_prob)


def make_default_valid_mask(
    occurrence_prob_map: np.ndarray,
    uniform_region: str = "positive",
) -> np.ndarray:
    """
    Create default valid mask when a brain mask is not provided.

    uniform_region options
    ----------------------
    positive:
        Use voxels where occurrence_prob_map > 0.
        This is safer than sampling over the whole volume.

    all:
        Use the whole 3D volume.
        This is not recommended for real augmentation because it can sample
        outside the brain. Use only for debugging.

    nonzero is accepted as an alias of positive.
    """
    if uniform_region in ("positive", "nonzero"):
        valid_mask = occurrence_prob_map > 0
    elif uniform_region == "all":
        valid_mask = np.ones_like(occurrence_prob_map, dtype=bool)
    else:
        raise ValueError(
            f"Unknown uniform_region: {uniform_region}. "
            f"Use 'positive' or 'all'."
        )

    if int(valid_mask.sum()) == 0:
        raise ValueError("Default valid mask has no valid voxels.")

    return valid_mask


def make_mixed_probability_map(
    occurrence_prob_map: np.ndarray,
    alpha: float = 0.8,
    valid_mask: Optional[np.ndarray] = None,
    uniform_region: str = "positive",
) -> np.ndarray:
    """
    Build final BRAIN-CP paste probability map.

    Formula
    -------
    P_paste = alpha * P_occurrence + (1 - alpha) * P_uniform

    Parameters
    ----------
    occurrence_prob_map:
        3D tumor occurrence probability map.
        Usually tumor_sampling_probability_WT.npy.
    alpha:
        Weight for occurrence-based sampling.
        alpha must be between 0 and 1.
    valid_mask:
        Optional 3D valid region mask.
        Later in braincp_augmentation.py, this should be the target brain mask.
    uniform_region:
        Used only when valid_mask is None.
        - positive: uniform over occurrence_prob_map > 0
        - all: uniform over whole volume

    Returns
    -------
    mixed_prob:
        3D probability map whose sum is 1.
    """
    validate_alpha(alpha)

    validate_probability_map(
        occurrence_prob_map,
        require_sum_one=False,
    )

    occurrence_prob = normalize_probability(occurrence_prob_map)

    if valid_mask is None:
        valid_mask = make_default_valid_mask(
            occurrence_prob,
            uniform_region=uniform_region,
        )
    else:
        if valid_mask.shape != occurrence_prob.shape:
            raise ValueError(
                f"valid_mask shape must match probability map shape. "
                f"Expected {occurrence_prob.shape}, got {valid_mask.shape}"
            )
        valid_mask = valid_mask > 0

    if int(valid_mask.sum()) == 0:
        raise ValueError("valid_mask has no valid voxels.")

    # Restrict occurrence probability to valid region.
    # Later, this prevents sampling outside the current target brain mask.
    occurrence_restricted = occurrence_prob * valid_mask.astype(np.float64)
    occurrence_restricted = normalize_probability(occurrence_restricted)

    uniform_prob = make_uniform_probability_map(
        shape=occurrence_prob.shape,
        valid_mask=valid_mask,
    )

    mixed_prob = alpha * occurrence_restricted + (1.0 - alpha) * uniform_prob
    mixed_prob = normalize_probability(mixed_prob)

    return mixed_prob


def _get_flat_probability(prob_map: np.ndarray) -> np.ndarray:
    """
    Convert 3D probability map into 1D probability vector.
    """
    validate_probability_map(prob_map, require_sum_one=False)
    flat_prob = prob_map.reshape(-1).astype(np.float64)
    flat_prob = normalize_probability(flat_prob)
    return flat_prob


def sample_paste_center(
    prob_map: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Center:
    """
    Sample one paste center from a probability map.

    The input prob_map can be:
    - pure occurrence probability map
    - mixed probability map
    - any valid 3D probability map
    """
    validate_probability_map(prob_map, require_sum_one=False)

    rng = _get_rng(rng)
    flat_prob = _get_flat_probability(prob_map)

    flat_index = rng.choice(flat_prob.size, p=flat_prob)
    center = np.unravel_index(flat_index, prob_map.shape)

    return int(center[0]), int(center[1]), int(center[2])


def sample_paste_center_from_occurrence(
    occurrence_prob_map: np.ndarray,
    alpha: float = 0.8,
    valid_mask: Optional[np.ndarray] = None,
    uniform_region: str = "positive",
    rng: Optional[np.random.Generator] = None,
) -> Center:
    """
    Sample one paste center using BRAIN-CP mixed sampling.

    This is the main function to call later from braincp_augmentation.py.
    """
    mixed_prob = make_mixed_probability_map(
        occurrence_prob_map=occurrence_prob_map,
        alpha=alpha,
        valid_mask=valid_mask,
        uniform_region=uniform_region,
    )

    return sample_paste_center(mixed_prob, rng=rng)


def sample_paste_centers(
    prob_map: np.ndarray,
    num_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample multiple paste centers from a probability map.

    Returns
    -------
    centers:
        NumPy array with shape (num_samples, 3).
        Each row is (i, j, k).
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")

    validate_probability_map(prob_map, require_sum_one=False)

    rng = _get_rng(rng)
    flat_prob = _get_flat_probability(prob_map)

    flat_indices = rng.choice(flat_prob.size, size=num_samples, p=flat_prob)
    centers = np.array(np.unravel_index(flat_indices, prob_map.shape)).T

    return centers.astype(int)


def sample_paste_centers_from_occurrence(
    occurrence_prob_map: np.ndarray,
    num_samples: int,
    alpha: float = 0.8,
    valid_mask: Optional[np.ndarray] = None,
    uniform_region: str = "positive",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample multiple paste centers using BRAIN-CP mixed sampling.
    """
    mixed_prob = make_mixed_probability_map(
        occurrence_prob_map=occurrence_prob_map,
        alpha=alpha,
        valid_mask=valid_mask,
        uniform_region=uniform_region,
    )

    return sample_paste_centers(
        prob_map=mixed_prob,
        num_samples=num_samples,
        rng=rng,
    )


def is_valid_center(
    center: Sequence[int],
    brain_mask: Optional[np.ndarray] = None,
    volume_shape: Optional[Sequence[int]] = None,
) -> bool:
    """
    Check whether a paste center is valid.

    If brain_mask is given:
        center must be inside brain_mask.

    If volume_shape is given:
        center must be inside volume bounds.
    """
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


def sample_valid_paste_center(
    occurrence_prob_map: np.ndarray,
    alpha: float = 0.8,
    brain_mask: Optional[np.ndarray] = None,
    uniform_region: str = "positive",
    max_tries: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Center:
    """
    Sample a valid paste center using BRAIN-CP mixed sampling.

    Important
    ---------
    This function only checks the center point.

    Patch boundary check should be done later in braincp_augmentation.py,
    because whether a full tumor patch fits depends on the patch size.
    """
    if max_tries <= 0:
        raise ValueError(f"max_tries must be > 0, got {max_tries}")

    validate_alpha(alpha)
    validate_probability_map(occurrence_prob_map, require_sum_one=False)

    rng = _get_rng(rng)

    # If brain_mask is given, we build the mixed map directly inside brain_mask.
    # This is better than repeatedly sampling invalid centers.
    mixed_prob = make_mixed_probability_map(
        occurrence_prob_map=occurrence_prob_map,
        alpha=alpha,
        valid_mask=brain_mask,
        uniform_region=uniform_region,
    )

    for _ in range(max_tries):
        center = sample_paste_center(mixed_prob, rng=rng)

        if is_valid_center(
            center,
            brain_mask=brain_mask,
            volume_shape=occurrence_prob_map.shape,
        ):
            return center

    raise RuntimeError(
        f"Failed to sample a valid paste center after {max_tries} tries."
    )


def summarize_probability_map(prob_map: np.ndarray) -> dict:
    """
    Return basic summary of probability map.
    """
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


def quick_sampling_check(
    occurrence_prob_map: np.ndarray,
    alpha: float = 0.8,
    num_samples: int = 10,
    seed: int = 2026,
    uniform_region: str = "positive",
) -> None:
    """
    Small CPU-only test.

    This samples centers from the mixed BRAIN-CP probability map and compares
    their probability values against uniform centers.

    Interpretation
    --------------
    sampled_occurrence_prob_mean:
        Average original occurrence probability at sampled centers.

    uniform_occurrence_prob_mean:
        Average original occurrence probability at uniformly sampled centers.

    If sampled_occurrence_prob_mean is higher, the sampler still prefers
    high-occurrence regions.

    If alpha is lower, this gap should become smaller, meaning diversity is larger.
    """
    validate_alpha(alpha)
    validate_probability_map(occurrence_prob_map, require_sum_one=False)

    rng = np.random.default_rng(seed)

    valid_mask = make_default_valid_mask(
        occurrence_prob_map,
        uniform_region=uniform_region,
    )

    mixed_prob = make_mixed_probability_map(
        occurrence_prob_map=occurrence_prob_map,
        alpha=alpha,
        valid_mask=valid_mask,
        uniform_region=uniform_region,
    )

    centers = sample_paste_centers(
        prob_map=mixed_prob,
        num_samples=num_samples,
        rng=rng,
    )

    sampled_occurrence_probs = np.array(
        [occurrence_prob_map[i, j, k] for i, j, k in centers],
        dtype=np.float64,
    )

    valid_indices = np.argwhere(valid_mask > 0)

    uniform_choice = rng.choice(
        valid_indices.shape[0],
        size=num_samples,
        replace=True,
    )
    uniform_centers = valid_indices[uniform_choice]

    uniform_occurrence_probs = np.array(
        [occurrence_prob_map[i, j, k] for i, j, k in uniform_centers],
        dtype=np.float64,
    )

    print("\n[Sampling setting]")
    print(f"alpha: {alpha}")
    print(f"uniform_region: {uniform_region}")
    print("Formula: P_paste = alpha * P_occurrence + (1 - alpha) * P_uniform")

    print("\n[Mixed probability map summary]")
    mixed_summary = summarize_probability_map(mixed_prob)
    for key, value in mixed_summary.items():
        print(f"{key}: {value}")

    print("\n[Sampled paste centers]")
    print("Coordinate order: (i, j, k) = NumPy index order\n")

    for idx, (i, j, k) in enumerate(centers, start=1):
        print(
            f"{idx:02d}: center=({i:3d}, {j:3d}, {k:3d}), "
            f"occ_prob={float(occurrence_prob_map[i, j, k]):.10e}, "
            f"mixed_prob={float(mixed_prob[i, j, k]):.10e}"
        )

    print("\n[Simple sampling sanity check]")
    print(
        f"sampled_occurrence_prob_mean : "
        f"{sampled_occurrence_probs.mean():.10e}"
    )
    print(
        f"uniform_occurrence_prob_mean : "
        f"{uniform_occurrence_probs.mean():.10e}"
    )
    print(
        f"sampled_occurrence_prob_max  : "
        f"{sampled_occurrence_probs.max():.10e}"
    )
    print(
        f"sampled_occurrence_prob_min  : "
        f"{sampled_occurrence_probs.min():.10e}"
    )

    if alpha == 1.0:
        print("[INFO] alpha=1.0 means pure occurrence-based sampling.")
    elif alpha == 0.0:
        print("[INFO] alpha=0.0 means pure uniform sampling over valid region.")
    else:
        print("[INFO] Mixed sampling is enabled.")

    if sampled_occurrence_probs.mean() > uniform_occurrence_probs.mean():
        print(
            "[OK] Sampled centers still tend to have higher occurrence "
            "probability than uniform centers."
        )
    else:
        print(
            "[WARN] Sampled centers are not higher than uniform centers "
            "in this small test."
        )
        print(
            "       This can happen when alpha is low or num_samples is small."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BRAIN-CP occurrence probability map sampler."
    )

    parser.add_argument(
        "--prob-map",
        required=True,
        type=str,
        help="Path to tumor_sampling_probability_WT.npy",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help=(
            "Weight for occurrence map. "
            "1.0=pure occurrence, 0.0=pure uniform."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of paste centers to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed.",
    )
    parser.add_argument(
        "--expected-shape",
        type=int,
        nargs=3,
        default=None,
        help="Optional expected shape, e.g. --expected-shape 182 218 182",
    )
    parser.add_argument(
        "--uniform-region",
        type=str,
        default="positive",
        choices=["positive", "nonzero", "all"],
        help=(
            "Where to define uniform probability when no brain mask is given. "
            "Use positive for normal testing. Use all only for debugging."
        ),
    )

    args = parser.parse_args()

    occurrence_prob_map = load_probability_map(args.prob_map)

    validate_probability_map(
        occurrence_prob_map,
        expected_shape=args.expected_shape,
        require_sum_one=True,
    )

    print("[Original occurrence probability map summary]")
    summary = summarize_probability_map(occurrence_prob_map)
    for key, value in summary.items():
        print(f"{key}: {value}")

    quick_sampling_check(
        occurrence_prob_map=occurrence_prob_map,
        alpha=args.alpha,
        num_samples=args.num_samples,
        seed=args.seed,
        uniform_region=args.uniform_region,
    )


if __name__ == "__main__":
    main()
