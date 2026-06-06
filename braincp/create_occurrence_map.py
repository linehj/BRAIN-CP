#!/usr/bin/env python3
"""
Create brain tumor occurrence probability maps for BRAIN-CP.

Main outputs:
1. tumor_occurrence_count_WT.npy
   - voxel-wise tumor occurrence count

2. tumor_occurrence_frequency_WT.npy
   - raw tumor occurrence frequency
   - count / number of cases
   - used for analysis and visualization

3. tumor_sampling_probability_WT.npy
   - Gaussian-smoothed frequency map
   - normalized so that sum == 1
   - used for BRAIN-CP paste location sampling

Visualization outputs:
- absolute heatmaps:
  actual occurrence frequency scale

- normalized heatmaps:
  visualization-only 0~1 normalized intensity
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create WT tumor occurrence maps for BRAIN-CP."
    )

    parser.add_argument(
        "--labels-dir",
        "--labels_dir",
        dest="labels_dir",
        type=str,
        required=True,
        help="Path to BraTS labelsTr directory.",
    )

    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default="occurrence_maps",
        help="Output directory for occurrence maps.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="WT",
        choices=["WT"],
        help="Tumor region mode. Current implementation supports WT only.",
    )

    parser.add_argument(
        "--smooth-sigma",
        "--smooth_sigma",
        dest="smooth_sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma for sampling probability map.",
    )

    parser.add_argument(
        "--make-figures",
        "--make_figures",
        dest="make_figures",
        action="store_true",
        help="Save PNG heatmap visualizations.",
    )

    return parser.parse_args()


def load_label(label_path: Path) -> np.ndarray:
    """
    Load a BraTS label NIfTI file.

    Returns
    -------
    label : np.ndarray
        3D label array.
    """
    img = nib.load(str(label_path))
    label = np.asanyarray(img.dataobj)
    return label.astype(np.uint8)


def make_tumor_mask(label: np.ndarray, mode: str) -> np.ndarray:
    """
    Convert BraTS label to binary tumor mask.

    WT means Whole Tumor.
    For the current BRAIN-CP first implementation:
        WT = all non-background tumor labels
        tumor_mask = label > 0
    """
    if mode == "WT":
        return label > 0

    raise ValueError(f"Unsupported mode: {mode}")


def get_shape_counts(label_files: Iterable[Path]) -> Dict[Tuple[int, int, int], int]:
    """
    Check shape distribution using NIfTI headers.
    """
    shape_counts: Dict[Tuple[int, int, int], int] = {}

    for f in label_files:
        shape = tuple(nib.load(str(f)).shape)
        shape_counts[shape] = shape_counts.get(shape, 0) + 1

    return shape_counts


def normalize_sum_to_one(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize an array so that its sum is 1.
    """
    arr = arr.astype(np.float32)
    total = float(arr.sum())

    if total < eps:
        raise RuntimeError("Cannot normalize: array sum is zero.")

    return arr / total


def normalize_for_visualization(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize array to 0~1 for visualization only.

    This does NOT change the scientific raw frequency map.
    It is only used for PNG heatmaps.
    """
    arr = arr.astype(np.float32)
    arr_min = float(arr.min())
    arr_max = float(arr.max())

    if arr_max - arr_min < eps:
        return np.zeros_like(arr, dtype=np.float32)

    return (arr - arr_min) / (arr_max - arr_min)


def choose_peak_indices(volume: np.ndarray) -> Dict[str, int]:
    """
    Choose slice index with the highest value for each anatomical view.

    NIfTI array axes are treated as:
        axis 0: sagittal direction
        axis 1: coronal direction
        axis 2: axial direction
    """
    sagittal_idx = int(np.argmax(volume.max(axis=(1, 2))))
    coronal_idx = int(np.argmax(volume.max(axis=(0, 2))))
    axial_idx = int(np.argmax(volume.max(axis=(0, 1))))

    return {
        "sagittal": sagittal_idx,
        "coronal": coronal_idx,
        "axial": axial_idx,
    }


def extract_slice(volume: np.ndarray, view: str, index: int) -> np.ndarray:
    """
    Extract 2D slice from 3D volume.
    """
    if view == "sagittal":
        return volume[index, :, :]
    if view == "coronal":
        return volume[:, index, :]
    if view == "axial":
        return volume[:, :, index]

    raise ValueError(f"Unsupported view: {view}")


def save_heatmap_png(
    arr2d: np.ndarray,
    out_path: Path,
    title: str,
    colorbar_label: str,
    vmin: float,
    vmax: float,
) -> None:
    """
    Save a 2D heatmap PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        np.rot90(arr2d),
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    plt.title(title)
    plt.axis("off")

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_visualizations(
    frequency_map: np.ndarray,
    output_dir: Path,
    mode: str,
) -> None:
    """
    Save occurrence map heatmap visualizations.

    Two visualization types are saved:
    1. absolute:
       colorbar uses actual occurrence frequency

    2. normalized:
       colorbar uses visualization-only 0~1 scale
    """
    freq = frequency_map.astype(np.float32)
    freq_max = float(freq.max())

    if freq_max <= 0:
        print("[WARNING] frequency map max is 0. Skipping visualizations.")
        return

    normalized = normalize_for_visualization(freq)
    indices = choose_peak_indices(freq)

    for view, idx in indices.items():
        raw_slice = extract_slice(freq, view, idx)
        norm_slice = extract_slice(normalized, view, idx)

        abs_path = output_dir / f"tumor_occurrence_frequency_{mode}_{view}_peak_absolute.png"
        norm_path = output_dir / f"tumor_occurrence_frequency_{mode}_{view}_peak_normalized.png"

        save_heatmap_png(
            arr2d=raw_slice,
            out_path=abs_path,
            title=f"{mode} occurrence frequency - {view} peak slice {idx}",
            colorbar_label="Occurrence frequency",
            vmin=0.0,
            vmax=freq_max,
        )

        save_heatmap_png(
            arr2d=norm_slice,
            out_path=norm_path,
            title=f"{mode} occurrence intensity - {view} peak slice {idx}",
            colorbar_label="Normalized occurrence intensity",
            vmin=0.0,
            vmax=1.0,
        )

        print(f"[SAVED] {abs_path}")
        print(f"[SAVED] {norm_path}")

    # Maximum projection gives a compact map-like overview.
    projections = {
        "sagittal": freq.max(axis=0),
        "coronal": freq.max(axis=1),
        "axial": freq.max(axis=2),
    }

    norm_projections = {
        key: normalize_for_visualization(value)
        for key, value in projections.items()
    }

    for view, proj in projections.items():
        abs_path = output_dir / f"tumor_occurrence_frequency_{mode}_{view}_max_projection_absolute.png"
        norm_path = output_dir / f"tumor_occurrence_frequency_{mode}_{view}_max_projection_normalized.png"

        save_heatmap_png(
            arr2d=proj,
            out_path=abs_path,
            title=f"{mode} occurrence frequency - {view} max projection",
            colorbar_label="Occurrence frequency",
            vmin=0.0,
            vmax=freq_max,
        )

        save_heatmap_png(
            arr2d=norm_projections[view],
            out_path=norm_path,
            title=f"{mode} occurrence intensity - {view} max projection",
            colorbar_label="Normalized occurrence intensity",
            vmin=0.0,
            vmax=1.0,
        )

        print(f"[SAVED] {abs_path}")
        print(f"[SAVED] {norm_path}")


def main() -> None:
    args = parse_args()

    labels_dir = Path(args.labels_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.nii.gz"))

    if not labels_dir.exists():
        raise FileNotFoundError(f"labels_dir does not exist: {labels_dir}")

    if len(label_files) == 0:
        raise RuntimeError(f"No .nii.gz label files found in: {labels_dir}")

    print("[INFO] BRAIN-CP occurrence map generation")
    print(f"[INFO] labels_dir: {labels_dir}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] mode: {args.mode}")
    print(f"[INFO] smooth_sigma: {args.smooth_sigma}")
    print(f"[INFO] num_label_files: {len(label_files)}")

    shape_counts = get_shape_counts(label_files)
    print("[INFO] shape counts:")
    for shape, count in shape_counts.items():
        print(f"  shape={shape}, count={count}")

    if len(shape_counts) != 1:
        raise RuntimeError(
            "Label files have multiple shapes. "
            "Occurrence map generation requires identical shapes."
        )

    label_shape = next(iter(shape_counts.keys()))
    occurrence_count = np.zeros(label_shape, dtype=np.float32)

    used_cases = 0

    for idx, label_path in enumerate(label_files, start=1):
        label = load_label(label_path)

        if tuple(label.shape) != tuple(label_shape):
            raise RuntimeError(
                f"Shape mismatch in {label_path.name}: "
                f"got {label.shape}, expected {label_shape}"
            )

        tumor_mask = make_tumor_mask(label, args.mode)

        if tumor_mask.sum() == 0:
            print(f"[WARNING] no tumor voxel found: {label_path.name}")
            continue

        occurrence_count += tumor_mask.astype(np.float32)
        used_cases += 1

        if idx % 50 == 0 or idx == len(label_files):
            print(f"[INFO] processed {idx}/{len(label_files)} files")

    if used_cases == 0:
        raise RuntimeError("No usable tumor cases found.")

    occurrence_frequency = occurrence_count / float(used_cases)

    if args.smooth_sigma > 0:
        smoothed_frequency = gaussian_filter(
            occurrence_frequency,
            sigma=args.smooth_sigma,
        )
    else:
        smoothed_frequency = occurrence_frequency.copy()

    sampling_probability = normalize_sum_to_one(smoothed_frequency)

    count_path = output_dir / f"tumor_occurrence_count_{args.mode}.npy"
    freq_path = output_dir / f"tumor_occurrence_frequency_{args.mode}.npy"
    sampling_path = output_dir / f"tumor_sampling_probability_{args.mode}.npy"
    metadata_path = output_dir / f"tumor_occurrence_metadata_{args.mode}.json"

    np.save(count_path, occurrence_count)
    np.save(freq_path, occurrence_frequency)
    np.save(sampling_path, sampling_probability)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "labels_dir": str(labels_dir),
        "output_dir": str(output_dir),
        "mode": args.mode,
        "smooth_sigma": args.smooth_sigma,
        "num_label_files": len(label_files),
        "used_cases": used_cases,
        "label_shape": list(label_shape),
        "frequency_min": float(occurrence_frequency.min()),
        "frequency_max": float(occurrence_frequency.max()),
        "frequency_mean": float(occurrence_frequency.mean()),
        "sampling_probability_sum": float(sampling_probability.sum()),
        "sampling_probability_min": float(sampling_probability.min()),
        "sampling_probability_max": float(sampling_probability.max()),
        "note": (
            "frequency map is raw count/used_cases; "
            "sampling probability map is smoothed and sum-normalized."
        ),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("[SAVED]", count_path)
    print("[SAVED]", freq_path)
    print("[SAVED]", sampling_path)
    print("[SAVED]", metadata_path)

    if args.make_figures:
        save_visualizations(
            frequency_map=occurrence_frequency,
            output_dir=output_dir,
            mode=args.mode,
        )

    print("\n[SUMMARY]")
    print(f"used_cases: {used_cases}")
    print(f"map_shape: {label_shape}")
    print(f"count_min/max: {occurrence_count.min():.6f} / {occurrence_count.max():.6f}")
    print(f"frequency_min/max: {occurrence_frequency.min():.8f} / {occurrence_frequency.max():.8f}")
    print(f"sampling_sum: {sampling_probability.sum():.8f}")
    print(f"sampling_min/max: {sampling_probability.min():.12f} / {sampling_probability.max():.12f}")

    if abs(float(sampling_probability.sum()) - 1.0) < 1e-5:
        print("[OK] sampling probability map sum is approximately 1.")
    else:
        print("[WARNING] sampling probability map sum is not close to 1.")


if __name__ == "__main__":
    main()
