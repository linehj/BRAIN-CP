#!/usr/bin/env python3
"""
Visualize BRAIN-CP tumor occurrence frequency map over a representative MRI image.

Recommended background modality for WT:
    FLAIR = channel 3 = *_0003.nii.gz

Inputs:
    - tumor_occurrence_frequency_WT.npy
    - one MRI image from imagesTr, usually FLAIR

Outputs:
    - overlay PNG images for axial/coronal/sagittal peak slices
    - overlay PNG images for axial/coronal/sagittal max projections
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay WT tumor occurrence frequency map on MRI background."
    )

    parser.add_argument(
        "--frequency-map",
        "--frequency_map",
        dest="frequency_map",
        type=str,
        required=True,
        help="Path to tumor_occurrence_frequency_WT.npy.",
    )

    parser.add_argument(
        "--images-dir",
        "--images_dir",
        dest="images_dir",
        type=str,
        required=True,
        help="Path to nnU-Net imagesTr directory.",
    )

    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        required=True,
        help="Output directory for overlay PNG images.",
    )

    parser.add_argument(
        "--modality-index",
        "--modality_index",
        dest="modality_index",
        type=int,
        default=3,
        help="MRI modality index. For this dataset: 0=T1, 1=T1ce, 2=T2, 3=FLAIR.",
    )

    parser.add_argument(
        "--case-id",
        "--case_id",
        dest="case_id",
        type=str,
        default=None,
        help=(
            "Optional case ID prefix, for example BraTS00005-100. "
            "If not provided, the first matching case is used."
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Maximum overlay transparency. Recommended: 0.4~0.7.",
    )

    parser.add_argument(
        "--threshold-frac",
        "--threshold_frac",
        dest="threshold_frac",
        type=float,
        default=0.05,
        help=(
            "Hide very low occurrence values below threshold_frac * max_frequency. "
            "This makes the background MRI easier to see."
        ),
    )

    return parser.parse_args()


def find_background_image(images_dir: Path, modality_index: int, case_id: str | None) -> Path:
    suffix = f"_{modality_index:04d}.nii.gz"

    if case_id is not None:
        candidate = images_dir / f"{case_id}{suffix}"
        if not candidate.exists():
            raise FileNotFoundError(f"Requested case image not found: {candidate}")
        return candidate

    files = sorted(images_dir.glob(f"*{suffix}"))
    if len(files) == 0:
        raise FileNotFoundError(
            f"No image files found for modality index {modality_index} in {images_dir}"
        )

    return files[0]


def load_nifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    return arr.astype(np.float32)


def robust_normalize_mri(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize MRI intensity for grayscale visualization.

    Percentile normalization is used because MRI intensity scale is arbitrary.
    """
    arr = arr.astype(np.float32)

    nonzero = arr[arr > 0]
    if nonzero.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    low = np.percentile(nonzero, 1)
    high = np.percentile(nonzero, 99)

    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low + eps)
    arr[arr < 0] = 0
    arr[arr > 1] = 1

    return arr.astype(np.float32)


def normalize_for_alpha(freq: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    max_val = float(freq.max())
    if max_val < eps:
        return np.zeros_like(freq, dtype=np.float32)
    return (freq / max_val).astype(np.float32)


def choose_peak_indices(freq: np.ndarray) -> Dict[str, int]:
    sagittal_idx = int(np.argmax(freq.max(axis=(1, 2))))
    coronal_idx = int(np.argmax(freq.max(axis=(0, 2))))
    axial_idx = int(np.argmax(freq.max(axis=(0, 1))))

    return {
        "sagittal": sagittal_idx,
        "coronal": coronal_idx,
        "axial": axial_idx,
    }


def extract_slice(volume: np.ndarray, view: str, index: int) -> np.ndarray:
    if view == "sagittal":
        return volume[index, :, :]
    if view == "coronal":
        return volume[:, index, :]
    if view == "axial":
        return volume[:, :, index]
    raise ValueError(f"Unsupported view: {view}")


def max_projection(volume: np.ndarray, view: str) -> np.ndarray:
    if view == "sagittal":
        return volume.max(axis=0)
    if view == "coronal":
        return volume.max(axis=1)
    if view == "axial":
        return volume.max(axis=2)
    raise ValueError(f"Unsupported view: {view}")


def save_overlay_png(
    background_2d: np.ndarray,
    freq_2d: np.ndarray,
    out_path: Path,
    title: str,
    alpha_max: float,
    threshold_frac: float,
    freq_vmax: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg = np.rot90(background_2d)
    freq = np.rot90(freq_2d)

    alpha_norm = normalize_for_alpha(freq)
    threshold = threshold_frac * float(freq_vmax)

    alpha_map = alpha_norm * alpha_max
    alpha_map[freq < threshold] = 0.0

    plt.figure(figsize=(7, 6))

    plt.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")

    im = plt.imshow(
        freq,
        cmap="jet",
        vmin=0.0,
        vmax=freq_vmax,
        alpha=alpha_map,
        interpolation="nearest",
    )

    plt.title(title)
    plt.axis("off")

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("WT occurrence frequency")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    print(f"[SAVED] {out_path}")


def main() -> None:
    args = parse_args()

    frequency_map_path = Path(args.frequency_map).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    freq = np.load(frequency_map_path).astype(np.float32)
    bg_path = find_background_image(
        images_dir=images_dir,
        modality_index=args.modality_index,
        case_id=args.case_id,
    )
    bg = load_nifti(bg_path)
    bg_norm = robust_normalize_mri(bg)

    print("[INFO] frequency_map:", frequency_map_path)
    print("[INFO] background_image:", bg_path)
    print("[INFO] frequency_shape:", freq.shape)
    print("[INFO] background_shape:", bg.shape)
    print("[INFO] frequency_min/max:", float(freq.min()), float(freq.max()))
    print("[INFO] modality_index:", args.modality_index)

    if tuple(freq.shape) != tuple(bg.shape):
        raise RuntimeError(
            f"Shape mismatch: frequency map {freq.shape}, background MRI {bg.shape}"
        )

    freq_vmax = float(freq.max())
    peak_indices = choose_peak_indices(freq)

    modality_name = {
        0: "T1",
        1: "T1ce",
        2: "T2",
        3: "FLAIR",
    }.get(args.modality_index, f"modality{args.modality_index}")

    # 1. Peak slices: actual 2D slices.
    for view, idx in peak_indices.items():
        bg_slice = extract_slice(bg_norm, view, idx)
        freq_slice = extract_slice(freq, view, idx)

        out_path = output_dir / f"overlay_WT_{modality_name}_{view}_peak_slice_{idx}.png"

        save_overlay_png(
            background_2d=bg_slice,
            freq_2d=freq_slice,
            out_path=out_path,
            title=f"WT occurrence frequency overlay on {modality_name} - {view} peak slice {idx}",
            alpha_max=args.alpha,
            threshold_frac=args.threshold_frac,
            freq_vmax=freq_vmax,
        )

    # 2. Max projections: compressed 3D summary.
    for view in ["axial", "coronal", "sagittal"]:
        bg_proj = max_projection(bg_norm, view)
        freq_proj = max_projection(freq, view)

        out_path = output_dir / f"overlay_WT_{modality_name}_{view}_max_projection.png"

        save_overlay_png(
            background_2d=bg_proj,
            freq_2d=freq_proj,
            out_path=out_path,
            title=f"WT occurrence frequency overlay on {modality_name} - {view} max projection",
            alpha_max=args.alpha,
            threshold_frac=args.threshold_frac,
            freq_vmax=freq_vmax,
        )

    print("[OK] Overlay visualization completed.")


if __name__ == "__main__":
    main()
