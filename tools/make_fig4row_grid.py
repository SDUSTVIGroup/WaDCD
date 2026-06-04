#!/usr/bin/env python3
"""Stitch a 4-row figure grid across categories.

Rows (top->bottom):
  1) Image (x)
  2) Recon. (xhat)
  3) Anomaly GT (binary mask)
  4) Anomaly map (anomaly_overlay)

Inputs are taken from an existing evaluation outputs folder:
  <visuals_root>/<category>/<sample_id>_x.png
  <visuals_root>/<category>/<sample_id>_xhat.png
  <visuals_root>/<category>/<sample_id>_anomaly_overlay.png

The GT mask is loaded from the dataset split via MVTECDataset/VISADataset,
and aligned by the dataset index implied by sample_id (batch_idx * batch_size + b).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def _parse_sample_id(sample_id: str, batch_size: int) -> int:
    """Convert '0000_b00' -> dataset index (batch_idx*batch_size + b)."""
    m = re.fullmatch(r"(\d{4})_b(\d{2})", sample_id)
    if not m:
        raise ValueError(f"Invalid sample_id '{sample_id}'. Expected like 0000_b00")
    batch_idx = int(m.group(1))
    b = int(m.group(2))
    return batch_idx * batch_size + b


def _load_rgb(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def _blank_rgb(size: Tuple[int, int]) -> Image.Image:
    return Image.new("RGB", size, color=(255, 255, 255))


def _safe_load_rgb(path: str, size: Tuple[int, int], missing: str) -> Image.Image:
    """missing: 'error'|'blank'"""
    try:
        img = _load_rgb(path)
        if img.size != size:
            img = img.resize(size)
        return img
    except FileNotFoundError:
        if missing == "blank":
            return _blank_rgb(size)
        raise


def _center_crop(img: Image.Image, crop_size: int) -> Image.Image:
    w, h = img.size
    if crop_size <= 0 or (crop_size == w and crop_size == h):
        return img
    left = max(0, (w - crop_size) // 2)
    top = max(0, (h - crop_size) // 2)
    right = min(w, left + crop_size)
    bottom = min(h, top + crop_size)
    return img.crop((left, top, right, bottom))


def _mask_path_from_split_row(data_dir: str, row: Dict[str, str]) -> Optional[str]:
    mask_rel = (row.get("mask") or "").strip()
    if not mask_rel:
        return None
    if mask_rel.lower() == "nan":
        return None
    return os.path.join(data_dir, mask_rel)


def _load_binary_mask_pil(
    mask_path: Optional[str],
    image_size: int,
    center_size: int,
    center_crop: bool,
) -> Image.Image:
    if not mask_path or not os.path.isfile(mask_path):
        base = Image.new("L", (image_size, image_size), color=0)
    else:
        base = Image.open(mask_path).convert("L")
        base = base.resize((image_size, image_size), resample=Image.NEAREST)
    if center_crop:
        base = _center_crop(base, center_size)
    # Binarize and map to 0/255
    base = base.point(lambda p: 255 if p > 0 else 0, mode="L")
    return base


def _mask_to_rgb(mask_l: Image.Image, size: Tuple[int, int]) -> Image.Image:
    pil = mask_l.resize(size, resample=Image.NEAREST)
    return pil.convert("RGB")


def _get_default_font(size: int = 16) -> ImageFont.ImageFont:
    # Prefer a truetype if available; fallback to default bitmap font.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _draw_centered_text(draw: ImageDraw.ImageDraw, xywh, text: str, font, fill=(0, 0, 0)):
    x, y, w, h = xywh
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x + (w - tw) // 2
    ty = y + (h - th) // 2
    draw.text((tx, ty), text, font=font, fill=fill)


def _read_split_rows(split_csv: str, dataset: str, category: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(split_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = (row.get("object") or "").strip()
            split = (row.get("split") or "").strip()
            if obj != category or split != "test":
                continue
            # Keep file order to match dataset iteration semantics.
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {split_csv} for category={category}, split=test")
    return rows


def _load_gt_mask_from_splits(
    dataset: str,
    data_dir: str,
    category: str,
    idx: int,
    image_size: int,
    center_size: int,
    center_crop: bool,
) -> Image.Image:
    if dataset == "mvtec":
        split_csv = os.path.join(os.path.dirname(__file__), "..", "splits", "mvtec-split.csv")
    elif dataset == "visa":
        split_csv = os.path.join(os.path.dirname(__file__), "..", "splits", "visa-split.csv")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    split_csv = os.path.abspath(split_csv)
    rows = _read_split_rows(split_csv, dataset=dataset, category=category)
    if idx < 0 or idx >= len(rows):
        raise IndexError(f"idx={idx} out of range for {category} test split (len={len(rows)})")
    row = rows[idx]
    mask_path = _mask_path_from_split_row(data_dir, row)
    return _load_binary_mask_pil(mask_path, image_size=image_size, center_size=center_size, center_crop=center_crop)


def make_grid(
    dataset: str,
    data_dir: str,
    visuals_root: str,
    categories: List[str],
    sample_id: str,
    out_path: str,
    batch_size: int = 8,
    image_size: int = 288,
    center_size: int = 256,
    center_crop: bool = True,
    title_height: int = 36,
    row_label_width: int = 120,
    pad: int = 10,
    gap: int = 8,
    missing: str = "error",
):
    idx = _parse_sample_id(sample_id, batch_size=batch_size)

    if not categories:
        raise ValueError("categories is empty")

    # Load one image to get panel size.
    first_cat = categories[0]
    first_x = _load_rgb(os.path.join(visuals_root, first_cat, f"{sample_id}_x.png"))
    panel_w, panel_h = first_x.size

    rows = 4
    cols = len(categories)
    canvas_w = pad * 2 + row_label_width + cols * panel_w + (cols - 1) * gap
    canvas_h = pad * 2 + title_height + rows * panel_h + (rows - 1) * gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font_title = _get_default_font(18)
    font_row = _get_default_font(16)

    # Row labels
    row_names = ["Image", "Recon.", "Anomaly\nGT", "Anomaly\nmap"]
    for r, name in enumerate(row_names):
        x0 = pad
        y0 = pad + title_height + r * (panel_h + gap)
        _draw_centered_text(draw, (x0, y0, row_label_width, panel_h), name, font_row)

    # Panels
    for c, cat in enumerate(categories):
        col_x = pad + row_label_width + c * (panel_w + gap)
        # Column title
        _draw_centered_text(draw, (col_x, pad, panel_w, title_height), cat, font_title)

        panel_size = (panel_w, panel_h)
        x_img = _safe_load_rgb(os.path.join(visuals_root, cat, f"{sample_id}_x.png"), panel_size, missing)
        xhat_img = _safe_load_rgb(os.path.join(visuals_root, cat, f"{sample_id}_xhat.png"), panel_size, missing)
        overlay_img = _safe_load_rgb(os.path.join(visuals_root, cat, f"{sample_id}_anomaly_overlay.png"), panel_size, missing)

        try:
            seg_l = _load_gt_mask_from_splits(
                dataset=dataset,
                data_dir=data_dir,
                category=cat,
                idx=idx,
                image_size=image_size,
                center_size=center_size,
                center_crop=center_crop,
            )
        except Exception:
            if missing == "blank":
                seg_l = Image.new("L", (center_size if center_crop else image_size, center_size if center_crop else image_size), color=0)
            else:
                raise
        gt_img = _mask_to_rgb(seg_l, (panel_w, panel_h))

        imgs = [x_img, xhat_img, gt_img, overlay_img]
        for r in range(rows):
            y = pad + title_height + r * (panel_h + gap)
            canvas.paste(imgs[r], (col_x, y))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Stitch 4-row grid figure across categories.")
    parser.add_argument("--dataset", choices=["mvtec", "visa"], default="mvtec")
    parser.add_argument("--data-dir", type=str, default="./mvtec-dataset/")
    parser.add_argument(
        "--visuals-root",
        type=str,
        default="./WaDCD_mvtec_all_UNet_L_256_CenterCrop/visuals",
        help="Root folder containing per-category visuals subfolders.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Categories (subfolder names under visuals-root). If omitted, auto-discover from visuals-root.",
    )
    parser.add_argument("--sample-id", type=str, default="0000_b00")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size used when saving visuals (default: 8).")
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--center-size", type=int, default=256)
    parser.add_argument("--center-crop", action="store_true", help="Apply center crop (after resizing) to match evaluation.")
    parser.add_argument("--missing", choices=["error", "blank"], default="error", help="How to handle missing per-category panels.")
    parser.add_argument("--out", type=str, default="./fig1_grid.png")

    args = parser.parse_args()

    # Auto-discover categories if not provided.
    categories = args.categories
    if categories is None:
        if not os.path.isdir(args.visuals_root):
            raise FileNotFoundError(args.visuals_root)
        categories = [
            name
            for name in sorted(os.listdir(args.visuals_root))
            if os.path.isdir(os.path.join(args.visuals_root, name))
        ]

    make_grid(
        dataset=args.dataset,
        data_dir=args.data_dir,
        visuals_root=args.visuals_root,
        categories=categories,
        sample_id=args.sample_id,
        out_path=args.out,
        batch_size=args.batch_size,
        image_size=args.image_size,
        center_size=args.center_size,
        center_crop=args.center_crop,
        missing=args.missing,
    )


if __name__ == "__main__":
    main()
