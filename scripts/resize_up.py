#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 Ricoh（或任意等距柱状）场景的图像放大 2 倍，得到新目录供 OpenMVG 估计位姿。
放大后的位姿与高分辨率图像一致，可用于 SPaGS/OmniGS 的 Ricoh360 加载器。

用法:
  python upscale_ricoh_for_openmvg.py SOURCE_SCENE OUTPUT_SCENE [--scale 2.0]

  SOURCE_SCENE: 原场景根目录（含 images/ 或 imgs/）
  OUTPUT_SCENE: 输出目录，将创建 OUTPUT_SCENE/images/ 并写入放大后的图
  --scale: 缩放倍数，默认 2.0（宽高各放大一倍）
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def find_image_dir(scene_root: Path) -> Path | None:
    for name in ("images", "imgs"):
        d = scene_root / name
        if d.is_dir():
            return d
    return None


def upscale_images(
    source_images_dir: Path,
    output_images_dir: Path,
    scale: float = 2.0,
) -> tuple[int, int | None, int | None]:
    """将 source 下所有图放大后写入 output，返回 (文件数, 首图宽, 首图高)。"""
    try:
        import cv2
    except ImportError:
        print("需要 opencv-python: pip install opencv-python", file=sys.stderr)
        sys.exit(1)

    output_images_dir.mkdir(parents=True, exist_ok=True)
    extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    count = 0
    first_w, first_h = None, None
    for p in sorted(source_images_dir.iterdir()):
        if not p.is_file() or p.suffix not in extensions:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        nw = int(round(w * scale))
        nh = int(round(h * scale))
        if first_w is None:
            first_w, first_h = nw, nh
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        out_path = output_images_dir / p.name
        cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1
        print(f"  已完成: {p.name}")
    return count, first_w, first_h


def main() -> None:
    parser = argparse.ArgumentParser(
        description="放大 Ricoh 场景图像，供 OpenMVG 估计位姿"
    )
    parser.add_argument("source", type=Path, help="原场景根目录（含 images/ 或 imgs/）")
    parser.add_argument("output", type=Path, help="输出场景根目录，将创建 output/images/")
    parser.add_argument("--scale", type=float, default=2.0, help="缩放倍数，默认 2.0")
    args = parser.parse_args()

    source_images = find_image_dir(args.source)
    if not source_images:
        print(f"错误: 未找到图像目录 {args.source}/images 或 {args.source}/imgs", file=sys.stderr)
        sys.exit(1)

    output_images = args.output / "images"
    print(f"从 {source_images} 放大 {args.scale}x -> {output_images}")
    n, w, h = upscale_images(source_images, output_images, args.scale)
    print(f"已处理 {n} 张图像，输出尺寸约 {w}x{h}")
    print(f"\n下一步：在该目录跑 OpenMVG 并转 Ricoh360 JSON：")
    print(f"  nerficg/scripts/run_openmvg_rarpano.sh {args.output.resolve()} [test_step]")


if __name__ == "__main__":
    main()
