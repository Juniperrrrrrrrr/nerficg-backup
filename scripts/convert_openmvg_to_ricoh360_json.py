#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
将 OpenMVG 的 SfM 导出结果整理成 Ricoh360 数据加载器可读的格式：
  - openMVG/data_openmvg_train.json
  - openMVG/data_openmvg_test.json
  - openMVG/scene.ply
并确保场景根目录下存在 imgs -> images 的链接（由 run_openmvg_rarpano.sh 创建）。
"""

import argparse
import copy
import json
import math
import shutil
from pathlib import Path
from typing import Tuple


def get_image_size(image_path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            return im.size[0], im.size[1]
    except Exception:
        return 0, 0


def rotation_center_from_w2c(R_w2c, t_w2c):
    """OpenMVG 外参为 world-to-camera [R|t]，相机中心 C = -R^T @ t。"""
    import numpy as np
    R = np.array(R_w2c, dtype=float)
    t = np.array(t_w2c, dtype=float).reshape(3)
    center = -R.T @ t
    return R.tolist(), center.tolist()


def load_openmvg_sfm(sfm_path: Path) -> dict:
    with open(sfm_path, "r") as f:
        return json.load(f)


def is_ricoh360_format(data: dict) -> bool:
    """判断是否已是 Ricoh360 所用的 Cereal 格式（含 ptr_wrapper）。"""
    if not data.get("views"):
        return False
    v = data["views"][0]
    return "value" in v and "ptr_wrapper" in (v.get("value") or {})


def convert_to_ricoh360_format(
    sfm_data: dict,
    image_dir: Path,
    focal_angle_deg: float = 0.0,
    train_ratio: float = 0.8,
    test_step: int = 0,
    train_step: int = 1,
) -> Tuple[dict, dict]:
    """
    将 OpenMVG 导出的 sfm_data（可能为另一种 JSON 结构）转为两个字典：
    train_data, test_data，每个均为 Ricoh360 所需的完整 JSON 结构（intrinsics, views, extrinsics）。
    若已是 ptr_wrapper 格式，则只做 train/test 划分并统一 filename 为仅文件名。
    """
    # 若已是 Cereal 格式，只做划分、规范化 intrinsic/view 为 Ricoh360 格式
    if is_ricoh360_format(sfm_data):
        return _split_ricoh360_like(sfm_data, train_ratio, test_step, image_dir, train_step)

    # 否则按 OpenMVG 常见导出结构解析：views, intrinsics, extrinsics 或 poses
    views_in = sfm_data.get("views") or []
    intrinsics_in = sfm_data.get("intrinsics") or sfm_data.get("Intrinsics") or []
    extrinsics_in = sfm_data.get("extrinsics") or sfm_data.get("Extrinsics") or {}
    poses_in = sfm_data.get("poses") or sfm_data.get("Poses") or {}

    if not views_in:
        raise ValueError("sfm_data 中缺少 views")

    # 取第一张图尺寸以生成内参
    first_id = next((v.get("key") or v.get("id") or list(views_in)[0]) for v in (views_in if isinstance(views_in, list) else [views_in]))
    if isinstance(views_in, dict):
        views_list = list(views_in.values())
    else:
        views_list = views_in
    first_view = views_list[0]
    filename = first_view.get("value", first_view).get("ptr_wrapper", first_view).get("data", first_view).get("filename", "") or first_view.get("path", "") or first_view.get("filename", "")
    if not filename:
        for v in views_list:
            fn = (v.get("value") or v).get("ptr_wrapper", v) if isinstance(v, dict) else v
            if isinstance(fn, dict):
                fn = (fn.get("data") or fn).get("filename", "") or fn.get("path", "")
            if fn:
                filename = fn
                break
    if isinstance(filename, str) and "/" in filename:
        filename = Path(filename).name
    first_img = image_dir / filename
    if not first_img.exists():
        for p in image_dir.iterdir():
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                first_img = p
                break
    width, height = get_image_size(first_img) if first_img.exists() else (1920, 960)
    focal_pix = width / (2 * math.pi * math.cos(focal_angle_deg * math.pi / 180))

    # 构建 Ricoh360 式 intrinsics（与官方一致：spherical + value0 仅 width/height，OmniGS 靠此建 remap）
    intrinsic_id = 0
    intrinsics_ricoh = [
        {
            "key": 0,
            "value": {
                "polymorphic_id": 2147483649,
                "polymorphic_name": "spherical",
                "ptr_wrapper": {
                    "id": 2147483699,
                    "data": {
                        "value0": {
                            "width": width,
                            "height": height,
                        },
                    },
                },
            },
        }
    ]

    # 解析每个 view 的 pose，生成 extrinsics（rotation 3x3 + center 3）
    extrinsics_ricoh = {}
    views_ricoh = []
    for idx, view in enumerate(views_list if isinstance(views_list, list) else [views_list]):
        if isinstance(view, dict):
            v = view.get("value", view)
            v = v.get("ptr_wrapper", v).get("data", v) if isinstance(v, dict) else v
            pose_id = v.get("id_pose", idx)
            fn = v.get("filename", "") or view.get("path", "") or view.get("filename", "")
        else:
            pose_id = idx
            fn = str(view)
        if isinstance(fn, str) and "/" in fn:
            fn = Path(fn).name

        # 从 poses 或 extrinsics 取 [R|t]
        pose_src = poses_in.get(str(pose_id)) or poses_in.get(pose_id) or extrinsics_in.get(str(pose_id)) or extrinsics_in.get(pose_id)
        if not pose_src:
            continue
        pv = pose_src.get("value", pose_src) if isinstance(pose_src, dict) else pose_src
        if isinstance(pv, dict):
            rot = pv.get("rotation")
            center = pv.get("center")
            if center is not None and rot is not None:
                pass
            else:
                R = pv.get("rotation") or pv.get("R")
                t = pv.get("center") or pv.get("translation") or pv.get("t")
                if R is not None and t is not None:
                    rot, center = rotation_center_from_w2c(R, t)
                else:
                    continue
        else:
            continue

        extrinsics_ricoh[pose_id] = {"key": pose_id, "value": {"rotation": rot, "center": center}}
        views_ricoh.append({
            "key": idx,
            "value": {
                "polymorphic_id": 1073741824,
                "ptr_wrapper": {
                    "id": 2147483649 + idx,
                    "data": {
                        "local_path": "",
                        "filename": fn,
                        "width": width,
                        "height": height,
                        "id_view": idx,
                        "id_intrinsic": intrinsic_id,
                        "id_pose": pose_id,
                    },
                },
            },
        })

    full_data = {
        "intrinsics": intrinsics_ricoh,
        "views": views_ricoh,
        "extrinsics": extrinsics_ricoh,
    }
    return _split_ricoh360_like(full_data, train_ratio, test_step, image_dir, train_step)


def _reindex_to_sequential_ricoh360(data: dict) -> None:
    """
    把 views 和 extrinsics 重排成 0,1,2,...，与官方 Ricoh360 一致。
    OmniGS 用 view.key 和 id_pose 做 lookup，重排后 key == id_pose == id_view，避免错位。
    """
    views = data.get("views", [])
    extrinsics = data.get("extrinsics", [])
    if not views or not extrinsics:
        return
    # 建 id_pose -> extrinsic value 的映射（extrinsics 可能是 list of {key, value}）
    ext_by_key = {}
    for e in (extrinsics if isinstance(extrinsics, list) else extrinsics.values()):
        k = e.get("key") if isinstance(e, dict) else None
        if k is not None:
            ext_by_key[int(k)] = e.get("value", e) if isinstance(e, dict) else e
    new_views = []
    new_extrinsics = []
    for i, v in enumerate(views):
        try:
            inner = v["value"]["ptr_wrapper"]["data"]
            old_pose = inner.get("id_pose")
            if old_pose is None or int(old_pose) not in ext_by_key:
                continue
            ext_val = ext_by_key[int(old_pose)]
        except (KeyError, TypeError):
            continue
        # 新 view：key=i, id_view=i, id_pose=i
        new_v = copy.deepcopy(v)
        new_v["key"] = i
        new_v["value"]["ptr_wrapper"]["id"] = 2147483649 + i
        new_v["value"]["ptr_wrapper"]["data"]["id_view"] = i
        new_v["value"]["ptr_wrapper"]["data"]["id_pose"] = i
        new_views.append(new_v)
        new_extrinsics.append({"key": i, "value": ext_val})
    if new_views:
        data["views"] = new_views
        data["extrinsics"] = new_extrinsics


def _normalize_ricoh360_omnigs(data: dict, image_dir: Path = None) -> None:
    """把 intrinsics/views 规范成 Ricoh360 官方格式，便于 OmniGS 建 remap（spherical + value0 仅 width/height）。"""
    views = data.get("views", [])
    intrinsics = data.get("intrinsics", [])
    if not views:
        return
    # 取 width/height：优先第一个 view 的 data，否则从 intrinsic 的 value0，否则读图
    v0 = views[0]
    try:
        d = v0["value"]["ptr_wrapper"]["data"]
        w, h = d.get("width"), d.get("height")
    except (KeyError, TypeError):
        w, h = None, None
    if (w is None or h is None) and intrinsics:
        try:
            val0 = intrinsics[0].get("value", {}) or {}
            pw = (val0.get("ptr_wrapper") or {}).get("data", {}) or {}
            v0 = (pw.get("value0") or pw)
            w, h = v0.get("width"), v0.get("height")
        except (KeyError, TypeError):
            pass
    if (w is None or h is None) and image_dir and views:
        fn = (views[0].get("value") or {}).get("ptr_wrapper", {}).get("data", {}).get("filename", "")
        if fn:
            fn = Path(fn).name
            w, h = get_image_size(Path(image_dir) / fn)
    if w is None or h is None:
        w, h = 1920, 960
    w, h = int(w), int(h)
    # 统一 intrinsic：spherical，value0 仅 width/height
    data["intrinsics"] = [{
        "key": 0,
        "value": {
            "polymorphic_id": 2147483649,
            "polymorphic_name": "spherical",
            "ptr_wrapper": {"id": 2147483699, "data": {"value0": {"width": w, "height": h}}},
        },
    }]
    # 给每个 view 的 data 补全 width, height, id_view, local_path（与 Ricoh360 一致）
    for i, v in enumerate(views):
        try:
            inner = v["value"]["ptr_wrapper"]["data"]
            if inner.get("width") is None:
                inner["width"] = w
            if inner.get("height") is None:
                inner["height"] = h
            if "id_view" not in inner:
                inner["id_view"] = i
            if "local_path" not in inner:
                inner["local_path"] = ""
        except (KeyError, TypeError):
            pass


def _get_view_filename(v: dict) -> str:
    """从 view 里取出 filename，用于按文件名排序，保证原图/2x 同一套 test 视角。"""
    try:
        inner = (v.get("value") or v).get("ptr_wrapper", {}) or {}
        inner = (inner.get("data") or inner)
        fn = inner.get("filename", "") or v.get("path", "") or v.get("filename", "")
        return (fn or "").split("/")[-1]
    except (TypeError, AttributeError, KeyError):
        return ""


def _split_ricoh360_like(
    data: dict,
    train_ratio: float = 0.8,
    test_step: int = 0,
    image_dir: Path = None,
    train_step: int = 1,
) -> Tuple[dict, dict]:
    """划分 train / test：test_step>0 时每 test_step 张取 1 张为测试集，否则按 train_ratio 比例划分。train_step>1 时对训练集再抽帧。
    划分前按 filename 排序，保证原数据集与 2x 等不同 run 的 test 视角一致（同一批帧）。"""
    views = data.get("views", [])
    # 按文件名排序，使原图 / 2x 两次跑 OpenMVG 得到的 test 集对应同一批视角（同一批文件名）
    views = sorted(views, key=_get_view_filename)
    data = dict(data)
    data["views"] = views
    n = len(views)
    if n == 0:
        return data, {k: [] if k == "views" else v for k, v in data.items()}
    if test_step > 0:
        # 每 test_step 张取 1 张为测试集（如 test_step=8 则第 0,8,16,... 张为 test）
        train_views = [v for i, v in enumerate(views) if i % test_step != 0]
        test_views = [v for i, v in enumerate(views) if i % test_step == 0]
        if not train_views:
            train_views = views[:-1] if len(views) > 1 else views
            test_views = views[-1:] if len(views) > 1 else []
    else:
        n_train = max(1, int(n * train_ratio))
        train_views = views[:n_train]
        test_views = views[n_train:]
    # 训练集再抽帧：每 train_step 张取 1 张（减显存）
    if train_step > 1 and len(train_views) > 1:
        train_views = train_views[::train_step]
        if not train_views:
            train_views = [views[0]]

    def get_pose_id(v):
        try:
            return v["value"]["ptr_wrapper"]["data"]["id_pose"]
        except (KeyError, TypeError):
            return None

    pose_ids_train = {get_pose_id(v) for v in train_views if get_pose_id(v) is not None}
    pose_ids_test = {get_pose_id(v) for v in test_views if get_pose_id(v) is not None}
    extrinsics = data.get("extrinsics", {})
    if isinstance(extrinsics, dict):
        ext_list = list(extrinsics.values())
    else:
        ext_list = list(extrinsics) if extrinsics else []
    extrinsics_by_id = {e.get("key", i): e for i, e in enumerate(ext_list)}

    # OmniGS C++ 按数组遍历 extrinsics，必须为 array；顺序按 pose_id 排即可，用 key 做 lookup
    train_ext_list = [extrinsics_by_id[pid] for pid in sorted(pose_ids_train) if pid in extrinsics_by_id]
    test_ext_list = [extrinsics_by_id[pid] for pid in sorted(pose_ids_test) if pid in extrinsics_by_id]
    train_data = {
        "intrinsics": data.get("intrinsics", []),
        "views": train_views,
        "extrinsics": train_ext_list,
    }
    test_data = {
        "intrinsics": data.get("intrinsics", []),
        "views": test_views,
        "extrinsics": test_ext_list,
    }
    _normalize_ricoh360_omnigs(train_data, image_dir)
    _normalize_ricoh360_omnigs(test_data, image_dir)
    # 重排为 0,1,2,...，与官方 Ricoh360 一致，避免 OmniGS 按 id_pose 查 extrinsic 错位
    _reindex_to_sequential_ricoh360(train_data)
    _reindex_to_sequential_ricoh360(test_data)
    return train_data, test_data


def main():
    parser = argparse.ArgumentParser(description="将 OpenMVG 输出转为 Ricoh360 格式并划分 train/test")
    parser.add_argument("--scene_root", type=Path, required=True, help="RaRPano 场景根目录")
    parser.add_argument("--openmvg_out", type=Path, required=True, help="OpenMVG 输出目录（含 reconstruction/）")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例（test_step=0 时生效）")
    parser.add_argument("--test_step", type=int, default=0, help="每 N 张取 1 张为测试集，如 8 表示第 0,8,16,... 张为 test；0 表示用 train_ratio 划分")
    parser.add_argument("--train_step", type=int, default=1, help="训练集再抽帧：每 N 张取 1 张（1=不抽帧），与 test_step 配合可减显存")
    parser.add_argument("--focal_angle", type=float, default=0.0, help="FOCAL_ANGLE 度，用于等距柱状焦距")
    args = parser.parse_args()

    scene_root = args.scene_root.resolve()
    openmvg_out = args.openmvg_out.resolve()
    # 图像目录：优先 images，否则 imgs（与官方 Ricoh360 数据集一致）
    image_dir = scene_root / "images" if (scene_root / "images").exists() else scene_root / "imgs"
    openmvg_dir = scene_root / "openMVG"
    openmvg_dir.mkdir(parents=True, exist_ok=True)

    sfm_json = openmvg_out / "reconstruction" / "sfm_data.json"
    sfm_bin = openmvg_out / "reconstruction" / "sfm_data.bin"
    if not sfm_json.exists() and sfm_bin.exists():
        raise FileNotFoundError(f"请先将 sfm_data.bin 转为 JSON: openMVG_main_ConvertSfM_DataFormat -i {sfm_bin} -o {sfm_json} -V -I -E")
    if not sfm_json.exists():
        raise FileNotFoundError(f"未找到 {sfm_json}")

    sfm_data = load_openmvg_sfm(sfm_json)
    train_data, test_data = convert_to_ricoh360_format(
        sfm_data, image_dir, args.focal_angle, args.train_ratio, args.test_step, args.train_step
    )
    # OmniGS 用 root_path 拼图像路径；用绝对路径可避免依赖进程 cwd
    root_path = str(image_dir.resolve())
    train_data["root_path"] = root_path
    test_data["root_path"] = root_path

    (openmvg_dir / "data_openmvg_train.json").write_text(json.dumps(train_data, indent=2), encoding="utf-8")
    (openmvg_dir / "data_openmvg_test.json").write_text(json.dumps(test_data, indent=2), encoding="utf-8")

    # 点云：优先 colorized.ply，否则 cloud_and_poses 或 reconstruction 下 ply
    for name in ("colorized.ply", "cloud_and_poses.ply", "scene.ply"):
        src = openmvg_out / "reconstruction" / name
        if not src.exists():
            src = openmvg_out / name
        if src.exists():
            shutil.copy2(src, openmvg_dir / "scene.ply")
            break
    else:
        print("未找到点云 .ply，请手动将重建点云复制为 openMVG/scene.ply")

    print("已写入:", openmvg_dir / "data_openmvg_train.json", openmvg_dir / "data_openmvg_test.json")


if __name__ == "__main__":
    main()
