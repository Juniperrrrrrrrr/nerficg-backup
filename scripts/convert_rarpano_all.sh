#!/usr/bin/env bash
# 对 RaR/pano 下所有场景仅重新跑「OpenMVG → Ricoh360 JSON」转换（不跑 OpenMVG SfM）。
# 用于修正 JSON 格式后批量重新生成 data_openmvg_train/test.json。
# 用法: ./convert_rarpano_all.sh [pano根目录] [test_step] [train_step]
# 示例: ./convert_rarpano_all.sh /home/bupt803/桌面/home/OmniGS/dataset/RaR/pano
#       ./convert_rarpano_all.sh /path/to/pano 8
#       ./convert_rarpano_all.sh /path/to/pano 8 2   # test_step=8, train_step=2

set -e
PANO_ROOT="${1:-/home/bupt803/桌面/home/OmniGS/dataset/RaR/pano}"
TEST_STEP="${2:-8}"
TRAIN_STEP="${3:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERT_PY="${SCRIPT_DIR}/convert_openmvg_to_ricoh360_json.py"

if [ ! -d "$PANO_ROOT" ]; then
  echo "错误: 目录不存在: $PANO_ROOT"
  echo "用法: $0 [RaR/pano 的路径] [test_step] [train_step]"
  exit 1
fi

if [ ! -f "$CONVERT_PY" ]; then
  echo "错误: 未找到: $CONVERT_PY"
  exit 1
fi

count=0
failed=()
for scene in "$PANO_ROOT"/*/; do
  [ -d "$scene" ] || continue
  scene="${scene%/}"
  name="$(basename "$scene")"
  sfm_json="${scene}/openmvg_out/reconstruction/sfm_data.json"
  if [ ! -f "$sfm_json" ]; then
    echo "跳过（无 openmvg_out/reconstruction/sfm_data.json）: $name"
    continue
  fi
  echo "========== 转换: $name =========="
  if python3 "$CONVERT_PY" \
    --scene_root "$scene" \
    --openmvg_out "${scene}/openmvg_out" \
    --test_step "$TEST_STEP" \
    --train_step "$TRAIN_STEP"; then
    count=$((count + 1))
    echo "  OK"
  else
    echo "  失败: $name"
    failed+=("$name")
  fi
  echo ""
done

echo "完成。共成功转换 $count 个场景。"
if [ ${#failed[@]} -gt 0 ]; then
  echo "失败的场景: ${failed[*]}"
  exit 1
fi
