#!/usr/bin/env bash
# 对 RaR/pano 下所有场景批量跑 OpenMVG SfM，并整理成 Ricoh360 可读格式。
# 训练/测试划分：每 8 张取 1 张为测试集（可通过第二参数修改）。
# 用法: ./run_openmvg_rarpano_all.sh [pano根目录] [test_step]
# 示例: ./run_openmvg_rarpano_all.sh /home/bupt803/桌面/home/OmniGS/dataset/RaR/pano
#       ./run_openmvg_rarpano_all.sh /path/to/pano 8

set -e
PANO_ROOT="${1:-/home/bupt803/桌面/home/OmniGS/dataset/RaR/pano}"
TEST_STEP="${2:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONE="${SCRIPT_DIR}/run_openmvg_rarpano.sh"

if [ ! -d "$PANO_ROOT" ]; then
  echo "错误: 目录不存在: $PANO_ROOT"
  echo "用法: $0 [RaR/pano 的路径]"
  exit 1
fi

if [ ! -x "$RUN_ONE" ]; then
  echo "错误: 未找到或不可执行: $RUN_ONE"
  exit 1
fi

count=0
for scene in "$PANO_ROOT"/*/; do
  [ -d "$scene" ] || continue
  if [ ! -d "${scene}images" ]; then
    echo "跳过（无 images/）: $scene"
    continue
  fi
  echo "========== 处理场景: $scene =========="
  if "$RUN_ONE" "$scene" "$TEST_STEP"; then
    count=$((count + 1))
  else
    echo "警告: 场景失败: $scene"
    echo "若 openmvg_out 已有内容但 openMVG 为空，可手动为该场景生成："
    echo "  python3 $SCRIPT_DIR/convert_openmvg_to_ricoh360_json.py --scene_root \"$scene\" --openmvg_out \"${scene}openmvg_out\" --test_step $TEST_STEP"
  fi
  echo ""
done

echo "完成。共成功处理 $count 个场景。"
