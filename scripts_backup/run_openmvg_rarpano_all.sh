#!/usr/bin/env bash
# 对 RaR/pano 下所有场景批量跑 OpenMVG SfM，并整理成 Ricoh360 可读格式。
# 用法: ./run_openmvg_rarpano_all.sh [pano根目录]
# 示例: ./run_openmvg_rarpano_all.sh /home/bupt803/桌面/home/OmniGS/dataset/RaR/pano

set -e
PANO_ROOT="${1:-/home/bupt803/桌面/home/OmniGS/dataset/RaR/pano}"
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
  if "$RUN_ONE" "$scene"; then
    count=$((count + 1))
  else
    echo "警告: 场景失败: $scene"
  fi
  echo ""
done

echo "完成。共成功处理 $count 个场景。"
