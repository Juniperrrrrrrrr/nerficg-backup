#!/usr/bin/env bash
# 用 OpenMVG 对 RaRPano 场景做 SfM，并整理成 Ricoh360  loader 可读的 openMVG 目录结构。
# 用法: ./run_openmvg_rarpano.sh /path/to/rarpano_scene
# 要求: 已安装 OpenMVG，场景下存在 images/ 目录。

set -e
SCENE_ROOT="${1:?用法: $0 /path/to/rarpano_scene}"
IMAGE_DIR="${SCENE_ROOT}/images"
OUTPUT_DIR="${SCENE_ROOT}/openmvg_out"
OPENMVG_DIR="${SCENE_ROOT}/openMVG"
IMGS_LINK="${SCENE_ROOT}/imgs"

# OpenMVG 可执行文件目录（若未装到 PATH）
if [ -n "$OPENMVG_BIN" ]; then
  OPENMVG_MAIN_SFMINIT="${OPENMVG_BIN}/openMVG_main_SfMInit_ImageListing"
  OPENMVG_COMPUTE_FEATURES="${OPENMVG_BIN}/openMVG_main_ComputeFeatures"
  OPENMVG_COMPUTE_MATCHES="${OPENMVG_BIN}/openMVG_main_ComputeMatches"
  OPENMVG_INCREMENTAL_SFM="${OPENMVG_BIN}/openMVG_main_IncrementalSfM"
  OPENMVG_CONVERT="${OPENMVG_BIN}/openMVG_main_ConvertSfM_DataFormat"
else
  OPENMVG_MAIN_SFMINIT="openMVG_main_SfMInit_ImageListing"
  OPENMVG_COMPUTE_FEATURES="openMVG_main_ComputeFeatures"
  OPENMVG_COMPUTE_MATCHES="openMVG_main_ComputeMatches"
  OPENMVG_INCREMENTAL_SFM="openMVG_main_IncrementalSfM"
  OPENMVG_CONVERT="openMVG_main_ConvertSfM_DataFormat"
fi

# 传感器数据库（可选，equirect 用 -f 时可不依赖）
SENSOR_DB="${SENSOR_DB:-/usr/share/openMVG/sensor_width_camera_database.txt}"
if [ ! -f "$SENSOR_DB" ]; then
  SENSOR_DB=""
fi

if [ ! -d "$IMAGE_DIR" ]; then
  echo "错误: 未找到图像目录 $IMAGE_DIR"
  exit 1
fi

# 取第一张图尺寸，计算等距柱状图焦距: focal = width / (2*pi)
FIRST_IMG=$(find "$IMAGE_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | head -1)
if [ -z "$FIRST_IMG" ]; then
  echo "错误: $IMAGE_DIR 下没有 jpg/png 图像"
  exit 1
fi
if command -v python3 &>/dev/null; then
  read -r WIDTH HEIGHT <<< $(python3 -c "
import sys
try:
    from PIL import Image
    with Image.open('$FIRST_IMG') as im:
        print(im.size[0], im.size[1])
except Exception:
    print('0 0')
")
else
  echo "请安装 Python3 或先设置好焦距，并在下方脚本中手动设置 FOCAL_PIX"
  exit 1
fi
if [ -z "$WIDTH" ] || [ "$WIDTH" = "0" ]; then
  echo "无法读取图像尺寸，请安装 Pillow: pip install Pillow"
  exit 1
fi
# 等距柱状 360° 水平对应整幅宽度: focal = width / (2*pi)
FOCAL_PIX=$(python3 -c "import math; print(int(round($WIDTH / (2 * math.pi))))")
echo "图像尺寸: ${WIDTH}x${HEIGHT}, 使用焦距(像素): $FOCAL_PIX"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# 1) 图像列表与内参（equirectangular 用 pinhole + 固定焦距）
INIT_ARGS=(-i "$IMAGE_DIR" -o "$OUTPUT_DIR" -c 1 -f "$FOCAL_PIX")
[ -n "$SENSOR_DB" ] && INIT_ARGS+=(-d "$SENSOR_DB")
if ! "$OPENMVG_MAIN_SFMINIT" "${INIT_ARGS[@]}"; then
  echo "SfMInit_ImageListing 失败"
  exit 1
fi

# 2) 特征
if ! "$OPENMVG_COMPUTE_FEATURES" -i "$OUTPUT_DIR/sfm_data.json" -o "$OUTPUT_DIR/matches" -m AKAZE_FLOAT; then
  echo "ComputeFeatures 失败，可尝试 -m SIFT 或 -m AKAZE"
  exit 1
fi

# 3) 匹配
if ! "$OPENMVG_COMPUTE_MATCHES" -i "$OUTPUT_DIR/sfm_data.json" -o "$OUTPUT_DIR/matches" -g e; then
  echo "ComputeMatches 失败"
  exit 1
fi

# 4) 增量 SfM
if ! "$OPENMVG_INCREMENTAL_SFM" -i "$OUTPUT_DIR/sfm_data.json" -m "$OUTPUT_DIR/matches" -o "$OUTPUT_DIR/reconstruction"; then
  echo "IncrementalSfM 失败"
  exit 1
fi

# 5) 导出 JSON（若为 bin 则先转换）
SFM_BIN="$OUTPUT_DIR/reconstruction/sfm_data.bin"
SFM_JSON="$OUTPUT_DIR/reconstruction/sfm_data.json"
if [ -f "$SFM_BIN" ] && [ ! -f "$SFM_JSON" ]; then
  "$OPENMVG_CONVERT" -i "$SFM_BIN" -o "$SFM_JSON" -V -I -E
fi

# 6) 整理为 Ricoh360 所需结构：openMVG/、data_openmvg_train/test.json、scene.ply、imgs
mkdir -p "$OPENMVG_DIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/convert_openmvg_to_ricoh360_json.py" ]; then
  python3 "$SCRIPT_DIR/convert_openmvg_to_ricoh360_json.py" \
    --scene_root "$SCENE_ROOT" \
    --openmvg_out "$OUTPUT_DIR" \
    --train_ratio 0.8
else
  echo "未找到 convert_openmvg_to_ricoh360_json.py，请手动："
  echo "  1) 将 $OUTPUT_DIR/reconstruction/ 中的 sfm_data.json 转为 data_openmvg_train/test.json 放入 $OPENMVG_DIR"
  echo "  2) 将点云复制为 $OPENMVG_DIR/scene.ply"
  echo "  3) 在 $SCENE_ROOT 下建立 imgs -> images 的符号链接"
fi

if [ ! -e "$IMGS_LINK" ]; then
  ln -sfn images "$IMGS_LINK"
  echo "已创建符号链接: $IMGS_LINK -> images"
fi

echo "完成。数据集路径: $SCENE_ROOT （在配置中使用 Ricoh360 类）"
