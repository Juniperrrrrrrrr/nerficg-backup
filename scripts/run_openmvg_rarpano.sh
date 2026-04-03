#!/usr/bin/env bash
# 用 OpenMVG 对 RaRPano / Ricoh360 等距柱状 360° 场景做 SfM，并整理成 Ricoh360 loader 可读的 openMVG 目录结构。
# 用法: ./run_openmvg_rarpano.sh /path/to/scene [test_step]
# test_step: 每 N 张取 1 张为测试集，默认 8；0 表示用比例划分。
# 要求: 已安装 OpenMVG，场景下存在 images/ 目录。
# 位姿/糊图: 默认用几何模型 a（angular，球面），若仍糊可设 OPENMVG_FEATURE_PRESET=ULTRA 再跑。

set -e
SCENE_ROOT="${1:?用法: $0 /path/to/rarpano_scene}"
TEST_STEP="${2:-8}"
IMAGE_DIR="${SCENE_ROOT}/images"
OUTPUT_DIR="${SCENE_ROOT}/openmvg_out"
OPENMVG_DIR="${SCENE_ROOT}/openMVG"

# OpenMVG 可执行文件目录（若未装到 PATH）
if [ -n "$OPENMVG_BIN" ]; then
  OPENMVG_MAIN_SFMINIT="${OPENMVG_BIN}/openMVG_main_SfMInit_ImageListing"
  OPENMVG_COMPUTE_FEATURES="${OPENMVG_BIN}/openMVG_main_ComputeFeatures"
  OPENMVG_PAIR_GENERATOR="${OPENMVG_BIN}/openMVG_main_PairGenerator"
  OPENMVG_COMPUTE_MATCHES="${OPENMVG_BIN}/openMVG_main_ComputeMatches"
  OPENMVG_GEOMETRIC_FILTER="${OPENMVG_BIN}/openMVG_main_GeometricFilter"
  OPENMVG_SFM="${OPENMVG_BIN}/openMVG_main_SfM"
  OPENMVG_INCREMENTAL_SFM="${OPENMVG_BIN}/openMVG_main_IncrementalSfM"
  OPENMVG_CONVERT="${OPENMVG_BIN}/openMVG_main_ConvertSfM_DataFormat"
else
  OPENMVG_MAIN_SFMINIT="openMVG_main_SfMInit_ImageListing"
  OPENMVG_COMPUTE_FEATURES="openMVG_main_ComputeFeatures"
  OPENMVG_PAIR_GENERATOR="openMVG_main_PairGenerator"
  OPENMVG_COMPUTE_MATCHES="openMVG_main_ComputeMatches"
  OPENMVG_GEOMETRIC_FILTER="openMVG_main_GeometricFilter"
  OPENMVG_SFM="openMVG_main_SfM"
  OPENMVG_INCREMENTAL_SFM="openMVG_main_IncrementalSfM"
  OPENMVG_CONVERT="openMVG_main_ConvertSfM_DataFormat"
fi

# 几何过滤: a=angular（球面，360° 等距柱状图必须用此，位姿才正确）/ f=fundamental（针孔，球面下易错位、糊图）
OPENMVG_GEOMETRIC_MODEL="${OPENMVG_GEOMETRIC_MODEL:-a}"
# 特征预设: NORMAL（快）/ HIGH（更准，推荐）/ ULTRA（最准最慢）
OPENMVG_FEATURE_PRESET="${OPENMVG_FEATURE_PRESET:-HIGH}"

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

# 在场景根目录下用相对路径运行，这样 sfm_data 里是相对路径，ComputeMatches 才能找到 .feat 文件
cd "$SCENE_ROOT"
mkdir -p openmvg_out
# 使用相对路径；-c 7 = spherical（与 Ricoh360 一致，不再用 pinhole 近似）
INIT_ARGS=(-i images -o openmvg_out -c 7 -f "$FOCAL_PIX")
[ -n "$SENSOR_DB" ] && INIT_ARGS+=(-d "$SENSOR_DB")
if ! "$OPENMVG_MAIN_SFMINIT" "${INIT_ARGS[@]}"; then
  echo "SfMInit_ImageListing 失败"
  exit 1
fi

# 2) 特征：输出到与 sfm_data 同一目录，ComputeMatches 按此目录找 .feat/.desc
#    -p HIGH 提高匹配质量，便于 -g f 时也能通过初始像对（位姿往往更稳）
if ! "$OPENMVG_COMPUTE_FEATURES" -i openmvg_out/sfm_data.json -o openmvg_out -m SIFT -f 1 -p "$OPENMVG_FEATURE_PRESET"; then
  echo "ComputeFeatures 失败"
  exit 1
fi

# 3) 生成匹配对（与官方流程一致）
if command -v "$OPENMVG_PAIR_GENERATOR" &>/dev/null || type "$OPENMVG_PAIR_GENERATOR" &>/dev/null 2>&1; then
  if ! "$OPENMVG_PAIR_GENERATOR" -i openmvg_out/sfm_data.json -o openmvg_out/pairs.bin; then
    echo "PairGenerator 失败，将尝试无 -p 的 ComputeMatches"
  fi
fi

# 4) 匹配：-o 为输出文件（.bin），不是目录
MATCHES_FILE="openmvg_out/matches.putative.bin"
if [ -f "openmvg_out/pairs.bin" ]; then
  if ! "$OPENMVG_COMPUTE_MATCHES" -i openmvg_out/sfm_data.json -p openmvg_out/pairs.bin -o "$MATCHES_FILE"; then
    echo "ComputeMatches 失败"
    exit 1
  fi
else
  if ! "$OPENMVG_COMPUTE_MATCHES" -i openmvg_out/sfm_data.json -o "$MATCHES_FILE"; then
    echo "ComputeMatches 失败"
    exit 1
  fi
fi

# 4b) 几何过滤：-g f=fundamental（位姿常更稳），-g a=angular（球面）；输出 matches.f.bin 供 SfM 读取
if ! "$OPENMVG_GEOMETRIC_FILTER" -i openmvg_out/sfm_data.json -m openmvg_out/matches.putative.bin -g "$OPENMVG_GEOMETRIC_MODEL" -o openmvg_out/matches.f.bin; then
  echo "GeometricFilter 失败（当前 -g $OPENMVG_GEOMETRIC_MODEL；球面 360 请用 a，若仍失败可试 OPENMVG_GEOMETRIC_MODEL=f）"
  exit 1
fi

# 5) 增量 SfM（优先用 openMVG_main_SfM --sfm_engine INCREMENTAL，新版本常用此入口）
# 不再重定向 stderr，便于看到 SfM 真实报错
if "$OPENMVG_SFM" --sfm_engine INCREMENTAL --input_file openmvg_out/sfm_data.json --match_dir openmvg_out --output_dir openmvg_out/reconstruction; then
  echo "SfM 完成（openMVG_main_SfM）"
elif [ -x "$OPENMVG_INCREMENTAL_SFM" ] && "$OPENMVG_INCREMENTAL_SFM" -i openmvg_out/sfm_data.json -m openmvg_out -o openmvg_out/reconstruction; then
  echo "SfM 完成（openMVG_main_IncrementalSfM）"
else
  echo "错误: SfM 步骤失败。若上方有 OpenMVG 报错，请根据报错排查；否则请设置 OPENMVG_BIN 或将 OpenMVG bin 加入 PATH。"
  exit 1
fi

# 6) 导出 JSON（若为 bin 则先转换）
SFM_BIN="$SCENE_ROOT/openmvg_out/reconstruction/sfm_data.bin"
SFM_JSON="$SCENE_ROOT/openmvg_out/reconstruction/sfm_data.json"
if [ -f "$SFM_BIN" ] && [ ! -f "$SFM_JSON" ]; then
  "$OPENMVG_CONVERT" -i "$SFM_BIN" -o "$SFM_JSON" -V -I -E
fi

# 7) 整理为 Ricoh360 所需结构：openMVG/、data_openmvg_train/test.json、scene.ply、imgs
mkdir -p "$OPENMVG_DIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/convert_openmvg_to_ricoh360_json.py" ]; then
  if ! python3 "$SCRIPT_DIR/convert_openmvg_to_ricoh360_json.py" \
    --scene_root "$SCENE_ROOT" \
    --openmvg_out "$SCENE_ROOT/openmvg_out" \
    --test_step "$TEST_STEP"; then
    echo "转换脚本执行失败。可手动运行："
    echo "  python3 $SCRIPT_DIR/convert_openmvg_to_ricoh360_json.py --scene_root $SCENE_ROOT --openmvg_out $SCENE_ROOT/openmvg_out --test_step $TEST_STEP"
    exit 1
  fi
  if [ ! -f "$OPENMVG_DIR/data_openmvg_train.json" ] || [ ! -f "$OPENMVG_DIR/data_openmvg_test.json" ]; then
    echo "警告: openMVG 下未生成 train/test.json，请检查上方报错并手动运行上述 python3 命令。"
  fi
else
  echo "未找到 convert_openmvg_to_ricoh360_json.py（当前查找: $SCRIPT_DIR/convert_openmvg_to_ricoh360_json.py），请手动："
  echo "  1) 将 $SCENE_ROOT/openmvg_out/reconstruction/ 中的 sfm_data 转为 data_openmvg_train/test.json 放入 $OPENMVG_DIR"
  echo "  2) 将点云复制为 $OPENMVG_DIR/scene.ply"
  echo "  3) 在 $SCENE_ROOT 下建立 imgs -> images 的符号链接"
fi

echo "完成。数据集路径: $SCENE_ROOT （在配置中使用 Ricoh360 类）"
