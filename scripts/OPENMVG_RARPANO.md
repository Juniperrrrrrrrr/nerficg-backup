# 用 OpenMVG 对 RaRPano 做 SfM 获取位姿

本文说明如何用 OpenMVG 对 RaRPano 数据集的某个场景跑 SfM，得到与 Ricoh360 兼容的位姿与点云，便于在 nerficg 中用 `Ricoh360` 数据加载器读取。

## 前提

- 已安装 **OpenMVG**（含 `openMVG_main_*` 等可执行文件）。
- 本机或 Docker 中能执行 OpenMVG 的 Python 流程（见 [OpenMVG Wiki](https://github.com/openMVG/openMVG/wiki/OpenMVG-on-your-image-dataset)）。
- RaRPano 某个场景目录结构类似：
  ```
  /path/to/rarpano_scene/
  └── images/          # 全景图，如 0000.jpg, 0001.jpg, ...
  ```
  若当前是 `images_2/`，也可先建 `images/` 或在后文脚本里改 `IMAGE_DIR`。

## 思路概述

1. **用 OpenMVG 做 SfM**：把该场景下的全景图当作“已知内参”的等距柱状图（equirectangular），用合适的焦距做图像列表初始化 → 特征 → 匹配 → 增量式 SfM，得到 `sfm_data.bin` 和点云。
2. **导出与整理**：把 `sfm_data.bin` 转为 JSON，并整理成 Ricoh360  loader 需要的目录和文件名（`openMVG/data_openmvg_*.json`、`openMVG/scene.ply`、`imgs/` 等）。

这样得到的位姿即可被项目中 `Ricoh360` 所用的 OpenMVG 格式直接读取。

## 方式一：使用 OpenMVG 官方 Python 流程（推荐先试）

OpenMVG 自带顺序/增量 SfM 流程，**等距柱状图通常用 pinhole 模型 + 手工设焦距** 即可。

### 1. 安装与路径

- 从 [openMVG](https://github.com/openMVG/openMVG) 编译安装，或使用官方 Docker。
- 记下 OpenMVG 的安装目录，例如：
  - 源码编译：`openMVG_Build/Linux-x86_64-Release/`
  - 或系统安装后：`OPENMVG_BIN` 为包含 `openMVG_main_SfMInit_ImageListing` 的目录。

### 2. 为等距柱状图指定焦距

等距柱状投影下，水平 360° 对应图像宽度，焦距（像素）常用：

```text
focal_pixels = width / (2 * π)
```

若图像尺寸为 `W x H`，可在初始化时用 **-f** 指定近似焦距，例如：

```text
-f 1.0
```

并在 **ImageListing** 阶段用 **lists.txt** 或 **已知内参** 方式为每张图设置相同的 `width, height, focal`。  
OpenMVG 的 `openMVG_main_SfMInit_ImageListing` 支持从 **lists.txt** 读入每张图的 `filename;width;height;focal;...`，这样可精确控制 equirectangular 内参。

### 3. 运行顺序 SfM 流程

在 OpenMVG 的 **software/SfM** 目录下执行（将路径换成你的）：

```bash
# 图像目录、输出目录
IMAGE_DIR="/path/to/rarpano_scene/images"
OUTPUT_DIR="/path/to/rarpano_scene/openmvg_output"

# 使用官方 Python 脚本（会调用 openMVG_main_*）
cd /path/to/openMVG_Build/software/SfM
python SfM_SequentialPipeline.py "$IMAGE_DIR" "$OUTPUT_DIR"
```

若脚本不支持直接传焦距，可在其内部或本地拷贝中修改 **SfMInit_ImageListing** 的调用，为 equirectangular 增加 **-f** 或通过 **lists.txt** 提供 `width, height, focal`（focal = width / (2*π)）。

### 4. 导出 JSON 与点云

- 将得到的 `sfm_data.bin` 转为 JSON（便于后续按 Ricoh360 格式切分 train/test）：
  ```bash
  openMVG_main_ConvertSfM_DataFormat -i "$OUTPUT_DIR/sfm_data.bin" -o "$OUTPUT_DIR/sfm_data.json" -V -I -E
  ```
- 点云：若流程已生成 `cloud_and_poses.ply` 或 `colorized.ply`，可将其复制为后续的 `scene.ply`；否则可用 OpenMVG 的 **ComputeStructureFromKnownPoses** 等步骤生成再导出 PLY。

## 方式二：使用本仓库脚本（一键流程）

在 **nerficg** 仓库中提供了脚本，在已安装 OpenMVG 的前提下，对指定 RaRPano 场景跑 SfM 并整理成 Ricoh360 所需结构。

### 1. 准备

- 设置 OpenMVG 可执行文件路径（若不在 `PATH` 中）：
  ```bash
  export OPENMVG_BIN="/path/to/openMVG/Build/Linux-x86_64-Release"
  # 或将 openMVG 的 bin 目录加入 PATH
  ```
- 场景目录需包含 `images/`（或脚本内配置的 `IMAGE_DIR`）。

### 2. 运行

```bash
cd /path/to/nerficg/scripts
./run_openmvg_rarpano.sh /path/to/rarpano_scene
```

脚本会：

- 用等距柱状图焦距（`focal = width / (2*π)`）生成 **lists.txt** 并调用 **SfMInit_ImageListing**。
- 依次执行 **ComputeFeatures**、**ComputeMatches**、**IncrementalSfM**。
- 将 `sfm_data.bin` 转为 JSON，并按比例（如 80% train / 20% test）拆成 `data_openmvg_train.json` 与 `data_openmvg_test.json`，放入 `openMVG/`。
- 将点云复制或导出为 `openMVG/scene.ply`。
- 为兼容 Ricoh360 的 `imgs/` 约定，建立 `imgs` 指向 `images` 的符号链接（或复制）。

### 3. 在 nerficg 中使用

- 数据集路径设为该场景根目录，例如：`/path/to/rarpano_scene`。
- 在配置中选择 **Ricoh360** 数据集类，即可用 OpenMVG 得到的位姿和点云进行训练/测试。

## 若 OpenMVG 导出 JSON 与 Ricoh360 格式不一致

Ricoh360 读取的 JSON 结构大致为：

- `intrinsics[0].value.ptr_wrapper.data.value0`: `width`, `height`
- `views[].value.ptr_wrapper.data`: `id_pose`, `filename`
- `extrinsics[id].value`: `rotation` (3x3), `center` (3,)

若 `openMVG_main_ConvertSfM_DataFormat` 导出的 JSON 键名或层级不同，可使用仓库中的 **convert_openmvg_to_ricoh360_json.py**，将导出结果转换为上述结构，并写回 `data_openmvg_train.json` / `data_openmvg_test.json`。

## 常见问题

- **“No intrinsic found”**：用 **-f** 或 **lists.txt** 为每张图提供焦距（equirectangular 下 `focal = width / (2*π)`）。
- **匹配过少**：可尝试 Global SfM（`SfM_GlobalPipeline.py`）或调整匹配策略；全景图重叠不足时需增加拍摄重叠或减少间隔。
- **坐标系/朝向**：若与 RaRPano 原 OpenSfM 结果不一致，可在训练前在数据加载或配置中做一次统一坐标系变换（与现有 Ricoh360 处理方式保持一致即可）。

完成以上步骤后，即可用 OpenMVG 为 RaRPano 获取位姿，并在同一套代码下用 Ricoh360 的 OpenMVG 位姿流程处理 RaRPano 场景。

---

## RaR/pano 多场景（Linux 示例）

若 RaRPano 已下载到 Linux，路径为 **`/home/bupt803/桌面/home/OmniGS/dataset/RaR/pano`**，其下每个子目录为一个场景（如 `O_lion`、`O_xxx` 等），每个场景内有 **`images/`** 目录。

### 单个场景

```bash
cd /home/bupt803/桌面/home/OmniGS/scripts   # 或你放脚本的目录

# 可选：若 OpenMVG 未加入 PATH，先设置
export OPENMVG_BIN="/path/to/openMVG/Build/Linux-x86_64-Release"

./run_openmvg_rarpano.sh /home/bupt803/桌面/home/OmniGS/dataset/RaR/pano/O_lion
```

### 批量处理所有场景

```bash
cd /home/bupt803/桌面/home/OmniGS/scripts

./run_openmvg_rarpano_all.sh /home/bupt803/桌面/home/OmniGS/dataset/RaR/pano
```

脚本会对 `pano` 下每个包含 **`images/`** 的子目录执行一次 OpenMVG 流程，并生成该场景下的 **`openMVG/`**、**`imgs`** 等，训练时用 **Ricoh360** 类即可。

### 训练时配置

数据集路径填**单个场景**的根目录，例如：

- `PATH: '/home/bupt803/桌面/home/OmniGS/dataset/RaR/pano/O_lion'`

数据集类选 **Ricoh360**。
