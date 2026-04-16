# SPaGS + 3DGS-MCMC 修改记录

本文档详细记录了将 3DGS-MCMC 改进点迁移到 SPaGS 的所有修改。

## 修改概览

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `src/Methods/SPaGS/reloc_utils.py` | 新增 | MCMC 辅助函数（CUDA relocation 计算） |
| `src/Optim/AdamUtils.py` | 修改 | 添加 `reset_optimizer_state` 函数 |
| `src/Methods/SPaGS/Model.py` | 修改 | 添加 MCMC 核心方法（relocate, add_new_gs, noise） |
| `src/Methods/SPaGS/Trainer.py` | 修改 | 添加 MCMC 训练逻辑和配置参数 |

---

## 1. 新增文件：`src/Methods/SPaGS/reloc_utils.py`

### 功能
提供 MCMC relocation 的 CUDA 辅助计算。

### 主要内容

#### `compute_relocation_cuda(opacity_old, scale_old, N)`
- **功能**：计算 relocation 后的新 opacity 和 scale
- **公式**（来自 3DGS-MCMC 论文 Equation 9）：
  - 新 opacity: $o_{new} = o_{old} / N$
  - 新 scale: $s_{new} = s_{old} \times \sqrt{N}$
- **参数**：
  - `opacity_old`: 父节点原始 opacity
  - `scale_old`: 父节点原始 scale (3D)
  - `N`: 采样次数 + 1

#### `op_sigmoid(x, k=100.0, x0=0.995)`
- **功能**：自适应噪声的 sigmoid 缩放函数
- **用途**：控制噪声强度，低 opacity 点获得更大噪声
- **公式**：$\sigma(-k(t-o))$，其中 $k=100$, $t=0.995$

---

## 2. 修改文件：`src/Optim/AdamUtils.py`

### 修改内容
在文件末尾添加 `reset_optimizer_state` 函数：

```python
def reset_optimizer_state(optimizer: torch.optim.Optimizer, indices: torch.Tensor | None = None) -> None:
    """Reset Adam optimizer state (exp_avg and exp_avg_sq) for specified indices."""
```

### 用途
在 MCMC relocation 后重置优化器的动量统计（exp_avg, exp_avg_sq），避免旧动量影响新参数。

### 调用位置
- `Model.py` 的 `relocate_gs()` 和 `add_new_gs()` 方法

---

## 3. 修改文件：`src/Methods/SPaGS/Model.py`

### 3.1 导入修改

**原代码：**
```python
from Optim.AdamUtils import replace_param_group_data, prune_param_groups, extend_param_groups
```

**修改为：**
```python
from Optim.AdamUtils import replace_param_group_data, prune_param_groups, extend_param_groups, reset_optimizer_state
from Methods.SPaGS.reloc_utils import compute_relocation_cuda, op_sigmoid
```

### 3.2 新增方法

#### `_sample_alives(probs, num, alive_indices=None)`
- **功能**：基于概率采样父节点
- **算法**：`torch.multinomial` 多项式采样（有放回）
- **用途**：Relocation 和 add_new_gs 中选择父节点
- **关键**：高 opacity 点更可能被选中

#### `_update_params_for_relocation(idxs, ratio)`
- **功能**：计算 relocation 后的新参数
- **实现**：调用 `compute_relocation_cuda` 计算新 opacity/scale
- **返回**：(positions, sh_0, sh_rest, new_opacities, new_scales, rotations)

#### `relocate_gs(dead_mask)`
- **功能**：MCMC Relocation（论文 3.4 节）
- **流程**：
  1. 识别死亡点（opacity < threshold）
  2. 基于 opacity 采样存活点作为父节点
  3. 更新死亡点参数（继承父节点但调整 opacity/scale）
  4. 同步更新父节点参数（保持总概率质量）
  5. 重置优化器状态

#### `add_new_gs(cap_max)`
- **功能**：动态添加新高斯点（论文 3.6 节）
- **流程**：
  1. 计算目标数量：`min(cap_max, 1.05 * current_num)`（增长 5%）
  2. 基于 opacity 采样父节点
  3. 计算子节点参数（分裂）
  4. 更新父节点参数
  5. 添加子节点到优化器
  6. 重置优化器状态

#### `add_mcmc_noise(iteration, position_lr, noise_lr)`
- **功能**：添加 SGLD 噪声（论文 3.3 节，Equation 7-8）
- **公式**：$\epsilon_\mu = \lambda_{noise} \cdot \lambda_{lr} \cdot \sigma(-k(t-o)) \cdot \Sigma \eta$
- **实现步骤**：
  1. 构建 3D 协方差矩阵 `covariance = L @ L.T`
  2. 计算自适应缩放 `op_sigmoid(1 - opacity)`
  3. 生成标准正态噪声 `eta ~ N(0, I)`
  4. 应用协方差变换 `covariance @ noise`
  5. 缩放到最终噪声强度
  6. 添加到位置参数

---

## 4. 修改文件：`src/Methods/SPaGS/Trainer.py`

### 4.1 新增配置参数

在 `@Framework.Configurable.configure` 中添加 MCMC 参数：

```python
# ==========================================================================
# MCMC Parameters (adapted from 3DGS-MCMC)
# ==========================================================================
USE_MCMC=False,           # 是否启用 MCMC 模式
CAP_MAX=-1,               # 高斯点数量上限（必须设置，-1 表示不启用 MCMC）
NOISE_LR=5e5,             # SGLD 噪声学习率 (lambda_noise)
OPACITY_REG=0.01,         # Opacity 正则化系数 (lambda_o)
SCALE_REG=0.01,           # Scale 正则化系数 (lambda_Sigma)
MCMC_OPACITY_THRESHOLD=0.005,  # MCMC 死亡点判定阈值
```

### 4.2 修改 `trainingIteration` 方法

**添加 MCMC 正则化（Equation 10）：**

```python
if self.USE_MCMC:
    gaussians = self.model.gaussians
    # Opacity regularization: lambda_o * sum(|o_i|)
    loss = loss + self.OPACITY_REG * gaussians.get_opacities.abs().mean()
    # Scale regularization: lambda_Sigma * sum(|scale|)
    loss = loss + self.SCALE_REG * gaussians.get_scales.abs().mean()
```

### 4.3 新增 MCMC 回调

#### `densifyMCMC(iteration, dataset)`
- **优先级**：92（高于传统 densify 的 90）
- **激活条件**：`USE_MCMC=True`
- **功能**：
  1. 调用 `relocate_gs()` 处理死亡点
  2. 调用 `add_new_gs(cap_max)` 动态增长
  3. 更新 3D filter（如果启用）

#### `addMCMCNoise(iteration, dataset)`
- **优先级**：88（在 densify 之后，optimizer step 之前）
- **激活条件**：`USE_MCMC=True`
- **功能**：在 optimizer step 后添加 SGLD 噪声

### 4.4 修改传统回调的激活条件

禁用传统 densification 当 MCMC 启用时：

| 回调 | 原激活条件 | 新激活条件 |
|------|-----------|-----------|
| `densify` | 无 | `not USE_MCMC` |
| `resetOpacities` | `USE_OPACITY_RESET` | `USE_OPACITY_RESET and not USE_MCMC` |
| `decayOpacities` | `USE_OPACITY_DECAY` | `USE_OPACITY_DECAY and not USE_MCMC` |

---

## 训练命令

### 方式 1：使用 YAML 配置文件（推荐）

SPaGS 使用 YAML 配置文件管理参数。我创建了示例配置 `configs/I_garden_mcmc.yaml`：

```yaml
TRAINING:
    # ... 其他原有参数 ...
    
    # MCMC 参数（新增）
    USE_MCMC: true              # 启用 MCMC
    CAP_MAX: 7000000            # 高斯点上限（7M）
    NOISE_LR: 500000.0          # SGLD 噪声学习率
    OPACITY_REG: 0.01           # Opacity 正则化
    SCALE_REG: 0.01             # Scale 正则化
    MCMC_OPACITY_THRESHOLD: 0.005  # 死亡点判定阈值
```

运行训练：

```bash
python scripts/train.py configs/I_garden_mcmc.yaml
```

### 方式 2：基于现有配置修改

复制现有配置并添加 MCMC 参数：

```bash
# 复制现有配置
cp configs/I_avenue.yaml configs/I_avenue_mcmc.yaml

# 编辑 configs/I_avenue_mcmc.yaml，在 TRAINING 部分添加：
# USE_MCMC: true
# CAP_MAX: 7000000
# ... 其他 MCMC 参数

# 运行
python scripts/train.py configs/I_avenue_mcmc.yaml
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `USE_MCMC` | `False` | 必须设为 `True` 启用 MCMC |
| `CAP_MAX` | `-1` | **必须设置正数**，推荐 1M-5M |
| `NOISE_LR` | `5e5` | 噪声强度，论文默认值 |
| `OPACITY_REG` | `0.01` | Opacity L1 正则系数 |
| `SCALE_REG` | `0.01` | Scale L1 正则系数 |
| `MCMC_OPACITY_THRESHOLD` | `0.005` | 死亡点判定阈值 |

### CAP_MAX 获取建议

`CAP_MAX` 是 MCMC 的核心参数，需要根据场景设置：

| 场景类型 | 推荐 CAP_MAX | 说明 |
|---------|-------------|------|
| 小场景（室内） | 500K - 1M | 细节丰富但范围小 |
| 中等场景（花园） | 1M - 2M | 平衡质量和效率 |
| 大场景（城市） | 2M - 5M | 需要更多高斯表示 |
| 超大规模 | 5M+ | 需要更多显存 |

**显存估算**：每个高斯点约 100-200 字节，1M 点约 200MB 显存。

**建议**：从 1M 开始，如果显存允许且质量不足，逐步增加。

---

## 与传统 SPaGS 的对比

| 特性 | 传统 SPaGS | SPaGS + MCMC |
|------|-----------|--------------|
| Densification | clone + split + prune | relocate + dynamic growth |
| 点数控制 | 无明确上限 | 严格上限 `cap_max` |
| 死亡点处理 | 直接删除 | 重定位到存活点 |
| 噪声 | 无 | SGLD 自适应噪声 |
| 正则化 | 无 | Opacity + Scale L1 |
| 调参难度 | 需要调 densify 阈值 | 主要调 `cap_max` |

---

## 验证清单

训练前检查：
- [ ] `reloc_utils.py` 已创建
- [ ] `AdamUtils.py` 已添加 `reset_optimizer_state`
- [ ] `Model.py` 已导入 `reloc_utils`
- [ ] `Trainer.py` 已设置 `USE_MCMC=True` 和 `CAP_MAX`
- [ ] 显存足够容纳 `CAP_MAX` 个高斯点

训练中检查：
- [ ] 日志显示 "MCMC densify: relocated X, added Y"
- [ ] 点数单调增长到 `cap_max` 后稳定
- [ ] 没有 OOM 错误

---

## 环境配置

### 使用 nerficg 环境

使用 **nerficg** 环境（SPaGS 原环境），不需要 3dgs-mcmc-env。

### CUDA 扩展编译

修改后的代码需要重新编译 SPaGS CUDA Backend：

```bash
cd src/Methods/SPaGS/SPaGSCudaBackend
rm -rf build
python setup.py build_ext --inplace

# 复制编译好的 .so 文件到模块目录
cp build/lib.linux-x86_64-cpython-311/SPaGSCudaBackend/*.so SPaGSCudaBackend/
```

### 验证 CUDA 安装

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from Methods.SPaGS.reloc_utils import HAS_CUDA_RELOC
print('CUDA available:', HAS_CUDA_RELOC)  # 应输出 True
"
```

---

## 参考文献

- 3DGS-MCMC: "3D Gaussian Splatting with MCMC" (arXiv:2404.09591)
- SPaGS: 原 SPaGS 实现
