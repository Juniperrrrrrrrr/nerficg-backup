# -- coding: utf-8 --
# 指定文件编码为UTF-8，确保中文字符能正常显示

"""SPaGS/Loss.py: Loss function."""  # 文件说明：SPaGS的损失函数模块

# 导入PyTorch深度学习框架
import torch
# 导入PyTorch度量标准模块，用于计算图像质量指标
import torchmetrics

# 从框架中导入配置参数列表类
from Framework import ConfigParameterList
# 从优化器损失模块导入基础损失类
from Optim.Losses.Base import BaseLoss
# 从优化器损失模块导入融合DSSIM损失函数
from Optim.Losses.FusedDSSIM import fused_dssim


# 定义SPaGS的损失函数类，继承自BaseLoss基类
class SPaGSLoss(BaseLoss):
    # 初始化方法，接收损失配置参数
    def __init__(self, loss_config: ConfigParameterList) -> None:
        # 调用父类BaseLoss的初始化方法
        super().__init__()

        # 添加L1颜色损失（绝对误差损失）
        # 参数说明：
        # 'L1_Color': 损失名称
        # torch.nn.functional.l1_loss: PyTorch的L1损失函数
        # loss_config.LAMBDA_L1: L1损失的权重系数（λ_L1）
        self.addLossMetric('L1_Color', torch.nn.functional.l1_loss, loss_config.LAMBDA_L1)

        # 添加DSSIM颜色损失（结构相似性损失）
        # 参数说明：
        # 'DSSIM_Color': 损失名称
        # fused_dssim: 融合的DSSIM损失函数（可能用CUDA加速）
        # loss_config.LAMBDA_DSSIM: DSSIM损失的权重系数（λ_DSSIM）
        self.addLossMetric('DSSIM_Color', fused_dssim, loss_config.LAMBDA_DSSIM)

        # 添加PSNR质量指标（峰值信噪比，用于评估图像质量，不参与梯度计算）
        # 参数说明：
        # 'PSNR': 指标名称
        # torchmetrics.functional.image.peak_signal_noise_ratio: PSNR计算函数
        # 注意：这是质量指标，不是损失函数，不参与反向传播
        self.addQualityMetric('PSNR', torchmetrics.functional.image.peak_signal_noise_ratio)

    # 前向传播方法，计算总损失
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 调用父类的forward方法，传入计算参数
        return super().forward({
            # L1损失的计算参数：input为预测图像，target为真实图像
            'L1_Color': {'input': input, 'target': target},
            # DSSIM损失的计算参数：同样使用预测图像和真实图像
            'DSSIM_Color': {'input': input, 'target': target},
            # PSNR指标的计算参数：preds为预测图像，target为真实图像，data_range=1.0表示像素值范围
            'PSNR': {'preds': input, 'target': target, 'data_range': 1.0}
        })