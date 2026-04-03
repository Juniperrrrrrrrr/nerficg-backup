# -- coding: utf-8 --

"""SPaGS/Trainer.py: Implementation of the trainer for SPaGS."""

import torch

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud
from Logging import Logger
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import preTrainingCallback, trainingCallback, postTrainingCallback
from Methods.SPaGS.Loss import SPaGSLoss
from Optim.Samplers.DatasetSamplers import DatasetSampler


def _export_spags_gaussians_to_point_cloud_ply(gaussians, ply_path, sh_degree: int, baked: bool) -> None:
    """
    导出 SPaGS 的高斯为 gaussian-splatting/SIBR 兼容的 `point_cloud.ply`。

    顶点属性顺序与 `gaussian-splatting/scene/gaussian_model.py` 的 `save_ply()` 保持一致：
    `x y z nx ny nz f_dc_* f_rest_* opacity scale_* rot_*`。
    
    注意：标准 gaussian-splatting 的 PLY 格式保存的是 raw 参数值（用于加载时重新应用激活函数）：
    - opacity: inverse_sigmoid 空间（通过 sigmoid 得到实际透明度）
    - scales: log 空间（通过 exp 得到实际尺度）
    - rotations: 四元数（应归一化）
    """
    import numpy as np
    import torch
    from plyfile import PlyData, PlyElement

    from Methods.GaussianSplatting.utils import inverse_sigmoid

    positions = gaussians._positions.detach().cpu().numpy().astype(np.float32)  # (N,3)
    normals = np.zeros_like(positions)

    sh0 = gaussians._sh_0.detach().cpu().numpy().astype(np.float32)  # (N,1,3)
    f_dc = sh0[:, 0, :]  # (N,3)

    sh_rest = gaussians._sh_rest.detach().cpu().numpy().astype(np.float32)  # (N,num_rest,3)
    num_rest = sh_rest.shape[1]
    expected_num_rest = (sh_degree + 1) ** 2 - 1
    if num_rest != expected_num_rest:
        raise ValueError(
            f"Unexpected sh_rest coeffs: got {num_rest}, expected {expected_num_rest} (sh_degree={sh_degree})."
        )

    # Match gaussian-splatting ordering:
    # f_rest = _features_rest.transpose(1,2).flatten(start_dim=1)
    # Here _features_rest is (N,num_rest,3) -> transpose -> (N,3,num_rest) -> flatten -> (N,3*num_rest)
    f_rest = sh_rest.transpose(0, 2, 1).reshape(sh_rest.shape[0], -1)  # (N,3*num_rest)

    N = positions.shape[0]

    if baked:
        # After bake_activations:
        # - _opacities 是实际概率值 (0-1 范围)
        # - _scales 是实际尺度值 (正数)
        # - _rotations 已经是归一化的四元数
        # PLY 格式需要 raw 空间的数据，所以这里需要转换回去
        opacity_actual = gaussians._opacities.detach().cpu().clamp(1e-6, 1.0 - 1e-6)
        opacity_raw = inverse_sigmoid(opacity_actual).numpy().astype(np.float32).reshape(N, 1)

        scales_actual = gaussians._scales.detach().cpu().clamp_min(1e-6)
        scales_raw = torch.log(scales_actual).numpy().astype(np.float32)  # (N,3)
        
        # rotations 已经归一化，直接保存
        rotations_raw = gaussians._rotations.detach().cpu().numpy().astype(np.float32)  # (N,4)
    else:
        # Initial: _opacities 和 _scales 是 raw 参数
        opacity_raw = gaussians._opacities.detach().cpu().numpy().astype(np.float32).reshape(N, 1)
        scales_raw = gaussians._scales.detach().cpu().numpy().astype(np.float32)  # (N,3)
        # 未 baked 时，四元数需要归一化
        rotations = torch.nn.functional.normalize(gaussians._rotations, dim=-1)
        rotations_raw = rotations.detach().cpu().numpy().astype(np.float32)  # (N,4)

    num_f_rest = f_rest.shape[1]
    if num_f_rest != 3 * num_rest:
        raise ValueError(f"Unexpected f_rest dim: got {num_f_rest}, expected {3 * num_rest}.")

    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
    ]
    dtype_full += [(f'f_dc_{i}', 'f4') for i in range(3)]
    dtype_full += [(f'f_rest_{i}', 'f4') for i in range(num_f_rest)]
    dtype_full += [('opacity', 'f4')]
    dtype_full += [(f'scale_{i}', 'f4') for i in range(3)]
    dtype_full += [(f'rot_{i}', 'f4') for i in range(4)]

    elements = np.empty(N, dtype=np.dtype(dtype_full))

    elements['x'] = positions[:, 0]
    elements['y'] = positions[:, 1]
    elements['z'] = positions[:, 2]
    elements['nx'] = 0.0
    elements['ny'] = 0.0
    elements['nz'] = 0.0

    for i in range(3):
        elements[f'f_dc_{i}'] = f_dc[:, i]
    for i in range(num_f_rest):
        elements[f'f_rest_{i}'] = f_rest[:, i]

    elements['opacity'] = opacity_raw[:, 0]
    for i in range(3):
        elements[f'scale_{i}'] = scales_raw[:, i]
    for i in range(4):
        elements[f'rot_{i}'] = rotations_raw[:, i]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(str(ply_path))

    # Quick header verification (helps确认 t3/t4: 字段名是否完整)
    try:
        header_bytes = b''
        with open(str(ply_path), 'rb') as f:
            while b'end_header' not in header_bytes:
                line = f.readline()
                if not line:
                    break
                header_bytes += line

        header_text = header_bytes.decode('utf-8', errors='ignore')
        prop_names = []
        for line in header_text.splitlines():
            line = line.strip()
            if line.startswith('property'):
                parts = line.split()
                if len(parts) >= 3:
                    prop_names.append(parts[-1])

        required = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_rest_0', 'opacity', 'scale_0', 'rot_0']
        missing = [k for k in required if k not in prop_names]
        if missing:
            Logger.logWarning(f'PLY header missing properties {missing} (file={ply_path})')
        else:
            f_rest_props = [n for n in prop_names if n.startswith('f_rest_')]
            expected_f_rest = 3 * ((sh_degree + 1) ** 2 - 1)
            Logger.logInfo(
                f'PLY header ok (file={ply_path}): f_rest_count={len(f_rest_props)} expected={expected_f_rest}'
            )
    except Exception as e:  # pragma: no cover
        Logger.logWarning(f'PLY header verification failed: {e}')


@Framework.Configurable.configure(
    NUM_ITERATIONS=30_000,
    LEARNING_RATE_POSITION_INIT=0.00016,
    LEARNING_RATE_POSITION_FINAL=0.0000016,
    LEARNING_RATE_POSITION_DELAY_MULT=0.01,
    LEARNING_RATE_POSITION_MAX_STEPS=30_000,
    LEARNING_RATE_FEATURE=0.0025,
    LEARNING_RATE_OPACITY=0.05,
    LEARNING_RATE_SCALING=0.005,
    LEARNING_RATE_ROTATION=0.001,
    PERCENT_DENSE=0.01,
    USE_3D_FILTER=True,
    USE_OPACITY_RESET=True,
    OPACITY_RESET_MAX_OPACITY=0.1,
    USE_OPACITY_DECAY=False,
    USE_VISIBILITY_PRUNING=False,
    VISIBILITY_PRUNING_THRESHOLD=0.01,
    USE_DISTANCE_SCALING=True,
    OPACITY_RESET_INTERVAL=3_000,
    OPACITY_THRESHOLD=0.005,
    DENSIFY_START_ITERATION=200,  # 500
    DENSIFY_END_ITERATION=15_000,
    DENSIFICATION_INTERVAL=100,
    DENSIFY_GRAD_THRESHOLD=0.00005,  # 0.0002
    REDUCE_OPACITY_AFTER_DENSIFY=0.001,  # 每次 densify 后对 opacity 的减量；设为 0 可减少被 bake 剪枝的高斯
    BAKE_MIN_OPACITY=0.00392156862,  # bake 时剪枝阈值 1/255；设为 0 不在 bake 时按透明度剪枝，保留更多高斯
    # 是否在初始化高斯时导出一份「训练前初始点云」到 PLY，方便用 Blender/CloudCompare 查看密度分布
    SAVE_INITIAL_PLY=False,
    # 初始点云 PLY 文件名；会保存在 checkpoint_directory 下
    INITIAL_PLY_NAME="initial_points.ply",
    # 训练结束、bake 之后是否导出最终高斯中心为 PLY（Blender 可直接导入）
    SAVE_FINAL_PLY=False,
    FINAL_PLY_NAME="final_points.ply",
    LOSS=Framework.ConfigParameterList(
        LAMBDA_L1=0.8,
        LAMBDA_DSSIM=0.2,
    ),
)
class SPaGSTrainer(GuiTrainer):
    """Defines the trainer for the SPaGS method."""

    def __init__(self, **kwargs) -> None:
        super(SPaGSTrainer, self).__init__(**kwargs)
        self.train_sampler = None
        self.loss = SPaGSLoss(loss_config=self.LOSS)

    @preTrainingCallback(priority=50)
    @torch.no_grad()
    def createSampler(self, _, dataset: 'BaseDataset') -> None:
        """Creates the sampler."""
        self.train_sampler = DatasetSampler(dataset=dataset.train(), random=True)

    @preTrainingCallback(priority=40)
    @torch.no_grad()
    def setupGaussians(self, _, dataset: 'BaseDataset') -> None:
        """Sets up the model."""
        dataset.train()
        camera_centers = torch.stack([camera_properties.T for camera_properties in dataset])
        radius = (1.1 * torch.max(torch.linalg.norm(camera_centers - torch.mean(camera_centers, dim=0), dim=1))).item()
        # radius = torch.linalg.norm(dataset.point_cloud.positions - torch.mean(camera_centers, dim=0), dim=1).mean().item()
        Logger.logInfo(f'Training cameras extent: {radius:.2f}')

        if dataset.point_cloud is not None:
            point_cloud = dataset.point_cloud
        else:
            n_random_points = 100_000
            min_bounds, max_bounds = dataset.getBoundingBox()
            extent = max_bounds - min_bounds
            point_cloud = BasicPointCloud(torch.rand(n_random_points, 3, dtype=torch.float32, device=min_bounds.device) * extent + min_bounds)
        self.model.gaussians.initialize_from_point_cloud(point_cloud, radius)

        # 可选：在训练真正开始前，把「初始化 Gaussians」导出为 gaussian-splatting 兼容的 PLY。
        # 这样得到的文件可直接被 SIBR 打开做可视化/诊断（查看不同区域的初始化密度）。
        if getattr(self, 'SAVE_INITIAL_PLY', False):
            try:
                from pathlib import Path
                import os

                gaussians = self.model.gaussians
                base_dir = Path(getattr(self, 'checkpoint_directory', Path('.')))
                os.makedirs(base_dir, exist_ok=True)
                ply_name = getattr(self, 'INITIAL_PLY_NAME', 'point_cloud_initial.ply')
                ply_path = base_dir / ply_name

                _export_spags_gaussians_to_point_cloud_ply(
                    gaussians=gaussians,
                    ply_path=ply_path,
                    sh_degree=gaussians.max_sh_degree,
                    baked=False,
                )
                Logger.logInfo(f'Initial Gaussians saved to: {ply_path}')
            except Exception as e:  # pragma: no cover
                Logger.logWarning(f'Could not save initial Gaussians PLY: {e}')

        self.model.gaussians.training_setup(self, dataset)

    @trainingCallback(priority=110, start_iteration=1000, iteration_stride=1000)
    @torch.no_grad()
    def increaseSHDegree(self, *_) -> None:
        """Increase the number of used SH coefficients up to a maximum degree."""
        self.model.gaussians.increase_used_sh_degree()

    @trainingCallback(active='USE_VISIBILITY_PRUNING', priority=105, start_iteration=15000, iteration_stride=1000)
    @torch.no_grad()
    def importanceBasedPruning(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Pruning from RadSplat (see https://arxiv.org/abs/2403.13806)."""
        if iteration in [16000, 24000]:
            max_blending_weights = self.renderer.computeMaxWeights(dataset.train(), threshold=self.VISIBILITY_PRUNING_THRESHOLD)
            self.model.gaussians.importance_pruning(max_blending_weights, threshold=self.VISIBILITY_PRUNING_THRESHOLD)

    @trainingCallback(priority=100)
    def trainingIteration(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Performs a training step without actually doing the optimizer step."""
        # init modes
        self.model.train()
        dataset.train()
        self.loss.train()
        # update learning rate
        self.model.gaussians.update_learning_rate(iteration + 1)
        # get random sample from dataset
        camera_properties = self.train_sampler.get(dataset=dataset)['camera_properties']
        dataset.camera.setProperties(camera_properties)
        # render sample
        image = self.renderer.renderImageTraining(
            camera=dataset.camera,
            update_densification_info=iteration <= self.DENSIFY_END_ITERATION,
            use_distance_scaling=self.USE_DISTANCE_SCALING,
        )
        # calculate loss
        loss = self.loss(image, camera_properties.rgb)
        loss.backward()

    @trainingCallback(priority=90, start_iteration='DENSIFY_START_ITERATION', end_iteration='DENSIFY_END_ITERATION', iteration_stride='DENSIFICATION_INTERVAL')
    @torch.no_grad()
    def densify(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Apply densification."""
        if iteration == self.DENSIFY_START_ITERATION:
            return
        # self.model.gaussians.densify_and_prune(self.DENSIFY_GRAD_THRESHOLD, self.OPACITY_THRESHOLD, iteration > self.OPACITY_RESET_INTERVAL)
        self.model.gaussians.densify_and_prune(self.DENSIFY_GRAD_THRESHOLD, self.OPACITY_THRESHOLD, prune_large_gaussians=False)

        if self.REDUCE_OPACITY_AFTER_DENSIFY > 0:
            self.model.gaussians.reduce_opacity(self.REDUCE_OPACITY_AFTER_DENSIFY)

        if self.USE_3D_FILTER:
            self.model.gaussians.compute_3d_filter(dataset.train())

    @trainingCallback(active='USE_OPACITY_RESET', priority=80, start_iteration='OPACITY_RESET_INTERVAL', end_iteration='DENSIFY_END_ITERATION', iteration_stride='OPACITY_RESET_INTERVAL')
    @torch.no_grad()
    def resetOpacities(self, iteration: int, _) -> None:
        """Reset opacities."""
        if iteration == self.DENSIFY_END_ITERATION:
            return
        self.model.gaussians.reset_opacities(max_opacity=self.OPACITY_RESET_MAX_OPACITY)

    @trainingCallback(active='USE_OPACITY_DECAY', priority=80, start_iteration='DENSIFY_START_ITERATION', end_iteration='DENSIFY_END_ITERATION', iteration_stride=50)
    @torch.no_grad()
    def decayOpacities(self, iteration: int, _) -> None:
        """Decay opacities."""
        if iteration == self.DENSIFY_START_ITERATION:
            return
        self.model.gaussians.decay_opacities(decay_factor=0.9995)

    @trainingCallback(active='USE_3D_FILTER', priority=75, start_iteration='DENSIFY_END_ITERATION', iteration_stride=100)
    @torch.no_grad()
    def recompute3DFilter(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Recompute 3D filter."""
        if self.DENSIFY_END_ITERATION < iteration < self.NUM_ITERATIONS - 100:
            self.model.gaussians.compute_3d_filter(dataset.train())

    @trainingCallback(priority=70)
    @torch.no_grad()
    def performOptimizerStep(self, *_) -> None:
        """Update parameters."""
        self.model.gaussians.optimizer.step()
        self.model.gaussians.optimizer.zero_grad()

    @trainingCallback(active='WANDB.ACTIVATE', priority=10, iteration_stride='WANDB.INTERVAL')
    @torch.no_grad()
    def logWandB(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Adds primitive count to default Weights & Biases logging."""
        Framework.wandb.log({
            'n_primitives': self.model.gaussians.get_positions.shape[0]
        }, step=iteration)
        # default logging
        super().logWandB(iteration, dataset)

    @postTrainingCallback(priority=1000)
    @torch.no_grad()
    def bakeActivations(self, *_) -> None:
        """Bake relevant activation functions after training."""
        self.model.gaussians.bake_min_opacity = getattr(self, 'BAKE_MIN_OPACITY', 0.00392156862)
        self.model.gaussians.bake_activations()
        n_final = self.model.gaussians.get_positions.shape[0]
        Logger.logInfo(f'Number of Gaussians after training (final): {n_final:,}')

        if getattr(self, 'SAVE_FINAL_PLY', False):
            try:
                import os
                from pathlib import Path
                gaussians = self.model.gaussians

                base_dir = Path(getattr(self, 'checkpoint_directory', Path('.')))
                os.makedirs(base_dir, exist_ok=True)
                ply_name = getattr(self, 'FINAL_PLY_NAME', 'point_cloud.ply')
                ply_path = base_dir / ply_name

                _export_spags_gaussians_to_point_cloud_ply(
                    gaussians=gaussians,
                    ply_path=ply_path,
                    sh_degree=gaussians.max_sh_degree,
                    baked=True,
                )
                Logger.logInfo(f'Final Gaussians saved to: {ply_path}')
            except Exception as e:  # pragma: no cover
                Logger.logWarning(f'Could not save final point cloud PLY: {e}')

        # delete optimizer to save memory
        self.model.gaussians.optimizer = None