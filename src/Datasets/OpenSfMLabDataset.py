# -- coding: utf-8 --

"""
Datasets/RaRPano.py: Provides a dataset class for panorama scenes from the Rounding and Roaming dataset.
Data available at: https://junboli-cn.github.io/spags/.
"""

import torch
import json
import numpy as np
import math
import os
from natsort import natsorted

import Framework
from Logging import Logger
from Cameras.Equirectangular import EquirectangularCamera
from Cameras.utils import CameraProperties
from Datasets.Base import BaseDataset
from Datasets.utils import CameraCoordinateSystemsTransformations, loadImagesParallel, WorldCoordinateSystemTransformations, transformPosesPCA
from Datasets.Colmap import quaternion_to_R, storePly, fetchPly, Image, Camera


def angle_axis_to_quaternion(angle_axis: np.ndarray):
    """
    将角轴（angle-axis）表示转换为四元数（quaternion）表示
    
    参数：
        angle_axis: 角轴向量，形状为 (3,)
                   格式：旋转轴 * 旋转角度
                   angle_axis = axis * angle
    
    返回：
        quaternion: 四元数，形状为 (4,)，格式为 [qw, qx, qy, qz]
    """
    angle = np.linalg.norm(angle_axis)
    
    if angle < 1e-8:  # 接近零旋转
        return np.array([1.0, 0.0, 0.0, 0.0])  # 单位四元数
    
    axis = angle_axis / angle
    half_angle = 0.5 * angle
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)
    
    return np.array([
        cos_half,           # qw
        axis[0] * sin_half, # qx  
        axis[1] * sin_half, # qy
        axis[2] * sin_half  # qz
    ])


def read_points3D_opensfm(reconstructions):
    """
    从OpenSfM重建结果中读取3D点云数据
    
    OpenSfM是开源的结构光运动恢复结构库，用于从图像序列进行3D重建
    这个函数解析OpenSfM的输出格式，提取点云信息
    
    参数:
        reconstructions: OpenSfM格式的重建数据，通常是字典列表
                        每个字典对应一个重建的3D场景
    
    返回:
        xyzs: 点云的三维坐标，形状为 (num_points, 3)
        rgbs: 点云的颜色值，形状为 (num_points, 3)
        errors: 每个点的重投影误差，形状为 (num_points, 1)
    """
    
    # 第一步：计算总点数
    num_points = 0
    for reconstruction in reconstructions:
        num_points = num_points + len(reconstruction["points"])
        # reconstruction["points"] 是字典，键是点ID，值是点属性
    
    # 预分配数组（提高效率）
    xyzs = np.empty((num_points, 3))   # 3D坐标 (X, Y, Z)
    rgbs = np.empty((num_points, 3))   # RGB颜色 (R, G, B)
    errors = np.empty((num_points, 1)) # 重投影误差（这里初始化为0）
    
    # 第二步：遍历所有重建场景，提取点数据
    count = 0  # 当前处理的点数索引
    for reconstruction in reconstructions:
        # 遍历当前重建场景中的所有点
        for point_id in reconstruction["points"]:  # point_id是点的唯一标识符
            # 获取点的颜色（通常是从图像中提取的平均颜色）
            color = reconstruction["points"][point_id]["color"]
            # 获取点的3D坐标
            coordinates = reconstruction["points"][point_id]["coordinates"]
            
            # 将坐标转换为numpy数组
            xyz = np.array([coordinates[0], coordinates[1], coordinates[2]])
            # 将颜色转换为numpy数组
            rgb = np.array([color[0], color[1], color[2]])
            # 错误值暂时设为0（OpenSfM原始数据中可能没有误差信息）
            error = np.array(0)
            
            # 将数据存储到预分配的数组中
            xyzs[count] = xyz
            rgbs[count] = rgb
            errors[count] = error
            count += 1
    
    return xyzs, rgbs, errors


def read_opensfm(reconstructions, focal_angle):
    """
    从OpenSfM重建结果中读取相机参数（内参和外参）
    
    参数:
        reconstructions: OpenSfM格式的重建数据列表
        focal_angle: 焦距角度（用于球面/等距柱状投影相机）
    
    返回:
        cameras: 相机内参字典 {camera_id: Camera对象}
        images: 图像位姿字典 {image_id: Image对象}
    """
    
    images = {}      # 存储图像位姿信息
    i = 0            # 图像ID计数器
    cameras = {}     # 存储相机内参
    camera_names = {} # 相机名称到ID的映射
    
    # 遍历所有重建场景（OpenSfM可能输出多个重建结果）
    for reconstruction in reconstructions:
        
        # 第一步：读取相机内参（重建中定义的相机模型）
        for i, camera in enumerate(reconstruction["cameras"]):
            camera_name = camera  # 相机名称（如"camera0"）
            camera_info = reconstruction["cameras"][camera]
            
            # 只支持球面/等距柱状投影相机（360度全景）
            if camera_info['projection_type'] in ['spherical', 'equirectangular']:
                camera_id = 11  # 固定ID（可能是硬编码的，或者应该自动生成）
                model = "SPHERICAL"  # 相机模型类型
                
                # 获取图像尺寸
                width = reconstruction["cameras"][camera]["width"]
                height = reconstruction["cameras"][camera]["height"]
                
                # 计算等距柱状投影相机的内参
                # 对于球面投影，需要特殊处理焦距
                # focal_x = width / (2 * π * cos(θ))，其中θ是焦距角度
                focal_x = width / (2 * math.pi * math.cos(focal_angle / 180 * math.pi))
                focal_y = focal_x  # 通常fx = fy
                center_x = width / 2  # 主点在图像中心
                center_y = height / 2
                
                # 相机参数数组：[fx, fy, cx, cy]
                params = np.array([focal_x, focal_y, center_x, center_y])
                
                # 创建Camera对象（来自Colmap模块）
                cameras[camera_id] = Camera(
                    id=camera_id, 
                    model=model, 
                    width=width, 
                    height=height, 
                    params=params
                )
                
                # 建立相机名称到ID的映射
                camera_names[camera_name] = camera_id
                
            else:
                # 只支持球面/等距柱状投影，其他模型（如针孔、鱼眼）未实现
                raise NotImplementedError(
                    f"{camera_info['projection_type']} camera model from OpenSfM data format is not implemented"
                )

        # 第二步：读取相机外参（每个图像的位姿）
        for shot in reconstruction["shots"]:
            # 获取平移向量
            translation = reconstruction["shots"][shot]["translation"]
            # 获取旋转（角轴表示）
            rotation = reconstruction["shots"][shot]["rotation"]
            
            # 将角轴转换为四元数
            qvec = angle_axis_to_quaternion(rotation)
            # 平移向量转为numpy数组
            tvec = np.array([translation[0], translation[1], translation[2]])
            
            # 查找相机ID
            camera_name = reconstruction["shots"][shot]["camera"]
            camera_id = camera_names.get(camera_name, 0)  # 默认0
            
            # 图像信息
            image_id = i
            image_name = shot  # 图像文件名
            
            # 虚拟数据（实际从OpenSfM读取可能需要这些）
            xys = np.array([0, 0])  # 特征点坐标（虚拟）
            point3D_ids = np.array([0, 0])  # 对应的3D点ID（虚拟）
            
            # 创建Image对象
            images[image_id] = Image(
                id=image_id, 
                qvec=qvec,      # 旋转（四元数）
                tvec=tvec,      # 平移
                camera_id=camera_id,  # 对应的相机ID
                name=image_name,      # 图像文件名
                xys=xys,              # 特征点坐标（这里未使用）
                point3D_ids=point3D_ids  # 3D点ID（这里未使用）
            )
            i += 1  # 递增图像ID
            
    return cameras, images


@Framework.Configurable.configure(
    PATH='/openbayes/home/nerficg/dataset/lab',      # 数据集路径
    BACKGROUND_COLOR=[0.0, 0.0, 0.0],                 # 背景颜色（黑色）
    NEAR_PLANE=0.2,                                   # 近裁剪平面距离
    FAR_PLANE=1000.0,                                 # 远裁剪平面距离
    TEST_STEP=8,                                      # 测试集采样步长
    FOCAL_ANGLE=0.0,                                  # 焦距角度
)


class CustomDataset(BaseDataset):
    """Dataset class for panorama scenes from the Roaming and Rounding dataset."""
    
    def __init__(self, path: str) -> None:
        super().__init__(
            path,  # 数据集路径
            EquirectangularCamera(0.2, 1000.0),  # 相机模型：等距柱状投影
            CameraCoordinateSystemsTransformations.LEFT_HAND,  # 左手坐标系
            WorldCoordinateSystemTransformations.XnZY,  # 世界坐标系变换
        )

    def load(self) -> dict[str, list[CameraProperties] | None]:
        """
            加载数据集，返回包含训练、评估和测试集相机属性的字典

            主要流程：
            1. 设置相机近远平面
            2. 加载OpenSfM重建数据
            3. 读取相机内外参
            4. 加载图像数据
            5. 创建相机属性
            6. 加载/生成点云
            7. 位姿PCA对齐
            8. 划分训练测试集

            返回：
                dict: {
                    'train': [CameraProperties, ...],  # 训练集
                    'test': [CameraProperties, ...],   # 测试集（如果划分）
                    'eval': [CameraProperties, ...]    # 评估集（可能为空）
                }
        """
        # set near and far plane
        self.camera.near_plane = self.NEAR_PLANE    # 近裁剪面：0.2米
        self.camera.far_plane = self.FAR_PLANE      # 远裁剪面：1000米
        # load dataset 初始化数据集字典
        """
            self.subsets 通常为 ['train', 'eval', 'test']
            创建空列表准备存储不同子集的相机属性
        """
        dataset: dict[str, list[CameraProperties]] = {subset: [] for subset in self.subsets}
        # load data  加载opensfm重建数据
        reconstruction_file = self.dataset_path / 'reconstruction.json'
        with open(reconstruction_file) as f:
            reconstruction = json.load(f)
        """
            相机内参字典 {camera_id: Camera对象}
            相机外参字典 {image_id: Image对象}
        """
        cam_intrinsics, cam_extrinsics = read_opensfm(reconstruction, self.FOCAL_ANGLE) 
        all_views: list[CameraProperties] = []
        for cam_idx, cam_data in enumerate(cam_intrinsics.values()):
            # load images
            images = [data for data in cam_extrinsics.values() if data.camera_id == cam_data.id]
            images = natsorted(images, key=lambda data: data.name)
            image_directory_name = 'images'
            image_scale_factor = self.IMAGE_SCALE_FACTOR
            match self.IMAGE_SCALE_FACTOR:
                case 0.5:
                    image_directory_name = 'images_2'
                    image_scale_factor = None
                case _:
                    pass
            image_filenames = [str(self.dataset_path / image_directory_name / image.name) for image in images]
            rgbs, alphas = loadImagesParallel(image_filenames, image_scale_factor, num_threads=-1, desc=f'camera {cam_data.id}')
            for idx, (image, rgb, alpha) in enumerate(zip(images, rgbs, alphas)):
                # extract c2w matrix
                rotation_matrix = torch.from_numpy(quaternion_to_R(image.qvec)).float()
                translation_vector = torch.from_numpy(image.tvec).float()
                w2c = torch.eye(4, device=torch.device('cpu'))
                w2c[:3, :3] = rotation_matrix
                w2c[:3, 3] = translation_vector
                c2w = torch.linalg.inv(w2c)
                # intrinsics
                focal_x = cam_data.params[0]
                focal_y = cam_data.params[1]
                principal_offset_x = cam_data.params[2] - cam_data.width / 2
                principal_offset_y = cam_data.params[3] - cam_data.height / 2
                if self.IMAGE_SCALE_FACTOR is not None:
                    scale_factor_intrinsics_x = rgb.shape[2] / cam_data.width
                    scale_factor_intrinsics_y = rgb.shape[1] / cam_data.height
                    focal_x *= scale_factor_intrinsics_x
                    focal_y *= scale_factor_intrinsics_y
                    principal_offset_x *= scale_factor_intrinsics_x
                    principal_offset_y *= scale_factor_intrinsics_y

                # create camera properties and subsets
                all_views.append(CameraProperties(
                    width=rgb.shape[2],
                    height=rgb.shape[1],
                    rgb=rgb,
                    alpha=alpha,
                    c2w=c2w,
                    focal_x=focal_x,
                    focal_y=focal_y,
                    principal_offset_x=principal_offset_x,
                    principal_offset_y=principal_offset_y,
                ))

        # load point cloud
        ply_path = self.dataset_path / 'reconstruction.ply'
        if not os.path.exists(ply_path):
            Logger.logInfo('Found new scene. Converting sparse SfM points to .ply format.')
            xyz, rgb, _ = read_points3D_opensfm(reconstruction)
            storePly(ply_path, xyz, rgb)
        try:
            self.point_cloud = fetchPly(ply_path)
        except Exception:
            raise Framework.DatasetError(f'Failed to load SfM point cloud')

        # rotate/scale poses to try aligning ground with xy plane
        c2ws = torch.stack([camera.c2w for camera in all_views])
        c2ws, transformation = transformPosesPCA(c2ws, rescale=False)
        for camera_properties, c2w in zip(all_views, c2ws):
            camera_properties.c2w = c2w
        self.point_cloud.transform(transformation)
        self.world_coordinate_system = None

        # perform test split
        if self.TEST_STEP > 0:
            for i in range(len(all_views)):
                if i % self.TEST_STEP == 0:
                    dataset['test'].append(all_views[i])
                else:
                    dataset['train'].append(all_views[i])
        else:
            dataset['train'] = all_views

        # return the dataset
        return dataset

    @classmethod
    def getDefaultParameters(cls):
        """Return default parameters for this dataset."""
        return {
            'PATH': '/openbayes/home/nerficg/dataset/lab',
            'BACKGROUND_COLOR': [0.0, 0.0, 0.0],
            'NEAR_PLANE': 0.2,
            'FAR_PLANE': 1000.0,
            'TEST_STEP': 8,
            'FOCAL_ANGLE': 0.0,
            'IMAGE_SCALE_FACTOR': 1.0,
        }
