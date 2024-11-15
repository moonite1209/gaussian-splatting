#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.active_sh_degree = 0
        self.optimizer_type = args.optimizer_type
        self.max_sh_degree = args.sh_degree
        self.sh_num = (args.sh_degree+1)**2

        self.xyz = nn.Parameter(torch.empty((0,3)), False)
        self.shs = nn.Parameter(torch.empty((0,3, self.sh_num)), False)
        self.scaling = nn.Parameter(torch.empty((0,3)), False)
        self.rotation = nn.Parameter(torch.empty((0,3)), False)
        self.opacity = nn.Parameter(torch.empty((0,1)), False)

        self.max_radii2D = torch.empty((0,1), requires_grad=False)
        self.xyz_gradient_accum = torch.empty((0,1), requires_grad=False)
        self.denom = torch.empty((0,1), requires_grad=False)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation) # [n, 3, 3]
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance) # [n, 6]
            return symm
        # use activation restrict number range
        self.scaling_activation = torch.exp # in (0, inf)
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid # in (0, 1)
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize # 单位四元数表示旋转

    def load_point_cloud(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(pcd.points) # [n, 3]
        fused_color = RGB2SH(torch.tensor(pcd.colors)) # [n, 3]
        shs = torch.zeros((fused_color.shape[0], 3, self.sh_num))
        shs[:, :3, 0 ] = fused_color
        shs[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.tensor(pcd.points, dtype=torch.float, device='cuda')), 0.0000001).cpu()
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4))
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float))

        self.xyz = nn.Parameter(fused_point_cloud, False)
        self.shs = nn.Parameter(shs, False)
        self.scaling = nn.Parameter(scales, False)
        self.rotation = nn.Parameter(rots, False)
        self.opacity = nn.Parameter(opacities, False)

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]))

    def forward():
        ...