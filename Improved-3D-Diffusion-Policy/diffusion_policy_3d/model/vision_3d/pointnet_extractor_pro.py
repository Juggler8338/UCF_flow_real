import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
import diffusion_policy_3d.model.vision_3d.point_process as point_process


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class StateEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU):
        super().__init__()
        self.state_key = 'full_state'
        self.state_shape = observation_space[self.state_key]
        cprint(f"[StateEncoder] state shape: {self.state_shape}", "yellow")
        
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[StateEncoder] output dim: {output_dim}", "red")
        self.output_dim = output_dim
        
    def output_shape(self):
        return self.output_dim
        
    def forward(self, observations: Dict) -> torch.Tensor:
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)
        return state_feat
    

class OCFEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='dp3_encoder',
                 point_downsample=True,
                 ):
        super().__init__()
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        self.num_points = pointcloud_encoder_cfg.num_points # 4096

        cprint(f"[OCFEncoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[OCFEncoder] state shape: {self.state_shape}", "yellow")
        
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        
        self.downsample = point_downsample
        if self.downsample:
            self.point_preprocess = point_process.uniform_sampling_torch
        else:
            self.point_preprocess = nn.Identity()
        
        if pointnet_type == "multi_stage_pointnet":
            from .multi_stage_pointnet import MultiStagePointNetEncoder
            self.extractor = MultiStagePointNetEncoder(out_channels=pointcloud_encoder_cfg.out_channels)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
            
        # ====================== 修改点 1：分离维度变量 ======================
        self.pc_dim = pointcloud_encoder_cfg.out_channels
        self.state_dim = state_mlp_size[-1]

        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], self.state_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[OCFEncoder] pc_dim: {self.pc_dim}, state_dim: {self.state_dim}", "red")

    def forward(self, observations: Dict) -> Dict[str, torch.Tensor]:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")

        if self.downsample:
            points = self.point_preprocess(points, self.num_points)
           
        pc_feat = self.extractor(points)    # B * pc_dim
         
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * state_dim
        
        # ====================== 修改点 2：返回分离特征的字典 ======================
        return {
            'pc_feat': pc_feat,
            'state_feat': state_feat
        }

    def output_shape(self) -> Dict[str, int]:
        # ====================== 修改点 3：返回分离维度的字典 ======================
        return {
            'pc_dim': self.pc_dim, 
            'state_dim': self.state_dim
        }