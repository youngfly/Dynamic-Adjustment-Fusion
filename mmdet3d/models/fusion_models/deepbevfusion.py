from typing import Any, Dict
import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
import math
from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS
from mmdet3d.core import LiDARInstance3DBoxes
from .base import Base3DFusionModel
import os
import matplotlib.pyplot as plt
from torch.nn import functional as F
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
import random
import cv2
from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention
from mmcv.cnn import ConvModule, build_conv_layer

# todo context local attention
from torch.autograd import Function
from localAttention import (similar_forward,
                            similar_backward,
                            weighting_forward,
                            weighting_backward_ori,
                            weighting_backward_weight)

# todo self attention
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
import warnings
import torch
# import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
from mmdet.core import (
    AssignResult,
    build_assigner,
    build_bbox_coder,
    build_sampler,
    multi_apply,
)
from mmdet3d.core import (
    PseudoSampler,
    circle_nms,
    draw_heatmap_gaussian,
    gaussian_radius,
    xywhr2xyxyr,
)
from mmdet3d.models.builder import build_loss
__all__ = ["DeepBEVFusion"]


# todo context local attention ****************************
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, bias='auto',
                 inplace=True, affine=True):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.use_norm = norm_layer is not None
        self.use_activation = activation_layer is not None
        if bias == 'auto':
            bias = not self.use_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.use_norm:
            self.bn = norm_layer(out_channels, affine=affine)
        if self.use_activation:
            self.activation = activation_layer(inplace=inplace)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x


class similarFunction(Function):
    @staticmethod
    def forward(ctx, x_ori, x_loc, kH, kW):
        ctx.save_for_backward(x_ori, x_loc)
        ctx.kHW = (kH, kW)
        output = similar_forward(x_ori, x_loc, kH, kW)

        return output

    @staticmethod
    #@once_differentiable
    def backward(ctx, grad_outputs):
        x_ori, x_loc = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = similar_backward(x_loc, grad_outputs, kH, kW, True)
        grad_loc = similar_backward(x_ori, grad_outputs, kH, kW, False)

        return grad_ori, grad_loc, None, None


class weightingFunction(Function):
    @staticmethod
    def forward(ctx, x_ori, x_weight, kH, kW):
        ctx.save_for_backward(x_ori, x_weight)
        ctx.kHW = (kH, kW)
        output = weighting_forward(x_ori, x_weight, kH, kW)

        return output

    @staticmethod
    #@once_differentiable
    def backward(ctx, grad_outputs):
        x_ori, x_weight = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = weighting_backward_ori(x_weight, grad_outputs, kH, kW)
        grad_weight = weighting_backward_weight(x_ori, grad_outputs, kH, kW)

        return grad_ori, grad_weight, None, None


class LocalContextAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, last_affine=True):
        super().__init__()

        self.f_similar = similarFunction.apply
        self.f_weighting = weightingFunction.apply

        self.kernel_size = kernel_size
        self.query_project = nn.Sequential(ConvBNReLU(in_channels,
                                                                  out_channels,
                                                                  kernel_size=1,
                                                                  norm_layer=nn.BatchNorm2d,
                                                                  activation_layer=nn.ReLU),
                                           ConvBNReLU(out_channels,
                                                                  out_channels,
                                                                  kernel_size=1,
                                                                  norm_layer=nn.BatchNorm2d,
                                                                  activation_layer=nn.ReLU))
        self.key_project = nn.Sequential(ConvBNReLU(in_channels,
                                                                out_channels,
                                                                kernel_size=1,
                                                                norm_layer=nn.BatchNorm2d,
                                                                activation_layer=nn.ReLU),
                                         ConvBNReLU(out_channels,
                                                                out_channels,
                                                                kernel_size=1,
                                                                norm_layer=nn.BatchNorm2d,
                                                                activation_layer=nn.ReLU))
        self.value_project = ConvBNReLU(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    norm_layer=nn.BatchNorm2d,
                                                    activation_layer=nn.ReLU,
                                                    affine=last_affine)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, target_feats, source_feats, **kwargs):
        query = self.query_project(target_feats).to(torch.float32)
        key = self.key_project(source_feats).to(torch.float32)
        value = self.value_project(source_feats).to(torch.float32)

        weight = self.f_similar(query, key, self.kernel_size, self.kernel_size)
        weight = nn.functional.softmax(weight / math.sqrt(key.size(1)), -1)
        out = self.f_weighting(value, weight, self.kernel_size, self.kernel_size)
        return out.to(torch.float16)


class LocalAttention(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, last_affine=True):
        super(LocalAttention, self).__init__()
        self.f_similar = similarFunction.apply
        self.f_weighting = weightingFunction.apply
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, target_feats, source_feats):
        query = self.conv1(target_feats).to(torch.float32)
        key = self.conv2(source_feats).to(torch.float32)
        value = self.conv3(source_feats).to(torch.float32)

        weight = self.f_similar(query, key, self.kernel_size, self.kernel_size)
        weight = nn.functional.softmax(weight / math.sqrt(key.size(1)), -1)
        out = self.f_weighting(value, weight, self.kernel_size, self.kernel_size)

        return out


class MMFEncoderLayer(nn.Module):
    def __init__(self, hidden_channel, context_kernel):
        super(MMFEncoderLayer, self).__init__()
        # todo get the mask
        self.COM_L2C_block = LocalAttention(hidden_channel, hidden_channel, context_kernel)
        self.COM_C2L_block = LocalAttention(hidden_channel, hidden_channel, context_kernel)
        self.UNQ_L2C_block = LocalAttention(hidden_channel, hidden_channel, context_kernel)
        self.UNQ_C2L_block = LocalAttention(hidden_channel, hidden_channel, context_kernel)

        # todo get the common_feat
        self.COM_L2C = LocalAttention(hidden_channel, hidden_channel, context_kernel)
        self.COM_C2L = LocalAttention(hidden_channel, hidden_channel, context_kernel)
        self.UNQ_L2C = LocalAttention(hidden_channel, hidden_channel, context_kernel)
        self.UNQ_C2L = LocalAttention(hidden_channel, hidden_channel, context_kernel)

    def forward(self, f_cam, f_lid):
        # todo get the normalization
        f_cam = F.sigmoid(f_cam)
        f_lid = F.sigmoid(f_lid)
        f_cam_res = 1 - f_cam
        f_lid_res = 1 - f_lid

        # todo get the common and unque
        MCL = self.COM_C2L_block(f_cam, f_lid)
        MLC = self.COM_L2C_block(f_lid, f_cam)
        MC =  self.UNQ_C2L_block(f_cam, f_lid_res)
        ML =  self.UNQ_L2C_block(f_lid, f_cam_res)

        # todo the multi-modal-fusion
        F_C_COM = self.COM_C2L(MCL, f_cam)
        F_L_COM = self.COM_L2C(MLC, f_lid)
        F_C_UNQ = self.UNQ_C2L(MC, f_cam)
        F_L_UNQ = self.UNQ_L2C(ML, f_lid)

        return F_C_COM, F_L_COM, F_C_UNQ, F_L_UNQ


class MMF(nn.Module):
    def __init__(self):
        super().__init__()

        # todo define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.m_channel = 512
        self.context_kernel = 9
        self.num_layers = 1
        self.bn_momentum = 0.1

        # todo translate the channel
        self.cam_conv = ConvBNReLU(self.cam_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)
        self.lid_conv = ConvBNReLU(self.hidden_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)
        self.fusion_blocks = nn.ModuleList()
        # todo our multi-modality-fusion
        for i in range(self.num_layers):
            self.fusion_blocks.append(MMFEncoderLayer(self.hidden_channel, self.context_kernel))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, feature_cam, feature_lid):
        # todo channel chanslate
        f_com_cam = self.cam_conv(feature_cam)
        f_com_lid = self.lid_conv(feature_lid)

        # todo multi-level-encode
        for i in range(self.num_layers):
            f_com_cam, f_com_lid, f_unq_cam, f_unq_lid = \
                self.fusion_blocks[i](f_com_cam, f_com_lid)
        return f_com_cam, f_com_lid, f_unq_cam,  f_unq_lid


# todo global attention ***********************************
class GlobalGraph(nn.Module):
    def __init__(self, hidden_size, attention_head_size=None, num_attention_heads=1):
        super(GlobalGraph, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.num_qkv = 1

        self.query = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.key = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.value = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)

    def get_extended_attention_mask(self, attention_mask):
        """
        1 in attention_mask stands for doing attention, 0 for not doing attention.
        After this function, 1 turns to 0, 0 turns to -10000.0
        Because the -10000.0 will be fed into softmax and -10000.0 can be thought as 0 in softmax.
        """
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads,
                              self.attention_head_size)
        # (batch, max_vector_num, head, head_size)
        x = x.view(*sz)
        # (batch, head, max_vector_num, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, return_scores=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = nn.functional.linear(hidden_states, self.key.weight)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        if attention_mask is not None:
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_scores:
            attention_probs = torch.squeeze(attention_probs, dim=1)
            return context_layer, attention_probs
        return context_layer


class CrossAttention(GlobalGraph):
    def __init__(self, hidden_size, attention_head_size=None, num_attention_heads=1, key_hidden_size=256,
                 query_hidden_size=256, last_affine=True):
        super(CrossAttention, self).__init__(hidden_size, attention_head_size, num_attention_heads)
        self.num_attention_heads = num_attention_heads  # todo 1
        self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.num_qkv = 1

        # todo construct the qkv weight
        # self.query = nn.Linear(query_hidden_size, self.all_head_size * self.num_qkv)
        # self.key = nn.Linear(key_hidden_size, self.all_head_size * self.num_qkv)
        # self.value = nn.Linear(key_hidden_size, self.all_head_size * self.num_qkv)

        self.query = ConvBNReLU(hidden_size,
                                hidden_size,
                                kernel_size=1,
                                norm_layer=nn.BatchNorm2d,
                                activation_layer=nn.ReLU)
        self.key = ConvBNReLU(hidden_size,
                              hidden_size,
                              kernel_size=1,
                              norm_layer=nn.BatchNorm2d,
                              activation_layer=nn.ReLU)

        self.value = ConvBNReLU(hidden_size,
                                hidden_size,
                                kernel_size=1,
                                norm_layer=nn.BatchNorm2d,
                                activation_layer=nn.ReLU,
                                affine=last_affine)

    def forward(self, hidden_states_query, hidden_states_key=None, attention_mask=None, return_scores=False):
        query_layer = self.query(hidden_states_query)
        key_layer = self.key(hidden_states_key)
        value_layer = self.value(hidden_states_key)

        # query_layer = self.transpose_for_scores(mixed_query_layer)
        # key_layer = self.transpose_for_scores(mixed_key_layer)
        # value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        # if attention_mask is not None:
        #     assert hidden_states_query.shape[1] == attention_mask.shape[1] \
        #            and hidden_states_key.shape[1] == attention_mask.shape[2]
        #     attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)
        # todo softmax
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # todo get the attention
        context_layer = torch.matmul(attention_probs, value_layer)

        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[
        #                           :-2] + (self.all_head_size,)
        # context_layer = context_layer.view(*new_context_layer_shape)
        # if return_scores:
        #     return context_layer, torch.squeeze(attention_probs, dim=1)
        return context_layer


class MMFEncoderLayer2(nn.Module):
    def __init__(self, hidden_channel, context_kernel):
        super(MMFEncoderLayer2, self).__init__()
        # todo get the mask
        self.COM_L2C_block = CrossAttention(hidden_channel)
        self.COM_C2L_block = CrossAttention(hidden_channel)
        self.UNQ_L2C_block = CrossAttention(hidden_channel)
        self.UNQ_C2L_block = CrossAttention(hidden_channel)

        # todo get the common_feat
        self.COM_L2C = CrossAttention(hidden_channel)
        self.COM_C2L = CrossAttention(hidden_channel)
        self.UNQ_L2C = CrossAttention(hidden_channel)
        self.UNQ_C2L = CrossAttention(hidden_channel)

    def forward(self, f_cam, f_lid):
        # todo get the normalization
        f_cam = F.sigmoid(f_cam)
        f_lid = F.sigmoid(f_lid)
        f_cam_res = 1 - f_cam
        f_lid_res = 1 - f_lid

        # todo get the common and unque
        MCL = self.COM_C2L_block(f_cam, f_lid)
        MLC = self.COM_L2C_block(f_lid, f_cam)
        MC =  self.UNQ_C2L_block(f_cam, f_lid_res)
        ML =  self.UNQ_L2C_block(f_lid, f_cam_res)

        # todo the multi-modal-fusion
        F_C_COM = self.COM_C2L(MCL, f_cam)
        F_L_COM = self.COM_L2C(MLC, f_lid)
        F_C_UNQ = self.UNQ_C2L(MC, f_cam)
        F_L_UNQ = self.UNQ_L2C(ML, f_lid)

        return F_C_COM, F_L_COM, F_C_UNQ, F_L_UNQ


class MMF2(nn.Module):
    def __init__(self):
        super().__init__()

        # todo define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.m_channel = 512
        self.context_kernel = 9
        self.num_layers = 1
        self.bn_momentum = 0.1

        # todo translate the channel
        self.cam_conv = ConvBNReLU(self.cam_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)
        self.lid_conv = ConvBNReLU(self.hidden_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)
        self.fusion_blocks = nn.ModuleList()
        # todo our multi-modality-fusion
        for i in range(self.num_layers):
            self.fusion_blocks.append(MMFEncoderLayer2(self.hidden_channel, self.context_kernel))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, feature_cam, feature_lid):
        # todo channel chanslate
        f_com_cam = self.cam_conv(feature_cam)
        f_com_lid = self.lid_conv(feature_lid)

        # todo multi-level-encode
        for i in range(self.num_layers):
            f_com_cam, f_com_lid, f_unq_cam, f_unq_lid = \
                self.fusion_blocks[i](f_com_cam, f_com_lid)
        return f_com_cam, f_com_lid, f_unq_cam,  f_unq_lid


# todo deformable attention *******************************
class TemporalSelfAttention(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        num_bev_queue (int): In this version, we only use one history BEV and one currenct BEV.
         the length of BEV queue is 2.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=1,
                 num_points=4,
                 num_bev_queue=1,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 operation=None):

        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False
        self.operation = operation
        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        # todo generatet the offsets
        self.sampling_offsets = nn.Linear(
            embed_dims*self.num_bev_queue, num_bev_queue*num_heads * num_levels * num_points * 2)
        # todo attention
        self.attention_weights = nn.Linear(embed_dims*self.num_bev_queue,
                                           num_bev_queue*num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',

                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        # todo (2, 256, 180, 180)-> (2, 256, 180*180).
        query = query.reshape(query.shape[0], query.shape[1], -1)
        # todo (2, 256, 180*180)-> (2, 180*180, 256).
        query = query.permute(0, 2, 1)
        # todo (2, 256, 180, 180)-> (2, 256, 180*180).
        value = value.reshape(value.shape[0], value.shape[1], -1)
        # todo (2, 256, 180*180)-> (2, 180*180, 256).
        value = value.permute(0, 2, 1)

        # todo copy the it self (2, 180*180, 256).
        if identity is None:
            identity = query

        # todo add the position
        if query_pos is not None:
            query = query + query_pos

        # todo change to (bs, num_query ,embed_dims)
        bs,  num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        # todo keep the nums==h*w
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 1

        # todo concat the query and value with embedding dimension
        # query = torch.cat([value[:bs], query], -1)
        # todo project
        value = self.value_proj(value)

        # todo not use
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        # todo reshape the value (2*1, 180*180, 8, 64)
        value = value.reshape(bs*self.num_bev_queue,
                              num_value, self.num_heads, -1)
        # todo sample-offset
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads,  self.num_bev_queue, self.num_levels, self.num_points, 2)
        # todo attention weight
        attention_weights = self.attention_weights(query).view(
            bs, num_query,  self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()

        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        # todo [2, 32400, 1, 2]
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        # todo calculate the attention
        if torch.cuda.is_available() and value.is_cuda:
            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            # todo get the prediction output shape (bs*num_bev_queue, num_query, embed_dims)
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        # todo (2, 180*180, 256)  the resdul is necessry
        output = self.output_proj(output)
        results = self.dropout(output) + identity

        # todo (2, 256, 180, 180) final
        results = results.permute(0, 2, 1)
        results = results.reshape(bs, embed_dims, spatial_shapes[0][0], spatial_shapes[0][1])
        return results


class MMFEncoderLayer3(nn.Module):
    def __init__(self):
        super(MMFEncoderLayer3, self).__init__()
        # todo  encoding the position
        positional_encoding = dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)

        # todo get the mask
        self.COM_L2C_block = TemporalSelfAttention()
        self.COM_C2L_block = TemporalSelfAttention()
        self.UNQ_L2C_block = TemporalSelfAttention()
        self.UNQ_C2L_block = TemporalSelfAttention()

        # todo get the common_feat
        self.COM_L2C = TemporalSelfAttention()
        self.COM_C2L = TemporalSelfAttention()
        self.UNQ_L2C = TemporalSelfAttention()
        self.UNQ_C2L = TemporalSelfAttention()

    def get_reference_points(self, H, W, Z=8,
                             num_points_in_pillar=4, dim='3d', bs=1,
                             device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def forward(self, f_cam, f_lid):
        # todo [2, 256, 180, 180]->[2, 256, 180*180]
        bs, embeding_c, H, W = f_cam.shape
        f_cam = f_cam.reshape(bs, embeding_c, -1)
        f_lid = f_lid.reshape(bs, embeding_c, -1)
        # todo get the normalization we use the softmax at the h*w channel
        f_cam = F.softmax(f_cam, dim=2)
        f_lid = F.softmax(f_lid, dim=2)
        # todo [2, 256, 180*180] -> [2, 256, 180, 180]
        f_cam = f_cam.reshape(bs, embeding_c, H, W)
        f_lid = f_lid.reshape(bs, embeding_c, H, W)
        f_cam_res = 1 - f_cam
        f_lid_res = 1 - f_lid

        # todo get the bev_pos
        ref_2d = self.get_reference_points(
            H, W, dim='2d', bs=f_cam.size(0), device=f_cam.device, dtype=f_cam.dtype)  # todo [2, 32400, 1, 2]
        dtype = f_cam.dtype
        bev_mask = torch.zeros((len(f_cam), H, W), device=f_cam.device).to(dtype) # todo [2, 180, 180]
        bev_pos = self.positional_encoding(bev_mask).to(dtype) # todo [2, 256, 180, 180]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # todo [32400, 2, 256]
        bev_pos = bev_pos.permute(1, 0, 2) # todo [2, 32400, 256]

        # todo deformable attention
        # todo get the common and unque mask
        MCL = self.COM_C2L_block(f_cam, f_lid, f_lid, None, query_pos=bev_pos, key_pos=bev_pos, reference_points=ref_2d,
                                 spatial_shapes=torch.tensor([[H, W]], device=f_cam.device),
                                 level_start_index=torch.tensor([0], device=f_cam.device))
        MLC = self.COM_L2C_block(f_lid, f_cam, f_cam, None, query_pos=bev_pos, key_pos=bev_pos, reference_points=ref_2d,
                                 spatial_shapes=torch.tensor([[H, W]], device=f_cam.device),
                                 level_start_index=torch.tensor([0], device=f_cam.device))
        MC =  self.UNQ_C2L_block(f_cam, f_lid_res, f_lid_res, None, query_pos=bev_pos, key_pos=bev_pos, reference_points=ref_2d,
                                 spatial_shapes=torch.tensor([[H, W]], device=f_cam.device),
                                 level_start_index=torch.tensor([0], device=f_cam.device))
        ML =  self.UNQ_L2C_block(f_lid, f_cam_res, f_cam_res, None, query_pos=bev_pos, key_pos=bev_pos, reference_points=ref_2d,
                                 spatial_shapes=torch.tensor([[H, W]], device=f_cam.device),
                                 level_start_index=torch.tensor([0], device=f_cam.device))

        # todo get the comom and unique feature
        F_C_COM = self.COM_C2L(MCL * f_cam, f_cam, f_cam, None, query_pos=bev_pos, key_pos=bev_pos, reference_points=ref_2d,
                                 spatial_shapes=torch.tensor([[H, W]], device=f_cam.device),
                                 level_start_index=torch.tensor([0], device=f_cam.device))
        F_L_COM = self.COM_L2C(MLC * f_lid, f_lid, f_lid, None, query_pos=bev_pos, key_pos=bev_pos, reference_points=ref_2d,
                                 spatial_shapes=torch.tensor([[H, W]], device=f_cam.device),
                                 level_start_index=torch.tensor([0], device=f_cam.device))
        F_C_UNQ = self.UNQ_C2L(MC * f_cam, f_cam, f_cam, None, query_pos=bev_pos, key_pos=bev_pos, reference_points=ref_2d,
                                 spatial_shapes=torch.tensor([[H, W]], device=f_cam.device),
                                 level_start_index=torch.tensor([0], device=f_cam.device))
        F_L_UNQ = self.UNQ_L2C(ML * f_lid, f_lid, f_lid, None, query_pos=bev_pos, key_pos=bev_pos, reference_points=ref_2d,
                                 spatial_shapes=torch.tensor([[H, W]], device=f_cam.device),
                                 level_start_index=torch.tensor([0], device=f_cam.device))

        return F_C_COM, F_L_COM, F_C_UNQ, F_L_UNQ, MCL, MLC, MC, ML


class MMF3(nn.Module):
    def __init__(self):
        super().__init__()

        # todo define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.m_channel = 512
        self.num_layers = 1
        self.bn_momentum = 0.1

        # todo translate the channel
        self.cam_conv = ConvBNReLU(self.cam_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)
        self.lid_conv = ConvBNReLU(self.hidden_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)
        self.fusion_blocks = nn.ModuleList()
        # todo our multi-modality-fusion
        for i in range(self.num_layers):
            self.fusion_blocks.append(MMFEncoderLayer3())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, feature_cam, feature_lid):
        # todo channel chanslate
        f_com_cam = self.cam_conv(feature_cam)
        f_com_lid = self.lid_conv(feature_lid)

        # todo multi-level-encode
        for i in range(self.num_layers):
            f_com_cam, f_com_lid, f_unq_cam, f_unq_lid, MCL, MLC, MC, ML = \
                self.fusion_blocks[i](f_com_cam, f_com_lid)
        return f_com_cam, f_com_lid, f_unq_cam, f_unq_lid, MCL, MLC, MC, ML


# todo basic fusion ***************************************
class Inter_local(nn.Module):
    def __init__(self):
        super().__init__()

        # todo channel define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.mix_channel = 1024
        self.out_channel = 1
        self.m_channel = 512

        # todo  encoding the position
        positional_encoding = dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)

        # todo translate the channel
        self.cam_conv = ConvBNReLU(self.cam_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)
        self.lid_conv = ConvBNReLU(self.hidden_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)

        # todo cross attention and self attention
        self.C2L_block = TemporalSelfAttention()
        self.L2C_block = TemporalSelfAttention()
        self.C2C_block = TemporalSelfAttention()
        self.L2L_block = TemporalSelfAttention()

        # todo
        self.mix_feature_cam = ConvBNReLU(self.m_channel, self.hidden_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)
        self.mix_feature_lid = ConvBNReLU(self.m_channel, self.hidden_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)

    def get_reference_points(self, H, W, Z=8,
                             num_points_in_pillar=4, dim='3d', bs=1,
                             device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def forward(self, feature_cam, feature_lid):
        # todo channel chanslate

        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)

        bs, embeding_c, H, W = f_cam_new.shape
        # todo get the bev_pos
        ref_2d = self.get_reference_points(
            H, W, dim='2d', bs=f_cam_new.size(0), device=f_cam_new.device, dtype=f_cam_new.dtype)  # todo [2, 32400, 1, 2]
        dtype = f_cam_new.dtype
        bev_mask = torch.zeros((len(f_cam_new), H, W), device=f_cam_new.device).to(dtype) # todo [2, 180, 180]
        bev_pos = self.positional_encoding(bev_mask).to(dtype) # todo [2, 256, 180, 180]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # todo [32400, 2, 256]
        bev_pos = bev_pos.permute(1, 0, 2) # todo [2, 32400, 256]

        # todo cross attention
        C2L = self.C2L_block(f_cam_new, f_lid_new, f_lid_new, None, query_pos=bev_pos, key_pos=bev_pos, reference_points=ref_2d,
                                 spatial_shapes=torch.tensor([[H, W]], device=f_cam_new.device),
                                 level_start_index=torch.tensor([0], device=f_cam_new.device))
        L2C = self.L2C_block(f_lid_new, f_cam_new, f_cam_new, None, query_pos=bev_pos, key_pos=bev_pos, reference_points=ref_2d,
                                 spatial_shapes=torch.tensor([[H, W]], device=f_cam_new.device),
                                 level_start_index=torch.tensor([0], device=f_cam_new.device))
        C2C = self.C2C_block(f_cam_new, f_cam_new, f_cam_new, None, query_pos=bev_pos, key_pos=bev_pos, reference_points=ref_2d,
                                 spatial_shapes=torch.tensor([[H, W]], device=f_cam_new.device),
                                 level_start_index=torch.tensor([0], device=f_cam_new.device))
        L2L = self.L2L_block(f_lid_new, f_lid_new, f_lid_new, None, query_pos=bev_pos, key_pos=bev_pos, reference_points=ref_2d,
                                 spatial_shapes=torch.tensor([[H, W]], device=f_cam_new.device),
                                 level_start_index=torch.tensor([0], device=f_cam_new.device))

        mix_fea_camera = self.mix_feature_cam((torch.cat((C2L, C2C), dim=1)))
        mix_fea_lidar = self.mix_feature_lid((torch.cat((L2C, L2L), dim=1)))


        return mix_fea_camera, mix_fea_lidar


# todo image attention and deformable attention without redul
class MMF4(nn.Module):
    def __init__(self):
        super().__init__()

        # todo define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.m_channel = 512
        self.num_layers = 1
        self.bn_momentum = 0.1

        # todo  encoding the position
        positional_encoding = dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)

        # todo translate the channel
        self.cam_conv = ConvBNReLU(self.cam_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)
        self.lid_conv = ConvBNReLU(self.hidden_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)

        # todo cross attention and self attention
        self.C2L_block = TemporalSelfAttention()
        self.L2C_block = TemporalSelfAttention()
        self.C2C_block = TemporalSelfAttention()
        self.L2L_block = TemporalSelfAttention()

        # todo fuse single attention feature
        self.mix_feature_scl = ConvBNReLU(self.m_channel, self.hidden_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)
        self.mix_feature_ccl = ConvBNReLU(self.m_channel, self.hidden_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)

        # todo SE attention for single feature
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1))
        self.conv = ConvBNReLU(self.hidden_channel, self.hidden_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)
        # todo SE att
        self.global_avg_pool_ccl = torch.nn.AdaptiveAvgPool2d((1))
        self.conv_cll = ConvBNReLU(self.hidden_channel, self.hidden_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)

    def get_reference_points(self, H, W, Z=8,
                             num_points_in_pillar=4, dim='3d', bs=1,
                             device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, feature_cam, feature_lid):
        # todo channel chanslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)

        bs, embeding_c, H, W = f_cam_new.shape
        # todo get the bev_pos
        ref_2d = self.get_reference_points(
            H, W, dim='2d', bs=f_cam_new.size(0), device=f_cam_new.device, dtype=f_cam_new.dtype)  # todo [2, 32400, 1, 2]
        dtype = f_cam_new.dtype
        bev_mask = torch.zeros((len(f_cam_new), H, W), device=f_cam_new.device).to(dtype) # todo [2, 180, 180]
        bev_pos = self.positional_encoding(bev_mask).to(dtype) # todo [2, 256, 180, 180]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # todo [32400, 2, 256]
        bev_pos = bev_pos.permute(1, 0, 2) # todo [2, 32400, 256]

        # todo get SA and CA
        C2L = self.C2L_block(f_cam_new, f_lid_new, f_lid_new, None, query_pos=bev_pos, key_pos=bev_pos,
                             reference_points=ref_2d,
                             spatial_shapes=torch.tensor([[H, W]], device=f_cam_new.device),
                             level_start_index=torch.tensor([0], device=f_cam_new.device))
        L2C = self.L2C_block(f_lid_new, f_cam_new, f_cam_new, None, query_pos=bev_pos, key_pos=bev_pos,
                             reference_points=ref_2d,
                             spatial_shapes=torch.tensor([[H, W]], device=f_cam_new.device),
                             level_start_index=torch.tensor([0], device=f_cam_new.device))
        C2C = self.C2C_block(f_cam_new, f_cam_new, f_cam_new, None, query_pos=bev_pos, key_pos=bev_pos,
                             reference_points=ref_2d,
                             spatial_shapes=torch.tensor([[H, W]], device=f_cam_new.device),
                             level_start_index=torch.tensor([0], device=f_cam_new.device))
        L2L = self.L2L_block(f_lid_new, f_lid_new, f_lid_new, None, query_pos=bev_pos, key_pos=bev_pos,
                             reference_points=ref_2d,
                             spatial_shapes=torch.tensor([[H, W]], device=f_cam_new.device),
                             level_start_index=torch.tensor([0], device=f_cam_new.device))

        # todo get Mask
        mask = f_cam_new * f_lid_new

        # todo fuse the SCL and CCL
        SCL = self.mix_feature_scl((torch.cat((C2C, L2L), dim=1)))
        CCL = self.mix_feature_ccl((torch.cat((C2L, L2C), dim=1)))
        CCL = CCL* mask

        # todo senet
        SCL_at = self.global_avg_pool(SCL)
        SCL_at = self.conv(SCL_at)
        SCl_at = F.sigmoid(SCL_at)

        CCL_at = self.global_avg_pool_ccl(CCL)
        CCL_at = self.conv_cll (CCL_at)
        CCl_at = F.sigmoid(CCL_at)

        # todo final output
        SCL = SCL * SCl_at
        CCL = CCL * CCl_at
        return SCL, CCL


# todo basic module ***************************************
class Deformable_AT(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=1,
                 num_points=4,
                 num_bev_queue=1,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 operation=None):
        super(Deformable_AT, self).__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False
        self.operation = operation
        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        # todo generatet the offsets
        self.sampling_offsets = nn.Linear(
            embed_dims*self.num_bev_queue, num_bev_queue*num_heads * num_levels * num_points * 2)
        # todo attention
        self.attention_weights = nn.Linear(embed_dims*self.num_bev_queue,
                                           num_bev_queue*num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                **kwargs):

        # todo copy the it self (2, 180*180, 256).
        if identity is None:
            identity = query

        # todo add the position
        if query_pos is not None:
            query = query + query_pos

        # todo change to (bs, num_query ,embed_dims)
        bs,  num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        # todo keep the nums==h*w
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 1

        # todo concat the query and value with embedding dimension
        # query = torch.cat([value[:bs], query], -1)
        # todo project
        value = self.value_proj(value)

        # todo not use
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        # todo reshape the value (2*1, 180*180, 8, 64)
        value = value.reshape(bs*self.num_bev_queue,
                              num_value, self.num_heads, -1)
        # todo sample-offset
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads,  self.num_bev_queue, self.num_levels, self.num_points, 2)
        # todo attention weight
        attention_weights = self.attention_weights(query).view(
            bs, num_query,  self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()

        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        # todo [2, 32400, 1, 2]
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        # todo calculate the attention
        if torch.cuda.is_available() and value.is_cuda:
            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            # todo get the prediction output shape (bs*num_bev_queue, num_query, embed_dims)
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        # todo (2, 180*180, 256)  the resdul is necessry
        output = self.output_proj(output)
        results = self.dropout(output) + identity
        return results


class Fusionformerlayer(nn.Module):
    def __init__(self,
                 embed_dims =256):
        super(Fusionformerlayer, self).__init__()

        self.embed_dims = embed_dims
        # todo attention module
        self.attention = Deformable_AT()

        # todo norm module
        norm_cfg = dict(type='LN')
        self.norm1 = build_norm_layer(norm_cfg, self.embed_dims)[1]
        self.norm2 = build_norm_layer(norm_cfg, self.embed_dims)[1]

        # todo ffn module
        ffn_cfgs = dict(
            type='FFN',
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.,
            act_cfg=dict(type='ReLU', inplace=True))
        # todo we just build one layer ffn
        self.ffns = build_feedforward_network(ffn_cfgs)

    def forward(self, query, key, bev_pos, ref_2d):
        # todo change the shape
        bs, embeding_c, H, W = query.shape
        # todo (2, 256, 180, 180)-> (2, 256, 180*180).
        query = query.reshape(query.shape[0], query.shape[1], -1)
        # todo (2, 256, 180*180)-> (2, 180*180, 256).
        query = query.permute(0, 2, 1)
        # todo (2, 256, 180, 180)-> (2, 256, 180*180).
        key = key.reshape(key.shape[0], key.shape[1], -1)
        # todo (2, 256, 180*180)-> (2, 180*180, 256).
        key = key.permute(0, 2, 1)

        # todo attention and add itself
        query = self.attention(query, key, key, None, query_pos=bev_pos, key_pos=bev_pos,
                             reference_points=ref_2d,
                             spatial_shapes=torch.tensor([[H, W]], device=query.device),
                             level_start_index=torch.tensor([0], device=query.device))
        identity = query
        # todo norm
        query = self.norm1(query)

        # todo ffn and add
        query = self.ffns(query, identity)

        # todo norm
        query = self.norm2(query)

        # todo change the shape
        # todo (2, 180*180, 256) ->(2, 256, 180, 180) final
        query = query.permute(0, 2, 1)
        query = query.reshape(bs, embeding_c, H, W)

        return query


class MMF5(nn.Module):
    def __init__(self):
        super().__init__()

        # todo define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.m_channel = 512
        self.fusion_layers = 4
        self.bn_momentum = 0.1

        # todo encoding the position
        positional_encoding = dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)

        # todo translate the channel
        self.cam_conv = ConvBNReLU(self.cam_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)
        self.lid_conv = ConvBNReLU(self.hidden_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)
        # todo fusion former layer
        self.C2L = Fusionformerlayer()
        self.C2C = Fusionformerlayer()
        self.L2C = Fusionformerlayer()
        self.L2L = Fusionformerlayer()

        # todo fuse single attention feature
        self.mix_feature_scl = ConvBNReLU(self.m_channel, self.hidden_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)
        self.mix_feature_ccl = ConvBNReLU(self.m_channel, self.hidden_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)

        # todo SEnet layer

        self.global_avg_pool_unq = torch.nn.AdaptiveAvgPool2d((1))
        self.conv_unq = ConvBNReLU(self.hidden_channel, self.hidden_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)



        self.global_avg_pool_com = torch.nn.AdaptiveAvgPool2d((1))
        self.conv_com = ConvBNReLU(self.hidden_channel, self.hidden_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)

        self.at_vis = False

        self.init_weights()

    def get_reference_points(self, H, W, Z=8,
                             num_points_in_pillar=4, dim='3d', bs=1,
                             device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum


    def forward(self, feature_cam, feature_lid):
        # todo channel traslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)

        bs, embeding_c, H, W = f_cam_new.shape
        # todo get the bev_pos
        ref_2d = self.get_reference_points(
            H, W, dim='2d', bs=f_cam_new.size(0), device=f_cam_new.device, dtype=f_cam_new.dtype)  # todo [2, 32400, 1, 2]
        dtype = f_cam_new.dtype
        bev_mask = torch.zeros((len(f_cam_new), H, W), device=f_cam_new.device).to(dtype) # todo [2, 180, 180]
        bev_pos = self.positional_encoding(bev_mask).to(dtype) # todo [2, 256, 180, 180]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # todo [32400, 2, 256]
        bev_pos = bev_pos.permute(1, 0, 2) # todo [2, 32400, 256]

        # todo
        C2L = self.C2L(f_cam_new, f_lid_new, bev_pos, ref_2d)
        L2C = self.L2C(f_lid_new, f_cam_new, bev_pos, ref_2d)
        C2C = self.C2C(f_cam_new, f_cam_new, bev_pos, ref_2d)
        L2L = self.L2L(f_lid_new, f_lid_new, bev_pos, ref_2d)

        # todo common mask
        mask = f_cam_new * f_lid_new

        # todo fuse single feature
        SCL = self.mix_feature_scl((torch.cat((C2C, L2L), dim=1)))
        # todo fuse the common feature
        CCL = self.mix_feature_ccl((torch.cat((C2L, L2C), dim=1)))
        CCL = CCL * mask

        # todo se layers
        SCL_at = self.global_avg_pool_unq(SCL)
        SCL_at = self.conv_unq(SCL_at)
        SCL_at = F.sigmoid(SCL_at)

        CCL_at = self.global_avg_pool_com(CCL)
        CCL_at = self.conv_com(CCL_at)
        CCL_at = F.sigmoid(CCL_at)

        # todo final output
        SCL = SCL * SCL_at
        CCL = CCL * CCL_at

        return SCL, CCL


class SKConv(nn.Module):
    def __init__(self, features =256, M =2, G =1, r = 2, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features,
                              features,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i,
                              groups=G), nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False)))

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # todo x = torch.ones((32, 256, 24, 24))
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        # todo get the fuse channel
        fea_U = torch.sum(feas, dim=1)
        # todo averge pooling
        fea_s = fea_U.mean(-1).mean(-1)
        # todo linear fc
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        # todo get the attention channel
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        # todo add themself
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


# todo defor at + SK
class MMF6(nn.Module):
    def __init__(self):
        super().__init__()

        # todo define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.m_channel = 512
        self.num_layers = 1
        self.bn_momentum = 0.1
        self.grid_size = [1440, 1440, 41]
        self.pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        self.voxel_size = [0.075, 0.075, 0.2]
        self.out_size_factor = 8
        self.gaussian_overlap = 0.1
        self.min_radius = 2

        self.consistency_loss_weight = 0.1
        self.loss_cam_heatmap_weight = 1.0
        self.loss_lid_heatmap_weight = 1.0

        # todo encoding the position
        positional_encoding = dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)

        # todo adapte the channel
        self.cam_conv = build_conv_layer(dict(type="Conv2d"), self.cam_channel, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)
        self.lid_conv = build_conv_layer(dict(type="Conv2d"), self.hidden_channel, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)


        # todo generate the heatmap
        cam_layers = []
        cam_layers.append( ConvModule( self.hidden_channel, self.hidden_channel, kernel_size=3,
                padding=1, bias='auto', conv_cfg=dict(type="Conv2d"),norm_cfg=dict(type="BN2d"),))
        cam_layers.append( build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, self.hidden_channel, kernel_size=3, padding=1,bias='auto',))
        self.cam_heatmap_head = nn.Sequential(*cam_layers)
        self.cam_heatmap_headhead = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 10, kernel_size=3, padding=1,bias='auto',)

        lid_layers = []
        lid_layers.append( ConvModule( self.hidden_channel, self.hidden_channel, kernel_size=3,
                padding=1, bias='auto', conv_cfg=dict(type="Conv2d"),norm_cfg=dict(type="BN2d"),))
        lid_layers.append( build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, self.hidden_channel ,kernel_size=3, padding=1,bias='auto',))
        self.lid_heatmap_head = nn.Sequential(*lid_layers)
        self.lid_heatmap_headhead = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 10, kernel_size=3, padding=1,bias='auto',)

        # todo fusion former layer
        self.C2L = Fusionformerlayer()
        self.C2C = Fusionformerlayer()
        self.L2C = Fusionformerlayer()
        self.L2L = Fusionformerlayer()

        self.mix_feature_camera = build_conv_layer(dict(type="Conv2d"), self.hidden_channel * 2, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)
        self.mix_feature_lidar = build_conv_layer(dict(type="Conv2d"), self.hidden_channel * 2, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)

        # todo channel attention
        self.cam_sk = SKConv(256, M=2, G=1, r=2)
        self.lid_sk = SKConv(256, M=2, G=1, r=2)

        loss_heatmap=dict(type="GaussianFocalLoss", reduction="mean")
        self.loss_heatmap = build_loss(loss_heatmap)

        self.init_weights()

    def get_reference_points(self, H, W, Z=8,
                             num_points_in_pillar=4, dim='3d', bs=1,
                             device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, feature_cam, feature_lid, gt_bboxes_3d, gt_labels_3d):
        # todo channel traslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)

        # todo get the bev_pos
        bs, embeding_c, H, W = f_cam_new.shape
        ref_2d = self.get_reference_points(
            H, W, dim='2d', bs=f_cam_new.size(0), device=f_cam_new.device, dtype=f_cam_new.dtype)  # todo [2, 32400, 1, 2]
        dtype = f_cam_new.dtype
        bev_mask = torch.zeros((len(f_cam_new), H, W), device=f_cam_new.device).to(dtype) # todo [2, 180, 180]
        bev_pos = self.positional_encoding(bev_mask).to(dtype) # todo [2, 256, 180, 180]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # todo [32400, 2, 256]
        bev_pos = bev_pos.permute(1, 0, 2) # todo [2, 32400, 256]

        # todo heatmap generate
        lidar_dense_heatmap = self.lid_heatmap_head(f_lid_new)
        lidar_dense_heat_heat = self.lid_heatmap_headhead(lidar_dense_heatmap)
        lidar_heatmap = lidar_dense_heatmap.sigmoid()

        camera_dense_heatmap = self.cam_heatmap_head(f_cam_new)
        camera_dense_heat_heat = self.cam_heatmap_headhead(camera_dense_heatmap)
        camera_heatmap = camera_dense_heatmap.sigmoid()

        # todo get the spatial attention
        C2L = self.C2L(camera_heatmap, lidar_heatmap , bev_pos, ref_2d)
        L2C = self.L2C(lidar_heatmap , camera_heatmap, bev_pos, ref_2d)
        C2C = self.C2C(camera_heatmap, camera_heatmap, bev_pos, ref_2d)
        L2L = self.L2L(lidar_heatmap , lidar_heatmap , bev_pos, ref_2d)

        # todo get the channel attention
        camera_mask = self.mix_feature_camera((torch.cat((C2C, C2L), dim=1)))
        lidar_mask = self.mix_feature_lidar((torch.cat((L2L, L2C), dim=1)))

        # # todo get the channel attention
        # cam_sk_feature = self.cam_sk(f_cam_new)
        # lid_sk_feature = self.lid_sk(f_lid_new)

        # todo channel->spatial
        f_cam_new_at = self.cam_sk(f_cam_new) * camera_mask
        f_lid_new_at = self.lid_sk(f_lid_new) * lidar_mask

        # todo spatial->channel
        # f_cam_new_at = self.cam_sk(camera_mask * f_cam_new)
        # f_lid_new_at = self.lid_sk(lidar_mask * f_lid_new)

        # todo consisteance loss
        loss_consistency = self.sup_direct_loss(f_cam_new, f_lid_new) * self.consistency_loss_weight
        # loss_consistency = self.sup_direct_loss(camera_dense_heat_heat, lidar_dense_heat_heat) * self.consistency_loss_weight
        # todo heatmap_gauss loss
        heatmap = self.get_targets(gt_bboxes_3d, gt_labels_3d, f_cam_new)
        loss_cam_heatmap = self.loss_heatmap(
            self.clip_sigmoid(camera_dense_heat_heat),
            heatmap,
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        ) * self.loss_cam_heatmap_weight
        loss_lid_heatmap = self.loss_heatmap(
            self.clip_sigmoid(lidar_dense_heat_heat),
            heatmap,
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        ) * self.loss_lid_heatmap_weight

        return f_cam_new_at,  f_lid_new_at, loss_consistency, loss_cam_heatmap, loss_lid_heatmap

    def sup_direct_loss(self, f_cam, f_lid):
        # todo cam f0[2, 80, 180, 180] lidar f1[2, 256, 180, 180]
        loss_fea = torch.sqrt(torch.pow(f_cam - f_lid, 2).sum())
        sup_loss = loss_fea / (f_cam.size(2) * f_cam.size(3) * f_cam.size(0))
        return sup_loss

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, feature):
        res_tuple = multi_apply(
            self.get_targets_single,
            gt_bboxes_3d,
            gt_labels_3d,
            feature,
        )
        heatmap = torch.cat(res_tuple[0], dim=0)
        return heatmap

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, feature):
        device = feature.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
        ).to(device)
        grid_size = torch.tensor(self.grid_size)
        pc_range = torch.tensor(self.pc_range)
        voxel_size = torch.tensor(self.voxel_size)
        feature_map_size = (
            grid_size[:2] // self.out_size_factor
        )  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(
            10, feature_map_size[1], feature_map_size[0]
        )
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.out_size_factor
            length = length / voxel_size[1] / self.out_size_factor
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width), min_overlap=self.gaussian_overlap
                )
                radius = max(self.min_radius, int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (
                    (x - pc_range[0])
                    / voxel_size[0]
                    / self.out_size_factor
                )
                coor_y = (
                    (y - pc_range[1])
                    / voxel_size[1]
                    / self.out_size_factor
                )

                center = torch.tensor(
                    [coor_x, coor_y], dtype=torch.float32, device=device
                )
                center_int = center.to(torch.int32)
                draw_heatmap_gaussian(
                    heatmap[gt_labels_3d[idx]], center_int[[1, 0]], radius
                )
        # todo heatmap[10,180,180]
        heatmap = heatmap.unsqueeze(0)
        # todo heatmap[1, 10,180,180]
        return (heatmap[None])

    def clip_sigmoid(self, x, eps=1e-4):
        y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
        return y


class MMF6_test(nn.Module):
    def __init__(self):
        super().__init__()

        # todo define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.m_channel = 512
        self.num_layers = 1
        self.bn_momentum = 0.1
        self.grid_size = [1440, 1440, 41]
        self.pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        self.voxel_size = [0.075, 0.075, 0.2]
        self.out_size_factor = 8
        self.gaussian_overlap = 0.1
        self.min_radius = 2

        self.consistency_loss_weight = 0.1
        self.loss_cam_heatmap_weight = 1.0
        self.loss_lid_heatmap_weight = 1.0

        # todo encoding the position
        positional_encoding = dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)

        # todo adapte the channel
        self.cam_conv = build_conv_layer(dict(type="Conv2d"), self.cam_channel, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)
        self.lid_conv = build_conv_layer(dict(type="Conv2d"), self.hidden_channel, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)


        # todo generate the heatmap
        cam_layers = []
        cam_layers.append( ConvModule( self.hidden_channel, self.hidden_channel, kernel_size=3,
                padding=1, bias='auto', conv_cfg=dict(type="Conv2d"),norm_cfg=dict(type="BN2d"),))
        cam_layers.append( build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, self.hidden_channel, kernel_size=3, padding=1,bias='auto',))
        self.cam_heatmap_head = nn.Sequential(*cam_layers)
        self.cam_heatmap_headhead = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 10, kernel_size=3, padding=1,bias='auto',)

        lid_layers = []
        lid_layers.append( ConvModule( self.hidden_channel, self.hidden_channel, kernel_size=3,
                padding=1, bias='auto', conv_cfg=dict(type="Conv2d"),norm_cfg=dict(type="BN2d"),))
        lid_layers.append( build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, self.hidden_channel ,kernel_size=3, padding=1,bias='auto',))
        self.lid_heatmap_head = nn.Sequential(*lid_layers)
        self.lid_heatmap_headhead = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 10, kernel_size=3, padding=1,bias='auto',)

        # todo fusion former layer
        self.C2L = Fusionformerlayer()
        self.C2C = Fusionformerlayer()
        self.L2C = Fusionformerlayer()
        self.L2L = Fusionformerlayer()

        self.mix_feature_camera = build_conv_layer(dict(type="Conv2d"), self.hidden_channel * 2, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)
        self.mix_feature_lidar = build_conv_layer(dict(type="Conv2d"), self.hidden_channel * 2, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)

        # todo channel attention
        self.cam_sk = SKConv(256, M=2, G=1, r=2)
        self.lid_sk = SKConv(256, M=2, G=1, r=2)

        loss_heatmap=dict(type="GaussianFocalLoss", reduction="mean")
        self.loss_heatmap = build_loss(loss_heatmap)

        self.init_weights()

    def get_reference_points(self, H, W, Z=8,
                             num_points_in_pillar=4, dim='3d', bs=1,
                             device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, feature_cam, feature_lid):
        # todo channel traslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)

        # todo get the bev_pos
        bs, embeding_c, H, W = f_cam_new.shape
        ref_2d = self.get_reference_points(
            H, W, dim='2d', bs=f_cam_new.size(0), device=f_cam_new.device, dtype=f_cam_new.dtype)  # todo [2, 32400, 1, 2]
        dtype = f_cam_new.dtype
        bev_mask = torch.zeros((len(f_cam_new), H, W), device=f_cam_new.device).to(dtype) # todo [2, 180, 180]
        bev_pos = self.positional_encoding(bev_mask).to(dtype) # todo [2, 256, 180, 180]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # todo [32400, 2, 256]
        bev_pos = bev_pos.permute(1, 0, 2) # todo [2, 32400, 256]

        # todo heatmap generate
        lidar_dense_heatmap = self.lid_heatmap_head(f_lid_new)
        lidar_dense_heat_heat = self.lid_heatmap_headhead(lidar_dense_heatmap)
        lidar_heatmap = lidar_dense_heatmap.sigmoid()

        camera_dense_heatmap = self.cam_heatmap_head(f_cam_new)
        camera_dense_heat_heat = self.cam_heatmap_headhead(camera_dense_heatmap)
        camera_heatmap = camera_dense_heatmap.sigmoid()

        # todo get the spatial attention
        C2L = self.C2L(camera_heatmap, lidar_heatmap , bev_pos, ref_2d)
        L2C = self.L2C(lidar_heatmap , camera_heatmap, bev_pos, ref_2d)
        C2C = self.C2C(camera_heatmap, camera_heatmap, bev_pos, ref_2d)
        L2L = self.L2L(lidar_heatmap , lidar_heatmap , bev_pos, ref_2d)

        # todo get the channel attention
        camera_mask = self.mix_feature_camera((torch.cat((C2C, C2L), dim=1)))
        lidar_mask = self.mix_feature_lidar((torch.cat((L2L, L2C), dim=1)))

        # # todo get the channel attention
        # cam_sk_feature = self.cam_sk(f_cam_new)
        # lid_sk_feature = self.lid_sk(f_lid_new)

        # todo channel->spatial
        f_cam_new_at = self.cam_sk(f_cam_new) * camera_mask
        f_lid_new_at = self.lid_sk(f_lid_new) * lidar_mask

        # todo spatial->channel
        # f_cam_new_at = self.cam_sk(camera_mask * f_cam_new)
        # f_lid_new_at = self.lid_sk(lidar_mask * f_lid_new)

        return f_cam_new_at,  f_lid_new_at

    def sup_direct_loss(self, f_cam, f_lid):
        # todo cam f0[2, 80, 180, 180] lidar f1[2, 256, 180, 180]
        loss_fea = torch.sqrt(torch.pow(f_cam - f_lid, 2).sum())
        sup_loss = loss_fea / (f_cam.size(2) * f_cam.size(3) * f_cam.size(0))
        return sup_loss

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, feature):
        res_tuple = multi_apply(
            self.get_targets_single,
            gt_bboxes_3d,
            gt_labels_3d,
            feature,
        )
        heatmap = torch.cat(res_tuple[0], dim=0)
        return heatmap

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, feature):
        device = feature.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
        ).to(device)
        grid_size = torch.tensor(self.grid_size)
        pc_range = torch.tensor(self.pc_range)
        voxel_size = torch.tensor(self.voxel_size)
        feature_map_size = (
            grid_size[:2] // self.out_size_factor
        )  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(
            10, feature_map_size[1], feature_map_size[0]
        )
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.out_size_factor
            length = length / voxel_size[1] / self.out_size_factor
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width), min_overlap=self.gaussian_overlap
                )
                radius = max(self.min_radius, int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (
                    (x - pc_range[0])
                    / voxel_size[0]
                    / self.out_size_factor
                )
                coor_y = (
                    (y - pc_range[1])
                    / voxel_size[1]
                    / self.out_size_factor
                )

                center = torch.tensor(
                    [coor_x, coor_y], dtype=torch.float32, device=device
                )
                center_int = center.to(torch.int32)
                draw_heatmap_gaussian(
                    heatmap[gt_labels_3d[idx]], center_int[[1, 0]], radius
                )
        # todo heatmap[10,180,180]
        heatmap = heatmap.unsqueeze(0)
        # todo heatmap[1, 10,180,180]
        return (heatmap[None])

    def clip_sigmoid(self, x, eps=1e-4):
        y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
        return y


# todo SKnet
class MMF7(nn.Module):
    def __init__(self):
        super().__init__()

        # todo define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.m_channel = 512
        self.num_layers = 1
        self.bn_momentum = 0.1
        self.grid_size = [1440, 1440, 41]
        self.pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        self.voxel_size = [0.075, 0.075, 0.2]
        self.out_size_factor = 8
        self.gaussian_overlap = 0.1
        self.min_radius = 2

        self.consistency_loss_weight = 0.1
        self.loss_cam_heatmap_weight = 0.5
        self.loss_lid_heatmap_weight = 0.5

        # todo adapte the channel
        self.cam_conv = build_conv_layer(dict(type="Conv2d"), self.cam_channel, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)
        self.lid_conv = build_conv_layer(dict(type="Conv2d"), self.hidden_channel, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)


        # todo generate the heatmap
        cam_layers = []
        cam_layers.append( ConvModule( self.hidden_channel, self.hidden_channel, kernel_size=3,
                padding=1, bias='auto', conv_cfg=dict(type="Conv2d"),norm_cfg=dict(type="BN2d"),))
        cam_layers.append( build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, self.hidden_channel, kernel_size=3, padding=1,bias='auto',))
        self.cam_heatmap_head = nn.Sequential(*cam_layers)

        lid_layers = []
        lid_layers.append( ConvModule( self.hidden_channel, self.hidden_channel, kernel_size=3,
                padding=1, bias='auto', conv_cfg=dict(type="Conv2d"),norm_cfg=dict(type="BN2d"),))
        lid_layers.append( build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, self.hidden_channel ,kernel_size=3, padding=1,bias='auto',))
        self.lid_heatmap_head = nn.Sequential(*lid_layers)


        # todo channel attention
        self.cam_sk = SKConv(256, M=2, G=1, r=2)
        self.lid_sk = SKConv(256, M=2, G=1, r=2)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, feature_cam, feature_lid, gt_bboxes_3d, gt_labels_3d):
        # todo channel traslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)

        # todo heatmap generate
        lidar_dense_heatmap = self.lid_heatmap_head(f_cam_new)
        camera_dense_heatmap = self.cam_heatmap_head(f_lid_new)

        # todo get the channel attention
        cam_sk_feature = self.cam_sk(camera_dense_heatmap)
        lid_sk_feature = self.lid_sk(lidar_dense_heatmap)

        return cam_sk_feature,  lid_sk_feature

    def sup_direct_loss(self, f_cam, f_lid):
        # todo cam f0[2, 80, 180, 180] lidar f1[2, 256, 180, 180]
        loss_fea = torch.sqrt(torch.pow(f_cam - f_lid, 2).sum())
        sup_loss = loss_fea / (f_cam.size(2) * f_cam.size(3) * f_cam.size(0))
        return sup_loss

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, feature):
        res_tuple = multi_apply(
            self.get_targets_single,
            gt_bboxes_3d,
            gt_labels_3d,
            feature,
        )
        heatmap = torch.cat(res_tuple[0], dim=0)
        return heatmap

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, feature):

        # # compute dense heatmap targets
        device = feature.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
        ).to(device)
        grid_size = torch.tensor(self.grid_size)
        pc_range = torch.tensor(self.pc_range)
        voxel_size = torch.tensor(self.voxel_size)
        feature_map_size = (
            grid_size[:2] // self.out_size_factor
        )  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(
            256, feature_map_size[1], feature_map_size[0]
        )
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.out_size_factor
            length = length / voxel_size[1] / self.out_size_factor
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width), min_overlap=self.gaussian_overlap
                )
                radius = max(self.min_radius, int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (
                    (x - pc_range[0])
                    / voxel_size[0]
                    / self.out_size_factor
                )
                coor_y = (
                    (y - pc_range[1])
                    / voxel_size[1]
                    / self.out_size_factor
                )

                center = torch.tensor(
                    [coor_x, coor_y], dtype=torch.float32, device=device
                )
                center_int = center.to(torch.int32)
                draw_heatmap_gaussian(
                    heatmap[gt_labels_3d[idx]], center_int[[1, 0]], radius
                )
        # todo heatmap[10,180,180]
        heatmap = heatmap.unsqueeze(0)
        # todo heatmap[1, 10,180,180]
        return (heatmap[None])

    def clip_sigmoid(self, x, eps=1e-4):
        y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
        return y


# todo common and unqiue
class MMF8(nn.Module):
    def __init__(self):
        super().__init__()

        # todo define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.m_channel = 512
        self.num_layers = 1
        self.bn_momentum = 0.1
        self.grid_size = [1440, 1440, 41]
        self.pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        self.voxel_size = [0.075, 0.075, 0.2]
        self.out_size_factor = 8
        self.gaussian_overlap = 0.1
        self.min_radius = 2

        self.consistency_loss_weight = 0.1
        self.loss_cam_heatmap_weight = 1.0
        self.loss_lid_heatmap_weight = 1.0

        # todo encoding the position
        positional_encoding = dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)

        # todo adapte the channel
        self.cam_conv = build_conv_layer(dict(type="Conv2d"), self.cam_channel, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)
        self.lid_conv = build_conv_layer(dict(type="Conv2d"), self.hidden_channel, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)


        # todo generate the heatmap
        cam_layers = []
        cam_layers.append( ConvModule( self.hidden_channel, self.hidden_channel, kernel_size=3,
                padding=1, bias='auto', conv_cfg=dict(type="Conv2d"),norm_cfg=dict(type="BN2d"),))
        cam_layers.append( build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, self.hidden_channel, kernel_size=3, padding=1,bias='auto',))
        self.cam_heatmap_head = nn.Sequential(*cam_layers)
        self.cam_heatmap_headhead = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel*2, 10, kernel_size=3, padding=1,bias='auto',)

        lid_layers = []
        lid_layers.append( ConvModule( self.hidden_channel, self.hidden_channel, kernel_size=3,
                padding=1, bias='auto', conv_cfg=dict(type="Conv2d"),norm_cfg=dict(type="BN2d"),))
        lid_layers.append( build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, self.hidden_channel ,kernel_size=3, padding=1,bias='auto',))
        self.lid_heatmap_head = nn.Sequential(*lid_layers)
        self.lid_heatmap_headhead = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel*2, 10, kernel_size=3, padding=1, bias='auto',)

        # todo fusion former layer
        self.CL = Fusionformerlayer()
        self.C2C = Fusionformerlayer()
        self.L2L = Fusionformerlayer()

        # todo mix the future
        self.mix_feature_common = build_conv_layer(dict(type="Conv2d"), self.hidden_channel * 2, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)
        self.mix_feature_camera = build_conv_layer(dict(type="Conv2d"), self.hidden_channel, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)
        self.mix_feature_lidar = build_conv_layer(dict(type="Conv2d"), self.hidden_channel, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)

        # todo channel attention
        self.com_sk = SKConv(256, M=2, G=1, r=2)
        self.cam_sk = SKConv(256, M=2, G=1, r=2)
        self.lid_sk = SKConv(256, M=2, G=1, r=2)

        loss_heatmap=dict(type="GaussianFocalLoss", reduction="mean")
        self.loss_heatmap = build_loss(loss_heatmap)

        self.init_weights()

    def get_reference_points(self, H, W, Z=8,
                             num_points_in_pillar=4, dim='3d', bs=1,
                             device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def get_bev_pos(self, f_cam_new):
        bs, embeding_c, H, W = f_cam_new.shape
        ref_2d = self.get_reference_points(
            H, W, dim='2d', bs=f_cam_new.size(0), device=f_cam_new.device, dtype=f_cam_new.dtype)  # todo [2, 32400, 1, 2]
        dtype = f_cam_new.dtype
        bev_mask = torch.zeros((len(f_cam_new), H, W), device=f_cam_new.device).to(dtype) # todo [2, 180, 180]
        bev_pos = self.positional_encoding(bev_mask).to(dtype) # todo [2, 256, 180, 180]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # todo [32400, 2, 256]
        bev_pos = bev_pos.permute(1, 0, 2) # todo [2, 32400, 256]

        return bev_pos, ref_2d

    def forward(self, feature_cam, feature_lid, gt_bboxes_3d, gt_labels_3d):
        # todo channel traslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)

        # todo get the bev_pos
        bev_pos, ref_2d = self.get_bev_pos(f_cam_new)

        # todo heatmap generate
        lidar_dense_heatmap = self.lid_heatmap_head(f_lid_new)
        lidar_heatmap = lidar_dense_heatmap.sigmoid()

        camera_dense_heatmap = self.cam_heatmap_head(f_cam_new)
        camera_heatmap = camera_dense_heatmap.sigmoid()

        # todo get the spatial attention
        C2L = self.CL(camera_heatmap, lidar_heatmap , bev_pos, ref_2d)
        L2C = self.CL(lidar_heatmap , camera_heatmap, bev_pos, ref_2d)
        C2C = self.C2C(camera_heatmap, camera_heatmap, bev_pos, ref_2d)
        L2L = self.L2L(lidar_heatmap , lidar_heatmap , bev_pos, ref_2d)

        # todo get the channel attention
        common_mask = self.mix_feature_common((torch.cat((C2L, L2C), dim=1)))
        camera_mask = self.mix_feature_camera(C2C)
        lidar_mask = self.mix_feature_lidar(L2L)

        # todo channel->spatial
        f_cam_new_at = self.cam_sk(f_cam_new) * camera_mask
        f_lid_new_at = self.lid_sk(f_lid_new) * lidar_mask
        f_common_at = self.com_sk(f_cam_new * f_lid_new) * common_mask

        # todo spatial->channel
        # f_cam_new_at = self.cam_sk(camera_mask * f_cam_new)
        # f_lid_new_at = self.lid_sk(lidar_mask * f_lid_new)
        # f_common_at = self.com_sk(common_mask * f_cam_new * f_lid_new)

        # todo consisteance loss
        loss_consistency = self.sup_direct_loss(f_cam_new, f_lid_new) * self.consistency_loss_weight
        # todo heatmap_gauss loss
        heatmap = self.get_targets(gt_bboxes_3d, gt_labels_3d, f_cam_new)
        # todo heatmap_loss
        camera_dense_heat_heat = self.cam_heatmap_headhead((torch.cat((C2L, C2C), dim=1)))
        lidar_dense_heat_heat = self.lid_heatmap_headhead((torch.cat((L2C, L2L), dim=1)))
        loss_cam_heatmap = self.loss_heatmap(
            self.clip_sigmoid(camera_dense_heat_heat),
            heatmap,
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        ) * self.loss_cam_heatmap_weight
        loss_lid_heatmap = self.loss_heatmap(
            self.clip_sigmoid(lidar_dense_heat_heat),
            heatmap,
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        ) * self.loss_lid_heatmap_weight

        return f_cam_new_at,  f_lid_new_at, f_common_at, loss_consistency, loss_cam_heatmap, loss_lid_heatmap

    def sup_direct_loss(self, f_cam, f_lid):
        # todo cam f0[2, 80, 180, 180] lidar f1[2, 256, 180, 180]
        loss_fea = torch.sqrt(torch.pow(f_cam - f_lid, 2).sum())
        sup_loss = loss_fea / (f_cam.size(2) * f_cam.size(3) * f_cam.size(0))
        return sup_loss

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, feature):
        res_tuple = multi_apply(
            self.get_targets_single,
            gt_bboxes_3d,
            gt_labels_3d,
            feature,
        )
        heatmap = torch.cat(res_tuple[0], dim=0)
        return heatmap

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, feature):
        device = feature.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
        ).to(device)
        grid_size = torch.tensor(self.grid_size)
        pc_range = torch.tensor(self.pc_range)
        voxel_size = torch.tensor(self.voxel_size)
        feature_map_size = (
            grid_size[:2] // self.out_size_factor
        )  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(
            10, feature_map_size[1], feature_map_size[0]
        )
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.out_size_factor
            length = length / voxel_size[1] / self.out_size_factor
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width), min_overlap=self.gaussian_overlap
                )
                radius = max(self.min_radius, int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (
                    (x - pc_range[0])
                    / voxel_size[0]
                    / self.out_size_factor
                )
                coor_y = (
                    (y - pc_range[1])
                    / voxel_size[1]
                    / self.out_size_factor
                )

                center = torch.tensor(
                    [coor_x, coor_y], dtype=torch.float32, device=device
                )
                center_int = center.to(torch.int32)
                draw_heatmap_gaussian(
                    heatmap[gt_labels_3d[idx]], center_int[[1, 0]], radius
                )
        # todo heatmap[10,180,180]
        heatmap = heatmap.unsqueeze(0)
        # todo heatmap[1, 10,180,180]
        return (heatmap[None])

    def clip_sigmoid(self, x, eps=1e-4):
        y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
        return y


# todo new common and unique
class MMF11(nn.Module):
    def __init__(self):
        super().__init__()

        # todo define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.m_channel = 512
        self.num_layers = 1
        self.bn_momentum = 0.1
        self.grid_size = [1440, 1440, 41]
        self.pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        self.voxel_size = [0.075, 0.075, 0.2]
        self.out_size_factor = 8
        self.gaussian_overlap = 0.1
        self.min_radius = 2

        self.consistency_loss_weight = 0.1
        self.loss_cam_heatmap_weight = 0.1
        self.loss_lid_heatmap_weight = 0.1
        self.mask_loss_weight = 1
        self.eps = 1e-4

        # todo encoding the position
        positional_encoding = dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)

        # todo adapte the channel
        self.cam_conv = build_conv_layer(dict(type="Conv2d"), self.cam_channel, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)
        self.lid_conv = build_conv_layer(dict(type="Conv2d"), self.hidden_channel, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)


        # todo generate the heatmap
        cam_layers = []
        cam_layers.append( ConvModule( self.hidden_channel, self.hidden_channel, kernel_size=3,
                padding=1, bias='auto', conv_cfg=dict(type="Conv2d"),norm_cfg=dict(type="BN2d"),))
        cam_layers.append( build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, self.hidden_channel, kernel_size=3, padding=1,bias='auto',))
        self.cam_heatmap_head = nn.Sequential(*cam_layers)
        self.cam_heatmap_headhead = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 10, kernel_size=3, padding=1,bias='auto',)

        lid_layers = []
        lid_layers.append( ConvModule( self.hidden_channel, self.hidden_channel, kernel_size=3,
                padding=1, bias='auto', conv_cfg=dict(type="Conv2d"),norm_cfg=dict(type="BN2d"),))
        lid_layers.append( build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, self.hidden_channel ,kernel_size=3, padding=1,bias='auto',))
        self.lid_heatmap_head = nn.Sequential(*lid_layers)
        self.lid_heatmap_headhead = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 10, kernel_size=3, padding=1,bias='auto',)

        # todo fusion former layer
        self.C2L = Fusionformerlayer()
        self.C2C = Fusionformerlayer()
        self.L2C = Fusionformerlayer()
        self.L2L = Fusionformerlayer()

        self.mix_feature_camera = build_conv_layer(dict(type="Conv2d"), self.hidden_channel * 2, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)
        self.mix_feature_lidar = build_conv_layer(dict(type="Conv2d"), self.hidden_channel * 2, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)

        # todo encode the sigma and mu
        self.mu = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 1, kernel_size=3, padding=1,bias='auto',)
        self.sigma = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 1, kernel_size=3, padding=1,bias='auto',)
        self.mu2 = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 1, kernel_size=3, padding=1,bias='auto',)
        self.sigma2 = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 1, kernel_size=3, padding=1,bias='auto',)


        # todo channel attention
        self.cam_sk = SKConv(256, M=2, G=1, r=2)
        self.lid_sk = SKConv(256, M=2, G=1, r=2)

        loss_heatmap=dict(type="GaussianFocalLoss", reduction="mean")
        self.loss_heatmap = build_loss(loss_heatmap)

        self.init_weights()

    def get_reference_points(self, H, W, Z=8,
                             num_points_in_pillar=4, dim='3d', bs=1,
                             device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, feature_cam, feature_lid, gt_bboxes_3d, gt_labels_3d):
        # todo channel traslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)

        # todo get the bev_pos
        bs, embeding_c, H, W = f_cam_new.shape
        ref_2d = self.get_reference_points(
            H, W, dim='2d', bs=f_cam_new.size(0), device=f_cam_new.device, dtype=f_cam_new.dtype)  # todo [2, 32400, 1, 2]
        dtype = f_cam_new.dtype
        bev_mask = torch.zeros((len(f_cam_new), H, W), device=f_cam_new.device).to(dtype) # todo [2, 180, 180]
        bev_pos = self.positional_encoding(bev_mask).to(dtype) # todo [2, 256, 180, 180]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # todo [32400, 2, 256]
        bev_pos = bev_pos.permute(1, 0, 2) # todo [2, 32400, 256]

        # todo heatmap generate
        lidar_dense_heatmap = self.lid_heatmap_head(f_lid_new)
        lidar_dense_heat_heat = self.lid_heatmap_headhead(lidar_dense_heatmap)
        lidar_heatmap = lidar_dense_heatmap.sigmoid()

        camera_dense_heatmap = self.cam_heatmap_head(f_cam_new)
        camera_dense_heat_heat = self.cam_heatmap_headhead(camera_dense_heatmap)
        camera_heatmap = camera_dense_heatmap.sigmoid()

        # todo get the spatial attention
        C2L = self.C2L(camera_heatmap, lidar_heatmap , bev_pos, ref_2d)
        L2C = self.L2C(lidar_heatmap , camera_heatmap, bev_pos, ref_2d)
        C2C = self.C2C(camera_heatmap, camera_heatmap, bev_pos, ref_2d)
        L2L = self.L2L(lidar_heatmap , lidar_heatmap , bev_pos, ref_2d)

        # todo get the weightattention
        camera_mask = self.mix_feature_camera((torch.cat((C2C, C2L), dim=1)))
        lidar_mask = self.mix_feature_lidar((torch.cat((L2L, L2C), dim=1)))

        # todo get the channel attention
        cam_sk_feature = self.cam_sk(f_cam_new)
        lid_sk_feature = self.lid_sk(f_lid_new)

        # todo encode the mu and sigma
        cam_mu = self.mu(camera_heatmap)
        cam_sigma = self.sigma(camera_heatmap)
        lid_mu = self.mu2(lidar_heatmap)
        lid_sigma = self.sigma2(lidar_heatmap)
        opt = [cam_mu, cam_sigma, lid_mu, lid_sigma, ref_2d]
        gauss_weight_cam = self.gauss_weight(cam_mu, cam_sigma)
        gasus_weight_lid = self.gauss_weight(lid_mu, lid_sigma)

        # todo second decision
        f_cam_new_at = cam_sk_feature * (camera_mask + gauss_weight_cam)
        f_lid_new_at = lid_sk_feature * (lidar_mask + gasus_weight_lid)

        # todo consisteance loss
        loss_consistency = self.sup_direct_loss(f_cam_new, f_lid_new) * self.consistency_loss_weight
        # todo heatmap_gauss loss
        heatmap = self.get_targets(gt_bboxes_3d, gt_labels_3d, f_cam_new)
        loss_cam_heatmap = self.loss_heatmap(
            self.clip_sigmoid(camera_dense_heat_heat),
            heatmap,
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        ) * self.loss_cam_heatmap_weight
        loss_lid_heatmap = self.loss_heatmap(
            self.clip_sigmoid(lidar_dense_heat_heat),
            heatmap,
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        ) * self.loss_lid_heatmap_weight

        return f_cam_new_at, f_lid_new_at, loss_consistency, loss_cam_heatmap, loss_lid_heatmap, opt

    def sup_direct_loss(self, f_cam, f_lid):
        # todo cam f0[2, 80, 180, 180] lidar f1[2, 256, 180, 180]
        loss_fea = torch.sqrt(torch.pow(f_cam - f_lid, 2).sum())
        sup_loss = loss_fea / (f_cam.size(2) * f_cam.size(3) * f_cam.size(0))
        return sup_loss

    def clip_sigmoid(self, x, eps=1e-4):
        y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
        return y

    def gauss_weight(self, mu, sigma, eps=1e-4):
        with torch.no_grad():
            # todo maybe there is zero which cause the issue?
            sigma = torch.clamp(sigma, min=eps,  max=sigma.max())
            mu_new = torch.pow(mu, 2)
            sigma_new = torch.pow(sigma, 2)
            x = torch.div(mu_new, sigma_new)
            y = torch.exp(-x)
            # y = (1/sigma) * torch.exp(-x)
        return y

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, feature):
        res_tuple = multi_apply(
            self.get_targets_single,
            gt_bboxes_3d,
            gt_labels_3d,
            feature,
        )
        heatmap = torch.cat(res_tuple[0], dim=0)
        return heatmap

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, feature):
        device = feature.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
        ).to(device)
        grid_size = torch.tensor(self.grid_size)
        pc_range = torch.tensor(self.pc_range)
        voxel_size = torch.tensor(self.voxel_size)
        feature_map_size = (
            grid_size[:2] // self.out_size_factor
        )  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(
            10, feature_map_size[1], feature_map_size[0]
        )
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.out_size_factor
            length = length / voxel_size[1] / self.out_size_factor
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width), min_overlap=self.gaussian_overlap
                )
                radius = max(self.min_radius, int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (
                    (x - pc_range[0])
                    / voxel_size[0]
                    / self.out_size_factor
                )
                coor_y = (
                    (y - pc_range[1])
                    / voxel_size[1]
                    / self.out_size_factor
                )

                center = torch.tensor(
                    [coor_x, coor_y], dtype=torch.float32, device=device
                )
                center_int = center.to(torch.int32)
                draw_heatmap_gaussian(
                    heatmap[gt_labels_3d[idx]], center_int[[1, 0]], radius
                )
        # todo heatmap[10,180,180]
        heatmap = heatmap.unsqueeze(0)
        # todo heatmap[1, 10,180,180]
        return (heatmap[None])

    def opt_loss(self, opt, gt_bboxes_3d, feature):
        # todo calculate every distance to ground truth
        ref_2d = opt[4] # todo [2, 32400, 1, 2]
        ref_2d = ref_2d.squeeze(2) # todo [2, 32400, 2]
        loss = 0
        # todo batch is necessary
        for i in range (len(gt_bboxes_3d)):
            center = self.opt_single(gt_bboxes_3d[i], feature) # todo [x ,2]
            # todo we calculate the distance, used the a^2+b^2-2ab
            gt_nums = len(gt_bboxes_3d[i])
            pot_nums = len(ref_2d[i,:,0])
            x1 = ref_2d[i,:,0].unsqueeze(1) # todo [32400, 1]
            y1 = ref_2d[i,:,1].unsqueeze(1) # todo [32400, 1]
            x2 = center[:, 0].unsqueeze(0) # todo [1, x]
            y2 = center[:, 1].unsqueeze(0) # todo [1, x]
            x1_copy = x1.repeat(1, gt_nums)
            x2_copy = x2.repeat(pot_nums, 1)
            y1_copy = y1.repeat(1, gt_nums)
            y2_copy = y2.repeat(pot_nums, 1)

            # dis_x = x1_copy - x2_copy
            # dis_y = y1_copy - y2_copy
            # distance = torch.sqrt(torch.pow(dis_x, 2) + torch.pow(dis_y, 2))
            distance = torch.pow(x1_copy, 2) + torch.pow(x2_copy, 2) + \
                       torch.pow(y1_copy, 2) + torch.pow(y2_copy, 2) - \
                       2*x1_copy*x2_copy - 2*y1_copy*y2_copy
            distance = torch.sqrt(distance)

            # todo get the mini distance
            distance, index = torch.min(distance, dim=1) # todo [32400]
            distance = distance / distance.max()

            # todo calculate the loss
            cam_mu = opt[0][i].squeeze(0).view(-1, 1).squeeze(1)
            cam_sigma = opt[1][i].squeeze(0).view(-1, 1).squeeze(1)
            lid_mu = opt[2][i].squeeze(0).view(-1, 1).squeeze(1)
            lid_sigma = opt[3][i].squeeze(0).view(-1, 1).squeeze(1)

            loss_cam = torch.sqrt(torch.pow(cam_mu - distance, 2).sum()) + torch.sqrt(torch.pow(cam_sigma - self.eps, 2).sum())
            loss_lid = torch.sqrt(torch.pow(lid_mu - distance, 2).sum()) + torch.sqrt(torch.pow(lid_sigma - self.eps, 2).sum())

            loss_all = (loss_cam + loss_lid) / (2 * pot_nums)
            loss = loss + loss_all
        return loss

    def opt_single(self, gt_bboxes_3d, feature):
        device = feature.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
        ).to(device)
        pc_range = torch.tensor(self.pc_range)
        voxel_size = torch.tensor(self.voxel_size)
        center_all = gt_bboxes_3d.new_zeros(len(gt_bboxes_3d), 2)
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.out_size_factor
            length = length / voxel_size[1] / self.out_size_factor
            if width > 0 and length > 0:
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]
                coor_x = (
                    (x - pc_range[0])
                    / voxel_size[0]
                    / self.out_size_factor
                )
                coor_y = (
                    (y - pc_range[1])
                    / voxel_size[1]
                    / self.out_size_factor
                )
                center = torch.tensor(
                    [coor_x, coor_y], dtype=torch.float32, device=device
                )
                center_int = center.to(torch.int32)
                center_all[idx, :] = center_int
        return center_all


# todo layer normalized and location align
class MMF11_new(nn.Module):
    def __init__(self):
        super().__init__()

        # todo define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.m_channel = 512
        self.num_layers = 1
        self.bn_momentum = 0.1
        self.grid_size = [1440, 1440, 41]
        self.pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        self.voxel_size = [0.075, 0.075, 0.2]
        self.out_size_factor = 8
        self.gaussian_overlap = 0.1
        self.min_radius = 2

        self.consistency_loss_weight = 0.1
        self.loss_cam_heatmap_weight = 1.0
        self.loss_lid_heatmap_weight = 1.0
        self.mask_loss_weight = 1
        self.eps = 1e-4

        # todo encoding the position
        positional_encoding = dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)

        # todo adapte the channel
        self.cam_conv = build_conv_layer(dict(type="Conv2d"), self.cam_channel, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)
        self.lid_conv = build_conv_layer(dict(type="Conv2d"), self.hidden_channel, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)


        # todo generate the heatmap
        cam_layers = []
        cam_layers.append( ConvModule( self.hidden_channel, self.hidden_channel, kernel_size=3,
                padding=1, bias='auto', conv_cfg=dict(type="Conv2d"),norm_cfg=dict(type="BN2d"),))
        cam_layers.append( build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, self.hidden_channel, kernel_size=3, padding=1,bias='auto',))
        self.cam_heatmap_head = nn.Sequential(*cam_layers)
        self.cam_heatmap_headhead = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 10, kernel_size=3, padding=1,bias='auto',)

        lid_layers = []
        lid_layers.append( ConvModule( self.hidden_channel, self.hidden_channel, kernel_size=3,
                padding=1, bias='auto', conv_cfg=dict(type="Conv2d"),norm_cfg=dict(type="BN2d"),))
        lid_layers.append( build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, self.hidden_channel ,kernel_size=3, padding=1,bias='auto',))
        self.lid_heatmap_head = nn.Sequential(*lid_layers)
        self.lid_heatmap_headhead = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 10, kernel_size=3, padding=1,bias='auto',)

        # todo fusion former layer
        self.C2L = Fusionformerlayer()
        self.C2C = Fusionformerlayer()
        self.L2C = Fusionformerlayer()
        self.L2L = Fusionformerlayer()

        self.mix_feature_camera = build_conv_layer(dict(type="Conv2d"), self.hidden_channel * 2, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)
        self.mix_feature_lidar = build_conv_layer(dict(type="Conv2d"), self.hidden_channel * 2, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)

        # todo encode the sigma and mu
        self.mu = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 1, kernel_size=3, padding=1,bias='auto',)
        self.sigma = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 1, kernel_size=3, padding=1,bias='auto',)
        self.mu2 = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 1, kernel_size=3, padding=1,bias='auto',)
        self.sigma2 = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 1, kernel_size=3, padding=1,bias='auto',)


        # todo channel attention
        self.cam_sk = SKConv(256, M=2, G=1, r=2)
        self.lid_sk = SKConv(256, M=2, G=1, r=2)

        loss_heatmap=dict(type="GaussianFocalLoss", reduction="mean")
        self.loss_heatmap = build_loss(loss_heatmap)

        self.init_weights()

    def get_reference_points(self, H, W, Z=8,
                             num_points_in_pillar=4, dim='3d', bs=1,
                             device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, feature_cam, feature_lid, gt_bboxes_3d, gt_labels_3d):
        # todo channel traslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)

        # todo get the bev_pos
        bs, embeding_c, H, W = f_cam_new.shape
        ref_2d = self.get_reference_points(
            H, W, dim='2d', bs=f_cam_new.size(0), device=f_cam_new.device, dtype=f_cam_new.dtype)  # todo [2, 32400, 1, 2]
        dtype = f_cam_new.dtype
        bev_mask = torch.zeros((len(f_cam_new), H, W), device=f_cam_new.device).to(dtype) # todo [2, 180, 180]
        bev_pos = self.positional_encoding(bev_mask).to(dtype) # todo [2, 256, 180, 180]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # todo [32400, 2, 256]
        bev_pos = bev_pos.permute(1, 0, 2) # todo [2, 32400, 256]

        # todo heatmap generate
        lidar_dense_heatmap = self.lid_heatmap_head(f_lid_new)
        lidar_dense_heat_heat = self.lid_heatmap_headhead(lidar_dense_heatmap)
        lidar_heatmap = lidar_dense_heatmap.sigmoid()

        camera_dense_heatmap = self.cam_heatmap_head(f_cam_new)
        camera_dense_heat_heat = self.cam_heatmap_headhead(camera_dense_heatmap)
        camera_heatmap = camera_dense_heatmap.sigmoid()

        # todo get the spatial attention
        C2L = self.C2L(camera_heatmap, lidar_heatmap , bev_pos, ref_2d)
        L2C = self.L2C(lidar_heatmap , camera_heatmap, bev_pos, ref_2d)
        C2C = self.C2C(camera_heatmap, camera_heatmap, bev_pos, ref_2d)
        L2L = self.L2L(lidar_heatmap , lidar_heatmap , bev_pos, ref_2d)

        # todo get the weightattention
        camera_mask = self.mix_feature_camera((torch.cat((C2C, C2L), dim=1)))
        lidar_mask = self.mix_feature_lidar((torch.cat((L2L, L2C), dim=1)))

        # todo get the channel attention
        cam_sk_feature = self.cam_sk(f_cam_new)
        lid_sk_feature = self.lid_sk(f_lid_new)

        # todo encode the mu and sigma
        cam_mu = self.mu(camera_heatmap)
        cam_sigma = self.sigma(camera_heatmap)
        lid_mu = self.mu2(lidar_heatmap)
        lid_sigma = self.sigma2(lidar_heatmap)
        opt = [cam_mu, cam_sigma, lid_mu, lid_sigma, ref_2d]
        gauss_weight_cam = self.gauss_weight(cam_mu, cam_sigma)
        gasus_weight_lid = self.gauss_weight(lid_mu, lid_sigma)

        # todo second decision
        f_cam_new_at = cam_sk_feature * (camera_mask + gauss_weight_cam)
        f_lid_new_at = lid_sk_feature * (lidar_mask + gasus_weight_lid)

        # todo consisteance loss
        loss_cons2 = self.sup_layer_loss(f_cam_new, f_lid_new) * self.consistency_loss_weight
        loss_cons1 = self.sup_direct_loss(C2C, L2L)* self.consistency_loss_weight
        loss_consistency = loss_cons1 + loss_cons2
        # todo heatmap_gauss loss
        heatmap = self.get_targets(gt_bboxes_3d, gt_labels_3d, f_cam_new)
        loss_cam_heatmap = self.loss_heatmap(
            self.clip_sigmoid(camera_dense_heat_heat),
            heatmap,
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        ) * self.loss_cam_heatmap_weight
        loss_lid_heatmap = self.loss_heatmap(
            self.clip_sigmoid(lidar_dense_heat_heat),
            heatmap,
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        ) * self.loss_lid_heatmap_weight

        return f_cam_new_at, f_lid_new_at, loss_consistency, loss_cam_heatmap, loss_lid_heatmap, opt

    def sup_direct_loss(self, f_cam, f_lid):
        # todo cam f0[2, 80, 180, 180] lidar f1[2, 256, 180, 180]
        loss_fea = torch.sqrt(torch.pow(f_cam - f_lid, 2).sum())
        sup_loss = loss_fea / (f_cam.size(2) * f_cam.size(3) * f_cam.size(0))
        return sup_loss

    def sup_layer_loss(self, f_cam, f_lid):
        cam_norm = torch.mean(f_cam, dim=2, keepdim=True)
        cam_norm = torch.mean(cam_norm, dim=3, keepdim=True)
        lid_norm = torch.mean(f_lid, dim=2, keepdim=True)
        lid_norm = torch.mean(lid_norm, dim=3, keepdim=True)
        loss_layer = torch.sqrt(torch.pow(cam_norm - lid_norm, 2).sum())
        sup_loss = loss_layer / f_cam.size(0)
        return sup_loss

    def clip_sigmoid(self, x, eps=1e-4):
        y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
        return y

    def gauss_weight(self, mu, sigma, eps=1e-4):
        with torch.no_grad():
            # todo maybe there is zero which cause the issue?
            sigma = torch.clamp(sigma, min=eps,  max=sigma.max())
            mu_new = torch.pow(mu, 2)
            sigma_new = torch.pow(sigma, 2)
            x = torch.div(mu_new, sigma_new)
            y = torch.exp(-x)
            # y = (1/sigma) * torch.exp(-x)
        return y

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, feature):
        res_tuple = multi_apply(
            self.get_targets_single,
            gt_bboxes_3d,
            gt_labels_3d,
            feature,
        )
        heatmap = torch.cat(res_tuple[0], dim=0)
        return heatmap

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, feature):
        device = feature.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
        ).to(device)
        grid_size = torch.tensor(self.grid_size)
        pc_range = torch.tensor(self.pc_range)
        voxel_size = torch.tensor(self.voxel_size)
        feature_map_size = (
            grid_size[:2] // self.out_size_factor
        )  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(
            10, feature_map_size[1], feature_map_size[0]
        )
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.out_size_factor
            length = length / voxel_size[1] / self.out_size_factor
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width), min_overlap=self.gaussian_overlap
                )
                radius = max(self.min_radius, int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (
                    (x - pc_range[0])
                    / voxel_size[0]
                    / self.out_size_factor
                )
                coor_y = (
                    (y - pc_range[1])
                    / voxel_size[1]
                    / self.out_size_factor
                )

                center = torch.tensor(
                    [coor_x, coor_y], dtype=torch.float32, device=device
                )
                center_int = center.to(torch.int32)
                draw_heatmap_gaussian(
                    heatmap[gt_labels_3d[idx]], center_int[[1, 0]], radius
                )
        # todo heatmap[10,180,180]
        heatmap = heatmap.unsqueeze(0)
        # todo heatmap[1, 10,180,180]
        return (heatmap[None])

    def opt_loss(self, opt, gt_bboxes_3d, feature):
        # todo calculate every distance to ground truth
        ref_2d = opt[4] # todo [2, 32400, 1, 2]
        ref_2d = ref_2d.squeeze(2) # todo [2, 32400, 2]
        loss = 0
        # todo batch is necessary
        for i in range (len(gt_bboxes_3d)):
            center = self.opt_single(gt_bboxes_3d[i], feature) # todo [x ,2]
            # todo we calculate the distance, used the a^2+b^2-2ab
            gt_nums = len(gt_bboxes_3d[i])
            pot_nums = len(ref_2d[i,:,0])
            x1 = ref_2d[i,:,0].unsqueeze(1) # todo [32400, 1]
            y1 = ref_2d[i,:,1].unsqueeze(1) # todo [32400, 1]
            x2 = center[:, 0].unsqueeze(0) # todo [1, x]
            y2 = center[:, 1].unsqueeze(0) # todo [1, x]
            x1_copy = x1.repeat(1, gt_nums)
            x2_copy = x2.repeat(pot_nums, 1)
            y1_copy = y1.repeat(1, gt_nums)
            y2_copy = y2.repeat(pot_nums, 1)

            # dis_x = x1_copy - x2_copy
            # dis_y = y1_copy - y2_copy
            # distance = torch.sqrt(torch.pow(dis_x, 2) + torch.pow(dis_y, 2))
            distance = torch.pow(x1_copy, 2) + torch.pow(x2_copy, 2) + \
                       torch.pow(y1_copy, 2) + torch.pow(y2_copy, 2) - \
                       2*x1_copy*x2_copy - 2*y1_copy*y2_copy
            distance = torch.sqrt(distance)

            # todo get the mini distance
            distance, index = torch.min(distance, dim=1) # todo [32400]
            distance = distance / distance.max()

            # todo calculate the loss
            cam_mu = opt[0][i].squeeze(0).view(-1, 1).squeeze(1)
            cam_sigma = opt[1][i].squeeze(0).view(-1, 1).squeeze(1)
            lid_mu = opt[2][i].squeeze(0).view(-1, 1).squeeze(1)
            lid_sigma = opt[3][i].squeeze(0).view(-1, 1).squeeze(1)

            loss_cam = torch.sqrt(torch.pow(cam_mu - distance, 2).sum()) + torch.sqrt(torch.pow(cam_sigma - self.eps, 2).sum())
            loss_lid = torch.sqrt(torch.pow(lid_mu - distance, 2).sum()) + torch.sqrt(torch.pow(lid_sigma - self.eps, 2).sum())

            loss_all = (loss_cam + loss_lid) / (2 * pot_nums)
            loss = loss + loss_all
        return loss

    def opt_single(self, gt_bboxes_3d, feature):
        device = feature.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
        ).to(device)
        pc_range = torch.tensor(self.pc_range)
        voxel_size = torch.tensor(self.voxel_size)
        center_all = gt_bboxes_3d.new_zeros(len(gt_bboxes_3d), 2)
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.out_size_factor
            length = length / voxel_size[1] / self.out_size_factor
            if width > 0 and length > 0:
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]
                coor_x = (
                    (x - pc_range[0])
                    / voxel_size[0]
                    / self.out_size_factor
                )
                coor_y = (
                    (y - pc_range[1])
                    / voxel_size[1]
                    / self.out_size_factor
                )
                center = torch.tensor(
                    [coor_x, coor_y], dtype=torch.float32, device=device
                )
                center_int = center.to(torch.int32)
                center_all[idx, :] = center_int
        return center_all


class MMF11_test(nn.Module):
    def __init__(self):
        super().__init__()

        # todo define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.m_channel = 512
        self.num_layers = 1
        self.bn_momentum = 0.1
        self.grid_size = [1440, 1440, 41]
        self.pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        self.voxel_size = [0.075, 0.075, 0.2]
        self.out_size_factor = 8
        self.gaussian_overlap = 0.1
        self.min_radius = 2

        self.consistency_loss_weight = 0.1
        self.loss_cam_heatmap_weight = 1.0
        self.loss_lid_heatmap_weight = 1.0
        self.mask_loss_weight = 10
        self.eps = 1e-4

        # todo encoding the position
        positional_encoding = dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)

        # todo adapte the channel
        self.cam_conv = build_conv_layer(dict(type="Conv2d"), self.cam_channel, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)
        self.lid_conv = build_conv_layer(dict(type="Conv2d"), self.hidden_channel, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)


        # todo generate the heatmap
        cam_layers = []
        cam_layers.append( ConvModule( self.hidden_channel, self.hidden_channel, kernel_size=3,
                padding=1, bias='auto', conv_cfg=dict(type="Conv2d"),norm_cfg=dict(type="BN2d"),))
        cam_layers.append( build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, self.hidden_channel, kernel_size=3, padding=1,bias='auto',))
        self.cam_heatmap_head = nn.Sequential(*cam_layers)
        # self.cam_heatmap_headhead = build_conv_layer(
        #         dict(type="Conv2d"), self.hidden_channel, 10, kernel_size=3, padding=1,bias='auto',)

        lid_layers = []
        lid_layers.append( ConvModule( self.hidden_channel, self.hidden_channel, kernel_size=3,
                padding=1, bias='auto', conv_cfg=dict(type="Conv2d"),norm_cfg=dict(type="BN2d"),))
        lid_layers.append( build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, self.hidden_channel ,kernel_size=3, padding=1,bias='auto',))
        self.lid_heatmap_head = nn.Sequential(*lid_layers)
        # self.lid_heatmap_headhead = build_conv_layer(
        #         dict(type="Conv2d"), self.hidden_channel, 10, kernel_size=3, padding=1,bias='auto',)

        # todo fusion former layer
        self.C2L = Fusionformerlayer()
        self.C2C = Fusionformerlayer()
        self.L2C = Fusionformerlayer()
        self.L2L = Fusionformerlayer()

        self.mix_feature_camera = build_conv_layer(dict(type="Conv2d"), self.hidden_channel * 2, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)
        self.mix_feature_lidar = build_conv_layer(dict(type="Conv2d"), self.hidden_channel * 2, self.hidden_channel,
            kernel_size=3, padding=1, bias='auto',)

        # todo encode the sigma and mu
        self.mu = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 1, kernel_size=3, padding=1,bias='auto',)
        self.sigma = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 1, kernel_size=3, padding=1,bias='auto',)
        self.mu2 = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 1, kernel_size=3, padding=1,bias='auto',)
        self.sigma2 = build_conv_layer(
                dict(type="Conv2d"), self.hidden_channel, 1, kernel_size=3, padding=1,bias='auto',)


        # todo channel attention
        self.cam_sk = SKConv(256, M=2, G=1, r=2)
        self.lid_sk = SKConv(256, M=2, G=1, r=2)

        # loss_heatmap=dict(type="GaussianFocalLoss", reduction="mean")
        # self.loss_heatmap = build_loss(loss_heatmap)

        self.init_weights()

    def get_reference_points(self, H, W, Z=8,
                             num_points_in_pillar=4, dim='3d', bs=1,
                             device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, feature_cam, feature_lid, gt_bboxes_3d, gt_labels_3d):
        # todo channel traslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)

        # todo get the bev_pos
        bs, embeding_c, H, W = f_cam_new.shape
        ref_2d = self.get_reference_points(
            H, W, dim='2d', bs=f_cam_new.size(0), device=f_cam_new.device, dtype=f_cam_new.dtype)  # todo [2, 32400, 1, 2]
        dtype = f_cam_new.dtype
        bev_mask = torch.zeros((len(f_cam_new), H, W), device=f_cam_new.device).to(dtype) # todo [2, 180, 180]
        bev_pos = self.positional_encoding(bev_mask).to(dtype) # todo [2, 256, 180, 180]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # todo [32400, 2, 256]
        bev_pos = bev_pos.permute(1, 0, 2) # todo [2, 32400, 256]

        # todo heatmap generate
        lidar_dense_heatmap = self.lid_heatmap_head(f_lid_new)
        # lidar_dense_heat_heat = self.lid_heatmap_headhead(lidar_dense_heatmap)
        lidar_heatmap = lidar_dense_heatmap.sigmoid()

        camera_dense_heatmap = self.cam_heatmap_head(f_cam_new)
        # camera_dense_heat_heat = self.cam_heatmap_headhead(camera_dense_heatmap)
        camera_heatmap = camera_dense_heatmap.sigmoid()

        # todo get the spatial attention
        C2L = self.C2L(camera_heatmap, lidar_heatmap , bev_pos, ref_2d)
        L2C = self.L2C(lidar_heatmap , camera_heatmap, bev_pos, ref_2d)
        C2C = self.C2C(camera_heatmap, camera_heatmap, bev_pos, ref_2d)
        L2L = self.L2L(lidar_heatmap , lidar_heatmap , bev_pos, ref_2d)

        # todo get the weightattention
        camera_mask = self.mix_feature_camera((torch.cat((C2C, C2L), dim=1)))
        lidar_mask = self.mix_feature_lidar((torch.cat((L2L, L2C), dim=1)))

        # todo get the channel attention
        cam_sk_feature = self.cam_sk(f_cam_new)
        lid_sk_feature = self.lid_sk(f_lid_new)

        # todo encode the mu and sigma
        cam_mu = self.mu(camera_heatmap)
        cam_sigma = self.sigma(camera_heatmap)
        lid_mu = self.mu2(lidar_heatmap)
        lid_sigma = self.sigma2(lidar_heatmap)
        # opt = [cam_mu, cam_sigma, lid_mu, lid_sigma, ref_2d]
        gauss_weight_cam = self.gauss_weight(cam_mu, cam_sigma)
        gasus_weight_lid = self.gauss_weight(lid_mu, lid_sigma)

        # todo second decision
        f_cam_new_at = cam_sk_feature * (camera_mask + gauss_weight_cam)
        f_lid_new_at = lid_sk_feature * (lidar_mask + gasus_weight_lid)

        # todo consisteance loss
        # loss_consistency = self.sup_direct_loss(f_cam_new, f_lid_new) * self.consistency_loss_weight
        # heatmap = self.get_targets(gt_bboxes_3d, gt_labels_3d, f_cam_new)
        # loss_cam_heatmap = self.loss_heatmap(
        #     self.clip_sigmoid(camera_dense_heat_heat),
        #     heatmap,
        #     avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        # ) * self.loss_cam_heatmap_weight
        # loss_lid_heatmap = self.loss_heatmap(
        #     self.clip_sigmoid(lidar_dense_heat_heat),
        #     heatmap,
        #     avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        # ) * self.loss_lid_heatmap_weight

        return f_cam_new_at, f_lid_new_at

    def sup_direct_loss(self, f_cam, f_lid):
        # todo cam f0[2, 80, 180, 180] lidar f1[2, 256, 180, 180]
        loss_fea = torch.sqrt(torch.pow(f_cam - f_lid, 2).sum())
        sup_loss = loss_fea / (f_cam.size(2) * f_cam.size(3) * f_cam.size(0))
        return sup_loss

    def clip_sigmoid(self, x, eps=1e-4):
        y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
        return y

    def gauss_weight(self, mu, sigma, eps=1e-4):
        with torch.no_grad():
            # todo maybe there is zero which cause the issue?
            sigma = torch.clamp(sigma, min=eps,  max=sigma.max())
            mu_new = torch.pow(mu, 2)
            sigma_new = torch.pow(sigma, 2)
            x = torch.div(mu_new, sigma_new)
            y = torch.exp(-x)
            # y = (1/sigma) * torch.exp(-x)
        return y

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, feature):
        res_tuple = multi_apply(
            self.get_targets_single,
            gt_bboxes_3d,
            gt_labels_3d,
            feature,
        )
        heatmap = torch.cat(res_tuple[0], dim=0)
        return heatmap

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, feature):
        device = feature.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
        ).to(device)
        grid_size = torch.tensor(self.grid_size)
        pc_range = torch.tensor(self.pc_range)
        voxel_size = torch.tensor(self.voxel_size)
        feature_map_size = (
            grid_size[:2] // self.out_size_factor
        )  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(
            10, feature_map_size[1], feature_map_size[0]
        )
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.out_size_factor
            length = length / voxel_size[1] / self.out_size_factor
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width), min_overlap=self.gaussian_overlap
                )
                radius = max(self.min_radius, int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (
                    (x - pc_range[0])
                    / voxel_size[0]
                    / self.out_size_factor
                )
                coor_y = (
                    (y - pc_range[1])
                    / voxel_size[1]
                    / self.out_size_factor
                )

                center = torch.tensor(
                    [coor_x, coor_y], dtype=torch.float32, device=device
                )
                center_int = center.to(torch.int32)
                draw_heatmap_gaussian(
                    heatmap[gt_labels_3d[idx]], center_int[[1, 0]], radius
                )
        # todo heatmap[10,180,180]
        heatmap = heatmap.unsqueeze(0)
        # todo heatmap[1, 10,180,180]
        return (heatmap[None])

    def opt_loss(self, opt, gt_bboxes_3d, feature):
        # todo calculate every distance to ground truth
        ref_2d = opt[4] # todo [2, 32400, 1, 2]
        ref_2d = ref_2d.squeeze(2) # todo [2, 32400, 2]
        loss = 0
        # todo batch is necessary
        for i in range (len(gt_bboxes_3d)):
            center = self.opt_single(gt_bboxes_3d[i], feature) # todo [x ,2]
            # todo we calculate the distance, used the a^2+b^2-2ab
            gt_nums = len(gt_bboxes_3d[i])
            pot_nums = len(ref_2d[i,:,0])
            x1 = ref_2d[i,:,0].unsqueeze(1) # todo [32400, 1]
            y1 = ref_2d[i,:,1].unsqueeze(1) # todo [32400, 1]
            x2 = center[:, 0].unsqueeze(0) # todo [1, x]
            y2 = center[:, 1].unsqueeze(0) # todo [1, x]
            x1_copy = x1.repeat(1, gt_nums)
            x2_copy = x2.repeat(pot_nums, 1)
            y1_copy = y1.repeat(1, gt_nums)
            y2_copy = y2.repeat(pot_nums, 1)

            # dis_x = x1_copy - x2_copy
            # dis_y = y1_copy - y2_copy
            # distance = torch.sqrt(torch.pow(dis_x, 2) + torch.pow(dis_y, 2))
            distance = torch.pow(x1_copy, 2) + torch.pow(x2_copy, 2) + \
                       torch.pow(y1_copy, 2) + torch.pow(y2_copy, 2) - \
                       2*x1_copy*x2_copy - 2*y1_copy*y2_copy
            distance = torch.sqrt(distance)

            # todo get the mini distance
            distance, index = torch.min(distance, dim=1) # todo [32400]
            distance = distance / distance.max()

            # todo calculate the loss
            cam_mu = opt[0][i].squeeze(0).view(-1, 1).squeeze(1)
            cam_sigma = opt[1][i].squeeze(0).view(-1, 1).squeeze(1)
            lid_mu = opt[2][i].squeeze(0).view(-1, 1).squeeze(1)
            lid_sigma = opt[3][i].squeeze(0).view(-1, 1).squeeze(1)

            loss_cam = torch.sqrt(torch.pow(cam_mu - distance, 2).sum()) + torch.sqrt(torch.pow(cam_sigma - self.eps, 2).sum())
            loss_lid = torch.sqrt(torch.pow(lid_mu - distance, 2).sum()) + torch.sqrt(torch.pow(lid_sigma - self.eps, 2).sum())

            loss_all = (loss_cam + loss_lid) / (2 * pot_nums)
            loss = loss + loss_all
        return loss

    def opt_single(self, gt_bboxes_3d, feature):
        device = feature.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
        ).to(device)
        pc_range = torch.tensor(self.pc_range)
        voxel_size = torch.tensor(self.voxel_size)
        center_all = gt_bboxes_3d.new_zeros(len(gt_bboxes_3d), 2)
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.out_size_factor
            length = length / voxel_size[1] / self.out_size_factor
            if width > 0 and length > 0:
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]
                coor_x = (
                    (x - pc_range[0])
                    / voxel_size[0]
                    / self.out_size_factor
                )
                coor_y = (
                    (y - pc_range[1])
                    / voxel_size[1]
                    / self.out_size_factor
                )
                center = torch.tensor(
                    [coor_x, coor_y], dtype=torch.float32, device=device
                )
                center_int = center.to(torch.int32)
                center_all[idx, :] = center_int
        return center_all


@FUSIONMODELS.register_module()
class DeepBEVFusion(Base3DFusionModel):
    def __init__(
            self,
            encoders: Dict[str, Any],
            fuser: Dict[str, Any],
            decoder: Dict[str, Any],
            heads: Dict[str, Any],
            **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        # todo build our Multi-modality fusion module
        # self.MMF = MMF5()
        # self.MMF = MMF6()
        # self.MMF = MMF6_test()
        # self.MMF = MMF7()
        # self.MMF = MMF8()
        self.MMF = MMF11()

        # todo orginal module
        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.grid_size = [1440, 1440, 41]
        self.pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        self.voxel_size = [0.075, 0.075, 0.2]
        self.out_size_factor = 8
        self.gaussian_overlap = 0.1
        self.min_radius = 2
        self.mask_loss_weight = 1
        self.eps = 1e-4

        self.init_weights()

    def freeze_encoders_camera(self):
        self.encoders["camera"].eval()
        for param in self.encoders["camera"].parameters():
            param.requires_grad = False

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
            self,
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)
        # todo fea after fpn
        # todo x=([12, 256, 32, 88]) ([12, 256, 16, 44])

        # todo just get the highest level feature, x = [12, 256, 32, 88]
        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        # todo x = [2, 6, 256, 32, 88]

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        # todo get bev feature x = [2, 80, 180, 180]
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
            self,
            img,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
            gt_masks_bev=None,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            **kwargs,
    ):

        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
            self,
            img,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
            gt_masks_bev=None,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            **kwargs,
    ):
        features = []

        # todo encoder
        for sensor in (
                self.encoders if self.training else list(self.encoders.keys())[::-1]):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        opt = None
        # todo MMF6
        # x, y, loss_consistency, loss_cam_heatmap, loss_lid_heatmap \
        #     = self.MMF(features[0], features[1], gt_bboxes_3d, gt_labels_3d)
        # features_new = []
        # features_new.append(x)
        # features_new.append(y)

        # todo MMF6_test
        # x, y = self.MMF(features[0], features[1])
        # features_new = []
        # features_new.append(x)
        # features_new.append(y)


        # todo MMF11
        x, y, loss_consistency, loss_cam_heatmap, loss_lid_heatmap, opt \
            = self.MMF(features[0], features[1], gt_bboxes_3d, gt_labels_3d)
        features_new = []
        features_new.append(x)
        features_new.append(y)

        if self.training:
            mask_loss = self.opt_loss(opt, gt_bboxes_3d, x) * self.mask_loss_weight

        # todo fuse
        if self.fuser is not None:
            x = self.fuser(features_new) # todo fusion shape torch.Size([2, 256, 180, 180])
        else:
            assert len(features_new) == 1, features_new
            x = features_new[0]

        # todo vis
        # self.vis(metas, features[0], features[1], points, gt_bboxes_3d, gt_labels_3d,
        #          x, f_com_cam, f_com_lid, f_unq_cam,  f_unq_lid, MCL, MLC, MC, ML)

        # todo decode the com_fea
        batch_size = x.shape[0]
        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)  # todo x=[1, [2, 512, 180, 180]]

        # todo head
        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    # pred_dict, query_labels_all = head(x, metas, y)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                    losses.update(dict(loss_consistence=loss_consistency))
                    losses.update(dict(loss_heatmap_cam=loss_cam_heatmap))
                    losses.update(dict(loss_heatmap_lid=loss_lid_heatmap))
                    losses.update(dict(loss_mask=mask_loss))
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    # pred_dict, query_labels_all = head(x, metas, y)
                    # bboxes = head.get_bboxes(pred_dict, metas, query_labels_all)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs


    # todo some vis tools
    def vis(self, metas, cam_fea, lid_fea, points, gt_bboxes_3d, gt_labels_3d,
                 x, f_com_cam, f_com_lid, f_unq_cam,  f_unq_lid, MCL, MLC, MC, ML):
        camera_path = "/root/paddlejob/workspace/yyr/code/bevfusion-main/mask/camera/"
        self.vis_feature(metas, cam_fea, camera_path, '/camera_fea.jpg')
        self.vis_feature(metas, lid_fea, camera_path, '/lidar_fea.jpg')
        self.vis_feature(metas, x, camera_path, '/fusion.jpg')
        self.vis_bev(metas, points, gt_bboxes_3d, gt_labels_3d, camera_path)
        self.vis_feature(metas, f_com_cam, camera_path, '/com_cam_fea.jpg')
        self.vis_feature(metas, f_com_lid, camera_path, '/com_lid_fea.jpg')
        self.vis_feature(metas, f_unq_cam, camera_path, '/unq_com_fea.jpg')
        self.vis_feature(metas, f_unq_lid, camera_path, '/unq_lid_fea.jpg')
        self.vis_feature(metas, MCL, camera_path, '/mask_com_CL.jpg')
        self.vis_feature(metas, MLC, camera_path, '/mask_com_LC.jpg')
        self.vis_feature(metas, MC, camera_path, '/mask_mc.jpg')
        self.vis_feature(metas, ML, camera_path, '/mask_ml.jpg')
        exit()

    def vis_feature(self, img_metas, heatmaps, path, name):
        for i, img_meta in enumerate(img_metas):
            filename = img_meta['filename'][0]
            path_pic = path + filename.split("/")[-1].split(".")[0].split("__")[0]
            self.mkdir(path_pic)
            plt.xticks(())
            plt.yticks(())
            heatmap = torch.sum(heatmaps[i], dim=0).cpu().clone().detach().numpy()
            plt.imshow(heatmap, interpolation='bilinear', cmap='rainbow')
            plt.xticks(())
            plt.yticks(())
            pic_name = path_pic + name
            plt.savefig(pic_name, bbox_inches='tight', pad_inches=-0.1)
            plt.close("all")

    def vis_bev(self, img_meta, points, gt_bboxes_3d, gt_labels_3d, path):
        for i, img_meta in enumerate(img_meta):
            bboxes = gt_bboxes_3d[i].tensor.numpy()
            labels = gt_labels_3d[i].cpu().clone().detach().numpy()

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
            lidar = points[i].cpu().clone().detach().numpy()
            point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
            object_classes = ['car', 'truck', 'construction_vehicle',
                              'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

            filename = img_meta['filename'][0]
            path_pic = path + filename.split("/")[-1].split(".")[0].split("__")[0]
            pic_name = path_pic + "/bev.jpg"
            visualize_lidar(
                pic_name,
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[point_cloud_range[d] for d in [0, 3]],
                ylim=[point_cloud_range[d] for d in [1, 4]],
                classes=object_classes,
            )

    def mkdir(self, path):
        isexists = os.path.exists(path)
        if not isexists:
            os.makedirs(path)
            print("file makes successfully")
        else:
            print("file is exist")


    def opt_loss(self, opt, gt_bboxes_3d, feature):
        # todo calculate every distance to ground truth
        ref_2d = opt[4] # todo [2, 32400, 1, 2]
        ref_2d = ref_2d.squeeze(2) # todo [2, 32400, 2]
        loss = 0
        # todo batch is necessary
        for i in range (len(gt_bboxes_3d)):
            center = self.opt_single(gt_bboxes_3d[i], feature) # todo [x ,2]
            # todo we calculate the distance, used the a^2+b^2-2ab
            gt_nums = len(gt_bboxes_3d[i])
            pot_nums = len(ref_2d[i,:,0])
            x1 = ref_2d[i,:,0].unsqueeze(1) # todo [32400, 1]
            y1 = ref_2d[i,:,1].unsqueeze(1) # todo [32400, 1]
            x2 = center[:, 0].unsqueeze(0) # todo [1, x]
            y2 = center[:, 1].unsqueeze(0) # todo [1, x]
            x1_copy = x1.repeat(1, gt_nums)
            x2_copy = x2.repeat(pot_nums, 1)
            y1_copy = y1.repeat(1, gt_nums)
            y2_copy = y2.repeat(pot_nums, 1)

            # dis_x = x1_copy - x2_copy
            # dis_y = y1_copy - y2_copy
            # distance = torch.sqrt(torch.pow(dis_x, 2) + torch.pow(dis_y, 2))
            distance = torch.pow(x1_copy, 2) + torch.pow(x2_copy, 2) + \
                       torch.pow(y1_copy, 2) + torch.pow(y2_copy, 2) - \
                       2*x1_copy*x2_copy - 2*y1_copy*y2_copy
            distance = torch.sqrt(distance)

            # todo get the mini distance
            distance, index = torch.min(distance, dim=1) # todo [32400]
            distance = distance / distance.max()

            # todo calculate the loss
            cam_mu = opt[0][i].squeeze(0).view(-1, 1).squeeze(1)
            cam_sigma = opt[1][i].squeeze(0).view(-1, 1).squeeze(1)
            lid_mu = opt[2][i].squeeze(0).view(-1, 1).squeeze(1)
            lid_sigma = opt[3][i].squeeze(0).view(-1, 1).squeeze(1)

            loss_cam = torch.sqrt(torch.pow(cam_mu - distance, 2).sum()) + torch.sqrt(torch.pow(cam_sigma - self.eps, 2).sum())
            loss_lid = torch.sqrt(torch.pow(lid_mu - distance, 2).sum()) + torch.sqrt(torch.pow(lid_sigma - self.eps, 2).sum())

            loss_all = (loss_cam + loss_lid) / (2 * pot_nums)
            loss = loss + loss_all
        return loss

    def opt_single(self, gt_bboxes_3d, feature):
        device = feature.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
        ).to(device)
        pc_range = torch.tensor(self.pc_range)
        voxel_size = torch.tensor(self.voxel_size)
        center_all = gt_bboxes_3d.new_zeros(len(gt_bboxes_3d), 2)
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.out_size_factor
            length = length / voxel_size[1] / self.out_size_factor
            if width > 0 and length > 0:
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]
                coor_x = (
                    (x - pc_range[0])
                    / voxel_size[0]
                    / self.out_size_factor
                )
                coor_y = (
                    (y - pc_range[1])
                    / voxel_size[1]
                    / self.out_size_factor
                )
                center = torch.tensor(
                    [coor_x, coor_y], dtype=torch.float32, device=device
                )
                center_int = center.to(torch.int32)
                center_all[idx, :] = center_int
        return center_all