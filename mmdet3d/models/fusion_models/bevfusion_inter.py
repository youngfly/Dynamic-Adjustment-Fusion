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

from torch.autograd import Function
from localAttention import (similar_forward,
                            similar_backward,
                            weighting_forward,
                            weighting_backward_ori,
                            weighting_backward_weight)
from .deep_inter import DeepInteractionEncoder

from collections import OrderedDict  # 导入此模块

__all__ = ["BEVFusion_inter"]


# todo grid mask
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        # todo Probability > defalut, return original  image
        if random.uniform(0, 1) > self.probability:
            return img

        # todo
        for attempt in range(100):
            # todo get the area
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


# todo global attention
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


# todo cross attention
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


# todo mask two path, x1, x2
class Interact(nn.Module):
    def __init__(self):
        super().__init__()

        # todo channel define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.out_channel = 1
        self.m_channel = 512

        self.cam_conv = ConvBNReLU(self.cam_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)
        self.lid_conv = ConvBNReLU(self.hidden_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)

        # todo query feature
        self.cam_cam = CrossAttention(self.hidden_channel)
        self.lid_lid = CrossAttention(self.hidden_channel)

        self.cam_query = CrossAttention(self.hidden_channel)
        self.lid_query = CrossAttention(self.hidden_channel)

        # todo mix the camera mask
        self.mix_feature_cam = ConvBNReLU(self.m_channel, self.cam_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)
        # todo mix lidar mask
        self.mix_feature_lid = ConvBNReLU(self.m_channel, self.hidden_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)

    def forward(self, feature_cam, feature_lid):
        # todo channel chanslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)
        # todo cross attention and self attention
        f_cam_cam = self.cam_cam(f_cam_new, f_cam_new)
        f_lid_lid = self.lid_lid(f_lid_new, f_lid_new)
        f_camq = self.cam_query(f_cam_new, f_lid_new)
        f_lidq = self.lid_query(f_lid_new, f_cam_new)

        # todo mix the single mask
        mix_camera = self.mix_feature_cam((torch.cat((f_cam_cam, f_camq), dim=1)))
        mix_lidar = self.mix_feature_lid((torch.cat((f_lid_lid, f_lidq), dim=1)))

        # todo norm by sigmoid
        mix_camera = F.sigmoid(mix_camera)
        mix_lidar = F.sigmoid(mix_lidar)
        # todo attention
        feature_new = []
        feature_cam_new = torch.zeros_like(feature_cam)
        feature_lidar_new = torch.zeros_like(feature_lid)
        for i in range(feature_cam.size(0)):
            feature_cam_new[i] = feature_cam[i] * mix_camera[i]
            feature_lidar_new[i] = feature_lid[i] * mix_lidar[i]
        feature_new.append(feature_cam_new)
        feature_new.append(feature_lidar_new)

        return feature_new, mix_camera, mix_lidar


# todo mask four path, x1, x2
class Interact_2(nn.Module):
    def __init__(self):
        super().__init__()

        # todo channel define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.mix_channel = 1024
        self.out_channel = 1
        self.m_channel = 512

        self.cam_conv = ConvBNReLU(self.cam_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)
        self.lid_conv = ConvBNReLU(self.hidden_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)

        # todo query feature
        self.cam_cam = CrossAttention(self.hidden_channel)
        self.lid_lid = CrossAttention(self.hidden_channel)

        self.cam_query = CrossAttention(self.hidden_channel)
        self.lid_query = CrossAttention(self.hidden_channel)

        # todo mix feature
        self.mix_feature_cam = ConvBNReLU(self.mix_channel, self.cam_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)
        self.mix_feature_lid = ConvBNReLU(self.mix_channel, self.hidden_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)

    def forward(self, feature_cam, feature_lid):
        # todo channel chanslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)
        # todo cross attention and self attention
        f_cam_cam = self.cam_cam(f_cam_new, f_cam_new)
        f_lid_lid = self.lid_lid(f_lid_new, f_lid_new)
        f_camq = self.cam_query(f_cam_new, f_lid_new)
        f_lidq = self.lid_query(f_lid_new, f_cam_new)

        # todo get the single mask
        mix_camera = self.mix_feature_cam((torch.cat((f_camq, f_lidq, f_cam_cam, f_lid_lid), dim=1)))
        mix_lidar = self.mix_feature_lid((torch.cat((f_camq, f_lidq, f_cam_cam, f_lid_lid), dim=1)))

        # todo norm by sigmoid
        mix_camera = F.sigmoid(mix_camera)
        mix_lidar = F.sigmoid(mix_lidar)

        # todo attention
        feature_new = []
        feature_cam_new = torch.zeros_like(feature_cam)
        feature_lidar_new = torch.zeros_like(feature_lid)
        for i in range(feature_cam.size(0)):
            feature_cam_new[i] = feature_cam[i] * mix_camera[i]
            feature_lidar_new[i] = feature_lid[i] * mix_lidar[i]
        feature_new.append(feature_cam_new)
        feature_new.append(feature_lidar_new)

        return feature_new, mix_camera, mix_lidar


# todo mask four path, x, 1-x
class Interact_3(nn.Module):
    def __init__(self):
        super().__init__()

        # todo channel define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.mix_channel = 1024
        self.out_channel = 1
        self.m_channel = 512

        self.cam_conv = ConvBNReLU(self.cam_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)
        self.lid_conv = ConvBNReLU(self.hidden_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)

        # todo query feature
        self.cam_cam = CrossAttention(self.hidden_channel)
        self.lid_lid = CrossAttention(self.hidden_channel)

        self.cam_query = CrossAttention(self.hidden_channel)
        self.lid_query = CrossAttention(self.hidden_channel)

        # todo mix feature
        self.mix_feature_all = ConvBNReLU(self.mix_channel, self.out_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)

    def forward(self, feature_cam, feature_lid):
        # todo channel chanslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)
        # todo cross attention and self attention
        f_cam_cam = self.cam_cam(f_cam_new, f_cam_new)
        f_lid_lid = self.lid_lid(f_lid_new, f_lid_new)
        f_camq = self.cam_query(f_cam_new, f_lid_new)
        f_lidq = self.lid_query(f_lid_new, f_cam_new)

        # todo get the single mask
        mix_fea_all = self.mix_feature_all((torch.cat((f_camq, f_lidq, f_cam_cam, f_lid_lid), dim=1)))

        # todo norm by sigmoid
        mix_fea_all = F.sigmoid(mix_fea_all)

        # todo attention
        feature_new = []
        feature_cam_new = torch.zeros_like(feature_cam)
        feature_lidar_new = torch.zeros_like(feature_lid)
        for i in range(feature_cam.size(0)):
            feature_cam_new[i] = feature_cam[i] * mix_fea_all[i]
            feature_lidar_new[i] = feature_lid[i] * (1 - mix_fea_all[i])
        feature_new.append(feature_cam_new)
        feature_new.append(feature_lidar_new)

        return feature_new, mix_fea_all, 1 - mix_fea_all


# todo mask four path, x, 1-x no self-attention
class Interact_4(nn.Module):
    def __init__(self):
        super().__init__()

        # todo channel define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.mix_channel = 1024
        self.out_channel = 1
        self.m_channel = 512

        self.cam_conv = ConvBNReLU(self.cam_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)
        self.lid_conv = ConvBNReLU(self.hidden_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)

        # todo query feature

        self.cam_query = CrossAttention(self.hidden_channel)
        self.lid_query = CrossAttention(self.hidden_channel)

        # todo mix feature
        self.mix_feature_all = ConvBNReLU(self.mix_channel, self.out_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)

    def forward(self, feature_cam, feature_lid):
        # todo channel chanslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)
        # todo cross attention and self attention
        f_camq = self.cam_query(f_cam_new, f_lid_new)
        f_lidq = self.lid_query(f_lid_new, f_cam_new)

        # todo get the single mask
        mix_fea_all = self.mix_feature_all((torch.cat((f_camq, f_lidq, f_cam_new, f_lid_new), dim=1)))

        # todo norm by sigmoid
        mix_fea_all = F.sigmoid(mix_fea_all)

        # todo attention
        feature_new = []
        feature_cam_new = torch.zeros_like(feature_cam)
        feature_lidar_new = torch.zeros_like(feature_lid)
        for i in range(feature_cam.size(0)):
            feature_cam_new[i] = feature_cam[i] * mix_fea_all[i]
            feature_lidar_new[i] = feature_lid[i] * (1 - mix_fea_all[i])
        feature_new.append(feature_cam_new)
        feature_new.append(feature_lidar_new)

        return feature_new, mix_fea_all, 1 - mix_fea_all


# todo new attention  **********************************
# todo CBR
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


# todo context local attention
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


# todo big
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


# todo mini
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
        # weight = F.softmax(weight, -1)
        weight = nn.functional.softmax(weight / math.sqrt(key.size(1)), -1)
        out = self.f_weighting(value, weight, self.kernel_size, self.kernel_size)

        return out


# todo local mask four path, x, 1-x without self-attention
class Inter_local_1(nn.Module):
    def __init__(self):
        super().__init__()

        # todo channel define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.mix_channel = 1024
        self.out_channel = 1
        self.m_channel = 512
        self.context_kernel = 9

        self.cam_conv = ConvBNReLU(self.cam_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)
        self.lid_conv = ConvBNReLU(self.hidden_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)

        # todo query feature
        self.cam_query = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)
        self.lid_query = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)

        # todo mix feature
        self.mix_feature_all = ConvBNReLU(self.mix_channel, self.out_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)

    def forward(self, feature_cam, feature_lid):
        # todo channel chanslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)
        # todo cross attention
        f_camq = self.cam_query(f_cam_new, f_lid_new)
        f_lidq = self.lid_query(f_lid_new, f_cam_new)

        # todo eliminate the background
        f_camq = f_camq * f_cam_new
        f_lidq = f_lidq * f_lid_new

        # todo get the single mask
        mix_fea_all = self.mix_feature_all((torch.cat((f_camq, f_lidq, f_cam_new, f_lid_new), dim=1)))

        # todo norm by sigmoid
        mix_fea_all = F.sigmoid(mix_fea_all)

        # todo attention
        feature_new = []
        feature_cam_new = torch.zeros_like(feature_cam)
        feature_lidar_new = torch.zeros_like(feature_lid)
        for i in range(feature_cam.size(0)):
            feature_cam_new[i] = feature_cam[i] * mix_fea_all[i]
            feature_lidar_new[i] = feature_lid[i] * (1 - mix_fea_all[i])
        feature_new.append(feature_cam_new)
        feature_new.append(feature_lidar_new)

        return feature_new, mix_fea_all, 1 - mix_fea_all


# todo local mask four path, x, 1-x with self-attention
class Inter_local_2(nn.Module):
    def __init__(self):
        super().__init__()

        # todo channel define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.mix_channel = 1024
        self.out_channel = 1
        self.m_channel = 512
        self.context_kernel = 9

        self.cam_conv = ConvBNReLU(self.cam_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)
        self.lid_conv = ConvBNReLU(self.hidden_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)

        # todo query feature
        self.cam_query = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)
        self.lid_query = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)
        self.cam_cam = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)
        self.lid_lid = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)


        # todo mix feature
        self.mix_feature_all = ConvBNReLU(self.mix_channel, self.out_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)

    def forward(self, feature_cam, feature_lid):
        # todo channel chanslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)
        # todo cross attention and self attention
        f_camq = self.cam_query(f_cam_new, f_lid_new)
        f_lidq = self.lid_query(f_lid_new, f_cam_new)
        f_cc =  self.cam_cam(f_cam_new, f_cam_new)
        f_ll =  self.lid_lid(f_lid_new, f_lid_new)

        # todo get the single mask
        mix_fea_all = self.mix_feature_all((torch.cat((f_camq, f_lidq, f_cc, f_ll), dim=1)))

        # todo norm by sigmoid
        mix_fea_all = F.sigmoid(mix_fea_all)

        # todo attention
        feature_new = []
        feature_cam_new = torch.zeros_like(feature_cam)
        feature_lidar_new = torch.zeros_like(feature_lid)
        for i in range(feature_cam.size(0)):
            feature_cam_new[i] = feature_cam[i] * mix_fea_all[i]
            feature_lidar_new[i] = feature_lid[i] * (1 - mix_fea_all[i])
        feature_new.append(feature_cam_new)
        feature_new.append(feature_lidar_new)

        return feature_new, mix_fea_all, 1 - mix_fea_all


# todo local mask four path x1, x2 without self-attention
class Inter_local_3(nn.Module):
    def __init__(self):
        super().__init__()

        # todo channel define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.mix_channel = 1024
        self.out_channel = 1
        self.m_channel = 512
        self.context_kernel = 9

        # todo translate the channel
        self.cam_conv = ConvBNReLU(self.cam_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)
        self.lid_conv = ConvBNReLU(self.hidden_channel, self.hidden_channel,
                                   kernel_size=1, norm_layer=nn.BatchNorm2d,
                                   activation_layer=nn.ReLU, bias='auto',
                                   inplace=True, affine=True)

        # todo cross attention
        self.cam_query = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)
        self.lid_query = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)

        # todo mix feature
        self.mix_feature_cam = ConvBNReLU(self.mix_channel, self.out_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)
        self.mix_feature_lid = ConvBNReLU(self.mix_channel, self.out_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)

    def forward(self, feature_cam, feature_lid):
        # todo channel chanslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)
        # todo cross attention
        f_camq = self.cam_query(f_cam_new, f_lid_new)
        f_lidq = self.lid_query(f_lid_new, f_cam_new)

        # todo eliminate the background
        f_camq = f_camq * f_cam_new
        f_lidq = f_lidq * f_lid_new


        # todo get the single mask
        mix_fea_camera = self.mix_feature_cam((torch.cat((f_camq, f_lidq, f_cam_new , f_lid_new), dim=1)))
        mix_fea_lidar = self.mix_feature_lid((torch.cat((f_camq, f_lidq, f_cam_new, f_lid_new), dim=1)))

        # todo norm by sigmoid
        mix_camera = F.sigmoid(mix_fea_camera)
        mix_lidar = F.sigmoid(mix_fea_lidar)

        # todo generate the mask to eliminate the background

        # todo attention
        feature_new = []
        feature_cam_new = torch.zeros_like(feature_cam)
        feature_lidar_new = torch.zeros_like(feature_lid)
        for i in range(feature_cam.size(0)):
            feature_cam_new[i] = feature_cam[i] * mix_camera[i]
            feature_lidar_new[i] = feature_lid[i] * mix_lidar[i]
        feature_new.append(feature_cam_new)
        feature_new.append(feature_lidar_new)

        return feature_new, mix_camera, mix_lidar, f_camq, f_lidq


# todo local mask four path x1, x2 with self-attention
class Inter_local_4(nn.Module):
    def __init__(self):
        super().__init__()

        # todo channel define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.mix_channel = 1024
        self.out_channel = 1
        self.m_channel = 512
        self.context_kernel = 9

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
        self.cam_query = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)
        self.lid_query = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)
        self.cam_cam = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)
        self.lid_lid = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)

        # todo mix feature
        self.mix_feature_cam = ConvBNReLU(self.mix_channel, self.out_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)
        self.mix_feature_lid = ConvBNReLU(self.mix_channel, self.out_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)

    def forward(self, feature_cam, feature_lid):
        # todo channel chanslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)
        # todo cross attention
        f_camq = self.cam_query(f_cam_new, f_lid_new)
        f_lidq = self.lid_query(f_lid_new, f_cam_new)
        f_cc =  self.cam_cam(f_cam_new, f_cam_new)
        f_ll =  self.lid_lid(f_lid_new, f_lid_new)

        # todo eliminate the background
        f_camq = f_camq * f_cam_new
        f_lidq = f_lidq * f_lid_new
        f_cc = f_cc * f_cam_new
        f_ll = f_ll * f_lid_new

        # todo get the single mask
        mix_fea_camera = self.mix_feature_cam((torch.cat((f_camq, f_lidq, f_cc, f_ll), dim=1)))
        mix_fea_lidar = self.mix_feature_lid((torch.cat((f_camq, f_lidq, f_cc, f_ll), dim=1)))

        # todo norm by sigmoid
        mix_camera = F.sigmoid(mix_fea_camera)
        mix_lidar = F.sigmoid(mix_fea_lidar)

        # todo generate the mask to eliminate the background

        # todo attention
        feature_new = []
        feature_cam_new = torch.zeros_like(feature_cam)
        feature_lidar_new = torch.zeros_like(feature_lid)
        for i in range(feature_cam.size(0)):
            feature_cam_new[i] = feature_cam[i] * mix_camera[i]
            feature_lidar_new[i] = feature_lid[i] * mix_lidar[i]
        feature_new.append(feature_cam_new)
        feature_new.append(feature_lidar_new)

        return feature_new, mix_camera, mix_lidar


# todo local mask two path x1, x2  cross-attention self
class Inter_local_5(nn.Module):
    def __init__(self):
        super().__init__()

        # todo channel define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.mix_channel = 1024
        self.out_channel = 1
        self.m_channel = 512
        self.context_kernel = 9

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
        self.cam_query = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)
        self.lid_query = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)

        # todo mix feature
        self.mix_feature_cam = ConvBNReLU(self.m_channel, self.out_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)
        self.mix_feature_lid = ConvBNReLU(self.m_channel, self.out_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)

    def forward(self, feature_cam, feature_lid):
        # todo channel chanslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)
        # todo cross attention
        f_camq = self.cam_query(f_cam_new, f_lid_new)
        f_lidq = self.lid_query(f_lid_new, f_cam_new)

        # todo eliminate the background
        f_camq = f_camq * f_cam_new
        f_lidq = f_lidq * f_lid_new

        # todo get the single mask
        mix_fea_camera = self.mix_feature_cam((torch.cat((f_camq, f_cam_new), dim=1)))
        mix_fea_lidar = self.mix_feature_lid((torch.cat((f_lidq, f_lid_new), dim=1)))

        # todo norm by sigmoid
        mix_camera = F.sigmoid(mix_fea_camera)
        mix_lidar = F.sigmoid(mix_fea_lidar)

        # todo attention
        feature_new = []
        feature_cam_new = torch.zeros_like(feature_cam)
        feature_lidar_new = torch.zeros_like(feature_lid)
        for i in range(feature_cam.size(0)):
            feature_cam_new[i] = feature_cam[i] * mix_camera[i]
            feature_lidar_new[i] = feature_lid[i] * mix_lidar[i]
        feature_new.append(feature_cam_new)
        feature_new.append(feature_lidar_new)

        return feature_new, mix_camera, mix_lidar


# todo lcoal mask two path x1, x2 cross-attention self-attention
class Inter_local_6(nn.Module):
    def __init__(self):
        super().__init__()

        # todo channel define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.mix_channel = 1024
        self.out_channel = 1
        self.m_channel = 512
        self.context_kernel = 9

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
        self.cam_query = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)
        self.lid_query = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)
        self.cam_cam = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)
        self.lid_lid = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)

        # todo mix feature
        self.mix_feature_cam = ConvBNReLU(self.m_channel, self.out_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)
        self.mix_feature_lid = ConvBNReLU(self.m_channel, self.out_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)

    def forward(self, feature_cam, feature_lid):
        # todo channel chanslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)
        # todo cross attention
        f_camq = self.cam_query(f_cam_new, f_lid_new)
        f_lidq = self.lid_query(f_lid_new, f_cam_new)
        f_cc =  self.cam_cam(f_cam_new, f_cam_new)
        f_ll =  self.lid_lid(f_lid_new, f_lid_new)

        # todo eliminate the background
        f_camq = f_camq * f_cam_new
        f_lidq = f_lidq * f_lid_new
        f_cc = f_cc * f_cam_new
        f_ll = f_ll * f_lid_new

        # todo get the single mask
        mix_fea_camera = self.mix_feature_cam((torch.cat((f_camq, f_cc), dim=1)))
        mix_fea_lidar = self.mix_feature_lid((torch.cat((f_lidq, f_ll), dim=1)))

        # todo norm by sigmoid
        mix_camera = F.sigmoid(mix_fea_camera)
        mix_lidar = F.sigmoid(mix_fea_lidar)

        # todo generate the mask to eliminate the background

        # todo attention
        feature_new = []
        feature_cam_new = torch.zeros_like(feature_cam)
        feature_lidar_new = torch.zeros_like(feature_lid)
        for i in range(feature_cam.size(0)):
            feature_cam_new[i] = feature_cam[i] * mix_camera[i]
            feature_lidar_new[i] = feature_lid[i] * mix_lidar[i]
        feature_new.append(feature_cam_new)
        feature_new.append(feature_lidar_new)

        return feature_new, mix_camera, mix_lidar


# todo lcoal mask two path x1, x2 cross-attention self-attention self
class Inter_local_7(nn.Module):
    def __init__(self):
        super().__init__()

        # todo channel define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.mix_channel = 1024
        self.out_channel = 1
        self.m_channel = 512
        self.all_channel = 768
        self.context_kernel = 9

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
        self.cam_query = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)
        self.lid_query = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)
        self.cam_cam = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)
        self.lid_lid = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)

        # todo mix feature
        self.mix_feature_cam = ConvBNReLU(self.all_channel, self.out_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)
        self.mix_feature_lid = ConvBNReLU(self.all_channel, self.out_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)

    def forward(self, feature_cam, feature_lid):
        # todo channel chanslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)
        # todo cross attention
        f_camq = self.cam_query(f_cam_new, f_lid_new)
        f_lidq = self.lid_query(f_lid_new, f_cam_new)
        f_cc =  self.cam_cam(f_cam_new, f_cam_new)
        f_ll =  self.lid_lid(f_lid_new, f_lid_new)

        # todo eliminate the background
        f_camq = f_camq * f_cam_new
        f_lidq = f_lidq * f_lid_new
        f_cc = f_cc * f_cam_new
        f_ll = f_ll * f_lid_new

        # todo get the single mask
        mix_fea_camera = self.mix_feature_cam((torch.cat((f_camq, f_cc, f_cam_new), dim=1)))
        mix_fea_lidar = self.mix_feature_lid((torch.cat((f_lidq, f_ll, f_lid_new), dim=1)))

        # todo norm by sigmoid
        mix_camera = F.sigmoid(mix_fea_camera)
        mix_lidar = F.sigmoid(mix_fea_lidar)

        # todo attention
        feature_new = []
        feature_cam_new = torch.zeros_like(feature_cam)
        feature_lidar_new = torch.zeros_like(feature_lid)
        for i in range(feature_cam.size(0)):
            feature_cam_new[i] = feature_cam[i] * mix_camera[i]
            feature_lidar_new[i] = feature_lid[i] * mix_lidar[i]
        feature_new.append(feature_cam_new)
        feature_new.append(feature_lidar_new)

        return feature_new, mix_camera, mix_lidar


# todo new local mask
class Inter_1(nn.Module):
    def __init__(self):
        super().__init__()

        # todo channel define
        self.hidden_channel = 256
        self.cam_channel = 80
        self.mix_channel = 1024
        self.out_channel = 1
        self.m_channel = 512
        self.all_channel = 768
        self.context_kernel = 9

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
        self.cam_query = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)
        self.lid_query = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)
        self.cam_cam = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)
        self.lid_lid = LocalContextAttentionBlock(self.hidden_channel,
                                                    self.hidden_channel,
                                                    self.context_kernel,
                                                    last_affine=True)

        # todo mix feature
        self.mix_feature_cam = ConvBNReLU(self.all_channel, self.out_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)
        self.mix_feature_lid = ConvBNReLU(self.all_channel, self.out_channel,
                                          kernel_size=1, norm_layer=nn.BatchNorm2d,
                                          activation_layer=None)

    def forward(self, feature_cam, feature_lid):
        # todo channel chanslate
        f_cam_new = self.cam_conv(feature_cam)
        f_lid_new = self.lid_conv(feature_lid)

        # todo cross attention
        f_camq = self.cam_query(f_cam_new, f_lid_new)
        f_lidq = self.lid_query(f_lid_new, f_cam_new)
        f_cc =  self.cam_cam(f_cam_new, f_cam_new)
        f_ll =  self.lid_lid(f_lid_new, f_lid_new)

        # todo eliminate the background
        f_camq = f_camq * f_cam_new
        f_lidq = f_lidq * f_lid_new
        f_cc = f_cc * f_cam_new
        f_ll = f_ll * f_lid_new

        # todo get the single mask
        mix_fea_camera = self.mix_feature_cam((torch.cat((f_camq, f_cc, f_cam_new), dim=1)))
        mix_fea_lidar = self.mix_feature_lid((torch.cat((f_lidq, f_ll, f_lid_new), dim=1)))

        # todo norm by sigmoid
        mix_camera = F.sigmoid(mix_fea_camera)
        mix_lidar = F.sigmoid(mix_fea_lidar)

        # todo attention
        feature_new = []
        feature_cam_new = torch.zeros_like(feature_cam)
        feature_lidar_new = torch.zeros_like(feature_lid)
        for i in range(feature_cam.size(0)):
            feature_cam_new[i] = feature_cam[i] * mix_camera[i]
            feature_lidar_new[i] = feature_lid[i] * mix_lidar[i]
        feature_new.append(feature_cam_new)
        feature_new.append(feature_lidar_new)

        return feature_new, mix_camera, mix_lidar


@FUSIONMODELS.register_module()
class BEVFusion_inter(Base3DFusionModel):
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

        self.init_weights()
        self.num_proposals = 200
        # self.freeze_encoders_camera()

        # todo global inter module
        # self.mask_generator = Interact()
        # self.mask_generator = Interact_2()
        # self.mask_generator = Interact_3()
        # self.mask_generator = Interact_4()

        # todo local inter nodule four path
        # self.mask_generator = Inter_local_1()
        # self.mask_generator = Inter_local_2()
        # self.mask_generator = Inter_local_3()
        # self.mask_generator = Inter_local_4()

        # todo local inter nodule two path
        # self.mask_generator = Inter_local_5()
        # self.mask_generator = Inter_local_6()
        # self.mask_generator = Inter_local_7()

        # todo init the deep interaction
        # self.inter_feature = DeepInteractionEncoder()

        # todo get a single high level bev feature
        # self.high_level_cam = ConvBNReLU(192, 256,
        #                            kernel_size=1, norm_layer=nn.BatchNorm2d,
        #                            activation_layer=nn.ReLU, bias='auto',
        #                            inplace=True, affine=True)

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

        # todo mask-modal-training
        # img, points = self.mask_modal_trainig(img, points)

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

        if self.fuser is not None:
            x = self.fuser(features) # todo fusion shape torch.Size([2, 256, 180, 180])
        else:
            assert len(features) == 1, features
            x = features[0]

        # todo fuse the feature
        # todo camera f0[2, 80, 180, 180] lidar f1[2, 256, 180, 180]
        # todo last two is the grid size, first is batchszie, second is channel
        # todo interact two_single_path mask

        # todo get the top k quires of the camera and lidar
        camera_top = self.get_the_local_max(features[0])
        lidar_top = self.get_the_local_max(features[1])

        # todo fusion the mask
        # features_new, mask_camera, mask_lidar = self.mask_generator(features[0], features[1])
        # if self.fuser is not None:
        #     x = self.fuser(features_new)
        # else:
        #     assert len(features_new) == 1, features_new
        #     x = features_new[0]

        # todo deepinteraction
        # cam_feature, lidar_feature = self.inter_feature(features[0], features[1])
        # features_new = []
        # features_new.append(cam_feature)
        # features_new.append(lidar_feature)
        # if self.fuser is not None:
        #     x = self.fuser(features_new)
        # else:
        #     assert len(features_new) == 1, features_new
        #     x = features_new[0]

        # todo vis the mask
        # camera_path = "/root/paddlejob/workspace/yyr/code/bevfusion-main/mask/camera/"
        # self.vis_feature(metas, features[0], camera_path, '/camera_fea.jpg')
        # self.vis_feature(metas, features[1], camera_path, '/lidar_fea.jpg')
        # self.vis_feature(metas, mask_camera, camera_path, '/mix_cam_mask.jpg')
        # self.vis_feature(metas, mask_lidar, camera_path, '/mix_lid_mask.jpg')
        # self.vis_feature(metas, x, camera_path, '/fusion.jpg')
        # self.vis_bev(metas, points, gt_bboxes_3d, gt_labels_3d, camera_path)
        # self.vis_feature(metas, features[0].sigmoid(), camera_path, '/camera_norm_fea.jpg')
        # self.vis_feature(metas, features[1].sigmoid(), camera_path, '/lidar_norm_fea.jpg')
        # self.vis_feature(metas, features_new[0].sigmoid(), camera_path, '/camera_mask_fea.jpg')
        # self.vis_feature(metas, features_new[1].sigmoid(), camera_path, '/lidar_mask_fea.jpg')
        # self.vis_feature(metas, f_camq, camera_path, '/camq.jpg')
        # self.vis_feature(metas, f_lidq, camera_path, '/lidq.jpg')
        # self.vis_feature(metas, camera_top, camera_path, '/cam_top.jpg')
        # self.vis_feature(metas, lidar_top, camera_path, '/lid_top.jpg')
        # exit()

        # todo we use the high_level_feature to generate the mask [2, 80, 180, 180]
        # f_cam_hl,  cam_6fea= self.get_HL_camera_features(img, points, camera2ego, lidar2ego, lidar2camera,
        #         lidar2image, camera_intrinsics, camera2lidar,img_aug_matrix, lidar_aug_matrix, metas)
        # self.vis_6_cam_features(metas, cam_6fea, camera_path)
        # self.vis_feature(metas, f_cam_hl, camera_path, '/swin_high_fea.jpg')
        # exit()

        # todo decoder
        batch_size = x.shape[0]
        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)  # todo x=[1, [2, 512, 180, 180]]

        # todo head
        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            # print("loss calculate")
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
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

    def get_the_local_max(self, map):
        heatmap = map.detach().sigmoid() # todo [2, 80, 180, 180]
        heatmap = torch.sum(heatmap, dim=1) # todo [2, 180, 180] range(1, 1.9526)
        heatmap = heatmap.view(len(map), -1) # todo [2, 32400]
        new_mask = torch.zeros_like(heatmap) # todo [2, 32400]
        # todo get the mask
        top_proposals = heatmap.argsort(dim=-1, descending=True)[
            ..., : self.num_proposals] # todo [2, 200] this is the  index

        # todo make top_query has number
        for i in range(len(map)):
            new_mask[i][top_proposals[i]] = 1
        new_mask = new_mask.view(len(map), 1, map.size(2), map.size(3))
        return new_mask

    def mask_modal_trainig(self, img, points):
        # todo images[batcsize [6, 3, 256, 704]] points[batchsize [226176, 5]]
        # todo mask-modal-training
        pro_cam = 0.1
        pro_lid = 0.2

        pro_train = random.uniform(0, 1)
        batch_size = len(img)
        mask_batch = int(random.uniform(0, batch_size))

        if pro_train >= 0 and pro_train < pro_cam:
            img[mask_batch] = img[mask_batch] * 0
        elif pro_train >= pro_cam and pro_train < pro_lid:
            points[mask_batch] = points[mask_batch] * 0

        return img, points

    # todo some mask tools
    def get_HL_camera_features(
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
        # todo only use the high_level_features to generate the bev features
        if not isinstance(x, torch.Tensor):
            x = x[0]
        x = self.high_level_cam(x)
        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        # todo y = (2, 6, C, H, W)
        y = x.clone().detach()
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
        return x, y

    # todo some vis tools
    # todo vis the 6 camera features from the swin block

    def vis_6_cam_features(self, img_meta, feature, path):
        for i, img_meta in enumerate(img_meta):
            filename = img_meta['filename'][0]
            path_pic = path + filename.split("/")[-1].split(".")[0].split("__")[0]
            self.mkdir(path_pic)

            # todo vis all the camera photo
            plt.xticks(())
            plt.yticks(())
            for k in range(len(feature[0])):
                imagename = img_meta['filename'][k]
                img = cv2.imread(imagename)
                plt.subplot(2, 3, k + 1)
                plt.imshow(img)
            img_name = path_pic + '/6_cam_image.jpg'
            plt.savefig(img_name)
            plt.close("all")

            # todo vis all the feature of cam photo
            plt.xticks(())
            plt.yticks(())
            for j in range(len(feature[0])):
                heat = torch.sum(feature[i][j], dim=0).cpu().clone().detach().numpy()
                plt.subplot(2, 3, j + 1)
                plt.imshow(heat, cmap='rainbow')
            pic_name = path_pic + "/6_cam_fea.jpg"
            plt.savefig(pic_name)
            plt.close("all")

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

    def vis_all(self, img_meta, feature1, feature2, feature3, feature4, path):
        for i, img_meta in enumerate(img_meta):
            filename = img_meta['filename'][0]
            path_pic = path + filename.split("/")[-1].split(".")[0].split("__")[0]
            self.mkdir(path_pic)
            plt.xticks(())
            plt.yticks(())
            heat1 = torch.sum(feature1[i], dim=0).cpu().clone().detach().numpy()
            heat2 = torch.sum(feature2[i], dim=0).cpu().clone().detach().numpy()
            heat3 = torch.sum(feature3[i], dim=0).cpu().clone().detach().numpy()
            heat4 = torch.sum(feature4[i], dim=0).cpu().clone().detach().numpy()
            plt.subplot(2, 2, 1)
            plt.imshow(heat1, cmap='rainbow')
            plt.subplot(2, 2, 2)
            plt.imshow(heat2, cmap='rainbow')
            plt.subplot(2, 2, 3)
            plt.imshow(heat3, cmap='rainbow')
            plt.subplot(2, 2, 4)
            plt.imshow(heat4, cmap='rainbow')
            pic_name = path_pic + "/feature.jpg"
            plt.savefig(pic_name)
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
