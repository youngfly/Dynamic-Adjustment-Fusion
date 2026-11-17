from mmcv.cnn import build_conv_layer
import torch
import math
import torch.nn as nn
from torch.autograd import Function
from localAttention import (similar_forward,
                            similar_backward,
                            weighting_forward,
                            weighting_backward_ori,
                            weighting_backward_weight)
import pdb


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
        return out


class DeepInteractionEncoderLayer(nn.Module):
    def __init__(self, hidden_channel):
        super(DeepInteractionEncoderLayer, self).__init__()

        self.I2P_block = LocalContextAttentionBlock(hidden_channel, hidden_channel, 9)
        self.P_IML = LocalContextAttentionBlock(hidden_channel, hidden_channel, 9)
        self.P_out_proj = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size=1, norm_layer=nn.BatchNorm2d,
                                     activation_layer=None)
        self.P_integration = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size=1, norm_layer=nn.BatchNorm2d,
                                        activation_layer=None)

        self.P2I_block = LocalContextAttentionBlock(hidden_channel, hidden_channel, 9)
        self.I_IML = LocalContextAttentionBlock(hidden_channel, hidden_channel, 9)
        self.I_out_proj = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size=1, norm_layer=nn.BatchNorm2d,
                                     activation_layer=None)
        self.I_integration = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size=1, norm_layer=nn.BatchNorm2d,
                                        activation_layer=None)

    def forward(self, img_feat, lidar_feat):
        I2P_feat = self.I2P_block(lidar_feat, img_feat)
        P2P_feat = self.P_IML(lidar_feat, lidar_feat)
        P_Aug_feat = self.P_out_proj(torch.cat((I2P_feat, P2P_feat), dim=1))
        new_lidar_feat = self.P_integration(torch.cat((P_Aug_feat, lidar_feat), dim=1))

        P2I_feat = self.P2I_block(img_feat, lidar_feat)
        I2I_feat = self.I_IML(img_feat, img_feat)
        I_Aug_feat = self.I_out_proj(torch.cat((P2I_feat, I2I_feat), dim=1))
        new_img_feat = self.I_integration(torch.cat((I_Aug_feat, img_feat), dim=1))
        return new_img_feat, new_lidar_feat


class DeepInteractionEncoder(nn.Module):
    def __init__(self,
                num_layers=1,
                in_channels_img=80,
                in_channels_pts=256,
                hidden_channel=256,
                bn_momentum=0.1,
                bias='auto',
                ):
        super(DeepInteractionEncoder, self).__init__()

        self.shared_conv_pts = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_pts,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        self.shared_conv_img = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_img,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        self.end_conv_img = build_conv_layer(
            dict(type='Conv2d'),
            hidden_channel,
            in_channels_img,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        self.num_layers = num_layers
        self.fusion_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.fusion_blocks.append(DeepInteractionEncoderLayer(hidden_channel))

        self.bn_momentum = bn_momentum
        self.init_weights()

    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, img_feats, pts_feats):
        new_img_feat = self.shared_conv_img(img_feats)
        new_pts_feat = self.shared_conv_pts(pts_feats)
        for i in range(self.num_layers):
            new_img_feat, new_pts_feat = self.fusion_blocks[i](new_img_feat, new_pts_feat)
        new_img_feat = self.end_conv_img(new_img_feat)
        return new_img_feat, new_pts_feat