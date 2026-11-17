from typing import Any, Dict
import os
import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmcv.parallel import MMDistributedDataParallel
from mmdet3d.models import FUSIONMODELS
from mmdet.utils import get_root_logger
from mmdet3d.core import LiDARInstance3DBoxes
from mmcv.runner import load_checkpoint
from .bevfusion import BEVFusion
from torch.nn import functional as F
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
import random
import cv2
from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
import warnings
import torch
from mmcv.cnn import xavier_init, constant_init
import math
from mmcv.utils import ext_loader
from .base import Base3DFusionModel
import matplotlib.pyplot as plt
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

__all__ = ["Dis_bevfusion"]

# todo the common and unique feature
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


class MMF(nn.Module):
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
        # todo fusion encoder former layer
        self.CL = Fusionformerlayer()
        self.C2C = Fusionformerlayer()
        self.L2L = Fusionformerlayer()

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

        # todo encode the features
        # todo common feature
        C2L = self.CL(f_cam_new, f_lid_new, bev_pos, ref_2d)
        L2C = self.CL(f_lid_new, f_cam_new, bev_pos, ref_2d)
        # todo unique feature
        C2C = self.C2C(f_cam_new, f_cam_new, bev_pos, ref_2d)
        L2L = self.L2L(f_lid_new, f_lid_new, bev_pos, ref_2d)

        return C2L, L2C, C2C, L2L


@FUSIONMODELS.register_module()
class Dis_bevfusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        tea_encoder: Dict[str, Any],
        tea_fuser: Dict[str, Any],
        tea_decoder: Dict[str, Any],
        tea_heads: Dict[str, Any],
        tea_pretrained = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # super(Dis_bevfusion, self).__init__(encoders, fuser, decoder, heads, tea_encoder, tea_fuser, tea_decoder, tea_heads)
        # super(Dis_bevfusion, self).__init__(encoders, fuser, decoder, heads)

        # todo build student model
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

        # todo build teacher model, build the encoder camera backbone is main
        # todo build the en_camera
        tea_encoder_camera= nn.ModuleDict(
            {
                "backbone": build_backbone(tea_encoder["camera"]["backbone"]),
                "neck": build_neck(tea_encoder["camera"]["neck"]),
                "vtransform": build_vtransform(tea_encoder["camera"]["vtransform"]),
            }
        )
        # todo build en_lidar
        if tea_encoder["lidar"]["voxelize"].get("max_num_points", -1) > 0:
            tea_voxelize_module = Voxelization(**tea_encoder["lidar"]["voxelize"])
        else:
            tea_voxelize_module = DynamicScatter(**tea_encoder["lidar"]["voxelize"])
        tea_encoder_lidar = nn.ModuleDict(
            {
                "voxelize": tea_voxelize_module,
                "backbone": build_backbone(tea_encoder["lidar"]["backbone"]),
            }
        )
        self.voxelize_reduce = tea_encoder["lidar"].get("voxelize_reduce", True)
        # todo build tea_encoder
        self.tea_encoder = nn.ModuleDict()
        self.tea_encoder["encoders"] = nn.ModuleDict(
            {
            "camera": tea_encoder_camera,
            "lidar": tea_encoder_lidar,
            }
        )
        # todo build the tea_fuser
        self.tea_fuser = nn.ModuleDict(
            {
                "fuser": build_fuser(tea_fuser),
            }
        )
        # todo build tea_decoder
        self.tea_decoder = nn.ModuleDict()
        self.tea_decoder["decoder"] = nn.ModuleDict(
            {
                "backbone": build_backbone(tea_decoder["backbone"]),
                "neck": build_neck(tea_decoder["neck"]),
            }
        )

        # todo build tea_head
        self.tea_head = nn.ModuleDict()
        for name in tea_heads:
            if tea_heads[name] is not None:
                self.tea_head[name] = build_head(tea_heads[name])

        # todo build our Multi-modality extraction module
        # self.stu_MMF = MMF()
        # self.tea_MMF = MMF()

        # todo the load the teacher model pth
        self.tea_load_weights(pretrain = tea_pretrained)
        # todo freeze model
        self.freeze_models()

        # todo weight design
        self.fuser_weight = 10
        self.encoders_weight = 10
        self.decoder_weight = 100
        self.common_weight = 1000
        self.specific_weight =1000
        self.CMD_weight = 10


    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def freeze_models(self):
        self.tea_encoder.eval()
        for param_e in self.tea_encoder.parameters():
            param_e.requires_grad = False
        self.tea_fuser.eval()
        for param_f in self.tea_fuser.parameters():
            param_f.requires_grad = False
        self.tea_decoder.eval()
        for param_d in self.tea_decoder.parameters():
            param_d.requires_grad = False
        for param_h in self.tea_head.parameters():
            param_h.requires_grad = False

    def tea_load_weights(self, pretrain):
        print("begin load the teacher model")
        logger = get_root_logger()
        load_checkpoint(self.tea_encoder, pretrain, strict=False, logger=logger)
        print("load teacher backbone success")
        load_checkpoint(self.tea_fuser, pretrain, strict=False, logger=logger)
        print("load teacher fuser success")
        load_checkpoint(self.tea_decoder, pretrain, strict=False, logger=logger)
        print("load teacher decoer success")
        load_checkpoint(self.tea_head, pretrain, strict=False, logger=logger)
        print("load teacher head success")
        print("load teacher model success")

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

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

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
        return x

    def tea_extract_camera_features(
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

        x = self.tea_encoder["encoders"]["camera"]["backbone"](x)
        x = self.tea_encoder["encoders"]["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.tea_encoder["encoders"]["camera"]["vtransform"](
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
        return x

    def tea_extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.tea_voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.tea_encoder["encoders"]["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x


    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x


    @torch.no_grad()
    @force_fp32()
    def tea_voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.tea_encoder["encoders"]["lidar"]["voxelize"](res)
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
        # todo get the stu encoder
        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
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

        # todo get the stu fuser
        if self.fuser is not None:
            stu_fuser_fea = self.fuser(features)
        else:
            assert len(features) == 1, features
            stu_fuser_fea = features[0]

        batch_size = stu_fuser_fea.shape[0]

        # todo get the stu decoder
        x = self.decoder["backbone"](stu_fuser_fea)
        x = self.decoder["neck"](x)

        # todo get the tea features
        self.tea_encoder.eval()
        self.tea_fuser.eval()
        self.tea_decoder.eval()
        self.tea_head.eval()
        with torch.no_grad():
            # todo get the encoder feature
            tea_features = []
            for sensor in (
                self.tea_encoder["encoders"] if self.training else list(self.tea_encoder["encoders"].keys())[::-1]
            ):
                if sensor == "camera":
                    tea_feature = self.tea_extract_camera_features(
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
                    tea_feature = self.tea_extract_lidar_features(points)
                else:
                    raise ValueError(f"unsupported sensor: {sensor}")
                tea_features.append(tea_feature)
            if not self.training:
                # avoid OOM
                tea_features = tea_features[::-1]
            # todo get the fuser feature
            if self.tea_fuser is not None:
                tea_fuser_features = self.tea_fuser["fuser"](tea_features)
            else:
                assert len(tea_features) == 1, tea_features
                tea_fuser_features = tea_features[0]
            # todo get the decoder feature
            tea_decoder_features = self.tea_decoder["decoder"]["backbone"](tea_fuser_features)
            tea_decoder_features = self.tea_decoder["decoder"]["neck"](tea_decoder_features)
            # todo get the head output
            for type, head in self.tea_head.items():
                if type == "object":
                    tea_pred_dict = head(tea_decoder_features, metas)

        # todo get the model_common feature model_specific_common
        # stu_C2L, stu_L2C, stu_C2C, stu_L2L = self.stu_MMF(features[0], features[1])
        # tea_C2L, tea_L2C, tea_C2C, tea_L2L = self.tea_MMF(tea_features[0], tea_features[1])

        # todo stu: features tea: tea_features   f0[2, 80, 180, 180] lidar f1[2, 256, 180, 180]
        # todo direct imitation loss
        sup_encoder_loss = self.sup_direct_loss(features, tea_features) * self.encoders_weight
        # todo cross_modal loss
        # sup_cross_modal_loss = self.sup_CM_loss(features, tea_features)

        # todo common loss [2, 256, 180, 180]
        # sup_common_loss = self.sup_common_loss(stu_C2L, stu_L2C, tea_C2L, tea_L2C) *  self.common_weight
        # todo specific loss
        # sup_specific_loss = self.sup_specific_loss(stu_C2C, stu_L2L, tea_C2C, tea_L2L) * self.specific_weight

        # todo fuse_modal loss
        # todo fuser: stu: stu_fuser_fea tea: tea_features  [2, 256, 180, 180]
        sup_fuser_loss = self.sup_fuser(stu_fuser_fea, tea_fuser_features) * self.fuser_weight
        # todo decoder: stu: x tea: tea_decoder_features   [[2,512,180,180]]
        sup_decoder_loss = self.sup_decoder(x[0], tea_decoder_features[0]) * self.decoder_weight


        # todo loss seems like a dict
        # todo {'loss_heatmap': tensor(4.7756, device='cuda:0', grad_fn=<MulBackward0>),
        #      'layer_-1_loss_cls': tensor(0.5004, device='cuda:0', grad_fn=<MulBackward0>),
        #      'layer_-1_loss_bbox': tensor(9.6442, device='cuda:0', grad_fn=<MulBackward0>),
        #      'matched_ious': tensor(0.0055, device='cuda:0')}

        # todo vis
        # self.vis(metas, features[0], features[1], points, gt_bboxes_3d, gt_labels_3d, tea_features[0], tea_features[1])

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                    # todo add the distillation loss
                    sup_head_loss = self.sup_head(pred_dict, tea_pred_dict)
                    # losses.update(dict(loss_CM=sup_cross_modal_loss))
                    # losses.update(dict(loss_common=sup_common_loss))
                    # losses.update(dict(loss_specific=sup_specific_loss))
                    losses.update(dict(loss_encoder=sup_encoder_loss))
                    losses.update(dict(loss_fuser=sup_fuser_loss))
                    losses.update(dict(loss_decoder=sup_decoder_loss))
                    losses.update(dict(loss_head=sup_head_loss))
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

    def sup_direct_loss(self, features, tea_features):
        # todo cam f0[2, 80, 180, 180] lidar f1[2, 256, 180, 180]
        loss_fea_cam = torch.sqrt(torch.pow(features[0] - tea_features[0], 2).sum())
        sup_loss_cam = loss_fea_cam / (features[0].size(2) * features[0].size(3) * features[0].size(0))

        loss_fea_lid = torch.sqrt(torch.pow(features[1] - tea_features[1], 2).sum())
        sup_loss_lid = loss_fea_lid / (features[1].size(2) * features[1].size(3) * features[1].size(0))

        sup_loss = sup_loss_cam + sup_loss_lid
        return sup_loss

    def sup_CM_loss(self, features, tea_features):
        sup_loss = 0
        return sup_loss


    def sup_common_loss(self, stu_C2L, stu_L2C, tea_C2L, tea_L2C):
        loss_fea_common_cam = torch.sqrt(torch.pow(stu_C2L - tea_C2L, 2).sum())
        sup_loss_cam = loss_fea_common_cam / (tea_C2L.size(2) * tea_C2L.size(3) * tea_C2L.size(0))

        loss_fea_common_lid = torch.sqrt(torch.pow(stu_L2C - tea_L2C, 2).sum())
        sup_loss_lid = loss_fea_common_lid / (stu_L2C.size(2) * stu_L2C.size(3) * stu_L2C.size(0))

        sup_loss = sup_loss_cam + sup_loss_lid

        return sup_loss

    def sup_specific_loss(self, stu_C2C, stu_L2L, tea_C2C, tea_L2L):
        loss_fea_common_cam = torch.sqrt(torch.pow(stu_C2C - tea_C2C, 2).sum())
        sup_loss_cam = loss_fea_common_cam / (stu_C2C.size(2) * stu_C2C.size(3) * stu_C2C.size(0))

        loss_fea_common_lid = torch.sqrt(torch.pow(stu_L2L - tea_L2L, 2).sum())
        sup_loss_lid = loss_fea_common_lid / (stu_L2L.size(2) * stu_L2L.size(3) * stu_L2L.size(0))

        sup_loss = sup_loss_cam + sup_loss_lid

        return sup_loss


    def sup_fuser(self, s_fea, t_fea):
        # todo fea = [2, 256, 180, 180]
        loss_fea = torch.sqrt(torch.pow(s_fea - t_fea, 2).sum())
        sup_loss = loss_fea / (s_fea.size(2) * s_fea.size(3) * s_fea.size(0))
        return sup_loss

    def sup_decoder(self, s_fea, t_fea):
        # todo fea = [2, 256, 180, 180]
        loss_fea = torch.sqrt(torch.pow(s_fea - t_fea, 2).sum())
        sup_loss = loss_fea / (s_fea.size(2) * s_fea.size(3) * s_fea.size(0))
        return sup_loss

    def sup_head(self, pred_dict, tea_pred_dict):
        stu_heat_map = pred_dict[0][0]["dense_heatmap"]
        stu_score = pred_dict[0][0]["heatmap"][..., 0 * 200: (0 + 1) * 200,]

        tea_heat_map = tea_pred_dict[0][0]["dense_heatmap"]
        tea_score = tea_pred_dict[0][0]["heatmap"][..., 0 * 200: (0 + 1) * 200,]

        loss_heat_map = torch.sqrt(torch.pow(stu_heat_map - tea_heat_map, 2).sum())
        sup_heat_loss = loss_heat_map / (stu_heat_map.size(2) * stu_heat_map.size(3) * stu_heat_map.size(0))

        loss_score = torch.sqrt(torch.pow(stu_score - tea_score, 2).sum())
        sup_score_loss = loss_score / stu_score.size(2) * stu_score.size(0)

        sup_loss = sup_heat_loss + sup_score_loss
        return sup_loss


    # todo some vis tools
    def vis(self, metas, cam_fea, lid_fea, points, gt_bboxes_3d, gt_labels_3d, tea_cam_features, tea_lid_feature):
        camera_path = "/root/paddlejob/workspace/yyr/code/bevfusion-main/mask/camera/"
        self.vis_feature(metas, cam_fea, camera_path, '/camera_fea.jpg')
        self.vis_feature(metas, lid_fea, camera_path, '/lidar_fea.jpg')
        # self.vis_feature(metas, x, camera_path, '/fusion.jpg')
        self.vis_bev(metas, points, gt_bboxes_3d, gt_labels_3d, camera_path)
        self.vis_feature(metas, tea_cam_features, camera_path, '/tea_cam_fea.jpg')
        self.vis_feature(metas, tea_lid_feature, camera_path, '/tea_lid_fea.jpg')
        # self.vis_feature(metas, f_unq_cam, camera_path, '/unq_com_fea.jpg')
        # self.vis_feature(metas, f_unq_lid, camera_path, '/unq_lid_fea.jpg')
        # self.vis_feature(metas, MCL, camera_path, '/mask_com_CL.jpg')
        # self.vis_feature(metas, MLC, camera_path, '/mask_com_LC.jpg')
        # self.vis_feature(metas, MC, camera_path, '/mask_mc.jpg')
        # self.vis_feature(metas, ML, camera_path, '/mask_ml.jpg')
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

