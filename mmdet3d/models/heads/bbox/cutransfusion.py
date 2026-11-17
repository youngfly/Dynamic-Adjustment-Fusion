import copy
import os
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.core import (
    PseudoSampler,
    circle_nms,
    draw_heatmap_gaussian,
    gaussian_radius,
    xywhr2xyxyr,
)
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import FFN, PositionEmbeddingLearned, TransformerDecoderLayer
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import (
    AssignResult,
    build_assigner,
    build_bbox_coder,
    build_sampler,
    multi_apply,
)
import matplotlib.pyplot as plt

__all__ = ["CuTransFusionHead"]
EPS = 1e-12

def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


@HEADS.register_module()
class CuTransFusionHead(nn.Module):
    def __init__(
        self,
        num_proposals=128,
        auxiliary=True,
        in_channels=128 * 3,
        hidden_channel=128,
        num_classes=4,
        # config for Transformer
        num_decoder_layers=3,
        num_heads=8,
        nms_kernel_size=1,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation="relu",
        # config for FFN
        common_heads=dict(),
        num_heatmap_convs=2,
        conv_cfg=dict(type="Conv1d"),
        norm_cfg=dict(type="BN1d"),
        bias="auto",
        # loss
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        loss_iou=dict(
            type="VarifocalLoss", use_sigmoid=True, iou_weighted=True, reduction="mean"
        ),
        loss_bbox=dict(type="L1Loss", reduction="mean"),
        loss_heatmap=dict(type="GaussianFocalLoss", reduction="mean"),
        # others
        train_cfg=None,
        test_cfg=None,
        bbox_coder=None,
    ):
        super(CuTransFusionHead, self).__init__()

        self.fp16_enabled = False

        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.nms_kernel_size = nms_kernel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.loss_heatmap = build_loss(loss_heatmap)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False

        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type="Conv2d"),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        self.fuse_conv = build_conv_layer(
            dict(type="Conv2d"),
            hidden_channel * 2,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        layers = []
        layers.append(
            ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=dict(type="BN2d"),
            )
        )
        layers.append(
            build_conv_layer(
                dict(type="Conv2d"),
                hidden_channel,
                num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            )
        )
        self.heatmap_head = nn.Sequential(*layers)
        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel,
                    num_heads,
                    ffn_channel,
                    dropout,
                    activation,
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                )
            )

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(
                FFN(
                    hidden_channel,
                    heads,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=bias,
                )
            )

        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.test_cfg["grid_size"][0] // self.test_cfg["out_size_factor"]
        y_size = self.test_cfg["grid_size"][1] // self.test_cfg["out_size_factor"]
        self.bev_pos = self.create_2D_grid(x_size, y_size)
        self.bev_pos_all = self.create_2D_grid(x_size * 3, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        # todo batch_x
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        # todo coord_base [2, 180, 180]
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        # todo [1, 2, 32400]->[1,32400,2]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, "query"):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def forward_single(self, inputs, img_inputs, metas, y1, y2, y3, y4):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        # todo get the original heatmap
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs) # todo [2, 128, 180, 180]

        # todo decode other modality feature # todo [2, 128, 180, 180]
        C2L = self.shared_conv(y1)
        L2C = self.shared_conv(y2)
        C2C = self.shared_conv(y3)
        L2L = self.shared_conv(y4)

        common = self.fuse_conv(torch.cat((C2L, L2C), dim=1))
        unique = self.fuse_conv(torch.cat((C2C, L2L), dim=1))

        # todo concat the multimdoal feature with the w dim [2, 128, 900, 180]
        # all_modal_featrue = torch.cat((C2C, C2L, L2L, L2C), dim=2)
        all_modal_featrue = torch.cat((lidar_feat, common, unique), dim=2)
        bev_pos_all = self.bev_pos_all.repeat(batch_size, 1, 1).to(all_modal_featrue.device)  # todo [2, 32400*5, 2]
        all_feat_flatten = all_modal_featrue.view(batch_size, all_modal_featrue.shape[1], -1)  # [BS, C, H*W] # todo [2, 128, 32400]



        # todo this is fusion feature in reality
        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1
        )  # [BS, C, H*W] # todo [2, 128, 32400]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device) # todo [2, 32400, 2]
        # todo get the dense_heatmap,which is cls prediction
        dense_heatmap = self.heatmap_head(lidar_feat)
        dense_heatmap_img = None
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # todo equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        # todo for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg["dataset"] == "nuScenes":
            local_max[
                :,
                8,
            ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[
                :,
                9,
            ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg["dataset"] == "Waymo":  # for Pedestrian & Cyclist in Waymo
            local_max[
                :,
                1,
            ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[
                :,
                2,
            ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max) # todo [2, 10, 180, 180]
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1) # todo [2,10,32400]

        # todo get the proposal top 200 among all classes [2, 32400*10]
        # todo this is the index of all classes
        top_proposals = heatmap.contiguous().view(batch_size, -1).argsort(dim=-1, descending=True)[
            ..., : self.num_proposals] # todo [2, 200]

        # todo heatmap.shape[-1] = 324000
        top_proposals_class = top_proposals // heatmap.shape[-1] # todo [2, 200] category shang int
        top_proposals_index = top_proposals % heatmap.shape[-1] # todo [2, 200] index yushu int

        # todo get the query_feat from all feature by top_index
        # todo lidar_feat_flattn = [2, 128, 32400]
        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, lidar_feat_flatten.shape[1], -1),
            dim=-1,) # todo [2, 128, 200] second is feature channel
        self.query_labels = top_proposals_class
        # todo add category embedding
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(
            0, 2, 1
        )
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding # todo [2, 128, 200]

        # todo get the query_pos by bev_pos
        # todo bev_pos = [2, 32400, 2]
        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :]
            .permute(0, 2, 1)
            .expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        ) # todo [2, 200, 2]  third is meaning [x, y]

        # todo decode the query feat
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            prefix = "last_" if (i == self.num_decoder_layers - 1) else f"{i}head_"

            # todo Transformer Decoder Layer
            # todo query_feat = [2, 128, 200], lidar_feat_flatten = [2, 128, 32400]
            # todo query_pos = [2, 200, 2], bev_pos = [2, 32400, 2]
            # query_feat = self.decoder[i](
            #     query_feat, lidar_feat_flatten, query_pos, bev_pos
            # ) # todo [2, 128, 200]
            # todo intreact with other modal
            query_feat = self.decoder[i](
                query_feat, all_feat_flatten, query_pos, bev_pos_all
            ) # todo [2, 128, 200]

            # todo Prediction FFN heads
            res_layer = self.prediction_heads[i](query_feat)
            # todo add the offset to the basic grid
            res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
            first_res_layer = res_layer
            ret_dicts.append(res_layer)

            # todo for next level positional embedding when there are multi level
            query_pos = res_layer["center"].detach().clone().permute(0, 2, 1)

            # todo vis_feature_map and query_position
            # camera_path = "/root/paddlejob/workspace/yyr/code/bevfusion-main/mask/camera/"
            # self.vis_feature(metas, dense_heatmap,camera_path,'/prediction_fea.jpg')
            # exit()

        # todo obtain the postive classification results from the heatmap
        # todo heatmap is cls prediction, dense_heatmap is original featuremap
        ret_dicts[0]["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        )  # [bs, num_classes, num_proposals]
        ret_dicts[0]["dense_heatmap"] = dense_heatmap

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # todo return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ["dense_heatmap", "dense_heatmap_old", "query_heatmap_score"]:
                new_res[key] = torch.cat(
                    [ret_dict[key] for ret_dict in ret_dicts], dim=-1
                )
            else:
                new_res[key] = ret_dicts[0][key]
        return [new_res]

    def forward(self, feats, metas, y1, y2, y3, y4):
        """Forward pass.
        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
        Returns:
            tuple(list[dict]): Output results. first index by level, second index by layer
        """
        if isinstance(feats, torch.Tensor):
            feats = [feats]
            y1 = [y1]
            y2 =[y2]
            y3 = [y3]
            y4 = [y4]
        res = multi_apply(self.forward_single, feats, [None], [metas], y1, y2, y3, y4)
        assert len(res) == 1, "only support one level features."
        return res

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        """Generate training targets.
        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dicts (tuple of dict): first index by layer (default 1)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)  [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                pred_dict[key] = preds_dict[0][key][batch_idx : batch_idx + 1]
            list_of_pred_dict.append(pred_dict)

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)

        res_tuple = multi_apply(
            self.get_targets_single,
            gt_bboxes_3d,
            gt_labels_3d,
            list_of_pred_dict,
            np.arange(len(gt_labels_3d)),
        )
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        heatmap = torch.cat(res_tuple[7], dim=0)
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        )

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.
        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dict (dict): dict of prediction result for a single sample
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask)  [1, num_proposals]
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
        """
        num_proposals = preds_dict["center"].shape[-1]

        # todo get pred boxes, carefully ! donot change the network outputs
        score = copy.deepcopy(preds_dict["heatmap"].detach())
        center = copy.deepcopy(preds_dict["center"].detach())
        height = copy.deepcopy(preds_dict["height"].detach())
        dim = copy.deepcopy(preds_dict["dim"].detach())
        rot = copy.deepcopy(preds_dict["rot"].detach())
        if "vel" in preds_dict.keys():
            vel = copy.deepcopy(preds_dict["vel"].detach())
        else:
            vel = None
        # todo decode the prediction to real world metric bbox
        boxes_dict = self.bbox_coder.decode(
            score, rot, dim, center, height, vel
        )
        bboxes_tensor = boxes_dict[0]["bboxes"]
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        # todo each layer should do label assign seperately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[
                self.num_proposals * idx_layer : self.num_proposals * (idx_layer + 1), :
            ]
            score_layer = score[
                ...,
                self.num_proposals * idx_layer : self.num_proposals * (idx_layer + 1),
            ]

            if self.train_cfg.assigner.type == "HungarianAssigner3D":
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    gt_labels_3d,
                    score_layer,
                    self.train_cfg,
                )
            elif self.train_cfg.assigner.type == "HeuristicAssigner":
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    None,
                    gt_labels_3d,
                    self.query_labels[batch_idx],
                )
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat([res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )
        sampling_result = self.bbox_sampler.sample(
            assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor
        )
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(
            center.device
        )
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(
            center.device
        )
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # todo both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        device = labels.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
        ).to(device)
        grid_size = torch.tensor(self.train_cfg["grid_size"])
        pc_range = torch.tensor(self.train_cfg["point_cloud_range"])
        voxel_size = torch.tensor(self.train_cfg["voxel_size"])
        feature_map_size = (
            grid_size[:2] // self.train_cfg["out_size_factor"]
        )  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(
            self.num_classes, feature_map_size[1], feature_map_size[0]
        )
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.train_cfg["out_size_factor"]
            length = length / voxel_size[1] / self.train_cfg["out_size_factor"]
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width), min_overlap=self.train_cfg["gaussian_overlap"]
                )
                radius = max(self.train_cfg["min_radius"], int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (
                    (x - pc_range[0])
                    / voxel_size[0]
                    / self.train_cfg["out_size_factor"]
                )
                coor_y = (
                    (y - pc_range[1])
                    / voxel_size[1]
                    / self.train_cfg["out_size_factor"]
                )

                center = torch.tensor(
                    [coor_x, coor_y], dtype=torch.float32, device=device
                )
                center_int = center.to(torch.int32)

                # original
                # draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)
                # NOTE: fix
                draw_heatmap_gaussian(
                    heatmap[gt_labels_3d[idx]], center_int[[1, 0]], radius
                )

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (
            labels[None],
            label_weights[None],
            bbox_targets[None],
            bbox_weights[None],
            ious[None],
            int(pos_inds.shape[0]),
            float(mean_iou),
            heatmap[None],
        )

    @force_fp32(apply_to=("preds_dicts"))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.
        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (list[list[dict]]): Output of forward function.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        # todo get all labels
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        ) = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        if hasattr(self, "on_the_image_mask"):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        # todo compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(preds_dict["dense_heatmap"]),
            heatmap,
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        )
        loss_dict["loss_heatmap"] = loss_heatmap

        # todo compute loss for each layer, just one layer in reality
        for idx_layer in range(self.num_decoder_layers if self.auxiliary else 1):
            # todo get the lay
            if idx_layer == self.num_decoder_layers - 1 or (
                idx_layer == 0 and self.auxiliary is False
            ):
                prefix = "layer_-1"
            else:
                prefix = f"layer_{idx_layer}"

            # todo our reweight
            senmatic_weight_cls, geometry_weight_cls, \
                senmatic_weight_reg, geometry_weight_reg, \
                    sg_weight_cls, sg_weight_reg_new = \
                self.reweight(preds_dict, bbox_targets, ious, idx_layer, gt_bboxes_3d)

            # todo get each layer_labels, which is related to classification *****************
            layer_labels = labels[
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ].reshape(-1)
            layer_label_weights = label_weights[
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ].reshape(-1)
            layer_score = preds_dict["heatmap"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes)

            # todo calculate the category loss
            # todo original
            layer_loss_cls = self.loss_cls(
                layer_cls_score,
                layer_labels,
                layer_label_weights,
                avg_factor=max(num_pos, 1),
            )

            # todo reweight  senmatic_weight_cls, geometry_weight_cls, sg_weight_cls
            # layer_loss_cls = self.loss_cls(
            #     layer_cls_score,
            #     layer_labels,
            #     layer_label_weights * senmatic_weight_cls,
            #     avg_factor=max(num_pos, 1),
            # )

            # todo get the layer_box, which is related to regression ********************
            layer_center = preds_dict["center"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ]
            layer_height = preds_dict["height"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ]
            layer_rot = preds_dict["rot"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ]
            layer_dim = preds_dict["dim"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ]
            # todo combine the preds between regression
            preds = torch.cat(
                [layer_center, layer_height, layer_dim, layer_rot], dim=1
            ).permute(
                0, 2, 1
            )  # [BS, num_proposals, code_size]
            # todo speed is added for necessary
            if "vel" in preds_dict.keys():
                layer_vel = preds_dict["vel"][
                    ...,
                    idx_layer
                    * self.num_proposals : (idx_layer + 1)
                    * self.num_proposals,
                ]
                preds = torch.cat(
                    [layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1
                ).permute(
                    0, 2, 1
                )  # [BS, num_proposals, code_size]
            # todo the weight for each regression
            code_weights = self.train_cfg.get("code_weights", None)
            # todo code_weights [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
            layer_bbox_weights = bbox_weights[
                :,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
                :,
            ]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(
                code_weights
            )

            # todo get the regression targets
            layer_bbox_targets = bbox_targets[
                :,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
                :,
            ]
            # todo calculate the regression loss
            # todo original
            layer_loss_bbox = self.loss_bbox(
                preds, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos, 1)
            )

            # todo reiweight senmatic_weight_reg, geometry_weight_reg, sg_weight_reg_new
            # layer_loss_bbox = self.loss_bbox(
            #     preds, layer_bbox_targets,
            #     layer_reg_weights * senmatic_weight_reg,
            #     avg_factor=max(num_pos, 1)
            # )

            # layer_iou = preds_dict['iou'][..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals].squeeze(1)
            # layer_iou_target = ious[..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals]
            # layer_loss_iou = self.loss_iou(layer_iou, layer_iou_target, layer_bbox_weights.max(-1).values, avg_factor=max(num_pos, 1))

            # todo get all loss
            loss_dict[f"{prefix}_loss_cls"] = layer_loss_cls
            loss_dict[f"{prefix}_loss_bbox"] = layer_loss_bbox
            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        loss_dict[f"matched_ious"] = layer_loss_cls.new_tensor(matched_ious)

        return loss_dict

    def get_bboxes(self, preds_dicts, metas, img=None, rescale=False, for_roi=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]["heatmap"].shape[0]
            batch_score = preds_dict[0]["heatmap"][..., -self.num_proposals :].sigmoid()
            # if self.loss_iou.loss_weight != 0:
            #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid())
            one_hot = F.one_hot(
                self.query_labels, num_classes=self.num_classes
            ).permute(0, 2, 1)
            batch_score = batch_score * preds_dict[0]["query_heatmap_score"] * one_hot

            batch_center = preds_dict[0]["center"][..., -self.num_proposals :]
            batch_height = preds_dict[0]["height"][..., -self.num_proposals :]
            batch_dim = preds_dict[0]["dim"][..., -self.num_proposals :]
            batch_rot = preds_dict[0]["rot"][..., -self.num_proposals :]
            batch_vel = None
            if "vel" in preds_dict[0]:
                batch_vel = preds_dict[0]["vel"][..., -self.num_proposals :]

            # todo 将预测的进行解码
            temp = self.bbox_coder.decode(
                batch_score,
                batch_rot,
                batch_dim,
                batch_center,
                batch_height,
                batch_vel,
                filter=True,
            )

            if self.test_cfg["dataset"] == "nuScenes":
                self.tasks = [
                    dict(
                        num_class=8,
                        class_names=[],
                        indices=[0, 1, 2, 3, 4, 5, 6, 7],
                        radius=-1,
                    ),
                    dict(
                        num_class=1,
                        class_names=["pedestrian"],
                        indices=[8],
                        radius=0.175,
                    ),
                    dict(
                        num_class=1,
                        class_names=["traffic_cone"],
                        indices=[9],
                        radius=0.175,
                    ),
                ]
            elif self.test_cfg["dataset"] == "Waymo":
                self.tasks = [
                    dict(num_class=1, class_names=["Car"], indices=[0], radius=0.7),
                    dict(
                        num_class=1, class_names=["Pedestrian"], indices=[1], radius=0.7
                    ),
                    dict(num_class=1, class_names=["Cyclist"], indices=[2], radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]["bboxes"]
                scores = temp[i]["scores"]
                labels = temp[i]["labels"]
                ## adopt circle nms for different categories
                if self.test_cfg["nms_type"] != None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task["indices"]:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task["radius"] > 0:
                            if self.test_cfg["nms_type"] == "circle":
                                boxes_for_nms = torch.cat(
                                    [
                                        boxes3d[task_mask][:, :2],
                                        scores[:, None][task_mask],
                                    ],
                                    dim=1,
                                )
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task["radius"],
                                    )
                                )
                            else:
                                boxes_for_nms = xywhr2xyxyr(
                                    metas[i]["box_type_3d"](
                                        boxes3d[task_mask][:, :7], 7
                                    ).bev
                                )
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_gpu(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task["radius"],
                                    pre_maxsize=self.test_cfg["pre_maxsize"],
                                    post_max_size=self.test_cfg["post_maxsize"],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(task_mask != 0)[0][
                                task_keep_indices
                            ]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(
                        bboxes=boxes3d[keep_mask],
                        scores=scores[keep_mask],
                        labels=labels[keep_mask],
                    )
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_layer.append(ret)
            rets.append(ret_layer)
        assert len(rets) == 1
        assert len(rets[0]) == 1
        res = [
            [
                metas[0]["box_type_3d"](
                    rets[0][0]["bboxes"], box_dim=rets[0][0]["bboxes"].shape[-1]
                ),
                rets[0][0]["scores"],
                rets[0][0]["labels"].int(),
            ]
        ]
        return res

    def reweight(self, preds_dict, bbox_targets, ious, idx_layer, gt_bboxes_3d):
        # todo our senmatic (cls+loc) + geometry fusion (distnace+volume)
        # todo cls
        score_all = preds_dict["heatmap"][
                    ...,
                    idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
                    ]  # todo [2, 10, 200]
        score_all = score_all.sigmoid().permute(0, 2, 1).reshape(-1, self.num_classes)  # todo score_all = [400,10]
        score_all_max, _ = torch.max(score_all, dim=1)  # todo [400]

        # todo loc
        iou_target = ious[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]  # todo [2,200]
        iou_all = iou_target.reshape(-1, 1).squeeze()  # todo [400,1]

        # todo geometry
        bbox_gt = bbox_targets[
                  :,
                  idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
                  :,
                  ]  # todo [2, 200, 10]
        bbox_all = bbox_gt.reshape(-1, 10)  # todo [400, 10]


        # todo need to decode
        center_x = bbox_all[:, 0]
        center_y = bbox_all[:, 1]
        height = bbox_all[:, 2]
        l, w, h = bbox_all[:, 3], bbox_all[:, 4], bbox_all[:, 5]

        # todo get the postive samples from 200 query according to one-2-one assign
        # todo postive samples = gt_nums
        gt_mask = (w != 0) + 0 # todo [400, 1]
        res_mask = torch.where(gt_mask == 0, 1, 0)

        # todo decode x, y and get the distance
        c_x = center_x * 8 * 0.075 - 54
        c_y = center_y * 8 * 0.075 - 54
        distance = torch.sqrt(c_x ** 2 + c_y ** 2) * gt_mask
        distance_norm = distance / distance.max()
        distance_weight = torch.exp(distance_norm) * gt_mask  # todo [1->2.736]

        # todo decode l, w, h and get the volume
        l_r, w_r, h_r = bbox_all[:, 3].exp(), bbox_all[:, 4].exp(), bbox_all[:, 5].exp()
        volume = w_r * h_r * l_r * gt_mask
        volume_norm = volume / volume.max()
        volume_weight = (torch.exp(-volume_norm) + 1) * gt_mask

        # todo distance from the gt to predict
        center_pd = preds_dict["center"][
                       ...,
                       idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
                       ] # todo [2, 2 , 200]
        center_pd = center_pd.permute(0,2,1)
        center_pd = center_pd.reshape(-1, 2) # todo [400, 2]
        center_pd = center_pd * 8 * 0.075 - 54
        center_gt = bbox_all[:, 0:2] * 8 * 0.075 - 54  # todo [400, 2]
        center_mask = (bbox_all[:, 0:2] != 0) + 0
        center_gt_pos = center_gt * center_mask
        center_pd_pos = center_pd * center_mask
        center_res = center_gt_pos - center_pd_pos
        center_dis = torch.sqrt(center_res[:,0] ** 2 + center_res[:,1]**2)

        gama = 1
        t = lambda x: (-2 / 7) * (x ** gama) + (8 / 7)

        def normalize(x):
            y = t(x)
            y[x < 0.5] = 1
            y[x > 4] = 0
            return y
        # todo [1, 2]
        p_distance = normalize(center_dis) * gt_mask + 1 # todo [400, 1]
        # todo dynamic wight
        alpha = 1
        theta = 1

        beta = 1

        # todo geo weight
        # g_weight = p_distance
        # g_weight = volume_weight
        g_weight = distance_weight

        # todo sen weight
        cls_pre = score_all_max * gt_mask
        iou_all = iou_all * gt_mask
        t_score = cls_pre * (iou_all ** theta)

        # todo make the cls weight [1->2.736] [400]
        senmatic_weight_cls = torch.exp(alpha * t_score) * gt_mask + res_mask
        geometry_weight_cls = g_weight
        sg_weight_cls = senmatic_weight_cls * geometry_weight_cls

        # todo reshape for reg weight[2,200]
        senmatic_weight_reg = senmatic_weight_cls.reshape(-1, 200).unsqueeze(2)
        geometry_weight_reg = geometry_weight_cls.reshape(-1, 200).unsqueeze(2)
        sg_weight_reg = sg_weight_cls.reshape(-1, 200).unsqueeze(2)

        # todo reduce the weight for reg because there 10 reg loss
        senmatic_weight_reg_new = senmatic_weight_reg.repeat(1, 1, 10)
        geometry_weight_reg_new = geometry_weight_reg.repeat(1, 1, 10)
        sg_weight_reg_new = sg_weight_reg.repeat(1, 1, 10)

        return senmatic_weight_cls, geometry_weight_cls, \
            senmatic_weight_reg_new, geometry_weight_reg_new, \
            sg_weight_cls, sg_weight_reg_new

    # todo vis
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

    def mkdir(self, path):
        isexists = os.path.exists(path)
        if not isexists:
            os.makedirs(path)
            print("file makes successfully")
        else:
            print("file is exist")
