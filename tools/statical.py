import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
# from torchpack.utils.tqdm import tqdm
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


from typing import List, Optional, Tuple
import cv2
from matplotlib import pyplot as plt

def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj

OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

gt_dict = {"car": 0, "truck": 0, "construction_vehicle": 0, "bus": 0,
    "trailer": 0, "barrier": 0, "motorcycle": 0, "bicycle": 0, "pedestrian": 0, "traffic_cone": 0}
dist_dict = {"car": 0, "truck": 0, "construction_vehicle": 0, "bus": 0,
    "trailer": 0, "barrier": 0, "motorcycle": 0, "bicycle": 0, "pedestrian": 0, "traffic_cone": 0}
area_dict = {"car": 0, "truck": 0, "construction_vehicle": 0, "bus": 0,
    "trailer": 0, "barrier": 0, "motorcycle": 0, "bicycle": 0, "pedestrian": 0, "traffic_cone": 0}

def volumw_cal(
    fpath: str,
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
        for index in range(coords.shape[0]):
            name = classes[labels[index]]

            box_volume_h = coords[index, 3].astype(np.int) - coords[index, 0].astype(np.int)
            box_volume_w = coords[index, 4].astype(np.int) - coords[index, 0].astype(np.int)
            box_volume_l = coords[index, 1].astype(np.int) - coords[index, 0].astype(np.int)
            box_volume = np.abs(box_volume_l * box_volume_w * box_volume_h)
            print(index)
            print(box_volume_l)
            print(name, box_volume)

def area_cal(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            coords[index, :, 0] = coords[index, :, 0] * 10
            coords[index, :, 1] = coords[index, :, 1] * 10
            x1 = coords[index, :, 0][0].numpy().astype(np.int)
            x2 = coords[index, :, 0][1].numpy().astype(np.int)
            x3 = coords[index, :, 0][2].numpy().astype(np.int)
            x4 = coords[index, :, 0][3].numpy().astype(np.int)
            y1 = coords[index, :, 1][0].numpy().astype(np.int)
            y2 = coords[index, :, 1][1].numpy().astype(np.int)
            y3 = coords[index, :, 1][2].numpy().astype(np.int)
            y4 = coords[index, :, 1][3].numpy().astype(np.int)

            bw = np.sqrt((x1 -x2)**2 + (y1 -y2)**2)
            bh = np.sqrt((x2 -x3)**2 + (y2 -y3)**2)
            area = bw * bh
            distance = np.sqrt(((x1+x2+x3+x4)/4)**2 + ((y1+y2+y3+y4)/4)**2)
            gt_dict["{}".format(name)] += 1
            area_dict["{}".format(name)] += area
            dist_dict["{}".format(name)] += distance



def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred", "all"])
    parser.add_argument("--checkpoint", type=str, default="pretrained/bevfusion-det.pth")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=0.5)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz/demo/")
    parser.add_argument("--sensor", type=str, default="all", choices=["camera", "lidar", "all"])
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    if args.mode == "pred" or args.mode == "all":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()

    for data in tqdm(dataflow):
        metas = data["metas"].data[0][0]
        name = "{}-{}".format(metas["timestamp"], metas["token"])

        # todo get the pd output
        if args.mode == "pred" or args.mode == "all":
            with torch.inference_mode():
                outputs = model(**data)

        # todo output the gt
        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        # todo output the pd
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        # todo get pd and gt together
        elif args.mode == "all" and "gt_bboxes_3d" in data and "boxes_3d" in outputs[0]:
            # todo get gt first
            bboxes_gt = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels_gt = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels_gt, args.bbox_classes)
                bboxes = bboxes_gt[indices]
                labels = labels_gt[indices]

            bboxes_gt[..., 2] -= bboxes_gt[..., 5] / 2
            # todo get the pd next
            bboxes_pd = outputs[0]["boxes_3d"].tensor.numpy()
            scores_pd = outputs[0]["scores_3d"].numpy()
            labels_pd = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices_pd = np.isin(labels_pd, args.bbox_classes)
                bboxes_pd = bboxes_pd[indices_pd]
                scores_pd = scores_pd[indices_pd]
                labels_pd = labels_pd[indices_pd]

            if args.bbox_score is not None:
                indices_pd = scores_pd >= args.bbox_score
                bboxes_pd = bboxes_pd[indices_pd]
                scores_pd = scores_pd[indices_pd]
                labels_pd = labels_pd[indices_pd]

            bboxes_pd[..., 2] -= bboxes_pd[..., 5] / 2
            # todo 将结果进行拼接
            bboxes = np.concatenate((bboxes_pd, bboxes_gt), axis=0)
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
            labels = np.concatenate((labels_pd, labels_gt), axis=0)
        else:
            bboxes = None
            labels = None

        # todo vloume still exist some issue
        # if "img" in data:
        #     for k, image_path in enumerate(metas["filename"]):
        #         image = mmcv.imread(image_path)
        #         volumw_cal(
        #             os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
        #             image,
        #             bboxes=bboxes,
        #             labels=labels,
        #             transform=metas["lidar2image"][k],
        #             classes=cfg.object_classes,
        #         )
        #         exit()
        # todo area
        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            area_cal(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )
    print(gt_dict)
    for (key, value) in area_dict.items():
        area_dict[key] = value / gt_dict[key]
    for (key, value) in dist_dict.items():
        dist_dict[key] = value / gt_dict[key]
    print(area_dict)
    print(dist_dict)


if __name__ == "__main__":
    main()



