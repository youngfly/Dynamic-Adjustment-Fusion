# weight/ — 训练权重与训练日志

本目录存放本项目（Dynamic-Adjustment-Fusion，基于 MIT BEVFusion 二次开发）在 nuScenes trainval 上训练得到的检测权重，以及对应的完整训练日志（`.log` 文本日志 + `.log.json` 逐 iteration 记录）。

所有指标均为 **nuScenes val 集**、单模型、无 TTA 的结果，从各目录训练日志中的 `Epoch(val)` 行提取（`object/map` 即 mAP，`object/nds` 即 NDS）。

## 权重下载（ModelScope）

全部权重与训练日志已发布至 ModelScope：**https://www.modelscope.cn/models/learnai2/3D-Detection-model**

仓库内目录结构与本目录一致（`camera_pretrain/`、`fusion256/`、`fusion384/`、`fusion512/`），可任选以下方式下载：

**方式一：ModelScope SDK（推荐，支持断点续传）**

```bash
pip install modelscope
```

```python
from modelscope import snapshot_download

# 下载全部权重（约 2.4 GB）
model_dir = snapshot_download('learnai2/3D-Detection-model')

# 只下载某个子目录（如 fusion384）
model_dir = snapshot_download('learnai2/3D-Detection-model', allow_patterns=['fusion384/*'])
```

**方式二：git clone（需 git-lfs）**

```bash
git lfs install
git clone https://www.modelscope.cn/models/learnai2/3D-Detection-model.git
```

**方式三：网页手动下载**

打开仓库页面 → "模型文件" 标签页 → 进入对应子目录，点击 `.pth` 文件右侧下载按钮。

下载后把对应子目录放回本 `weight/` 目录（或任意路径），测试时用 `--load_from` / 命令行位置参数指向 `.pth` 即可，例如：

```bash
torchpack dist-run -np 8 python tools/test.py \
    configs_yyr/transfusion/deepbevfusion.yaml \
    weight/fusion384/epoch_10.pth --eval bbox
```

## 目录总览

| 目录                 | 模型                                               | 相机输入分辨率 | 保留权重                  | 该 epoch 的 val mAP / NDS         |
| -------------------- | -------------------------------------------------- | -------------- | ------------------------- | --------------------------------- |
| `camera_pretrain/` | BEVFusion_inter + CenterHead（纯相机，lidar=None） | 256×704       | `epoch_20.pth` (515 MB) | 日志仅记录到 epoch 13，见下文说明 |
| `fusion256/`       | DeepBEVFusion + TransFusionHead_inter              | 256×704       | `epoch_11.pth` (613 MB) | **0.6914 / 0.7191**         |
| `fusion384/`       | DeepBEVFusion + TransFusionHead_inter              | 384×1056      | `epoch_10.pth` (618 MB) | **0.6995 / 0.7233**         |
| `fusion512/`       | DeepBEVFusion + TransFusionHead_inter              | 512×1408      | `epoch_9.pth` (624 MB)  | **0.7095 / 0.7223**         |

三个 fusion 权重均含 MMF11 动态融合模块的辅助损失（日志中可见 `loss_consistence` / `loss_heatmap_cam` / `loss_heatmap_lid` / `loss_mask`），即"空间注意力 × SKConv 通道注意力 × 高斯置信图"的融合前动态加权方案（`mmdet3d/models/fusion_models/deepbevfusion.py` 中的 `MMF11`）。

## 各目录说明

### camera_pretrain/ — 相机分支预训练

- 配置要点：SwinTransformer (embed_dims=96, nuImages 预训练初始化) + GeneralizedLSSFPN + DepthLSSTransform，CenterHead 检测头；`point_cloud_range=[-51.2, 51.2]`、`voxel_size=0.1`（与后续 fusion 阶段的 ±54 / 0.075 不同）。
- 计划训练 20 epoch；`epoch_20.pth` 即三个 fusion 实验配置中 `load_from_camera` 所加载的相机预训练权重。
- 注意：目录内日志 `20230206_173619.log` 只记录到 epoch 13 开头（日志不完整），最后一次完整验证为 epoch 12：mAP 0.4057 / NDS 0.4004。epoch 20 的验证指标不在本目录日志中。

### fusion256/ — 融合模型，256×704

- 训练 12 epoch，fp16；初始化：`load_from=pretrained/lidar-only-det.pth`（LiDAR 分支）+ `load_from_camera=pretrained/epoch_20.pth`（即上面的相机预训练）。
- `epoch_11.pth` 对应全程最优验证：mAP 0.6914 / NDS 0.7191（epoch 12 略回落至 0.6896 / 0.7170）。

### fusion384/ — 融合模型，384×1056

- 训练 12 epoch，初始化同上。
- `epoch_10.pth` 对应全程最优验证：mAP 0.6995 / NDS 0.7233。

### fusion512/ — 融合模型，512×1408

- 训练 12 epoch，初始化同上。分两段日志：
  - `20230424_092930.log`：epoch 1–9 完整，epoch 10 中途中断；
  - `20230426_151819.log`：续训完成 epoch 10–12（最终 mAP 0.7088 / NDS 0.7229）。
- 保留的 `epoch_9.pth` 对应 mAP 0.7095 / NDS 0.7223，是全程 **mAP 最高**的 checkpoint。

## 使用方法

权重与 `configs_yyr/default_256_704.yaml` / `default_384_1056.yaml` 等数据配置及 `configs_yyr/transfusion/deepbevfusion.yaml` 模型配置配套使用。测试示例（以 fusion384 为例，需按实际配置组合数据与模型 yaml）：

```bash
torchpack dist-run -np 8 python tools/test.py \
    configs_yyr/transfusion/deepbevfusion.yaml \
    weight/fusion384/epoch_10.pth --eval bbox
```

注意事项：

1. 训练时的模型组合是 `DeepBEVFusion + TransFusionHead_inter`（以各日志头部 Config 为准）；当前仓库 `deepbevfusion.yaml` 中的头已改为 `DeepTransFusionHead`，直接加载本目录权重测试时需将头部配置改回 `TransFusionHead_inter`，否则结构不匹配。
2. 日志中的所有路径（`dataset_root`、`load_from` 等）均为训练时云平台（`/root/paddlejob/...`）的绝对路径，本机复现需自行替换。
3. 相机 backbone 为 Swin-T，其 nuImages 预训练权重（`swint-nuimages-pretrained.pth`）与 LiDAR 预训练权重（`lidar-only-det.pth`）不在本目录内，需另行下载（参见根目录 README 与 `tools/download_pretrained.sh`）。
