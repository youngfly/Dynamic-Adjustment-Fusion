# Dynamic-Adjustment-Fusion

**Paper**: [Explore the LiDAR-Camera Dynamic Adjustment Fusion for 3D Object Detection](https://arxiv.org/pdf/2407.15334) (arXiv:2407.15334)

![示例图片](DAF.jpg "这是示例图片的标题")

基于 [BEVFusion](https://github.com/mit-han-lab/bevfusion)（MIT，nuScenes 相机+LiDAR 统一 BEV 空间多模态融合）的二次开发项目，核心研究方向是 **融合前的跨模态动态权重调整**：在 camera BEV 与 lidar BEV 送入 ConvFuser 之前，用可学习的空间注意力 × 通道注意力（SKConv）× GT 监督的高斯置信图对两模态特征逐位置加权（MMF11 模块），并辅以跨模态一致性损失与双模态 heatmap 监督。

围绕主线还包含多条实验线：编码器/解码器侧交互消融、模态缺失鲁棒性（noise mask）、共模/特异特征分解（CUBEVFusion + Sinkhorn 最优传输）、跨模态一致性监督（DqnBEVFusion，名字遗留、无强化学习）、以及 Swin-T → ResNet-18 的四层特征蒸馏。

> 上游 BEVFusion 的原始说明保留在 [README_base.md](README_base.md)。

## 文档导航

| 文档                                                                               | 内容                                                                       |
| ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **[configs_yyr/README.md](configs_yyr/README.md)**                            | 全部 19 个实验配置的含义、继承/合并机制、消融关系链、已知问题清单          |
| **[weight/README.md](weight/README.md)**                                      | 已训练权重与日志：各分辨率 mAP/NDS、ModelScope 下载方式、复现注意事项      |
| [note_yyr.txt](note_yyr.txt)                                                        | 历史实验命令原始记录（本文 §训练/测试 已整理收录）                        |
| [README_base.md](README_base.md)                                                    | 上游 BEVFusion 原始 README（环境/数据准备的权威参考）                      |
| [ModelScope 模型仓库](https://www.modelscope.cn/models/learnai2/3D-Detection-model) | 权重在线发布页（camera_pretrain + fusion256/384/512，约 2.4 GB）           |
| [论文 arXiv:2407.15334](https://arxiv.org/pdf/2407.15334)                           | Explore the LiDAR-Camera Dynamic Adjustment Fusion for 3D Object Detection |

## 目录结构

```
Dynamic-Adjustment-Fusion/
├── configs_yyr/            # 本项目全部实验配置（详见其 README.md）
│   ├── default*.yaml       #   数据管线/训练流程配置（6 个，手动切换）
│   ├── centerhead/         #   纯相机 CenterHead 基线（3 个）
│   ├── distill/            #   ResNet-18 学生蒸馏线（2 个）
│   └── transfusion/        #   融合检测主线消融（8 个）
├── configs/                # 上游 BEVFusion 原版配置
├── mmdet3d/
│   ├── models/fusion_models/   # 自定义融合模型（DeepBEVFusion/MMF11、CUBEVFusion 等）
│   ├── models/heads/bbox/      # 检测头变体（TransFusionHead_inter / Deep / Cu）
│   ├── models/utils_yyr/       # 自定义工具（LCAB 局部注意力 CUDA 算子等）
│   └── ...                     # 其余为上游 mmdet3d fork
├── tools/                  # train / test / visualize / statical 等入口
├── weight/                 # 已训练权重 + 训练日志（详见其 README.md）
└── docker/                 # 上游 Dockerfile
```

## 环境安装

依赖与上游一致（详见 [README_base.md](README_base.md)）：

- Python >= 3.8, < 3.9
- PyTorch >= 1.9, <= 1.10.2
- mmcv = 1.4.0，mmdetection = 2.20.0
- OpenMPI 4.0.4 + mpi4py 3.0.3（torchpack 需要）
- Pillow 8.4.0、tqdm、[torchpack](https://github.com/mit-han-lab/torchpack)、nuscenes-dev-kit

编译安装（`setup.py develop` 会编译自定义 CUDA 扩展：bev_pool、voxelization 等）：

```bash
python setup.py develop
```

或使用 Docker：

```bash
cd docker && docker build . -t bevfusion
nvidia-docker run -it -v `pwd`/../data:/dataset --shm-size 16g bevfusion /bin/bash
```

注意：`mmdet3d/models/utils_yyr/ops/locatt_ops` 的局部注意力 CUDA 算子为 JIT 现场编译，交互实验线需要可用 CUDA 环境。

## 数据准备

按 [mmdetection3d nuScenes 说明](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/datasets/nuscenes_det.md) 下载并预处理 nuScenes（检测集 + map 扩展）。**info 文件请用本仓库重新生成**（fork 早于 mmdet3d 坐标系重构，不能直接用其成品 pkl）：

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

完成后目录形如：

```
data/nuscenes/
├── maps/  samples/  sweeps/
├── v1.0-trainval/  v1.0-test/
├── nuscenes_database/
├── nuscenes_infos_train.pkl / _val.pkl / _test.pkl
└── nuscenes_dbinfos_train.pkl
```

**本机运行前必改**：configs_yyr 内所有 `dataset_root`、`init_cfg.checkpoint`、`load_from*`、`tea_pretrained` 均为训练时云平台绝对路径（`/root/paddlejob/...`），需替换为本地路径。数据配置的选择与切换方式见 [configs_yyr/README.md §1–§2](configs_yyr/README.md)。

## 预训练与已训练权重

**上游预训练权重**（训练初始化用，需要 `swint-nuimages-pretrained.pth`（相机）与 `lidar-only-det.pth`（LiDAR 分支））：

```bash
./tools/download_pretrained.sh
```

**本项目已训练权重**（nuScenes val，单模型无 TTA）：

| 模型                                 | 分辨率    | val mAP / NDS                          |
| ------------------------------------ | --------- | -------------------------------------- |
| camera_pretrain（纯相机 CenterHead） | 256×704  | 见 [weight/README.md](weight/README.md) |
| DeepBEVFusion（fusion256）           | 256×704  | 0.6914 / 0.7191                        |
| DeepBEVFusion（fusion384）           | 384×1056 | 0.6995 / 0.7233                        |
| DeepBEVFusion（fusion512）           | 512×1408 | 0.7095 / 0.7223                        |

下载方式（SDK / git-lfs / 网页）、各 epoch 指标细节与**复现注意事项（检测头漂移等）**见 **[weight/README.md](weight/README.md)**，在线仓库：[learnai2/3D-Detection-model](https://www.modelscope.cn/models/learnai2/3D-Detection-model)。

## 训练

配置加载机制（torchpack 自动合并 `configs_yyr/default.yaml` + 叶子 yaml）与全部配置含义见 **[configs_yyr/README.md](configs_yyr/README.md)**。以下命令整理自 [note_yyr.txt](note_yyr.txt)。

### 1. 基线三件套（官方流程）

**LiDAR-only（TransFusion-L）**，产出 `lidar-only-det.pth`：

```bash
torchpack dist-run -np 8 python tools/train.py configs_yyr/transfusion/lidar.yaml
```

**纯相机 CenterHead**：

```bash
torchpack dist-run -np 8 python tools/train.py configs_yyr/centerhead/camera.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

**BEVFusion 融合基线**（双分支分别初始化）：

```bash
torchpack dist-run -np 8 python tools/train.py configs_yyr/transfusion/fusion.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
    --load_from pretrained/lidar-only-det.pth
```

### 2. 主线与消融（编码器侧 / 解码器侧 / 鲁棒性）

**主线：DeepBEVFusion + MMF11 动态加权**：

```bash
torchpack dist-run -np 8 python tools/train.py configs_yyr/transfusion/deepbevfusion.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
    --load_from pretrained/lidar-only-det.pth
```

**编码器侧交互**：

```bash
torchpack dist-run -np 8 python tools/train.py configs_yyr/transfusion/fusion_inter.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
    --load_from pretrained/lidar-only-det.pth
```

**解码器侧样本重加权**（`--load_from` 也可换成 `pretrained/lidar-only-det.pth`）：

```bash
torchpack dist-run -np 8 python tools/train.py configs_yyr/transfusion/fusion_head_inter.yaml \
    --load_from pretrained/bevfusion-det.pth
```

**模态区域掩码鲁棒性**：

```bash
torchpack dist-run -np 8 python tools/train.py configs_yyr/transfusion/fusion_noise.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
    --load_from pretrained/lidar-only-det.pth
```

**从自己的 checkpoint 续训**：

```bash
torchpack dist-run -np 8 python tools/train.py configs_yyr/transfusion/fusion_inter.yaml \
    --resume_from runs/fusion_F3_cc+cl_ll+lc/epoch_3.pth
```

### 3. 蒸馏（ResNet-18 学生）

**学生对照组（无蒸馏）**：

```bash
torchpack dist-run -np 8 python tools/train.py configs_yyr/distill/r18.yaml \
    --load_from pretrained/lidar-only-det.pth
```

**四层特征蒸馏（Swin-T 教师冻结）**：

```bash
torchpack dist-run -np 8 python tools/train.py configs_yyr/distill/D_r18.yaml \
    --load_from pretrained/lidar-only-det.pth
```

### 4. 多卡分组 / 单卡调试

**8 卡机拆两组 4 卡并行跑不同实验**：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=28456 tools/dist_train.sh configs_yyr/transfusion/fusion.yaml 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=28457 tools/dist_train.sh configs_yyr/transfusion/fusion.yaml 4
```

**单卡调试**：

```bash
torchpack dist-run -np 1 python tools/train.py configs_yyr/transfusion/fusion_inter.yaml \
    --load_from pretrained/bevfusion-det.pth
torchpack dist-run -np 1 python tools/train.py configs_yyr/centerhead/camera_inter.yaml \
    --load_from pretrained/camera-only-det.pth
```

## 测试与评估

**官方预训练模型**：

```bash
torchpack dist-run -np 8 python tools/test.py configs_yyr/transfusion/fusion.yaml \
    pretrained/bevfusion-det.pth --eval bbox
```

**自己训练的模型**（`runs/` 下）：

```bash
torchpack dist-run -np 8 python tools/test.py configs_yyr/transfusion/fusion.yaml \
    runs/run-51c5d35d/epoch_6.pth --eval bbox
```

**已发布权重**（从 ModelScope 下载后；注意检测头漂移问题，见 [weight/README.md](weight/README.md)）：

```bash
torchpack dist-run -np 8 python tools/test.py configs_yyr/transfusion/fusion.yaml \
    weight/fusion384/epoch_10.pth --eval bbox
```

**可视化 / 数据统计**：

```bash
torchpack dist-run -np 8 python tools/visualize.py configs_yyr/transfusion/fusion.yaml
torchpack dist-run -np 8 python tools/statical.py configs_yyr/transfusion/fusion.yaml
```

### nuScenes 官方榜单提交（evalai）

token 从 evalai 个人账户页获取，勿提交到仓库：

```bash
evalai set_token <你的 evalai refresh token>
evalai challenge 356 phase 744 submit --file results_nusc.json --large --private
```

test 集推理需使用 `default_test.yaml` / `default_trainvaltest.yaml` 数据配置（注意前者当前含坏参数，见 [configs_yyr/README.md §6](configs_yyr/README.md)）。

## 致谢与引用

本项目基于 [BEVFusion](https://github.com/mit-han-lab/bevfusion)（mmdetection3d fork），并受 LSS、BEVDet、TransFusion、CenterPoint、DeepInteraction 等工作启发。若使用本项目代码/模型，请引用我们的论文：

```bibtex
@article{yang2024explore,
  title={Explore the LiDAR-Camera Dynamic Adjustment Fusion for 3D Object Detection},
  author={Yang, Yiran and Gao, Xu and Wang, Tong and Hao, Xin and Shi, Yifeng and Tan, Xiao and Ye, Xiaoqing and Wang, Jingdong},
  journal={arXiv preprint arXiv:2407.15334},
  year={2024}
}
```

若使用上游代码/模型，请引用：

```bibtex
@article{liu2022bevfusion,
  title={BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation},
  author={Liu, Zhijian and Tang, Haotian and Amini, Alexander and Yang, Xingyu and Mao, Huizi and Rus, Daniela and Han, Song},
  journal={arXiv},
  year={2022}
}
```
