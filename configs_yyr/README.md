# configs_yyr — 配置说明与继承关系

本目录是 Dynamic-Adjustment-Fusion（基于 MIT BEVFusion 二次开发）的全部实验配置，共 19 个 yaml，分为 **根目录数据配置（6 个）** 与 **三个模型配置子目录（13 个）**：

```
configs_yyr/
├── default*.yaml          # 数据管线 / 训练流程配置（6 个，见 §2）
├── centerhead/            # 纯相机 CenterHead 基线（3 个，见 §3）
├── distill/               # ResNet-18 学生蒸馏线（2 个，见 §4）
└── transfusion/           # 融合检测主线消融（8 个，见 §5）
```

## 1. 配置加载与"继承"机制

训练入口 `tools/train.py` 使用 torchpack 的 `configs.load(path, recursive=True)`：

1. **层级合并**：沿目录树向上，自动合并所有祖先目录中名为 `default.yaml` 的文件，最后叠加命令行指定的叶子 yaml（同名键后者覆盖前者）。
   例如运行 `configs_yyr/transfusion/fusion.yaml` 时，实际生效配置 = `configs_yyr/default.yaml`（数据管线/seed/fp16/日志）+ `fusion.yaml`（模型/优化器）。
2. **`${...}` 插值**：合并完成后由 `mmdet3d/utils/config.py` 的 `recursive_eval` 统一求值（本质是 `eval`）。因此叶子 yaml 里覆盖 `voxel_size` / `point_cloud_range` / `image_size`，会同步作用于数据管线中的体素化、范围过滤、图像增广等所有引用处。
3. **只有 `default.yaml` 会被自动加载**。根目录其余 5 个 `default_*.yaml` 是"手动替换 default.yaml 后再跑"的备选快照，不会被任何命令自动读取。
4. 命令行 `--key value` 形式的 opts 可继续覆盖任意键（见根目录 note_yyr.txt 中的用法）。

> 注意：`${}` 插值即 `eval`，配置文件等同可执行代码，不要加载不可信来源的 yaml。

## 2. 根目录数据配置（6 个）

### 演化谱系

```
default.yaml                     相机预训练阶段基座
   │  数据根→trainval、sweeps 9→10、gridmask 0→0.5、加载相机预训练权重
   ▼
default_256_704.yaml             融合阶段基座（256×704）
   ├──► default_384_1056.yaml    仅改 2 行：分辨率 384×1056 + resize 增广按 1.5× 缩放
   │
   └──► default_st.yaml          + RandomFlip3D 强翻转参数（sync_2d / 水平垂直各 0.5）
           │  + test 集支持
           ▼
        default_test.yaml        test_pipeline 去 GT、data.test 指向官方 test 集
           │  删翻转参数 + train/val 合并训练
           ▼
        default_trainvaltest.yaml  全量 24 epoch 重训 + test 提交（最终形态）
```

### 逐个说明

| 文件                          | 内容                                                                                                                                                                                                                 | 用途                                                          |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `default.yaml`              | 基准数据配置：`alldata/` 数据根、±51.2 / voxel 0.1、256×704、sweeps 9、gridmask 概率 0、`load_from_camera: null`、20 epoch、CustomEpochBasedRunner                                                             | 相机预训练阶段（与 weight/camera_pretrain 的训练日志吻合）    |
| `default_256_704.yaml`      | 相对 default：`trainval/` 数据根、sweeps 10、gridmask 0.5、`load_from_camera: pretrained/epoch_20.pth`、use_dim 改列表写法                                                                                       | 融合训练 256×704（与 weight/fusion256 日志的数据设置吻合）   |
| `default_384_1056.yaml`     | 相对 256_704 仅 2 行：`image_size: [384,1056]`、resize 增广 [0.38,0.55]/0.48 → [0.57,0.825]/0.72                                                                                                                  | 融合训练 384×1056                                            |
| `default_st.yaml`           | 相对 256_704 仅给 RandomFlip3D 加 `sync_2d: True` + 水平/垂直翻转各 0.5（st = 强增广）                                                                                                                             | 增广消融。**当前代码下会崩，见 §6**                    |
| `default_test.yaml`         | 继承 st 全部；+ `testdataset_root`；test_pipeline 去掉 GT 加载（`with_bbox_3d: False`、Collect 去 gt 键）；`data.test` 指向 `nuscenes_infos_test.pkl`                                                        | nuScenes test 集推理/提交。**同样带坏翻转参数，见 §6** |
| `default_trainvaltest.yaml` | 删掉 st/test 的翻转参数；train 的 ann_file 变为 **train+val 两个 pkl 列表**；`total_epochs: 24` + 标准 EpochBasedRunner；新增带 GT 的独立 `val_pipeline`（val 已并入训练，验证需单独管线）；保留 data.test | 最终刷榜：全量数据重训 + test 提交                            |

### 关键自定义字段（代码消费处）

- `load_from` / `load_from_camera`：`mmdet3d/apis/train.py` 先后两次 `load_checkpoint`，实现 LiDAR 分支与相机分支分别用各自预训练权重初始化。
- `CustomEpochBasedRunner`（`mmdet3d/runner/epoch_based_runner.py`）：每 epoch 调 `dataset.set_epoch`，支持 GT-paste 增广的 epoch 调度（`gt_paste_stop_epoch`）。
- `fp16.loss_scale.growth_interval: 2000`：动态 loss scale，全部实验均开 fp16。

## 3. centerhead/ — 纯相机 CenterHead 基线（3 个）

共享骨架：`type: BEVFusion` 但 **`encoders.lidar: null`、`fuser: null`**；
Swin-T → GeneralizedLSSFPN → 视角变换 → 80ch BEV → GeneralizedResNet + LSSFPN → **CenterHead**（6 任务分组、circle/rotate NMS、code_size 9）。
优化器与 transfusion 系不同：**AdamW lr 2e-4 + cyclic 策略**（target_ratio 5），相机 backbone lr×0.1，本地覆盖弱化 `augment3d`（scale 0.95–1.05、rot ±22.5°、无平移）。

| 文件                   | 视角变换                                  | BEV 网格                        | 分辨率              | backbone 初始化             | 定位                                                                                            |
| ---------------------- | ----------------------------------------- | ------------------------------- | ------------------- | --------------------------- | ----------------------------------------------------------------------------------------------- |
| `camera.yaml`        | LSSTransform（无深度估计）                | ±51.2 / 0.4，grid 1024         | 256×704            | ImageNet 版 swin_tiny       | 原始相机基线，配 default.yaml 数据体系                                                          |
| `camera_fusion.yaml` | **DepthLSSTransform**（带深度分布） | **±54 / 0.3，grid 1440** | **384×1056** | 同上                        | **与融合阶段网格对齐**的相机预训练，产出可被融合实验 `load_from_camera` 直接加载的权重  |
| `camera_inter.yaml`  | LSSTransform                              | ±51.2，grid 1024               | 256×704            | **nuImages 版** swint | 仅 3 处差异：type→BEVFusion_inter、换 nuImages 初始化、`max_epochs: 6`；交互实验的相机侧对照 |

关系：`camera.yaml` 是基座；`camera_fusion` 改"网格 + 深度变换"服务融合预训练；`camera_inter` 改"模型类 + 初始化"服务交互消融。

对应代码：`mmdet3d/models/fusion_models/bevfusion.py`、`bevfusion_inter.py`；`vtransforms/lss.py`、`depth_lss.py`；`heads/bbox/centerpoint.py`。

## 4. distill/ — ResNet-18 学生蒸馏线（2 个）

两者都是完整融合模型（相机+LiDAR，±54/0.075，6 epoch，cosine 1e-4），学生相机 backbone 均为 **ResNet-18**（FPN 输入 [128,256,512]，norm_eval=True），其余结构与 fusion.yaml 相同。

| 文件           | model type        | 说明                                                                                                                                            |
| -------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `r18.yaml`   | `BEVFusion`     | **无蒸馏对照组**：R18 学生单独训练，量化换小 backbone 的精度损失                                                                          |
| `D_r18.yaml` | `Dis_bevfusion` | 在学生之外完整复制一份教师：`tea_encoder / tea_fuser / tea_decoder / tea_heads`（Swin-T 相机 + SparseEncoder），`tea_pretrained` 加载后冻结 |

蒸馏机制（在代码中，yaml 不体现）：teacher 于 `no_grad` 下跑全链路，学生在 **encoder 输出 / fuser 输出 / decoder 输出 / head 预测** 四层做 L2 特征模仿 + heatmap 分数对齐。loss 权重硬编码于 `mmdet3d/models/fusion_models/dis_bevfusion.py`（encoders 10、fuser 10、decoder 100；共模/特异分解蒸馏分支已注释未启用）。`r18` 与 `D_r18` 的指标差即蒸馏收益。

## 5. transfusion/ — 融合检测主线（8 个）

共同骨架：Swin-T + DepthLSS（80ch camera BEV）⊕ SparseEncoder（256ch lidar BEV）→ ConvFuser → SECOND + SECONDFPN → TransFusion 系检测头（200 proposals、匈牙利匹配、1 层 decoder）。
统一超参：±54 / voxel 0.075、grid 1440、**6 epoch**（cubevfusion 为 12）、AdamW 1e-4、cosine + 500 iter warmup、grad_clip 35、cyclic momentum。
**8 个文件互为消融，彼此差异通常只有 2–4 处**（type / 检测头 / fuser 通道 / lr 策略）。

| 文件                       | model type              | 检测头                                     | 关键差异                                                                | 实验目的                                                                                                             |
| -------------------------- | ----------------------- | ------------------------------------------ | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `lidar.yaml`             | BEVFusion（无相机分支） | TransFusionHead                            | `fuser: null`、cyclic lr、`gt_paste_stop_epoch: 15`                 | **TransFusion-L 纯 LiDAR 基线**，产出 lidar-only-det.pth 供所有融合实验初始化                                  |
| `fusion.yaml`            | BEVFusion               | TransFusionHead                            | fuser [80,256]                                                          | **原版 BEVFusion 基线**（复现锚点）                                                                            |
| `fusion_inter.yaml`      | BEVFusion_inter         | TransFusionHead                            | 仅 type 一行                                                            | 编码器侧交互消融（交互算子当前在代码中全被注释，实际等价基线）                                                       |
| `fusion_head_inter.yaml` | BEVFusion_inter         | **TransFusionHead_inter**            | + encoder lr×0.1                                                       | 解码器侧"语义×几何"样本重加权消融（reweight 已实现但默认关闭）                                                      |
| `fusion_noise.yaml`      | BEVFusion_noise         | TransFusionHead                            | 融合前区域掩码：camera 留外环、lidar 留中心 25:155                      | 模态缺失鲁棒性                                                             |
| `deepbevfusion.yaml` ⭐  | DeepBEVFusion（MMF11）  | **DeepTransFusionHead**（提案数×2） | fuser **[256,256]**（MMF11 先把 camera 80→256）+ encoder lr×0.1 | **主线**：空间注意力 × SKConv 通道注意力 × 高斯置信图的融合前动态加权，附一致性/双模态 heatmap/mask 辅助损失 |
| `cubevfusion.yaml`       | CUBEVFusion             | **CuTransFusionHead**                | **max_epochs 12**；fuser 仍 [80,256]                              | 共模/特异模态分解 + Sinkhorn 最优传输对齐 + 重建/正交/相似约束（分解输出未接主路径，只贡献辅助损失）                 |
| `dqnbevfusion.yaml`      | DqnBEVFusion            | TransFusionHead_inter                      | fuser [256,256] + encoder lr×0.1                                       | 跨模态一致性损失 + camera/lidar/fuser 三路 heatmap 辅助监督。**名字遗留，无任何 DQN/强化学习**                 |

消融逻辑链：

```
lidar（单模态锚点）→ fusion（融合锚点）
   → fusion_inter / fusion_head_inter（交互位置消融：编码器 vs 解码器）
   → fusion_noise（鲁棒性）
   → dqnbevfusion（一致性监督）→ cubevfusion（特征分解）
   → deepbevfusion（集大成的 MMF11 动态加权）
```

对应代码：`mmdet3d/models/fusion_models/` 下同名文件；检测头在 `mmdet3d/models/heads/bbox/`（transfusion.py / transfusion_inter.py / deeptransfusion.py / cutransfusion.py）；ConvFuser 在 `mmdet3d/models/fusers/conv.py`。

## 6. 已知问题

1. **`default_st.yaml` / `default_test.yaml` 与代码不兼容**：给 RandomFlip3D 传 `sync_2d / flip_ratio_bev_*`，但本 fork 的 RandomFlip3D（`mmdet3d/datasets/pipelines/transforms_3d.py`）无 `__init__`、不接收任何参数（内部固定 50/50 翻转），build pipeline 即 TypeError。`default_trainvaltest.yaml` 已删掉这些参数（可视为修正版）。
2. **配置与已发布权重存在漂移**：weight/ 下训练日志显示融合实验为 12 epoch、头为 `TransFusionHead_inter`；现存 yaml 为 6 epoch、头为 `DeepTransFusionHead` 等。**现存 yaml 并非训练时原样**，用 ModelScope 权重（learnai2/3D-Detection-model）复现指标时，应以各权重日志头部的 Config 为准（尤其检测头类型与提案数，否则推理行为不同、指标无法精确复现）。
3. **路径均为云平台绝对路径**（`/root/paddlejob/...`），本机运行需替换 `dataset_root`、各 `init_cfg.checkpoint`、`load_from*`、`tea_pretrained`。
4. 部分关键超参不在 yaml 中：蒸馏 loss 权重、MMF 变体选择（MMF→MMF11）、交互算子开关、reweight 开关均在代码里硬编码或靠注释切换，消融记录不完整。

## 7. 典型运行命令

```bash
# 相机预训练（centerhead 基线）
torchpack dist-run -np 8 python tools/train.py configs_yyr/centerhead/camera.yaml

# 融合主线（自动合并 configs_yyr/default.yaml 数据配置）
torchpack dist-run -np 8 python tools/train.py configs_yyr/transfusion/deepbevfusion.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
    --load_from pretrained/lidar-only-det.pth

# 测试
torchpack dist-run -np 8 python tools/test.py configs_yyr/transfusion/fusion.yaml \
    weight/fusion384/epoch_10.pth --eval bbox
```

更多历史命令见根目录 `note_yyr.txt`；权重与指标见 `weight/README.md`（已同步至 ModelScope 仓库 README）。
