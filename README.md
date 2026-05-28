# PSOD: Point-Supervised Object Detection (单点监督目标检测)

本项目提供了一个完整的"单点监督目标检测"基线（Baseline）。核心流程为：利用 SAM (Segment Anything Model) 将单点标注转化为伪边界框（Pseudo-boxes），随后使用 DEIM 目标检测器进行模型训练与评估。

---

## 1. 数据准备 (Data Preparation)

本项目默认使用 **NWPU-VHR-10** 遥感数据集。
* **图片目录**：`/home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/images`
* **原始标注**：`/home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/annotations/train.json`

### 1.1 划分真实训练/验证集 (极其重要)

在开始实验前，**必须**从带有标注的训练集中划分出真实的验证集（Ground Truth Validation Set）：

```bash
cd /home/pc/gxy/PSOD

python scripts/split_coco_train_val.py \
  --coco-json /home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/annotations/train.json \
  --out-dir /home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/annotations \
  --ratio 0.2 \
  --seed 42
```

* **输出文件**：
  * 划分后的训练集标注：`.../annotations/train_gt.json`
  * 划分后的验证集标注：`.../annotations/val_gt.json` *(后续的评估请务必使用此文件)*

---

## 2. 使用方法 (Usage)

项目统一入口为 `run.py`，支持三种运行模式：`train`、`val`、`pseudo`。

### 2.1 基本用法

```bash
cd /home/pc/gxy/PSOD

# 训练模式
python run.py train -c configs/nwpu_vhr10/deim_hgnetv2_n_pseudo_box.yml

# 验证模式
python run.py val -c configs/nwpu_vhr10/deim_hgnetv2_n_pseudo_box.yml -r outputs/xxx/best_stg1.pth

# 伪标签生成模式
python run.py pseudo --coco-json /path/to/train.json --image-root /path/to/images --out-dir /path/to/output
```

### 2.2 完整参数说明

#### train / val 模式参数

| 参数 | 说明 |
|------|------|
| `-c` / `--config` | YAML 配置文件路径（必填） |
| `-r` / `--resume` | 从 checkpoint 恢复训练 |
| `-t` / `--tuning` | 从 checkpoint 微调 |
| `-d` / `--device` | 设备（如 `cuda:0`） |
| `--seed` | 随机种子 |
| `--use-amp` | 启用混合精度训练 |
| `--output-dir` | 覆盖输出目录 |
| `-u` / `--update` | 动态覆盖 YAML 配置项 |
| `-p` / `--path` | ONNX/Engine 模型路径（val 模式） |
| `--onnx-mode` | 模型模式：`det` / `mask` |

#### pseudo 模式参数

| 参数 | 说明 |
|------|------|
| `--coco-json` | COCO 标注文件路径（必填） |
| `--image-root` | 图片目录路径（必填） |
| `--out-dir` | 输出目录 |
| `--output-name` | 输出文件名（默认 `train_pseudo.json`） |
| `--weights` | SAM 权重路径 |
| `--device` | 设备 |
| `--max-images` | 最大处理图片数 |

### 2.3 使用 `--update` 覆盖配置

当需要临时修改配置（如更换验证集标注文件）时，使用 `-u` 参数：

```bash
python run.py val \
  -c configs/nwpu_vhr10/deim_hgnetv2_n_pseudo_box.yml \
  -r outputs/nwpu_vhr10_pseudo_box_fullscale3/best_stg1.pth \
  -u val_dataloader.dataset.ann_file=/home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/annotations/val_gt.json \
  -u val_dataloader.dataset.img_folder=/home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/images
```

---

## 3. 生成伪边界框 (Generate Pseudo-boxes)

使用 SAM 根据单点坐标生成伪边界框。该脚本会读取原始标注（提取中心点），并输出包含伪框尺寸的全新 COCO 格式 JSON。

```bash
cd /home/pc/gxy/PSOD

python run.py pseudo \
  --coco-json /home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/annotations/train.json \
  --image-root /home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/images \
  --out-dir /home/pc/gxy/PSOD \
  --output-name train_pseudo.json
```

* **输出文件**：`/home/pc/gxy/PSOD/train_pseudo.json`（将作为下一步训练的监督信号）。

---

## 4. 模型训练 (Training)

将生成的 `train_pseudo.json` 配置到您的 YAML 文件中，启动 DEIM 模型的全监督训练。

```bash
cd /home/pc/gxy/PSOD

python run.py train -c configs/nwpu_vhr10/deim_hgnetv2_n_pseudo_box.yml
```

### DEIM 两阶段权重切换机制 (Stage 1 -> Stage 2)

本框架支持先进的两阶段训练策略（如训练末期关闭数据增强）：

1. **Stage 1 (常规训练阶段)**：训练过程中会持续保存 `last.pth`，并在验证指标提升时保存 `best_stg1.pth`。
2. **Stage 2 (微调阶段)**：当 `epoch == stop_epoch` 时，引擎会自动触发阶段切换。
   * 优先尝试加载 `best_stg1.pth` 作为 Stage 2 的起点。
   * 若不存在则回退加载 `last.pth`；若皆不存在则不进行加载。
3. **保存**：Stage 2 期间指标提升时，会独立保存为 `best_stg2.pth`。

*(如需进行最小复现测试，可运行: `python scripts/repro_deim_stage2_resume.py`)*

---

## 5. 模型评估 (Evaluation)

测试训练好的权重。**请确保 YAML 配置文件中的 `val_dataloader.dataset.ann_file` 指向的是您在第 1 步生成的 `val_gt.json`，而非空的 `test.json`。**

```bash
cd /home/pc/gxy/PSOD

python run.py val \
  -c configs/nwpu_vhr10/deim_hgnetv2_n_pseudo_box.yml \
  -r outputs/nwpu_vhr10_pseudo_box_fullscale/best_stg1.pth
```

### 常见故障排查：指标全部为 `-1.000`

如果您在运行评估时，终端打印的 COCO AP/AR 指标全部为 `-1.0`（并伴随 TIDE `division by zero` 报错）：

* **原因**：验证集 JSON 文件中没有真实标注框（GT），即 `"annotations": []`。COCOeval 在计算 Precision/Recall 时分母为零，强制返回 `-1`。
* **解决方案**：检查配置文件的验证集路径。如果您手中只有一份带有标注的训练集，请务必先使用本文档第 1 节的 `split_coco_train_val.py` 脚本切分出真实的验证集 (`val_gt.json`)，并使用它进行评估。

---

## 6. 可视化 (Visualization)

### 6.1 可视化模型推理结果

使用训练好的权重对图片进行实时推理并可视化：

```bash
python visualize.py \
  -c configs/nwpu_vhr10/deim_hgnetv2_n_pseudo_box.yml \
  -r outputs/nwpu_vhr10_pseudo_box_fullscale/best_stg1.pth \
  --image /path/to/image_or_directory \
  --output vis_output \
  --conf-thresh 0.5
```

### 6.2 可视化已有预测结果

可视化 `eval` 生成的 `pred.json` 文件：

```bash
python vis_pred.py \
  --pred-json outputs/nwpu_vhr10_pseudo_box_fullscale_eval/pred.json \
  --ann-file /home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/annotations/val_gt.json \
  --images-dir /home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/images \
  --out-dir vis_pred_output \
  --score-thr 0.3
```

---

## 7. 项目结构

```
PSOD/
├── run.py                  # 统一入口（train/val/pseudo）
├── visualize.py            # 模型推理可视化
├── vis_pred.py             # 预测结果可视化
├── configs/                # 配置文件目录
│   └── nwpu_vhr10/
│       └── deim_hgnetv2_n_pseudo_box.yml
├── psod/                   # 核心代码包
│   ├── sam/                # SAM 模型代码
│   ├── sam_point_adapter.py
│   ├── pseudo/             # 伪标签生成逻辑
│   └── deim/               # DEIM 检测器代码
├── scripts/                # 工具脚本
│   ├── split_coco_train_val.py
│   └── repro_deim_stage2_resume.py
├── weights/                # 模型权重（不上传 git）
└── outputs/                # 训练输出（不上传 git）
```
