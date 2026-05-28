# PSOD

## 数据

- 训练集：/home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/images
- 训练标注：/home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/annotations/train.json
- 测试集：/home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-test/images
- 测试标注：/home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-test/annotations/test.json

## 伪框

```bash
cd /home/pc/gxy/PSOD
python pseudo.py \
  --coco-json /home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/annotations/train.json \
  --image-root /home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/images \
  --out-dir /home/pc/gxy/PSOD

或

python -m psod pseudo \
  --coco-json /home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/annotations/train.json \
  --image-root /home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/images \
  --out-dir /home/pc/gxy/PSOD
```

输出：/home/pc/gxy/PSOD/train_pseudo.json

## 训练

```bash
cd /home/pc/gxy/PSOD
python train.py --config /home/pc/gxy/PSOD/configs/nwpu_vhr10/deim_hgnetv2_n_pseudo_box.yml

或

python -m psod train --config /home/pc/gxy/PSOD/configs/nwpu_vhr10/deim_hgnetv2_n_pseudo_box.yml
```

### 两阶段权重切换（DEIM）

- Stage1 训练过程中会持续保存 `last.pth`，并在指标提升时保存 `best_stg1.pth`
- 进入 Stage2（epoch==stop_epoch）时优先加载 `best_stg1.pth`；若不存在则回退加载 `last.pth`；两者都不存在则跳过加载
- Stage2 指标提升时保存 `best_stg2.pth`
- 最小复现：`python /home/pc/gxy/PSOD/scripts/repro_deim_stage2_resume.py`

## 评测

```bash
cd /home/pc/gxy/PSOD
python eval.py --config /home/pc/gxy/PSOD/configs/nwpu_vhr10/deim_hgnetv2_n_pseudo_box.yml --resume /path/to/checkpoint.pth

或

python -m psod eval --config /home/pc/gxy/PSOD/configs/nwpu_vhr10/deim_hgnetv2_n_pseudo_box.yml --resume /path/to/checkpoint.pth
```

### 指标为 -1 的原因与用法

- 当 `val_dataloader.dataset.ann_file` 中没有 GT（例如 `annotations/test.json` 为空标注），COCOeval 的 `precision/recall` 会被填充为 -1（表示该项不可计算），最终汇总出来的 AP/AR 也会是 -1
- 因此评测必须指向“含 GT 标注”的 COCO json；如果你只有训练集带标注，可以先用训练集标注做 val（当前默认配置即如此），或者从训练集标注切分出一个 val_gt

### 切分 COCO 标注（train_gt/val_gt）

```bash
python /home/pc/gxy/PSOD/scripts/split_coco_train_val.py \
  --coco-json /home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/annotations/train.json \
  --out-dir /home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/annotations \
  --ratio 0.2 \
  --seed 42
```

输出：

- /home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/annotations/train_gt.json
- /home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/annotations/val_gt.json

### 最小验证（确认 val annotations > 0）

```bash
python - <<'PY'
import json
path = "/home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train/annotations/val_gt.json"
j = json.load(open(path, "r", encoding="utf-8"))
print("images:", len(j.get("images", [])))
print("annotations:", len(j.get("annotations", [])))
PY
```
