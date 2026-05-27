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

## 评测

```bash
cd /home/pc/gxy/PSOD
python eval.py --config /home/pc/gxy/PSOD/configs/nwpu_vhr10/deim_hgnetv2_n_pseudo_box.yml --resume /path/to/checkpoint.pth

或

python -m psod eval --config /home/pc/gxy/PSOD/configs/nwpu_vhr10/deim_hgnetv2_n_pseudo_box.yml --resume /path/to/checkpoint.pth
```
