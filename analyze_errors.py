import json

def calculate_iou(box1, box2):
    """计算两个框的 IoU (Intersection over Union)。框格式: [x, y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 计算交集坐标
    ixmin = max(x1, x2)
    iymin = max(y1, y2)
    ixmax = min(x1 + w1, x2 + w2)
    iymax = min(y1 + h1, y2 + h2)
    
    iw = max(ixmax - ixmin, 0.)
    ih = max(iymax - iymin, 0.)
    inters = iw * ih
    
    uni = (w1 * h1) + (w2 * h2) - inters
    return inters / uni if uni > 0 else 0

def main():
    # 替换为你实际的路径
    gt_path = "/home/pc/gxy/Point-DEIM/dataset/nwpu_vhr10/annotations/instances_train.json"
    pseudo_path = "/home/pc/gxy/PSOD/train_pseudo.json"

    print(f"Loading Ground Truth: {gt_path}")
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
        
    print(f"Loading Pseudo Labels: {pseudo_path}")
    with open(pseudo_path, 'r') as f:
        pseudo_data = json.load(f)

    # 用字典存储，按 annotation ID 对齐
    gt_dict = {ann['id']: ann for ann in gt_data['annotations']}
    
    total_iou = 0
    valid_boxes = 0
    
    over_seg_count = 0  # 过度分割 (SAM 框太大)
    under_seg_count = 0 # 不完全分割 (SAM 框太小)
    terrible_count = 0  # IoU 极低的烂框

    for pseudo_ann in pseudo_data['annotations']:
        ann_id = pseudo_ann['id']
        if ann_id in gt_dict:
            gt_ann = gt_dict[ann_id]
            
            iou = calculate_iou(gt_ann['bbox'], pseudo_ann['bbox'])
            total_iou += iou
            valid_boxes += 1
            
            # 分析错误类型
            gt_area = gt_ann['area']
            pseudo_area = pseudo_ann['area']
            
            if iou < 0.3:
                terrible_count += 1
            if pseudo_area > gt_area * 2.0:
                over_seg_count += 1
            elif pseudo_area < gt_area * 0.5:
                under_seg_count += 1

    mean_iou = total_iou / valid_boxes
    
    print("\n" + "="*40)
    print("📊 SAM 伪标签质量分析报告 (Error Analysis)")
    print("="*40)
    print(f"总计分析目标数 : {valid_boxes}")
    print(f"平均 IoU (mIoU) : {mean_iou:.4f}  <-- 这个值决定了你单点监督的底线性能")
    print("-" * 40)
    print("🚨 典型错误模式统计:")
    print(f"1. 极低质量框 (IoU < 0.3) : {terrible_count} 个 (占 {terrible_count/valid_boxes*100:.1f}%)")
    print(f"2. 框太大 (包含过多背景)   : {over_seg_count} 个 (占 {over_seg_count/valid_boxes*100:.1f}%)")
    print(f"3. 框太小 (只框住部分结构) : {under_seg_count} 个 (占 {under_seg_count/valid_boxes*100:.1f}%)")
    print("="*40)
    print("💡 下一步建议: 根据以上比例最高的错误，在 points_to_pseudoboxes.py 中加入对应的过滤策略！")

if __name__ == "__main__":
    main()