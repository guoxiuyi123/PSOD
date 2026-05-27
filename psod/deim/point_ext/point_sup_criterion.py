from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from engine.core import register
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from engine.deim.pseudo_box_memory import PseudoBoxConfig, PseudoBoxMemory

from .dinov2_vit import load_dinov2_vits14_reg4


@register()
class PointSupDEIMCriterionV2(DEIMCriterion):
    __share__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        no_weight_vfl_epoch=-1,
        ccm_params=None,
        density_recall_penalty=1.1,
        density_precision_penalty=1.3,
        mask_point_sample_ratio=8,
        density_area_aware: bool = False,
        density_area_low: float = 0.0025,
        density_area_high: float = 0.09,
        density_recall_penalty_mult_small: float = 1.0,
        density_recall_penalty_mult_large: float = 1.0,
        density_precision_penalty_mult_small: float = 1.0,
        density_precision_penalty_mult_large: float = 1.0,
        density_area_reduce: str = "median",
        dino_semantic_enable: bool = False,
        dino_semantic_ckpt: str | None = None,
        dino_semantic_mode: str = "variance",
        dino_semantic_weight: float = 0.0,
        dino_semantic_input_size: int = 560,
        dino_semantic_max_boxes: int = 64,
        dino_semantic_tau: float = 0.02,
        dino_semantic_sim_thresh: float = 0.65,
        dino_semantic_min_mask: int = 4,
        dino_semantic_min_wh: float | None = None,
        dino_semantic_prop_iters: int = 12,
        dino_semantic_hr_size: int = 0,
        dino_semantic_interval: int = 1,
        update_topk: int = 0,
        update_use_aux_outputs: bool = False,
        update_use_enc_aux_outputs: bool = False,
        update_use_pre_outputs: bool = False,
        update_burnin_epochs: int = 0,
        reg_quality_weight: str = "none",
        reg_quality_power: float = 1.0,
        pseudo_box=None,
        score_quality_mode: str = "max",
    ):
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
            reg_max=reg_max,
            boxes_weight_format=boxes_weight_format,
            share_matched_indices=share_matched_indices,
            mal_alpha=mal_alpha,
            use_uni_set=use_uni_set,
            no_weight_vfl_epoch=no_weight_vfl_epoch,
            ccm_params=ccm_params,
            density_recall_penalty=density_recall_penalty,
            density_precision_penalty=density_precision_penalty,
            mask_point_sample_ratio=mask_point_sample_ratio,
        )
        cfg = PseudoBoxConfig()
        if isinstance(pseudo_box, dict):
            for k, v in pseudo_box.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        self.pseudo_box_cfg = cfg
        self.pseudo_box_memory = PseudoBoxMemory(cfg)
        self.update_topk = int(update_topk)
        self.update_use_aux_outputs = bool(update_use_aux_outputs)
        self.update_use_enc_aux_outputs = bool(update_use_enc_aux_outputs)
        self.update_use_pre_outputs = bool(update_use_pre_outputs)
        self.update_burnin_epochs = int(update_burnin_epochs)
        self.reg_quality_weight = str(reg_quality_weight)
        self.reg_quality_power = float(reg_quality_power)
        self.density_area_aware = bool(density_area_aware)
        self.density_area_low = float(density_area_low)
        self.density_area_high = float(density_area_high)
        self.density_recall_penalty_mult_small = float(density_recall_penalty_mult_small)
        self.density_recall_penalty_mult_large = float(density_recall_penalty_mult_large)
        self.density_precision_penalty_mult_small = float(density_precision_penalty_mult_small)
        self.density_precision_penalty_mult_large = float(density_precision_penalty_mult_large)
        self.density_area_reduce = str(density_area_reduce)
        self.score_quality_mode = str(score_quality_mode).lower()  # "max" or "mixed"
        if isinstance(pseudo_box, dict):
            if "density_recall_penalty" in pseudo_box:
                self.density_recall_penalty = float(pseudo_box["density_recall_penalty"])
            if "density_precision_penalty" in pseudo_box:
                self.density_precision_penalty = float(pseudo_box["density_precision_penalty"])

        self.dino_semantic_enable = bool(dino_semantic_enable)
        self.dino_semantic_ckpt = None if dino_semantic_ckpt is None else str(dino_semantic_ckpt)
        self.dino_semantic_mode = str(dino_semantic_mode)
        self.dino_semantic_weight = float(dino_semantic_weight)
        self.dino_semantic_input_size = int(dino_semantic_input_size)
        self.dino_semantic_max_boxes = int(dino_semantic_max_boxes)
        self.dino_semantic_tau = float(dino_semantic_tau)
        self.dino_semantic_sim_thresh = float(dino_semantic_sim_thresh)
        self.dino_semantic_min_mask = int(dino_semantic_min_mask)
        self.dino_semantic_min_wh = None if dino_semantic_min_wh is None else float(dino_semantic_min_wh)
        self.dino_semantic_prop_iters = int(dino_semantic_prop_iters)
        self.dino_semantic_hr_size = int(dino_semantic_hr_size)
        self.dino_semantic_interval = int(dino_semantic_interval)
        self._dino = None
        self._dino_device = None
        if self.dino_semantic_enable:
            if not self.dino_semantic_ckpt:
                raise ValueError("dino_semantic_enable=True requires dino_semantic_ckpt")
            self._dino = load_dinov2_vits14_reg4(self.dino_semantic_ckpt, device=None)

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        if self.reg_quality_weight != "cls_score":
            return super().loss_boxes(outputs, targets, indices, num_boxes, boxes_weight=boxes_weight)

        assert "pred_boxes" in outputs and "pred_logits" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        src_logits = outputs["pred_logits"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_labels = torch.cat([t["labels"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        if src_boxes.numel() == 0:
            z = src_boxes.sum()
            return {"loss_bbox": z, "loss_giou": z}

        if self.matcher.use_focal_loss:
            prob = src_logits.sigmoid()
            w = prob.gather(1, target_labels[:, None]).squeeze(1)
        else:
            prob = src_logits.softmax(-1)
            w = prob.gather(1, target_labels[:, None]).squeeze(1)
        w = w.clamp(min=0.0).pow(float(self.reg_quality_power)).detach()
        denom = w.sum().clamp(min=1.0)

        loss_bbox = torch.nn.functional.l1_loss(src_boxes, target_boxes, reduction="none").sum(dim=1)
        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        losses = {
            "loss_bbox": (loss_bbox * w).sum() / denom,
            "loss_giou": (loss_giou * w).sum() / denom,
        }
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        if loss == "point":
            return self.loss_points(outputs, targets, indices, num_boxes)
        return super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)

    def loss_points(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_points = torch.cat([t["points"][j] for t, (_, j) in zip(targets, indices)], dim=0)
        loss = F.smooth_l1_loss(src_boxes[:, :2], target_points, reduction="none").sum() / num_boxes
        return {"loss_point": loss}

    def forward(self, outputs, targets, **kwargs):
        device = next(iter(outputs.values())).device
        dtype = outputs["pred_boxes"].dtype if "pred_boxes" in outputs else torch.float32

        images_input = kwargs.get("images", None)
        images_tensor = None
        if images_input is not None:
            if hasattr(images_input, "tensors"):
                images_tensor = images_input.tensors
            elif torch.is_tensor(images_input):
                images_tensor = images_input

        dino_feat = None
        dino_hw = None
        dino_valid = (
            bool(getattr(self, "dino_semantic_enable", False))
            and (self._dino is not None)
            and (images_tensor is not None)
            and float(getattr(self, "dino_semantic_weight", 0.0)) > 0.0
        )
        if dino_valid:
            step = kwargs.get("step", 0)
            interval = int(getattr(self, "dino_semantic_interval", 1))
            if interval <= 0:
                interval = 1
            dino_valid = (int(step) % interval) == 0

        if dino_valid:
            if images_tensor.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                images_tensor = images_tensor.float()
            if float(images_tensor.max().item()) > 1.5:
                images_tensor = images_tensor / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=images_tensor.device, dtype=images_tensor.dtype)[None, :, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], device=images_tensor.device, dtype=images_tensor.dtype)[None, :, None, None]
            patch = 14
            s = int(getattr(self, "dino_semantic_input_size", 560))
            s = int(max(patch, (s // patch) * patch))
            imgs = torch.nn.functional.interpolate(images_tensor, size=(s, s), mode="bilinear", align_corners=False)
            imgs = (imgs - mean) / std

            if self._dino_device != imgs.device:
                self._dino.to(device=imgs.device)
                self._dino_device = imgs.device

            with torch.no_grad():
                out = self._dino.forward_features(imgs)
                feat = out["x_norm_patchtokens"]
                hw = out["hw"]
                hp, wp = int(hw[0].item()), int(hw[1].item())
                dino_feat = feat.reshape(int(feat.shape[0]), hp, wp, int(feat.shape[-1]))
                dino_hw = (hp, wp)

        pseudo_targets = []
        for t in targets:
            points = t.get("points", None)
            if points is None:
                raise ValueError('PointSupDEIMCriterion requires targets[*]["points"]')

            idx_tensor = t.get("idx", None)
            if idx_tensor is None:
                raise ValueError('PointSupDEIMCriterion requires targets[*]["idx"]')
            sample_idx = int(idx_tensor.item()) if hasattr(idx_tensor, "item") else int(idx_tensor)

            pb = self.pseudo_box_memory.get(sample_idx, points=points, device=device, dtype=dtype)
            refined_points = (
                self.pseudo_box_memory.get_points(sample_idx, points=points, device=device, dtype=dtype)
                if hasattr(self.pseudo_box_memory, "get_points")
                else points
            )
            pt = dict(t)
            pt["boxes"] = pb
            pt["points"] = refined_points
            pseudo_targets.append(pt)

        old_drp = float(getattr(self, "density_recall_penalty", 1.1))
        old_dpp = float(getattr(self, "density_precision_penalty", 1.3))
        if bool(getattr(self, "density_area_aware", False)) and len(pseudo_targets) > 0:
            boxes_all = [pt["boxes"] for pt in pseudo_targets if "boxes" in pt and pt["boxes"].numel() > 0]
            if len(boxes_all) > 0:
                boxes_cat = torch.cat(boxes_all, dim=0)
                area = (boxes_cat[:, 2] * boxes_cat[:, 3]).clamp(min=1e-6)
                if str(getattr(self, "density_area_reduce", "median")) == "mean":
                    a = float(area.mean().item())
                else:
                    a = float(area.median().item())
                lo = max(float(getattr(self, "density_area_low", 0.0025)), 1e-6)
                hi = max(float(getattr(self, "density_area_high", 0.09)), lo + 1e-6)
                t_area = float((torch.tensor(a).log() - torch.tensor(lo).log()) / (torch.tensor(hi).log() - torch.tensor(lo).log()))
                t_area = float(max(0.0, min(1.0, t_area)))
                rr0 = float(getattr(self, "density_recall_penalty_mult_small", 1.0))
                rr1 = float(getattr(self, "density_recall_penalty_mult_large", 1.0))
                rp0 = float(getattr(self, "density_precision_penalty_mult_small", 1.0))
                rp1 = float(getattr(self, "density_precision_penalty_mult_large", 1.0))
                self.density_recall_penalty = old_drp * float(rr0 + (rr1 - rr0) * t_area)
                self.density_precision_penalty = old_dpp * float(rp0 + (rp1 - rp0) * t_area)

        losses = super().forward(outputs, pseudo_targets, **kwargs)
        self.density_recall_penalty = old_drp
        self.density_precision_penalty = old_dpp

        epoch = int(kwargs.get("epoch", 0))
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}
        num_queries_list = outputs.get("num_queries_list", None)
        use_aux = bool(self.update_use_aux_outputs) and "aux_outputs" in outputs and epoch >= int(self.update_burnin_epochs)
        use_enc_aux = bool(self.update_use_enc_aux_outputs) and "enc_aux_outputs" in outputs and epoch >= int(self.update_burnin_epochs)
        use_pre = bool(self.update_use_pre_outputs) and "pre_outputs" in outputs and epoch >= int(self.update_burnin_epochs)

        layers = [outputs_without_aux]
        if use_pre:
            layers.append(outputs["pre_outputs"])
        if use_aux:
            layers.extend(list(outputs.get("aux_outputs", [])))
        if use_enc_aux:
            layers.extend(list(outputs.get("enc_aux_outputs", [])))

        topk = int(self.update_topk)
        if topk > 0:
            all_indices = [
                self.matcher(layer, pseudo_targets, return_topk=topk, epoch=epoch, num_queries_list=num_queries_list)[
                    "indices_o2m"
                ]
                for layer in layers
            ]
        else:
            all_indices = [
                self.matcher(layer, pseudo_targets, epoch=epoch, num_queries_list=num_queries_list)["indices"] for layer in layers
            ]

        matched = 0
        upd_total = 0.0
        upd_radius_ok = 0.0
        upd_score_ok = 0.0
        upd_inside_ok = 0.0
        upd_wh_ok = 0.0
        upd_ok = 0.0
        upd_clip_up = 0.0
        upd_clip_down = 0.0
        upd_sum_mean_max_scale_ok = 0.0
        upd_sum_mean_area_ratio_ok = 0.0
        upd_score_topk_used = 0.0
        upd_sum_score_thresh_eff = 0.0
        upd_images = 0.0
        upd_frozen_images = 0.0
        upd_tgt_total = 0.0
        upd_tgt_updated = 0.0
        n_pts = int(sum(int(t["points"].shape[0]) for t in targets))
        dino_loss_sum = None
        dino_loss_n = 0.0
        for b in range(len(pseudo_targets)):
            idx_tensor = targets[b]["idx"]
            sample_idx = int(idx_tensor.item()) if hasattr(idx_tensor, "item") else int(idx_tensor)

            src_all = []
            tgt_all = []
            logits_all = []
            boxes_all = []
            centers_all = []

            for layer, idx_list in zip(layers, all_indices):
                src_idx, tgt_idx = idx_list[b]
                if tgt_idx.numel() == 0:
                    continue
                src_all.append(src_idx)
                tgt_all.append(tgt_idx)
                logits_all.append(layer["pred_logits"][b][src_idx])
                boxes_all.append(layer["pred_boxes"][b][src_idx])
                if "pred_attn_centers" in layer:
                    centers_all.append(layer["pred_attn_centers"][b][src_idx])

            if len(tgt_all) == 0:
                continue

            src_idx = torch.cat(src_all, dim=0)
            tgt_idx = torch.cat(tgt_all, dim=0)
            logits = torch.cat(logits_all, dim=0)
            pred_boxes = torch.cat(boxes_all, dim=0)
            pred_centers = torch.cat(centers_all, dim=0) if len(centers_all) > 0 else None

            matched += int(tgt_idx.numel())
            labels = pseudo_targets[b]["labels"][tgt_idx]

            # ============================================================================
            # 🔧 修复：改进的置信度计算策略
            # 原问题：在预热期后使用真实标签的置信度，导致置信度过低，伪框膨胀不足
            # 解决方案：根据 score_quality_mode 选择置信度策略
            # ============================================================================
            if self.matcher.use_focal_loss:
                prob = logits.sigmoid()
                max_scores = prob.max(dim=-1).values
                
                if self.score_quality_mode == "mixed":
                    # 混合策略：主要用最大值，逐渐混入真实标签置信度
                    if epoch < int(self.pseudo_box_cfg.class_agnostic_warmup_epochs):
                        scores = max_scores
                    else:
                        # 线性插值：随着epoch增加，逐渐增加真实标签权重
                        label_scores = prob.gather(1, labels[:, None]).squeeze(1)
                        alpha = min(0.3, (epoch - int(self.pseudo_box_cfg.class_agnostic_warmup_epochs)) / 50.0)
                        scores = (1.0 - alpha) * max_scores + alpha * label_scores
                else:
                    # 默认"max"模式：全程使用类别最大置信度
                    scores = max_scores
            else:
                prob = logits.softmax(-1)
                max_scores = prob.max(dim=-1).values
                
                if self.score_quality_mode == "mixed":
                    if epoch < int(self.pseudo_box_cfg.class_agnostic_warmup_epochs):
                        scores = max_scores
                    else:
                        label_scores = prob.gather(1, labels[:, None]).squeeze(1)
                        alpha = min(0.3, (epoch - int(self.pseudo_box_cfg.class_agnostic_warmup_epochs)) / 50.0)
                        scores = (1.0 - alpha) * max_scores + alpha * label_scores
                else:
                    scores = max_scores

            if dino_feat is not None and dino_hw is not None and int(scores.numel()) > 0:
                k = int(getattr(self, "dino_semantic_max_boxes", 64))
                k = max(1, min(k, int(scores.numel())))
                sel = scores.detach().topk(k=k, largest=True).indices
                hp, wp = dino_hw
                mode = str(getattr(self, "dino_semantic_mode", "variance"))
                if mode == "point_sim":
                    boxes_sel = pred_boxes[sel]
                    pts_sel = pseudo_targets[b]["points"][tgt_idx][sel]
                    px = torch.clamp((pts_sel[:, 0] * float(wp)).to(dtype=torch.long), min=0, max=wp - 1)
                    py = torch.clamp((pts_sel[:, 1] * float(hp)).to(dtype=torch.long), min=0, max=hp - 1)
                    flat = dino_feat[b].reshape(hp * wp, -1)
                    flat_n = F.normalize(flat, dim=1)
                    idx = (py * wp + px).to(dtype=torch.long)
                    f_p = flat_n[idx]
                    sim = f_p @ flat_n.t()
                    thr = float(getattr(self, "dino_semantic_sim_thresh", 0.65))
                    sim_mask_2d = (sim > thr).view(k, 1, hp, wp).to(dtype=boxes_sel.dtype)
                    seed_mask = torch.zeros((k, 1, hp, wp), device=device, dtype=boxes_sel.dtype)
                    seed_mask[torch.arange(k, device=device), 0, py, px] = 1.0
                    iters = int(getattr(self, "dino_semantic_prop_iters", 12))
                    if iters <= 0:
                        iters = 1
                    for _ in range(iters):
                        dilated = F.max_pool2d(seed_mask, kernel_size=3, stride=1, padding=1)
                        seed_mask = dilated * sim_mask_2d
                    hr = int(getattr(self, "dino_semantic_hr_size", 0))
                    if hr and hr > 0:
                        seed_mask_hr = F.interpolate(seed_mask, size=(hr, hr), mode="bilinear", align_corners=False)
                        mask = seed_mask_hr.view(k, hr * hr) > 0.5
                        min_mask = int(getattr(self, "dino_semantic_min_mask", 4))
                        min_mask_hr = int(float(min_mask) * (float(hr * hr) / float(hp * wp)))
                        cnt = mask.sum(dim=1)
                        valid = cnt >= int(max(1, min_mask_hr))
                        if bool(valid.any()):
                            x_flat = (torch.arange(hr * hr, device=device) % hr).view(1, -1).expand(k, -1)
                            y_flat = (torch.arange(hr * hr, device=device) // hr).view(1, -1).expand(k, -1)
                            big = int(10**9)
                            x1i = x_flat.masked_fill(~mask, big).min(dim=1).values
                            x2i = x_flat.masked_fill(~mask, -big).max(dim=1).values
                            y1i = y_flat.masked_fill(~mask, big).min(dim=1).values
                            y2i = y_flat.masked_fill(~mask, -big).max(dim=1).values
                            x1 = (x1i.to(dtype=boxes_sel.dtype) / float(hr)).clamp(0.0, 1.0)
                            y1 = (y1i.to(dtype=boxes_sel.dtype) / float(hr)).clamp(0.0, 1.0)
                            x2 = ((x2i.to(dtype=boxes_sel.dtype) + 1.0) / float(hr)).clamp(0.0, 1.0)
                            y2 = ((y2i.to(dtype=boxes_sel.dtype) + 1.0) / float(hr)).clamp(0.0, 1.0)
                            cx = ((x1 + x2) * 0.5).clamp(0.0, 1.0)
                            cy = ((y1 + y2) * 0.5).clamp(0.0, 1.0)
                            w = (x2 - x1).clamp(min=1e-6)
                            h = (y2 - y1).clamp(min=1e-6)
                            min_wh = self.dino_semantic_min_wh
                            if min_wh is None:
                                min_wh = float(self.pseudo_box_cfg.min_wh)
                            w = w.clamp(min=float(min_wh), max=float(self.pseudo_box_cfg.max_wh))
                            h = h.clamp(min=float(min_wh), max=float(self.pseudo_box_cfg.max_wh))
                            dino_boxes = torch.stack([cx, cy, w, h], dim=1)
                            if float(getattr(self, "dino_semantic_weight", 0.0)) > 0.0:
                                l = F.l1_loss(boxes_sel[valid], dino_boxes[valid], reduction="mean") * float(
                                    getattr(self, "dino_semantic_weight", 0.0)
                                )
                                dino_loss_sum = l if dino_loss_sum is None else (dino_loss_sum + l)
                                dino_loss_n += 1.0
                    else:
                        mask = seed_mask.view(k, hp * wp) > 0.5
                        min_mask = int(getattr(self, "dino_semantic_min_mask", 4))
                        cnt = mask.sum(dim=1)
                        valid = cnt >= int(max(1, min_mask))
                        if bool(valid.any()):
                            x_flat = (torch.arange(hp * wp, device=device) % wp).view(1, -1).expand(k, -1)
                            y_flat = (torch.arange(hp * wp, device=device) // wp).view(1, -1).expand(k, -1)
                            big = int(10**9)
                            x1i = x_flat.masked_fill(~mask, big).min(dim=1).values
                            x2i = x_flat.masked_fill(~mask, -big).max(dim=1).values
                            y1i = y_flat.masked_fill(~mask, big).min(dim=1).values
                            y2i = y_flat.masked_fill(~mask, -big).max(dim=1).values
                            x1 = (x1i.to(dtype=boxes_sel.dtype) / float(wp)).clamp(0.0, 1.0)
                            y1 = (y1i.to(dtype=boxes_sel.dtype) / float(hp)).clamp(0.0, 1.0)
                            x2 = ((x2i.to(dtype=boxes_sel.dtype) + 1.0) / float(wp)).clamp(0.0, 1.0)
                            y2 = ((y2i.to(dtype=boxes_sel.dtype) + 1.0) / float(hp)).clamp(0.0, 1.0)
                            cx = ((x1 + x2) * 0.5).clamp(0.0, 1.0)
                            cy = ((y1 + y2) * 0.5).clamp(0.0, 1.0)
                            w = (x2 - x1).clamp(min=1e-6)
                            h = (y2 - y1).clamp(min=1e-6)
                            min_wh = self.dino_semantic_min_wh
                            if min_wh is None:
                                min_wh = float(self.pseudo_box_cfg.min_wh)
                            w = w.clamp(min=float(min_wh), max=float(self.pseudo_box_cfg.max_wh))
                            h = h.clamp(min=float(min_wh), max=float(self.pseudo_box_cfg.max_wh))
                            dino_boxes = torch.stack([cx, cy, w, h], dim=1)
                            if float(getattr(self, "dino_semantic_weight", 0.0)) > 0.0:
                                l = F.l1_loss(boxes_sel[valid], dino_boxes[valid], reduction="mean") * float(
                                    getattr(self, "dino_semantic_weight", 0.0)
                                )
                                dino_loss_sum = l if dino_loss_sum is None else (dino_loss_sum + l)
                                dino_loss_n += 1.0
                else:
                    boxes_sel = pred_boxes[sel].detach()
                    cx = boxes_sel[:, 0]
                    cy = boxes_sel[:, 1]
                    bw = boxes_sel[:, 2].clamp(min=1e-6)
                    bh = boxes_sel[:, 3].clamp(min=1e-6)
                    x1 = (cx - 0.5 * bw).clamp(0.0, 1.0)
                    y1 = (cy - 0.5 * bh).clamp(0.0, 1.0)
                    x2 = (cx + 0.5 * bw).clamp(0.0, 1.0)
                    y2 = (cy + 0.5 * bh).clamp(0.0, 1.0)
                    fx1 = torch.floor(x1 * float(wp)).to(dtype=torch.long)
                    fy1 = torch.floor(y1 * float(hp)).to(dtype=torch.long)
                    fx2 = torch.ceil(x2 * float(wp)).to(dtype=torch.long)
                    fy2 = torch.ceil(y2 * float(hp)).to(dtype=torch.long)
                    fx1 = fx1.clamp(min=0, max=wp - 1)
                    fy1 = fy1.clamp(min=0, max=hp - 1)
                    fx2 = fx2.clamp(min=1, max=wp)
                    fy2 = fy2.clamp(min=1, max=hp)
                    tau = float(getattr(self, "dino_semantic_tau", 0.02))
                    tau = max(tau, 1e-6)
                    sem = torch.ones((k,), device=device, dtype=scores.dtype)
                    for j in range(k):
                        a = int(fy1[j].item())
                        b2 = int(fy2[j].item())
                        c = int(fx1[j].item())
                        d2 = int(fx2[j].item())
                        if b2 <= a or d2 <= c:
                            continue
                        tok = dino_feat[b, a:b2, c:d2, :].reshape(-1, dino_feat.shape[-1])
                        if int(tok.shape[0]) <= 1:
                            continue
                        v = tok.var(dim=0, unbiased=False).mean()
                        sem[j] = torch.exp((-v).to(dtype=scores.dtype) / float(tau))
                    scores = scores * sem.new_ones(scores.shape[0])
                    scores[sel] = scores[sel] * sem
                    if float(getattr(self, "dino_semantic_weight", 0.0)) > 0.0:
                        sem_clamped = sem.clamp(min=1e-6)
                        l = (-torch.log(sem_clamped).detach() * scores[sel]).mean() * float(
                            getattr(self, "dino_semantic_weight", 0.0)
                        )
                        dino_loss_sum = l if dino_loss_sum is None else (dino_loss_sum + l)
                        dino_loss_n += 1.0

            upd = self.pseudo_box_memory.update(
                sample_idx=sample_idx,
                tgt_indices=tgt_idx,
                pred_boxes=pred_boxes,
                pred_scores=scores,
                points=pseudo_targets[b]["points"],
                epoch=epoch,
                pred_centers=pred_centers,
            )
            upd_images += 1.0
            upd_total += float(upd["total"])
            upd_radius_ok += float(upd["radius_ok"])
            upd_score_ok += float(upd["score_ok"])
            upd_inside_ok += float(upd["inside_ok"])
            upd_wh_ok += float(upd["wh_ok"])
            upd_ok += float(upd["ok"])
            upd_clip_up += float(upd["clip_up"])
            upd_clip_down += float(upd["clip_down"])
            upd_sum_mean_max_scale_ok += float(upd["mean_max_scale_ok"]) * float(upd["ok"])
            upd_sum_mean_area_ratio_ok += float(upd["mean_area_ratio_ok"]) * float(upd["ok"])
            upd_score_topk_used += float(upd.get("score_topk_used", 0.0))
            upd_sum_score_thresh_eff += float(upd.get("score_thresh_eff", float(self.pseudo_box_memory.cfg.score_thresh))) * float(
                upd["total"]
            )
            upd_frozen_images += float(upd.get("frozen", 0.0))
            upd_tgt_total += float(upd.get("tgt_total", 0.0))
            upd_tgt_updated += float(upd.get("tgt_updated", 0.0))

        step = kwargs.get("step", None)
        if step is not None:
            try:
                s = int(step)
            except Exception:
                s = None
            if s is not None and s % 20 == 0:
                n_pts = int(sum(int(t["points"].shape[0]) for t in targets))
                ratio = float(matched) / float(max(1, n_pts))
                print(
                    f"[PointSup] epoch={epoch} step={s} matched={matched} points={n_pts} ratio={ratio:.3f} "
                    f"score_thresh={float(self.pseudo_box_memory.cfg.score_thresh):.3f}"
                )

        losses.update(
            {
                "point_matched": torch.as_tensor(float(matched), device=device),
                "point_num_points": torch.as_tensor(float(n_pts), device=device),
                "point_match_ratio": torch.as_tensor(float(matched) / float(max(1, n_pts)), device=device),
                "pseudo_score_thresh": torch.as_tensor(float(self.pseudo_box_memory.cfg.score_thresh), device=device),
                "pseudo_update_total": torch.as_tensor(float(upd_total), device=device),
                "pseudo_update_ok_ratio": torch.as_tensor(float(upd_ok) / float(max(1.0, upd_total)), device=device),
                "pseudo_update_radius_ok_ratio": torch.as_tensor(
                    float(upd_radius_ok) / float(max(1.0, upd_total)), device=device
                ),
                "pseudo_update_score_ok_ratio": torch.as_tensor(float(upd_score_ok) / float(max(1.0, upd_total)), device=device),
                "pseudo_update_inside_ok_ratio": torch.as_tensor(
                    float(upd_inside_ok) / float(max(1.0, upd_total)), device=device
                ),
                "pseudo_update_wh_ok_ratio": torch.as_tensor(float(upd_wh_ok) / float(max(1.0, upd_total)), device=device),
                "pseudo_update_clip_up_ratio": torch.as_tensor(float(upd_clip_up) / float(max(1.0, upd_total)), device=device),
                "pseudo_update_clip_down_ratio": torch.as_tensor(
                    float(upd_clip_down) / float(max(1.0, upd_total)), device=device
                ),
                "pseudo_update_mean_max_scale_ok": torch.as_tensor(
                    float(upd_sum_mean_max_scale_ok) / float(max(1.0, upd_ok)), device=device
                ),
                "pseudo_update_mean_area_ratio_ok": torch.as_tensor(
                    float(upd_sum_mean_area_ratio_ok) / float(max(1.0, upd_ok)), device=device
                ),
                "pseudo_update_score_topk_used_ratio": torch.as_tensor(
                    float(upd_score_topk_used) / float(max(1.0, upd_images)), device=device
                ),
                "pseudo_update_score_thresh_eff": torch.as_tensor(
                    float(upd_sum_score_thresh_eff) / float(max(1.0, upd_total)), device=device
                ),
                "pseudo_update_frozen_ratio": torch.as_tensor(float(upd_frozen_images) / float(max(1.0, upd_images)), device=device),
                "pseudo_update_tgt_total": torch.as_tensor(float(upd_tgt_total), device=device),
                "pseudo_update_tgt_updated_ratio": torch.as_tensor(
                    float(upd_tgt_updated) / float(max(1.0, upd_tgt_total)), device=device
                ),
            }
        )
        if dino_loss_sum is not None and dino_loss_n > 0:
            losses["loss_dino_semantic"] = dino_loss_sum / float(dino_loss_n)
        return losses
