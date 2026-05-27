""" 
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""    
 
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms.v2 as T 
import torchvision.transforms.v2.functional as F   
from torchvision.transforms.v2 import InterpolationMode  
   
import PIL
import PIL.Image 
 
from typing import Any, Dict, List, Optional, Union, Tuple

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes  

from ...core import register
torchvision.disable_beta_transforms_warning()  
   
from ...logger_module import get_logger    
    
RandomPhotometricDistort = register()(T.RandomPhotometricDistort) 
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor) 
# ConvertDtype = register()(T.ConvertDtype) 
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)   
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize) 
RandomErasing = register()(T.RandomErasing)

logger = get_logger(__name__)
   
@register()
class EmptyTransform(T.Transform):  
    def __init__(self, ) -> None:
        super().__init__()    

    def forward(self, *inputs): 
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs     
 
@register()   
class ResizeLongestEdge(T.Transform):   
    """
    将最长边缩放到指定大小，保持宽高比   
     
    支持:
        - PIL.Image
        - torch.Tensor    
        - torchvision.tv_tensors (Image, Video, Mask, BoundingBoxes)     
   
    Args:     
        size (int): 最长边的目标尺寸
        interpolation (InterpolationMode): 插值模式，默认 BILINEAR
        max_size (int, optional): 最短边的最大值限制 
        antialias (bool): 是否抗锯齿，默认 True     
    
    Examples:     
        >>> transform = ResizeLongestEdge(size=640)
        >>> img = Image.new('RGB', (1920, 1080)) 
        >>> out = transform(img)    
        >>> print(out.size)  # (640, 360)
  
        >>> # 配合其他变换
        >>> transform = T.Compose([    
        ...     ResizeLongestEdge(640),     
        ...     T.ConvertImageDtype(torch.float32),
        ...     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ... ])  
    """
 
    _transformed_types = (  
        PIL.Image.Image,  
        Image, 
        Video, 
        Mask,
        BoundingBoxes,
    )     
    
    def __init__(
        self,
        size: int,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,   
        max_size: int = None,
        antialias: bool = True    
    ):
        super().__init__()    
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size     
        self.antialias = antialias
    
    def _get_spatial_size(self, inpt: Any) -> Tuple[int, int]:
        """  
        获取空间尺寸 (height, width)  
        兼容多版本 torchvision
        """
        # 尝试使用 torchvision 的函数 
        if hasattr(F, 'get_size'):   
            return F.get_size(inpt) 
        elif hasattr(F, 'get_spatial_size'):
            return F.get_spatial_size(inpt)     
        
        # 手动处理
        if isinstance(inpt, PIL.Image.Image):
            w, h = inpt.size   
            return (h, w)   
        elif isinstance(inpt, torch.Tensor): 
            return tuple(inpt.shape[-2:])
        elif hasattr(inpt, 'shape'):   
            return tuple(inpt.shape[-2:])
        else:
            raise TypeError(f"Cannot get spatial size from {type(inpt)}") 
    
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        """计算缩放后的尺寸""" 
        orig_h, orig_w = self._get_spatial_size(flat_inputs[0])
        
        # 找到最长边   
        longest_edge = max(orig_h, orig_w)     
   
        # 计算缩放比例
        scale = self.size / longest_edge
     
        # 计算新尺寸    
        new_h = int(orig_h * scale) 
        new_w = int(orig_w * scale)   
        
        # 如果设置了 max_size，限制最短边     
        if self.max_size is not None: 
            shortest_edge = min(new_h, new_w)     
            if shortest_edge > self.max_size:
                scale = self.max_size / min(orig_h, orig_w)
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)
        
        return {     
            'size': (new_h, new_w),
            'scale': scale   
        }     
 
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """应用缩放变换"""    
        size = params['size'] 
    
        # 对不同类型使用不同的插值模式
        if isinstance(inpt, (BoundingBoxes, Mask)):     
            # BoundingBoxes 和 Mask 使用最近邻插值
            return F.resize(
                inpt,
                size=size,  
                interpolation=InterpolationMode.NEAREST    
            )
        else:  
            # 图像使用指定的插值模式 
            return F.resize(
                inpt,
                size=size,
                interpolation=self.interpolation,    
                antialias=self.antialias  
            )    
    
    def __repr__(self) -> str:    
        return ( 
            f"{self.__class__.__name__}("
            f"size={self.size}, "
            f"interpolation={self.interpolation}, "    
            f"max_size={self.max_size}, " 
            f"antialias={self.antialias})"     
        )

@register()     
class PadToSize(T.Pad):
    _transformed_types = (  
        PIL.Image.Image,
        Image,   
        Video,   
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:    
        if hasattr(F, 'get_size'):
            sp = F.get_size(flat_inputs[0])     
        elif hasattr(F, 'get_spatial_size'): 
            sp = F.get_spatial_size(flat_inputs[0])
        else:    
            if isinstance(flat_inputs[0], PIL.Image.Image):
                w, h = flat_inputs[0].size
                sp = (h, w)
        print(sp)
        h, w = self.size[1] - sp[0], self.size[0] - sp[1] 
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)
  
    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode) 

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        padding = params['padding']     
        return F.pad(inpt, padding=padding, fill=self.fill, padding_mode=self.padding_mode)  # type: ignore[arg-type] 
 
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params) 

    def __call__(self, *inputs: Any) -> Any:     
        outputs = super().forward(*inputs)  
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)   
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):   
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):  
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:    
            return inputs if len(inputs) > 1 else inputs[0] 
   
        return super().forward(*inputs)     
     
  
@register()
class PointAwareCopyPaste(T.Transform):
    def __init__(
        self,
        p: float = 0.5,
        max_paste: int = 3,
        max_area_ratio: float = 0.01,
        bbox_pad: int = 0,
        max_iou: float = 0.1,
        max_trials: int = 50,
    ) -> None:
        super().__init__()
        self.p = float(p)
        self.max_paste = int(max_paste)
        self.max_area_ratio = float(max_area_ratio)
        self.bbox_pad = int(bbox_pad)
        self.max_iou = float(max_iou)
        self.max_trials = int(max_trials)

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            return sample

        image = sample[0]
        target = sample[1]
        if not isinstance(target, dict) or "boxes" not in target or "labels" not in target:
            return sample
        if not isinstance(image, PIL.Image.Image):
            return sample
        if self.p <= 0 or torch.rand(1).item() >= self.p:
            return sample

        boxes = target["boxes"]
        labels = target["labels"]
        if boxes.numel() == 0:
            return sample

        w, h = image.size
        img_area = float(max(1, w * h))
        b = boxes.to(dtype=torch.float32)
        wh = (b[:, 2:] - b[:, :2]).clamp(min=0.0)
        area = (wh[:, 0] * wh[:, 1]).cpu()
        small = area <= (float(self.max_area_ratio) * img_area)
        if not bool(small.any()):
            return sample

        cand = torch.nonzero(small, as_tuple=False).squeeze(1).tolist()
        if len(cand) == 0:
            return sample

        n_paste = min(int(self.max_paste), len(cand))
        perm = torch.randperm(len(cand))[:n_paste].tolist()
        pick = [cand[i] for i in perm]

        out_image = image.copy()
        out_boxes = b.clone()
        out_labels = labels.clone()

        for idx in pick:
            x1, y1, x2, y2 = out_boxes[idx].tolist()
            x1 = int(max(0, min(w - 1, round(x1) - self.bbox_pad)))
            y1 = int(max(0, min(h - 1, round(y1) - self.bbox_pad)))
            x2 = int(max(0, min(w, round(x2) + self.bbox_pad)))
            y2 = int(max(0, min(h, round(y2) + self.bbox_pad)))
            pw = int(max(1, x2 - x1))
            ph = int(max(1, y2 - y1))
            if pw <= 1 or ph <= 1:
                continue
            patch = out_image.crop((x1, y1, x2, y2))

            placed = False
            for _ in range(int(self.max_trials)):
                nx1 = int(torch.randint(low=0, high=max(1, w - pw + 1), size=(1,)).item())
                ny1 = int(torch.randint(low=0, high=max(1, h - ph + 1), size=(1,)).item())
                nx2 = nx1 + pw
                ny2 = ny1 + ph
                new_box = torch.tensor([[nx1, ny1, nx2, ny2]], dtype=torch.float32)
                iou = torchvision.ops.box_iou(new_box, out_boxes).max().item()
                if float(iou) > float(self.max_iou):
                    continue
                out_image.paste(patch, (nx1, ny1))
                out_boxes = torch.cat([out_boxes, new_box], dim=0)
                out_labels = torch.cat([out_labels, out_labels[idx : idx + 1]], dim=0)
                placed = True
                break
            if not placed:
                continue

        if int(out_boxes.shape[0]) == int(boxes.shape[0]):
            return sample

        spatial_size = getattr(boxes, _boxes_keys[1])
        target = dict(target)
        target["boxes"] = convert_to_tv_tensor(out_boxes, key="boxes", box_format=boxes.format.value, spatial_size=spatial_size)
        target["labels"] = out_labels

        if isinstance(sample, tuple):
            return (out_image, target, *sample[2:])
        return [out_image, target, *sample[2:]]


@register()
class ConvertBoxes(T.Transform):   
    _transformed_types = (
        BoundingBoxes, 
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()     
        self.fmt = fmt
        self.normalize = normalize
     
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])     
        if self.fmt:  
            in_fmt = inpt.format.value.lower()    
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower()) 
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize: 
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt   

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)
 

@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image, 
    )
    def __init__(self, dtype='float32', scale=True, dinov3_norm=False) -> None:  
        super().__init__()
        self.dtype = dtype
        self.scale = scale
        self.dinov3_norm = dinov3_norm
        
        self.dinov3_normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),  
        ) 

        if self.dinov3_norm:
            logger.info(f'Using Dinov3 Normalize:{self.dinov3_normalize}')
   
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':  
            inpt = inpt.float() 
        
        if self.scale:    
            inpt = inpt / 255.     
            if self.dinov3_norm: 
                inpt = self.dinov3_normalize(inpt)  

        inpt = Image(inpt)
  
        return inpt   
    
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any: 
        return self._transform(inpt, params)
