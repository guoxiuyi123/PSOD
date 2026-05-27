from __future__ import annotations

import torch
from torch import nn


class TinyViT(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError("Mobile-SAM encoder is not included in PSOD")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Mobile-SAM encoder is not included in PSOD")

