import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np
from MLP import MLP
from LePEAttention import LePEAttention

import torch
import torch.nn as nn


class CustomCSWinBlock(nn.Module):
    def __init__(
        self,
        mode: str,
        input_dim,
        patch_size,
        num_heads,
        split_size=7,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path=0.0,
        activation_layer=nn.GELU,
        normalization_layer=nn.LayerNorm,
        last_stage=False,
    ):
        super().__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.patches_resolution = patch_size
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(input_dim, input_dim * 3, bias=qkv_bias)
        self.norm1 = normalization_layer(input_dim)
        # check mode, if norm then branch_num = 2, else branch_num = 4
        if self.mode == "norm":
            self.branch_num = 1 if last_stage else 2
        elif self.mode == "sw1" or self.mode == "sw2":
            self.branch_num = 1 if last_stage else 4
        else:
            raise ValueError("mode must be 'norm', 'sw1', or 'sw2'")
        self.proj = nn.Linear(input_dim, input_dim)
        # self.proj_drop = nn.Dropout(dropout)

        if self.mode == "norm":
            self.attns = nn.ModuleList(
                [
                    LePEAttention(
                        mode = mode,
                        dim = input_dim // self.branch_num,
                        resolution=self.patches_resolution,
                        idx=i if self.branch_num == 2 else -1,
                        split_size=split_size,
                        num_heads=num_heads // self.branch_num,
                        dim_out=input_dim // self.branch_num,
                        qk_scale=qk_scale,
                        attn_drop=attention_dropout,
                        proj_drop=dropout,
                    )
                    for i in range(self.branch_num)
                ]
            )
        elif self.mode == "sw1" or self.mode == "sw2":
            self.attns = nn.ModuleList(
                [
                    LePEAttention(
                        mode = mode,
                        dim = input_dim // self.branch_num,
                        resolution=self.patches_resolution,
                        idx=i if self.branch_num == 4 else -1,
                        split_size=split_size,
                        num_heads=num_heads // self.branch_num,
                        dim_out=input_dim // self.branch_num,
                        qk_scale=qk_scale,
                        attn_drop=attention_dropout,
                        proj_drop=dropout,
                    )
                    for i in range(self.branch_num)
                ]
            )
        else:
            raise ValueError("mode must be 'norm', 'sw1', or 'sw2'")

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = MLP(
            in_features=input_dim,
            hidden_features=int(input_dim * mlp_ratio),
            out_features=input_dim,
            act_layer=activation_layer,
            drop=dropout,
        )
        self.norm2 = normalization_layer(input_dim)

    def forward(self, x):
        """
        x: B, H * W, C
        """

        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        # attended_x = torch.cat(
        #     [self.attns[i](qkv[:, :, :, i * (C // 2) : C // 2 if i == 0 else None]) for i in
        #      range(self.branch_num)], input_dim=2) if self.branch_num == 2 else self.attns[0](qkv)

        if self.mode == "norm":
            if self.branch_num == 2:
                x1 = self.attns[0](qkv[:, :, :, : C // 2])
                x2 = self.attns[1](qkv[:, :, :, C // 2 :])
                attended_x = torch.cat([x1, x2], dim=2)
            else:
                attended_x = self.attns[0](qkv)

        elif self.mode == "sw1":
            if self.branch_num == 4:
                c1 = C // 4
                c2 = 2 * C // 4
                c3 = 3 * C // 4
                x1 = self.attns[0](qkv[:, :, :, :c1])
                x2 = self.attns[1](qkv[:, :, :, c1:c2])
                x3 = self.attns[2](qkv[:, :, :, c2:c3])
                x4 = self.attns[3](qkv[:, :, :, c3:])
                attended_x = torch.cat([x1, x2, x3, x4], dim=2)
            else:
                attended_x = self.attns[0](qkv)
        
        elif self.mode == "sw2":
            if self.branch_num == 4:
                c1 = C // 4
                c2 = 2 * C // 4
                c3 = 3 * C // 4
                x1 = self.attns[0](qkv[:, :, :, :c1])
                x2 = self.attns[1](qkv[:, :, :, c1:c2])
                x3 = self.attns[2](qkv[:, :, :, c2:c3])
                x4 = self.attns[3](qkv[:, :, :, c3:])
                attended_x = torch.cat([x1, x2, x3, x4], dim=2)
            else:
                attended_x = self.attns[0](qkv)

        else:
            raise ValueError("mode must be 'norm', 'sw1', or 'sw2'")

        attended_x = self.proj(attended_x)
        x = x + self.drop_path(attended_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
