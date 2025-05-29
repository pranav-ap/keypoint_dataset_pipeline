import torch
import torch.nn as nn
from torchvision import transforms as T
import math
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
import torchmetrics
import lightning.pytorch as pl


torch.set_float32_matmul_precision("medium")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = "D:/thesis_code/basalt/finalnets/lotus_57_linux/best_checkpoint.ckpt"

patch_normalize = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ]
)


class RoPENd(torch.nn.Module):
    """N-dimensional Rotary Positional Embedding."""
    def __init__(self, shape, base=10000):
        super(RoPENd, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0, f'shape[-1] ({feature_dim}) is not divisible by 2 * len(shape[:-1]) ({2 * len(channel_dims)})'

        # tensor of angles to use
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))

        # create a stack of angles multiplied by position
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # convert to complex number to allow easy rotation
        rotations = torch.polar(torch.ones_like(angles), angles)

        # store in a buffer so it can be saved in model parameters
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        # convert input into complex numbers to perform rotation
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = self.rotations * x
        return torch.view_as_real(pe_x).flatten(-2)


def positionalencoding2d(d_model, height, width):
    assert d_model % 4 == 0, f"Cannot use sin/cos positional encoding with odd dimension (got dim={d_model})"

    pe = torch.zeros(d_model, height, width, requires_grad=False)
    d_model = d_model // 2
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))

    pos_w = torch.arange(width).unsqueeze(1)
    pos_h = torch.arange(height).unsqueeze(1)

    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


"""
Model
"""



class PreActDSBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.depthwise1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.depthwise2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False)
        self.pointwise2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

        self.shortcut = None
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut else x

        out = self.depthwise1(out)
        out = self.pointwise1(out)
        out = F.relu(self.bn2(out))
        out = self.depthwise2(out)
        out = self.pointwise2(out)

        out += shortcut
        return out


class PreActBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = None
        
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut

        return out


class MatcherModel(nn.Module):
    def __init__(self):
        super().__init__()

        patch_size = 32 
        in_channels = 1
        embedding_length = 64
        out_channels = 512

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, embedding_length, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embedding_length),
        )
        
        # self.register_buffer(
        #     'positional_encoding', 
        #     positionalencoding2d(embedding_length, height=patch_size, width=patch_size).unsqueeze(0)
        # )

        self.positional_encoding = RoPENd((patch_size, patch_size, embedding_length))

        layers = [
            (embedding_length * 2, 256, 1),
            (256, 512, 1),
            (512, out_channels, 2),
        ]

        self.backbone = nn.ModuleList([
            PreActDSBasicBlock(in_c, out_c, stride) for in_c, out_c, stride in layers
        ])

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(out_channels + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
     

    def forward(self, ref_patches, tar_patches, estimates):
        _, _, height, width = ref_patches.shape

        # normalize to 0 to 1
        estimates = estimates / (height - 1)
        # normalize to -1 to 1
        estimates = estimates * 2 - 1

        """
        Patch Embedding
        """

        ref_patches = self.patch_embedding(ref_patches)
        
        # ref_patches = ref_patches * self.positional_encoding
        
        # (b, c, h, w) -> (b, h, w, c)
        ref_patches = ref_patches.permute(0, 2, 3, 1).contiguous()
        ref_patches = self.positional_encoding(ref_patches)
        # (b, h, w, c) -> (b, c, h, w)
        ref_patches = ref_patches.permute(0, 3, 1, 2).contiguous()

        tar_patches = self.patch_embedding(tar_patches)
        
        # tar_patches = tar_patches * self.positional_encoding
        
        # (b, c, h, w) -> (b, h, w, c)
        tar_patches = tar_patches.permute(0, 2, 3, 1).contiguous()
        tar_patches = self.positional_encoding(tar_patches)
        # (b, h, w, c) -> (b, c, h, w)
        tar_patches = tar_patches.permute(0, 3, 1, 2).contiguous()

        """
        PreActBasicBlock ResNet 
        """

        x = torch.cat([ref_patches, tar_patches], dim=1)

        for layer in self.backbone:
            x = layer(x)

        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)

        """
        Linear Layers for Separate Heads
        """
        
        x = torch.cat([x, estimates], dim=1)
        x = self.head(x)
        
        coords = F.tanh(x[:, :2]) 
        confs = F.sigmoid(x[:, 2:])  # (B, 2)
        confs_box, confs_point = confs[:, 0], confs[:, 1]  # (B,), (B,)
        
        return coords, confs_box * confs_point

