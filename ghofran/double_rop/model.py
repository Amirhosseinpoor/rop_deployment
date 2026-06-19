"""EyeNet network architecture for binocular keratoconus classification.

This is a faithful port of the architecture defined in the original
``double_rop/utils.py``. It is a *dual-branch* network: a shared-design but
independent encoder for each eye, each followed by an attention-enhanced
residual block (CBAM + multi-head self-attention), L2-normalised features, and
three classification heads:

* ``left_fc``  — class of the left eye  (5 classes)
* ``right_fc`` — class of the right eye (5 classes)
* ``z_fc``     — a combined class from both eyes concatenated (2 classes)

Why a separate module: the architecture must match the saved checkpoint
*exactly* (layer names are part of the ``state_dict`` keys), so it is isolated
here, unchanged, away from the inference glue in ``service.py``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import efficientnet_b0, resnet18


class _Config:
    """Architecture hyper-parameters (mirrors the original ``Config``)."""

    num_classes_xy = 5
    classes_xy = ["Normal", "ATN", "NEIr", "EIr", "eKCN"]
    num_classes_z = 2
    classes_z = ["SfRS", "NSfRS"]
    enable_dropout = True
    dropout_prob = 0.2
    enable_l2norm = True
    l2norm_dim = 1


config = _Config()


# --------------------------------------------------------------------------- #
# CBAM: Convolutional Block Attention Module (channel + spatial attention).
# --------------------------------------------------------------------------- #
class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, planes: int, ratio: int = 8, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x_out = x * self.channel_attention(x)
        return x_out * self.spatial_attention(x_out)


# --------------------------------------------------------------------------- #
# Transformer-style bottleneck block used inside the enhanced residual block.
# --------------------------------------------------------------------------- #
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_dim: int, num_heads: int = 8):
        super().__init__()
        assert in_dim % num_heads == 0, "in_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.dim_per_head = in_dim // num_heads
        self.query_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        bsz, C, width, height = x.size()
        queries = self.query_conv(x).view(bsz, self.num_heads, self.dim_per_head, -1)
        keys = self.key_conv(x).view(bsz, self.num_heads, self.dim_per_head, -1)
        values = self.value_conv(x).view(bsz, self.num_heads, self.dim_per_head, -1)

        queries = queries.permute(0, 1, 3, 2)
        keys = keys.permute(0, 1, 2, 3)
        values = values.permute(0, 1, 3, 2)

        attention_scores = torch.matmul(queries, keys) / (self.dim_per_head ** 0.5)
        attention_probs = self.softmax(attention_scores)
        out = torch.matmul(attention_probs, values)
        out = out.permute(0, 1, 3, 2).contiguous().view(bsz, C, width, height)
        return self.gamma * out + x


class BottleneckTransformer(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None, heads=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.attn = MultiHeadSelfAttention(in_planes, num_heads=heads)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.cbam = CBAM(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.attn(out)
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.transformer_block = BottleneckTransformer(in_planes, out_planes, stride=stride)
        self.cbam = CBAM(out_planes)
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.transformer_block(x)
        out = self.cbam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)


# --------------------------------------------------------------------------- #
# The full dual-branch network.
# --------------------------------------------------------------------------- #
class EyeNet(nn.Module):
    """Dual-encoder eye classifier producing left, right and combined-Z logits."""

    def __init__(self, backbone="resnet50", num_classes_xy=5, num_classes_z=2):
        super().__init__()

        # Pick a backbone for each eye. Only 'resnet50' is used by the shipped
        # checkpoint, but the other branches are preserved for completeness.
        if backbone == "resnet18":
            base_model_left = resnet18(weights="DEFAULT")
            base_model_right = resnet18(weights="DEFAULT")
            in_features = base_model_left.fc.in_features
        elif backbone == "resnet50":
            base_model_left = models.resnet50(weights="DEFAULT")
            base_model_right = models.resnet50(weights="DEFAULT")
            in_features = base_model_left.fc.in_features
        elif backbone == "efficientnet_b0":
            base_model_left = efficientnet_b0(weights="DEFAULT")
            base_model_right = efficientnet_b0(weights="DEFAULT")
            in_features = base_model_left.classifier[1].in_features
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented.")

        if "resnet" in backbone:
            self.left_features = nn.Sequential(*list(base_model_left.children())[:-2])
            self.left_enhanced_block = EnhancedResidualBlock(in_features, in_features)
            self.left_pool = nn.AdaptiveAvgPool2d(1)

            self.right_features = nn.Sequential(*list(base_model_right.children())[:-2])
            self.right_enhanced_block = EnhancedResidualBlock(in_features, in_features)
            self.right_pool = nn.AdaptiveAvgPool2d(1)
        elif "efficientnet" in backbone:
            self.left_features = base_model_left.features
            self.left_enhanced_block = EnhancedResidualBlock(in_features, in_features)
            self.left_pool = nn.AdaptiveAvgPool2d(1)

            self.right_features = base_model_right.features
            self.right_enhanced_block = EnhancedResidualBlock(in_features, in_features)
            self.right_pool = nn.AdaptiveAvgPool2d(1)

        if config.enable_dropout:
            self.left_dropout = nn.Dropout(p=config.dropout_prob)
            self.right_dropout = nn.Dropout(p=config.dropout_prob)
            self.z_dropout = nn.Dropout(p=config.dropout_prob)
        else:
            self.left_dropout = nn.Identity()
            self.right_dropout = nn.Identity()
            self.z_dropout = nn.Identity()

        self.left_norm = nn.LayerNorm(in_features)
        self.right_norm = nn.LayerNorm(in_features)
        self.z_norm = nn.LayerNorm(in_features * 2)

        self.left_fc = nn.Linear(in_features, num_classes_xy)
        self.right_fc = nn.Linear(in_features, num_classes_xy)
        self.z_fc = nn.Linear(in_features * 2, num_classes_z)

    def forward(self, left_image, right_image, return_features=False):
        # Left branch.
        left_feat = self.left_features(left_image)
        left_feat = self.left_enhanced_block(left_feat)
        left_feat = self.left_pool(left_feat).view(left_image.size(0), -1)
        left_feat = self.left_norm(self.left_dropout(left_feat))
        if config.enable_l2norm:
            left_feat = F.normalize(left_feat, p=2, dim=config.l2norm_dim)
        left_xy_output = self.left_fc(left_feat)

        # Right branch.
        right_feat = self.right_features(right_image)
        right_feat = self.right_enhanced_block(right_feat)
        right_feat = self.right_pool(right_feat).view(right_image.size(0), -1)
        right_feat = self.right_norm(self.right_dropout(right_feat))
        if config.enable_l2norm:
            right_feat = F.normalize(right_feat, p=2, dim=config.l2norm_dim)
        right_xy_output = self.right_fc(right_feat)

        # Combined Z head from concatenated features.
        combined_feat = torch.cat((left_feat, right_feat), dim=1)
        combined_feat = self.z_norm(self.z_dropout(combined_feat))
        if config.enable_l2norm:
            combined_feat = F.normalize(combined_feat, p=2, dim=config.l2norm_dim)
        z_output = self.z_fc(combined_feat)

        if return_features:
            return left_xy_output, right_xy_output, z_output, combined_feat
        return left_xy_output, right_xy_output, z_output
