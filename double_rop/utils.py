import io
import base64
import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# Image Transformation Pipeline
# -------------------------------------------------------
simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# -------------------------------------------------------
# Model Configuration Class
# -------------------------------------------------------
class Config:
    num_classes_xy = 5
    classes_xy = ['Normal', 'ATN', 'NEIr', 'EIr', 'eKCN']
    num_classes_z = 2
    classes_z = ['SfRS', 'NSfRS']
    device = device
    enable_dropout = True
    dropout_prob = 0.2
    enable_l2norm = True
    l2norm_dim = 1


config = Config()


# -------------------------------------------------------
# Model Architecture Components (CBAM, Transformers)
# -------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x_out = x * self.channel_attention(x)
        x_out = x_out * self.spatial_attention(x_out)
        return x_out


# -------------------------------------------------------
# Complete EyeNet Model Implementation
# -------------------------------------------------------


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
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
        out = self.gamma * out + x
        return out


class BottleneckTransformer(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None, heads=4):
        super(BottleneckTransformer, self).__init__()
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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.attn(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.transformer_block = BottleneckTransformer(in_planes, out_planes, stride=stride)
        self.cbam = CBAM(out_planes)

        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
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


class EyeNet(nn.Module):
    def __init__(self, backbone='resnet50', num_classes_xy=5, num_classes_z=2):
        super().__init__()
        # Backbone initialization
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(base_model.children())[:-2]

        # Load chosen backbone
        if backbone == 'resnet18':
            base_model_left = resnet18(weights='DEFAULT')
            base_model_right = resnet18(weights='DEFAULT')
            in_features = base_model_left.fc.in_features
        elif backbone == 'resnet50':
            base_model_left = models.resnet50(weights='DEFAULT')
            base_model_right = models.resnet50(weights='DEFAULT')
            in_features = base_model_left.fc.in_features
        elif backbone == 'efficientnet_b0':
            base_model_left = efficientnet_b0(weights='DEFAULT')
            base_model_right = efficientnet_b0(weights='DEFAULT')
            in_features = base_model_left.classifier[1].in_features
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented.")

        # Remove final layers
        if 'resnet' in backbone:
            modules_left = list(base_model_left.children())[:-2]
            self.left_features = nn.Sequential(*modules_left)
            self.left_enhanced_block = EnhancedResidualBlock(in_features, in_features)
            self.left_pool = nn.AdaptiveAvgPool2d(1)

            modules_right = list(base_model_right.children())[:-2]
            self.right_features = nn.Sequential(*modules_right)
            self.right_enhanced_block = EnhancedResidualBlock(in_features, in_features)
            self.right_pool = nn.AdaptiveAvgPool2d(1)

        elif 'efficientnet' in backbone:
            self.left_features = base_model_left.features
            self.left_enhanced_block = EnhancedResidualBlock(in_features, in_features)
            self.left_pool = nn.AdaptiveAvgPool2d(1)

            self.right_features = base_model_right.features
            self.right_enhanced_block = EnhancedResidualBlock(in_features, in_features)
            self.right_pool = nn.AdaptiveAvgPool2d(1)

        # Dropout layers
        if config.enable_dropout:
            self.left_dropout = nn.Dropout(p=config.dropout_prob)
            self.right_dropout = nn.Dropout(p=config.dropout_prob)
            self.z_dropout = nn.Dropout(p=config.dropout_prob)
        else:
            self.left_dropout = nn.Identity()
            self.right_dropout = nn.Identity()
            self.z_dropout = nn.Identity()

        # LayerNorm
        self.left_norm = nn.LayerNorm(in_features)
        self.right_norm = nn.LayerNorm(in_features)
        self.z_norm = nn.LayerNorm(in_features * 2)

        # Classification heads
        self.left_fc = nn.Linear(in_features, num_classes_xy)
        self.right_fc = nn.Linear(in_features, num_classes_xy)
        self.z_fc = nn.Linear(in_features * 2, num_classes_z)

    def forward(self, left_image, right_image, return_features=False):
        # Left branch
        left_feat = self.left_features(left_image)
        left_feat = self.left_enhanced_block(left_feat)
        left_feat = self.left_pool(left_feat)
        left_feat = left_feat.view(left_feat.size(0), -1)
        left_feat = self.left_dropout(left_feat)
        left_feat = self.left_norm(left_feat)
        if config.enable_l2norm:
            left_feat = F.normalize(left_feat, p=2, dim=config.l2norm_dim)
        left_xy_output = self.left_fc(left_feat)

        # Right branch
        right_feat = self.right_features(right_image)
        right_feat = self.right_enhanced_block(right_feat)
        right_feat = self.right_pool(right_feat)
        right_feat = right_feat.view(right_feat.size(0), -1)
        right_feat = self.right_dropout(right_feat)
        right_feat = self.right_norm(right_feat)
        if config.enable_l2norm:
            right_feat = F.normalize(right_feat, p=2, dim=config.l2norm_dim)
        right_xy_output = self.right_fc(right_feat)

        # Combined for Z
        combined_feat = torch.cat((left_feat, right_feat), dim=1)
        combined_feat = self.z_dropout(combined_feat)
        combined_feat = self.z_norm(combined_feat)
        if config.enable_l2norm:
            combined_feat = F.normalize(combined_feat, p=2, dim=config.l2norm_dim)
        z_output = self.z_fc(combined_feat)

        if return_features:
            return left_xy_output, right_xy_output, z_output, combined_feat
        else:
            return left_xy_output, right_xy_output, z_output


# -------------------------------------------------------
# Model Initialization
# -------------------------------------------------------
model = EyeNet().to(device)
model.load_state_dict(torch.load("model/best_model.pth", map_location=device, weights_only=True))
model.eval()


# -------------------------------------------------------
# Image Processing and Prediction Functions
# -------------------------------------------------------
def process_image(image_bytes):
    """Convert uploaded image bytes to model-ready tensor"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return simple_transform(image).unsqueeze(0).to(device)
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")


def encode_image(image_bytes):
    """Create base64 encoded string for web display"""
    return f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"


def get_prediction(left_bytes, right_bytes):
    """Main prediction pipeline"""
    try:
        # Process both images
        left_tensor = process_image(left_bytes)
        right_tensor = process_image(right_bytes)

        # Run model inference
        with torch.no_grad():
            left_out, right_out, z_out = model(left_tensor, right_tensor)

        # Convert outputs to probabilities
        left_probs = F.softmax(left_out, dim=1).squeeze()
        right_probs = F.softmax(right_out, dim=1).squeeze()
        z_probs = F.softmax(z_out, dim=1).squeeze()


        # Format results
        return {
            "left_eye": {
                "label": config.classes_xy[torch.argmax(left_probs).item()],
                "probability": f"{torch.max(left_probs).item():.4f}"
            },
            "right_eye": {
                "label": config.classes_xy[torch.argmax(right_probs).item()],
                "probability": f"{torch.max(right_probs).item():.4f}"
            },
            "z_class": {
                "label": config.classes_z[torch.argmax(z_probs).item()],
                "probability": f"{torch.max(z_probs).item():.4f}"
            },
            "image_data": {
                "left": encode_image(left_bytes),
                "right": encode_image(right_bytes)
            }
        }

    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")