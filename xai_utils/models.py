"""The models used in the posts, and a wrapper that returns their spatial features.

Every model turns a 224x224 image into a grid of feature vectors:
CNNs and hierarchical transformers into 7x7 cells, plain ViTs into one
token per 16x16 (or 14x14) patch. ``FeatureExtractor`` always returns this
grid as a tensor of shape (batch, channels, height, width).
"""
import gc
from dataclasses import dataclass

import timm
import torch
import torchvision.transforms as T
from torch import nn
from torchvision.models import resnet50

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DINO_RESNET50_URL = 'https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth'


@dataclass(frozen=True)
class ModelInfo:
    label: str         # display name
    training: str      # 'supervised', 'image-text' or 'self-supervised'
    architecture: str  # 'CNN', 'local attention' or 'plain ViT'
    checkpoint: str    # timm model name (a URL for DINOv1 ResNet-50)
    family: str        # decides how the features are read out and which LRP rules apply


MODELS = {
    # the first gallery: CNNs and transformers with local attention
    'resnet50':        ModelInfo('ResNet-50', 'supervised', 'CNN', 'resnet50.tv_in1k', 'resnet'),
    'convnext':        ModelInfo('ConvNeXt Base', 'supervised', 'CNN', 'convnext_base.fb_in1k', 'convnext'),
    'swin':            ModelInfo('Swin Base', 'supervised', 'local attention', 'swin_base_patch4_window7_224.ms_in1k', 'swin'),
    'maxvit':          ModelInfo('MaxViT Base', 'supervised', 'local attention', 'maxvit_base_tf_224.in1k', 'maxvit'),
    'clip_convnext':   ModelInfo('CLIP ConvNeXt', 'image-text', 'CNN', 'convnext_base.clip_laion2b', 'convnext'),
    'dino_resnet50':   ModelInfo('DINOv1 ResNet-50', 'self-supervised', 'CNN', DINO_RESNET50_URL, 'resnet'),
    'dinov3_convnext': ModelInfo('DINOv3 ConvNeXt', 'self-supervised', 'CNN', 'convnext_base.dinov3_lvd1689m', 'convnext'),
    # plain vision transformers
    'vit':             ModelInfo('ViT-B/16', 'supervised', 'plain ViT', 'vit_base_patch16_224.orig_in21k_ft_in1k', 'vit'),
    'clip_vit':        ModelInfo('CLIP ViT-B/16', 'image-text', 'plain ViT', 'vit_base_patch16_clip_224.openai', 'vit'),
    'dinov2':          ModelInfo('DINOv2 ViT-B/14', 'self-supervised', 'plain ViT', 'vit_base_patch14_dinov2.lvd142m', 'vit'),
    'dino_vit':        ModelInfo('DINOv1 ViT-B/16', 'self-supervised', 'plain ViT', 'vit_base_patch16_224.dino', 'vit'),
    'dinov3_vit':      ModelInfo('DINOv3 ViT-B/16', 'self-supervised', 'plain ViT', 'vit_base_patch16_dinov3.lvd1689m', 'eva'),
    'mae':             ModelInfo('MAE ViT-B/16', 'self-supervised', 'plain ViT', 'vit_base_patch16_224.mae', 'vit'),
    # more models, used only for the checks in section 11
    'vit_augreg':      ModelInfo('ViT-B/16 (AugReg)', 'supervised', 'plain ViT', 'vit_base_patch16_224.augreg_in21k_ft_in1k', 'vit'),
    'deit':            ModelInfo('DeiT-B', 'supervised', 'plain ViT', 'deit_base_patch16_224.fb_in1k', 'vit'),
    'deit3':           ModelInfo('DeiT III-B', 'supervised', 'plain ViT', 'deit3_base_patch16_224.fb_in1k', 'vit'),
    'dinov2_reg':      ModelInfo('DINOv2 ViT-B/14 + registers', 'self-supervised', 'plain ViT', 'vit_base_patch14_reg4_dinov2.lvd142m', 'vit'),
}


class FeatureExtractor(nn.Module):
    """Wraps a pretrained model so that calling it returns the (B, C, H, W) feature grid."""

    def __init__(self, info, model):
        super().__init__()
        self.info, self.model = info, model
        cfg = getattr(model, 'pretrained_cfg', {}) or {}
        # each checkpoint expects the input normalization it was trained with
        self.mean, self.std = tuple(cfg.get('mean', IMAGENET_MEAN)), tuple(cfg.get('std', IMAGENET_STD))
        self.transform = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])   # PIL image -> (3, H, W) tensor

    def forward(self, x):
        if isinstance(self.model, nn.Sequential):        # DINOv1 ResNet-50: the torchvision trunk
            return self.model(x)
        y = self.model.forward_features(x)
        if self.info.family == 'swin':                   # timm Swin returns (B, H, W, C)
            return y.permute(0, 3, 1, 2)
        if self.info.family in ('vit', 'eva'):           # tokens: drop CLS/register tokens, then fold into a grid
            y = y[:, self.model.num_prefix_tokens:]
            ph, pw = self.model.patch_embed.patch_size
            return y.transpose(1, 2).reshape(x.shape[0], y.shape[-1], x.shape[-2] // ph, x.shape[-1] // pw)
        return y                                         # CNNs, MaxViT: already (B, C, H, W)


def load_model(key, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Download (on first use) and return the model in eval mode, on the GPU if there is one.

    Memory held by models that are no longer in use is released first.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    info = MODELS[key]
    if key == 'dino_resnet50':
        model = resnet50(weights=None)
        model.load_state_dict(torch.hub.load_state_dict_from_url(info.checkpoint, map_location='cpu'), strict=False)
        model = nn.Sequential(*list(model.children())[:-2])  # everything before global pooling; the checkpoint has no classifier
    else:
        kwargs = {'img_size': 224} if 'dinov2' in info.checkpoint else {}  # DINOv2 defaults to 518 px
        model = timm.create_model(info.checkpoint, pretrained=True, **kwargs)
    return FeatureExtractor(info, model).eval().to(device)
