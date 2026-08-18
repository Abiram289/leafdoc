"""
Builds a pretrained backbone with a fresh classification head via timm.
Same factory serves Stage 1 (binary) and Stage 2 (38-class) -- only
num_classes differs, so the two stages are trivially comparable and the
optional Week-2 multi-task merge (shared backbone, two heads) can reuse
this without rewriting anything.
"""
import timm
import torch.nn as nn


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    backbone: any timm model name, e.g. 'efficientnet_b0', 'resnet50'.
    timm handles swapping the final classifier layer to num_classes for us.
    """
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
    return model
