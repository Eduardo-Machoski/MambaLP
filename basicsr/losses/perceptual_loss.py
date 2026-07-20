import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from basicsr.utils.registry import LOSS_REGISTRY
from .contextual_loss import contextual_loss


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    # normalize by number of elements
    return G / (ch * h * w)


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    def __init__(self,
                 layer_weights=None,
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 contextual_weight=0,
                 style_weight=0.0,
                 vgg_type='vgg19'):
        """
        layer_weights: dict mapping vgg layer index (as string) to weight
                       e.g. {'2':1.0, '7':1.0, '12':1.0, '21':1.0}
        use_input_norm: whether to normalize input by ImageNet mean/std
        range_norm: if True, assumes inputs in [-1,1] and rescales to [0,1]
        perceptual_weight: multiplier for perceptual loss
        style_weight: multiplier for style loss (if 0.0 style loss is skipped)
        vgg_type: which torchvision vgg weights to use (default vgg19)
        """
        super().__init__()

        # load pretrained VGG features
        # use the new torchvision weights API if available
        if vgg_type.lower().startswith('vgg19'):
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        else:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval()

        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg_layers = vgg

        # default layers if not provided
        self.layer_weights = layer_weights or {
            '2': 1.0,   # relu1_2
            '7': 1.0,   # relu2_2
            '12': 1.0,  # relu3_2
            '21': 1.0   # relu4_2
        }

        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        self.perceptual_weight = float(perceptual_weight)
        self.style_weight = float(style_weight)
        self.contextual_weight = contextual_weight

        if self.use_input_norm:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x, target):
        """
        Returns: (perceptual_loss (scalar tensor), style_loss (scalar tensor))
        Both tensors are on the same device as x.
        """
        # x: predicted image (N,3,H,W), target: ground truth (N,3,H,W)
        if self.range_norm:
            x = (x + 1) / 2
            target = (target + 1) / 2

        if self.use_input_norm:
            x = self.normalize(x)
            target = self.normalize(target)

        loss_perceptual = 0.0
        loss_style = 0.0
        contextual_loss_value = 0.0

        out_x = x
        out_t = target

        # iterate through vgg layers sequentially and collect losses where needed
        for name, layer in self.vgg_layers._modules.items():
            out_x = layer(out_x)
            out_t = layer(out_t)

            if name in self.layer_weights:
                w = self.layer_weights[name]
                loss_perceptual = loss_perceptual + w * F.l1_loss(out_x, out_t)

                # contextual loss
                if self.contextual_weight > 0:
                    contextual_loss_value = contextual_loss_value + w * contextual_loss(
                        out_x,
                        out_t
                    )

                # style loss via Gram matrix if requested
                if self.style_weight > 0:
                    Gx = gram_matrix(out_x)
                    Gt = gram_matrix(out_t)
                    loss_style = loss_style + w * F.l1_loss(Gx, Gt)

        # ensure tensors
        if isinstance(loss_perceptual, float):
            loss_perceptual = torch.tensor(loss_perceptual, device=x.device, dtype=x.dtype)
        if isinstance(loss_style, float):
            loss_style = torch.tensor(loss_style, device=x.device, dtype=x.dtype)
        if isinstance(contextual_loss_value, float):
            contextual_loss_value = torch.tensor(contextual_loss_value, device=x.device, dtype=x.dtype)

        # apply weights
        loss_perceptual = loss_perceptual * self.perceptual_weight
        loss_style = loss_style * self.style_weight
        contextual_loss_value = contextual_loss_value * self.contextual_weight

        return loss_perceptual, loss_style, contextual_loss_value

