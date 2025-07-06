import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from typing import Callable, Optional, Sequence

from a_unet import (
    ClassifierFreeGuidancePlugin,
    Conv,
    Module,
    TextConditioningPlugin,
    TimeConditioningPlugin,
    default,
    exists,
)
from a_unet.apex import (
    AttentionItem,
    CrossAttentionItem,
    InjectChannelsItem,
    ModulationItem,
    ResnetItem,
    SkipCat,
    SkipModulate,
    XBlock,
    XUNet,
)

def colorize_flow(flow):
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    flow_magnitude, flow_angle = cv2.cartToPolar(flow[0,:,:].astype(np.float32), flow[1,:,:].astype(np.float32))

    # A couple times, we've gotten NaNs out of the above...
    nans = np.isnan(flow_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        flow_magnitude[nans] = 0.

    # Normalize
    hsv[..., 0] = flow_angle * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img

"""
UNets (built with a-unet: https://github.com/archinetai/a-unet)
"""
def UNetV0(
    dim: int,
    in_channels: int,
    channels: Sequence[int],
    factors: Sequence[int],
    items: Sequence[int],
    attentions: Optional[Sequence[int]] = None,
    cross_attentions: Optional[Sequence[int]] = None,
    context_channels: Optional[Sequence[int]] = None,
    attention_features: Optional[int] = None,
    attention_heads: Optional[int] = None,
    embedding_features: Optional[int] = None,
    resnet_groups: int = 8,
    use_modulation: bool = True,
    modulation_features: int = 1024,
    embedding_max_length: Optional[int] = None,
    use_time_conditioning: bool = True,
    use_embedding_cfg: bool = False,
    use_text_conditioning: bool = False,
    out_channels: Optional[int] = None,
):
    # Set defaults and check lengths
    num_layers = len(channels)
    attentions = default(attentions, [0] * num_layers)
    cross_attentions = default(cross_attentions, [0] * num_layers)
    context_channels = default(context_channels, [0] * num_layers)
    xs = (channels, factors, items, attentions, cross_attentions, context_channels)
    assert all(len(x) == num_layers for x in xs)  # type: ignore

    # Define UNet type
    UNetV0 = XUNet

    if use_embedding_cfg:
        msg = "use_embedding_cfg requires embedding_max_length"
        assert exists(embedding_max_length), msg
        UNetV0 = ClassifierFreeGuidancePlugin(UNetV0, embedding_max_length)

    if use_text_conditioning:
        UNetV0 = TextConditioningPlugin(UNetV0)

    if use_time_conditioning:
        assert use_modulation, "use_time_conditioning requires use_modulation=True"
        UNetV0 = TimeConditioningPlugin(UNetV0)

    # Build
    return UNetV0(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        blocks=[
            XBlock(
                channels=channels,
                factor=factor,
                context_channels=ctx_channels,
                items=(
                    [ResnetItem]
                    + [ModulationItem] * use_modulation
                    + [InjectChannelsItem] * (ctx_channels > 0)
                    + [AttentionItem] * att
                    + [CrossAttentionItem] * cross
                )
                * items,
            )
            for channels, factor, items, att, cross, ctx_channels in zip(*xs)  # type: ignore # noqa
        ],
        skip_t=SkipModulate if use_modulation else SkipCat,
        attention_features=attention_features,
        attention_heads=attention_heads,
        embedding_features=embedding_features,
        modulation_features=modulation_features,
        resnet_groups=resnet_groups,
    )


class FlowDecoder(nn.Module):
    def __init__(self):
        super(FlowDecoder, self).__init__()

        self.model = nn.Sequential(*[
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=(2,1), output_padding = (0,1)),
            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=(2,1), output_padding=1),
            nn.BatchNorm2d(256),

            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),

            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),

            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 2, kernel_size=3, stride=1, padding=1),
        ])
        
    def forward(self, x):
        x = self.model(x)
        # img = colorize_flow(x[0].cpu().detach().numpy())
        # cv2.imwrite('./vis/flow_est.png', img)
        return x
    

class MaskDecoder(nn.Module):
    def __init__(self):
        super(MaskDecoder, self).__init__()

        self.model = nn.Sequential(*[
            nn.ReLU(),
            nn.ConvTranspose2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),

            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),

            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),

            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),

            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),

            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),

            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),

            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),

            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),

            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),

            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2),

            nn.Sigmoid(),
        ])
        
    def forward(self, x):
        x = self.model(x)
        return x

class DepthDecoder(nn.Module):
    def __init__(self):
        super(DepthDecoder, self).__init__()

        self.model = nn.Sequential(*[
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=(2,1), output_padding = (0,1)),
            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=(2,1), output_padding=1),
            nn.BatchNorm2d(256),

            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),

            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),

            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 2, kernel_size=3, stride=1, padding=1),
        ])
        
    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        # cv2.imwrite('./vis/depth_est.png', x[0][0].cpu().detach().numpy()*80)
        return x
