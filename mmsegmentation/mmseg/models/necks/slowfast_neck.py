# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import ConvModule
from ..builder import NECKS


@NECKS.register_module()
class SlowFastNeck(BaseModule):
    """The segmentation neck for SlowFast.
    concatenate slow pathway and fast pathway.

    Args:
        in_channels (List[Tuple[int]]): Number of input channels of fast pathway and slow pathway on each features.
        out_channels (List[int]): Number of output channels on each features.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 **kwargs):
        super().__init__(init_cfg, **kwargs)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, None, None))  # GrobalAveragePooling at `T` dim.
        self.convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.convs.append(
                ConvModule(
                    sum(in_channels[i]),  # (ch_slow, ch_fast)
                    out_channels[i],
                    kernel_size=1,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
            )

    def concat_pathway(self, x_slow, x_fast, conv):
        x_slow = self.avg_pool(x_slow)  # (N,ch_slow,T,H,W) -> (N,ch_slow,1,H,W)
        x_fast = self.avg_pool(x_fast)  # (N,ch_fast,T,H,W) -> (N,ch_fast,1,H,W)
        # [(N,ch_slow,1,H,W), (N,ch_fast,1,H,W)] -> (N,ch_slow+ch_fast,1 H,W)
        x = torch.cat((x_slow, x_fast), dim=1)
        x = x.squeeze(2)  # -> (N,ch_slow+ch_fast,H,W)
        out = conv(x)  # -> (N,out_channel,H,W)
        return out

    @auto_fp16()
    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (List[Tuple[torch.Tensor]]): The input data.
                                          tuple[0] is output of slow pathway, tuple[1] is output of fast pathway,

        Returns:
            outs (List[torch.Tensor]): concatenate data of slow pathway and fast pathway.
        """
        outs = [
            self.concat_pathway(x_slow, x_fast, conv)
            for (x_slow, x_fast), conv in zip(x, self.convs)
        ]
        return outs
