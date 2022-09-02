# Copyright (c) Yamada Atsushi. All rights reserved.
import torch.nn as nn
from ..builder import NECKS


@NECKS.register_module()
class SelectiveLevel(nn.Module):
    """select levels through head
    Args:
        out_index (Union[List|Tuple|int]): select indices of feature levels
    """

    def __init__(self, out_index):
        super().__init__()
        assert isinstance(out_index, (int, list, tuple))
        self.out_index = out_index

    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (List[torch.Tensor]): each levels from backbone.
        Returns:
            outs (List[torch.Tensor]): selected features.
        """
        if isinstance(self.out_index, (list, tuple)):
            outs = [x[i] for i in self.out_index]
        else:
            outs = [x[self.out_index]]
        return outs
