import torch
import torch.nn.functional as F


class Dilate:
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, src: torch.Tensor):
        assert src.dim() == 4
        C = src.size(1)
        src = src.float()
        padding = (self.kernel.shape[0] - 1) // 2
        kernel = src.new_tensor(self.kernel).repeat(C, 1, 1, 1)
        dst = F.conv2d(src, kernel,
                       padding=padding, groups=C)
        dst = torch.clamp(dst, 0, 1)
        return dst
