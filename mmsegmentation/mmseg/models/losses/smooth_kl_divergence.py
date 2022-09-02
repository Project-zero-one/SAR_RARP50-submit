# Copyright (c) Yamada, Atsushi. All rights reserved.
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
import cv2
from .morphology import Dilate

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def kl_divergence(pred,
                  target,
                  valid_mask,
                  detach_target=True,
                  **kwargs):
    """Loss function using KL divergence.

    Args:
        pred (torch.Tensor): Predicted logits with shape (N,C,H,W).
        target (torch.Tensor): Target logits with shape (N,C,H,W).
        valid_mask (torch.Tensor): Not ignore pixels with shape (N,H,W).
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: class weighted Loss tensor with shape (1,).
    """
    assert pred.size() == target.size()

    if detach_target:
        target = target.detach()
    kld = F.kl_div(pred * valid_mask.unsqueeze(1), target, reduction='none')
    return kld


def constant_smooth(src, alpha=0.1, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dst = Dilate(kernel)(src)
    smooth = (dst - src) * alpha + src * (1 - alpha)
    return smooth


def gaussian_blur(src, sigma=1, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = Dilate(kernel)(src)
    dst = GaussianBlur(kernel_size, sigma=sigma)(dilated)
    smooth = torch.where(src > 0, src.float(), dst)
    return smooth


@LOSSES.register_module()
class SmoothKLDivergence(nn.Module):
    """Loss function of Kullback–Leibler divergence using label smoothing.

    Args:
        kernel_size (int): Kernel(filter) size for dilation or blur. 
            Must be odd number. Default to 5.
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_kl'.
        One of the following is required:
            alpha (float in [0., 1.]): Noise factor. The correct label of the
                training data is incorrect with probability alpha. Default to 0.1.
            sigma (float): Variant of normal distribution. Default to 1.
    """

    def __init__(self,
                 kernel_size=5,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_kl',
                 **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.alpha = kwargs.get('alpha')
        self.sigma = kwargs.get('sigma')
        self._loss_name = loss_name

        assert (self.alpha is None and self.sigma is not None) \
            or (self.alpha is not None and self.sigma is None)

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.log_softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes).permute(0, 3, 1, 2)
        valid_mask = (target != self.ignore_index).long()

        if self.alpha:
            # label smoothing with constant value
            soft_target = constant_smooth(one_hot_target, self.alpha, self.kernel_size)
        elif self.sigma:
            # label distribution by gaussian
            soft_target = gaussian_blur(one_hot_target, self.sigma, self.kernel_size)
        else:
            NotImplementedError

        loss = self.loss_weight * kl_divergence(
            pred,
            soft_target,
            valid_mask=valid_mask,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss

    @property
    def loss_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
