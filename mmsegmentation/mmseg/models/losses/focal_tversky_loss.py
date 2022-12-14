# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from
https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
(Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss
from .tversky_loss import binary_tversky_loss


@weighted_loss
def focal_tversky_loss(pred,
                       target,
                       smooth=1e-8,
                       class_weight=None,
                       alpha=0.3,
                       beta=0.7,
                       gamma=1.3,
                       ignore_index=255):
    num_classes = pred.shape[1]
    one_hot_target = F.one_hot(
        torch.clamp(target.long(), 0, num_classes - 1),
        num_classes=num_classes)
    valid_mask = (target != ignore_index).long()
    assert pred.shape[0] == one_hot_target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            focal_tversky_loss = torch.pow(binary_tversky_loss(
                pred[:, i],
                one_hot_target[..., i],
                valid_mask=valid_mask,
                smooth=smooth,
                alpha=alpha,
                beta=beta), gamma)
            if class_weight is not None:
                focal_tversky_loss *= class_weight[i]
            total_loss += focal_tversky_loss
    return total_loss / num_classes


@LOSSES.register_module()
class FocalTverskyLoss(nn.Module):
    """TverskyLoss. This loss is proposed in `Tversky loss function for image
    segmentation using 3D fully convolutional deep networks.
    <https://arxiv.org/abs/1706.05721>`_.
    Args:
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        alpha(float, in [0, 1]):
            The coefficient of false positives. Default: 0.3.
        beta (float, in [0, 1]):
            The coefficient of false negatives. Default: 0.7.
            Note: alpha + beta = 1.
        gamma (float, in [0, 3]):
            gamma > 1, focus on hard example(thus class imbalanced), gamma < 1, focus on easy example.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_tversky'.
    """

    def __init__(self,
                 smooth=1e-8,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 alpha=0.3,
                 beta=0.7,
                 gamma=1.3,
                 loss_name='loss_focal_tversky'):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._loss_name = loss_name

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

        pred = F.softmax(pred, dim=1)

        loss = self.loss_weight * focal_tversky_loss(
            pred,
            target,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            class_weight=class_weight,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            **kwargs)
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
