# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/LikeLy-Journey/SegmenTron/blob/master/
segmentron/solver/loss.py (Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss


@weighted_loss
def mcc_loss(pred,
             target,
             valid_mask,
             smooth=1e-8,
             class_weight=None,
             ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            mcc_loss = binary_mcc_loss(
                pred[:, i],
                target[..., i],
                valid_mask=valid_mask,
                smooth=smooth)
            if class_weight is not None:
                mcc_loss *= class_weight[i]
            total_loss += mcc_loss
    return total_loss / num_classes


@weighted_loss
def binary_mcc_loss(pred, target, valid_mask, alpha=1, smooth=1e-8, **kwards):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    TP = torch.sum(torch.mul(pred, target) * valid_mask, dim=1)
    FP = torch.sum(torch.mul(pred, 1 - target) * valid_mask, dim=1)
    FN = torch.sum(torch.mul(1 - pred, target) * valid_mask, dim=1)
    TN = torch.sum(torch.mul(1 - pred, 1 - target) * valid_mask, dim=1)

    numerator = torch.mul(TP, TN) - torch.mul(FP, FN)
    denominator = torch.sqrt(
        torch.add(TP, FP, alpha=alpha)
        * torch.add(TP, FN, alpha=alpha)
        * torch.add(TN, FP, alpha=alpha)
        * torch.add(TN, FN, alpha=alpha)
    )

    # Adding 1 to the denominator to avoid divide-by-zero errors.
    mcc = torch.div(numerator.sum(), denominator.sum() + smooth)
    return 1 - mcc


@LOSSES.register_module()
class MCCLoss(nn.Module):
    """MCCLoss.

    This loss is proposed in `MATTHEWS CORRELATION COEFFICIENT LOSS FOR DEEP CONVOLUTIONAL
    NETWORKS: APPLICATION TO SKIN LESION SEGMENTATION <https://arxiv.org/abs/2010.13454>`_.

    Args:
        loss_type (str, optional): Binary or multi-class loss.
            Default: 'multi_class'. Options are "binary" and "multi_class".
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_dice'.
    """

    def __init__(self,
                 smooth=1e-8,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_mcc',
                 **kwards):
        super(MCCLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwards):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * mcc_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
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
