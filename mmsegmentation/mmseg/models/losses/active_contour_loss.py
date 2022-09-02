# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss

@weighted_loss
def active_contour_loss(pred,
                        target,
                        valid_mask,
                        w_region=1.0,
                        w_region_in=1.0,
                        w_region_out=1.0,
                        smooth=1e-8,
                        class_weight=None,
                        ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            ac_loss = binary_active_contour_loss(
                pred[:, i, :, :],
                target[:, :, :, i],
                valid_mask=valid_mask,
                w_region=w_region,
                w_region_in=w_region_in,
                w_region_out=w_region_out,
                smooth=smooth)
            if class_weight is not None:
                ac_loss *= class_weight[i]
            total_loss += ac_loss
    return total_loss / num_classes

@weighted_loss
def binary_active_contour_loss(pred,
                               target,
                               valid_mask,
                               w_region=1.0,
                               w_region_in=1.0,
                               w_region_out=1.0,
                               smooth=1e-8):
    """
    length term
    """
    x = pred[:,1:,:] - pred[:,:-1,:]
    y = pred[:,:,1:] - pred[:,:,:-1]

    delta_x = torch.pow(x[:,1:,:-2], 2)
    delta_y = torch.pow(y[:,:-2,1:], 2)
    delta_u = torch.abs(torch.add(delta_x, delta_y) * valid_mask[:, 1:-1, 1:-1])

    length = torch.mean(torch.sqrt(delta_u + smooth))

    """
    region term
    """
    region_in = torch.abs(torch.mean(
        pred
        * torch.pow(target - 1, 2)
        * valid_mask
    ))
    region_out = torch.abs(torch.mean(
        (1 - pred)
        * torch.pow(target, 2)
        * valid_mask
    ))
    region = w_region_in * region_in + w_region_out * region_out
    loss = length + w_region * region

    return loss

@LOSSES.register_module()
class ActiveContourLoss(nn.Module):
    """ActiveContourLoss.

    Paper: Chen, Xu, et al. "Learning active contour models for medical image segmentation." CVPR 2019.
    Code reference: <https://github.com/xuuuuuuchen/Active-Contour-Loss>

    Args:
        w_region (float): weight for region term
        w_region_in (float): weight for region inner term 
        w_region_out (float): weight for region outer term 
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
                 w_region=1.0,
                 w_region_in=1.0,
                 w_region_out=1.0,
                 smooth=1e-8,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_active_contour',
                 **kwards):
        super(ActiveContourLoss, self).__init__()
        self.w_region = w_region
        self.w_region_in = w_region_in
        self.w_region_out = w_region_out
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
        """
        Args:
            pred (Tensor): prediction tensor of shape [batch_size, C, H, W]
            target (Tensor): ground truth tensor of shape [batch_size, H, W]
            avg_factor (optional): not to use
            reduction_override (str, optional): if defined, override self.reduction

        Returns:
            loss (float)
        """
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
            num_classes=num_classes) # shape of [batch_size, H, W, C]
        valid_mask = (target != self.ignore_index).long()  # shape of [batch_size, H, W]

        loss = self.loss_weight * active_contour_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            w_region=self.w_region,
            w_region_in=self.w_region_in,
            w_region_out=self.w_region_out,
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
