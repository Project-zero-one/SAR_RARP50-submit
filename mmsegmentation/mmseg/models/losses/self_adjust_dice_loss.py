# Copyright (c) Yamada, Atsushi. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss


def sadice_loss(pred,
                target,
                gamma=1.,
                smooth=1.,
                class_weight=None,
                valid_mask=None,
                reduction='mean',
                avg_factor=None):
    '''
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
        ("Dice Loss for Data-imbalanced NLP Tasks" paper)
   Args:
        alpha (float): a factor to push down the weight of easy examples
        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    '''
    assert pred.shape[0] == target.shape[0]

    probs = torch.gather(pred, dim=1, index=target.unsqueeze(1))
    probs_with_factor = torch.pow((1 - probs), gamma) * probs
    loss = 1 - (2 * probs_with_factor + smooth) / (probs_with_factor + 1 + smooth)

    final_weight = torch.ones(1, loss.size(1)).type_as(loss)
    if class_weight is not None:
        final_weight = final_weight * pred.new_tensor(class_weight)
    if valid_mask is not None:
        final_weight = final_weight * valid_mask
    loss = weight_reduce_loss(loss, final_weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class SelfAdjustDiceLoss(nn.Module):
    """SelfAdjustDiceLoss.

    This loss is proposed in `Dice Loss for Data-imbalanced NLP Tasks
                            <https://arxiv.org/pdf/1911.02855v3.pdf>`.

    Args:
        loss_type (str, optional): Binary or multi-class loss.
            Default: 'multi_class'. Options are "binary" and "multi_class".
        gamma (float):  a factor to push down the weight of easy examples
            Default: 1
        smooth (float): a factor added to both the nominator and the denominator for smoothing purposes
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
                 gamma=1.,
                 smooth=1.,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_sadice',
                 **kwards):
        super(SelfAdjustDiceLoss, self).__init__()
        assert isinstance(ignore_index, int), \
            'ignore_index must be of type int'

        self.gamma = gamma
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

        assert reduction_override in (None, 'none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert pred.shape == target.shape or \
            (pred.size(0) == target.size(0) and
             pred.shape[2:] == target.shape[1:]), \
            "The shape of pred doesn't match the shape of target"

        pred = F.softmax(pred, dim=1)
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()

        # target with shape [B, d_1, d_2, ...]
        # transform it's shape into [N, ]
        target = target.view(-1).contiguous()
        valid_mask = (target != self.ignore_index).view(-1, 1)
        # avoid raising error when using F.one_hot()
        target = torch.where(target == self.ignore_index, target.new_tensor(0),
                             target)

        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        loss = self.loss_weight * sadice_loss(
            pred,
            target.long(),
            gamma=self.gamma,
            smooth=self.smooth,
            class_weight=class_weight,
            valid_mask=valid_mask,
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
