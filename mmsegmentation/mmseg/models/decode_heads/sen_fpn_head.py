# Copyright (c) Yamada, Atsushi. All rights reserved.
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from mmseg.ops import Upsample, resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..losses import accuracy


@HEADS.register_module()
class SenFPNHead(BaseDecodeHead):
    """Self-ensemble Feature Pyramid Networks.

    This head is inspired of `SenFormer
    <https://arxiv.org/abs/2111.13280>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
        reduction (str, optional): Ensemble method. Defaults to 'sum'.
            Options are "none", "mean" and "sum".
    """

    def __init__(self, feature_strides, reduction='sum', **kwargs):
        super().__init__(
            input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        # rescale
        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))
        # ensemble
        self.num_learners = len(self.in_index)**2 - 1  # in_indexのうちから少なくとも1つを選ぶ組合せ
        self.learners = nn.ModuleList()
        for _ in range(self.num_learners):
            learner = []
            if self.dropout is not None:
               learner.append(self.dropout)
            learner.append(
                nn.Conv2d(self.channels, self.num_classes, kernel_size=1))
            self.learners.append(nn.Sequential(*learner))

    def forward(self, inputs):
        x = self._transform_inputs(inputs)

        # rescale multi levels
        levels = [self.scale_heads[0](x[0])]
        out_shape = levels[0].shape[2:]
        for i in range(1, len(self.in_index)):
            # non inplace
            levels.append(resize(
                self.scale_heads[i](x[i]),
                size=out_shape,
                mode='bilinear',
                align_corners=self.align_corners))

        # each learner prediction
        logit_outputs = []  # save logits for learners' supervision
        prob_outputs = []  # save probabilty maps for ensemble prediction
        en = 0  # learner index
        for i in range(len(self.in_index)):
            # how many use feature levels
            for outs in itertools.combinations(levels, i + 1):
                x = torch.stack(outs, dim=0).sum(dim=0)  # feature aggregation by summention
                learner_pred = self.learners[en](x)  # learner prediction
                logit_outputs.append(learner_pred)
                prob_outputs.append(F.softmax(learner_pred, dim=1))
                en += 1

        # Ensemble prediction
        ensemble_pred = torch.stack(prob_outputs, dim=0)
        if self.reduction == 'sum':
            ensemble_pred = ensemble_pred.sum(dim=0)
        elif self.reduction == 'mean':
            ensemble_pred = ensemble_pred.mean(dim=0)
        return logit_outputs, ensemble_pred

    def forward_test(self, inputs, img_metas, test_cfg):
        _, ensemble_pred = self.forward(inputs)
        return ensemble_pred

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        logit_outputs, ensemble_pred = seg_logit  # unpack

        # Upscale outputs to the ground truth size
        gt_shape = seg_label.shape[2:]
        # Ensemble predicition
        ensemble_pred = resize(
            input=ensemble_pred,
            size=gt_shape,
            mode='bilinear',
            align_corners=self.align_corners)
        # Learners predicitions
        seg_log_logit = [resize(
            input=logit_outputs[i],
            size=gt_shape,
            mode='bilinear',
            align_corners=self.align_corners)
            for i in range(len(logit_outputs))]

        # Losses
        loss = dict()
        if self.sampler is not None:
            classic_weight = self.sampler.sample(ensemble_pred, seg_label)
            seg_weight = [self.sampler.sample(seg_logit, seg_label) for seg_logit in seg_log_logit]
        else:
            classic_weight = None
            seg_weight = [None] * len(seg_log_logit)
        seg_label = seg_label.squeeze(1)

        for loss_decode in self.loss_decode:
            name = loss_decode.loss_name
            if name + '_classic' not in loss:
                # Loss for the ensemble prediction
                loss[name + '_classic'] = loss_decode(
                    ensemble_pred,
                    seg_label,
                    weight=classic_weight,
                    ignore_index=self.ignore_index)
                # Loss for each learner
                loss[name + '_seg'] = torch.stack([
                    loss_decode(
                        seg_log_logit[i],
                        seg_label,
                        weight=seg_weight[i],
                        ignore_index=self.ignore_index)
                    for i in range(len(seg_log_logit))
                ], dim=0).sum() / len(seg_log_logit)
            else:
                # Loss for the ensemble prediction
                loss[name + '_classic'] += loss_decode(
                    ensemble_pred,
                    seg_label,
                    weight=classic_weight,
                    ignore_index=self.ignore_index)
                # Loss for each learner
                loss[name + '_seg'] += torch.stack([
                    loss_decode(
                        seg_log_logit[i],
                        seg_label,
                        weight=seg_weight[i],
                        ignore_index=self.ignore_index)
                    for i in range(len(seg_log_logit))
                ], dim=0).sum() / len(seg_log_logit)

        loss['acc_seg'] = accuracy(ensemble_pred, seg_label)
        return loss
