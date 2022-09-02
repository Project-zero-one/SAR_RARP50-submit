# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from mmseg.ops import Upsample, resize
from ..losses import accuracy
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_indices, point_coords


class PointMatcher(nn.Module):
    """
        Simple Point Matcher
    """

    def __init__(self, dim, kernel_size=3):
        super(PointMatcher, self).__init__()
        self.match_conv = nn.Conv2d(dim * 2, 1, kernel_size, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_high, x_low = x
        x_low = F.upsample(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        certainty = self.match_conv(torch.cat([x_high, x_low], dim=1))
        return self.sigmoid(certainty)


class PointFlowModuleWithMaxAvgpool(nn.Module):
    def __init__(self, in_planes, dim=64, maxpool_size=8, avgpool_size=8, matcher_kernel_size=3,
                 edge_points=64):
        super(PointFlowModuleWithMaxAvgpool, self).__init__()
        self.dim = dim
        self.point_matcher = PointMatcher(dim, matcher_kernel_size)
        self.down_h = nn.Conv2d(in_planes, dim, 1)
        self.down_l = nn.Conv2d(in_planes, dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.maxpool_size = maxpool_size
        self.avgpool_size = avgpool_size
        self.edge_points = edge_points
        self.max_pool = nn.AdaptiveMaxPool2d((maxpool_size, maxpool_size), return_indices=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((avgpool_size, avgpool_size))
        self.edge_final = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes, out_channels=1, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        x_high, x_low = x
        stride_ratio = x_low.shape[2] / x_high.shape[2]
        x_high_embed = self.down_h(x_high)
        x_low_embed = self.down_l(x_low)
        N, C, H, W = x_low.shape
        N_h, C_h, H_h, W_h = x_high.shape

        certainty_map = self.point_matcher([x_high_embed, x_low_embed])
        avgpool_grid = self.avg_pool(certainty_map)
        _, _, map_h, map_w = certainty_map.size()
        avgpool_grid = F.interpolate(avgpool_grid, size=(map_h, map_w), mode="bilinear", align_corners=True)

        # edge part
        x_high_edge = x_high - x_high * avgpool_grid
        edge_pred = self.edge_final(x_high_edge)
        point_indices, point_coords = get_uncertain_point_coords_on_grid(edge_pred, num_points=self.edge_points)
        sample_x = point_indices % W_h * stride_ratio
        sample_y = point_indices // W_h * stride_ratio
        low_edge_indices = sample_x + sample_y * W
        low_edge_indices = low_edge_indices.unsqueeze(1).expand(-1, C, -1).long()
        high_edge_feat = point_sample(x_high, point_coords)
        low_edge_feat = point_sample(x_low, point_coords)
        affinity_edge = torch.bmm(high_edge_feat.transpose(2, 1), low_edge_feat).transpose(2, 1)
        affinity = self.softmax(affinity_edge)
        high_edge_feat = torch.bmm(affinity, high_edge_feat.transpose(2, 1)).transpose(2, 1)
        fusion_edge_feat = high_edge_feat + low_edge_feat

        # residual part
        maxpool_grid, maxpool_indices = self.max_pool(certainty_map)
        maxpool_indices = maxpool_indices.expand(-1, C, -1, -1)
        maxpool_grid = F.interpolate(maxpool_grid, size=(map_h, map_w), mode="bilinear", align_corners=True)
        x_indices = maxpool_indices % W_h * stride_ratio
        y_indices = maxpool_indices // W_h * stride_ratio
        low_indices = x_indices + y_indices * W
        low_indices = low_indices.long()
        x_high = x_high + maxpool_grid * x_high
        flattened_high = x_high.flatten(start_dim=2)
        high_features = flattened_high.gather(dim=2, index=maxpool_indices.flatten(start_dim=2)).view_as(maxpool_indices)
        flattened_low = x_low.flatten(start_dim=2)
        low_features = flattened_low.gather(dim=2, index=low_indices.flatten(start_dim=2)).view_as(low_indices)
        feat_n, feat_c, feat_h, feat_w = high_features.shape
        high_features = high_features.view(feat_n, -1, feat_h * feat_w)
        low_features = low_features.view(feat_n, -1, feat_h * feat_w)
        affinity = torch.bmm(high_features.transpose(2, 1), low_features).transpose(2, 1)
        affinity = self.softmax(affinity)  # b, n, n
        high_features = torch.bmm(affinity, high_features.transpose(2, 1)).transpose(2, 1)
        fusion_feature = high_features + low_features
        mp_b, mp_c, mp_h, mp_w = low_indices.shape
        low_indices = low_indices.view(mp_b, mp_c, -1)

        final_features = x_low.reshape(N, C, H * W).scatter(2, low_edge_indices, fusion_edge_feat)
        final_features = final_features.scatter(2, low_indices, fusion_feature).view(N, C, H, W)

        return final_features, edge_pred


@HEADS.register_module()
class PointFlowHead(BaseDecodeHead):
    '''PointFlow: Flowing Semantics Through Points for Aerial Image Segmentation
    This head is the implementation of `PointFlow
    <https://arxiv.org/abs/2103.06564>`_.

    Args:

    '''

    def __init__(self, pool_scales=(1, 2, 3, 6),  # PSP Module
                 reduce_dim=64, max_pool_size=8, avgpool_size=8, edge_points=32,  # PointFlow Module
                 use_edge=False,  # use JointEdgeSegLightLoss
                 **kwargs):
        super(PointFlowHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # FPN and PointFLow Module
        self.fpn_in = nn.ModuleList()
        self.fpn_out = nn.ModuleList()
        self.fpn_out_align = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            # lateral
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            # PFM
            pfm = PointFlowModuleWithMaxAvgpool(
                self.channels,
                dim=reduce_dim,
                maxpool_size=max_pool_size,
                avgpool_size=avgpool_size,
                edge_points=edge_points)
            # fpn output
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.fpn_in.append(l_conv)
            self.fpn_out_align.append(pfm)
            self.fpn_out.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.use_edge = use_edge

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build neck
        psp_out = self.psp_forward(inputs)
        f = psp_out
        fpn_feature_list = [f]
        edge_preds = []

        for i in reversed(range(len(inputs) - 1)):
            conv_x = inputs[i]
            # build laterals
            conv_x = self.fpn_in[i](conv_x)
            # operate PointFlowModule
            f, edge_pred = self.fpn_out_align[i]([f, conv_x])
            f = conv_x + f
            edge_preds.append(edge_pred)
            # build outputs
            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        # concat every feature map
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(resize(
                fpn_feature_list[i],
                size=output_size,
                mode='bilinear',
                align_corners=self.align_corners))
        fusion_out = torch.cat(fusion_list, 1)
        output = self.fpn_bottleneck(fusion_out)
        # channel of class
        output = self.cls_seg(output)
        return output, edge_preds

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        # return only mask (exclude edges)
        return self.forward(inputs)[0]

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit, edge_preds = seg_logit
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        edge_preds = [
            resize(
                input=edge_pred,
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            for edge_pred in edge_preds
        ]
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)  # squeeze Channel dim
        # create edge_mask from seg_label
        device = seg_label.device
        edge_label = torch.from_numpy(np.asarray([self._get_boundary(mask) for mask in seg_label])).clone().to(device)
        for loss_decode in self.loss_decode:
            if loss_decode.loss_name not in loss:
                if loss_decode.loss_name == 'loss_joint_edge_seg':
                    tmp = loss_decode(
                        [seg_logit, edge_preds],
                        [seg_label, edge_label],
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                    for k, v in tmp.items():
                        loss[k] = v
                else:
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logit,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
            else:
                if loss_decode.loss_name == 'loss_joint_edge_seg':
                    tmp = loss_decode(
                        [seg_logit, edge_preds],
                        [seg_label, edge_label],
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                    for k, v in tmp.items():
                        loss[k] += v
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_logit,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss

    def _get_boundary(self, mask, thicky=8):
        tmp = mask.data.cpu().numpy().astype('uint8')
        contour, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundary = np.zeros_like(tmp)
        boundary = cv2.drawContours(boundary, contour, -1, 1, thicky)
        boundary = boundary.astype(np.float32)
        return boundary
