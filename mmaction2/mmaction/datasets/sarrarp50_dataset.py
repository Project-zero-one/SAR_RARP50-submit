# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from glob import glob
import random

import mmcv

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class SarRarp50Dataset(BaseDataset):
    """
    Example of a annotation file like action_discrete.txt

    .. code-block:: txt
        frame_ind class_id

        00000 0
        00006 1 
        00012 2 
        00018 3 

    Args:
        ann_file (str): Annotation file name.
        pipeline (list[dict | callable]): A sequence of data transforms.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
        dara_type (str): Type of input data. Support 'frame', 'video'
        load_start_ind (int): Start frame index to load.
            If model predict frames after index(30), set load_start_ind 30
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix,
                 modality='RGB',
                 data_type='frame',
                 load_start_ind=0,
                 **kwargs):
        assert (data_type == 'frame') or (data_type == 'video')
        self.data_type = data_type
         # if read annotation after id:30, set load_start_ind 30
        self.load_start_ind = load_start_ind
        super().__init__(ann_file, pipeline, data_prefix=data_prefix, modality=modality, **kwargs)

    def load_annotations(self):
        video_infos = []
        ann_file_paths = glob(osp.join(self.data_prefix, '*', self.ann_file))

        for ann_file_path in ann_file_paths:
            video_infos.extend(self._load_annotation(ann_file_path))

        return video_infos

    def _load_annotation(self, ann_file_path):
        video_infos = []

        with open(ann_file_path, 'r') as fin:
            for line in fin:
                line_split = line.strip().split(',')
                frame_ind = int(line_split[0])
                label = int(line_split[1])

                if frame_ind < self.load_start_ind:
                    continue
                
                video_name = osp.dirname(ann_file_path)
                if self.data_type == 'video':
                    data_path = osp.join(self.data_prefix, video_name, 'video_left.avi')
                
                elif self.data_type == 'frame':
                    if self.modality == 'RGB':
                        data_path = osp.join(self.data_prefix, video_name, 'rgb')
                    elif self.modality == 'Flow':
                        data_path = osp.join(self.data_prefix, video_name, 'flow')
                    
                video_infos.append(
                    dict(
                        data_path=data_path,
                        frame_ind=frame_ind,
                        label=label)
                )
        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        return self.pipeline(results)
