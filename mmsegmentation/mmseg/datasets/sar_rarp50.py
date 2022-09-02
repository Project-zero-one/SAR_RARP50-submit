# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from glob import glob
from pathlib import Path
import csv

import mmcv
from mmcv.utils import print_log
from .builder import DATASETS
from .custom import CustomDataset
from mmseg.datasets.pipelines import Compose, LoadAnnotations
from mmseg.utils import get_root_logger


@DATASETS.register_module()
class SARRARP50(CustomDataset):
    """SAR-RARP50 dataset
    https://www.synapse.org/#!Synapse:syn27618412/wiki/616881

    The sample files: ::
        training_set_1/
           ├── video_1
           │   ├── rgb
           │   │   ├── 000000000.png
           │   │   ├── 000000006.png
           │   │   └── ...
           │   ├── segmentation
           │   │   ├── 000000000.png
           │   │   ├── 000000060.png
           │   │   └── ...
           │   ├── action_continues.txt
           │   ├── action_discrete.txt
           │   └── video_left.avi
           ├── ...

    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory or image directory name.
        ann_dir (str): Path to annotation directory or annotation directory name.
        img_suffix (str): Suffix of images. Default: '.jpg'
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default: None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
        gt_seg_map_loader_cfg (dict, optional): build LoadAnnotations to
            load gt for evaluation, load from disk by default. Default: None.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """
    CLASSES = ('background',
               'Tool clasper',
               'Tool wrist',
               'Tool shaft',
               'Suturing needle',
               'Thread',
               'Suction tool',
               'Needle holder',
               'Clamps',
               'Catheter')
    PALETTE = [[0, 0, 0],
               [228, 26, 28],
               [55, 126, 184],
               [77, 175, 74],
               [152, 78, 163],
               [255, 127, 0],
               [255, 255, 51],
               [166, 86, 40],
               [247, 129, 191],
               [153, 153, 153]]

    def __init__(self,
                 pipeline,
                 img_dir,
                 ann_dir,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 dump_path=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=dict(flag='grayscale'),
                 file_client_args=dict(backend='disk')):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        self.gt_seg_map_loader = LoadAnnotations(
        ) if gt_seg_map_loader_cfg is None else LoadAnnotations(
            **gt_seg_map_loader_cfg)

        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient.infer_client(self.file_client_args)

        if test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` or `classes` should be specified when testing'

        # delete code `join paths if data_root is specified`

        # load annotations
        self.img_infos = self.load_annotations(self.data_root, self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

        # dump annotations as csv
        if dump_path:
            self.dump_annotations(dump_path)

    def load_annotations(self, data_root, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            data_root (str): Path to video directory.
            img_dir (str): image directory name.
            img_suffix (str): Suffix of images.
            ann_dir (str|None): annotation directory name.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            data_root = Path(self.data_root)
            # only read images if annotations are existing
            # data_root / video_1 / segmentation / 000000000.png
            # NOTE: pathlib.Path.globではsymbolic linkをglobで取ってこれないので、glob.globで取得する
            for seg_map_p in glob(str(data_root / f'**/{ann_dir}/*{img_suffix}'), recursive=True):
                seg_map = Path(*Path(seg_map_p).parts[-3:])  # video_1 / segmentation / 000000000.png
                img_info = dict(
                    ann=dict(seg_map=str(seg_map)),
                    video=seg_map.parts[0])
                img = Path(str(seg_map).replace(ann_dir, img_dir))
                img = img.with_suffix(img_suffix)
                img_info['filename'] = str(img)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.data_root
        results['seg_prefix'] = self.data_root
        if self.custom_classes:
            results['label_map'] = self.label_map

    def dump_annotations(self, dump_path):
        annotations = []
        for idx in range(len(self)):
            img_info = self.img_infos[idx]
            ann_info = self.get_ann_info(idx)
            results = dict(img_info=img_info, ann_info=ann_info)
            self.pre_pipeline(results)

            if results.get('img_prefix') is not None:
                img_filename = osp.join(results['img_prefix'],
                                        results['img_info']['filename'])
            else:
                img_filename = results['img_info']['filename']
            if results.get('seg_prefix', None) is not None:
                ann_filename = osp.join(results['seg_prefix'],
                                        results['ann_info']['seg_map'])
            else:
                ann_filename = results['ann_info']['seg_map']
            annotations.append([img_filename, ann_filename])

        with open(dump_path, 'w') as fw:
            writer = csv.writer(fw)
            writer.writerows(annotations)
        print_log(f'Dumped annotations to {dump_path}', logger=get_root_logger())
