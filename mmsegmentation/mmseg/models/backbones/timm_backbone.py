# Copyright (c) OpenMMLab. All rights reserved.
try:
    import timm
except ImportError:
    timm = None

from mmcv.cnn.bricks.registry import NORM_LAYERS
from mmcv.runner import BaseModule

from ..builder import BACKBONES


@BACKBONES.register_module()
class TIMMBackbone(BaseModule):
    """Wrapper to use backbones from timm library. More details can be found in
    `timm <https://github.com/rwightman/pytorch-image-models>`_ .
    [table with available encoders](https://smp.readthedocs.io/en/latest/encoders_timm.html)

    Args:
        model_name (str): Name of timm model to instantiate.
        pretrained (bool): Load pretrained weights if True.
        checkpoint_path (str): Path of checkpoint to load after
            model is initialized.
        in_channels (int): Number of input image channels. Default: 3.
        init_cfg (dict, optional): Initialization config dict
        **kwargs: Other timm & model specific arguments.

    Check features channel:
    >>> import timm
    >>> m = timm.create_model('MODEL_NAME', features_only=True)
    >>> m.feature_info.channels()
    [32, 64, 96, 224, 640]

    Check to be able to use pretrained weights or not
    (example EfficientNet)
    1. Access [here](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py)
    2. `default_cfgs` shows available backbone, and if that has `url` not empty, you can use pretrained weights.

    Or you can check pretrained model list like bellow code:
    >>> import timm
    >>> from pprint import pprint
    >>> timm.list_models(pretrained=True)
    ['adv_inception_v3',
    'bat_resnext26ts',
    ...]
    """

    def __init__(
        self,
        model_name,
        features_only=True,
        pretrained=True,
        checkpoint_path='',
        in_channels=3,
        init_cfg=None,
        **kwargs,
    ):
        if timm is None:
            raise RuntimeError('timm is not installed')
        super(TIMMBackbone, self).__init__(init_cfg)
        if 'norm_layer' in kwargs:
            kwargs['norm_layer'] = NORM_LAYERS.get(kwargs['norm_layer'])
        self.timm_model = timm.create_model(
            model_name=model_name,
            features_only=features_only,
            pretrained=pretrained,
            in_chans=in_channels,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )

        # Make unused parameters None
        self.timm_model.global_pool = None
        self.timm_model.fc = None
        self.timm_model.classifier = None

        # Hack to use pretrained weights from timm
        if pretrained or checkpoint_path:
            self._is_init = True

    def forward(self, x):
        features = self.timm_model(x)
        return features
