import numpy as np
from tqdm import tqdm
import shutil
import scipy.stats as stats
import mmcv
from mmseg.apis import init_segmentor, inference_segmentor


class EnsembleSegmentor:
    def __init__(self, config_checkpoint_list, ori_img_shape, gpus=0):
        self.config_checkpoint_list = config_checkpoint_list
        self.gpus = gpus
        self.ori_img_shape = ori_img_shape  # (height, width)

    def _load_models(self, config, checkpoint, gpus):
        model = init_segmentor(
            str(config),
            str(checkpoint),
            f'cuda:{gpus}' if gpus >= 0 else 'cpu',
        )
        if hasattr(model, 'module'):
            model = model.module
        return model

    def ensemble_majority_decision(self, video_dir):
        rgb_paths = sorted([img_p for img_p in (video_dir/'rgb').iterdir()])[::10]# processing one every 10 rgb images(10Hz) is equivilant to processing the video at 1Hz
        for i, [config, checkpoint] in enumerate(self.config_checkpoint_list):
            model = self._load_models(config, checkpoint, self.gpus)
            class_num = len(model.CLASSES)
            if i == 0:
                h, w = self.ori_img_shape
                all_model_masks = np.zeros((len(rgb_paths), class_num, h, w), dtype="float16")
            for i, rgb_p in enumerate(tqdm(rgb_paths, desc=f'segmentation files for {video_dir.name}', leave=False, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
                all_model_masks[i] += np.array(inference_segmentor(model, str(rgb_p), return_logits=True)[0], dtype="float16")

        # (フレーム数, c, h, w)の軸になってる
        ensemble_masks = np.argmax(all_model_masks, axis=1)
        return rgb_paths, ensemble_masks
