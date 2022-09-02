import cv2
import numpy as np
import torch

from mmcv import Config
from mmcv.parallel import collate, scatter

from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose
from post_processor.prob_postproc import CheckTransProbZero


class ActionRecognizer:
    def __init__(self, config, checkpoint, stride=6, device='cuda:0'):
        """
        Video input action inference 

        Args:
            model: mmaction model
            stride (int, optional): Frame interval to predict. Defaults to 6.
            device (str, optional): Defaults to 'cuda:0'.
        """
        self.model = init_recognizer(
            Config.fromfile(config), 
            checkpoint, 
            device=torch.device(device))
        self.stride = stride
        self.device = device
        self.exclude_steps = ['LoadFrames']
        self.prob_post_processor = CheckTransProbZero()

        # prepare test pipeline from non-camera pipeline
        self.data = dict(img_shape=None, modality='RGB', label=-1)
        cfg = self.model.cfg
        sample_length = 0

        pipeline = cfg.data.test.pipeline
        pipeline_ = pipeline.copy()
        for step in pipeline:
            if 'SimpleSampleFrames' in step['type']:
                self.data['num_clips'] = 1
                self.data['clip_len'] = step['clip_len']
                self.data['frame_interval'] = step['frame_interval']
                pipeline_.remove(step)
            if step['type'] in self.exclude_steps:
                # remove step to decode frames
                pipeline_.remove(step)
        self.test_pipeline = Compose(pipeline_)

    def predict_video(self, video_path):
        frame_idxs, results = self._inference_video(
            self.model,
            video_path,
            self.data,
            self.test_pipeline,
            stride=self.stride,
            device=self.device,
            prob_post_processor=self.prob_post_processor
        )
        return frame_idxs, results

    def _inference_video(
            self, 
            model, 
            video_path, 
            data, 
            test_pipeline, 
            stride=6, 
            device='cuda:0', 
            prob_post_processor=None
        ):
        """
        Video input action inference 

        Args:
            model: mmaction model
            video_path (str): video_path
            data (dict): Input data
            test_pipeline: Composed pipeline to apply frame.
            stride (int, optional): Frame interval to predict. Defaults to 6.
            device (str, optional): Defaults to 'cuda:0'.

        Raises:
            Error: raise if frame is none

        Returns:
            results (list): list of tuples (frame index, predicted class, predicted scores)
        """
        results = list()
        frame_idxs = list()

        clip_len = data['clip_len']
        frame_interval = data['frame_interval']
        PRED_START_IDX = frame_interval * (clip_len - 1)
        pred_idx = frame_interval * (clip_len - 1)

        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs = []
        while pred_idx < num_frames:
            frame_clip = list()
            start_idx = pred_idx - frame_interval * (clip_len - 1)

            for frame_idx in np.arange(start_idx, pred_idx + frame_interval, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if frame is None:
                    raise Error(f'frame is None. index: {frame_idx}')

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_clip.append(frame)

            scores = self._inference_clip(model, data, test_pipeline, frame_clip, device)

            # append same results for frames before start index
            # if pred start at frame index(30), fill results for frame index(0, ..., 30-stride)
            if pred_idx == PRED_START_IDX:
                for _idx in range(0, PRED_START_IDX, stride):
                    print(f'fill {_idx}')
                    results.append(scores)
                    frame_idxs.append(_idx)

            # append frame index, predicted class and predicted scores
            results.append(scores)
            frame_idxs.append(pred_idx)

            pred_idx += stride

        cap.release()
        return frame_idxs, results

    def _inference_clip(self, model, data, test_pipeline, frame_clip, device='cuda:0'):
        inp_data = data.copy()
        inp_data['imgs'] = frame_clip
        inp_data['original_shape'] = frame_clip[0].shape[:2]
        inp_data['img_shape'] = frame_clip[0].shape[:2]

        inp_data = test_pipeline(inp_data)
        inp_data = collate([inp_data], samples_per_gpu=1)

        if next(model.parameters()).is_cuda:
            inp_data = scatter(inp_data, [device])[0]

        with torch.no_grad():
            scores = model(return_loss=False, **inp_data)[0]

        return scores


if __name__ == '__main__':
    config = '/mnt/cloudy_z/result/SAR-RARP50/ActionRecognition/VideoClassification/slowfast_r50_4x16x1_256e_kinetics400_rgb/coloraug/cv1/cv1_coloraug.py'
    checkpoint = '/mnt/cloudy_z/result/SAR-RARP50/ActionRecognition/VideoClassification/slowfast_r50_4x16x1_256e_kinetics400_rgb/coloraug/cv1/epoch_10.pth'
    video_path = '/mnt/cloudy_z/src/yharai/notebook/SAR-RARP50/testdataset/videos/video_34/video_left.avi'
    out_dir_path = '/mnt/cloudy_z/src/yharai/notebook/SAR-RARP50/testdataset/videos/video_34/'

    inference_action(
        config,
        checkpoint,
        video_path,
        out_dir_path=out_dir_path,
        stride=6,
        device='cuda:0'
    )
