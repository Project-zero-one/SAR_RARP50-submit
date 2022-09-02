import cv2
import os.path as osp
import mmcv
import numpy as np
import torch

from mmcv import Config
from mmcv.parallel import collate, scatter

from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose


def save_action_discrete(results, out_dir_path):
    with open(osp.join(out_dir_path, 'action_discrete.txt'), mode='w') as f:
        for result in results:
            f.writelines(f'{result[0]:09d},{result[1]}\n')  # frame_index,class


def score_to_class(scores):
    pred_class = np.argmax(scores)
    return pred_class


def inference_video(model, video_path, data, test_pipeline, stride=6, device='cuda:0'):
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
        results (list): list of tuples (frame index, predicted class)
    """
    results = list()

    clip_len = data['clip_len']
    frame_interval = data['frame_interval']
    PRED_START_IDX = frame_interval * (clip_len - 1)
    pred_idx = frame_interval * (clip_len - 1)

    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while pred_idx < num_frames:
        frame_clip = list()
        start_idx = pred_idx - frame_interval * (clip_len - 1)
        frame_idxs = list(range(start_idx, pred_idx + frame_interval, frame_interval))
        frame_idxs = np.array(frame_idxs, dtype=np.int)

        for frame_idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if frame is None:
                raise Error(f'frame is None. index: {frame_idx}')

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_clip.append(frame)

        scores = inference_clip(model, data, test_pipeline, frame_clip, device)
        pred_class = score_to_class(scores)
        print(f'[DBG] pred_idx: {pred_idx}  pred_class: {pred_class}')

        # append same results for frames before start index
        # if pred start at frame index(30), fill results for frame index(0, ..., 30-stride)
        if pred_idx == PRED_START_IDX:
            for _idx in range(0, PRED_START_IDX, stride):
                print(f'fill {_idx}')
                results.append((_idx, pred_class))

        # append frame index and predicted class
        results.append((pred_idx, pred_class))

        pred_idx += stride

    cap.release()
    return results


def inference_clip(model, data, test_pipeline, frame_clip, device='cuda:0'):
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


def inference_action(
    config: str,
    checkpoint: str,
    video_path: str,
    out_dir_path=None,
    stride=6,
    device='cuda:0'
):
    EXCLUED_STEPS = ['LoadFrames']

    device = torch.device(device)
    cfg = Config.fromfile(config)

    model = init_recognizer(cfg, checkpoint, device=device)
    data = dict(img_shape=None, modality='RGB', label=-1)

    # prepare test pipeline from non-camera pipeline
    cfg = model.cfg
    sample_length = 0

    pipeline = cfg.data.test.pipeline
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SimpleSampleFrames' in step['type']:
            data['num_clips'] = 1
            data['clip_len'] = step['clip_len']
            data['frame_interval'] = step['frame_interval']
            pipeline_.remove(step)
        if step['type'] in EXCLUED_STEPS:
            # remove step to decode frames
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)

    results = inference_video(
        model,
        video_path,
        data,
        test_pipeline,
        stride=stride,
        device=device
    )

    if out_dir_path is None:
        out_dir_path = osp.dirname(video_path)

    save_action_discrete(results, out_dir_path)


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
