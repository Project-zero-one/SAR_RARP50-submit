import argparse
from pathlib import Path
import logging
import shutil
import numpy as np
import csv
import cv2
from tqdm import tqdm
import mmcv
from sarrarp50.utils import TqdmLoggingHandler
from predictor.mmseg_predictor import EnsembleSegmentor
from predictor.mmaction_predictor import ActionRecognizer
from post_processor.prob_postproc import CheckTransProbZero
from post_processor.smooth_prediction import smooth_pred_window


MMSEG_ENSEMBLE_DIR = Path("./weights/mmseg/ensemble")
MMACT_ENSEMBLE_DIR = Path("./weights/mmaction/ensemble")
ORIGINAL_IMAGE_SHAPE = (1080, 1920)


def main(args):
    test_root_dir = Path(args.test_dir)
    test_video_dirs = [n for n in test_root_dir.iterdir() if n.is_dir()]
    dst_root_dir = Path(args.prediction_dir)
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler("predictions.log"),
                        TqdmLoggingHandler()])
    do_action = True if args.task in ['actions' , 'multitask'] else False
    do_segmentation = True if args.task in ['segmentation' , 'multitask'] else False

    try:
        dst_root_dir.mkdir()
    except FileExistsError as e:
        if not args.overwrite:
            logging.error(f"{dst_root_dir.resolve()} exist, please delete it  manually or call this script using thhe --overwirte flag to replace previous predictions")
            return 1
        try:
            shutil.rmtree(dst_root_dir)
        except NotADirectoryError as e:
            logging.error(f"{dst_root_dir.resolve()} is not a directory, please remove it manually")
            return 1

    # creating the output directories
    for video_dir in test_video_dirs:
        (dst_root_dir/video_dir.name/'segmentation').mkdir(parents=True, exist_ok=True)

    # creating action recognition predictions:
    if do_action:
        for video_dir in tqdm(test_video_dirs, desc='generate action predicitons', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            video_p = video_dir / "video_left.avi"

            # mmaction predict
            mmact_results = []
            for config_path in MMACT_ENSEMBLE_DIR.glob("**/*.py"):
                model_path = config_path.parent / "latest.pth"
                action_recognizer = ActionRecognizer(str(config_path), str(model_path), stride=6, device='cuda:0')
                frame_idxs, results = action_recognizer.predict_video(str(video_p))
                mmact_results.append( results )
            
            # ensemble
            ensemble_scores = np.sum(mmact_results, axis=0) # (model num, frame num, class num) -> (frame num, class num)

            # post process
            prob_post_processor = CheckTransProbZero()
            ensemble_label_ids = []
            for scores in ensemble_scores:
                ensemble_label_ids.append( prob_post_processor(scores) )
            
            smoothed_label_ids =smooth_pred_window(ensemble_label_ids, window_size=1)
            smoothed_label_ids =smooth_pred_window(smoothed_label_ids, window_size=15)
            with open(dst_root_dir/video_dir.name/'action_discrete.txt', 'w') as f:
                for frame_idx, label_id in zip(frame_idxs, smoothed_label_ids):
                    f.writelines(f'{frame_idx},{label_id}\n')  # frame_index,label id
                logging.info(f" generate action recognition predicitons for {video_dir.name}, at {dst_root_dir/video_dir.name/'action_discrete.txt'}")

    # create 3 channel 1920x1080 segmentation images
    if do_segmentation:
        config_checkpoint_list = []
        for config_path in MMSEG_ENSEMBLE_DIR.glob("**/config.py"):
            config_checkpoint_list.append(
                [
                    config_path,
                    config_path.parent/"latest.pth"
                ]
            )
        segmentor = EnsembleSegmentor(config_checkpoint_list, ORIGINAL_IMAGE_SHAPE, gpus=0)
        for video_dir in tqdm(test_video_dirs, desc='generate segmentation predictions', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            rgb_paths, all_masks = segmentor.ensemble_majority_decision(video_dir)
            for i, rgb_p in enumerate(rgb_paths):
                mask = all_masks[i]
                cv2.imwrite(
                    str(dst_root_dir/video_dir.name/'segmentation'/rgb_p.name), 
                    mask)
            logging.info(f"generate segmentation predicitons for {video_dir.name}, under {dst_root_dir/video_dir.name/'segmentation'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir', help='root_directory of the test set')
    parser.add_argument('prediction_dir', help='directory to store predictions')
    parser.add_argument('task', help='segmentation or actions or multitask')
    parser.add_argument('-o','--overwrite', help='overwrite the predictions of prediction_dir', action='store_true')
    
    SystemExit(main(parser.parse_args()))