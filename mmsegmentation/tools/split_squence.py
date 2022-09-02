import os.path as osp
from pathlib import Path
import argparse
from tqdm import tqdm
import cv2
from PIL import Image


DATA_SRC = Path("/mnt/data_src")


def get_videos():
    ignore = "使用不可"
    video_paths = []
    for data_p in DATA_SRC.glob("**/*.mp4"):
        if ignore in str(data_p):
            continue
        video_paths.append(data_p)
    return video_paths


def find_video_from_case(case):
    video_path = None
    for data_p in get_videos():
        if case in data_p.stem:
            video_path = str(data_p)
            break
    return video_path


def get_cv_db_frame_id(p, separator='_'):
    comp = str(p).split(osp.sep)
    cv = comp[-4]
    db = comp[-3].split(separator)[0]
    base = osp.splitext(comp[-1])[0]
    frame = base.split(separator)[1]
    return cv, db, int(frame)


def sample_frames(frame_num, clip_len, frame_interval):
    frame_inds = list(range(
        frame_num,
        frame_num - clip_len * frame_interval,
        -frame_interval))[::-1]
    return frame_inds


def get_frame_sequence(video_p, frame_inds):
    frames = []
    video = cv2.VideoCapture(str(video_p))
    for i in frame_inds:
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        frames.append(frame)
    video.release()
    return frames


def save_frame(fr, save_p, height, width):
    save_p.parent.mkdir(parents=True, exist_ok=True)
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    fr_p = Image.fromarray(fr)
    if fr_p.size != (width, height):
        fr_p = fr_p.resize((width, height), Image.LANCZOS)
    fr_p.save(save_p)


def main(args):
    with tqdm(args.data_dir.glob('**/label/*.png')) as pbar:
        for label_path in pbar:
            cv, case, frame_num = get_cv_db_frame_id(label_path)
            pbar.set_description(f'DB {case}/Frame {frame_num:06d}')

            frame_inds = sample_frames(frame_num, args.num_input_frames, args.frame_interval)
            assert len(frame_inds) == args.num_input_frames

            video_path = find_video_from_case(case)
            frames = get_frame_sequence(video_path, frame_inds)
            for i, idx in enumerate(frame_inds):
                pbar.set_postfix({'Frame': f'{idx:06d}'})
                if args.save_dir is None:
                    save_path = args.data_dir / cv / case / 'sequence' / f'frame_{idx:06d}.png'
                else:
                    save_path = args.save_dir / cv / case / 'sequence' / f'frame_{idx:06d}.png'
                save_frame(frames[i], save_path, args.height, args.width)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path)  # cvの手前までのパス
    parser.add_argument('--save-dir', type=Path, default=None)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--num-input-frames', type=int)
    parser.add_argument('--frame-interval', type=int, default=1)

    args = parser.parse_args()

    main(args)
