import argparse
from pathlib import Path
from tqdm import trange
import numpy as np
import cv2
import imageio
from mmseg.apis import init_segmentor, inference_segmentor


class VideoWriter:
    def __init__(self, video_path, fps):
        self.writer = imageio.get_writer(video_path, format='ffmpeg', fps=fps)

    def release(self):
        self.writer.close()

    def write(self, image):
        self.writer.append_data(image)


def add_overlay(image, overlay, alpha=0.5):
    gray = cv2.cvtColor(overlay, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    blend = np.uint8(img_gray * (overlay / 255))
    # blend = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    bg = cv2.bitwise_and(image, image, mask=mask_inv)
    fg = cv2.bitwise_and(blend, blend, mask=mask)
    cv2.add(bg, fg, image)


def render_default(image, mask, color=(255, 255, 255)):
    image[mask > 0] = color


def load_models(config, checkpoint, gpus=-1):
    model = init_segmentor(
        str(config),
        str(checkpoint),
        f'cuda:{gpus}' if gpus >= 0 else 'cpu')
    if hasattr(model, 'module'):
        model = model.module
    return model


def pre_process(frame):
    # no process
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def predict_frame(model, img):
    # predict
    mask = inference_segmentor(model, img)[0]
    return mask


def overlay(img, mask, alpha=0.5, color=(0, 255, 255)):
    dst = img.astype(np.uint8).copy()
    ovl = np.zeros(dst.shape, dst.dtype)
    render_default(ovl, mask, color=color)
    add_overlay(dst, ovl, alpha=alpha)
    return dst


def main(args):
    if args.work_dir is not None:
        # config = args.work_dir / f'{args.work_dir.name}.py'
        config = args.work_dir / 'config.py'
        # checkpoint = args.work_dir / 'latest.pth'
        checkpoint = args.work_dir / 'model.pth'
        save_dir = args.work_dir / 'prediction'
    else:
        config = args.config
        checkpoint = args.checkpoint
        save_dir = args.save_dir

    model = load_models(config, checkpoint, args.gpus)
    print(model.CLASSES, model.PALETTE)
    sample_inds = list(range(args.num_input_frames * args.frame_interval - 1,
                             0,
                             -args.frame_interval))[::-1]

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{args.video_path.stem}_{args.start_frame}_{args.end_frame}.mp4"

    video = cv2.VideoCapture(str(args.video_path))
    video.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    writer = VideoWriter(save_path, fps=30.0)

    frame_buffer = []
    for i in trange(args.start_frame, args.end_frame + 1):
        # read frame
        ret, frame = video.read()  # BGR
        if not ret:
            break
        frame = pre_process(frame)
        frame_buffer.append(frame)

        # predict
        if args.num_input_frames <= 1:
            mask = predict_frame(model, frame)  # input BGR
        else:
            if len(frame_buffer) < args.num_input_frames * args.frame_interval:
                continue
            else:
                buffer_arr = np.array(frame_buffer)
                # 現在フレームから過去フレームに向かってframe_interval分sampleする
                if args.frame_interval > 1:
                    buffer_arr = buffer_arr[sample_inds]
                mask = predict_frame(model, buffer_arr)
        frame_buffer.pop(0)

        # overlay
        dst = overlay(frame[..., ::-1], mask)
        # frame番号表示
        cv2.putText(dst, f'frame: {i:06d}', (25, 40),
                    cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255))
        # save overlayed image
        writer.write(dst)
    video.release()
    writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # work-dirがあればconfig, checkpoint, save-dirいらない
    parser.add_argument('--work-dir', type=Path, default=None)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--save-dir', type=Path, default=Path('./prediction'))
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--video-path', type=Path)
    parser.add_argument('--start-frame', type=int)
    parser.add_argument('--end-frame', type=int)
    parser.add_argument('--num-input-frames', type=int, default=1)
    parser.add_argument('--frame-interval', type=int, default=1)

    args = parser.parse_args()

    main(args)
