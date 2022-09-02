import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import mmcv
from mmseg.apis import init_segmentor, inference_segmentor


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


def load_models(config, checkpoint, gpus):
    model = init_segmentor(
        str(config),
        str(checkpoint),
        f'cuda:{gpus}' if gpus >= 0 else 'cpu',
    )
    if hasattr(model, 'module'):
        model = model.module
    return model


def predict_frame(model, img, return_logits=False):
    # predict
    return inference_segmentor(model, img, return_logits=return_logits)[0]


def overlay(img, mask, alpha=0.5, color=(0, 255, 255)):
    dst = img.astype(np.uint8).copy()
    ovl = np.zeros(dst.shape, dst.dtype)
    render_default(ovl, mask, color=color)
    add_overlay(dst, ovl, alpha=alpha)
    return dst


def main(args):
    if args.work_dir is not None:
        config = args.work_dir / f'{args.work_dir.name}.py'
        checkpoint = args.work_dir / 'latest.pth'
        save_dir = args.work_dir / 'prediction'
    else:
        config = args.config
        checkpoint = args.checkpoint
        save_dir = args.save_dir

    save_dir.mkdir(parents=True, exist_ok=True)

    model = load_models(config, checkpoint, args.gpus)
    print(model.CLASSES, model.PALETTE)
    dataset = [p for p in args.dataset_dir.glob('*/rgb/*.png')]


    for img_path in tqdm(dataset):
        anno_segm_img = str(img_path).replace("/rgb/", "/segmentation/")
        if not Path(anno_segm_img).exists():
            continue
        video_name = img_path.parts[-3]
        save_path = save_dir / video_name / "segmentation" / img_path.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # predict
        # img = mmcv.imread(img_path)[..., ::-1]  # for overlay(not need inference)
        mask = predict_frame(model, str(img_path))
        cv2.imwrite(str(save_path), mask)
        # # overlay
        # dst = overlay(img, mask)[..., ::-1]  # RGB->BGR
        # # save overlayed image
        # mmcv.imwrite(dst, str(save_path))

        # save_path = save_dir / video_name / "segmentation" / f"{img_path.stem}.npy"
        # save_path.parent.mkdir(parents=True, exist_ok=True)
        # scores = predict_frame(model, str(img_path), return_logits=True)
        # result = np.asarray(scores).astype("float16")
        # np.save(str(save_path), result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--work-dir', type=Path, default=None)
    parser.add_argument('--save-dir', type=Path, default=Path('./prediction'))

    args = parser.parse_args()

    main(args)
