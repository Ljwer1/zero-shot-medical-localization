import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from CLIP.clip import create_model
from CLIP.localization_adapter import LocalizationCLIP
from train_zero import ADAPTER_LAYERS, setup_seed
from utils import encode_text_with_prompt_ensemble
from visualize_zero import (
    build_prediction_mask,
    build_sample_panel,
    compose_figure,
    compute_score_map,
    device,
    get_checkpoint_path,
    process_score_map,
    resolve_mask_path,
    use_cuda,
)


def compute_overlap_metrics(prediction: np.ndarray, gt_mask: np.ndarray):
    prediction_fg = prediction > 0
    gt_fg = gt_mask > 0
    intersection = np.logical_and(prediction_fg, gt_fg).sum()
    prediction_area = prediction_fg.sum()
    gt_area = gt_fg.sum()
    union = np.logical_or(prediction_fg, gt_fg).sum()

    dice = (2.0 * intersection) / (prediction_area + gt_area + 1e-8)
    iou = intersection / (union + 1e-8)
    return float(dice), float(iou), int(prediction_area), int(gt_area)


def parse_args():
    parser = argparse.ArgumentParser(description="Rank localization samples by overlap with GT")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--pretrain", type=str, default="openai")
    parser.add_argument("--obj", type=str, default="Brain")
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--checkpoint_dir", type=str, default="./ckpt/localization/")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--img_size", type=int, default=240)
    parser.add_argument("--display_size", type=int, default=180)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--foreground_threshold", type=float, default=0.03)
    parser.add_argument("--apply_foreground_mask", type=int, default=1)
    parser.add_argument("--keep_largest_component", type=int, default=1)
    parser.add_argument("--min_component_area", type=int, default=20)
    parser.add_argument("--fill_holes", type=int, default=0)
    parser.add_argument("--renormalize_within_foreground", type=int, default=1)
    parser.add_argument("--threshold_mode", type=str, default="hybrid", choices=["absolute", "percentile", "hybrid"])
    parser.add_argument("--threshold_percentile", type=float, default=0.94)
    parser.add_argument("--score_blur_radius", type=float, default=1.0)
    parser.add_argument("--closing_kernel_size", type=int, default=5)
    parser.add_argument("--opening_kernel_size", type=int, default=3)
    parser.add_argument("--min_prediction_area", type=int, default=0)
    parser.add_argument("--output", type=str, default="./images/localization_topk.png")
    parser.add_argument("--seed", type=int, default=111)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)

    image_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size), Image.BICUBIC),
            transforms.ToTensor(),
        ]
    )

    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained=args.pretrain,
        require_pretrained=True,
    )
    clip_model.eval()

    model = LocalizationCLIP(clip_model=clip_model, features=ADAPTER_LAYERS).to(device)
    model.eval()
    checkpoint_path = get_checkpoint_path(args.checkpoint_dir, args.obj)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_localization_state_dict(checkpoint)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
        text_features = encode_text_with_prompt_ensemble(clip_model, args.obj, device)

    image_dir = os.path.join(args.data_path, f"{args.obj}_AD", args.split, "Ungood", "img")
    image_names = sorted(os.listdir(image_dir))
    records = []

    for index, image_name in enumerate(image_names, start=1):
        image_path = os.path.join(image_dir, image_name)
        mask_path = resolve_mask_path(image_path)

        image = Image.open(image_path).convert("RGB")
        resized_image = image.resize((args.img_size, args.img_size), Image.BICUBIC)
        image_tensor = image_transform(image).unsqueeze(0).to(device)
        score_map = compute_score_map(model, image_tensor, text_features, args.img_size)

        processed_score_map, foreground_mask = process_score_map(
            image=resized_image,
            score_map=score_map,
            foreground_threshold=args.foreground_threshold,
            apply_foreground_mask=bool(args.apply_foreground_mask),
            renormalize_within_foreground=bool(args.renormalize_within_foreground),
            score_blur_radius=args.score_blur_radius,
        )

        prediction, effective_threshold = build_prediction_mask(
            score_map=processed_score_map,
            foreground_mask=foreground_mask,
            threshold=args.threshold,
            apply_foreground_mask=bool(args.apply_foreground_mask),
            keep_largest=bool(args.keep_largest_component),
            min_component_area=args.min_component_area,
            fill_holes=bool(args.fill_holes),
            threshold_mode=args.threshold_mode,
            threshold_percentile=args.threshold_percentile,
            closing_kernel_size=args.closing_kernel_size,
            opening_kernel_size=args.opening_kernel_size,
        )

        gt_mask = Image.open(mask_path).convert("L").resize((args.img_size, args.img_size), Image.NEAREST)
        gt_mask = (np.asarray(gt_mask, dtype=np.uint8) > 127).astype(np.uint8)

        dice, iou, prediction_area, gt_area = compute_overlap_metrics(prediction, gt_mask)
        if prediction_area < args.min_prediction_area:
            continue

        records.append(
            {
                "image_name": image_name,
                "image_path": image_path,
                "mask_path": mask_path,
                "dice": dice,
                "iou": iou,
                "prediction_area": prediction_area,
                "gt_area": gt_area,
                "effective_threshold": float(effective_threshold),
            }
        )

        if index % 50 == 0:
            print(f"processed {index}/{len(image_names)} abnormal samples")

    records.sort(key=lambda item: (item["dice"], item["iou"]), reverse=True)
    selected = records[: args.top_k]

    print(f"Top {len(selected)} samples by Dice:")
    for rank, item in enumerate(selected, start=1):
        print(
            f"{rank}. {item['image_name']} | dice={item['dice']:.4f} | "
            f"iou={item['iou']:.4f} | pred_area={item['prediction_area']} | "
            f"gt_area={item['gt_area']} | thr={item['effective_threshold']:.4f}"
        )

    panels = []
    for item in selected:
        panels.append(
            build_sample_panel(
                image_path=item["image_path"],
                mask_path=item["mask_path"],
                model=model,
                text_features=text_features,
                image_transform=image_transform,
                display_size=args.display_size,
                img_size=args.img_size,
                threshold=args.threshold,
                overlay_alpha=0.45,
                foreground_threshold=args.foreground_threshold,
                apply_foreground_mask=bool(args.apply_foreground_mask),
                keep_largest=bool(args.keep_largest_component),
                min_component_area=args.min_component_area,
                fill_holes=bool(args.fill_holes),
                renormalize_within_foreground=bool(args.renormalize_within_foreground),
                threshold_mode=args.threshold_mode,
                threshold_percentile=args.threshold_percentile,
                score_blur_radius=args.score_blur_radius,
                closing_kernel_size=args.closing_kernel_size,
                opening_kernel_size=args.opening_kernel_size,
            )
        )

    compose_figure([(args.obj, panels)], display_size=args.display_size, output_path=args.output)
    print(f"saved top-{len(selected)} visualization to {args.output}")


if __name__ == "__main__":
    main()
