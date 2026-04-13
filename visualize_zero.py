import argparse
import os
from collections import deque
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from torchvision import transforms

from CLIP.clip import create_model
from CLIP.localization_adapter import LocalizationCLIP
from dataset.medical_localization import LOCALIZATION_CLASS_NAMES
from train_zero import ADAPTER_LAYERS, build_layer_probability_map, setup_seed
from utils import encode_text_with_prompt_ensemble, fuse_layer_outputs


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def get_checkpoint_path(checkpoint_dir: str, obj: str) -> str:
    return os.path.join(
        checkpoint_dir,
        f"{obj}_localization_spatial_text_guided_{ADAPTER_LAYERS[0]}_{ADAPTER_LAYERS[1]}.pth",
    )


def resolve_mask_path(image_path: str) -> str:
    return image_path.replace("\\img\\", "\\anomaly_mask\\").replace("/img/", "/anomaly_mask/")


def normalize_map(score_map: np.ndarray) -> np.ndarray:
    score_map = score_map.astype(np.float32)
    return (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)


def normalize_map_in_mask(score_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(score_map, dtype=np.float32)
    valid = mask > 0
    if not np.any(valid):
        return normalize_map(score_map)

    foreground_values = score_map[valid].astype(np.float32)
    value_min = foreground_values.min()
    value_max = foreground_values.max()
    normalized[valid] = (foreground_values - value_min) / (value_max - value_min + 1e-8)
    return normalized


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_foreground_mask(image: Image.Image, threshold: float) -> np.ndarray:
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    grayscale = image_array.mean(axis=-1)
    return (grayscale >= threshold).astype(np.uint8)


def find_connected_components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    height, width = mask.shape
    visited = np.zeros((height, width), dtype=bool)
    components: List[List[Tuple[int, int]]] = []

    for row in range(height):
        for col in range(width):
            if mask[row, col] == 0 or visited[row, col]:
                continue

            queue = deque([(row, col)])
            visited[row, col] = True
            component: List[Tuple[int, int]] = []

            while queue:
                current_row, current_col = queue.popleft()
                component.append((current_row, current_col))

                for next_row in range(max(0, current_row - 1), min(height, current_row + 2)):
                    for next_col in range(max(0, current_col - 1), min(width, current_col + 2)):
                        if visited[next_row, next_col] or mask[next_row, next_col] == 0:
                            continue
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))

            components.append(component)

    return components


def components_to_mask(shape: Tuple[int, int], components: List[List[Tuple[int, int]]]) -> np.ndarray:
    output = np.zeros(shape, dtype=np.uint8)
    for component in components:
        for row, col in component:
            output[row, col] = 1
    return output


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask

    components = find_connected_components(mask)
    kept_components = [component for component in components if len(component) >= min_area]
    return components_to_mask(mask.shape, kept_components)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    components = find_connected_components(mask)
    if not components:
        return mask
    largest_component = max(components, key=len)
    return components_to_mask(mask.shape, [largest_component])


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    inverse_mask = 1 - mask.astype(np.uint8)
    height, width = inverse_mask.shape
    visited = np.zeros((height, width), dtype=bool)
    queue = deque()

    for row in range(height):
        for col in [0, width - 1]:
            if inverse_mask[row, col] == 1 and not visited[row, col]:
                visited[row, col] = True
                queue.append((row, col))

    for col in range(width):
        for row in [0, height - 1]:
            if inverse_mask[row, col] == 1 and not visited[row, col]:
                visited[row, col] = True
                queue.append((row, col))

    while queue:
        current_row, current_col = queue.popleft()
        for next_row in range(max(0, current_row - 1), min(height, current_row + 2)):
            for next_col in range(max(0, current_col - 1), min(width, current_col + 2)):
                if visited[next_row, next_col] or inverse_mask[next_row, next_col] == 0:
                    continue
                visited[next_row, next_col] = True
                queue.append((next_row, next_col))

    holes = (inverse_mask == 1) & (~visited)
    filled_mask = mask.copy().astype(np.uint8)
    filled_mask[holes] = 1
    return filled_mask


def smooth_score_map(score_map: np.ndarray, blur_radius: float) -> np.ndarray:
    if blur_radius <= 0:
        return score_map.astype(np.float32)

    score_image = Image.fromarray(np.uint8(np.clip(score_map, 0.0, 1.0) * 255), mode="L")
    score_image = score_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.asarray(score_image, dtype=np.float32) / 255.0


def process_score_map(
    image: Image.Image,
    score_map: np.ndarray,
    foreground_threshold: float,
    apply_foreground_mask: bool,
    renormalize_within_foreground: bool,
    score_blur_radius: float,
):
    foreground_mask = build_foreground_mask(image, threshold=foreground_threshold)

    masked_score_map = score_map.copy()
    if apply_foreground_mask:
        masked_score_map = masked_score_map * foreground_mask
    if renormalize_within_foreground and apply_foreground_mask:
        masked_score_map = normalize_map_in_mask(masked_score_map, foreground_mask)
    masked_score_map = smooth_score_map(masked_score_map, blur_radius=score_blur_radius)
    return masked_score_map, foreground_mask


def _normalize_kernel_size(kernel_size: int) -> int:
    if kernel_size <= 1:
        return 0
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = _normalize_kernel_size(kernel_size)
    if kernel_size == 0:
        return mask.astype(np.uint8)
    mask_image = Image.fromarray(np.uint8(mask) * 255, mode="L")
    dilated = mask_image.filter(ImageFilter.MaxFilter(size=kernel_size))
    return (np.asarray(dilated, dtype=np.uint8) > 127).astype(np.uint8)


def erode_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = _normalize_kernel_size(kernel_size)
    if kernel_size == 0:
        return mask.astype(np.uint8)
    mask_image = Image.fromarray(np.uint8(mask) * 255, mode="L")
    eroded = mask_image.filter(ImageFilter.MinFilter(size=kernel_size))
    return (np.asarray(eroded, dtype=np.uint8) > 127).astype(np.uint8)


def close_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = _normalize_kernel_size(kernel_size)
    if kernel_size == 0:
        return mask.astype(np.uint8)
    return erode_mask(dilate_mask(mask, kernel_size), kernel_size)


def open_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = _normalize_kernel_size(kernel_size)
    if kernel_size == 0:
        return mask.astype(np.uint8)
    return dilate_mask(erode_mask(mask, kernel_size), kernel_size)


def compute_effective_threshold(
    score_map: np.ndarray,
    foreground_mask: np.ndarray,
    threshold: float,
    threshold_mode: str,
    threshold_percentile: float,
) -> float:
    if threshold_mode == "absolute":
        return float(threshold)

    valid = foreground_mask > 0
    if not np.any(valid):
        return float(threshold)

    values = score_map[valid].astype(np.float32)
    percentile_value = float(np.quantile(values, np.clip(threshold_percentile, 0.0, 1.0)))
    if threshold_mode == "percentile":
        return percentile_value
    return max(float(threshold), percentile_value)


def build_prediction_mask(
    score_map: np.ndarray,
    foreground_mask: np.ndarray,
    threshold: float,
    apply_foreground_mask: bool,
    keep_largest: bool,
    min_component_area: int,
    fill_holes: bool,
    threshold_mode: str,
    threshold_percentile: float,
    closing_kernel_size: int,
    opening_kernel_size: int,
):
    effective_threshold = compute_effective_threshold(
        score_map=score_map,
        foreground_mask=foreground_mask if apply_foreground_mask else np.ones_like(score_map, dtype=np.uint8),
        threshold=threshold,
        threshold_mode=threshold_mode,
        threshold_percentile=threshold_percentile,
    )

    prediction = (score_map >= effective_threshold).astype(np.uint8)
    if apply_foreground_mask:
        prediction = prediction * foreground_mask

    prediction = close_mask(prediction, kernel_size=closing_kernel_size)
    prediction = open_mask(prediction, kernel_size=opening_kernel_size)
    prediction = remove_small_components(prediction, min_area=min_component_area)
    if keep_largest and prediction.sum() > 0:
        prediction = keep_largest_component(prediction)
    if fill_holes and prediction.sum() > 0:
        prediction = fill_mask_holes(prediction)
    return prediction, effective_threshold


def list_abnormal_samples(
    data_path: str,
    obj: str,
    split: str,
    num_samples: int,
    start_index: int,
) -> List[Tuple[str, str]]:
    image_dir = os.path.join(data_path, f"{obj}_AD", split, "Ungood", "img")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Abnormal image directory not found: {image_dir}")

    image_names = sorted(os.listdir(image_dir))
    selected_names = image_names[start_index : start_index + num_samples]
    if not selected_names:
        raise ValueError(
            f"No abnormal samples found for class={obj}, split={split}, "
            f"start_index={start_index}, num_samples={num_samples}"
        )

    samples = []
    for image_name in selected_names:
        image_path = os.path.join(image_dir, image_name)
        mask_path = resolve_mask_path(image_path)
        samples.append((image_path, mask_path))
    return samples


def compute_score_map(
    model: LocalizationCLIP,
    image_tensor: torch.Tensor,
    text_features: torch.Tensor,
    img_size: int,
) -> np.ndarray:
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
        _, seg_patch_tokens = model(image_tensor, text_context=text_features)
        seg_layer_weights = model.get_fusion_weights().to(device)

        anomaly_maps = []
        for layer_tokens in seg_patch_tokens:
            anomaly_prob = build_layer_probability_map(layer_tokens, text_features, img_size)
            anomaly_maps.append(anomaly_prob[:, 1, :, :])

        final_score_map = fuse_layer_outputs(anomaly_maps, seg_layer_weights)
    return normalize_map(final_score_map.squeeze(0).detach().cpu().numpy())


def resize_score_map(score_map: np.ndarray, display_size: int) -> np.ndarray:
    score_image = Image.fromarray(np.uint8(np.clip(score_map, 0.0, 1.0) * 255), mode="L")
    score_image = score_image.resize((display_size, display_size), Image.BILINEAR)
    return np.asarray(score_image, dtype=np.float32) / 255.0


def apply_jet_colormap(score_map: np.ndarray) -> np.ndarray:
    x = np.clip(score_map, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def build_heatmap_overlay(image: Image.Image, score_map: np.ndarray, alpha: float) -> Image.Image:
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    heatmap = apply_jet_colormap(score_map)
    blended = np.clip((1.0 - alpha) * image_array + alpha * heatmap, 0.0, 1.0)
    return Image.fromarray(np.uint8(blended * 255))


def build_binary_mask(mask_array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(mask_array) * 255, mode="L").convert("RGB")


def build_ground_truth_image(mask_path: str, display_size: int) -> Image.Image:
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((display_size, display_size), Image.NEAREST)
    mask_array = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.uint8)
    return build_binary_mask(mask_array)


def add_border(image: Image.Image, border_size: int = 1) -> Image.Image:
    return ImageOps.expand(image, border=border_size, fill=(40, 40, 40))


def build_sample_panel(
    image_path: str,
    mask_path: str,
    model: LocalizationCLIP,
    text_features: torch.Tensor,
    image_transform: transforms.Compose,
    display_size: int,
    img_size: int,
    threshold: float,
    overlay_alpha: float,
    foreground_threshold: float,
    apply_foreground_mask: bool,
    keep_largest: bool,
    min_component_area: int,
    fill_holes: bool,
    renormalize_within_foreground: bool,
    threshold_mode: str,
    threshold_percentile: float,
    score_blur_radius: float,
    closing_kernel_size: int,
    opening_kernel_size: int,
) -> Dict[str, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    display_image = image.resize((display_size, display_size), Image.BICUBIC)
    model_image = image.resize((img_size, img_size), Image.BICUBIC)
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    score_map = compute_score_map(model, image_tensor, text_features, img_size)
    processed_score_map, foreground_mask = process_score_map(
        image=model_image,
        score_map=score_map,
        foreground_threshold=foreground_threshold,
        apply_foreground_mask=apply_foreground_mask,
        renormalize_within_foreground=renormalize_within_foreground,
        score_blur_radius=score_blur_radius,
    )
    prediction, effective_threshold = build_prediction_mask(
        score_map=processed_score_map,
        foreground_mask=foreground_mask,
        threshold=threshold,
        apply_foreground_mask=apply_foreground_mask,
        keep_largest=keep_largest,
        min_component_area=min_component_area,
        fill_holes=fill_holes,
        threshold_mode=threshold_mode,
        threshold_percentile=threshold_percentile,
        closing_kernel_size=closing_kernel_size,
        opening_kernel_size=opening_kernel_size,
    )

    display_score_map = resize_score_map(processed_score_map, display_size)
    prediction_image = Image.fromarray(np.uint8(prediction) * 255, mode="L").resize(
        (display_size, display_size), Image.NEAREST
    )

    heatmap = build_heatmap_overlay(display_image, display_score_map, alpha=overlay_alpha)
    prediction_image = prediction_image.convert("RGB")
    ground_truth_image = build_ground_truth_image(mask_path, display_size)

    return {
        "image": add_border(display_image),
        "heatmap": add_border(heatmap),
        "result": add_border(prediction_image),
        "ground_truth": add_border(ground_truth_image),
        "name": os.path.basename(image_path),
        "effective_threshold": effective_threshold,
    }


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    x_center: int,
    y_top: int,
    fill: Tuple[int, int, int],
) -> None:
    text_width, text_height = measure_text(draw, text, font)
    draw.text((x_center - text_width // 2, y_top), text, font=font, fill=fill)


def draw_dashed_separator(
    draw: ImageDraw.ImageDraw,
    x: int,
    y0: int,
    y1: int,
    dash_length: int = 12,
    gap_length: int = 10,
) -> None:
    current = y0
    while current < y1:
        draw.line((x, current, x, min(current + dash_length, y1)), fill=(0, 0, 0), width=3)
        current += dash_length + gap_length


def compose_figure(
    grouped_samples: Sequence[Tuple[str, List[Dict[str, Image.Image]]]],
    display_size: int,
    output_path: str,
) -> None:
    row_keys = ["image", "heatmap", "result", "ground_truth"]
    row_labels = ["(a) Image", "(b) Heatmap", "(c) Result", "(d) Ground Truth"]

    title_font = load_font(30)
    label_font = load_font(24)
    sample_font = load_font(14)
    tmp_canvas = Image.new("RGB", (32, 32), color="white")
    tmp_draw = ImageDraw.Draw(tmp_canvas)

    row_label_width = max(measure_text(tmp_draw, label, label_font)[0] for label in row_labels) + 36
    title_height = measure_text(tmp_draw, "Brain MRI", title_font)[1]
    sample_name_height = measure_text(tmp_draw, "00000_00.png", sample_font)[1]

    outer_margin = 24
    group_gap = 56
    sample_gap = 16
    row_gap = 16
    header_gap = 16

    group_widths = []
    for _, samples in grouped_samples:
        group_width = len(samples) * (display_size + 2) + max(0, len(samples) - 1) * sample_gap
        group_widths.append(group_width)

    content_width = sum(group_widths) + max(0, len(group_widths) - 1) * group_gap
    canvas_width = outer_margin * 2 + row_label_width + content_width
    canvas_height = (
        outer_margin * 2
        + title_height
        + header_gap
        + sample_name_height
        + header_gap
        + len(row_keys) * (display_size + 2)
        + (len(row_keys) - 1) * row_gap
    )

    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
    draw = ImageDraw.Draw(canvas)

    x_cursor = outer_margin + row_label_width
    group_centers = []
    for group_width in group_widths:
        group_centers.append(x_cursor + group_width // 2)
        x_cursor += group_width + group_gap

    y_title = outer_margin
    y_sample_names = outer_margin + title_height + header_gap
    y_tiles = y_sample_names + sample_name_height + header_gap

    for group_idx, (class_name, samples) in enumerate(grouped_samples):
        draw_centered_text(draw, class_name, title_font, group_centers[group_idx], y_title, fill=(0, 0, 0))

        sample_x = outer_margin + row_label_width + sum(group_widths[:group_idx]) + group_idx * group_gap
        for sample in samples:
            sample_center = sample_x + (display_size + 2) // 2
            draw_centered_text(draw, sample["name"], sample_font, sample_center, y_sample_names, fill=(90, 90, 90))

            for row_idx, row_key in enumerate(row_keys):
                y_position = y_tiles + row_idx * ((display_size + 2) + row_gap)
                canvas.paste(sample[row_key], (sample_x, y_position))

            sample_x += (display_size + 2) + sample_gap

        if group_idx < len(grouped_samples) - 1:
            separator_x = (
                outer_margin
                + row_label_width
                + sum(group_widths[: group_idx + 1])
                + group_idx * group_gap
                + group_gap // 2
            )
            draw_dashed_separator(draw, separator_x, y_title, canvas_height - outer_margin)

    for row_idx, row_label in enumerate(row_labels):
        label_width, label_height = measure_text(draw, row_label, label_font)
        row_y = y_tiles + row_idx * ((display_size + 2) + row_gap) + ((display_size + 2) - label_height) // 2
        draw.text((outer_margin, row_y), row_label, font=label_font, fill=(0, 0, 0))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate localization-only visualization panels")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--pretrain", type=str, default="openai")
    parser.add_argument("--obj", nargs="+", default=["Brain"], choices=LOCALIZATION_CLASS_NAMES)
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--checkpoint_dir", type=str, default="./ckpt/localization/")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--img_size", type=int, default=240)
    parser.add_argument("--display_size", type=int, default=180)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
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
    parser.add_argument("--output", type=str, default="./images/localization_visualize.png")
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

    grouped_samples = []
    for class_name in args.obj:
        checkpoint_path = get_checkpoint_path(args.checkpoint_dir, class_name)
        if not os.path.exists(checkpoint_path):
            print(f"skip {class_name}: checkpoint not found at {checkpoint_path}")
            continue

        samples = list_abnormal_samples(
            data_path=args.data_path,
            obj=class_name,
            split=args.split,
            num_samples=args.num_samples,
            start_index=args.start_index,
        )

        model = LocalizationCLIP(clip_model=clip_model, features=ADAPTER_LAYERS).to(device)
        model.eval()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_localization_state_dict(checkpoint)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
            text_features = encode_text_with_prompt_ensemble(clip_model, class_name, device)

        rendered_samples = []
        print(f"rendering {class_name}:")
        for image_path, mask_path in samples:
            print(f"  {os.path.basename(image_path)}")
            sample_panel = build_sample_panel(
                image_path=image_path,
                mask_path=mask_path,
                model=model,
                text_features=text_features,
                image_transform=image_transform,
                display_size=args.display_size,
                img_size=args.img_size,
                threshold=args.threshold,
                overlay_alpha=args.overlay_alpha,
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
            print(f"    effective threshold: {sample_panel['effective_threshold']:.4f}")
            rendered_samples.append(sample_panel)

        grouped_samples.append((class_name, rendered_samples))

    if not grouped_samples:
        raise FileNotFoundError(
            "No visualization panels were generated. "
            "Please confirm the selected classes have trained checkpoints in checkpoint_dir."
        )

    compose_figure(grouped_samples, display_size=args.display_size, output_path=args.output)
    print(f"saved visualization to {args.output}")


if __name__ == "__main__":
    main()
