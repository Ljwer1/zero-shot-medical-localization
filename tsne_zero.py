import argparse
import os
import inspect
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from CLIP.clip import create_model
from CLIP.localization_adapter import LocalizationCLIP
from dataset.medical_localization import LOCALIZATION_CLASS_NAMES, LocalizationEvalDataset
from train_zero import ADAPTER_LAYERS, get_checkpoint_path, setup_seed
from utils import encode_text_with_prompt_ensemble, fuse_layer_outputs


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

NORMAL_LABEL = 0
ABNORMAL_LABEL = 1


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Brain-style normal/abnormal feature separation with t-SNE")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--pretrain", type=str, default="openai")
    parser.add_argument("--obj", type=str, default="Brain", choices=LOCALIZATION_CLASS_NAMES)
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--checkpoint_dir", type=str, default="./ckpt/localization/")
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--img_size", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--samples_per_class", type=int, default=300)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iterations", type=int, default=1000)
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--point_radius", type=int, default=4)
    parser.add_argument("--output", type=str, default="./images/brain_tsne_comparison.png")
    parser.add_argument("--seed", type=int, default=111)
    return parser.parse_args()


def split_indices_by_label(dataset: LocalizationEvalDataset) -> Tuple[List[int], List[int]]:
    normal_indices = []
    abnormal_indices = []
    for index, (_, mask_path) in enumerate(dataset.samples):
        if mask_path is None:
            normal_indices.append(index)
        else:
            abnormal_indices.append(index)
    return normal_indices, abnormal_indices


def sample_balanced_subset(dataset: LocalizationEvalDataset, samples_per_class: int, seed: int):
    normal_indices, abnormal_indices = split_indices_by_label(dataset)
    rng = np.random.default_rng(seed)

    normal_count = min(samples_per_class, len(normal_indices))
    abnormal_count = min(samples_per_class, len(abnormal_indices))
    selected_normal = rng.choice(normal_indices, size=normal_count, replace=False).tolist()
    selected_abnormal = rng.choice(abnormal_indices, size=abnormal_count, replace=False).tolist()

    selected_indices = selected_normal + selected_abnormal
    rng.shuffle(selected_indices)
    return Subset(dataset, selected_indices), normal_count, abnormal_count


def infer_labels(mask_batch: torch.Tensor) -> np.ndarray:
    return (mask_batch.view(mask_batch.shape[0], -1).amax(dim=1) > 0.5).long().cpu().numpy()


def mean_pool_patch_tokens(layer_tokens: torch.Tensor) -> torch.Tensor:
    patch_tokens = layer_tokens[:, 1:, :]
    pooled = patch_tokens.mean(dim=1)
    return pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)


def extract_base_features(clip_model, data_loader):
    features = []
    labels = []

    for images, masks in data_loader:
        images = images.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
            _, patch_tokens = clip_model.visual(images, ADAPTER_LAYERS)
            layer_features = [mean_pool_patch_tokens(layer_token) for layer_token in patch_tokens]
            fused = torch.stack(layer_features, dim=0).mean(dim=0)
            fused = fused / (fused.norm(dim=-1, keepdim=True) + 1e-8)

        features.append(fused.float().cpu().numpy())
        labels.append(infer_labels(masks))

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def extract_adapter_features(model, data_loader, text_features):
    features = []
    labels = []

    for images, masks in data_loader:
        images = images.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
            _, seg_patch_tokens = model(images, text_context=text_features)
            fusion_weights = model.get_fusion_weights().to(device)
            layer_features = [mean_pool_patch_tokens(layer_token) for layer_token in seg_patch_tokens]
            fused = fuse_layer_outputs(layer_features, fusion_weights)
            fused = fused / (fused.norm(dim=-1, keepdim=True) + 1e-8)

        features.append(fused.float().cpu().numpy())
        labels.append(infer_labels(masks))

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def run_tsne(features: np.ndarray, args) -> Tuple[np.ndarray, float]:
    feature_matrix = features.astype(np.float32)
    feature_matrix = feature_matrix - feature_matrix.mean(axis=0, keepdims=True)
    feature_matrix = feature_matrix / (feature_matrix.std(axis=0, keepdims=True) + 1e-8)

    if args.pca_dim > 0:
        pca_dim = min(args.pca_dim, feature_matrix.shape[0], feature_matrix.shape[1])
        if pca_dim >= 2:
            feature_matrix = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(feature_matrix)

    sample_count = feature_matrix.shape[0]
    max_perplexity = max(1.0, min(args.perplexity, float(sample_count - 1)))
    used_perplexity = min(max_perplexity, max(5.0, (sample_count - 1) / 3.0))
    used_perplexity = min(used_perplexity, max_perplexity)

    tsne_kwargs = {
        "n_components": 2,
        "perplexity": used_perplexity,
        "init": "pca",
        "learning_rate": 200.0,
        "random_state": args.seed,
    }
    tsne_signature = inspect.signature(TSNE.__init__)
    if "max_iter" in tsne_signature.parameters:
        tsne_kwargs["max_iter"] = args.tsne_iterations
    else:
        tsne_kwargs["n_iter"] = args.tsne_iterations

    embedding = TSNE(**tsne_kwargs).fit_transform(feature_matrix)
    return embedding, used_perplexity


def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int, font: ImageFont.ImageFont, point_radius: int):
    legend_items = [
        ("Normal", (59, 130, 189)),
        ("Abnormal", (244, 127, 50)),
    ]
    row_height = 32
    for index, (label, color) in enumerate(legend_items):
        cy = y + index * row_height + row_height // 2
        draw.ellipse(
            (x, cy - point_radius, x + point_radius * 2, cy + point_radius),
            fill=color,
            outline=color,
        )
        draw.text((x + point_radius * 2 + 12, y + index * row_height), label, fill="black", font=font)


def project_points(points: np.ndarray, left: int, top: int, width: int, height: int, padding: int):
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    normalized = (points - min_xy) / span

    x_coords = left + padding + normalized[:, 0] * (width - 2 * padding)
    y_coords = top + height - padding - normalized[:, 1] * (height - 2 * padding)
    return np.stack([x_coords, y_coords], axis=1)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    labels: np.ndarray,
    panel_box: Tuple[int, int, int, int],
    title: str,
    subtitle: str,
    font_title: ImageFont.ImageFont,
    font_label: ImageFont.ImageFont,
    point_radius: int,
    show_legend: bool,
):
    left, top, right, bottom = panel_box
    width = right - left
    height = bottom - top

    draw.rectangle(panel_box, outline=(60, 60, 60), width=2)
    draw.text((left + width // 2, top - 34), title, anchor="mm", fill="black", font=font_title)

    projected = project_points(points, left, top, width, height, padding=26)
    colors = {
        NORMAL_LABEL: (59, 130, 189),
        ABNORMAL_LABEL: (244, 127, 50),
    }

    for (x_coord, y_coord), label in zip(projected, labels):
        color = colors[int(label)]
        draw.ellipse(
            (
                x_coord - point_radius,
                y_coord - point_radius,
                x_coord + point_radius,
                y_coord + point_radius,
            ),
            fill=color,
            outline=color,
        )

    draw.text((left + width // 2, bottom + 26), subtitle, anchor="mm", fill="black", font=font_label)
    if show_legend:
        draw_legend(draw, right - 132, top + 16, font_label, point_radius)


def create_comparison_figure(
    base_embedding: np.ndarray,
    adapter_embedding: np.ndarray,
    labels: np.ndarray,
    args,
    normal_count: int,
    abnormal_count: int,
    base_perplexity: float,
    adapter_perplexity: float,
):
    canvas_width = 1320
    canvas_height = 760
    image = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(image)

    font_header = load_font(34)
    font_title = load_font(26)
    font_text = load_font(20)
    font_small = load_font(18)

    draw.text(
        (canvas_width // 2, 48),
        f"{args.obj} {args.split.capitalize()} Feature t-SNE",
        anchor="mm",
        fill="black",
        font=font_header,
    )
    draw.text(
        (canvas_width // 2, 90),
        "Normal and abnormal samples are balanced before visualization.",
        anchor="mm",
        fill=(70, 70, 70),
        font=font_small,
    )

    panel_width = 500
    panel_height = 470
    gap = 80
    left_panel = (120, 150, 120 + panel_width, 150 + panel_height)
    right_panel = (120 + panel_width + gap, 150, 120 + panel_width + gap + panel_width, 150 + panel_height)

    draw_panel(
        draw=draw,
        points=base_embedding,
        labels=labels,
        panel_box=left_panel,
        title="Without Adapter",
        subtitle="(a) Pretrained visual encoder features",
        font_title=font_title,
        font_label=font_text,
        point_radius=args.point_radius,
        show_legend=False,
    )
    draw_panel(
        draw=draw,
        points=adapter_embedding,
        labels=labels,
        panel_box=right_panel,
        title="With Adapter",
        subtitle="(b) Multi-level adapter features",
        font_title=font_title,
        font_label=font_text,
        point_radius=args.point_radius,
        show_legend=True,
    )

    footer_lines = [
        (
            f"{args.obj} {args.split} subset: {normal_count} normal + {abnormal_count} abnormal samples"
        ),
        (
            f"t-SNE perplexity: base={base_perplexity:.1f}, adapter={adapter_perplexity:.1f} | "
            f"adapter layers={ADAPTER_LAYERS}"
        ),
    ]
    for line_index, line in enumerate(footer_lines):
        draw.text(
            (canvas_width // 2, 680 + line_index * 28),
            line,
            anchor="mm",
            fill=(70, 70, 70),
            font=font_small,
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    image.save(args.output)


def main():
    args = parse_args()
    setup_seed(args.seed)

    dataset = LocalizationEvalDataset(
        dataset_path=args.data_path,
        class_name=args.obj,
        resize=args.img_size,
        split=args.split,
    )
    subset, normal_count, abnormal_count = sample_balanced_subset(
        dataset=dataset,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )

    data_loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
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
    checkpoint_path = get_checkpoint_path(
        argparse.Namespace(
            obj=args.obj,
            save_path=args.checkpoint_dir,
        )
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_localization_state_dict(checkpoint)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
        text_features = encode_text_with_prompt_ensemble(clip_model, args.obj, device)

    print(
        f"running t-SNE on {args.obj} {args.split} with "
        f"{normal_count} normal and {abnormal_count} abnormal samples"
    )

    base_features, labels = extract_base_features(clip_model=clip_model, data_loader=data_loader)
    adapter_features, adapter_labels = extract_adapter_features(
        model=model,
        data_loader=data_loader,
        text_features=text_features,
    )

    if not np.array_equal(labels, adapter_labels):
        raise RuntimeError("Label mismatch between base-feature and adapter-feature extraction")

    base_embedding, base_perplexity = run_tsne(base_features, args)
    adapter_embedding, adapter_perplexity = run_tsne(adapter_features, args)

    create_comparison_figure(
        base_embedding=base_embedding,
        adapter_embedding=adapter_embedding,
        labels=labels,
        args=args,
        normal_count=normal_count,
        abnormal_count=abnormal_count,
        base_perplexity=base_perplexity,
        adapter_perplexity=adapter_perplexity,
    )
    print(f"saved t-SNE comparison to {args.output}")


if __name__ == "__main__":
    main()
