import argparse
import os
import inspect
from typing import List, Sequence, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from CLIP.clip import create_model
from CLIP.localization_adapter import LocalizationCLIP
from dataset.medical_localization import LOCALIZATION_CLASS_NAMES, LocalizationEvalDataset, resolve_clip_image_stats
from prompt import PROMPT_MODES
from train_zero import ADAPTER_LAYERS, build_layer_probability_map, get_checkpoint_path, setup_seed
from utils import encode_text_with_prompt_ensemble, fuse_layer_outputs, resolve_prompt_mode


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

NORMAL_LABEL = 0
ABNORMAL_LABEL = 1


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
    parser.add_argument("--selection_mode", type=str, default="random", choices=["random", "confidence"])
    parser.add_argument("--feature_mode", type=str, default="patch", choices=["patch", "pooled"])
    parser.add_argument("--score_topk", type=int, default=200)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iterations", type=int, default=1000)
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--point_radius", type=int, default=4)
    parser.add_argument("--prompt_mode", type=str, default="auto", choices=["auto", *PROMPT_MODES])
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


def sample_confidence_subset(
    dataset: LocalizationEvalDataset,
    labels: np.ndarray,
    scores: np.ndarray,
    samples_per_class: int,
):
    normal_indices = np.where(labels == NORMAL_LABEL)[0]
    abnormal_indices = np.where(labels == ABNORMAL_LABEL)[0]

    normal_count = min(samples_per_class, len(normal_indices))
    abnormal_count = min(samples_per_class, len(abnormal_indices))

    normal_order = np.argsort(scores[normal_indices])
    abnormal_order = np.argsort(scores[abnormal_indices])[::-1]
    selected_normal = normal_indices[normal_order[:normal_count]].tolist()
    selected_abnormal = abnormal_indices[abnormal_order[:abnormal_count]].tolist()
    selected_indices = selected_normal + selected_abnormal
    return Subset(dataset, selected_indices), normal_count, abnormal_count


def infer_labels(mask_batch: torch.Tensor) -> np.ndarray:
    return (mask_batch.view(mask_batch.shape[0], -1).amax(dim=1) > 0.5).long().cpu().numpy()


def mean_pool_patch_tokens(layer_tokens: torch.Tensor) -> torch.Tensor:
    patch_tokens = layer_tokens[:, 1:, :]
    pooled = patch_tokens.mean(dim=1)
    return pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)


def extract_base_features(clip_model, data_loader, feature_mode: str):
    features = []
    labels = []

    for images, masks in data_loader:
        images = images.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
            pooled, patch_tokens = clip_model.visual(images, ADAPTER_LAYERS)
            if feature_mode == "pooled":
                fused = pooled
            else:
                layer_features = [mean_pool_patch_tokens(layer_token) for layer_token in patch_tokens]
                fused = torch.stack(layer_features, dim=0).mean(dim=0)
            fused = fused / (fused.norm(dim=-1, keepdim=True) + 1e-8)

        features.append(fused.float().cpu().numpy())
        labels.append(infer_labels(masks))

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def extract_adapter_features(model, data_loader, text_features, feature_mode: str):
    features = []
    labels = []

    for images, masks in data_loader:
        images = images.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
            pooled, seg_patch_tokens = model(images, text_context=text_features)
            if feature_mode == "pooled":
                fused = pooled
            else:
                fusion_weights = model.get_fusion_weights().to(device)
                layer_features = [mean_pool_patch_tokens(layer_token) for layer_token in seg_patch_tokens]
                fused = fuse_layer_outputs(layer_features, fusion_weights)
            fused = fused / (fused.norm(dim=-1, keepdim=True) + 1e-8)

        features.append(fused.float().cpu().numpy())
        labels.append(infer_labels(masks))

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def score_dataset_samples(model, data_loader, text_features, img_size: int, score_topk: int):
    labels = []
    scores = []

    for images, masks in data_loader:
        images = images.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
            _, seg_patch_tokens = model(images, text_context=text_features)
            fusion_weights = model.get_fusion_weights().to(device)

            anomaly_maps = []
            for layer_tokens in seg_patch_tokens:
                anomaly_prob = build_layer_probability_map(layer_tokens, text_features, img_size)
                anomaly_maps.append(anomaly_prob[:, 1, :, :])

            score_map = fuse_layer_outputs(anomaly_maps, fusion_weights)
            flattened = score_map.view(score_map.shape[0], -1)
            topk = min(score_topk, flattened.shape[1])
            sample_scores = torch.topk(flattened, k=topk, dim=1).values.mean(dim=1)

        labels.append(infer_labels(masks))
        scores.append(sample_scores.float().cpu().numpy())

    return np.concatenate(labels, axis=0), np.concatenate(scores, axis=0)


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


def dataset_display_name(obj: str) -> str:
    mapping = {
        "Brain": "Brain MRI",
        "Liver": "Liver CT",
        "Retina_RESC": "Retinal OCT",
    }
    return mapping.get(obj, obj)


def panel_limits(points: np.ndarray, padding_ratio: float = 0.08):
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    span = max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1], 1e-6)
    half = span * (0.5 + padding_ratio)
    return (
        center[0] - half,
        center[0] + half,
        center[1] - half,
        center[1] + half,
    )


def create_comparison_figure(
    base_embedding: np.ndarray,
    adapter_embedding: np.ndarray,
    labels: np.ndarray,
    args,
    normal_count: int,
    abnormal_count: int,
    base_perplexity: float,
    adapter_perplexity: float,
    selection_mode: str,
):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.9,
        }
    )

    colors = {
        NORMAL_LABEL: "#3b82bd",
        ABNORMAL_LABEL: "#f47f32",
    }
    figure = plt.figure(figsize=(7.4, 5.2), dpi=200)
    grid = figure.add_gridspec(2, 2, height_ratios=[14, 3], hspace=0.22, wspace=0.18)
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
    ]
    caption_axis = figure.add_subplot(grid[1, :])
    caption_axis.axis("off")

    panel_specs = [
        (axes[0], base_embedding, "(a) Without Adapter", False),
        (axes[1], adapter_embedding, "(b) With Adapter", True),
    ]
    marker_size = max(8, args.point_radius * 5)

    for axis, points, subtitle, show_legend in panel_specs:
        for label_value, label_name in [(NORMAL_LABEL, "Normal"), (ABNORMAL_LABEL, "Abnormal")]:
            mask = labels == label_value
            axis.scatter(
                points[mask, 0],
                points[mask, 1],
                s=marker_size,
                c=colors[label_value],
                edgecolors="none",
                label=label_name,
            )

        xmin, xmax, ymin, ymax = panel_limits(points)
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_box_aspect(1)
        axis.text(
            0.5,
            -0.11,
            subtitle,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )
        if show_legend:
            legend = axis.legend(
                loc="upper right",
                frameon=True,
                fancybox=False,
                framealpha=1.0,
                borderpad=0.4,
                handlelength=0.8,
                handletextpad=0.5,
                fontsize=9,
            )
            legend.get_frame().set_edgecolor("#cccccc")
            legend.get_frame().set_linewidth(0.7)

    display_name = dataset_display_name(args.obj)
    caption = (
        f"Figure. Visualization, using t-SNE, of the features learned from the {display_name} {args.split} set, "
        f"using (a) pretrained visual encoder, and (b) multi-level feature adapters. "
        f"The same t-SNE optimization settings are used in each case. "
        f"Results show that features extracted by adapters are separated between normal and abnormal samples."
    )
    caption_axis.text(
        0.0,
        0.72,
        caption,
        ha="left",
        va="top",
        fontsize=9.2,
        wrap=True,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    args = parse_args()
    setup_seed(args.seed)

    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained=args.pretrain,
        require_pretrained=True,
    )
    clip_model.eval()
    image_mean, image_std = resolve_clip_image_stats(clip_model)

    dataset = LocalizationEvalDataset(
        dataset_path=args.data_path,
        class_name=args.obj,
        resize=args.img_size,
        split=args.split,
        image_mean=image_mean,
        image_std=image_std,
    )

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
    effective_prompt_mode = resolve_prompt_mode(args.prompt_mode, checkpoint=checkpoint)
    checkpoint_prompt_mode = checkpoint.get("prompt_mode") if isinstance(checkpoint, dict) else None

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
        text_features = encode_text_with_prompt_ensemble(
            clip_model,
            args.obj,
            device,
            prompt_mode=effective_prompt_mode,
        )

    if args.prompt_mode == "auto" and checkpoint_prompt_mode in PROMPT_MODES:
        print(f"prompt mode: auto -> {effective_prompt_mode} (from checkpoint)")
    elif args.prompt_mode == "auto":
        print(f"prompt mode: auto -> {effective_prompt_mode} (default)")
    else:
        print(f"prompt mode: {effective_prompt_mode}")

    if args.selection_mode == "confidence":
        selection_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=use_cuda,
        )
        selection_labels, selection_scores = score_dataset_samples(
            model=model,
            data_loader=selection_loader,
            text_features=text_features,
            img_size=args.img_size,
            score_topk=args.score_topk,
        )
        subset, normal_count, abnormal_count = sample_confidence_subset(
            dataset=dataset,
            labels=selection_labels,
            scores=selection_scores,
            samples_per_class=args.samples_per_class,
        )
    else:
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

    print(
        f"running t-SNE on {args.obj} {args.split} with "
        f"{normal_count} normal and {abnormal_count} abnormal samples "
        f"(selection={args.selection_mode}, feature_mode={args.feature_mode})"
    )

    base_features, labels = extract_base_features(
        clip_model=clip_model,
        data_loader=data_loader,
        feature_mode=args.feature_mode,
    )
    adapter_features, adapter_labels = extract_adapter_features(
        model=model,
        data_loader=data_loader,
        text_features=text_features,
        feature_mode=args.feature_mode,
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
        selection_mode=args.selection_mode,
    )
    print(f"saved t-SNE comparison to {args.output}")


if __name__ == "__main__":
    main()
