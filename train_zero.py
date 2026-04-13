import argparse
import os
import random

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from CLIP.clip import create_model
from CLIP.localization_adapter import LocalizationCLIP
from dataset.medical_localization import LOCALIZATION_CLASS_NAMES, LocalizationEvalDataset, LocalizationSourceTrainDataset
from loss import BinaryDiceLoss, FocalLoss
from utils import encode_text_with_prompt_ensemble, fuse_layer_outputs


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
ADAPTER_LAYERS = [12, 24]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_checkpoint_path(args):
    return os.path.join(
        args.save_path,
        f"{args.obj}_localization_spatial_text_guided_{ADAPTER_LAYERS[0]}_{ADAPTER_LAYERS[1]}.pth",
    )


def format_weight_values(values):
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def log_model_weights(model):
    seg_fusion = model.get_fusion_weights().detach().cpu().tolist()
    residual_weights = model.get_residual_weights().detach().cpu()
    seg_reg = model.get_fusion_regularization().detach().cpu().item()

    print(f"seg fusion weights: {format_weight_values(seg_fusion)}")
    print(f"fusion regularization: seg={seg_reg:.6f}")
    for feature_idx, layer_id in enumerate(model.features):
        layer_weights = residual_weights[feature_idx].tolist()
        print(
            f"layer {layer_id} residual weights: "
            f"base={layer_weights[0]:.4f}, seg={layer_weights[1]:.4f}"
        )


def build_text_feature_dict(clip_model):
    text_feature_dict = {}
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
        for class_name in LOCALIZATION_CLASS_NAMES:
            text_feature_dict[class_name] = encode_text_with_prompt_ensemble(clip_model, class_name, device)
    return text_feature_dict


def build_layer_probability_map(layer_tokens, text_features, img_size):
    layer_tokens = layer_tokens[:, 1:, :]
    layer_tokens = layer_tokens / layer_tokens.norm(dim=-1, keepdim=True)
    anomaly_logits = 100.0 * layer_tokens @ text_features
    batch_size, patch_count, _ = anomaly_logits.shape
    grid_size = int(np.sqrt(patch_count))
    anomaly_logits = F.interpolate(
        anomaly_logits.permute(0, 2, 1).view(batch_size, 2, grid_size, grid_size),
        size=img_size,
        mode="bilinear",
        align_corners=True,
    )
    return torch.softmax(anomaly_logits, dim=1)


def evaluate(model, data_loader, text_features, img_size):
    gt_masks = []
    score_maps = []

    for image, mask in tqdm(data_loader):
        image = image.to(device)
        mask = mask.to(device)
        mask = (mask > 0.5).float()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
            _, seg_patch_tokens = model(image, text_context=text_features)
            seg_layer_weights = model.get_fusion_weights().to(device)

            anomaly_maps = []
            for layer_tokens in seg_patch_tokens:
                anomaly_prob = build_layer_probability_map(layer_tokens, text_features, img_size)
                anomaly_maps.append(anomaly_prob[:, 1, :, :])

            final_score_map = fuse_layer_outputs(anomaly_maps, seg_layer_weights).cpu().numpy()
            score_maps.append(final_score_map)
            gt_masks.append(mask.squeeze(1).cpu().numpy())

    gt_masks = np.concatenate(gt_masks, axis=0)
    gt_masks = (gt_masks > 0).astype(np.int_)
    score_maps = np.concatenate(score_maps, axis=0)
    score_maps = (score_maps - score_maps.min()) / (score_maps.max() - score_maps.min() + 1e-8)

    seg_roc_auc = roc_auc_score(gt_masks.flatten(), score_maps.flatten())
    return seg_roc_auc


def main():
    parser = argparse.ArgumentParser(description="Localization-only zero-shot training")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--pretrain", type=str, default="openai")
    parser.add_argument("--obj", type=str, default="Brain", choices=LOCALIZATION_CLASS_NAMES)
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--save_path", type=str, default="./ckpt/localization/")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=240)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seg_fusion_reg_weight", type=float, default=0.01)
    parser.add_argument("--learn_residual_weights", type=int, default=0)
    parser.add_argument("--log_weights", type=int, default=1)
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()

    setup_seed(args.seed)
    if args.batch_size != 1:
        raise ValueError("Localization-only training currently expects --batch_size 1")
    os.makedirs(args.save_path, exist_ok=True)

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
    model.residual_weight_logits.requires_grad_(bool(args.learn_residual_weights))
    model.seg_fusion_logits.requires_grad_(True)

    for param in model.seg_adapters.parameters():
        param.requires_grad = True

    trainable_parameters = list(model.seg_adapters.parameters()) + [model.seg_fusion_logits]
    if args.learn_residual_weights:
        trainable_parameters.append(model.residual_weight_logits)

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )

    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    train_dataset = LocalizationSourceTrainDataset(
        dataset_path=args.data_path,
        target_class=args.obj,
        resize=args.img_size,
        split="valid",
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    valid_dataset = LocalizationEvalDataset(
        dataset_path=args.data_path,
        class_name=args.obj,
        resize=args.img_size,
        split="valid",
    )
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, **kwargs)

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    text_feature_dict = build_text_feature_dict(clip_model)

    print("task mode: localization-only")
    print("adapter: spatial_text_guided")
    print(f"adapter layers: {ADAPTER_LAYERS}")
    print(f"target class: {args.obj}")
    print(f"source classes: {[name for name in LOCALIZATION_CLASS_NAMES if name != args.obj]}")

    best_score = 0.0
    best_epoch = -1
    for epoch in range(args.epoch):
        print(f"epoch {epoch}:")
        loss_list = []

        for image, mask, class_name in tqdm(train_loader):
            image = image.to(device)
            mask = mask.to(device)
            mask = (mask > 0.5).float()
            text_features = text_feature_dict[class_name[0]]

            with torch.cuda.amp.autocast(enabled=use_cuda):
                _, seg_patch_tokens = model(image, text_context=text_features)
                seg_layer_weights = model.get_fusion_weights().to(device)

                seg_loss_terms = []
                for layer_tokens in seg_patch_tokens:
                    anomaly_prob = build_layer_probability_map(layer_tokens, text_features, args.img_size)
                    seg_loss_terms.append(
                        loss_focal(anomaly_prob, mask) + loss_dice(anomaly_prob[:, 1, :, :], mask)
                    )

                seg_loss = fuse_layer_outputs(seg_loss_terms, seg_layer_weights)
                fusion_reg = args.seg_fusion_reg_weight * model.get_fusion_regularization()
                loss = seg_loss + fusion_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        valid_score = evaluate(
            model=model,
            data_loader=valid_loader,
            text_features=text_feature_dict[args.obj],
            img_size=args.img_size,
        )
        print(f"{args.obj} valid pAUC : {round(valid_score, 4)}")
        print(f"Loss: {np.mean(loss_list):.6f}")

        if args.log_weights:
            log_model_weights(model)

        if valid_score >= best_score:
            best_score = valid_score
            best_epoch = epoch
            checkpoint_path = get_checkpoint_path(args)
            checkpoint = model.localization_state_dict()
            checkpoint["best_valid_score"] = float(best_score)
            checkpoint["best_epoch"] = int(best_epoch)
            torch.save(checkpoint, checkpoint_path)
            print(f"best checkpoint saved to {checkpoint_path}")
            print(f"best valid pAUC so far: {best_score:.4f} at epoch {best_epoch}")

        print("")


if __name__ == "__main__":
    main()
