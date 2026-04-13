import argparse
import os
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from CLIP.clip import create_model
from CLIP.localization_adapter import LocalizationCLIP
from dataset.medical_localization import LOCALIZATION_CLASS_NAMES, LocalizationEvalDataset
from train_zero import ADAPTER_LAYERS, build_layer_probability_map, get_checkpoint_path, setup_seed
from utils import encode_text_with_prompt_ensemble, fuse_layer_outputs, resolve_fusion_weights


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def evaluate(
    model,
    data_loader,
    text_features,
    img_size,
    obj,
    split,
    fusion_mode="learned",
    selected_layer=None,
    show_progress=True,
):
    gt_masks = []
    score_maps = []

    for image, mask in tqdm(data_loader, disable=not show_progress):
        image = image.to(device)
        mask = mask.to(device)
        mask = (mask > 0.5).float()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
            _, seg_patch_tokens = model(image, text_context=text_features)
            learned_weights = model.get_fusion_weights().to(device)
            seg_layer_weights = resolve_fusion_weights(
                features=model.features,
                learned_weights=learned_weights,
                fusion_mode=fusion_mode,
                selected_layer=selected_layer,
            )

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
    if fusion_mode == "single":
        mode_desc = f"layer{selected_layer}"
    else:
        mode_desc = fusion_mode
    print(f"{obj} {split} pAUC ({mode_desc}) : {round(seg_roc_auc, 4)}")
    return seg_roc_auc


def main():
    parser = argparse.ArgumentParser(description="Localization-only zero-shot testing")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--pretrain", type=str, default="openai")
    parser.add_argument("--obj", type=str, default="Brain", choices=LOCALIZATION_CLASS_NAMES)
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--save_path", type=str, default="./ckpt/localization/")
    parser.add_argument("--img_size", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fusion_mode", type=str, default="learned", choices=["learned", "equal", "single"])
    parser.add_argument("--single_layer", type=int, default=None, choices=ADAPTER_LAYERS)
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()

    if args.fusion_mode == "single" and args.single_layer is None:
        raise ValueError("--single_layer must be set when --fusion_mode single")
    if args.fusion_mode != "single" and args.single_layer is not None:
        print("warning: --single_layer is ignored unless --fusion_mode single")

    setup_seed(args.seed)

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

    checkpoint = torch.load(get_checkpoint_path(args))
    model.load_localization_state_dict(checkpoint)
    if isinstance(checkpoint, dict):
        best_valid_score = checkpoint.get("best_valid_score")
        best_epoch = checkpoint.get("best_epoch")
        if best_valid_score is not None:
            if best_epoch is not None:
                print(f"checkpoint metadata: best valid pAUC={best_valid_score:.4f}, epoch={best_epoch}")
            else:
                print(f"checkpoint metadata: best valid pAUC={best_valid_score:.4f}")

    kwargs = {"num_workers": args.num_workers, "pin_memory": use_cuda}
    test_dataset = LocalizationEvalDataset(
        dataset_path=args.data_path,
        class_name=args.obj,
        resize=args.img_size,
        split=args.split,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
        text_features = encode_text_with_prompt_ensemble(clip_model, args.obj, device)

    print("task mode: localization-only")
    print("adapter: spatial_text_guided")
    print(f"adapter layers: {ADAPTER_LAYERS}")
    evaluate(
        model,
        test_loader,
        text_features,
        args.img_size,
        args.obj,
        args.split,
        fusion_mode=args.fusion_mode,
        selected_layer=args.single_layer,
    )


if __name__ == "__main__":
    main()
