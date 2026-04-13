import argparse
import os

import torch
from torch.utils.data import DataLoader

from CLIP.clip import create_model
from CLIP.localization_adapter import LocalizationCLIP
from dataset.medical_localization import LOCALIZATION_CLASS_NAMES, LocalizationEvalDataset, resolve_clip_image_stats
from prompt import PROMPT_MODES
from test_zero import evaluate
from train_zero import ADAPTER_LAYERS, setup_seed
from utils import encode_text_with_prompt_ensemble, resolve_prompt_mode


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

ABLATION_SETTINGS = [
    (f"layer{ADAPTER_LAYERS[0]} only", "single", ADAPTER_LAYERS[0]),
    (f"layer{ADAPTER_LAYERS[1]} only", "single", ADAPTER_LAYERS[1]),
    (f"{ADAPTER_LAYERS[0]}+{ADAPTER_LAYERS[1]} equal fusion", "equal", None),
    (f"{ADAPTER_LAYERS[0]}+{ADAPTER_LAYERS[1]} learned fusion", "learned", None),
]


def format_markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]

    for row in rows:
        formatted = [row[0]] + [f"{value:.4f}" for value in row[1:]]
        lines.append("| " + " | ".join(formatted) + " |")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation-time fusion ablations for localization-only checkpoints")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--pretrain", type=str, default="openai")
    parser.add_argument("--obj", nargs="+", default=LOCALIZATION_CLASS_NAMES, choices=LOCALIZATION_CLASS_NAMES)
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--save_path", type=str, default="./ckpt/localization/")
    parser.add_argument("--img_size", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_md", type=str, default="")
    parser.add_argument("--prompt_mode", type=str, default="auto", choices=["auto", *PROMPT_MODES])
    parser.add_argument("--seed", type=int, default=111)
    return parser.parse_args()


def get_checkpoint_path(save_path, class_name):
    return os.path.join(
        save_path,
        f"{class_name}_localization_spatial_text_guided_{ADAPTER_LAYERS[0]}_{ADAPTER_LAYERS[1]}.pth",
    )


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

    headers = ["Target"] + [setting[0] for setting in ABLATION_SETTINGS]
    rows = []

    kwargs = {"num_workers": args.num_workers, "pin_memory": use_cuda}

    for class_name in args.obj:
        print(f"\n=== {class_name} ===")
        model = LocalizationCLIP(clip_model=clip_model, features=ADAPTER_LAYERS).to(device)
        model.eval()

        checkpoint = torch.load(get_checkpoint_path(args.save_path, class_name), map_location=device)
        model.load_localization_state_dict(checkpoint)
        effective_prompt_mode = resolve_prompt_mode(args.prompt_mode, checkpoint=checkpoint)
        checkpoint_prompt_mode = checkpoint.get("prompt_mode") if isinstance(checkpoint, dict) else None

        dataset = LocalizationEvalDataset(
            dataset_path=args.data_path,
            class_name=class_name,
            resize=args.img_size,
            split=args.split,
            image_mean=image_mean,
            image_std=image_std,
        )
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
            text_features = encode_text_with_prompt_ensemble(
                clip_model,
                class_name,
                device,
                prompt_mode=effective_prompt_mode,
            )

        if args.prompt_mode == "auto" and checkpoint_prompt_mode in PROMPT_MODES:
            print(f"prompt mode: auto -> {effective_prompt_mode} (from checkpoint)")
        elif args.prompt_mode == "auto":
            print(f"prompt mode: auto -> {effective_prompt_mode} (default)")
        else:
            print(f"prompt mode: {effective_prompt_mode}")

        row = [class_name]
        for _, fusion_mode, selected_layer in ABLATION_SETTINGS:
            score = evaluate(
                model=model,
                data_loader=data_loader,
                text_features=text_features,
                img_size=args.img_size,
                obj=class_name,
                split=args.split,
                fusion_mode=fusion_mode,
                selected_layer=selected_layer,
                show_progress=False,
            )
            row.append(score)

        rows.append(row)

    average_row = ["Average"]
    for column_idx in range(1, len(headers)):
        average_row.append(sum(row[column_idx] for row in rows) / len(rows))
    rows.append(average_row)

    table = format_markdown_table(headers, rows)
    print("\n" + table)

    if args.output_md:
        with open(args.output_md, "w", encoding="utf-8") as file:
            file.write(table + "\n")
        print(f"\nsaved markdown table to {args.output_md}")


if __name__ == "__main__":
    main()
