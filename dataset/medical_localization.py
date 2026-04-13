import os
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LOCALIZATION_CLASS_NAMES = ["Brain", "Liver", "Retina_RESC"]


def _build_image_transform(resize: int):
    return transforms.Compose(
        [
            transforms.Resize((resize, resize), Image.BICUBIC),
            transforms.ToTensor(),
        ]
    )


def _build_mask_transform(resize: int):
    return transforms.Compose(
        [
            transforms.Resize((resize, resize), Image.NEAREST),
            transforms.ToTensor(),
        ]
    )


def _resolve_mask_path(image_path: str) -> str:
    return image_path.replace("\\img\\", "\\anomaly_mask\\").replace("/img/", "/anomaly_mask/")


class LocalizationSourceTrainDataset(Dataset):
    def __init__(self, dataset_path="./data/", target_class="Brain", resize=240, split="valid"):
        assert target_class in LOCALIZATION_CLASS_NAMES, (
            f"target_class: {target_class}, should be in {LOCALIZATION_CLASS_NAMES}"
        )
        self.dataset_path = dataset_path
        self.target_class = target_class
        self.resize = resize
        self.split = split
        self.source_classes = [name for name in LOCALIZATION_CLASS_NAMES if name != target_class]
        self.transform_x = _build_image_transform(resize)
        self.transform_mask = _build_mask_transform(resize)
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[str, str, str]]:
        samples = []
        for class_name in self.source_classes:
            class_root = os.path.join(self.dataset_path, f"{class_name}_AD", self.split)

            normal_dir = os.path.join(class_root, "good", "img")
            for image_name in sorted(os.listdir(normal_dir)):
                samples.append((class_name, os.path.join(normal_dir, image_name), None))

            abnormal_dir = os.path.join(class_root, "Ungood", "img")
            for image_name in sorted(os.listdir(abnormal_dir)):
                image_path = os.path.join(abnormal_dir, image_name)
                mask_path = _resolve_mask_path(image_path)
                samples.append((class_name, image_path, mask_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        class_name, image_path, mask_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform_x(image)

        if mask_path is None:
            mask = torch.zeros((1, self.resize, self.resize), dtype=torch.float32)
        else:
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
        return image, mask, class_name


class LocalizationEvalDataset(Dataset):
    def __init__(self, dataset_path="./data/", class_name="Brain", resize=240, split="test"):
        assert class_name in LOCALIZATION_CLASS_NAMES, (
            f"class_name: {class_name}, should be in {LOCALIZATION_CLASS_NAMES}"
        )
        self.dataset_path = os.path.join(dataset_path, f"{class_name}_AD", split)
        self.class_name = class_name
        self.resize = resize
        self.split = split
        self.transform_x = _build_image_transform(resize)
        self.transform_mask = _build_mask_transform(resize)
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[str, str]]:
        samples = []
        normal_dir = os.path.join(self.dataset_path, "good", "img")
        for image_name in sorted(os.listdir(normal_dir)):
            samples.append((os.path.join(normal_dir, image_name), None))

        abnormal_dir = os.path.join(self.dataset_path, "Ungood", "img")
        for image_name in sorted(os.listdir(abnormal_dir)):
            image_path = os.path.join(abnormal_dir, image_name)
            mask_path = _resolve_mask_path(image_path)
            samples.append((image_path, mask_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform_x(image)

        if mask_path is None:
            mask = torch.zeros((1, self.resize, self.resize), dtype=torch.float32)
        else:
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
        return image, mask
