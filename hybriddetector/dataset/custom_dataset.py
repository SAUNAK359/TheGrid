# hybriddetector/dataset/custom_dataset.py

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

from .transforms import get_transforms

class CustomDataset(Dataset):
    """
    Custom dataset for object detection.
    Supports:
    1) YOLO format (preferred): images dir + labels dir containing per-image .txt files
       Each line: class_id x_center y_center width height (all normalized 0-1)
    2) CSV annotation (legacy): ['image_path', 'x_center','y_center','w','h','class_id']
    """
    def __init__(self, csv_file, img_dir, transform=None, num_classes: int | None = None):
        self.img_dir = str(img_dir)
        self.transform = transform or get_transforms()
        self.num_classes = num_classes

        # Backward-compatible parameter name: csv_file may actually be a YOLO labels directory.
        labels_or_csv = Path(csv_file)

        self.mode = None
        self.annotations = None
        self.image_groups = None
        self.image_list = []
        self.labels_dir = None

        # Decide mode:
        # - If it's a .csv path -> CSV mode
        # - Otherwise -> YOLO labels directory mode
        if labels_or_csv.suffix.lower() == ".csv":
            if not labels_or_csv.exists():
                raise FileNotFoundError(
                    f"CSV annotations file not found: {labels_or_csv}. "
                    "Either provide a valid .csv file or pass a YOLO labels directory (e.g. dataset/labels/train)."
                )
            self.mode = "csv"
            self.annotations = pd.read_csv(labels_or_csv)
            self.image_groups = self.annotations.groupby('image_path')
            self.image_list = list(self.image_groups.groups.keys())
        else:
            # YOLO mode
            if not labels_or_csv.exists() or not labels_or_csv.is_dir():
                raise FileNotFoundError(
                    f"YOLO labels directory not found: {labels_or_csv}. "
                    "Expected a folder containing per-image .txt files (YOLO format). "
                    "Update config paths or create dataset/labels/train and dataset/images/train."
                )
            self.mode = "yolo"
            self.labels_dir = str(labels_or_csv)
            self.image_list = self._build_yolo_image_list()

    def _build_yolo_image_list(self):
        img_dir = Path(self.img_dir)
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = []
        for p in sorted(img_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                images.append(p.name)
        if len(images) == 0:
            raise RuntimeError(f"No images found in {img_dir}")
        return images

    def _load_yolo_labels(self, image_filename: str):
        if self.labels_dir is None:
            raise RuntimeError("labels_dir is not set (dataset is not in YOLO mode)")

        labels_path = Path(self.labels_dir) / (Path(image_filename).stem + ".txt")
        if not labels_path.exists():
            # No labels file -> no boxes
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        boxes = []
        labels = []
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Invalid YOLO label line in {labels_path}: {line}")
                class_id, xc, yc, w, h = parts
                labels.append(int(float(class_id)))
                boxes.append([float(xc), float(yc), float(w), float(h)])

        if len(boxes) == 0:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def _sanitize_yolo_boxes(
        boxes: torch.Tensor,
        labels: torch.Tensor,
        *,
        eps: float = 1e-9,
        clip: bool = True,
        num_classes: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remove/clip invalid YOLO-format boxes (xc, yc, w, h) normalized to [0, 1].

        Albumentations will raise if a box becomes degenerate during conversion
        (e.g., w==0 or h==0 -> x_max <= x_min).
        """
        if boxes.numel() == 0:
            return boxes.reshape(0, 4).to(dtype=torch.float32), labels.reshape(0).to(dtype=torch.long)

        boxes = boxes.to(dtype=torch.float32)
        labels = labels.to(dtype=torch.long)

        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError(f"Expected boxes to have shape [N,4] (YOLO), got {tuple(boxes.shape)}")
        if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
            raise ValueError(
                f"Expected labels to have shape [N] matching boxes, got labels={tuple(labels.shape)} boxes={tuple(boxes.shape)}"
            )

        finite = torch.isfinite(boxes).all(dim=1)
        w = boxes[:, 2]
        h = boxes[:, 3]
        positive = (w > eps) & (h > eps)
        keep = finite & positive
        if num_classes is not None:
            keep = keep & (labels >= 0) & (labels < int(num_classes))

        boxes = boxes[keep]
        labels = labels[keep]

        if boxes.numel() == 0:
            return boxes.reshape(0, 4), labels.reshape(0)

        if clip:
            boxes = boxes.clamp_(0.0, 1.0)

        # Re-check after clipping (a bad label might clip to 0 width/height).
        w2 = boxes[:, 2]
        h2 = boxes[:, 3]
        keep2 = (w2 > eps) & (h2 > eps) & torch.isfinite(boxes).all(dim=1)
        boxes = boxes[keep2]
        labels = labels[keep2]

        return boxes.reshape(-1, 4), labels.reshape(-1)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        # Albumentations expects images as NumPy arrays (HWC).
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.mode == "yolo":
            boxes, labels = self._load_yolo_labels(img_name)
        else:
            if self.image_groups is None:
                raise RuntimeError("image_groups is not set (dataset is not in CSV mode)")
            boxes = self.image_groups.get_group(img_name)[['x_center','y_center','w','h']].values
            labels = self.image_groups.get_group(img_name)['class_id'].values
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

        # Prevent Albumentations from crashing on degenerate/invalid boxes (YOLO mode only).
        if self.mode == "yolo":
            boxes, labels = self._sanitize_yolo_boxes(boxes, labels, num_classes=self.num_classes)

        target = {'boxes': boxes, 'labels': labels}

        if self.transform:
            # Albumentations bbox pipeline expects Python lists (not torch tensors).
            bboxes_in = boxes.tolist() if isinstance(boxes, torch.Tensor) else boxes.tolist()
            labels_in = labels.tolist() if isinstance(labels, torch.Tensor) else labels.tolist()

            try:
                transformed = self.transform(image=image, bboxes=bboxes_in, class_labels=labels_in)
            except ValueError as e:
                # Common cause: dataset contains a YOLO line with w==0 or h==0 (or NaNs),
                # which becomes x_max<=x_min during conversion.
                # Fall back to transforming the image without boxes so training/eval can continue.
                print(
                    f"[CustomDataset] Albumentations bbox error for image '{img_path}': {e}. "
                    "Dropping bboxes for this sample."
                )
                transformed = self.transform(image=image, bboxes=[], class_labels=[])

            image = transformed['image']
            bboxes_out = transformed.get('bboxes', [])
            labels_out = transformed.get('class_labels', [])

            if len(bboxes_out) == 0:
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            else:
                target['boxes'] = torch.tensor(bboxes_out, dtype=torch.float32)

            if len(labels_out) == 0:
                target['labels'] = torch.zeros((0,), dtype=torch.long)
            else:
                target['labels'] = torch.tensor(labels_out, dtype=torch.long)

        return image, target


if __name__ == "__main__":
    dataset = CustomDataset(csv_file="annotations.csv", img_dir="images")
    img, target = dataset[0]
    print("Image type:", type(img))
    print("Boxes:", target['boxes'])
    print("Labels:", target['labels'])
