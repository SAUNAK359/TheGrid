# hybriddetector/dataset/custom_dataset.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from .transforms import get_transforms

class CustomDataset(Dataset):
    """
    Custom dataset for object detection.
    Supports CSV annotation: ['image_path', 'x_center','y_center','w','h','class_id']
    """
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform or get_transforms()

        # Group annotations by image
        self.image_groups = self.annotations.groupby('image_path')

        self.image_list = list(self.image_groups.groups.keys())

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        boxes = self.image_groups.get_group(img_name)[['x_center','y_center','w','h']].values
        labels = self.image_groups.get_group(img_name)['class_id'].values

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        target = {'boxes': boxes, 'labels': labels}

        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = transformed['image']
            target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(transformed['class_labels'], dtype=torch.long)

        return image, target


if __name__ == "__main__":
    dataset = CustomDataset(csv_file="annotations.csv", img_dir="images")
    img, target = dataset[0]
    print("Image shape:", img.shape)
    print("Boxes:", target['boxes'])
    print("Labels:", target['labels'])
