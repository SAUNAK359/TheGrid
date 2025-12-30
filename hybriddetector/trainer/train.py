# hybriddetector/trainer/train.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..losses import bbox_loss, cls_loss, obj_loss

class Trainer:
    """
    Trainer for Hybrid Detector
    """
    def __init__(self, model, train_dataset, optimizer, scheduler=None, device='cuda', batch_size=4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.bbox_loss_fn = bbox_loss.giou_loss
        self.cls_loss_fn = cls_loss.FocalLoss()
        self.obj_loss_fn = obj_loss.ObjectnessLoss()

    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        return images, targets

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader)
        for images, targets in loop:
            images = images.to(self.device)
            # Forward pass
            outputs = self.model(images)

            # Extract predictions
            pred_boxes = outputs['boxes']
            pred_obj = outputs['objectness']
            pred_cls = outputs['class_probs']

            # Extract targets
            target_boxes = torch.stack([t['boxes'] for t in targets]).to(self.device)
            target_labels = torch.stack([t['labels'] for t in targets]).to(self.device)
            target_obj = (target_boxes.sum(-1) > 0).float().unsqueeze(-1)

            # Compute losses
            loss_bbox = self.bbox_loss_fn(pred_boxes, target_boxes)
            loss_cls = self.cls_loss_fn(pred_cls, target_labels)
            loss_obj = self.obj_loss_fn(pred_obj, target_obj)

            loss = loss_bbox + loss_cls + loss_obj

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            loop.set_description(f"Loss: {loss.item():.4f}")

        return total_loss / len(self.train_loader)
