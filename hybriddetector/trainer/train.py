# hybriddetector/trainer/train.py

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
import json
from ..losses import bbox_loss, cls_loss, obj_loss


class Trainer:
    """
    Enhanced Trainer for Hybrid Detector with checkpointing, mixed precision, and logging.
    """
    def __init__(self, model, train_dataset, optimizer, scheduler=None, device='cuda', 
                 batch_size=4, use_amp=True, checkpoint_dir='./checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                       collate_fn=self.collate_fn, num_workers=4, pin_memory=True)
        
        # Loss functions
        self.bbox_loss_fn = bbox_loss.giou_loss
        self.cls_loss_fn = cls_loss.FocalLoss()
        self.obj_loss_fn = obj_loss.ObjectnessLoss()
        
        # Mixed precision training
        self.use_amp = use_amp and device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss history tracking
        self.loss_history = {
            'total': [],
            'bbox': [],
            'cls': [],
            'obj': []
        }
        
        self.current_epoch = 0
        self.best_loss = float('inf')

    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        return images, targets

    def train_epoch(self):
        """Train for one epoch with loss tracking."""
        self.model.train()
        total_loss = 0
        bbox_loss_total = 0
        cls_loss_total = 0
        obj_loss_total = 0
        
        loop = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, targets) in enumerate(loop):
            images = images.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
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
                
                # Backward with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training without AMP
                outputs = self.model(images)
                
                pred_boxes = outputs['boxes']
                pred_obj = outputs['objectness']
                pred_cls = outputs['class_probs']
                
                target_boxes = torch.stack([t['boxes'] for t in targets]).to(self.device)
                target_labels = torch.stack([t['labels'] for t in targets]).to(self.device)
                target_obj = (target_boxes.sum(-1) > 0).float().unsqueeze(-1)
                
                loss_bbox = self.bbox_loss_fn(pred_boxes, target_boxes)
                loss_cls = self.cls_loss_fn(pred_cls, target_labels)
                loss_obj = self.obj_loss_fn(pred_obj, target_obj)
                
                loss = loss_bbox + loss_cls + loss_obj
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Track losses
            total_loss += loss.item()
            bbox_loss_total += loss_bbox.item()
            cls_loss_total += loss_cls.item()
            obj_loss_total += loss_obj.item()
            
            # Update progress bar
            loop.set_postfix({
                'loss': f'{loss.item():.4f}',
                'bbox': f'{loss_bbox.item():.4f}',
                'cls': f'{loss_cls.item():.4f}',
                'obj': f'{loss_obj.item():.4f}'
            })
        
        # Calculate average losses
        avg_total_loss = total_loss / len(self.train_loader)
        avg_bbox_loss = bbox_loss_total / len(self.train_loader)
        avg_cls_loss = cls_loss_total / len(self.train_loader)
        avg_obj_loss = obj_loss_total / len(self.train_loader)
        
        # Update loss history
        self.loss_history['total'].append(avg_total_loss)
        self.loss_history['bbox'].append(avg_bbox_loss)
        self.loss_history['cls'].append(avg_cls_loss)
        self.loss_history['obj'].append(avg_obj_loss)
        
        self.current_epoch += 1
        
        return {
            'total': avg_total_loss,
            'bbox': avg_bbox_loss,
            'cls': avg_cls_loss,
            'obj': avg_obj_loss
        }
    
    def save_checkpoint(self, epoch, is_best=False, additional_info=None):
        """
        Save model checkpoint with optimizer and scheduler state.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            additional_info: Additional information to save (dict)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'best_loss': self.best_loss
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save loss history as JSON
        loss_history_path = self.checkpoint_dir / 'loss_history.json'
        with open(loss_history_path, 'w') as f:
            json.dump(self.loss_history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.loss_history = checkpoint.get('loss_history', self.loss_history)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}")
        
        return checkpoint
    
    def get_loss_history(self):
        """Return the loss history."""
        return self.loss_history
