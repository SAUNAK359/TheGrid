# hybriddetector/trainer/train.py

import torch
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
from pathlib import Path
import json
from ..losses import bbox_loss, cls_loss, obj_loss


class Trainer:
    """
    Enhanced Trainer for Hybrid Detector with checkpointing, mixed precision, and logging.
    """
    def __init__(self, model, train_dataset, optimizer, scheduler=None, device='cuda', 
                 batch_size=4, use_amp=True, checkpoint_dir='./checkpoints',
                 grad_accum_steps: int = 1,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 prefetch_factor: int = 2,
                 freeze_cnn_epochs: int = 0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.freeze_cnn_epochs = max(0, int(freeze_cnn_epochs))

        # DataLoader performance tuning
        dl_kwargs = {
            "batch_size": batch_size,
            "shuffle": True,
            "collate_fn": self.collate_fn,
            "num_workers": int(num_workers),
            "pin_memory": bool(pin_memory),
        }
        if dl_kwargs["num_workers"] > 0:
            dl_kwargs["persistent_workers"] = bool(persistent_workers)
            dl_kwargs["prefetch_factor"] = int(prefetch_factor)
        self.train_loader = DataLoader(train_dataset, **dl_kwargs)
        
        # Loss functions
        self.bbox_loss_fn = bbox_loss.giou_loss
        self.cls_loss_fn = cls_loss.FocalLoss()
        self.obj_loss_fn = obj_loss.ObjectnessLoss()
        
        # Mixed precision training
        self.use_amp = bool(use_amp) and str(device).startswith('cuda')
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
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

        # Optionally freeze/unfreeze CNN backbone early
        if hasattr(self.model, "cnn") and self.freeze_cnn_epochs > 0:
            should_freeze = self.current_epoch < self.freeze_cnn_epochs
            for p in self.model.cnn.parameters():
                p.requires_grad = not should_freeze

        total_loss = 0
        bbox_loss_total = 0
        cls_loss_total = 0
        obj_loss_total = 0
        
        loop = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, targets) in enumerate(loop):
            images = images.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                assert self.scaler is not None
                with autocast(device_type='cuda'):
                    outputs = self.model(images)

                    pred_boxes = outputs['boxes']
                    pred_obj = outputs['objectness']
                    pred_cls = outputs['class_probs']

                    target_boxes, target_labels, target_obj = self._encode_dense_targets(pred_boxes, targets)

                    # Objectness on all anchors
                    loss_obj = self.obj_loss_fn(pred_obj, target_obj)

                    # Class only on positive anchors (background uses ignore_index=-1)
                    loss_cls = self.cls_loss_fn(pred_cls, target_labels)

                    # BBox only on positive anchors
                    pos_mask = target_obj.squeeze(-1) > 0.5
                    if pos_mask.any():
                        loss_bbox = self.bbox_loss_fn(pred_boxes[pos_mask], target_boxes[pos_mask])
                    else:
                        loss_bbox = pred_boxes.sum() * 0.0

                    loss = (loss_bbox + loss_cls + loss_obj) / self.grad_accum_steps

                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images)

                pred_boxes = outputs['boxes']
                pred_obj = outputs['objectness']
                pred_cls = outputs['class_probs']

                target_boxes, target_labels, target_obj = self._encode_dense_targets(pred_boxes, targets)

                loss_obj = self.obj_loss_fn(pred_obj, target_obj)
                loss_cls = self.cls_loss_fn(pred_cls, target_labels)

                pos_mask = target_obj.squeeze(-1) > 0.5
                if pos_mask.any():
                    loss_bbox = self.bbox_loss_fn(pred_boxes[pos_mask], target_boxes[pos_mask])
                else:
                    loss_bbox = pred_boxes.sum() * 0.0

                loss = (loss_bbox + loss_cls + loss_obj) / self.grad_accum_steps

                loss.backward()

            # Optimizer step on accumulation boundary
            is_step = ((batch_idx + 1) % self.grad_accum_steps == 0) or ((batch_idx + 1) == len(self.train_loader))
            if is_step:
                if self.use_amp:
                    assert self.scaler is not None
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler:
                    self.scheduler.step()
            
            # Track losses
            # Track *unscaled* losses for reporting
            total_loss += (loss_bbox + loss_cls + loss_obj).item()
            bbox_loss_total += loss_bbox.item()
            cls_loss_total += loss_cls.item()
            obj_loss_total += loss_obj.item()
            
            # Update progress bar
            loop.set_postfix({
                'loss': f'{(loss_bbox + loss_cls + loss_obj).item():.4f}',
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

    def _encode_dense_targets(self, pred_boxes: torch.Tensor, targets: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode YOLO-format GT boxes into dense anchor targets aligned with model outputs.

        pred_boxes: [B, N, 4] where N=H*W*A and boxes are predicted in (xc,yc,w,h) normalized space.
        targets: list of dicts with keys 'boxes' [Gi,4] normalized and 'labels' [Gi]

        Returns:
          target_boxes: [B, N, 4]
          target_labels: [B, N] with -1 for background
          target_obj: [B, N, 1] float 0/1
        """
        B, N, _ = pred_boxes.shape
        device = pred_boxes.device

        # Infer number of anchors from model if available; otherwise default to 3.
        num_anchors = 3
        if hasattr(self.model, "box_head") and hasattr(self.model.box_head, "num_anchors"):
            num_anchors = int(self.model.box_head.num_anchors)

        if N % num_anchors != 0:
            raise ValueError(f"Pred count N={N} not divisible by num_anchors={num_anchors}")

        grid_cells = N // num_anchors
        grid_size = int(grid_cells ** 0.5)
        if grid_size * grid_size != grid_cells:
            raise ValueError(f"Expected square grid; got grid_cells={grid_cells} from N={N} and A={num_anchors}")

        H = W = grid_size

        target_boxes = torch.zeros((B, N, 4), dtype=torch.float32, device=device)
        target_labels = torch.full((B, N), -1, dtype=torch.long, device=device)
        target_obj = torch.zeros((B, N, 1), dtype=torch.float32, device=device)

        for bi in range(B):
            gt_boxes = targets[bi]["boxes"]
            gt_labels = targets[bi]["labels"]
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes_t = gt_boxes.to(device=device, dtype=torch.float32)
            else:
                gt_boxes_t = torch.tensor(gt_boxes, device=device, dtype=torch.float32)
            if isinstance(gt_labels, torch.Tensor):
                gt_labels_t = gt_labels.to(device=device, dtype=torch.long)
            else:
                gt_labels_t = torch.tensor(gt_labels, device=device, dtype=torch.long)

            if gt_boxes_t.numel() == 0:
                continue

            # Assign each GT to a single anchor at its center cell (simple YOLO-style assignment)
            for (xc, yc, bw, bh), cls in zip(gt_boxes_t, gt_labels_t):
                # clamp to [0,1]
                xc = xc.clamp(0.0, 1.0)
                yc = yc.clamp(0.0, 1.0)
                bw = bw.clamp(0.0, 1.0)
                bh = bh.clamp(0.0, 1.0)

                gi = int((xc * W).clamp(0, W - 1).item())
                gj = int((yc * H).clamp(0, H - 1).item())

                a = 0  # simplest: use first anchor
                idx = (gj * W + gi) * num_anchors + a

                target_boxes[bi, idx] = torch.stack([xc, yc, bw, bh])
                target_labels[bi, idx] = cls
                target_obj[bi, idx, 0] = 1.0

        return target_boxes, target_labels, target_obj
    
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
