# hybriddetector/trainer/train.py

import torch
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
from pathlib import Path
import json
from ..losses import bbox_loss, cls_loss, obj_loss


class ModelEMA:
    """Exponential Moving Average (EMA) of model weights.

    Stores a shadow copy of state_dict and updates it after optimizer steps.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9998):
        self.decay = float(decay)
        self.shadow = {}
        self._init_from(model)

    @torch.no_grad()
    def _init_from(self, model: torch.nn.Module) -> None:
        self.shadow = {}
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                self.shadow[k] = v.detach().float().clone()
            else:
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                if torch.is_floating_point(v):
                    self.shadow[k] = v.detach().float().clone()
                else:
                    self.shadow[k] = v.detach().clone()
                continue

            if torch.is_floating_point(v):
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=(1.0 - d))
            else:
                # For non-floating buffers, keep latest.
                self.shadow[k].copy_(v.detach())

    def state_dict(self) -> dict:
        return self.shadow

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise TypeError(f"EMA state must be a dict, got: {type(state)!r}")
        self.shadow = state


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
                 freeze_cnn_epochs: int = 0,
                 save_only_best: bool | None = None):
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

        # Checkpoint saving policy
        self.save_only_best = False
        try:
            from ..utils.config import Config

            if save_only_best is None:
                self.save_only_best = bool(getattr(Config, "SAVE_ONLY_BEST", False))
            else:
                self.save_only_best = bool(save_only_best)
        except Exception:
            self.save_only_best = bool(save_only_best) if save_only_best is not None else False

        # EMA (best-effort; controlled by Config.USE_EMA)
        self.use_ema = False
        self.ema = None
        try:
            from ..utils.config import Config

            self.use_ema = bool(getattr(Config, "USE_EMA", False)) and str(device).startswith('cuda')
            ema_decay = float(getattr(Config, "EMA_DECAY", 0.9998))
            if self.use_ema:
                self.ema = ModelEMA(self.model, decay=ema_decay)
        except Exception:
            self.use_ema = False
            self.ema = None

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
        
        loop = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            mininterval=2.0,
            leave=False,
            dynamic_ncols=True,
        )
        
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, targets) in enumerate(loop):
            images = images.to(self.device)

            loss_bbox, loss_cls, loss_obj = self._forward_loss(images, targets)
            loss = (loss_bbox + loss_cls + loss_obj) / self.grad_accum_steps

            if self.use_amp:
                assert self.scaler is not None
                self.scaler.scale(loss).backward()
            else:
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

                if self.use_ema and self.ema is not None:
                    self.ema.update(self.model)

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

    @torch.no_grad()
    def eval_epoch(self, val_dataset, batch_size: int = 4, num_workers: int = 4,
                   pin_memory: bool = True, persistent_workers: bool = True,
                   prefetch_factor: int = 2):
        """Evaluate average losses on a validation dataset (no backprop).

        This is intended for early stopping / model selection.
        """
        self.model.eval()

        dl_kwargs = {
            "batch_size": int(batch_size),
            "shuffle": False,
            "collate_fn": self.collate_fn,
            "num_workers": int(num_workers),
            "pin_memory": bool(pin_memory),
        }
        if dl_kwargs["num_workers"] > 0:
            dl_kwargs["persistent_workers"] = bool(persistent_workers)
            dl_kwargs["prefetch_factor"] = int(prefetch_factor)

        val_loader = DataLoader(val_dataset, **dl_kwargs)
        total_loss = 0.0
        bbox_loss_total = 0.0
        cls_loss_total = 0.0
        obj_loss_total = 0.0

        loop = tqdm(
            val_loader,
            desc=f"Val (epoch {self.current_epoch})",
            mininterval=2.0,
            leave=False,
            dynamic_ncols=True,
        )
        for images, targets in loop:
            images = images.to(self.device)
            loss_bbox, loss_cls, loss_obj = self._forward_loss(images, targets)

            total = (loss_bbox + loss_cls + loss_obj).item()
            total_loss += total
            bbox_loss_total += loss_bbox.item()
            cls_loss_total += loss_cls.item()
            obj_loss_total += loss_obj.item()

            loop.set_postfix({
                'loss': f'{total:.4f}',
                'bbox': f'{loss_bbox.item():.4f}',
                'cls': f'{loss_cls.item():.4f}',
                'obj': f'{loss_obj.item():.4f}'
            })

        n = max(1, len(val_loader))
        return {
            'total': total_loss / n,
            'bbox': bbox_loss_total / n,
            'cls': cls_loss_total / n,
            'obj': obj_loss_total / n,
        }

    def _forward_loss(self, images: torch.Tensor, targets: list[dict]):
        """Compute per-component losses for a batch."""
        if self.use_amp:
            with autocast(device_type='cuda'):
                outputs = self.model(images)
                return self._loss_from_outputs(outputs, targets)

        outputs = self.model(images)
        return self._loss_from_outputs(outputs, targets)

    def _loss_from_outputs(self, outputs: dict, targets: list[dict]):
        from ..utils.box_activation import decode_boxes_cxcywh

        pred_boxes = decode_boxes_cxcywh(outputs['boxes'])
        pred_obj = outputs['objectness']
        pred_cls = outputs['class_probs']

        target_boxes, target_labels, target_obj = self._encode_dense_targets(
            pred_boxes,
            targets,
            meta=outputs.get('meta'),
        )

        loss_obj = self.obj_loss_fn(pred_obj, target_obj)
        loss_cls = self.cls_loss_fn(pred_cls, target_labels)

        pos_mask = target_obj.squeeze(-1) > 0.5
        if pos_mask.any():
            loss_bbox = self.bbox_loss_fn(pred_boxes[pos_mask], target_boxes[pos_mask])
        else:
            loss_bbox = pred_boxes.sum() * 0.0

        return loss_bbox, loss_cls, loss_obj

    def _encode_dense_targets(
        self,
        pred_boxes: torch.Tensor,
        targets: list[dict],
        meta: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Anchor priors (normalized w,h). Best-effort: if missing/mismatched, fall back to uniform anchors.
        anchors_wh = None
        try:
            from ..utils.config import Config

            cfg_anchors = getattr(Config, "ANCHORS", None)
            if isinstance(cfg_anchors, (list, tuple)) and len(cfg_anchors) == num_anchors:
                anchors_wh = torch.tensor(cfg_anchors, device=device, dtype=torch.float32)  # [A,2]
        except Exception:
            anchors_wh = None

        if anchors_wh is None:
            anchors_wh = torch.full((num_anchors, 2), 0.10, device=device, dtype=torch.float32)

        if N % num_anchors != 0:
            raise ValueError(f"Pred count N={N} not divisible by num_anchors={num_anchors}")

        # Multi-scale: if model provides per-level shapes, compute offsets into the concatenated tensor.
        # Expected concat order matches model forward: [P3, P4, P5].
        levels = None
        if isinstance(meta, dict):
            lv = meta.get("levels")
            if isinstance(lv, list) and len(lv) == 3:
                ok = True
                parsed = []
                for item in lv:
                    if not isinstance(item, dict) or "h" not in item or "w" not in item:
                        ok = False
                        break
                    h = int(item["h"])  # type: ignore[arg-type]
                    w = int(item["w"])  # type: ignore[arg-type]
                    if h <= 0 or w <= 0:
                        ok = False
                        break
                    parsed.append((h, w))
                if ok:
                    levels = parsed

        if levels is None:
            # Backward-compatible single-scale path (assume square grid)
            grid_cells = N // num_anchors
            grid_size = int(grid_cells ** 0.5)
            if grid_size * grid_size != grid_cells:
                raise ValueError(
                    f"Expected square grid; got grid_cells={grid_cells} from N={N} and A={num_anchors}"
                )
            levels = [(grid_size, grid_size)]

        level_sizes = [h * w * num_anchors for (h, w) in levels]
        if sum(level_sizes) != N:
            raise ValueError(
                f"Meta levels imply N={sum(level_sizes)} but model outputs N={N}. "
                f"levels={levels}, A={num_anchors}"
            )

        level_offsets = [0]
        for s in level_sizes[:-1]:
            level_offsets.append(level_offsets[-1] + s)

        target_boxes = torch.zeros((B, N, 4), dtype=torch.float32, device=device)
        target_labels = torch.full((B, N), -1, dtype=torch.long, device=device)
        target_obj = torch.zeros((B, N, 1), dtype=torch.float32, device=device)
        best_match = torch.full((B, N), -1.0, dtype=torch.float32, device=device)

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

                # Select best anchor by IoU in (w,h) space (boxes assumed centered).
                gt_wh = torch.stack([bw, bh])  # [2]
                inter = torch.min(anchors_wh[:, 0], gt_wh[0]) * torch.min(anchors_wh[:, 1], gt_wh[1])
                union = anchors_wh[:, 0] * anchors_wh[:, 1] + gt_wh[0] * gt_wh[1] - inter + 1e-9
                iou_wh = inter / union
                a = int(torch.argmax(iou_wh).item())
                match_score = float(iou_wh[a].item())

                # Choose best pyramid level by how well the object fits the grid resolution.
                # Heuristic: object should cover ~4 cells at the selected level.
                best_level = 0
                best_level_score = None
                for li, (h, w) in enumerate(levels):
                    cells_w = float((bw * w).item())
                    cells_h = float((bh * h).item())
                    size = max(cells_w, cells_h)
                    level_score = -abs(torch.log(torch.tensor(size + 1e-6)) - torch.log(torch.tensor(4.0)))
                    level_score = float(level_score.item())
                    if best_level_score is None or level_score > best_level_score:
                        best_level_score = level_score
                        best_level = li

                H_l, W_l = levels[best_level]
                gi = int((xc * W_l).clamp(0, W_l - 1).item())
                gj = int((yc * H_l).clamp(0, H_l - 1).item())

                idx = level_offsets[best_level] + (gj * W_l + gi) * num_anchors + a

                if match_score > float(best_match[bi, idx].item()):
                    best_match[bi, idx] = match_score
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
        model_state = self.model.state_dict()
        ema_state = self.ema.state_dict() if (self.use_ema and self.ema is not None) else None

        checkpoint = {
            'epoch': epoch,
            # Keep raw weights for resume; best_model.pth will use EMA weights if enabled.
            'model_state_dict': model_state,
            'ema_state_dict': ema_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'best_loss': self.best_loss,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # If configured to save only best, skip all non-best checkpoints.
        if self.save_only_best and not is_best:
            loss_history_path = self.checkpoint_dir / 'loss_history.json'
            with open(loss_history_path, 'w') as f:
                json.dump(self.loss_history, f, indent=2)
            return

        # Save best model (single file)
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            best_ckpt = dict(checkpoint)
            if ema_state is not None:
                best_ckpt['model_state_dict'] = ema_state
            torch.save(best_ckpt, best_path)
            print(f"Best model saved: {best_path}")

        # Save regular/latest checkpoints only when not in save-only-best mode
        if not self.save_only_best:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

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

        if self.use_ema and self.ema is not None and 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
            try:
                self.ema.load_state_dict(checkpoint['ema_state_dict'])
            except Exception:
                # Best-effort; resume is still functional with raw weights.
                pass
        
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
