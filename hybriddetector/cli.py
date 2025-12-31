from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import optim
from PIL import Image
import torchvision.transforms as TV

from hybriddetector.main import HybridDetector
from hybriddetector.dataset import custom_dataset, transforms as hd_transforms
from hybriddetector.trainer import train as trainer_mod, scheduler as scheduler_mod
from hybriddetector.inference.predictor import Predictor
from hybriddetector.inference.visualize import save_detection_image
from hybriddetector.utils import config, seed


def _require_yaml():
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required for --data YAML parsing. Install with: pip install pyyaml"
        ) from e
    return yaml


def load_data_yaml(data_yaml: str) -> Dict[str, Any]:
    """Loads a YOLO-style data.yaml.

    Supports keys: path, train, val, names/nc.
    Returns resolved absolute paths for train/val.
    """
    yaml = _require_yaml()

    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid data.yaml (expected mapping): {yaml_path}")

    base_dir = yaml_path.parent.resolve()
    root = data.get("path")
    if root is None:
        root_dir = base_dir
    else:
        root_dir = Path(root)
        if not root_dir.is_absolute():
            root_dir = (base_dir / root_dir).resolve()

    def _resolve(key: str) -> Optional[Path]:
        v = data.get(key)
        if v is None:
            return None
        p = Path(str(v))
        if not p.is_absolute():
            p = (root_dir / p).resolve()
        return p

    train_images = _resolve("train")
    val_images = _resolve("val")

    names = data.get("names")
    if names is None:
        nc = data.get("nc")
        if isinstance(nc, int):
            names = [f"class_{i}" for i in range(nc)]

    if names is not None and not isinstance(names, list):
        raise ValueError("data.yaml: 'names' must be a list")

    return {
        "root": root_dir,
        "train_images": train_images,
        "val_images": val_images,
        "names": names,
    }


def infer_labels_dir(images_dir: Path) -> Path:
    """Infer labels dir from images dir using YOLO convention.

    Examples:
      /.../images/train -> /.../labels/train
      /.../images/val   -> /.../labels/val
    """
    images_dir = images_dir.resolve()

    # Common YOLO layout:
    #   .../images/train  -> .../labels/train
    #   .../images/val    -> .../labels/val
    if images_dir.parent.name == "images":
        return images_dir.parent.parent / "labels" / images_dir.name

    # Kaggle/common alt layout:
    #   .../train/images  -> .../train/labels
    #   .../val/images    -> .../val/labels
    if images_dir.name == "images":
        return images_dir.parent / "labels"

    # Fallbacks:
    #   .../<root>/<split> -> .../<root>/labels/<split>
    candidate1 = images_dir.parent / "labels" / images_dir.name
    candidate2 = images_dir.parent / "labels"
    if candidate1.exists():
        return candidate1
    if candidate2.exists():
        return candidate2
    return candidate1


def load_weights(model: torch.nn.Module, weights_path: str, device: str) -> None:
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise TypeError(
            "Checkpoint must be a state_dict mapping or contain 'model_state_dict'. "
            f"Got: {type(state)!r}"
        )

    # If the checkpoint was trained with a different class count, adapt the head.
    # Common key for this project: cls_head.conv.weight has shape [A*C, 256, 1, 1]
    try:
        from hybriddetector.heads.class_head import ClassHead

        if hasattr(model, "cls_head") and "cls_head.conv.weight" in state:
            w = state.get("cls_head.conv.weight")
            if isinstance(w, torch.Tensor) and w.ndim == 4:
                out_channels = int(w.shape[0])
                in_channels = int(w.shape[1])

                num_anchors = int(getattr(getattr(model, "cls_head"), "num_anchors", 1))
                if num_anchors <= 0:
                    num_anchors = 1
                if out_channels % num_anchors == 0:
                    num_classes = out_channels // num_anchors
                else:
                    # Fallback: treat out_channels as class count if not divisible.
                    num_classes = out_channels

                current_classes = getattr(getattr(model, "cls_head"), "num_classes", None)
                current_out = getattr(getattr(getattr(model, "cls_head"), "conv", None), "out_channels", None)
                if current_classes != num_classes or current_out != out_channels:
                    head_device = next(model.parameters()).device
                    model.cls_head = ClassHead(
                        in_channels=in_channels,
                        num_anchors=num_anchors,
                        num_classes=num_classes,
                    ).to(head_device)
                    print(
                        "Adjusted cls_head to match checkpoint: "
                        f"num_anchors={num_anchors}, num_classes={num_classes}"
                    )
    except Exception:
        # Best-effort only; we still perform mismatch-stripping below.
        pass

    # strict=False still errors on shape mismatches, so drop those keys.
    model_state = model.state_dict()
    filtered: Dict[str, Any] = {}
    mismatched: List[Tuple[str, Any, Any]] = []
    for k, v in state.items():
        if k in model_state and isinstance(v, torch.Tensor) and isinstance(model_state[k], torch.Tensor):
            if v.shape != model_state[k].shape:
                mismatched.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                continue
        filtered[k] = v

    incompatible = model.load_state_dict(filtered, strict=False)

    if mismatched:
        print("Warning: skipped mismatched checkpoint tensors:")
        for k, ckpt_shape, model_shape in mismatched:
            print(f"  - {k}: checkpoint={ckpt_shape} model={model_shape}")
    if getattr(incompatible, "missing_keys", None):
        print(f"Warning: missing keys when loading weights: {len(incompatible.missing_keys)}")
    if getattr(incompatible, "unexpected_keys", None):
        print(f"Warning: unexpected keys in checkpoint: {len(incompatible.unexpected_keys)}")


def cmd_train(args: argparse.Namespace) -> None:
    seed.set_seed(42)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    data = load_data_yaml(args.data)
    train_images: Path = data["train_images"]
    if train_images is None:
        raise ValueError("data.yaml must contain a 'train' path")
    train_labels = infer_labels_dir(train_images)

    if not train_labels.exists():
        raise FileNotFoundError(
            "YOLO labels directory not found. "
            f"From train images path: {train_images} -> inferred labels dir: {train_labels}. "
            "Expected per-image .txt files in YOLO format. "
            "Fix your dataset layout or adjust data.yaml 'train' to point to the correct images folder."
        )

    val_images: Optional[Path] = data.get("val_images")
    val_labels: Optional[Path] = infer_labels_dir(val_images) if val_images else None

    class_names: Optional[List[str]] = data.get("names")
    num_classes = len(class_names) if class_names else config.Config.NUM_CLASSES

    train_tf = hd_transforms.get_transforms(train=True, img_size=args.img)
    train_ds = custom_dataset.CustomDataset(
        csv_file=str(train_labels),
        img_dir=str(train_images),
        transform=train_tf,
        num_classes=num_classes,
    )

    val_ds = None
    if val_images and val_labels and val_images.exists() and val_labels.exists():
        val_tf = hd_transforms.get_transforms(train=False, img_size=args.img)
        val_ds = custom_dataset.CustomDataset(
            csv_file=str(val_labels),
            img_dir=str(val_images),
            transform=val_tf,
            num_classes=num_classes,
        )

    if bool(getattr(args, "audit", True)):
        try:
            from hybriddetector.dataset.audit import audit_yolo_dataset, format_audit_summary

            stats, examples = audit_yolo_dataset(
                images_dir=train_images,
                labels_dir=train_labels,
                num_classes=num_classes,
                max_print=int(getattr(args, "audit_max_print", 5)),
            )
            print(format_audit_summary("Train", stats, examples))
            if bool(getattr(args, "audit_strict", False)):
                if stats.images_missing_label_file > 0 or stats.bad_boxes > 0 or stats.label_out_of_range > 0:
                    raise ValueError("Dataset audit failed (train). Fix labels/images or disable --audit-strict.")

            if val_images and val_labels and val_images.exists() and val_labels.exists():
                vstats, vex = audit_yolo_dataset(
                    images_dir=val_images,
                    labels_dir=val_labels,
                    num_classes=num_classes,
                    max_print=int(getattr(args, "audit_max_print", 5)),
                )
                print(format_audit_summary("Val", vstats, vex))
                if bool(getattr(args, "audit_strict", False)):
                    if vstats.images_missing_label_file > 0 or vstats.bad_boxes > 0 or vstats.label_out_of_range > 0:
                        raise ValueError("Dataset audit failed (val). Fix labels/images or disable --audit-strict.")
        except Exception as e:
            # Keep training runnable even if audit deps fail, unless strict mode requested.
            if bool(getattr(args, "audit_strict", False)):
                raise
            print(f"Warning: dataset audit skipped/failed: {e}")

    # Apply runtime backbone choice (keeps global Config minimal but allows CLI override).
    if getattr(args, "backbone", ""):
        config.Config.BACKBONE = str(args.backbone)
    if getattr(args, "backbone_pretrained", None) is not None:
        config.Config.BACKBONE_PRETRAINED = bool(args.backbone_pretrained)

    model = HybridDetector(img_size=args.img)
    # Keep config-driven architecture but set runtime class count for head
    if hasattr(model, "cls_head") and getattr(model.cls_head, "num_classes", None) != num_classes:
        from hybriddetector.heads.class_head import ClassHead

        model.cls_head = ClassHead(num_classes=num_classes)
    model.to(device)

    # Discriminative learning rates: backbone gets a smaller LR than heads.
    backbone_lr_mult = float(getattr(args, "backbone_lr_mult", getattr(config.Config, "BACKBONE_LR_MULT", 0.1)))
    backbone_lr_mult = max(0.0, min(1.0, backbone_lr_mult))

    param_groups = []
    if hasattr(model, "cnn"):
        backbone_params = [p for p in model.cnn.parameters() if p.requires_grad]
        other_params = [p for n, p in model.named_parameters() if (not n.startswith("cnn.")) and p.requires_grad]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": args.lr * backbone_lr_mult})
        if other_params:
            param_groups.append({"params": other_params, "lr": args.lr})

    optimizer = optim.AdamW(param_groups if param_groups else model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_ds) // args.batch)
    max_lr = [args.lr * backbone_lr_mult, args.lr] if param_groups else args.lr
    lr_scheduler = scheduler_mod.get_scheduler(
        optimizer,
        max_lr=max_lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
    )

    trainer = trainer_mod.Trainer(
        model,
        train_ds,
        optimizer,
        scheduler=lr_scheduler,
        device=device,
        batch_size=args.batch,
        use_amp=bool(args.amp),
        checkpoint_dir=args.project,
        grad_accum_steps=args.accumulate,
        num_workers=args.workers,
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers),
        prefetch_factor=args.prefetch_factor,
        freeze_cnn_epochs=args.freeze_cnn_epochs,
        save_only_best=bool(getattr(args, "save_only_best", False)),
    )

    start_epoch = 0
    if args.resume:
        ckpt = trainer.load_checkpoint(args.resume)
        start_epoch = int(ckpt.get("epoch", 0))

    print(f"\nTrain images: {train_images}")
    print(f"Train labels: {train_labels}")
    if val_ds is None:
        print("Val: (skipped) - provide val in data.yaml and labels/val")
    else:
        print(f"Val images: {val_images}")
        print(f"Val labels: {val_labels}")

    # Early stopping uses val loss when val is available.
    best_metric = float("inf")
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        trainer.current_epoch = epoch
        loss = trainer.train_epoch()
        print(
            f"epoch {epoch+1}/{args.epochs} | loss={loss['total']:.4f} "
            f"(bbox={loss['bbox']:.4f}, cls={loss['cls']:.4f}, obj={loss['obj']:.4f})"
        )

        val_loss = None
        if val_ds is not None and args.val_every > 0 and ((epoch + 1) % args.val_every == 0):
            val_loss = trainer.eval_epoch(
                val_ds,
                batch_size=args.batch,
                num_workers=args.workers,
                pin_memory=bool(args.pin_memory),
                persistent_workers=bool(args.persistent_workers),
                prefetch_factor=args.prefetch_factor,
            )
            print(
                f"val   {epoch+1}/{args.epochs} | loss={val_loss['total']:.4f} "
                f"(bbox={val_loss['bbox']:.4f}, cls={val_loss['cls']:.4f}, obj={val_loss['obj']:.4f})"
            )

        # Choose metric for best-model + early stopping
        metric = val_loss['total'] if val_loss is not None else loss['total']
        is_improved = metric < best_metric
        if is_improved:
            best_metric = metric
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if bool(getattr(args, "save_only_best", False)):
            # Only write checkpoints when we improve.
            if is_improved:
                trainer.best_loss = metric
                trainer.save_checkpoint(epoch + 1, is_best=True)
        else:
            if (epoch + 1) % args.save_period == 0 or (epoch + 1) == args.epochs:
                is_best = is_improved
                if is_best:
                    trainer.best_loss = metric
                trainer.save_checkpoint(epoch + 1, is_best=is_best)

        if args.early_stop and (epoch + 1) >= args.min_epochs and epochs_no_improve >= args.patience:
            print(
                f"Early stopping: no improvement for {epochs_no_improve} epochs "
                f"(patience={args.patience}). Best metric={best_metric:.4f}"
            )
            break

    print(f"\nDone. Checkpoints in: {Path(args.project).resolve()}")

    # ---------------------------------------------------------------------
    # Post-train: evaluation plots + sample visualizations
    # ---------------------------------------------------------------------
    ckpt_dir = Path(args.project)
    best_weights = ckpt_dir / "best_model.pth"
    latest_weights = ckpt_dir / "latest_checkpoint.pth"
    if bool(getattr(args, "save_only_best", False)):
        weights_to_use = best_weights
    else:
        weights_to_use = best_weights if best_weights.exists() else latest_weights

    if not weights_to_use.exists():
        print("Warning: no checkpoint found for post-train evaluation/visualization.")
        return

    if val_ds is None:
        if bool(getattr(args, "eval_after", False)) or bool(getattr(args, "vis_after", False)):
            print("Post-train eval/vis skipped: no validation split available.")
        return

    # Reload best weights for fair evaluation/visualizations.
    load_weights(model, str(weights_to_use), device)

    if bool(getattr(args, "eval_after", True)):
        try:
            from torch.utils.data import DataLoader
            from hybriddetector.trainer.evaluate import evaluate_model

            val_loader = DataLoader(
                val_ds,
                batch_size=int(getattr(args, "eval_batch", 4)),
                shuffle=False,
                collate_fn=trainer_mod.Trainer.collate_fn,
                num_workers=int(args.workers),
                pin_memory=bool(args.pin_memory),
            )
            eval_dir = ckpt_dir / "eval"
            evaluate_model(
                model=model,
                dataloader=val_loader,
                device=device,
                num_classes=num_classes,
                save_dir=str(eval_dir),
                conf_thresh=float(getattr(args, "eval_conf", 0.3)),
            )
        except Exception as e:
            print(f"Warning: post-train evaluation failed: {e}")

    if bool(getattr(args, "vis_after", True)):
        try:
            import random

            from hybriddetector.inference.predictor import Predictor
            from hybriddetector.inference.visualize import save_detection_image

            pred = Predictor(
                model,
                conf_thresh=float(getattr(args, "conf", float(config.Config.CONF_THRESH))),
                iou_thresh=float(getattr(args, "iou", float(config.Config.IOU_THRESH))),
                device=device,
            )

            vis_dir = ckpt_dir / "vis_samples"
            vis_dir.mkdir(parents=True, exist_ok=True)

            n = min(int(getattr(args, "vis_samples", 16)), len(val_ds))
            idxs = list(range(len(val_ds)))
            random.Random(42).shuffle(idxs)
            idxs = idxs[:n]

            for di in idxs:
                img_t, _tgt = val_ds[di]
                boxes, scores, labels = pred.predict_single_image(img_t)
                out_path = vis_dir / f"val_{di:06d}_pred.jpg"
                save_detection_image(
                    img_t,
                    boxes,
                    labels,
                    scores,
                    class_names=class_names,
                    save_path=out_path,
                )

            print(f"Saved sample visualizations to: {vis_dir.resolve()}")
        except Exception as e:
            print(f"Warning: post-train visualizations failed: {e}")


def _iter_images(source: Path) -> List[Path]:
    if source.is_file():
        return [source]

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in sorted(source.rglob("*")) if p.is_file() and p.suffix.lower() in exts]


def cmd_predict(args: argparse.Namespace) -> None:
    seed.set_seed(42)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Prefer class names from data.yaml; otherwise fall back to config.
    class_names = None
    if args.data:
        data = load_data_yaml(args.data)
        class_names = data.get("names")

    model = HybridDetector().to(device)
    load_weights(model, args.weights, device)

    # Ensure we always have readable class names.
    num_classes = getattr(getattr(model, "cls_head", None), "num_classes", None)
    if not class_names:
        cfg_names = list(getattr(config.Config, "CLASS_NAMES", []) or [])
        if isinstance(num_classes, int) and num_classes > 0:
            class_names = [cfg_names[i] if i < len(cfg_names) else f"class_{i}" for i in range(num_classes)]
        else:
            class_names = cfg_names if cfg_names else None
    else:
        if isinstance(num_classes, int) and num_classes > 0 and len(class_names) != num_classes:
            # Pad/truncate to match checkpoint class count.
            class_names = [class_names[i] if i < len(class_names) else f"class_{i}" for i in range(num_classes)]

    pred = Predictor(model, conf_thresh=args.conf, iou_thresh=args.iou, device=device)

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"--source not found: {source}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Match training/val preprocessing for the model input, but keep an unnormalized
    # copy for visualization (OpenCV expects 0-255-ish pixel space).
    tf_resize = TV.Resize((args.img, args.img))
    tf_to_tensor = TV.ToTensor()
    tf_norm = TV.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    images = _iter_images(source)
    if not images:
        raise RuntimeError(f"No images found in: {source}")

    verify_rows = []

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        img = tf_resize(img)
        t_raw = tf_to_tensor(img)
        t = tf_norm(t_raw)

        if not isinstance(t, torch.Tensor) or not isinstance(t_raw, torch.Tensor):
            raise TypeError("Expected torchvision transform to return a torch.Tensor")

        if args.verify:
            boxes, scores, labels, verified, verify_iou = pred.predict_single_image_verified(
                t, verify_iou=args.verify_iou
            )

            # Save a simple per-detection verification report
            for b, s, lab, ok, iou in zip(
                boxes.cpu().tolist(),
                scores.cpu().tolist(),
                labels.cpu().tolist(),
                verified.cpu().tolist(),
                verify_iou.cpu().tolist(),
            ):
                name = class_names[int(lab)] if class_names and int(lab) < len(class_names) else f"class_{int(lab)}"
                verify_rows.append([
                    str(img_path.name),
                    int(lab),
                    name,
                    float(s),
                    float(iou),
                    bool(ok),
                    float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                ])
        else:
            boxes, scores, labels = pred.predict_single_image(t)

        out_path = save_dir / f"{img_path.stem}_pred.jpg"
        save_detection_image(t_raw, boxes, labels, scores, class_names=class_names, save_path=str(out_path))

    if args.verify and verify_rows:
        import csv

        report_path = save_dir / "verification.csv"
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "image",
                "class_id",
                "class_name",
                "score",
                "verify_iou",
                "verified",
                "x1", "y1", "x2", "y2",
            ])
            w.writerows(verify_rows)
        print(f"Saved verification report to: {report_path.resolve()}")

    print(f"Saved predictions to: {save_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hybriddetector", description="YOLO-like CLI for TheGrid")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train using a YOLO data.yaml")
    t.add_argument("--data", type=str, required=True, help="Path to YOLO data.yaml")
    t.add_argument("--epochs", type=int, default=config.Config.EPOCHS)
    t.add_argument("--batch", type=int, default=config.Config.BATCH_SIZE)
    t.add_argument("--img", type=int, default=config.Config.IMG_SIZE)
    t.add_argument("--device", type=str, default=config.Config.DEVICE)
    t.add_argument(
        "--backbone",
        type=str,
        default=str(getattr(config.Config, "BACKBONE", "resnet50")),
        help="Backbone name (e.g. resnet50, resnet101, convnext_tiny).",
    )
    t.add_argument(
        "--backbone-pretrained",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(config.Config, "BACKBONE_PRETRAINED", True)),
        help="Enable/disable pretrained backbone weights.",
    )
    t.add_argument("--lr", type=float, default=config.Config.LR)
    t.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=float(getattr(config.Config, "BACKBONE_LR_MULT", 0.1)),
        help="Backbone LR multiplier relative to --lr (0..1).",
    )
    t.add_argument("--weight-decay", type=float, default=config.Config.WEIGHT_DECAY)
    t.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=bool(config.Config.USE_AMP),
        help="Enable/disable mixed precision (CUDA only).",
    )
    t.add_argument("--accumulate", type=int, default=int(getattr(config.Config, "GRAD_ACCUM_STEPS", 1)))
    t.add_argument("--workers", type=int, default=int(getattr(config.Config, "NUM_WORKERS", 4)))
    t.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(config.Config, "PIN_MEMORY", True)),
    )
    t.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(config.Config, "PERSISTENT_WORKERS", True)),
    )
    t.add_argument("--prefetch-factor", type=int, default=int(getattr(config.Config, "PREFETCH_FACTOR", 2)))
    t.add_argument("--freeze-cnn-epochs", type=int, default=int(getattr(config.Config, "FREEZE_CNN_EPOCHS", 0)))
    t.add_argument("--project", type=str, default=str(config.Config.SAVE_DIR))
    t.add_argument("--save-period", type=int, default=int(getattr(config.Config, "SAVE_EVERY_N_EPOCHS", 5)))
    t.add_argument(
        "--save-only-best",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(config.Config, "SAVE_ONLY_BEST", False)),
        help="Only save best_model.pth (no latest/per-epoch checkpoints).",
    )
    t.add_argument("--resume", type=str, default="", help="Path to checkpoint .pth to resume")

    t.add_argument(
        "--audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a fast dataset/label sanity audit before training.",
    )
    t.add_argument(
        "--audit-strict",
        action="store_true",
        help="Fail training if audit finds missing labels / invalid boxes / out-of-range class ids.",
    )
    t.add_argument(
        "--audit-max-print",
        type=int,
        default=5,
        help="Max number of audit example issues to print per split.",
    )

    t.add_argument(
        "--val-every",
        type=int,
        default=int(getattr(config.Config, "EVAL_EVERY_N_EPOCHS", 5)),
        help="Run validation-loss evaluation every N epochs (0 disables).",
    )
    t.add_argument(
        "--early-stop",
        action="store_true",
        help="Enable early stopping based on validation loss when val is available (else train loss).",
    )
    t.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs).")
    t.add_argument("--min-epochs", type=int, default=5, help="Minimum epochs before early stopping can trigger.")

    # Post-train outputs (enabled by default for Kaggle usability)
    t.add_argument(
        "--eval-after",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After training, run evaluation on val and save PR curves/confusion matrix under <project>/eval.",
    )
    t.add_argument(
        "--eval-batch",
        type=int,
        default=4,
        help="Batch size used for post-train evaluation.",
    )
    t.add_argument(
        "--eval-conf",
        type=float,
        default=0.3,
        help="Confidence threshold for post-train evaluation.",
    )
    t.add_argument(
        "--vis-after",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After training, save a few predicted images under <project>/vis_samples.",
    )
    t.add_argument("--vis-samples", type=int, default=16, help="Number of val images to visualize.")
    t.add_argument("--conf", type=float, default=float(config.Config.CONF_THRESH), help="Conf threshold for vis.")
    t.add_argument("--iou", type=float, default=float(config.Config.IOU_THRESH), help="IoU threshold for NMS in vis.")
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="Run inference on an image or folder")
    pr.add_argument("--weights", type=str, required=True, help="Path to checkpoint (.pth)")
    pr.add_argument("--source", type=str, required=True, help="Image file or folder")
    pr.add_argument("--data", type=str, default="", help="Optional data.yaml (for class names)")
    pr.add_argument("--img", type=int, default=config.Config.IMG_SIZE)
    pr.add_argument("--device", type=str, default=config.Config.DEVICE)
    pr.add_argument("--conf", type=float, default=float(config.Config.CONF_THRESH))
    pr.add_argument("--iou", type=float, default=float(config.Config.IOU_THRESH))
    pr.add_argument("--save-dir", type=str, default=str(config.Config.VIS_DIR))
    pr.add_argument(
        "--verify",
        action="store_true",
        help="Run a flip-consistency verification pass and write verification.csv to --save-dir.",
    )
    pr.add_argument(
        "--verify-iou",
        type=float,
        default=0.5,
        help="IoU threshold for verification match (only used with --verify).",
    )
    pr.set_defaults(func=cmd_predict)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "train" and args.resume == "":
        args.resume = ""

    args.func(args)


if __name__ == "__main__":
    main()
