# hybriddetector/trainer/evaluate.py

import torch
import numpy as np
from hybriddetector.utils import metrics
from pathlib import Path
from hybriddetector.utils.box_activation import decode_boxes_cxcywh
from hybriddetector.utils.nms import non_max_suppression


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.
    Boxes format: [x1, y1, x2, y2]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    
    # Union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / (union_area + 1e-6)


def precision_recall(pred_boxes, pred_labels, pred_scores, target_boxes, target_labels, iou_thresh=0.5):
    """
    Compute precision and recall for one batch
    """
    tp, fp, fn = 0, 0, 0
    for pb, pl, ps, tb, tl in zip(pred_boxes, pred_labels, pred_scores, target_boxes, target_labels):
        if len(pb) == 0:
            fn += len(tb)
            continue
        # IoU matching
        matched_targets = set()
        for i, p_box in enumerate(pb):
            matched = False
            for j, t_box in enumerate(tb):
                if j in matched_targets:
                    continue
                iou = calculate_iou(p_box, t_box)
                if iou > iou_thresh and pl[i] == tl[j]:
                    tp += 1
                    matched = True
                    matched_targets.add(j)
                    break
            if not matched:
                fp += 1
        fn += len(tb) - len(matched_targets)
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return precision, recall


def evaluate_model(
    model,
    dataloader,
    device,
    num_classes,
    save_dir: str = './results',
    conf_thresh: float = 1e-3,
    iou_thresh: float = 0.5,
    max_det: int = 300,
):
    """
    Comprehensive model evaluation with mAP, confusion matrix, and PR curves.
    
    Args:
        model: The detection model
        dataloader: Validation/test dataloader
        device: Device to run evaluation on
        num_classes: Number of object classes
        save_dir: Directory to save evaluation results
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    print("Running evaluation...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            
            # Get model predictions
            outputs = model(images)
            
            # Decode predictions similarly to Predictor
            boxes_cxcywh = decode_boxes_cxcywh(outputs['boxes'])  # [B,N,4]
            obj = torch.sigmoid(outputs['objectness']).squeeze(-1)  # [B,N]
            cls_prob = torch.softmax(outputs['class_probs'], dim=-1)  # [B,N,C]
            cls_conf, cls_label = torch.max(cls_prob, dim=-1)  # [B,N]
            conf = cls_conf * obj

            xc, yc, w, h = boxes_cxcywh.unbind(dim=-1)
            x1 = (xc - w / 2).clamp(0.0, 1.0)
            y1 = (yc - h / 2).clamp(0.0, 1.0)
            x2 = (xc + w / 2).clamp(0.0, 1.0)
            y2 = (yc + h / 2).clamp(0.0, 1.0)
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

            # Process each image in the batch
            for img_idx in range(len(images)):
                # Keep predictions in torch for NMS/top-k, then move to numpy.
                pred_boxes_t = boxes_xyxy[img_idx]  # [N,4]
                pred_scores_t = conf[img_idx]       # [N]
                pred_labels_t = cls_label[img_idx]  # [N]

                # NOTE: For mAP/AP you generally should not apply an aggressive fixed threshold
                # before ranking-by-score. We keep a very small threshold by default only
                # to reduce compute/memory.
                thr = float(conf_thresh) if conf_thresh is not None else 0.0
                if thr > 0.0:
                    conf_mask = pred_scores_t > thr
                    pred_boxes_t = pred_boxes_t[conf_mask]
                    pred_scores_t = pred_scores_t[conf_mask]
                    pred_labels_t = pred_labels_t[conf_mask]

                # Class-aware NMS to cut duplicates without suppressing across classes.
                if pred_boxes_t.numel() > 0:
                    keep_all = []
                    for c in pred_labels_t.unique():
                        c = int(c.item())
                        m = pred_labels_t == c
                        keep = non_max_suppression(pred_boxes_t[m], pred_scores_t[m], float(iou_thresh))
                        if keep.numel() > 0:
                            # Map back to original indices.
                            idxs = torch.nonzero(m, as_tuple=False).squeeze(1)
                            keep_all.append(idxs[keep])

                    if keep_all:
                        keep_idx = torch.cat(keep_all, dim=0)
                        # Optional top-k by score
                        if max_det is not None and int(max_det) > 0 and keep_idx.numel() > int(max_det):
                            topk = torch.topk(pred_scores_t[keep_idx], k=int(max_det)).indices
                            keep_idx = keep_idx[topk]

                        pred_boxes_t = pred_boxes_t[keep_idx]
                        pred_scores_t = pred_scores_t[keep_idx]
                        pred_labels_t = pred_labels_t[keep_idx]

                pred_boxes = pred_boxes_t.detach().cpu().numpy()
                pred_scores = pred_scores_t.detach().cpu().numpy()
                pred_labels = pred_labels_t.detach().cpu().numpy()
                
                all_predictions.append({
                    'boxes': pred_boxes,
                    'scores': pred_scores,
                    'labels': pred_labels
                })
                
                # Extract ground truths (targets is a list[dict])
                # Dataset provides YOLO boxes (xc,yc,w,h) normalized to [0,1].
                # Metrics expect xyxy, so convert here.
                gt_boxes_yolo = targets[img_idx]['boxes']
                gt_labels = targets[img_idx]['labels'].cpu().numpy()

                if gt_boxes_yolo.numel() == 0:
                    gt_boxes = gt_boxes_yolo.cpu().numpy().reshape(0, 4)
                else:
                    gt_boxes_yolo = gt_boxes_yolo.to(dtype=torch.float32)
                    xc, yc, w, h = gt_boxes_yolo.unbind(dim=-1)
                    x1 = (xc - w / 2).clamp(0.0, 1.0)
                    y1 = (yc - h / 2).clamp(0.0, 1.0)
                    x2 = (xc + w / 2).clamp(0.0, 1.0)
                    y2 = (yc + h / 2).clamp(0.0, 1.0)
                    gt_boxes = torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy()
                
                all_ground_truths.append({
                    'boxes': gt_boxes,
                    'labels': gt_labels
                })
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    print("\nCalculating metrics...")
    
    # Calculate mAP
    mean_ap, per_class_ap = metrics.calculate_map(
        all_predictions, 
        all_ground_truths, 
        num_classes, 
        iou_threshold=0.5
    )
    
    print(f"\nmAP@0.5: {mean_ap:.4f}")
    print("\nPer-class AP:")
    for class_id, ap in per_class_ap.items():
        print(f"  Class {class_id}: {ap:.4f}")
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Plot and save PR curves
    print("\nGenerating PR curves...")
    metrics.plot_pr_curves(
        all_predictions, 
        all_ground_truths, 
        num_classes,
        save_path=str(save_path / 'pr_curves.png')
    )
    
    # Plot and save confusion matrix
    print("Generating confusion matrix...")
    metrics.plot_confusion_matrix(
        all_predictions,
        all_ground_truths,
        num_classes,
        save_path=str(save_path / 'confusion_matrix.png')
    )
    
    # Compile results
    results = {
        'mAP': mean_ap,
        'per_class_ap': per_class_ap,
        'predictions': all_predictions,
        'ground_truths': all_ground_truths
    }
    
    print(f"\nEvaluation complete! Results saved to {save_dir}")
    
    return results
