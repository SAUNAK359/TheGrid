# hybriddetector/utils/metrics.py

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import seaborn as sns
from pathlib import Path


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


def calculate_ap(recall, precision):
    """
    Calculate Average Precision (AP) using 11-point interpolation.
    """
    # Add sentinel values at the end
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    
    # Compute the precision envelope
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    
    return ap


def calculate_map(predictions, ground_truths, num_classes, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP) across all classes.
    
    Args:
        predictions: List of dicts with keys 'boxes', 'scores', 'labels'
        ground_truths: List of dicts with keys 'boxes', 'labels'
        num_classes: Total number of classes
        iou_threshold: IoU threshold for matching predictions to ground truth
    
    Returns:
        mAP value and per-class AP dictionary
    """
    aps = {}
    
    for class_id in range(num_classes):
        # Collect all predictions and ground truths for this class
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []
        image_ids = []
        
        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            # Get predictions for this class
            class_mask = pred['labels'] == class_id
            pred_boxes = pred['boxes'][class_mask]
            pred_scores = pred['scores'][class_mask]
            
            for box, score in zip(pred_boxes, pred_scores):
                all_pred_boxes.append(box)
                all_pred_scores.append(score)
                image_ids.append(img_idx)
            
            # Get ground truths for this class
            gt_class_mask = gt['labels'] == class_id
            gt_boxes = gt['boxes'][gt_class_mask]
            all_gt_boxes.extend([(img_idx, box) for box in gt_boxes])
        
        if len(all_gt_boxes) == 0:
            aps[class_id] = 0.0
            continue
        
        # Sort predictions by confidence score (descending)
        sorted_indices = np.argsort(all_pred_scores)[::-1]
        all_pred_boxes = [all_pred_boxes[i] for i in sorted_indices]
        all_pred_scores = [all_pred_scores[i] for i in sorted_indices]
        image_ids = [image_ids[i] for i in sorted_indices]
        
        # Track which ground truths have been matched
        gt_matched = [False] * len(all_gt_boxes)
        
        tp = np.zeros(len(all_pred_boxes))
        fp = np.zeros(len(all_pred_boxes))
        
        for pred_idx, (pred_box, img_id) in enumerate(zip(all_pred_boxes, image_ids)):
            # Find ground truths in the same image
            max_iou = 0
            max_gt_idx = -1
            
            for gt_idx, (gt_img_id, gt_box) in enumerate(all_gt_boxes):
                if gt_img_id != img_id:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            # Check if match is above threshold and not already matched
            if max_iou >= iou_threshold and max_gt_idx != -1 and not gt_matched[max_gt_idx]:
                tp[pred_idx] = 1
                gt_matched[max_gt_idx] = True
            else:
                fp[pred_idx] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / len(all_gt_boxes)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Calculate AP for this class
        ap = calculate_ap(recall, precision)
        aps[class_id] = ap
    
    # Calculate mAP
    mean_ap = np.mean(list(aps.values()))
    
    return mean_ap, aps


def plot_pr_curves(predictions, ground_truths, num_classes, class_names=None, save_path='pr_curves.png'):
    """
    Plot Precision-Recall curves for each class.
    """
    plt.figure(figsize=(12, 8))
    
    for class_id in range(num_classes):
        # Collect all predictions and ground truths for this class
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []
        image_ids = []
        
        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            class_mask = pred['labels'] == class_id
            pred_boxes = pred['boxes'][class_mask]
            pred_scores = pred['scores'][class_mask]
            
            for box, score in zip(pred_boxes, pred_scores):
                all_pred_boxes.append(box)
                all_pred_scores.append(score)
                image_ids.append(img_idx)
            
            gt_class_mask = gt['labels'] == class_id
            gt_boxes = gt['boxes'][gt_class_mask]
            all_gt_boxes.extend([(img_idx, box) for box in gt_boxes])
        
        if len(all_gt_boxes) == 0:
            continue
        
        # Sort by confidence
        sorted_indices = np.argsort(all_pred_scores)[::-1]
        all_pred_boxes = [all_pred_boxes[i] for i in sorted_indices]
        image_ids = [image_ids[i] for i in sorted_indices]
        
        gt_matched = [False] * len(all_gt_boxes)
        tp = np.zeros(len(all_pred_boxes))
        fp = np.zeros(len(all_pred_boxes))
        
        for pred_idx, (pred_box, img_id) in enumerate(zip(all_pred_boxes, image_ids)):
            max_iou = 0
            max_gt_idx = -1
            
            for gt_idx, (gt_img_id, gt_box) in enumerate(all_gt_boxes):
                if gt_img_id != img_id:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            if max_iou >= 0.5 and max_gt_idx != -1 and not gt_matched[max_gt_idx]:
                tp[pred_idx] = 1
                gt_matched[max_gt_idx] = True
            else:
                fp[pred_idx] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / len(all_gt_boxes)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Plot PR curve
        label = f'Class {class_id}' if class_names is None else class_names[class_id]
        plt.plot(recall, precision, label=label, linewidth=2)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PR curves saved to {save_path}")


def plot_confusion_matrix(predictions, ground_truths, num_classes, class_names=None, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix for classification.
    """
    all_pred_labels = []
    all_true_labels = []
    
    for pred, gt in zip(predictions, ground_truths):
        # For each ground truth, find the best matching prediction
        for gt_box, gt_label in zip(gt['boxes'], gt['labels']):
            max_iou = 0
            pred_label = -1
            
            for pred_box, pred_lbl in zip(pred['boxes'], pred['labels']):
                iou = calculate_iou(gt_box, pred_box)
                if iou > max_iou:
                    max_iou = iou
                    pred_label = pred_lbl
            
            if max_iou >= 0.5 and pred_label != -1:
                all_true_labels.append(gt_label)
                all_pred_labels.append(pred_label)
    
    if len(all_true_labels) == 0:
        print("No valid predictions for confusion matrix")
        return
    
    # Compute confusion matrix
    cm = sk_confusion_matrix(all_true_labels, all_pred_labels, labels=list(range(num_classes)))
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def plot_loss_curves(loss_history, save_path='loss_curves.png'):
    """
    Plot training loss curves over epochs.
    
    Args:
        loss_history: Dict with keys 'total', 'bbox', 'cls', 'obj' containing lists of losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    epochs = list(range(1, len(loss_history['total']) + 1))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history['total'], 'b-', linewidth=2, label='Total Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Total Training Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if 'bbox' in loss_history:
        plt.plot(epochs, loss_history['bbox'], 'r-', linewidth=2, label='BBox Loss')
    if 'cls' in loss_history:
        plt.plot(epochs, loss_history['cls'], 'g-', linewidth=2, label='Class Loss')
    if 'obj' in loss_history:
        plt.plot(epochs, loss_history['obj'], 'orange', linewidth=2, label='Objectness Loss')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Component Losses', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curves saved to {save_path}")
