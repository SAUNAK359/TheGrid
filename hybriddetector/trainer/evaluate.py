# hybriddetector/trainer/evaluate.py

import torch

def precision_recall(pred_boxes, pred_labels, pred_scores, target_boxes, target_labels, iou_thresh=0.5):
    """
    Compute precision and recall for one batch
    """
    # This is simplified: in practice, use pycocotools or custom mAP calculation
    tp, fp, fn = 0, 0, 0
    for pb, pl, ps, tb, tl in zip(pred_boxes, pred_labels, pred_scores, target_boxes, target_labels):
        if len(pb) == 0:
            fn += len(tb)
            continue
        # naive IoU matching
        for t_box in tb:
            matched = False
            for p_box in pb:
                iou = bbox_iou(p_box.unsqueeze(0), t_box.unsqueeze(0))
                if iou > iou_thresh:
                    tp += 1
                    matched = True
                    break
            if not matched:
                fn += 1
        fp += len(pb) - tp
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return precision, recall
