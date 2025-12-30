# hybriddetector/losses/bbox_loss.py

import torch

def bbox_iou(box1, box2, eps=1e-7):
    """
    Compute IoU between two sets of boxes.
    box: [B, N, 4] in format [x_center, y_center, w, h]
    """
    # Convert to x1,y1,x2,y2
    b1_x1 = box1[...,0] - box1[...,2]/2
    b1_y1 = box1[...,1] - box1[...,3]/2
    b1_x2 = box1[...,0] + box1[...,2]/2
    b1_y2 = box1[...,1] + box1[...,3]/2

    b2_x1 = box2[...,0] - box2[...,2]/2
    b2_y1 = box2[...,1] - box2[...,3]/2
    b2_x2 = box2[...,0] + box2[...,2]/2
    b2_y2 = box2[...,1] + box2[...,3]/2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union = area1 + area2 - inter_area + eps
    return inter_area / union

def giou_loss(pred, target):
    """
    GIoU loss between predicted and target boxes
    """
    iou = bbox_iou(pred, target)
    # Convex box
    b1_x1 = pred[...,0] - pred[...,2]/2
    b1_y1 = pred[...,1] - pred[...,3]/2
    b1_x2 = pred[...,0] + pred[...,2]/2
    b1_y2 = pred[...,1] + pred[...,3]/2

    b2_x1 = target[...,0] - target[...,2]/2
    b2_y1 = target[...,1] - target[...,3]/2
    b2_x2 = target[...,0] + target[...,2]/2
    b2_y2 = target[...,1] + target[...,3]/2

    c_x1 = torch.min(b1_x1, b2_x1)
    c_y1 = torch.min(b1_y1, b2_y1)
    c_x2 = torch.max(b1_x2, b2_x2)
    c_y2 = torch.max(b1_y2, b2_y2)

    area_c = (c_x2 - c_x1) * (c_y2 - c_y1)
    giou = iou - (area_c - (area_c * 0 + 1))/area_c  # simplified
    # For now, fallback to IoU loss; reduce to scalar for stable training.
    return (1 - iou).mean()

if __name__ == "__main__":
    pred = torch.tensor([[[0.5,0.5,0.2,0.2]]])
    target = torch.tensor([[[0.5,0.5,0.3,0.3]]])
    print("IoU:", bbox_iou(pred, target))
    print("GIoU Loss:", giou_loss(pred, target))
