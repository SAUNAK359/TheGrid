# hybriddetector/losses/bbox_loss.py

import torch


def _to_xyxy(box: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # box: [...,4] in cxcywh
    cx, cy, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2

def bbox_iou(box1, box2, eps=1e-7):
    """
    Compute IoU between two sets of boxes.
    box: [B, N, 4] in format [x_center, y_center, w, h]
    """
    # Convert to x1,y1,x2,y2
    b1_x1, b1_y1, b1_x2, b1_y2 = _to_xyxy(box1)
    b2_x1, b2_y1, b2_x2, b2_y2 = _to_xyxy(box2)

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union = area1 + area2 - inter_area + eps
    return inter_area / union


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Complete IoU (CIoU) for cxcywh boxes.

    Returns CIoU in [-1, 1].
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = _to_xyxy(box1)
    b2_x1, b2_y1, b2_x2, b2_y2 = _to_xyxy(box2)

    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Areas
    area1 = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
    area2 = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)
    union = area1 + area2 - inter_area + eps
    iou = inter_area / union

    # Center distance term
    b1_cx = (b1_x1 + b1_x2) / 2
    b1_cy = (b1_y1 + b1_y2) / 2
    b2_cx = (b2_x1 + b2_x2) / 2
    b2_cy = (b2_y1 + b2_y2) / 2
    rho2 = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

    # Enclosing box diagonal
    c_x1 = torch.min(b1_x1, b2_x1)
    c_y1 = torch.min(b1_y1, b2_y1)
    c_x2 = torch.max(b1_x2, b2_x2)
    c_y2 = torch.max(b1_y2, b2_y2)
    c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

    # Aspect ratio term
    w1 = (b1_x2 - b1_x1).clamp(min=eps)
    h1 = (b1_y2 - b1_y1).clamp(min=eps)
    w2 = (b2_x2 - b2_x1).clamp(min=eps)
    h2 = (b2_y2 - b2_y1).clamp(min=eps)
    v = (4 / (torch.pi ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / c2) - (alpha * v)
    return ciou


def giou_loss(pred, target):
    """Compatibility wrapper.

    Historically named giou_loss, but we use CIoU here for stronger localization.
    """
    ciou = bbox_ciou(pred, target)
    return (1 - ciou).mean()

if __name__ == "__main__":
    pred = torch.tensor([[[0.5,0.5,0.2,0.2]]])
    target = torch.tensor([[[0.5,0.5,0.3,0.3]]])
    print("IoU:", bbox_iou(pred, target))
    print("GIoU Loss:", giou_loss(pred, target))
