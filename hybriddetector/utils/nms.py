# hybriddetector/utils/nms.py

import torch

def non_max_suppression(boxes, scores, iou_thresh=0.5):
    """
    boxes: Tensor [N,4] in xyxy format
    scores: Tensor [N]
    returns: indices of kept boxes
    """
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long)

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thresh]

    return torch.tensor(keep, dtype=torch.long)
