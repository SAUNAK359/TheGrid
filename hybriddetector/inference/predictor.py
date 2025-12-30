# hybriddetector/inference/predictor.py

import torch
from ..utils.nms import non_max_suppression

class Predictor:
    """
    Runs inference for the hybrid detector.
    Combines backbone + heads + fusion outputs.
    """
    def __init__(self, model, conf_thresh=0.3, iou_thresh=0.5, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device

    @torch.no_grad()
    def predict(self, images):
        """
        images: Tensor [B, 3, H, W] normalized
        Returns: list of predictions per image: [boxes, scores, labels]
        """
        images = images.to(self.device)
        # Forward pass
        outputs = self.model(images)

        # outputs: dict with keys ['boxes', 'objectness', 'class_probs']
        boxes = outputs['boxes']       # [B, N, 4]
        obj = outputs['objectness']    # [B, N, 1]
        cls = outputs['class_probs']   # [B, N, num_classes]

        conf_scores, labels = torch.max(cls * obj, dim=-1)  # [B, N]

        # Filter low-confidence boxes
        mask = conf_scores > self.conf_thresh
        filtered_boxes = [b[m] for b, m in zip(boxes, mask)]
        filtered_scores = [s[m] for s, m in zip(conf_scores, mask)]
        filtered_labels = [l[m] for l, m in zip(labels, mask)]

        # Apply NMS
        final_boxes, final_scores, final_labels = [], [], []
        for b, s, l in zip(filtered_boxes, filtered_scores, filtered_labels):
            keep = non_max_suppression(b, s, self.iou_thresh)
            final_boxes.append(b[keep])
            final_scores.append(s[keep])
            final_labels.append(l[keep])

        return final_boxes, final_scores, final_labels
