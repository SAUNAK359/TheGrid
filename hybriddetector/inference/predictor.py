# hybriddetector/inference/predictor.py

import torch
import json
import csv
from pathlib import Path
from tqdm import tqdm
from ..utils.nms import non_max_suppression


class Predictor:
    """
    Enhanced predictor with batch inference and result saving.
    Runs inference for the hybrid detector and saves results in various formats.
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
        Predict on a batch of images.
        
        Args:
            images: Tensor [B, 3, H, W] normalized
        
        Returns:
            List of predictions per image: [boxes, scores, labels]
        """
        images = images.to(self.device)
        
        # Forward pass
        outputs = self.model(images)

        # outputs: dict with keys ['boxes', 'objectness', 'class_probs']
        # boxes are (xc,yc,w,h) in normalized space (post-sigmoid)
        boxes_cxcywh = torch.sigmoid(outputs['boxes'])       # [B, N, 4]
        obj = torch.sigmoid(outputs['objectness']).squeeze(-1)  # [B, N]
        cls_prob = torch.softmax(outputs['class_probs'], dim=-1)  # [B, N, C]

        cls_conf, labels = torch.max(cls_prob, dim=-1)  # [B, N]
        conf_scores = cls_conf * obj

        # Convert to xyxy for NMS
        xc, yc, w, h = boxes_cxcywh.unbind(dim=-1)
        x1 = (xc - w / 2).clamp(0.0, 1.0)
        y1 = (yc - h / 2).clamp(0.0, 1.0)
        x2 = (xc + w / 2).clamp(0.0, 1.0)
        y2 = (yc + h / 2).clamp(0.0, 1.0)
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        # Filter low-confidence boxes
        mask = conf_scores > self.conf_thresh
        filtered_boxes = [b[m] for b, m in zip(boxes_xyxy, mask)]
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
    
    @torch.no_grad()
    def predict_batch_dataset(self, dataloader, class_names=None, save_json=True, save_csv=True, 
                             output_dir='./results'):
        """
        Run batch inference on entire dataset and save results.
        
        Args:
            dataloader: DataLoader for the dataset
            class_names: List of class names (optional)
            save_json: Whether to save results as JSON
            save_csv: Whether to save results as CSV
            output_dir: Directory to save results
        
        Returns:
            List of all predictions
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        print("Running batch inference...")
        for batch_idx, (images, image_info) in enumerate(tqdm(dataloader)):
            # Get predictions
            boxes, scores, labels = self.predict(images)
            
            # Process each image in batch
            for img_idx in range(len(images)):
                # Get image info (filename, etc.)
                img_name = image_info[img_idx].get('filename', f'image_{batch_idx}_{img_idx}.jpg')
                
                # Convert predictions to CPU and numpy
                pred_boxes = boxes[img_idx].cpu().numpy().tolist()
                pred_scores = scores[img_idx].cpu().numpy().tolist()
                pred_labels = labels[img_idx].cpu().numpy().tolist()
                
                # Create result dictionary
                result = {
                    'image': img_name,
                    'detections': []
                }
                
                # Add each detection
                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    detection = {
                        'bbox': box,  # [x1, y1, x2, y2]
                        'score': float(score),
                        'class_id': int(label),
                        'class_name': class_names[int(label)] if class_names else f'class_{int(label)}'
                    }
                    result['detections'].append(detection)
                
                all_results.append(result)
        
        print(f"\nProcessed {len(all_results)} images")
        
        # Save results as JSON
        if save_json:
            json_path = output_path / 'predictions.json'
            with open(json_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Results saved to {json_path}")
        
        # Save results as CSV
        if save_csv:
            csv_path = output_path / 'predictions.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image', 'x1', 'y1', 'x2', 'y2', 'confidence', 'class_id', 'class_name'])
                
                for result in all_results:
                    img_name = result['image']
                    for det in result['detections']:
                        x1, y1, x2, y2 = det['bbox']
                        writer.writerow([
                            img_name,
                            x1, y1, x2, y2,
                            det['score'],
                            det['class_id'],
                            det['class_name']
                        ])
            
            print(f"Results saved to {csv_path}")
        
        return all_results
    
    def predict_single_image(self, image_tensor):
        """
        Predict on a single image.
        
        Args:
            image_tensor: Tensor [3, H, W] or [1, 3, H, W]
        
        Returns:
            boxes, scores, labels (all as tensors)
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        boxes, scores, labels = self.predict(image_tensor)
        
        return boxes[0], scores[0], labels[0]

    @torch.no_grad()
    def predict_single_image_verified(
        self,
        image_tensor: torch.Tensor,
        verify_iou: float = 0.5,
    ):
        """Predict on a single image and verify detections via flip test-time augmentation.

        The "verification" here is a lightweight consistency check:
        - Run inference on the original image
        - Run inference on a horizontally flipped image
        - Flip the boxes back
        - Mark a detection as verified if it has a same-class match with IoU >= verify_iou

        Returns:
            boxes, scores, labels, verified_mask, verify_iou_scores
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Original predictions
        boxes_o, scores_o, labels_o = self.predict(image_tensor)
        boxes_o, scores_o, labels_o = boxes_o[0], scores_o[0], labels_o[0]

        # Flipped predictions
        flipped = torch.flip(image_tensor, dims=[3])
        boxes_f, scores_f, labels_f = self.predict(flipped)
        boxes_f, scores_f, labels_f = boxes_f[0], scores_f[0], labels_f[0]

        # Flip boxes back (xyxy in normalized coords)
        if boxes_f.numel() > 0:
            x1 = boxes_f[:, 0]
            y1 = boxes_f[:, 1]
            x2 = boxes_f[:, 2]
            y2 = boxes_f[:, 3]
            boxes_f = torch.stack([1.0 - x2, y1, 1.0 - x1, y2], dim=-1)

        verified = torch.zeros((boxes_o.shape[0],), dtype=torch.bool, device=boxes_o.device)
        best_iou = torch.zeros((boxes_o.shape[0],), dtype=torch.float32, device=boxes_o.device)

        if boxes_o.numel() == 0 or boxes_f.numel() == 0:
            return boxes_o, scores_o, labels_o, verified, best_iou

        def _iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # a: [4], b: [M,4]
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

            ix1 = torch.maximum(ax1, bx1)
            iy1 = torch.maximum(ay1, by1)
            ix2 = torch.minimum(ax2, bx2)
            iy2 = torch.minimum(ay2, by2)

            iw = (ix2 - ix1).clamp(min=0.0)
            ih = (iy2 - iy1).clamp(min=0.0)
            inter = iw * ih

            area_a = ((ax2 - ax1).clamp(min=0.0)) * ((ay2 - ay1).clamp(min=0.0))
            area_b = ((bx2 - bx1).clamp(min=0.0)) * ((by2 - by1).clamp(min=0.0))
            union = area_a + area_b - inter
            return inter / (union + 1e-6)

        # For each original detection, find best IoU among same-class flipped detections.
        for i in range(boxes_o.shape[0]):
            same_class = labels_f == labels_o[i]
            if not torch.any(same_class):
                continue
            ious = _iou_xyxy(boxes_o[i], boxes_f[same_class])
            m = torch.max(ious)
            best_iou[i] = m
            verified[i] = m >= float(verify_iou)

        return boxes_o, scores_o, labels_o, verified, best_iou
