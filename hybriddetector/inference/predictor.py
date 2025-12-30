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
