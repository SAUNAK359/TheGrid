# hybriddetector/inference/visualize.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np
from pathlib import Path
import cv2


def visualize_predictions(image, boxes, labels=None, scores=None, class_names=None, save_path=None, show=True):
    """
    Visualize predictions with bounding boxes and save to file.
    
    Args:
        image: Tensor [3,H,W] or numpy array
        boxes: Tensor [N,4] in xyxy format
        labels: optional Tensor [N]
        scores: optional Tensor [N]
        class_names: optional list of class names
        save_path: optional path to save the image
        show: whether to display the image
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1,2,0).cpu().numpy()
    
    # Ensure image is in [0, 1] range for display
    if image.max() > 1.0:
        image = image / 255.0
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)

    H, W = image.shape[:2]
    
    # Convert boxes to numpy if tensor
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    
    # Generate random colors for different classes
    np.random.seed(42)
    colors = plt.cm.rainbow(np.linspace(0, 1, 20))

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        
        # Denormalize if needed (if coordinates are in [0,1] range)
        if x2 <= 1.0 and y2 <= 1.0:
            x1 *= W
            y1 *= H
            x2 *= W
            y2 *= H
        
        width = x2 - x1
        height = y2 - y1
        
        # Get color for this class
        if labels is not None:
            label = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
            color = colors[label % len(colors)]
        else:
            color = 'r'

        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, 
                                 edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Add label text
        if labels is not None:
            label = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
            if class_names and label < len(class_names):
                label_text = f"{class_names[label]}"
            else:
                label_text = f"Class {label}"
            
            if scores is not None:
                score = scores[i].item() if isinstance(scores[i], torch.Tensor) else scores[i]
                label_text += f": {score:.2f}"
            
            ax.text(x1, y1 - 5, label_text, color='white', fontsize=10,
                    bbox=dict(facecolor=color, alpha=0.7, pad=2))

    ax.axis('off')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_detection_image(image, boxes, labels, scores, class_names=None, save_path='detection.jpg'):
    """
    Save detection results as an image file using OpenCV.
    
    Args:
        image: Tensor [3,H,W] or numpy array [H,W,3]
        boxes: Tensor [N,4] in xyxy format
        labels: Tensor [N]
        scores: Tensor [N]
        class_names: List of class names
        save_path: Path to save the image
    """
    # Convert tensor to numpy
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    
    # Convert to BGR for OpenCV
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    H, W = image_bgr.shape[:2]
    
    # Convert boxes to numpy
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    # Generate colors for each class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(20, 3), dtype=np.uint8)
    
    # Draw each detection
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        
        # Denormalize if needed
        if x2 <= 1.0 and y2 <= 1.0:
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
        else:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get color for this class
        color = tuple(map(int, colors[int(label) % len(colors)]))
        
        # Draw bounding box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        
        # Create label text
        if class_names and int(label) < len(class_names):
            label_text = f"{class_names[int(label)]}: {score:.2f}"
        else:
            label_text = f"Class {int(label)}: {score:.2f}"
        
        # Get text size for background rectangle
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw background rectangle for text
        cv2.rectangle(image_bgr, (x1, y1 - text_h - baseline - 5), 
                     (x1 + text_w, y1), color, -1)
        
        # Draw text
        cv2.putText(image_bgr, label_text, (x1, y1 - baseline - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Save image
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), image_bgr)
    print(f"Saved detection image to {save_path}")


def visualize_batch_predictions(images, all_boxes, all_labels, all_scores, 
                                class_names=None, save_dir='./results/visualizations', max_images=16):
    """
    Visualize predictions for a batch of images and save them.
    
    Args:
        images: Tensor [B, 3, H, W]
        all_boxes: List of boxes for each image
        all_labels: List of labels for each image
        all_scores: List of scores for each image
        class_names: List of class names
        save_dir: Directory to save visualizations
        max_images: Maximum number of images to visualize
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    num_images = min(len(images), max_images)
    
    print(f"Visualizing {num_images} images...")
    
    for i in range(num_images):
        image = images[i]
        boxes = all_boxes[i]
        labels = all_labels[i]
        scores = all_scores[i]
        
        output_file = save_path / f'detection_{i:04d}.jpg'
        
        save_detection_image(
            image, boxes, labels, scores,
            class_names=class_names,
            save_path=output_file
        )
    
    print(f"Saved {num_images} visualizations to {save_dir}")
