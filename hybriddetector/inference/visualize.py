# hybriddetector/inference/visualize.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

def visualize_predictions(image, boxes, labels=None, scores=None, class_names=None):
    """
    image: Tensor [3,H,W] or numpy array
    boxes: Tensor [N,4] in xywh normalized format (0-1)
    labels: optional Tensor [N]
    scores: optional Tensor [N]
    class_names: optional list of class names
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1,2,0).cpu().numpy()
    
    fig, ax = plt.subplots(1,1, figsize=(12,12))
    ax.imshow(image)

    H, W = image.shape[:2]

    for i, box in enumerate(boxes):
        x_c, y_c, w, h = box
        x = (x_c - w/2) * W
        y = (y_c - h/2) * H
        w *= W
        h *= H

        rect = patches.Rectangle((x,y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        if labels is not None:
            label = labels[i].item()
            if class_names:
                label_text = f"{class_names[label]}"
            else:
                label_text = str(label)
            if scores is not None:
                score = scores[i].item()
                label_text += f":{score:.2f}"
            ax.text(x, y, label_text, color='white', fontsize=12,
                    bbox=dict(facecolor='red', alpha=0.5, pad=1))

    plt.show()
