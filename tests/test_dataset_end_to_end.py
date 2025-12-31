import numpy as np
import torch
from PIL import Image

from hybriddetector.dataset.custom_dataset import CustomDataset
from hybriddetector.dataset.transforms import get_transforms


def _write_rgb_image(path, size=(80, 60), color=(10, 20, 30)):
    img = Image.new("RGB", size, color=color)
    img.save(path)


def test_sanitize_filters_invalid_and_clips():
    # YOLO format: [xc, yc, w, h] normalized
    boxes = torch.tensor(
        [
            [0.5, 0.5, 0.2, 0.3],  # valid
            [0.1, 0.1, 0.0, 0.2],  # invalid: w==0
            [0.9, 0.9, 0.2, 0.0],  # invalid: h==0
            [float("nan"), 0.2, 0.1, 0.1],  # invalid: NaN
            [-0.5, 1.5, 0.1, 0.1],  # out-of-range centers (will be clipped)
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)

    out_boxes, out_labels = CustomDataset._sanitize_yolo_boxes(boxes, labels)

    assert out_boxes.shape == (2, 4)
    assert out_labels.shape == (2,)

    expected_boxes = torch.tensor(
        [
            [0.5, 0.5, 0.2, 0.3],
            [0.0, 1.0, 0.1, 0.1],
        ],
        dtype=torch.float32,
    )
    expected_labels = torch.tensor([1, 5], dtype=torch.long)

    assert torch.allclose(out_boxes, expected_boxes, atol=1e-6)
    assert torch.equal(out_labels, expected_labels)


def test_dataset_getitem_end_to_end_with_albumentations(tmp_path):
    # Layout matches CustomDataset YOLO mode expectations:
    # img_dir contains images; labels_dir contains per-image .txt with same stem.
    img_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    img_dir.mkdir()
    labels_dir.mkdir()

    img_path = img_dir / "sample.jpg"
    _write_rgb_image(img_path)

    (labels_dir / "sample.txt").write_text(
        "\n".join(
            [
                # valid
                "1 0.5 0.5 0.2 0.3",
                # invalid (w==0) -> should be filtered
                "2 0.1 0.1 0.0 0.2",
                # invalid (h==0) -> should be filtered
                "3 0.9 0.9 0.2 0.0",
                # out-of-range center -> should be clipped and kept
                "5 -0.5 1.5 0.1 0.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    transform = get_transforms(train=False, img_size=64)
    ds = CustomDataset(csv_file=str(labels_dir), img_dir=str(img_dir), transform=transform)

    image, target = ds[0]

    assert isinstance(image, torch.Tensor)
    assert tuple(image.shape) == (3, 64, 64)

    boxes = target["boxes"]
    labels = target["labels"]

    assert isinstance(boxes, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert boxes.shape == (2, 4)
    assert labels.shape == (2,)

    # End-to-end expected bboxes after sanitize + deterministic val transform.
    # With bbox clipping enabled (clip=True), boxes that extend beyond the image boundary
    # are clipped in corner-coordinates, which changes (xc, yc, w, h).
    expected_boxes = torch.tensor(
        [
            [0.5, 0.5, 0.2, 0.3],
            # From sanitize: [0.0, 1.0, 0.1, 0.1]
            # Corners: x=[-0.05, 0.05], y=[0.95, 1.05] -> clipped to x=[0.0, 0.05], y=[0.95, 1.0]
            # Back to YOLO: xc=0.025, yc=0.975, w=0.05, h=0.05
            [0.025, 0.975, 0.05, 0.05],
        ],
        dtype=torch.float32,
    )
    expected_labels = torch.tensor([1, 5], dtype=torch.long)

    assert torch.allclose(boxes, expected_boxes, atol=1e-4)
    assert torch.equal(labels, expected_labels)


def test_dataset_transform_error_fallback_drops_boxes(tmp_path):
    img_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    img_dir.mkdir()
    labels_dir.mkdir()

    img_path = img_dir / "sample.jpg"
    _write_rgb_image(img_path)

    (labels_dir / "sample.txt").write_text(
        "\n".join(
            [
                "1 0.5 0.5 0.2 0.3",
                "5 -0.5 1.5 0.1 0.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class FlakyTransform:
        def __call__(self, *, image, bboxes, class_labels):
            # Simulate Albumentations raising on bbox processing.
            if len(bboxes) > 0:
                raise ValueError("simulated bbox failure")
            # Second call (with empty bboxes) succeeds.
            return {
                "image": torch.zeros((3, 32, 32), dtype=torch.float32),
                "bboxes": [],
                "class_labels": [],
            }

    ds = CustomDataset(csv_file=str(labels_dir), img_dir=str(img_dir), transform=FlakyTransform())
    image, target = ds[0]

    assert tuple(image.shape) == (3, 32, 32)
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)


def test_dataset_drops_out_of_range_class_ids(tmp_path):
    img_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    img_dir.mkdir()
    labels_dir.mkdir()

    img_path = img_dir / "sample.jpg"
    _write_rgb_image(img_path)

    # class_id=5 should be dropped when num_classes=3
    (labels_dir / "sample.txt").write_text(
        "\n".join(
            [
                "1 0.5 0.5 0.2 0.3",
                "5 0.4 0.4 0.2 0.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    transform = get_transforms(train=False, img_size=32)
    ds = CustomDataset(
        csv_file=str(labels_dir),
        img_dir=str(img_dir),
        transform=transform,
        num_classes=3,
    )

    _, target = ds[0]
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].shape == (1,)
    assert int(target["labels"][0].item()) == 1
