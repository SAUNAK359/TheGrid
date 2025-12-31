from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class AuditStats:
    images_total: int = 0
    images_missing_label_file: int = 0
    images_empty_label_file: int = 0

    lines_total: int = 0
    boxes_total: int = 0

    bad_format_lines: int = 0
    parse_errors: int = 0
    non_finite: int = 0
    zero_or_negative_size: int = 0
    coord_out_of_range: int = 0
    label_out_of_range: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "images_total": self.images_total,
            "images_missing_label_file": self.images_missing_label_file,
            "images_empty_label_file": self.images_empty_label_file,
            "lines_total": self.lines_total,
            "boxes_total": self.boxes_total,
            "bad_format_lines": self.bad_format_lines,
            "parse_errors": self.parse_errors,
            "non_finite": self.non_finite,
            "zero_or_negative_size": self.zero_or_negative_size,
            "coord_out_of_range": self.coord_out_of_range,
            "label_out_of_range": self.label_out_of_range,
        }

    @property
    def bad_boxes(self) -> int:
        # Any of these mean the bbox line should not be trusted for training.
        return (
            self.bad_format_lines
            + self.parse_errors
            + self.non_finite
            + self.zero_or_negative_size
            + self.coord_out_of_range
            + self.label_out_of_range
        )


def _iter_images(images_dir: Path) -> List[Path]:
    if images_dir.is_file():
        return [images_dir]
    return [p for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in _IMG_EXTS]


def audit_yolo_dataset(
    *,
    images_dir: Path,
    labels_dir: Path,
    num_classes: Optional[int] = None,
    max_print: int = 5,
) -> tuple[AuditStats, List[str]]:
    """Audit a YOLO-format dataset (per-image .txt files).

    Expected line format: class_id x_center y_center width height
    Normalized coords are expected to be in [0, 1].

    Returns (stats, examples) where examples are human-readable strings for the first few issues.
    """
    images_dir = images_dir.resolve()
    labels_dir = labels_dir.resolve()

    stats = AuditStats()
    examples: List[str] = []

    images = _iter_images(images_dir)
    stats.images_total = len(images)

    for img_path in images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            stats.images_missing_label_file += 1
            if len(examples) < max_print:
                examples.append(f"missing label file: {label_path}")
            continue

        txt = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
        nonempty_lines = [ln.strip() for ln in txt if ln.strip()]
        if not nonempty_lines:
            stats.images_empty_label_file += 1
            continue

        for ln_i, line in enumerate(nonempty_lines, 1):
            stats.lines_total += 1
            parts = line.split()
            if len(parts) != 5:
                stats.bad_format_lines += 1
                if len(examples) < max_print:
                    examples.append(f"{label_path}:{ln_i} bad format ({len(parts)} fields): {line}")
                continue

            try:
                class_id = int(float(parts[0]))
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except Exception:
                stats.parse_errors += 1
                if len(examples) < max_print:
                    examples.append(f"{label_path}:{ln_i} parse error: {line}")
                continue

            stats.boxes_total += 1

            if not all(map(lambda v: v == v and abs(v) != float("inf"), [xc, yc, w, h])):
                stats.non_finite += 1
                if len(examples) < max_print:
                    examples.append(f"{label_path}:{ln_i} non-finite: {line}")
                continue

            if w <= 0 or h <= 0:
                stats.zero_or_negative_size += 1
                if len(examples) < max_print:
                    examples.append(f"{label_path}:{ln_i} non-positive size: {line}")

            # Normalized coords expectation
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                stats.coord_out_of_range += 1
                if len(examples) < max_print:
                    examples.append(f"{label_path}:{ln_i} out of [0,1] range: {line}")

            if num_classes is not None and (class_id < 0 or class_id >= int(num_classes)):
                stats.label_out_of_range += 1
                if len(examples) < max_print:
                    examples.append(
                        f"{label_path}:{ln_i} class_id {class_id} out of range [0,{int(num_classes)-1}]: {line}"
                    )

    return stats, examples


def format_audit_summary(name: str, stats: AuditStats, examples: List[str]) -> str:
    pct = (100.0 * stats.bad_boxes / max(1, stats.boxes_total))
    lines = [
        f"[{name}] images={stats.images_total}",
        f"[{name}] missing_label_files={stats.images_missing_label_file} empty_label_files={stats.images_empty_label_file}",
        f"[{name}] boxes_total={stats.boxes_total} bad_boxes={stats.bad_boxes} ({pct:.2f}%)",
        f"[{name}] issues: bad_format={stats.bad_format_lines} parse_errors={stats.parse_errors} non_finite={stats.non_finite} zero_or_negative_size={stats.zero_or_negative_size} coord_oob={stats.coord_out_of_range} label_oob={stats.label_out_of_range}",
    ]
    if examples:
        lines.append(f"[{name}] examples:\n  - " + "\n  - ".join(examples))
    return "\n".join(lines)
