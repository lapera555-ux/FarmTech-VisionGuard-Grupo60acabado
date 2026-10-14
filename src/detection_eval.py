from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
import json
import numpy as np
from PIL import Image

from .dataset_tools import yolo_xywhn_to_xyxy


@dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    map50: float
    ap50_person: float
    ap50_car: float
    mean_inference_ms: float
    num_images: int

    def to_dict(self) -> dict:
        return asdict(self)


def box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO/PASCAL-style interpolated area under precision-recall curve."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_predictions(ground_truth: dict, predictions: list[dict], class_ids=(0, 1), iou_threshold=0.5):
    """
    ground_truth:
      {image_id: [{"class_id": 0, "box": [x1,y1,x2,y2]}, ...]}
    predictions:
      [{"image_id": "...", "class_id": 0, "confidence": .9, "box": [...]}, ...]
    """
    aps = {}
    total_tp = total_fp = total_gt = 0

    for class_id in class_ids:
        class_gt = {}
        n_gt = 0
        for image_id, items in ground_truth.items():
            boxes = [x["box"] for x in items if x["class_id"] == class_id]
            class_gt[image_id] = {"boxes": boxes, "matched": [False] * len(boxes)}
            n_gt += len(boxes)

        preds = sorted(
            [p for p in predictions if p["class_id"] == class_id],
            key=lambda p: p["confidence"],
            reverse=True,
        )
        tp, fp = [], []
        for pred in preds:
            image_id = pred["image_id"]
            candidate = class_gt.get(image_id, {"boxes": [], "matched": []})
            best_iou, best_idx = 0.0, -1
            for idx, gt_box in enumerate(candidate["boxes"]):
                if candidate["matched"][idx]:
                    continue
                iou = box_iou(pred["box"], gt_box)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                candidate["matched"][best_idx] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        tp_arr = np.cumsum(np.asarray(tp, dtype=float))
        fp_arr = np.cumsum(np.asarray(fp, dtype=float))
        recall = tp_arr / max(n_gt, 1)
        precision = tp_arr / np.maximum(tp_arr + fp_arr, 1e-12)
        aps[class_id] = compute_ap(recall, precision) if len(preds) else 0.0

        total_tp += int(tp_arr[-1]) if len(tp_arr) else 0
        total_fp += int(fp_arr[-1]) if len(fp_arr) else 0
        total_gt += n_gt

    precision_global = total_tp / max(total_tp + total_fp, 1)
    recall_global = total_tp / max(total_gt, 1)
    f1 = 2 * precision_global * recall_global / max(precision_global + recall_global, 1e-12)
    return {
        "precision": precision_global,
        "recall": recall_global,
        "f1": f1,
        "map50": float(np.mean([aps[c] for c in class_ids])),
        "ap50_person": aps.get(0, 0.0),
        "ap50_car": aps.get(1, 0.0),
    }


def load_ground_truth(dataset_root: str | Path, split: str = "test") -> dict:
    root = Path(dataset_root)
    output = {}
    image_dir = root / "images" / split
    label_dir = root / "labels" / split

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        items = []
        label_path = label_dir / f"{image_path.stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls, x, y, w, h = line.split()
            items.append(
                {
                    "class_id": int(float(cls)),
                    "box": list(yolo_xywhn_to_xyxy((x, y, w, h), width, height)),
                }
            )
        output[image_path.name] = items
    return output


def evaluate_torchhub_model(
    model,
    dataset_root: str | Path,
    split: str = "test",
    class_map: dict[int, int] | None = None,
    confidence_threshold: float = 0.001,
    image_size: int = 512,
) -> DetectionMetrics:
    """
    Avalia qualquer modelo YOLOv5 carregado via torch.hub.
    class_map permite mapear COCO -> projeto, ex.: {0: 0, 2: 1}.
    """
    root = Path(dataset_root)
    gt = load_ground_truth(root, split)
    predictions = []
    times_ms = []
    class_map = {0: 0, 1: 1} if class_map is None else class_map
    # Evita que o AutoShape elimine predições antes do cálculo de AP.
    if hasattr(model, "conf"):
        model.conf = confidence_threshold

    for image_path in sorted((root / "images" / split).iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        start = perf_counter()
        result = model(str(image_path), size=image_size)
        times_ms.append((perf_counter() - start) * 1000)

        rows = result.xyxy[0].detach().cpu().numpy()
        for x1, y1, x2, y2, conf, cls in rows:
            source_cls = int(cls)
            if source_cls not in class_map or float(conf) < confidence_threshold:
                continue
            predictions.append(
                {
                    "image_id": image_path.name,
                    "class_id": int(class_map[source_cls]),
                    "confidence": float(conf),
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                }
            )

    metrics = evaluate_predictions(gt, predictions)
    return DetectionMetrics(
        **metrics,
        mean_inference_ms=float(np.mean(times_ms)) if times_ms else 0.0,
        num_images=len(times_ms),
    )


def save_metrics(metrics: DetectionMetrics | dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = metrics.to_dict() if hasattr(metrics, "to_dict") else dict(metrics)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
