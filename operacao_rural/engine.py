"""Adaptadores de inferência real. Não geram resultados sintéticos."""
from __future__ import annotations
from pathlib import Path
from typing import Protocol
import numpy as np
from .core import Detection

VALID_CLASSES = {'person', 'car'}

class Detector(Protocol):
    def detect(self, image_bgr: np.ndarray, confidence: float) -> list[Detection]: ...


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError('Imagem inválida: esperado frame OpenCV BGR uint8 com 3 canais')


def _bbox(label: str, conf: float, coords: list[float], w: int, h: int) -> Detection | None:
    if label not in VALID_CLASSES:
        return None
    x1,y1,x2,y2 = (float(t) for t in coords)
    det = Detection(label, float(conf), max(0.0,x1), max(0.0,y1), min(float(w),x2), min(float(h),y2))
    return det if det.validate(w,h) else None


class UltralyticsDetector:
    """Modelo público COCO para teste imediato, NÃO é modelo treinado no dataset FIAP."""
    def __init__(self, weights: str = 'yolo11n.pt'):
        from ultralytics import YOLO
        self.model = YOLO(weights)  # Primeiro uso pode baixar pesos da fonte oficial.
        self.weights = weights

    def detect(self, image_bgr: np.ndarray, confidence: float=0.45) -> list[Detection]:
        _validate_frame(image_bgr)
        h,w = image_bgr.shape[:2]
        out = self.model.predict(source=image_bgr, imgsz=640, conf=confidence, verbose=False, device='cpu')[0]
        names = out.names
        found=[]
        for box in out.boxes:
            i = int(box.cls.item())
            label = names[i] if isinstance(names, dict) else names[i]
            d = _bbox(label, float(box.conf.item()), box.xyxy[0].tolist(), w,h)
            if d is not None:
                found.append(d)
        return found


class CustomYOLOv5Detector:
    """Inferência da YOLOv5s treinada na Entrega 1 (pesos best.pt locais)."""
    def __init__(self, weights: Path, repository: Path):
        from .core import load_yolov5
        self.model = load_yolov5(Path(weights),Path(repository), confidence=0.45)
        self.weights = str(weights)

    def detect(self, image_bgr: np.ndarray, confidence: float=0.45) -> list[Detection]:
        _validate_frame(image_bgr)
        self.model.conf = confidence
        from .core import predict
        return predict(self.model, image_bgr)


def build_detector(mode: str, project_root: Path) -> Detector:
    if mode == 'demonstracao_coco':
        return UltralyticsDetector('yolo11n.pt')
    if mode == 'yolov5_treinada':
        weights=project_root/'models'/'best.pt'
        repo=project_root/'third_party'/'yolov5'
        return CustomYOLOv5Detector(weights,repo)
    raise ValueError('Modo de detector desconhecido')
