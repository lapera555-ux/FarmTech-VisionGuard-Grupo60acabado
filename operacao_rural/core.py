"""Camada de monitoramento rural. Detecções não equivalem a intrusão."""
from __future__ import annotations
import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Detection:
    name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def validate(self, width: int, height: int) -> bool:
        return (self.name in ('person', 'car') and 0 <= self.confidence <= 1 and
                0 <= self.x1 < self.x2 <= width and 0 <= self.y1 < self.y2 <= height)


def center_in_zone(det: Detection, zone: tuple[float,float,float,float], width: int, height: int) -> bool:
    """Zone is normalized (x1, y1, x2, y2); bbox center is tested."""
    x1,y1,x2,y2 = zone
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError('Zona inválida: requer coordenadas normalizadas crescentes.')
    return x1 <= (det.x1 + det.x2)/(2*width) <= x2 and y1 <= (det.y1 + det.y2)/(2*height) <= y2


class EventThrottle:
    """Throttle per class/camera; not multi-object tracking."""
    def __init__(self, seconds: float=30):
        if seconds < 0:
            raise ValueError('Intervalo de eventos deve ser não negativo')
        self.seconds = seconds
        self.last: dict[tuple[str,str],float] = {}

    def accept(self, camera: str, name: str, monotonic_now: float) -> bool:
        key = (camera,name)
        previous = self.last.get(key)
        if previous is not None and monotonic_now - previous < self.seconds:
            return False
        self.last[key] = monotonic_now
        return True


def append_event(csv_path: Path, camera: str, detection: Detection, snapshot: str, zone: bool):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open('a', encoding='utf-8', newline='') as f:
        writer=csv.writer(f)
        if not exists:
            writer.writerow(['timestamp_utc','camera','class','confidence','zone_match','snapshot'])
        writer.writerow([dt.datetime.now(dt.timezone.utc).isoformat(),camera,detection.name,
                         f'{detection.confidence:.4f}',str(zone),snapshot])


def load_yolov5(weights: Path, repository: Path, confidence: float=0.45):
    """Loads locally pinned YOLOv5 clone and custom weights; no inference without weights."""
    if not weights.is_file():
        raise FileNotFoundError(f'Pesos não encontrados: {weights}. Treine no Colab e copie best.pt.')
    if not (repository/'hubconf.py').is_file():
        raise FileNotFoundError(f'Repositório YOLOv5 ausente: {repository}. Consulte GUIA_OPERACAO_RURAL.md')
    import os
    # Necessário para checkpoints YOLOv5 v7.0 legados em PyTorch >=2.6.
    # ALERTA: desativa carregamento restrito; nunca use pesos de fonte desconhecida.
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    import torch
    model = torch.hub.load(str(repository), 'custom', path=str(weights.resolve()), source='local', trust_repo=True)
    model.conf = confidence
    names = getattr(model, 'names', {})
    if isinstance(names, (tuple, list)):
        names = dict(enumerate(names))
    model.classes = [i for i, label in names.items() if label in ('person', 'car')] or None
    return model


def predict(model, frame):
    """YOLOv5 AutoShape accepts BGR OpenCV arrays via RGB conversion."""
    import cv2
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model(image, size=640)
    names = model.names
    detections=[]
    for x1,y1,x2,y2,conf,cls in result.xyxy[0].detach().cpu().numpy().tolist():
        label = names[int(cls)] if isinstance(names, (dict,list,tuple)) else str(cls)
        # only objects in scope; custom model may index car as 1
        if label in ('person','car'):
            d=Detection(label,float(conf),float(x1),float(y1),float(x2),float(y2))
            if d.validate(frame.shape[1],frame.shape[0]):
                detections.append(d)
    return detections
