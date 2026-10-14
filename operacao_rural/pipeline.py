"""Quadro a quadro: detecção, visualização, zona e eventos persistentes."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
import time
import cv2
import numpy as np
from .core import Detection, EventThrottle, center_in_zone
from .event_store import EventStore
from .engine import Detector, _validate_frame

@dataclass(frozen=True)
class MonitorConfig:
    camera: str='porteira_1'
    zone: tuple[float,float,float,float]=(0.10,0.10,0.90,0.90)
    confidence: float=0.45
    model_mode: str='demonstracao_coco'
    source_kind: str='imagem'
    save_events: bool=True

    def __post_init__(self):
        x1,y1,x2,y2 = self.zone
        if not (0<=x1<x2<=1 and 0<=y1<y2<=1): raise ValueError('Zona inválida')
        if not 0<=self.confidence<=1: raise ValueError('Confiança inválida')
        if not self.camera.strip() or len(self.camera)>80: raise ValueError('Nome da câmera inválido')

@dataclass
class FrameResult:
    annotated: np.ndarray
    detections: list[Detection]
    zone_detections: list[Detection]
    recorded: int
    inference_ms: float


def process_frame(frame: np.ndarray, detector: Detector, config: MonitorConfig,
                  store: EventStore | None=None, throttle: EventThrottle | None=None,
                  monotonic_now: float | None=None) -> FrameResult:
    _validate_frame(frame)
    start=time.perf_counter()
    detections=detector.detect(frame,config.confidence)
    inference_ms=(time.perf_counter()-start)*1000
    annotated=frame.copy()
    h,w=frame.shape[:2]
    x1,y1,x2,y2=config.zone
    cv2.rectangle(annotated,(round(x1*w),round(y1*h)),(round(x2*w),round(y2*h)),(255,190,30),2)
    cv2.putText(annotated,'Zona monitorada',(round(x1*w),max(20,round(y1*h)-8)),cv2.FONT_HERSHEY_SIMPLEX,0.56,(255,190,30),2)
    inside=[]; accepted=[]
    for d in detections:
        # Não assume que todo detector externo retorna caixas válidas.
        if not d.validate(w,h): continue
        match=center_in_zone(d,config.zone,w,h)
        if match:inside.append(d)
        color=(20,145,250) if match else (70,220,95)
        cv2.rectangle(annotated,(int(d.x1),int(d.y1)),(int(d.x2),int(d.y2)),color,2)
        cv2.putText(annotated,f'{d.name}: {d.confidence:.2f}',(int(d.x1),max(20,int(d.y1)-7)),cv2.FONT_HERSHEY_SIMPLEX,.57,color,2)
        if match and config.save_events and store is not None:
            if throttle is None or throttle.accept(config.camera,d.name,time.monotonic() if monotonic_now is None else monotonic_now):
                accepted.append(d)
    recorded=0
    if accepted and store is not None:
        path=store.snaps/f'{uuid4().hex}.jpg'
        if not cv2.imwrite(str(path),annotated):raise OSError('Não foi possível salvar imagem do evento')
        for d in accepted:
            store.record(config.camera,d.name,d.confidence,str(path.relative_to(store.root)),config.model_mode,config.source_kind)
            recorded+=1
    return FrameResult(annotated, detections,inside,recorded,inference_ms)
