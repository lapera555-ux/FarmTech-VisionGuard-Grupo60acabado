"""Processamento de arquivo de vídeo isolado do painel para testes de integração."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import cv2
from .engine import Detector
from .pipeline import MonitorConfig,process_frame
from .event_store import EventStore
from .core import EventThrottle

@dataclass
class VideoStats:
    frames_analyzed: int
    detections_total: int
    events_saved: int
    mean_inference_ms: float
    preview_jpeg: bytes
    output_file: Path


def analyze_video(src: Path,out: Path,detector: Detector,config: MonitorConfig,
                  store: EventStore | None = None, every_n: int = 5,max_frames: int = 120,
                  cooldown: float = 30.0,progress: Callable[[int],None] | None = None) -> VideoStats:
    if every_n<1 or max_frames<1:raise ValueError('Parâmetros de amostragem inválidos')
    cap=cv2.VideoCapture(str(src))
    if not cap.isOpened():
        cap.release()
        raise ValueError('Não foi possível abrir o arquivo de vídeo')
    out=Path(out)
    out.parent.mkdir(exist_ok=True,parents=True)
    writer=None
    count=sequence=total=events=0
    latencies=[]
    preview=None
    throttle=EventThrottle(cooldown)
    ok_finished=False
    try:
        fps=cap.get(cv2.CAP_PROP_FPS)
        fps=fps if 1<=fps<=120 else 25.
        outfps=max(1.,fps/every_n)
        while count<max_frames:
            ok,frame=cap.read()
            if not ok:break
            sequence+=1
            if sequence%every_n:continue
            if frame.shape[0]*frame.shape[1]>25_000_000:raise ValueError('Quadro maior que 25 MP')
            res=process_frame(frame,detector,config,store,throttle)
            if writer is None:
                h,w=frame.shape[:2]
                writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'MJPG'),outfps,(w,h))
                if not writer.isOpened():raise RuntimeError('Codec de vídeo MJPG indisponível')
            writer.write(res.annotated)
            if preview is None:
                ok,img=cv2.imencode('.jpg',res.annotated)
                if not ok:raise OSError('Erro ao codificar primeiro quadro')
                preview=img.tobytes()
            count+=1
            total+=len(res.detections)
            events+=res.recorded
            latencies.append(res.inference_ms)
            if progress:progress(count)
        if count==0:raise ValueError('Vídeo sem quadros selecionados')
        ok_finished=True
    finally:
        cap.release()
        if writer is not None:writer.release()
        if not ok_finished:out.unlink(missing_ok=True)
    return VideoStats(count,total,events,sum(latencies)/len(latencies),preview,out)
