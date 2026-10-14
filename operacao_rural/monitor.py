"""Monitor local de fazenda: webcam, RTSP, arquivo de vídeo ou imagem."""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import cv2
from operacao_rural.core import EventThrottle, append_event, center_in_zone, load_yolov5, predict


def run(args):
    model=load_yolov5(Path(args.weights),Path(args.yolov5_repo),args.confidence)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    zone=tuple(float(v) for v in args.zone.split(','))
    if len(zone)!=4 or not (0 <= zone[0] < zone[2] <=1 and 0 <= zone[1] < zone[3] <=1):
        raise ValueError('Use --zone 0.1,0.1,0.9,0.9')
    src=int(args.source) if args.source.isdigit() else args.source
    cap=cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError('Não foi possível abrir fonte de imagem. Confira webcam, caminho ou RTSP.')
    throttle=EventThrottle(args.cooldown)
    idx=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            idx+=1
            if idx % args.every_n != 0: continue
            h,w=frame.shape[:2]
            cv2.rectangle(frame,(int(zone[0]*w),int(zone[1]*h)),(int(zone[2]*w),int(zone[3]*h)),(255,180,0),2)
            for d in predict(model,frame):
                in_zone=center_in_zone(d,zone,w,h)
                color=(0,165,255) if in_zone else (60,200,80)
                cv2.rectangle(frame,(int(d.x1),int(d.y1)),(int(d.x2),int(d.y2)),color,2)
                cv2.putText(frame,f'{d.name}: {d.confidence:.2f}',(int(d.x1),max(20,int(d.y1)-8)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
                if in_zone and throttle.accept(args.camera,d.name,time.monotonic()):
                    filename=f'{time.time_ns()}_{d.name}.jpg'
                    snapshot=output/'snapshots'/filename
                    snapshot.parent.mkdir(parents=True,exist_ok=True)
                    if not cv2.imwrite(str(snapshot),frame): raise OSError('Falha ao salvar snapshot')
                    append_event(output/'events.csv',args.camera,d,str(snapshot),True)
                    print(f'EVENTO {d.name} {d.confidence:.2f}: {snapshot}')
            if not args.headless:
                cv2.imshow('FarmTech VisionGuard Rural (pressione Q para sair)',frame)
                if cv2.waitKey(1)&0xFF == ord('q'): break
            if args.max_frames and idx>=args.max_frames: break
    finally:
        cap.release()
        if not args.headless: cv2.destroyAllWindows()


def main():
    p=argparse.ArgumentParser(description='Monitoramento rural experimental; não aciona portões automaticamente')
    p.add_argument('--source',default='0',help='0 para webcam, caminho mp4 ou URL RTSP (não divulgue credenciais)')
    p.add_argument('--weights',default='runs/train/exp/weights/best.pt')
    p.add_argument('--yolov5-repo',default='yolov5')
    p.add_argument('--output',default='monitoramento_local')
    p.add_argument('--camera',default='porteira_1')
    p.add_argument('--confidence',type=float,default=.45)
    p.add_argument('--zone',default='0.1,0.1,0.9,0.9')
    p.add_argument('--cooldown',type=float,default=30.)
    p.add_argument('--every-n',type=int,default=1)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--max-frames',type=int,default=0)
    args=p.parse_args()
    if not 0 <= args.confidence <= 1 or args.every_n<1 or args.max_frames<0: p.error('Parâmetros inválidos')
    run(args)

if __name__=='__main__': main()
