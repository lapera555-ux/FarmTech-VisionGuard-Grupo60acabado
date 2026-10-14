"""Promove best.pt para o aplicativo SOMENTE conforme validação (sem usar teste)."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import shutil

EXPERIMENTS=('yolo_30_epochs','yolo_60_epochs')

def validation_score(csv_path: Path) -> float:
    if not csv_path.is_file():raise FileNotFoundError(f'CSV de validação ausente: {csv_path}')
    with csv_path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames:raise ValueError('Resultados sem cabeçalho')
        keys={k.strip():k for k in reader.fieldnames}
        candidates=['metrics/mAP_0.5','metrics/mAP50(B)','metrics/mAP50']
        field=next((keys[k] for k in candidates if k in keys),None)
        if not field:raise ValueError('Coluna de mAP50 de validação não identificada')
        scores=[]
        for row in reader:
            try:
                value=float(row[field])
                if math.isfinite(value) and 0<=value<=1:scores.append(value)
            except (TypeError,ValueError):pass
    if not scores:raise ValueError(f'Sem mAP50 válido em {csv_path}')
    return max(scores)


def promote(root: Path) -> dict:
    root=Path(root)
    observations={}
    for exp in EXPERIMENTS:
        scores=root/'results'/exp/'results.csv'
        weight=root/'results'/'weights'/f'{exp}_best.pt'
        value=validation_score(scores)
        if not weight.is_file() or weight.stat().st_size<1024:
            raise FileNotFoundError(f'Checkpoint real ausente ou muito pequeno: {weight}')
        observations[exp]={'validation_map50':value,'checkpoint':str(weight)}
    selected=max(EXPERIMENTS,key=lambda n:observations[n]['validation_map50'])
    target=root/'models'/'best.pt'
    target.parent.mkdir(exist_ok=True,parents=True)
    shutil.copyfile(observations[selected]['checkpoint'],target)
    payload={'selection_criterion':'best validation mAP50 (NOT test mAP50)',
             'selected_experiment':selected,'selected_validation_map50':observations[selected]['validation_map50'],
             'experiments':observations}
    (root/'models'/'model_provenance.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    return payload

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    args=p.parse_args()
    print(json.dumps(promote(Path(args.root)),indent=2,ensure_ascii=False))
