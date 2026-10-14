"""Checagem não invasiva: diagnóstico do ambiente, não substitui testes de modelo."""
from __future__ import annotations
import importlib.util
import platform
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
print('FARMTECH VISIONGUARD - PRÉ-VERIFICAÇÃO')
print('Sistema:', platform.platform())
print('Python:', sys.version.split()[0])
print('Diretório:', ROOT)
for module in ['cv2','streamlit','ultralytics','torch','pandas','numpy','nbformat']:
    print(f'{module:14s}:', 'instalado' if importlib.util.find_spec(module) else 'não instalado')
print('Notebook principal:', 'presente' if (ROOT/'notebooks/PedroGomes_rm571466_pbl_fase6.ipynb').is_file() else 'AUSENTE')
print('Custom best.pt:', 'presente' if (ROOT/'models/best.pt').is_file() else 'não disponibilizado (esperado antes do treino)')
print('Repo YOLOv5 v7.0:', 'presente' if (ROOT/'third_party/yolov5/hubconf.py').is_file() else 'não preparado (necessário só no modo customizado)')
print('ATENÇÃO: existência de arquivos não prova desempenho do modelo.')
