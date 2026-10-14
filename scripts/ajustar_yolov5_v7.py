"""Compatibilidade limitada para YOLOv5 v7.0 com Pillow>=10.

Não faz downgrade de Pillow, não altera instalações do sistema.
Altera somente o checkout LOCAL temporário do YOLOv5.
"""
from __future__ import annotations
import argparse
from pathlib import Path

HELPER='''\n\ndef _farmtech_pil_text_size(font, text):\n    """Compatibilidade Pillow 9/10+ para fontes TrueType."""\n    if hasattr(font, 'getsize'):\n        return font.getsize(text)\n    left, top, right, bottom = font.getbbox(text)\n    return right - left, bottom - top\n\n'''

def patch(repo: Path) -> bool:
    path=Path(repo)/'utils'/'plots.py'
    if not path.exists():raise FileNotFoundError(f'Arquivo YOLOv5 v7.0 ausente: {path}')
    original=path.read_text(encoding='utf-8')
    if '_farmtech_pil_text_size(' in original:
        return False
    substitutions={
        'self.font.getsize(label)':'_farmtech_pil_text_size(self.font, label)',
        'self.font.getsize(text)':'_farmtech_pil_text_size(self.font, text)',
    }
    updated=original
    for old,new in substitutions.items():
        updated=updated.replace(old,new)
    if updated==original:
        raise ValueError('Fonte YOLOv5 diferente de v7.0: atualize a rotina de compatibilidade com inspeção manual')
    marker='# Settings'
    if marker not in updated:raise ValueError('Ponto de inserção não encontrado')
    updated=updated.replace(marker,HELPER+marker,1)
    compile(updated,str(path),'exec')
    path.write_text(updated,encoding='utf-8')
    return True

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',required=True,type=Path)
    args=p.parse_args()
    changed=patch(args.repo)
    print('Patch Pillow para checkout YOLOv5:', 'aplicado' if changed else 'já aplicado')
