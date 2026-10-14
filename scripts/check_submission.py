"""Auditoria conservadora: nunca inferir execução real da mera existência de um arquivo."""
from __future__ import annotations
from pathlib import Path
import argparse, csv, json, math, re, sys
from urllib.parse import urlparse
import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.dataset_tools import validate_dataset

REQUIRED_RESULTS = {
    "yolo_custom_30_metrics.json": ("precision", "recall", "f1", "map50", "ap50_person", "ap50_car", "mean_inference_ms", "num_images", "training_seconds"),
    "yolo_custom_60_metrics.json": ("precision", "recall", "f1", "map50", "ap50_person", "ap50_car", "mean_inference_ms", "num_images", "training_seconds"),
    "yolo_standard_metrics.json": ("precision", "recall", "f1", "map50", "ap50_person", "ap50_car", "mean_inference_ms", "num_images"),
    "cnn_metrics.json": ("accuracy", "precision", "recall", "f1", "mean_inference_ms", "num_images", "training_seconds"),
}


def check_results(results: Path) -> list[str]:
    errors = []
    for filename, keys in REQUIRED_RESULTS.items():
        path = results / filename
        if not path.exists():
            errors.append(f"Resultado ausente: results/{filename}")
            continue
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(d, dict): raise ValueError("JSON precisa ser um objeto")
            for key in keys:
                if key not in d: raise ValueError(f"campo ausente {key}")
                v = d[key]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise ValueError(f"valor não numérico/finito: {key}")
                if key in ("num_images",) and v != 8: raise ValueError("teste não contém exatamente 8 imagens")
                if key in ("mean_inference_ms", "training_seconds") and v <= 0: raise ValueError(f"tempo inválido: {key}")
                if key not in ("num_images", "mean_inference_ms", "training_seconds") and not 0 <= v <= 1:
                    raise ValueError(f"métrica fora de [0,1]: {key}")
        except (ValueError, OSError, json.JSONDecodeError) as e:
            errors.append(f"Métrica inválida {filename}: {e}")
    csvpath = results / "metrics_summary.csv"
    if not csvpath.exists():
        errors.append("Tabela consolidada ausente: results/metrics_summary.csv")
    else:
        try:
            with csvpath.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            if len(rows) < 4: errors.append("Tabela comparativa sem os quatro experimentos")
        except Exception as e: errors.append(f"Tabela comparativa ilegível: {e}")
    return errors


def check_notebook(path: Path) -> list[str]:
    if not path.exists(): return ["Notebook principal não encontrado"]
    errors = []
    nb = nbformat.read(path, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    incomplete = [i for i, c in enumerate(code, 1) if c.get("execution_count") is None]
    if incomplete: errors.append(f"Notebook tem {len(incomplete)} células de código sem execução: {incomplete}")
    for i,c in enumerate(code, 1):
        if any(o.get("output_type") == "error" for o in c.get("outputs", [])):
            errors.append(f"Notebook contém exceção registrada na célula {i}")
    if len(code) < 15: errors.append("Notebook inesperadamente curto")
    return errors


def check_links(root: Path) -> list[str]:
    errors=[]
    readme_path=root/'README.md'
    if not readme_path.exists(): return ['README não encontrado']
    readme=readme_path.read_text(encoding='utf-8')
    if re.search(r'PREENCHER|INSIRA_URL|<INSIRA', readme): errors.append('README tem placeholders')
    links = re.findall(r'https?://[^\s)<>]+', readme)
    required = {'Colab': ('colab.research.google.com',), 'YouTube': ('youtube.com','youtu.be'), 'GitHub': ('github.com',)}
    for name,domains in required.items():
        if not any(urlparse(url).hostname in domains for url in links):
            errors.append(f'Link {name} ausente ou com domínio inválido no README')
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--dataset', default='dataset')
    args=ap.parse_args()
    root=Path(args.root).resolve()
    errors=check_notebook(root/'notebooks/PedroGomes_rm571466_pbl_fase6.ipynb')
    errors.extend(check_links(root))
    dataset=root/args.dataset
    if (dataset/'manifest.csv').exists():
        errors.extend(validate_dataset(dataset, require_labels=True).errors)
        mode=dataset/'LABEL_MODE.txt'
        if not mode.exists() or not mode.read_text(encoding='utf-8').startswith('FINAL_MAKESENSE_TRAIN'):
            errors.append('Make Sense não finalizado')
        audit=dataset/'makesense_quality_audit.csv'
        if not audit.exists(): errors.append('Auditoria de qualidade Make Sense ausente')
        else:
            with audit.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
            if len(rows)!=64 or any(r.get('status')!='OK' for r in rows):
                errors.append('Anotações Make Sense precisam revisão')
    else: errors.append('Dataset e manifest ausentes')
    errors.extend(check_results(root/'results'))
    if errors:
        print('CHECKLIST: PENDÊNCIAS')
        for e in errors: print(' -',e)
        raise SystemExit(1)
    print('CHECKLIST: OK (verificação estática; revisar manualmente execução, qualidade das imagens e links públicos)')

if __name__=='__main__':main()
