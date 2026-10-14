from __future__ import annotations

import argparse
import csv
import shutil
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_tools import validate_yolo_label_file


def read_boxes(path: Path):
    boxes = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        cls, x, y, w, h = map(float, raw.split())
        boxes.append((
            int(cls),
            [x-w/2, y-h/2, x+w/2, y+h/2]
        ))
    return boxes


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    area_a = max(0,ax2-ax1)*max(0,ay2-ay1)
    area_b = max(0,bx2-bx1)*max(0,by2-by1)
    union = area_a + area_b - inter
    return 0 if union <= 0 else inter/union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--export-dir", required=True)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    export_dir = Path(args.export_dir)

    with (dataset/"manifest.csv").open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    train = [r for r in manifest if r["split"] == "train"]

    missing = []
    audit = []
    candidates = {}

    for row in train:
        stem = Path(row["filename"]).stem
        src = export_dir/f"{stem}.txt"
        if not src.exists():
            missing.append(src.name)
            continue

        errors = validate_yolo_label_file(src)
        if errors:
            raise ValueError("\n".join(errors))

        candidates[src.name] = src

        expected = int(row["primary_class_id"])
        all_manual = read_boxes(src)
        manual = [b for c,b in all_manual if c == expected]
        ref_path = dataset/"labels_reference"/"train"/src.name
        ref = [b for c,b in read_boxes(ref_path) if c == expected]

        best = 0.0
        if manual and ref:
            best = max(iou(a,b) for a in manual for b in ref)

        audit.append({
            "filename": row["filename"],
            "class": row["primary_class"],
            "manual_boxes": len(manual),
            "reference_boxes": len(ref),
            "best_iou_vs_reference": round(best, 4),
            "status": "OK" if best >= .50 and len(all_manual) == len(manual) and len(manual) == len(ref) else "REVISAR",
        })

    if missing:
        raise FileNotFoundError(
            f"Faltam {len(missing)} labels do Make Sense. "
            f"Primeiros: {missing[:8]}"
        )

    audit_path = dataset/"makesense_quality_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit[0].keys()))
        writer.writeheader()
        writer.writerows(audit)

    flagged = [r for r in audit if r["status"] == "REVISAR"]

    if flagged:
        raise ValueError(
            f"{len(flagged)} anotações precisam revisão. Labels não foram aplicadas. "
            f"Consulte {audit_path}."
        )
    # Somente após validação de todas as 64 anotações, aplicar substituições.
    dst_dir = dataset / "labels" / "train"
    for filename, src in candidates.items():
        shutil.copy2(src, dst_dir / filename)
    (dataset/"LABEL_MODE.txt").write_text(
        "FINAL_MAKESENSE_TRAIN\n"
        "Treino rotulado no Make Sense; validação/teste usam ground truth Open Images.\n",
        encoding="utf-8"
    )

    print("✅ 64 labels de treino substituídas pelas labels do Make Sense.")
    print("Auditoria:", audit_path)
    print("Itens para revisar:", len(flagged))
    for r in flagged[:10]:
        print(" -", r["filename"], r["best_iou_vs_reference"])


if __name__ == "__main__":
    main()
