from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv
import math

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED = {
    "train": {"person": 32, "car": 32},
    "val": {"person": 4, "car": 4},
    "test": {"person": 4, "car": 4},
}
CLASS_TO_ID = {"person": 0, "car": 1}


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str]
    counts: dict[str, dict[str, int]]
    label_files: dict[str, int]

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError("Dataset inválido:\n- " + "\n- ".join(self.errors))


def _images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def infer_primary_class(filename: str) -> str | None:
    stem = Path(filename).stem.lower()
    if stem.startswith("person_"):
        return "person"
    if stem.startswith("car_"):
        return "car"
    return None


def validate_yolo_label_file(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"Label ausente: {path.name}"]

    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{line_number} deve possuir 5 campos YOLO")
            continue
        try:
            if parts[0] not in ("0", "1"):
                errors.append(f"{path.name}:{line_number} classe deve ser inteiro 0 ou 1")
                continue
            cls = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"{path.name}:{line_number} contém valor não numérico")
            continue
        if cls not in (0, 1):
            errors.append(f"{path.name}:{line_number} classe inválida {cls}; esperado 0 ou 1")
        if any(not math.isfinite(v) for v in coords):
            errors.append(f"{path.name}:{line_number} contém NaN ou infinito")
            continue
        if any(v < 0 or v > 1 for v in coords):
            errors.append(f"{path.name}:{line_number} coordenadas devem estar entre 0 e 1")
        if coords[2] <= 0 or coords[3] <= 0:
            errors.append(f"{path.name}:{line_number} largura/altura devem ser > 0")
        x, y, w, h = coords
        if x - w / 2 < -1e-6 or x + w / 2 > 1+1e-6 or y - h / 2 < -1e-6 or y + h / 2 > 1+1e-6:
            errors.append(f"{path.name}:{line_number} box ultrapassa limites da imagem")
    return errors


def validate_dataset(root: str | Path, require_labels: bool = True) -> ValidationReport:
    root = Path(root)
    errors: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    label_files: dict[str, int] = {}

    for split, expected in EXPECTED.items():
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = _images(image_dir)

        split_counts = {"person": 0, "car": 0}
        for image in images:
            cls = infer_primary_class(image.name)
            if cls is None:
                errors.append(
                    f"{split}/{image.name}: nome deve começar com person_ ou car_ "
                    "para manter a rastreabilidade da classe primária"
                )
                continue
            split_counts[cls] += 1

            if require_labels:
                label_path = label_dir / f"{image.stem}.txt"
                errors.extend(validate_yolo_label_file(label_path))

        counts[split] = split_counts
        label_files[split] = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0

        for cls, expected_count in expected.items():
            actual = split_counts[cls]
            if actual != expected_count:
                errors.append(
                    f"{split}/{cls}: encontrado {actual}; esperado exatamente {expected_count}"
                )

        if require_labels and label_files[split] != len(images):
            errors.append(
                f"{split}: {label_files[split]} labels para {len(images)} imagens. "
                "Para avaliação profissional, todas as 80 imagens devem estar rotuladas."
            )

    manifest_path = root / "manifest.csv"
    if manifest_path.exists():
        rows = read_manifest(manifest_path)
        ids = [r.get("source_image_id") for r in rows]
        if len(rows) != 80 or len(set(ids)) != 80:
            errors.append("Manifest deve conter 80 IDs de origem distintos (sem vazamento entre splits)")
        pairs = {(r.get("split"), r.get("filename")) for r in rows}
        if len(pairs) != 80:
            errors.append("Manifest tem nomes duplicados")
        for r in rows:
            if not (root / "images" / r.get("split", "") / r.get("filename", "")).exists():
                errors.append(f"Arquivo listado no manifest ausente: {r.get('filename')}")
    total_images = sum(sum(v.values()) for v in counts.values())
    if total_images != 80:
        errors.append(f"Total de imagens = {total_images}; esperado 80")

    return ValidationReport(ok=not errors, errors=errors, counts=counts, label_files=label_files)


def yolo_xywhn_to_xyxy(values: Iterable[float], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x, y, w, h = map(float, values)
    x1 = (x - w / 2) * image_width
    y1 = (y - h / 2) * image_height
    x2 = (x + w / 2) * image_width
    y2 = (y + h / 2) * image_height
    return x1, y1, x2, y2


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
