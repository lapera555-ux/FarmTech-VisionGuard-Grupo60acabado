"""
FarmTech VisionGuard - aquisição de 80 fotografias reais do Open Images.

Sem FiftyOne. Usa:
- CSV oficial de bounding boxes
- bucket público Open Images em S3/CVDF via boto3 UNSIGNED

Classes:
  person = /m/01g317
  car    = /m/0k4j

Gera:
- 40 person + 40 car
- 32/4/4 por classe
- manifest.csv
- labels_reference/ com ground truth oficial
- labels/val e labels/test oficiais
- labels/train inicialmente oficiais em MODO_DEMO
  (depois devem ser substituídos pelas labels feitas no Make Sense)
"""
from __future__ import annotations

import argparse
import csv
import shutil
import urllib.request
from pathlib import Path

import boto3
import botocore
import pandas as pd
from tqdm.auto import tqdm

ANNOTATIONS_URLS = [
    "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "https://storage.googleapis.com/openimages/v6/oidv6-validation-annotations-bbox.csv",
]
PERSON_MID = "/m/01g317"
CAR_MID = "/m/0k4j"
CLASS_META = {
    PERSON_MID: ("person", 0),
    CAR_MID: ("car", 1),
}
SPLIT_PLAN = (("train", 32), ("val", 4), ("test", 4))


def download_first_available(urls: list[str], destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 100_000:
        return "cache"
    last_error = None
    for url in urls:
        try:
            print(f"Baixando anotações: {url}")
            urllib.request.urlretrieve(url, destination)
            if destination.stat().st_size > 100_000:
                return url
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Não foi possível baixar o CSV oficial: {last_error}")


def choose_samples(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    target = df[df["LabelName"].isin(CLASS_META.keys())].copy()

    # Uma única ocorrência entre person/car por imagem.
    per_image = target.groupby("ImageID").size()
    one_target_ids = set(per_image[per_image == 1].index)
    target = target[target["ImageID"].isin(one_target_ids)].copy()

    # Presets progressivos: começa rigoroso e relaxa apenas se necessário.
    presets = [
        # area min/max, exigir não-ocluído
        (0.06, 0.70, True),
        (0.04, 0.80, True),
        (0.03, 0.85, False),
    ]

    for area_min, area_max, strict in presets:
        work = target.copy()
        flag_cols = ["IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"]
        if strict:
            flag_cols = ["IsOccluded"] + flag_cols
        for col in flag_cols:
            if col in work.columns:
                work = work[work[col].fillna(0).astype(int) == 0]

        work["BoxArea"] = (
            (work["XMax"] - work["XMin"]) *
            (work["YMax"] - work["YMin"])
        )
        work = work[
            (work["BoxArea"] >= area_min) &
            (work["BoxArea"] <= area_max)
        ]

        result = {}
        enough = True
        for mid, (name, _) in CLASS_META.items():
            subset = work[work["LabelName"] == mid].copy()
            if len(subset) < 40:
                enough = False
                break
            result[name] = subset.sample(n=40, random_state=seed).reset_index(drop=True)
        if enough:
            print(
                f"Seleção: area={area_min:.2f}-{area_max:.2f}, "
                f"strict={strict}"
            )
            return result

    raise RuntimeError(
        "Não encontrei 40 imagens adequadas por classe após os filtros."
    )


def yolo_line(row: pd.Series, class_id: int) -> str:
    xmin, xmax = float(row.XMin), float(row.XMax)
    ymin, ymax = float(row.YMin), float(row.YMax)
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"


def download_images(mapping, cache_dir: Path, workers: int = 8):
    # boto3 public bucket, no credentials.
    config = botocore.config.Config(
        signature_version=botocore.UNSIGNED,
        max_pool_connections=max(10, workers + 2),
    )
    bucket = boto3.resource("s3", config=config).Bucket("open-images-dataset")
    raw = cache_dir / "images"
    raw.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def one(item):
        image_id = item["source_image_id"]
        dst = raw / f"{image_id}.jpg"
        if dst.exists() and dst.stat().st_size > 1000:
            return dst
        bucket.download_file(f"validation/{image_id}.jpg", str(dst))
        return dst

    futures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for item in mapping:
            futures.append(ex.submit(one, item))
        for f in tqdm(as_completed(futures), total=len(futures), desc="80 imagens"):
            f.result()
    return raw


def prepare(output: Path, seed: int = 42, workers: int = 8):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "_source_cache"
    cache.mkdir(parents=True, exist_ok=True)

    csv_path = cache / "validation-annotations-bbox.csv"
    source_url = download_first_available(ANNOTATIONS_URLS, csv_path)

    print("Lendo anotações...")
    df = pd.read_csv(csv_path)
    required = {"ImageID","LabelName","XMin","XMax","YMin","YMax"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV sem colunas: {sorted(missing)}")

    selected = choose_samples(df, seed)

    # Proteção contra destruição acidental das anotações manuais já validadas.
    state = output / "LABEL_MODE.txt"
    if state.exists() and state.read_text(encoding="utf-8").startswith("FINAL_MAKESENSE_TRAIN"):
        raise RuntimeError("Dataset final Make Sense detectado. Faça backup e utilize outro --output para regenerar.")

    # Reset only generated folders.
    for folder in ["images", "labels", "labels_reference"]:
        path = output / folder
        if path.exists():
            shutil.rmtree(path)
        for split in ["train", "val", "test"]:
            (path / split).mkdir(parents=True, exist_ok=True)

    manifest = []
    mapping = []

    for class_name, frame in selected.items():
        class_id = 0 if class_name == "person" else 1
        cursor = 0
        serial = 1
        for split, amount in SPLIT_PLAN:
            part = frame.iloc[cursor:cursor+amount]
            for _, row in part.iterrows():
                image_id = str(row["ImageID"])
                filename = f"{class_name}_{serial:03d}.jpg"
                item = {
                    "filename": filename,
                    "primary_class_id": class_id,
                    "primary_class": class_name,
                    "split": split,
                    "source_dataset": "Open Images V6/V7 boxable subset",
                    "source_split": "validation",
                    "source_image_id": image_id,
                    "annotation_csv": source_url,
                    "seed": seed,
                }
                manifest.append(item)
                mapping.append(item | {"row": row})
                serial += 1
            cursor += amount

    if len({item["source_image_id"] for item in manifest}) != 80:
        raise RuntimeError("Seleção duplicada entre os splits; interrompido por risco de vazamento")
    raw = download_images(manifest, cache, workers=workers)

    for item in mapping:
        row = item["row"]
        image_id = item["source_image_id"]
        split = item["split"]
        filename = item["filename"]
        class_id = int(item["primary_class_id"])

        src = raw / f"{image_id}.jpg"
        if not src.exists() or src.stat().st_size < 1000:
            raise RuntimeError(f"Imagem inválida: {image_id}")
        shutil.copy2(src, output/"images"/split/filename)

        line = yolo_line(row, class_id)
        stem = Path(filename).stem

        # Ground truth oficial preservado sempre.
        (output/"labels_reference"/split/f"{stem}.txt").write_text(line)

        # Dataset roda imediatamente em modo DEMO com labels oficiais.
        # Make Sense substituirá SOMENTE train para a versão final acadêmica.
        (output/"labels"/split/f"{stem}.txt").write_text(line)

    with (output/"manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    (output/"LABEL_MODE.txt").write_text(
        "DEMO_OPENIMAGES_OFFICIAL\n"
        "Antes da entrega final, substitua labels/train pelas labels do Make Sense.\n",
        encoding="utf-8"
    )

    train_files = [x["filename"] for x in manifest if x["split"] == "train"]
    (output/"MAKESENSE_TRAIN_FILES.txt").write_text(
        "\n".join(train_files) + "\n", encoding="utf-8"
    )

    print("\n✅ DATASET PRONTO")
    print("80 fotografias reais")
    print("40 person | 40 car")
    print("Treino 64 | Validação 8 | Teste 8")
    print("Modo atual: DEMO com labels oficiais Open Images")
    print("Para a entrega: rotule apenas as 64 de treino no Make Sense.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    prepare(Path(args.output), seed=args.seed, workers=args.workers)


if __name__ == "__main__":
    main()
