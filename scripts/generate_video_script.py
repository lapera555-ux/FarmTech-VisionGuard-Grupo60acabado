from pathlib import Path
import json


RESULTS = Path("results")


def load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x):
    return f"{100*float(x):.1f}%"


def main():
    y30 = load("yolo_custom_30_metrics.json")
    y60 = load("yolo_custom_60_metrics.json")
    std = load("yolo_standard_metrics.json")
    cnn = load("cnn_metrics.json")

    missing = [n for n, v in [("YOLO30", y30), ("YOLO60", y60), ("YOLO padrão", std), ("CNN", cnn)] if v is None]
    if missing:
        raise FileNotFoundError("Execute os experimentos antes. Faltam: " + ", ".join(missing))

    text = f"""# Roteiro automático - vídeo de até 5 minutos

## 0:00-0:30 - Problema
Apresente o FarmTech VisionGuard: detecção de pessoas e carros para demonstrar visão computacional em controle de acesso.

## 0:30-1:10 - Dataset
Mostre a divisão 64 treino / 8 validação / 8 teste, com 40 imagens por classe, e a rotulação em bounding boxes.

## 1:10-2:10 - YOLO customizada
30 épocas: mAP50 {pct(y30['map50'])}, precision {pct(y30['precision'])}, recall {pct(y30['recall'])}.
60 épocas: mAP50 {pct(y60['map50'])}, precision {pct(y60['precision'])}, recall {pct(y60['recall'])}.
Explique se dobrar as épocas trouxe ganho mensurável ou sinais de overfitting.

## 2:10-3:05 - YOLO padrão
YOLOv5s COCO no mesmo teste: mAP50 {pct(std['map50'])}, precision {pct(std['precision'])}, recall {pct(std['recall'])}.
Compare com a versão customizada e destaque custo de treinamento versus generalização.

## 3:05-3:55 - CNN do zero
CNN: accuracy {pct(cnn['accuracy'])}, precision {pct(cnn['precision'])}, recall {pct(cnn['recall'])}.
Explique que classificação responde 'qual classe domina a imagem', enquanto YOLO localiza objetos com bounding boxes.

## 3:55-4:35 - Demonstração
Mostre 2 ou 3 imagens inéditas com as bounding boxes e confiança.

## 4:35-5:00 - Conclusão
Mostre limitações: dataset pequeno, apenas duas classes e cenário acadêmico. Encerre indicando o GitHub e o notebook executado.
"""
    out = Path("docs/ROTEIRO_VIDEO_PREENCHIDO.md")
    out.write_text(text, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
