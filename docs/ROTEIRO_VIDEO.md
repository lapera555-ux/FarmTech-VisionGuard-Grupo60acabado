# Roteiro profissional do vídeo - até 5 minutos

## 0:00-0:30 - Problema
Apresente o FarmTech VisionGuard e o cenário de controle de acesso rural.

## 0:30-1:10 - Dataset
Mostre:
- 80 fotografias reais;
- 40 person + 40 car;
- divisão 64 treino / 8 validação / 8 teste;
- rotulação do treino no Make Sense.

## 1:10-2:10 - YOLO customizada
Compare 30 e 60 épocas por mAP50, precision, recall, losses e tempo.

## 2:10-3:00 - YOLO padrão
Mostre a YOLOv5 padrão no mesmo conjunto de teste e compare com a customizada.

## 3:00-3:50 - CNN do zero
Mostre accuracy, F1, matriz de confusão e explique que classificação é diferente de detecção.

## 3:50-4:35 - Demonstração
Mostre 2 ou 3 imagens de teste processadas com bounding boxes e confiança.

## 4:35-5:00 - Conclusão
Diga qual abordagem foi melhor no experimento e apresente as limitações.

Após os experimentos:

```bash
python scripts/generate_video_script.py
```

Isso cria `docs/ROTEIRO_VIDEO_PREENCHIDO.md` usando as métricas reais.
