# Resultados reais consolidados | FIAP Fase 6

Pedro Vinicius Gomes dos Santos | RM 571446 | Grupo 60

Modo de rótulos registrado no Drive: `DEMO_OPENIMAGES_OFFICIAL
Antes da entrega final, substitua labels/train pelas labels do Make Sense.`

| Modelo | mAP50 | Accuracy | Precision | Recall | Inferência (ms) |
|---|---:|---:|---:|---:|---:|
| YOLOv5 custom 30 | 76.25% | não aplicável | 0.31% | 100.00% | 86.5 |
| YOLOv5 custom 60 | 89.58% | não aplicável | 2.25% | 100.00% | 26.9 |
| YOLOv5 padrão COCO | 87.50% | não aplicável | 5.23% | 100.00% | 24.0 |
| CNN do zero | não aplicável | 50.00% | 0.00% | 0.00% | 20.4 |

## Interpretação crítica
mAP50 avalia desempenho de localização/classificação de objetos a IoU 0,50; não representa porcentagem geral de acertos.
Acurácia da CNN não é diretamente comparável ao mAP dos detectores.
Precision calculada em limiar de confiança muito baixo pode produzir muitos falsos positivos; por isso deve ser interpretada junto ao AP.
O teste contém apenas 8 imagens; qualquer conclusão sobre operação contínua em uma fazenda exige validação adicional.
Se o modo de rótulo não for FINAL_MAKESENSE_TRAIN, a exigência de rotulagem manual continua pendente.
