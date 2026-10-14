# Metodologia

## Problema

Demonstrar para um cliente da FarmTech Solutions o funcionamento de visão computacional
em um cenário de controle de acesso, comparando detecção customizada, detecção genérica
pré-treinada e classificação com CNN criada do zero.

## Classes

- Pessoa (`person`)
- Carro (`car`)

A escolha permite comparar o modelo customizado com uma YOLOv5 padrão treinada em COCO,
pois ambas são classes conhecidas pelo modelo pré-treinado.

## Experimentos

### YOLOv5 customizada - 30 épocas
Transfer learning com `yolov5s.pt`.

### YOLOv5 customizada - 60 épocas
Mesma configuração e mesma seed; apenas o número de épocas muda. Isso permite atribuir
diferenças de desempenho principalmente à duração do treinamento.

### YOLOv5 padrão
`yolov5s.pt` sem fine-tuning. As classes COCO `person=0` e `car=2` são remapeadas para
`person=0` e `car=1` na avaliação, usando o mesmo conjunto de teste e IoU=0.50.

### CNN do zero
Rede convolucional binária implementada em TensorFlow/Keras, sem backbone pré-treinado.
Avaliada por accuracy, precision, recall, F1 e matriz de confusão.

## Métricas

Detecção:
- Precision
- Recall
- F1
- AP50 por classe
- mAP50
- tempo médio de inferência

Classificação:
- Accuracy
- Precision
- Recall
- F1
- matriz de confusão
- tempo médio de inferência

mAP de detecção não é tratado como equivalente à accuracy de classificação.
