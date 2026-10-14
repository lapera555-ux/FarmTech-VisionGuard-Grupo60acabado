
### Avaliação crítica final

**Facilidade de uso/integração.** A YOLOv5 padrão foi a alternativa mais simples de
integrar porque não precisou de customização. A YOLO customizada exigiu rotulação e
treinamento, mas passou a otimizar diretamente as classes do projeto. A CNN do zero
teve implementação conceitualmente simples, porém resolve classificação, e não localização.

**Precisão/desempenho.** A melhor YOLO customizada foi **60 épocas**
(mAP50=0.896). A YOLO padrão obteve
mAP50=0.875. A CNN atingiu accuracy=0.500 e
macro-F1=0.333. Esses números devem ser comparados dentro
da tarefa correspondente: mAP para detecção e accuracy/F1 para classificação.

**Custo de treinamento.** O treino de 30 épocas levou 89.0s e
o de 60 épocas 104.5s (razão aproximada 1.17x).
A YOLO padrão não exigiu customização. A CNN levou 10.0s.

**Tempo de inferência.** Entre os experimentos medidos, a menor média foi de
**CNN do zero**, com 20.43 ms/imagem no ambiente desta execução.
Os tempos dependem do hardware e devem ser interpretados como comparação deste Colab,
não como benchmark universal.

**Limitações.** O teste contém apenas 8 imagens, conforme a divisão solicitada no
enunciado. Portanto, pequenas diferenças absolutas devem ser interpretadas com cautela.
As conclusões demonstram o comportamento do protótipo, não desempenho de produção.
