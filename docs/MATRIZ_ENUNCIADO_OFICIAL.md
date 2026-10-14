# Matriz completa de aderência ao enunciado FIAP Fase 6

## Entrega 1

| Exigência | Artefato | Condição real para concluir |
|---|---|---|
| 40 fotos de objeto A + 40 objeto B | `scripts/download_openimages.py` | Executar, verificar imagens reais |
| 32/4/4 por classe | `dataset/manifest.csv` e estrutura | Conferir validação automatizada |
| Google Drive | `notebooks/00_PREPARAR_DATASET_GOOGLE.ipynb` | Executar no Drive do aluno |
| Rotular treinamento no Make Sense | `scripts/organize_makesense_labels.py` | 64 rotulagens humanas com exportação YOLO |
| Colab passo a passo | `notebooks/PedroGomes_rm571446_pbl_fase6.ipynb` | Executar e salvar saídas |
| Treino 30 e 60 épocas | Mesmo notebook | Rodar ambos em GPU |
| Comparar acurácia/erro/desempenho | JSONs, curvas, `analise_critica.md` | Gerar métricas reais e interpretar |
| Prints das detecções em teste | Célula 43 do notebook | Executar e salvar figuras/saídas |
| GitHub público com nome correto | README e notebook | Criar repositório real com arquivo nomeado |
| Vídeo YouTube <= 5 minutos | `docs/ROTEIRO_VIDEO.md` | Gravar e postar não listado |

## Entrega 2

| Exigência | Artefato | Condição |
|---|---|---|
| YOLO padrão | Célula 29 do notebook principal | Executar inferência real COCO v5s |
| CNN do zero | Células 31 a 34 | Treinar do zero e testar |
| Comparação de precisão, integração e tempo | Células 35 a 41; JSONs | Gerar métricas, avaliar e explicar diferenças |
| Notebook com saídas e conclusões | Notebook principal | Salvar todas as células executadas |

## Itens opcionais Ir Além

O notebook `ir_alem/PedroGomes_rm571446_ir_alem_fase6.ipynb` contempla MobileNetV2 transfer learning/fine-tuning e segmentação DeepLabV3. Deve ser executado e avaliado antes de afirmar que o extra foi concluído. A aplicação local com webcam não representa, sozinha, implementação ESP32-CAM via Wi-Fi.

## Barema oficial (10 pontos)

| Critério | Pontos | Evidência necessária |
|---|---:|---|
| GitHub público, nome e prazo | 1,5 | URL funcional, sem commits após prazo |
| Notebook executado e analítico | 3,0 | Execução, saída, análises e conclusões |
| README | 2,0 | Introdução, link Colab e YouTube |
| Vídeo | 2,0 | Demonstração real de até 5 minutos |
| Organização | 1,5 | Estrutura clara, código e imagens legíveis |

**Status neste ZIP:** fonte implementada, mas métricas e evidências de execução ainda ausentes. O comando `scripts/check_submission.py` deverá reportar pendências até serem produzidas. Não é correto antecipar a nota ou declarar aprovação total.
