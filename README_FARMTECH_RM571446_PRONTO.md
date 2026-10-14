# FarmTech VisionGuard Rural | FIAP Fase 6

**Aluno:** Pedro Gomes12  
**RM:** 571446  
**Grupo:** 60  
**Curso:** Tecnologia em Inteligência Artificial (On-Line)  
**Atividade:** Capítulo 1, O Despertar da Rede Neural | Fase 6

> **Projeto acadêmico executado.** As métricas abaixo provêm dos arquivos de resultados do notebook. O experimento foi realizado com rótulos oficiais do Open Images (`DEMO_OPENIMAGES_OFFICIAL`). **A rotulagem manual pelo Make Sense, exigida no enunciado, não está comprovada e permanece pendente.** A aplicação rural não foi validada em operação de campo.

## Acesso rápido

| Recurso | Link |
|---|---|
| Notebook executado no Google Colab | [Abrir notebook](https://colab.research.google.com/drive/1vvTeSznq2Ij-vpL9D-kuqE7l_I1J5fR1?usp=sharing) |
| Notebook versionado neste repositório | [Abrir arquivo](PedroViniciusGomesdosSantos_rm571446_pbl_fase6.ipynb) |
| Vídeo demonstrativo (YouTube não listado) | [Assistir vídeo](https://youtube.com/shorts/eRFRtGnIlJY) |
| Repositório público | [FarmTech VisionGuard](https://github.com/lapera555-ux/FarmTech-VisionGuard-Grupo60acabado) |
| Relatório técnico ilustrado | [Abrir PDF](reports/RELATORIO_TECNICO_COMPLETO_FARMTECH_RM571446.pdf) |
| Resultados e limitações | [Abrir análise](RESULTADOS_E_LIMITACOES.md) |

**Verificação antes da correção:** testar o acesso ao Colab e ao vídeo em janela anônima. A publicação do link não garante permissão de visualização.

## 1. Contexto e problema

A FarmTech Solutions pretende demonstrar o uso de visão computacional em cenários como segurança patrimonial e acompanhamento de entradas em propriedades rurais. Este projeto cria um **protótipo demonstrativo de detecção de pessoas e carros**, explorando seu uso em porteiras, acessos e áreas de circulação. Uma detecção indica presença visual e **não** determina autorização, identidade ou invasão.

## 2. Objetivo e arquitetura

1. Montar um dataset com 80 imagens reais, sendo 40 de `person` e 40 de `car`.
2. Comparar duas YOLOv5 customizadas com 30 e 60 épocas.
3. Comparar ambas a uma YOLOv5 padrão pré-treinada em COCO.
4. Treinar e avaliar uma CNN binária do zero.
5. Evidenciar os resultados com métricas, curvas e exemplos visuais.
6. Disponibilizar um demonstrador local em Windows (`app_fazenda.py`) para fotos, vídeos e fontes de câmera compatíveis.

**Fluxo científico:** imagens e bounding boxes → divisão treino/validação/teste → YOLOv5 e CNN → métricas no teste → análise crítica → protótipo rural. A YOLO detecta e localiza objetos; a CNN classifica a imagem como uma das duas classes. As duas tarefas não devem ser comparadas por uma única métrica.

## 3. Dataset e rótulos

| Classe | Treino | Validação | Teste | Total |
|---|---:|---:|---:|---:|
| Pessoa | 32 | 4 | 4 | 40 |
| Carro | 32 | 4 | 4 | 40 |
| **Total** | **64** | **8** | **8** | **80** |

As imagens são oriundas do Open Images, com identificadores e bounding boxes de referência. A amostra de teste tem apenas oito imagens e não é representativa de todas as condições de uma fazenda. **Integridade acadêmica:** o estado registrado no pacote é `DEMO_OPENIMAGES_OFFICIAL`; não afirmar que as 64 imagens de treino foram manualmente anotadas no Make Sense sem evidências. Uma etapa posterior deve exportar os rótulos manualmente no formato YOLO e repetir os treinamentos para satisfazer integralmente o requisito.

## 4. Resultados quantitativos observados

| Modelo | mAP@50 | Precision* | Recall* | Tempo médio de inferência | Treinamento |
|---|---:|---:|---:|---:|---:|
| YOLOv5 custom, 30 épocas | 76,25% | 0,31% | 100% | 86,48 ms | 89,00 s |
| YOLOv5 custom, 60 épocas | 89,58% | 2,25% | 100% | 26,86 ms | 104,46 s |
| YOLOv5 padrão COCO | 87,50% | 5,23% | 100% | 23,96 ms | não aplicado |
| CNN do zero | não aplicável | 0% | 0% | 20,43 ms | 9,98 s |

**CNN:** accuracy = **50%**, macro-F1 = **33,33%**. O AP@50 por classe da YOLO 60 foi **79,17% (person)** e **100% (car)**. Os arquivos de origem estão em `results/*.json`.

\* A avaliação de detecção utilizou um limiar de confiança bastante baixo, podendo incluir numerosos falsos positivos. Os valores de precision/recall do ponto operacional **não** são equivalentes ao mAP, que resume uma curva de precisão e recall. A comparação de tempos também é indicativa, pois difere conforme carregamento, contexto e hardware.

## 5. Interpretação crítica

A YOLO customizada com 60 épocas obteve maior mAP@50 nesta amostra (+13,33 pontos percentuais em relação a 30 épocas). Entretanto, o conjunto de teste com apenas oito imagens produz alta instabilidade estatística. A precision muito baixa sugere excesso de previsões de baixa confiança e exige reavaliação em um limiar operacional. A CNN treinada do zero apresentou apenas 50% de accuracy; a hipótese de classificação confiável não foi corroborada. **Nenhum resultado comprova prontidão de segurança operacional em fazendas.**

## 6. Estrutura do repositório

```text
.
├── README.md
├── PedroViniciusGomesdosSantos_rm571446_pbl_fase6.ipynb
├── app_fazenda.py
├── INICIAR_FAZENDA_WINDOWS.bat
├── index.html
├── dataset_80_imagens_e_rotulos.zip
├── results/             # métricas JSON, CSV e figuras reais
├── reports/             # relatório técnico completo em PDF
├── docs/                # metodologia e instruções
├── src/                 # avaliação e utilitários acadêmicos
├── scripts/             # auditoria, dataset e relatórios
├── operacao_rural/       # ingestão de câmera, detecção, eventos e SQLite
└── tests/               # testes automatizados, quando presentes
```

## 7. Como reproduzir os experimentos

1. Abra o [notebook do Colab](https://colab.research.google.com/drive/1vvTeSznq2Ij-vpL9D-kuqE7l_I1J5fR1?usp=sharing) e confirme que tem permissão de visualização.
2. Utilize GPU no Colab e monte o Drive, conforme as células do notebook.
3. Preserve as partições e a ordem de classes definidas no projeto.
4. Para cumprimento literal do enunciado, rotule manualmente as 64 imagens de treino no [Make Sense](https://www.makesense.ai/), exporte para YOLO, audite as labels e só então execute novamente os treinamentos.
5. Execute os experimentos, analise os registros e compare os resultados nas mesmas oito imagens de teste.
6. Confira as curvas e as imagens de teste processadas no notebook e em `results/`.

**Evite executar novamente um notebook só para reproduzir números sem fixar versões, seed e dataset.** O registro do ambiente consta em `results/runtime_metadata.json`.

## 8. Demonstração rural no Windows

Pré-requisito recomendado: Python 3.11, acesso à Internet para instalar as bibliotecas e obter os pesos. Na pasta do projeto execute `INICIAR_FAZENDA_WINDOWS.bat`. O aplicativo Streamlit abre normalmente em `http://127.0.0.1:8501`. A opção com modelo COCO é um modo de demonstração; para usar o detector customizado, configure os pesos `best.pt` gerados no treinamento. O sistema oferece entrada de imagens, arquivos de vídeo, captura webcam/RTSP, zona de interesse e histórico local SQLite.

**Limites:** o aplicativo não identifica indivíduos nem autoriza entradas; não foi ensaiado em produção e não substitui vigilância humana. Uma câmera rural requer validação com vídeos próprios, análise de falsos alarmes, proteção de credenciais e política de retenção de imagens.

## 9. Principais limitações e próximos passos

- Rotulagem manual obrigatória no Make Sense **não comprovada**.
- Apenas oito imagens de teste; generalização limitada.
- Classes genéricas pessoa/carro não cobrem caminhões, motocicletas, tratores ou cenários noturnos.
- Precision em limiar baixo requer avaliação adicional com limiar operacional e F1.
- CNN do zero não apresentou discriminação confiável.
- Ausência de validação independente de câmera, estabilidade 24/7 e segurança operacional.

## 10. Referências

- [Open Images](https://storage.googleapis.com/openimages/web/download.html)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [Ultralytics](https://docs.ultralytics.com/)
- [Make Sense](https://www.makesense.ai/)
- [PyTorch](https://pytorch.org/)
- [TensorFlow](https://www.tensorflow.org/)

---

**Autor:** Pedro Gomes12 · **RM 571446** · **Grupo 60** · **FIAP, Fase 6**. Resultados derivados de artefatos executados, sem métricas inventadas.
