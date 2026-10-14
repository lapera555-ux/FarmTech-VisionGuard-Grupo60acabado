# FarmTech VisionGuard Rural | FIAP Fase 6

**Pedro Vinicius Gomes dos Santos | RM 571446 | Grupo 60**  
**Projeto:** visão computacional para monitoramento experimental de acessos rurais e comparação científica de modelos de aprendizado profundo.

O repositório integra **duas entregas acadêmicas** e uma **aplicação rural local**. A aplicação demonstra a viabilidade de detectar pessoas e carros em imagens, gravações e câmeras; as entregas acadêmicas comparam YOLOv5 customizada (30 versus 60 épocas), YOLOv5 padrão COCO e CNN treinada do zero.

> **Status de execução:** código e testes locais preparados; os experimentos acadêmicos ainda devem ser executados no Colab e documentados com métricas reais. Nenhuma métrica artificial está sendo apresentada como resultado. O modelo COCO da aplicação não substitui a YOLOv5 customizada da FIAP.


**APRESENTAÇÃO VISUAL:** abra [`index.html`](index.html) no navegador sem instalar bibliotecas. Esta é uma apresentação estática; o painel de detecção funciona no `app_fazenda.py`.

## Abra primeiro

**Nova edição explicativa com painel profissional e guia completo:** [Manual passo a passo](docs/PASSO_A_PASSO_COMPLETO.md) | [PDF ilustrado](docs/GUIA_COMPLETO_FARMTECH_FIAP.pdf).

- [Instruções de instalação e fluxo completo](COMECE_AQUI.md)
- [Guia da aplicação rural](docs/GUIA_FAZENDA_APLICACAO.md)
- [Matriz requisito por requisito do enunciado](docs/MATRIZ_ENUNCIADO_OFICIAL.md)
- [Plano de validação em campo](docs/VALIDACAO_EM_CAMPO.md)
- [Status de testes e pendências](docs/STATUS_VERIFICADO.md)

## Aplicação para uso local na fazenda

No Windows, extraia o ZIP e execute **`INICIAR_FAZENDA_WINDOWS.bat`**. Na primeira execução, conexão com Internet é necessária para instalar dependências e obter os pesos públicos. O navegador acessa `http://127.0.0.1:8501`.

**Recursos:** análise de fotos; processamento de vídeos e download AVI; captura de webcam e RTSP; zonas configuráveis; detecções reais com `person/car`; armazenamento SQLite local, imagens dos eventos e exportação CSV.

**Motor de inferência:**

| Modo | Origem dos pesos | Finalidade |
|---|---|---|
| YOLO11n COCO | pesos públicos baixados no primeiro uso | demonstração imediata, não comprovante da FIAP |
| YOLOv5 customizada | `models/best.pt`, somente após treinamento real | demonstrar o modelo desenvolvido na FIAP |

**Limitações:** não identifica pessoas, placas, permissões nem invasões; não controla porteiras ou alarmes; a atualização depende do hardware. O painel está vinculado à interface local por padrão e não possui autenticação para publicação pública. Segurança real requer validação no local e infraestrutura apropriada.

## Dataset acadêmico

Duas classes: **pessoa (`person`) e automóvel (`car`)**, com 40 fotografias por classe do Open Images e seleção reprodutível por `ImageID`.

| Classe | Treino | Validação | Teste | Total |
|---|---:|---:|---:|---:|
| Pessoa | 32 | 4 | 4 | 40 |
| Carro | 32 | 4 | 4 | 40 |
| **Total** | **64** | **8** | **8** | **80** |

As **64 imagens de treino devem ser rotuladas manualmente no Make Sense**. O script de auditoria não permite promover labels com discrepâncias acima do limite configurado. As referências do Open Images são preservadas para validação e teste. Os diretórios de imagens são criados somente quando o download real termina.

## Colab e experimento obrigatório

**Notebook:** [`notebooks/PedroGomes_rm571446_pbl_fase6.ipynb`](notebooks/PedroGomes_rm571446_pbl_fase6.ipynb)

**Link público do Colab executado:** `INSIRA_URL_PUBLICA_DO_COLAB`

O notebook explica o dataset, faz controle de integridade, apresenta os dois treinamentos YOLOv5 (30 e 60 épocas), compara a YOLOv5 padrão, treina a CNN do zero, mede precisão/recall/F1/mAP50 e tempos, apresenta curvas, exemplos e conclusões. mAP de localização e acurácia de classificação não são tratados como equivalentes.

**Notebook extra (opcional):** [`ir_alem/PedroGomes_rm571446_ir_alem_fase6.ipynb`](ir_alem/PedroGomes_rm571446_ir_alem_fase6.ipynb), com MobileNetV2 transfer learning/fine-tuning e segmentação DeepLabV3. **Não considerar concluído antes de executar.**

## Repositório e demonstração

**Link do repositório GitHub público:** `INSIRA_URL_DO_GITHUB`

**Vídeo do YouTube não listado (até 5 minutos):** `INSIRA_URL_DO_VIDEO`

Insira URLs verdadeiras após a publicação e antes de executar o verificador. Não faça novos commits após o horário limite informado pela FIAP. Consulte [`docs/ROTEIRO_VIDEO.md`](docs/ROTEIRO_VIDEO.md) para a demonstração.

## Estrutura essencial

```text
FarmTech_VisionGuard/
  app_fazenda.py                 # painel local
  INICIAR_FAZENDA_WINDOWS.bat   # instalação e início Windows
  operacao_rural/               # YOLO, SQLite, frame/zone/evento
  models/                       # insira best.pt treinado (não incluso)
  notebooks/                    # entregas FIAP
  ir_alem/                      # entrega opcional
  scripts/                      # download, Make Sense, relatórios, auditoria
  dataset/                      # estrutura do dataset (80 fotos após download)
  results/                      # métricas reais após executar Colab
  tests/                        # testes automatizados da lógica
  docs/                         # metodologia e manuais
```

## Verificação e integridade

No diretório raiz:

```bash
python -m pytest -q
python -m compileall -q src scripts operacao_rural app_fazenda.py
python scripts/verificar_ambiente.py
python scripts/check_submission.py --root . --dataset dataset
```

O último comando **DEVE informar pendências** antes de Make Sense, execução real do notebook, geração de métricas e publicação dos links. A verificação de código e os testes de componentes não constituem evidência de mAP ou FPS real de câmera.

## Referências técnicas

- Open Images: https://storage.googleapis.com/openimages/web/download.html
- YOLOv5: https://github.com/ultralytics/yolov5
- Ultralytics YOLO: https://docs.ultralytics.com/
- Make Sense: https://www.makesense.ai/
- TensorFlow/Keras: https://www.tensorflow.org/
- PyTorch: https://pytorch.org/
