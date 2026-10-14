# Manual técnico de operação: FarmTech VisionGuard Rural

## 1. Arquitetura e componentes

```text
[foto] [arquivo de vídeo] [webcam] [câmera RTSP]
                  |
                  v
    [Aplicação local: Streamlit / OpenCV]
                  |
                  v
       [Motor de detecção YOLO]
       |                   |
  COCO YOLO11n       YOLOv5 custom
 (pré-treinada)       (best.pt FIAP)
                  |
                  v
       [Confiança + caixa + zona]
                  |
                  v
     [Evento revisável + fotografia]
                  |
                  v
      [SQLite local + histórico CSV]
```

**Justificativa da escolha:** sem serviço de nuvem obrigatório, funciona a partir de um PC e abre no navegador do próprio PC. Custo recorrente de hospedagem zero. O primeiro uso exige Internet para instalar bibliotecas e baixar pesos COCO; o uso futuro pode depender da disponibilidade das bibliotecas/modelos em cache. Webcam local usa a câmera **do PC que executa o aplicativo**, não necessariamente a câmera do celular que abre o navegador.

## 2. Instalação Windows

- Instale Python 3.11 x64. Marque `Add Python to PATH`.
- Descompacte tudo em uma pasta sem caracteres especiais ou permissão restrita (por exemplo, Documentos/FarmTech_VisionGuard).
- Clique `INICIAR_FAZENDA_WINDOWS.bat`.
- Aguarde a criação da pasta `.venv/` e instalação de `streamlit`, `ultralytics`, `torch`, `opencv-python` e bibliotecas auxiliares.
- Acesse `http://127.0.0.1:8501` se o navegador não abrir. Nunca publique a porta 8501 na Internet.
- No primeiro processamento, a YOLO11n COCO baixa `yolo11n.pt`. Uma falha de rede ou instalação não indica que o modelo foi executado: confira a mensagem de erro.

Se a primeira execução falhar, rode em um terminal `python scripts/verificar_ambiente.py` e copie o erro para diagnóstico. Em CPU modesta, a detecção poderá ficar lenta. Use primeiro fotografias e vídeos curtos.

## 3. Procedimentos por tela

### Imagem

1. Selecione o detector no menu lateral.
2. Escolha a zona de interesse, a confiança e se deseja persistir eventos.
3. Faça upload JPG/PNG e clique **Analisar imagem**.
4. Confira caixas, classes, confiança e se o centro da caixa entrou no retângulo.
5. Baixe a imagem anotada; o histórico é criado somente se o salvamento estiver ligado e houver objetos na zona.

### Vídeo gravado

1. Envie um arquivo MP4/MOV/AVI de até 100 MB.
2. Ajuste a amostragem `1 quadro a cada N` e limite de quadros.
3. Clique **Processar vídeo**. O sistema analisa apenas os quadros selecionados.
4. Baixe AVI com os quadros anotados, sem áudio. Contagens são **detecções acumuladas**, não indivíduos únicos.

### Webcam ou câmera IP

1. Configure `Webcam USB/integrada` com índice 0 (ou outro índice) OU informe uma URL RTSP válida.
2. Clique **Iniciar monitoramento**. O navegador atualiza a imagem em intervalos solicitados de cerca de 1 segundo, condicionados à velocidade do hardware.
3. Monitore os resultados e encerre com **Parar câmera**.
4. Use RTSP somente em rede própria ou autorizada. Não publique credenciais, prints sensíveis ou arquivo SQLite.

### Histórico

- Os eventos ficam em `dados_locais/events.sqlite3` e as fotografias em `dados_locais/snapshots/`.
- `Evento` = detecção de classe dentro de zona, limitada por temporizador, não identificação de acesso não autorizado.
- O intervalo de repetição é por `câmera+classe`. Objetos distintos da mesma classe podem ser agregados e ocorrências podem se perder; **não há rastreamento individual**.
- Exporte CSV pelo botão da interface. Mantenha o PC protegido e aplique retenção adequada aos seus dados.

## 4. Diferença entre os dois modelos

| Aspecto | YOLO11n COCO | YOLOv5 customizada FIAP |
|---|---|---|
| Objetivo | Disponibilizar demonstração inicial | Avaliar dados rotulados da Fase 6 |
| Treinada com suas 64 imagens? | Não | Somente após Colab + Make Sense |
| Detecção `person`/`car` | Sim | Sim, quando treinamento funcionar |
| Dependência de `best.pt` | Não | Sim |
| Inferência real disponível neste ZIP sem instalar/baixar pesos? | Não | Não |
| É prova das entregas 1 e 2? | Não | Apenas com notebook completo e saídas reais |

**Peso do modelo não está incluso no ZIP.** Nunca carregue `.pt` desconhecido: checkouts antigos YOLOv5 e checkpoints PyTorch podem usar desserialização que executa código.

## 5. Gestão de riscos em propriedades rurais

- **Falsos negativos:** distância, baixa luz, borrões e poeira podem impedir a detecção.
- **Falsos positivos:** padrões visuais enganosos podem ativar alertas.
- **Risco operacional:** queda de energia, rede, câmera ou PC interrompe a inspeção. Não é um sistema de segurança certificado ou redundante.
- **Classes limitadas:** automóveis e pessoas. Tratores, vacas, caminhões e motocicletas exigiriam classes/dados específicos para detecção confiável.
- **Segurança:** não há login nem HTTPS no protótipo. Por isso a aplicação fica somente em `127.0.0.1`. Acesso externo requer autenticação e TLS por solução apropriada.
- **Privacidade:** se houver imagens de funcionários, visitantes ou placas, definir política de acesso, finalidade, retenção e descarte, conforme a legislação aplicável.
- **Decisão humana:** alertas são registros para revisão; não autorizam acionamento de fechaduras, portões, sirenes ou denúncia automática.

## 6. Evolução para operação 24/7 (fora do escopo implementado)

Para colocar em produção: instalar serviço supervisionado, autenticação, backup criptografado, política de retenção automática, observabilidade, reconexão com backoff, UPS, autorização e validação com amostras reais do local. Esses elementos **não são fornecidos como funcionalidades já testadas** neste pacote.
