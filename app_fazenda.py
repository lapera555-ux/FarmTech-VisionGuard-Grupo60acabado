"""FarmTech VisionGuard Rural — painel web local para Windows/Linux.

Execute a partir da raiz: python -m streamlit run app_fazenda.py
Não expõe interfaces na Internet por padrão. Não controla portões nem toma decisões de segurança.
"""
from __future__ import annotations
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from operacao_rural.core import EventThrottle
from operacao_rural.engine import build_detector
from operacao_rural.event_store import EventStore
from operacao_rural.pipeline import MonitorConfig, process_frame
from operacao_rural.video_tools import analyze_video

ROOT=Path(__file__).resolve().parent
DATA_LOCAL=ROOT/'dados_locais'

st.set_page_config(page_title='FarmTech VisionGuard | Monitoramento Rural',page_icon='🌾',layout='wide',initial_sidebar_state='expanded')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
:root{--farm-dark:#092d27;--farm-green:#087b61;--farm-lime:#badc88;--farm-muted:#66766f}
html,body,[class*=css],p,li{font-family:'DM Sans',sans-serif}
h1,h2,h3{font-family:'Outfit','DM Sans',sans-serif!important;letter-spacing:-.02em}
.block-container{padding-top:1.1rem;max-width:1420px;padding-bottom:4rem}
section[data-testid='stSidebar']{background:linear-gradient(160deg,#f2f7f2,#fafcf9)}
[data-testid='stSidebar'] .stRadio label{font-size:.88rem}
[data-testid='stMetric']{background:linear-gradient(180deg,#fff,#f6faf7);border:1px solid #d7e6dd;border-radius:15px;padding:14px 18px;box-shadow:0 3px 12px rgba(12,59,44,.03)}
[data-testid='stMetricValue']{font-family:'Outfit',sans-serif;font-weight:750;color:#0b5947}
button[data-baseweb='tab']{padding:12px 14px!important;font-weight:650!important}
.stButton>button[kind='primary']{background:#087b61;border-color:#087b61;border-radius:11px;font-weight:700}
.stButton>button[kind='primary']:hover{background:#075c4b;border-color:#075c4b}
.stDownloadButton>button{border-radius:10px;border-color:#94baa7}
.farm-hero{background:radial-gradient(circle at 90% 5%,rgba(180,233,162,.3),transparent 32%),linear-gradient(110deg,#073c35 0%,#075a4d 57%,#08866a 100%);border-radius:22px;padding:34px 36px;color:white;margin:0 0 20px;position:relative;overflow:hidden}
.farm-kicker{color:#d6f0d5;font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;font-weight:700}
.farm-hero h1{color:white!important;font-size:2.2rem;font-weight:800;margin:8px 0!important;letter-spacing:-.04em}
.farm-hero p{color:#dcefe9;font-size:1.02rem;max-width:900px;line-height:1.55;margin:0 0 16px}
.farm-pill{display:inline-block;border:1px solid #7ebdaf9f;background:#ffffff12;padding:7px 12px;color:#e5fbf0;border-radius:100px;margin-right:8px;margin-bottom:7px;font-size:.74rem;font-weight:600}
.farm-section{font-size:.72rem;color:#078465;text-transform:uppercase;letter-spacing:.13em;font-weight:800;margin:20px 0 7px}
.farm-card{background:#fff;border:1px solid #d9e8df;border-radius:17px;padding:18px 19px;min-height:145px;margin-bottom:12px;box-shadow:0 5px 20px rgba(15,64,49,.04)}
.farm-card h3{margin:0 0 7px;font-size:1.1rem;color:#133e35!important}
.farm-card p{font-size:.88rem;color:#4d6258;line-height:1.55;margin-bottom:0}
.farm-tag{display:inline-block;padding:4px 10px;border-radius:99px;background:#e8f5ea;color:#136448;font-size:.73rem;font-weight:800;margin-bottom:8px}
.farm-warning{background:#fff8eb;border:1px solid #eedab5;border-radius:12px;padding:13px 17px;color:#705225;font-size:.89rem;line-height:1.55;margin:12px 0}
.farm-flow{background:#f1f7f3;border:1px solid #d7e7db;border-radius:15px;padding:17px 14px;text-align:center;color:#194a3b;font-weight:750;font-size:.93rem;margin:3px 0 11px}
.farm-sidebar{background:#e5f2e7;padding:18px;border-radius:17px;border:1px solid #bed9c9;margin-bottom:18px;color:#113f34}
.farm-sidebar h3{font-size:1.03rem;margin:0 0 7px!important}
.farm-sidebar p{font-size:.79rem;margin:0;color:#456354}
.farm-caption{font-size:.84rem;color:#657b70;line-height:1.6}
@media(max-width:680px){.farm-hero{padding:22px 20px;border-radius:15px}.farm-hero h1{font-size:1.65rem}.block-container{padding-left:1rem;padding-right:1rem}}
</style>
""",unsafe_allow_html=True)
st.markdown("""
<div class='farm-hero'>
  <div class='farm-kicker'>FARMTECH SOLUTIONS · VISÃO COMPUTACIONAL · FIAP FASE 6</div>
  <h1>VisionGuard <span style='color:#badc88'>Rural</span></h1>
  <p>Monitoramento inteligente de acessos rurais. Explore como redes neurais detectam pessoas e veículos, acompanhe eventos e conheça as limitações de cada modelo.</p>
  <span class='farm-pill'>Pedro Gomes</span><span class='farm-pill'>RM 571466</span><span class='farm-pill'>Grupo 60</span><span class='farm-pill'>Protótipo educacional</span>
</div>""",unsafe_allow_html=True)

@st.cache_resource(show_spinner='Carregando rede neural. A primeira inicialização pode baixar pesos oficiais...')
def load_detector(mode: str):
    return build_detector(mode,ROOT)

@st.cache_resource
def event_store():
    return EventStore(DATA_LOCAL)

store=event_store()
st.sidebar.markdown("""<div class='farm-sidebar'><h3>🌱 CENTRAL RURAL</h3><p>Projeto de Pedro Gomes<br>RM 571466 · Grupo 60</p></div>""",unsafe_allow_html=True)
st.sidebar.header('Configuração do detector')
st.sidebar.caption('Escolha qual rede neural irá processar as imagens. O modelo COCO é público; o modelo FIAP requer pesos realmente treinados.')
mode=st.sidebar.radio('Modelo de visão computacional',options=['demonstracao_coco','yolov5_treinada'],
    format_func=lambda v: 'YOLO11n pré-treinada COCO (uso imediato)' if v=='demonstracao_coco' else 'YOLOv5 customizada FIAP (best.pt)')
if mode=='demonstracao_coco':
    st.sidebar.info('Modelo público COCO, com as classes person e car. NÃO representa o treinamento acadêmico da FIAP.')
else:
    st.sidebar.info('Exige models/best.pt e third_party/yolov5. Use o script PREPARAR_MODELO_CUSTOM_WINDOWS.bat.')
confidence=st.sidebar.slider('Confiança mínima',0.10,0.95,0.45,0.05)
st.sidebar.subheader('Zona monitorada')
x_bounds=st.sidebar.slider('Extensão horizontal (%)',0,100,(10,90))
y_bounds=st.sidebar.slider('Extensão vertical (%)',0,100,(10,90))
zone=(x_bounds[0]/100,y_bounds[0]/100,x_bounds[1]/100,y_bounds[1]/100)
camera=st.sidebar.text_input('Nome do ponto monitorado','porteira_1',max_chars=80)
save_events=st.sidebar.toggle('Salvar eventos e imagens no computador',value=True)
cooldown=st.sidebar.number_input('Intervalo mínimo entre eventos da mesma classe (segundos)',min_value=0,max_value=3600,value=30,step=5)
st.sidebar.caption('O retângulo representa uma região de interesse. Uma caixa só dispara registro quando seu centro estiver na zona. Não indica invasão.')

if not camera.strip():
    st.error('Preencha o nome do ponto monitorado.');st.stop()

def cfg(source_kind: str) -> MonitorConfig:
    return MonitorConfig(camera=camera.strip(),zone=zone,confidence=confidence,model_mode=mode,source_kind=source_kind,save_events=save_events)

def model():
    try:
        return load_detector(mode)
    except Exception as e:
        st.error('Não foi possível iniciar o detector. Verifique a conexão no primeiro uso e a instalação das dependências.')
        st.code(f'{type(e).__name__}: {str(e)[:500]}')
        st.stop()

def image_from_bytes(blob: bytes):
    if len(blob)>25*1024*1024:raise ValueError('Arquivo superior a 25 MB')
    arr=np.frombuffer(blob,np.uint8)
    img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if img is None:raise ValueError('Não foi possível decodificar esta imagem')
    if img.shape[0]*img.shape[1]>25_000_000:raise ValueError('Imagem com resolução excessiva (>25 MP)')
    return img

def show_metrics(dets,within,events,latency):
    cols=st.columns(4)
    for c,lab,val in zip(cols,['Detecções','Na zona','Eventos salvos','Inferência por quadro'],
                         [len(dets),len(within),events,f'{latency:.0f} ms']):
        c.metric(lab,val)

home_tab,image_tab,video_tab,live_tab,history_tab,guide_tab=st.tabs(['🏡 Início','📷 Fotografias','🎬 Vídeos','📡 Câmera ao vivo','📚 Histórico','🎓 Ciência e FIAP'])

with home_tab:
    st.markdown('<div class="farm-section">BOAS-VINDAS E OBJETIVO</div>',unsafe_allow_html=True)
    st.subheader('Visão computacional aplicada à rotina de uma fazenda')
    st.write('O objetivo é identificar visualmente duas classes, **pessoa** (`person`) e **automóvel** (`car`), em imagens e câmeras. Os eventos podem ser registrados para uma pessoa consultar depois. O sistema não identifica indivíduos nem determina se uma movimentação é suspeita.')
    a,b,c=st.columns(3)
    with a: st.markdown('<div class="farm-card"><span class="farm-tag">01 · CAPTURA</span><h3>Fotografia, vídeo ou câmera</h3><p>O programa recebe imagens de arquivo, gravações ou uma webcam/câmera RTSP conectada à rede local.</p></div>',unsafe_allow_html=True)
    with b: st.markdown('<div class="farm-card"><span class="farm-tag">02 · DETECÇÃO</span><h3>Rede neural YOLO</h3><p>A IA procura objetos, estima caixas delimitadoras e atribui um índice de confiança a cada detecção.</p></div>',unsafe_allow_html=True)
    with c: st.markdown('<div class="farm-card"><span class="farm-tag">03 · DECISÃO HUMANA</span><h3>Eventos e histórico</h3><p>Ao detectar um objeto dentro da região escolhida, o sistema pode salvar imagem e registro local para análise.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="farm-section">FLUXO DE OPERAÇÃO</div>',unsafe_allow_html=True)
    x,y,z,w=st.columns(4)
    for col,title,desc in zip([x,y,z,w],['📸 Capturar','🧠 Analisar','🟩 Filtrar','📋 Registrar'],['Imagem da câmera','YOLO + confiança','Zona monitorada','SQLite + imagem']):
        with col: st.markdown(f'<div class="farm-flow">{title}<div class="farm-caption">{desc}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="farm-section">COMECE EM TRÊS PASSOS</div>',unsafe_allow_html=True)
    st.markdown('''**1.** Na barra lateral, deixe selecionado **YOLO11n COCO** para testar sem um treinamento seu. No primeiro uso os pesos podem precisar de Internet.

**2.** Vá à aba **Fotografias**, envie uma imagem JPG ou PNG e clique em **Analisar imagem**. Veja as caixas, a confiança e a região monitorada.

**3.** Consulte a aba **Histórico** para verificar o que foi registrado. Explore também vídeos ou câmera ao vivo quando disponíveis.''')
    st.markdown('<div class="farm-warning"><b>Interpretação responsável:</b> detectar uma pessoa não equivale a reconhecer uma invasão. As duas classes deste projeto não incluem explicitamente trator, motocicleta, animal ou identidade de funcionários.</div>',unsafe_allow_html=True)
    st.markdown('### O que os termos significam?')
    with st.expander('O que é YOLO e o que significa detecção?',expanded=True):
        st.write('**YOLO (You Only Look Once)** é uma família de modelos de detecção de objetos. O algoritmo processa a imagem e estima, para cada objeto reconhecido, sua **classe**, uma **bounding box** (retângulo da posição) e uma **confiança**. Detectar é diferente de classificar a imagem inteira.')
    with st.expander('O que é confiança mínima?'):
        st.write('É o limiar numérico usado para filtrar previsões. Por exemplo, 0,45 significa que previsões com escore inferior a 0,45 são descartadas. O escore **não deve ser interpretado automaticamente como uma probabilidade calibrada** de estar correto.')
    with st.expander('O que é uma zona monitorada?'):
        st.write('É um retângulo configurável sobre a imagem. Neste protótipo, um objeto gera evento quando o **centro de sua bounding box** está dentro da zona e o salvamento está ativo. Isso ajuda a focar no acesso à porteira.')
    with st.expander('Onde os registros são guardados?'):
        st.write('O histórico é armazenado em SQLite dentro da pasta `dados_locais/` no computador que executa o painel. Ele não é enviado a um servidor externo pelo código do projeto. Proteja a máquina, restrinja acessos e estabeleça um prazo de retenção.')



with image_tab:
    st.markdown('<div class="farm-section">MÓDULO 01 · DETECÇÃO POR IMAGEM</div>',unsafe_allow_html=True)
    st.subheader('Analisar uma fotografia')
    st.info('Como interpretar: cada retângulo colorido indica um objeto detectado; a zona delimitada informa quais objetos geram registros. Nenhuma classe representa uma pessoa identificada.')
    st.write('Envie uma imagem de porteira, estacionamento ou acesso rural. A detecção real é executada pelo modelo selecionado.')
    uploaded=st.file_uploader('Imagem JPG ou PNG',type=['jpg','jpeg','png'],key='farm_photo')
    if uploaded is not None and st.button('Analisar imagem',type='primary'):
        try:
            original=image_from_bytes(uploaded.getvalue())
            result=process_frame(original,model(),cfg('imagem'),store,EventThrottle(0))
            ok,jpg=cv2.imencode('.jpg',result.annotated)
            if not ok:raise OSError('Falha ao codificar imagem')
            st.session_state['image_last']=(uploaded.name,jpg.tobytes(),len(result.detections),len(result.zone_detections),result.recorded,result.inference_ms,mode)
        except Exception as exc:
            st.error(f'Erro no processamento: {exc}')
    if st.session_state.get('image_last'):
        name,jpg,n,inside,events,ms,last_mode=st.session_state['image_last']
        st.image(jpg,caption=f'Modelo utilizado: {last_mode} | Imagem: {name}',use_container_width=True)
        show_metrics(range(n),range(inside),events,ms)
        st.download_button('Baixar fotografia com detecções',jpg,file_name='farmtech_deteccoes.jpg',mime='image/jpeg')

with video_tab:
    st.markdown('<div class="farm-section">MÓDULO 02 · VÍDEOS GRAVADOS</div>',unsafe_allow_html=True)
    st.subheader('Analisar gravação de câmera')
    st.info('Vídeos são analisados em quadros amostrados para reduzir o custo computacional. Os totais são detecções de quadros, não contagem de pessoas ou veículos únicos.')
    st.write('Processamento offline de um trecho limitado. Ao terminar, é possível baixar o vídeo anotado e consultar os eventos.')
    video=st.file_uploader('Vídeo MP4, MOV ou AVI (até 100 MB)',type=['mp4','mov','avi'],key='farm_video')
    c1,c2=st.columns(2)
    with c1: every_n=st.number_input('Processar um quadro a cada N',min_value=1,max_value=30,value=5)
    with c2: max_frames=st.number_input('Máximo de quadros analisados',min_value=10,max_value=600,value=120,step=10)
    if video is not None and st.button('Processar vídeo',type='primary'):
        if video.size>100*1024*1024:
            st.error('Vídeo superior a 100 MB')
        else:
            infile=None
            try:
                suffix=Path(video.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
                    infile=Path(tmp.name);tmp.write(video.getvalue())
                outdir=DATA_LOCAL/'videos'
                outfile=outdir/f'farmtech_{uuid4().hex[:12]}.avi'
                progress=st.progress(0,text='Processando vídeo localmente...')
                def update_progress(count):
                    if count%5==0:progress.progress(min(1.,count/int(max_frames)),text=f'Quadros processados: {count}/{max_frames}')
                stats=analyze_video(infile,outfile,model(),cfg('video'),store,
                                    every_n=int(every_n),max_frames=int(max_frames),
                                    cooldown=float(cooldown),progress=update_progress)
                st.session_state['video_last']=(str(stats.output_file),stats.preview_jpeg,
                    stats.frames_analyzed,stats.detections_total,stats.events_saved,stats.mean_inference_ms,mode)
                progress.progress(1.,text=f'Análise finalizada: {stats.frames_analyzed} quadros')
            except Exception as exc:
                st.error(f'Não foi possível processar o vídeo: {exc}')
            finally:
                if infile is not None:infile.unlink(missing_ok=True)
    if st.session_state.get('video_last'):
        path,thumb,frames,n_dets,events,ms,last_mode=st.session_state['video_last']
        st.image(thumb,caption='Primeiro quadro analisado',use_container_width=True)
        a,b,c=st.columns(3)
        a.metric('Quadros analisados',frames);b.metric('Detecções acumuladas',n_dets);c.metric('Eventos registrados',events)
        st.caption(f'Modelo utilizado: {last_mode} | Tempo médio de inferência: {ms:.1f} ms por quadro. AVI MJPG sem áudio.')
        if Path(path).is_file():
            with Path(path).open('rb') as f: st.download_button('Baixar vídeo anotado (AVI)',f,file_name='farmtech_monitoramento.avi',mime='video/x-msvideo')

with live_tab:
    st.markdown('<div class="farm-section">MÓDULO 03 · MONITORAMENTO EM REDE LOCAL</div>',unsafe_allow_html=True)
    st.subheader('Monitoramento de webcam ou câmera IP')
    st.info('Webcam: conectada ao mesmo computador. RTSP: fluxo autorizado de uma câmera acessível pela rede. Não exponha credenciais ou a interface do sistema na Internet.')
    st.warning('Funciona no computador que executa o aplicativo. A câmera deve estar acessível a esse computador. A detecção é assistiva e requer supervisão humana.')
    source_type=st.radio('Fonte de vídeo',['Webcam USB / integrada','Câmera de rede RTSP'],horizontal=True)
    if source_type=='Webcam USB / integrada':
        webcam_id=st.number_input('Índice da câmera',min_value=0,max_value=5,value=0)
        source=int(webcam_id)
    else:
        rtsp=st.text_input('URL RTSP da câmera (será usada somente nesta sessão)',type='password',placeholder='rtsp://usuario:senha@192.168.1.10:554/stream')
        source=rtsp.strip()
    a,b=st.columns(2)
    with a: start=st.button('▶ Iniciar monitoramento',type='primary')
    with b: stop=st.button('■ Parar câmera')
    if stop:
        cap=st.session_state.pop('live_cap',None)
        if cap is not None:cap.release()
        st.session_state['live_running']=False
    if start:
        if source_type.startswith('Câmera') and not (str(source).startswith('rtsp://') or str(source).startswith('rtsps://')):
            st.error('Endereço RTSP inválido')
        else:
            old=st.session_state.pop('live_cap',None)
            if old is not None:old.release()
            cap=cv2.VideoCapture(source)
            if not cap.isOpened():
                cap.release();st.error('Não foi possível conectar à câmera. Confira índice, endereço, rede ou credenciais.')
            else:
                st.session_state['live_cap']=cap
                st.session_state['live_running']=True
                st.session_state['live_throttle']=EventThrottle(float(cooldown))
                st.session_state['live_count']=0
    @st.fragment(run_every='1s')
    def render_live():
        if not st.session_state.get('live_running'):
            st.info('Câmera parada. Clique em iniciar para visualizar detecções ao vivo.')
            return
        cap=st.session_state.get('live_cap')
        if cap is None or not cap.isOpened():
            st.session_state['live_running']=False
            st.error('Conexão com a câmera interrompida')
            return
        ok,frame=cap.read()
        if not ok:
            cap.release();st.session_state['live_running']=False
            st.warning('Fluxo finalizado ou desconectado. Inicie novamente para reconectar.')
            return
        try:
            result=process_frame(frame,model(),cfg('camera_ao_vivo'),store,st.session_state['live_throttle'])
            st.session_state['live_count']+=1
            st.image(cv2.cvtColor(result.annotated,cv2.COLOR_BGR2RGB),channels='RGB',use_container_width=True)
            show_metrics(result.detections,result.zone_detections,result.recorded,result.inference_ms)
            st.caption(f"Quadros efetivamente analisados nesta sessão: {st.session_state['live_count']} | Atualização solicitada a cada 1 segundo (depende do hardware).")
        except Exception as e:
            cap.release();st.session_state['live_running']=False
            st.error(f'Falha no processamento do fluxo: {e}')
    render_live()

with history_tab:
    st.markdown('<div class="farm-section">MÓDULO 04 · RASTREABILIDADE LOCAL</div>',unsafe_allow_html=True)
    st.subheader('Eventos armazenados no computador')
    st.caption(f'Banco SQLite local: {store.db.relative_to(ROOT)} | Registros não são transmitidos para servidores do projeto.')
    col1,col2,col3=st.columns(3)
    total=store.recent(limit=5000);cts=store.counts()
    col1.metric('Eventos recentes',len(total));col2.metric('Pessoas',cts['person']);col3.metric('Carros',cts['car'])
    if total:
        df=pd.DataFrame(total)
        st.dataframe(df[['occurred_utc','camera','object_class','confidence','model_mode','source_kind']],use_container_width=True,hide_index=True)
        for item in total[:1]:
            img=(store.root/item['snapshot']).resolve()
            if img.is_file() and store.root in img.parents:
                st.image(str(img),caption='Registro mais recente, sujeito à revisão humana',use_container_width=True)
        st.download_button('Exportar histórico CSV',store.csv_bytes(),file_name='eventos_farmtech.csv',mime='text/csv')
    else:st.info('Ainda não foram registrados eventos. Analise uma imagem ou inicie uma câmera com o salvamento habilitado.')

with guide_tab:
    st.markdown('<div class="farm-section">LABORATÓRIO ACADÊMICO · FIAP FASE 6</div>',unsafe_allow_html=True)
    st.subheader('Entenda a ciência e acompanhe a entrega')
    st.write('Esta página explica como os experimentos acadêmicos diferem do monitoramento rural. Os resultados abaixo são apresentados **somente se já existirem arquivos de métricas reais**.')
    dataset_ready=(ROOT/'dataset'/'manifest.csv').exists()
    custom_ready=(ROOT/'models'/'best.pt').exists()
    result_paths=[ROOT/'results'/name for name in ['yolo_custom_30_metrics.json','yolo_custom_60_metrics.json','yolo_standard_metrics.json','cnn_metrics.json']]
    m1,m2,m3=st.columns(3)
    m1.metric('Dataset de imagens', 'Encontrado' if dataset_ready else 'Pendente')
    m2.metric('Métricas experimentais',f'{sum(p.exists() for p in result_paths)}/4')
    m3.metric('Checkpoint customizado','Encontrado' if custom_ready else 'Pendente')
    st.markdown('### Como os modelos são comparados?')
    st.markdown('''| Modelo | O que faz | O que o aluno demonstra |
|---|---|---|
| YOLOv5 adaptada, 30 épocas | Detecta objetos | Treinamento e avaliação |
| YOLOv5 adaptada, 60 épocas | Detecta objetos | Efeito da duração do treino |
| YOLOv5 padrão (COCO) | Detecta sem fine-tuning | Benchmark com pesos públicos |
| CNN do zero | Classifica a imagem inteira | Comparação de abordagem e custo |''')
    st.caption('mAP50 dos detectores NÃO equivale à accuracy da CNN; as métricas devem ser interpretadas por tarefa.')
    data=[]
    for path in result_paths:
        if path.is_file():
            try:
                import json
                obj=json.loads(path.read_text(encoding='utf-8'))
                data.append({'Experimento':path.stem,'Precision':obj.get('precision'),'Recall':obj.get('recall'),'F1':obj.get('f1'),'mAP50':obj.get('map50'),'Accuracy':obj.get('accuracy'),'Inferência (ms)':obj.get('mean_inference_ms')})
            except (OSError,ValueError) as exc:st.warning(f'Arquivo de resultados inválido: {path.name}: {exc}')
    if data:
        st.markdown('### Resultados importados do treinamento')
        st.dataframe(pd.DataFrame(data),use_container_width=True,hide_index=True)
    else: st.info('Nenhum resultado de treinamento encontrado. Execute os notebooks no Colab e importe as métricas reais. Este painel não cria números de demonstração.')
    pdf=ROOT/'docs'/'GUIA_COMPLETO_FARMTECH_FIAP.pdf'
    if pdf.exists():
        st.download_button('📄 Baixar guia completo em PDF',pdf.read_bytes(),file_name=pdf.name,mime='application/pdf')
    st.markdown('### Etapas acadêmicas obrigatórias')
    for title,body in [
        ('1. Dataset e separação','40 fotografias de `person` + 40 de `car`, divididas em treino 32/32, validação 4/4, teste 4/4.'),
        ('2. Make Sense','Rotule pessoalmente as 64 imagens de treino, exporte YOLO, salve no Drive e audite as boxes.'),
        ('3. Experimentos YOLO','Treine a YOLOv5 em 30 e 60 épocas; avalie a YOLO padrão nas mesmas imagens e registre gráficos/prints.'),
        ('4. CNN e comparação','Treine CNN do zero, compare acertos e erros, tempos, facilidade de integração, pontos fortes e limitações.'),
        ('5. Entrega no portal','Notebook com todas as células executadas; GitHub público; README com Colab e YouTube; vídeo não listado de até 5 minutos.')]:
        with st.expander(title): st.write(body)
    st.warning('Nota máxima não é garantida por código, testes locais ou visual do aplicativo. O professor avalia também execução real, justificativas, evidências, apresentação, publicação e prazo.')
    st.subheader('Manual de operação, limitações e segurança')
    st.markdown('''
**Fluxo:** imagem ou câmera → YOLO → caixas de objetos → filtro de zona → registro local → painel.

**Rótulos:** `person` (pessoa) e `car` (automóvel). Caminhões, tratores e motocicletas não são classes explicitamente configuradas; um veículo agrícola pode não ser detectado corretamente.

**Demonstração:** a YOLO11n COCO permite detectar objetos reais mesmo antes de treinar o notebook da FIAP. Ela não deve ser apresentada como a YOLOv5 customizada do trabalho.

**Modelo FIAP:** depois do Colab, copie `best.pt` para `models/`, prepare o repositório YOLOv5 v7.0 em `third_party/yolov5/` e selecione a opção customizada.

**Segurança:** as detecções não identificam pessoas, não autorizam ou negam acesso e não comprovam invasão. Não conecte o sistema diretamente a alarmes críticos nem a motores de portões sem validação específica.

**Privacidade:** o histórico fica em `dados_locais/`. Mantenha a máquina protegida, defina política de retenção e respeite as obrigações de privacidade aplicáveis.

**Acesso pelo celular:** por padrão, este painel funciona apenas no próprio computador. Acesso remoto seguro exige configuração adicional de rede, autenticação e HTTPS. Não publique o Streamlit diretamente na Internet.
''')
    st.markdown('Documentação adicional: `docs/GUIA_FAZENDA_APLICACAO.md` e `COMECE_AQUI.md`.')
