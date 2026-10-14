"""Read-only local event dashboard. Start: streamlit run operacao_rural/dashboard.py"""
from pathlib import Path
import pandas as pd
import streamlit as st
st.set_page_config(page_title='VisionGuard Rural',page_icon='🌾',layout='wide')
st.title('FarmTech VisionGuard | Painel Rural')
st.caption('Protótipo demonstrativo. Detecção não comprova invasão nem autoriza ação automática.')
path=Path(st.sidebar.text_input('Arquivo CSV de eventos','monitoramento_local/events.csv'))
if not path.is_file():
    st.info('Nenhum registro encontrado. Execute o monitor com pesos reais best.pt.')
    st.stop()
try:
    df=pd.read_csv(path)
    needed={'timestamp_utc','camera','class','confidence','zone_match','snapshot'}
    if not needed.issubset(df.columns): raise ValueError('Colunas obrigatórias ausentes')
    st.metric('Eventos registrados',len(df))
    a,b=st.columns(2)
    a.metric('Pessoas',int((df['class']=='person').sum()))
    b.metric('Veículos',int((df['class']=='car').sum()))
    st.dataframe(df.sort_values('timestamp_utc',ascending=False),use_container_width=True)
    st.subheader('Última imagem do evento')
    for item in df.sort_values('timestamp_utc',ascending=False)['snapshot']:
        if isinstance(item,str) and Path(item).is_file():
            st.image(item,caption='Registro de detecção (revisão humana necessária)',width=760)
            break
except Exception as exc:
    st.error(f'Falha ao abrir registros: {exc}')
