"""Preenche links reais e gera PDF para o portal FIAP (não publica sozinho)."""
from __future__ import annotations
import argparse
from pathlib import Path
from urllib.parse import urlparse
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

TOKENS={'colab':'INSIRA_URL_PUBLICA_DO_COLAB','video':'INSIRA_URL_DO_VIDEO','github':'INSIRA_URL_DO_GITHUB'}
DOMAIN={'colab':{'colab.research.google.com'},'video':{'youtube.com','www.youtube.com','youtu.be'},'github':{'github.com','www.github.com'}}

def validate_url(kind: str, url: str) -> str:
    value=url.strip()
    parsed=urlparse(value)
    if parsed.scheme!='https' or parsed.hostname not in DOMAIN[kind] or not parsed.path.strip('/'):
        raise ValueError(f'Link {kind} inválido. Use HTTPS com endereço específico em {sorted(DOMAIN[kind])}.')
    return value

def update_readme(text: str, github: str, colab: str, video: str) -> str:
    values={'github':validate_url('github',github),'colab':validate_url('colab',colab),'video':validate_url('video',video)}
    for kind,token in TOKENS.items():
        if token not in text and values[kind] not in text:
            raise ValueError(f'Placeholder {token} não encontrado no README. Verifique se já foi finalizado.')
        text=text.replace(token,values[kind])
    return text

def build_portal_pdf(root: Path, rm: str, github: str, colab: str, video: str):
    """Gera um documento de links com identidade visual, somente após receber URLs reais."""
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.colors import HexColor
    from reportlab.lib.utils import simpleSplit
    from reportlab.lib.pagesizes import A4
    out=root/'reports'/'Entrega_Portal_FIAP_FINAL.pdf'
    out.parent.mkdir(parents=True,exist_ok=True)
    canvas=Canvas(str(out),pagesize=A4)
    canvas.setTitle('FarmTech VisionGuard | Entrega FIAP Fase 6')
    canvas.setAuthor('Pedro Gomes | RM 571466 | Grupo 60')
    w,h=A4
    dark=HexColor('#0a332b');green=HexColor('#087961');mint=HexColor('#edf7f0');line=HexColor('#d8e8de')
    canvas.setFillColor(dark);canvas.rect(0,h-235,w,235,fill=1,stroke=0)
    canvas.setFillColor(HexColor('#145848'));canvas.circle(w-35,h-36,150,fill=1,stroke=0)
    canvas.setFillColor(HexColor('#c6df9e'));canvas.setFont('Helvetica-Bold',10)
    canvas.drawString(46,h-59,'FIAP  |  INTELIGÊNCIA ARTIFICIAL  |  FASE 6')
    canvas.setFillColor(HexColor('#ffffff'));canvas.setFont('Helvetica-Bold',30)
    canvas.drawString(46,h-115,'FarmTech VisionGuard')
    canvas.setFont('Helvetica-Bold',30);canvas.drawString(46,h-154,'Rural')
    canvas.setFillColor(HexColor('#d0ede0'));canvas.setFont('Helvetica',11)
    canvas.drawString(46,h-187,'Documento de acesso à entrega acadêmica')
    canvas.setFillColor(dark);canvas.setFont('Helvetica-Bold',12);canvas.drawString(46,h-275,'PEDRO GOMES')
    canvas.setFillColor(HexColor('#496a59'));canvas.setFont('Helvetica',11)
    canvas.drawString(46,h-298,f'RM {rm}  |  Grupo 60')
    items=[('01','REPOSITÓRIO GITHUB',github),('02','NOTEBOOK GOOGLE COLAB',colab),('03','VÍDEO YOUTUBE NÃO LISTADO',video)]
    y=h-355
    for num,label,link in items:
        canvas.setFillColor(mint);canvas.roundRect(44,y-70,w-88,80,10,fill=1,stroke=0)
        canvas.setFillColor(green);canvas.setFont('Helvetica-Bold',13);canvas.drawString(60,y-16,num)
        canvas.setFillColor(dark);canvas.setFont('Helvetica-Bold',11);canvas.drawString(103,y-16,label)
        # Use smaller font when necessary to keep the whole clickable URL inside the card.
        available=w-164
        font_size=9.3
        while canvas.stringWidth(link,'Helvetica',font_size)>available and font_size>6.2:font_size-=0.2
        canvas.setFont('Helvetica',font_size);canvas.setFillColor(green)
        if canvas.stringWidth(link,'Helvetica',font_size)<=available:
            canvas.drawString(103,y-41,link)
        else:
            pieces=simpleSplit(link,'Helvetica',font_size,available)[:2]
            for k,fragment in enumerate(pieces):canvas.drawString(103,y-38-13*k,fragment)
        canvas.linkURL(link,(100,y-55,w-60,y-17),relative=0)
        y-=97
    canvas.setFillColor(HexColor('#fff6e8'));canvas.roundRect(44,103,w-88,80,10,fill=1,stroke=0)
    canvas.setFillColor(HexColor('#755423'));canvas.setFont('Helvetica-Bold',10);canvas.drawString(57,157,'ATENÇÃO ANTES DO ENVIO')
    canvas.setFont('Helvetica',9)
    for i,t in enumerate(['Confirme que o Colab contém células executadas e métricas reais.',
                          'Teste os 3 links em janela anônima, revise o vídeo e o prazo.',
                          'Não faça novos commits após o horário de entrega da FIAP.']):
        canvas.drawString(57,140-i*15,t)
    canvas.setStrokeColor(line);canvas.line(44,68,w-44,68)
    canvas.setFillColor(HexColor('#567568'));canvas.setFont('Helvetica',8)
    canvas.drawString(44,52,'FarmTech VisionGuard Rural | Entrega 1 + Entrega 2')
    canvas.drawRightString(w-44,52,'Documento gerado com URLs informadas pelo aluno')
    canvas.save()
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rm',required=True)
    p.add_argument('--colab',required=True)
    p.add_argument('--video',required=True)
    p.add_argument('--github',required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if args.rm!='571466':p.error('RM precisa ser 571466 para este projeto')
    try:
        for kind in TOKENS:validate_url(kind,getattr(args,kind))
        path=root/'README.md'
        updated=update_readme(path.read_text(encoding='utf-8'),args.github,args.colab,args.video)
        # Preparar PDF antes de escrever README; evita README alterado se PDF falhar.
        pdf=build_portal_pdf(root,args.rm,args.github,args.colab,args.video)
        path.write_text(updated,encoding='utf-8')
    except (ValueError,OSError) as exc:
        p.error(str(exc))
    print('Links reais incorporados no README. PDF gerado:',pdf)
    print('ATENÇÃO: ainda é necessário conferir link, notebook, treino e prazo manualmente.')

if __name__=='__main__':main()
