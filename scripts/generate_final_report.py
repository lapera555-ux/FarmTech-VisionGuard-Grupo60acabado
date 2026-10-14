from __future__ import annotations
from pathlib import Path
import argparse, json

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    res = root/"results"

    files = {
        "YOLO 30": res/"yolo_custom_30_metrics.json",
        "YOLO 60": res/"yolo_custom_60_metrics.json",
        "YOLO padrão": res/"yolo_standard_metrics.json",
        "CNN": res/"cnn_metrics.json",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Resultados ausentes: " + ", ".join(missing))

    data = {k:load(v) for k,v in files.items()}
    out = root/"reports/Relatorio_Final_Resultados_Reais.pdf"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], textColor=colors.HexColor("#16324F")))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9.5, leading=14))

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.6*cm, rightMargin=1.6*cm)
    story = [
        Paragraph("FarmTech VisionGuard - Resultados Reais", styles["Title"]),
        Paragraph("Pedro Gomes - RM 571466 - Grupo 60", styles["Bodyx"]),
        Spacer(1, .4*cm),
        Paragraph("Resumo", styles["H1x"]),
        Paragraph(
            "Este relatório é gerado automaticamente a partir dos arquivos JSON produzidos "
            "pela execução real do notebook. Nenhuma métrica é preenchida manualmente.",
            styles["Bodyx"]
        ),
    ]

    rows = [["Modelo","Precision","Recall","F1","mAP50/Accuracy","Treino (s)","Inferência (ms)"]]
    for name, d in data.items():
        metric = d.get("map50", d.get("accuracy", ""))
        rows.append([
            name,
            f"{d.get('precision',''):.4f}" if isinstance(d.get("precision"),(int,float)) else "",
            f"{d.get('recall',''):.4f}" if isinstance(d.get("recall"),(int,float)) else "",
            f"{d.get('f1',''):.4f}" if isinstance(d.get("f1"),(int,float)) else "",
            f"{metric:.4f}" if isinstance(metric,(int,float)) else "",
            f"{d.get('training_seconds',0):.1f}",
            f"{d.get('mean_inference_ms',0):.2f}",
        ])

    table = Table(rows, colWidths=[3.1*cm,2.2*cm,2.2*cm,2.0*cm,2.8*cm,2.1*cm,2.4*cm])
    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#16324F")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#BBBBBB")),
        ("FONTSIZE",(0,0),(-1,-1),7.5),
        ("ALIGN",(1,1),(-1,-1),"CENTER"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F4F7F9")]),
    ]))
    story += [table, Spacer(1,.5*cm)]

    analysis = res/"analise_critica.md"
    if analysis.exists():
        text = analysis.read_text(encoding="utf-8").replace("#","")
        story += [
            Paragraph("Análise crítica", styles["H1x"]),
            Paragraph(text.replace("\n","<br/>"), styles["Bodyx"])
        ]

    mode = (root/"dataset/LABEL_MODE.txt")
    if mode.exists():
        story += [
            Paragraph("Proveniência das anotações", styles["H1x"]),
            Paragraph(mode.read_text(encoding="utf-8").replace("\n","<br/>"), styles["Bodyx"])
        ]

    doc.build(story)
    print(out)

if __name__ == "__main__":
    main()
