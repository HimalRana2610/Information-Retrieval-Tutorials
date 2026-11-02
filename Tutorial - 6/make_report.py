"""Generate U23AI053_Tutorial_6.pdf from pipeline outputs and plots."""

from __future__ import annotations

import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "U23AI053_Tutorial_6.pdf")
RUN = os.path.join(HERE, "run_output.txt")


def styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(
        name="TitleCenter", parent=s["Title"], alignment=TA_CENTER, fontSize=16,
        spaceAfter=6))
    s.add(ParagraphStyle(
        name="SubCenter", parent=s["Normal"], alignment=TA_CENTER, fontSize=11,
        textColor=colors.HexColor("#333333"), spaceAfter=18))
    s.add(ParagraphStyle(
        name="H", parent=s["Heading2"], fontSize=13, spaceBefore=14,
        spaceAfter=6, textColor=colors.HexColor("#1a1a1a")))
    s.add(ParagraphStyle(
        name="BodyJust", parent=s["Normal"], alignment=TA_JUSTIFY, fontSize=10,
        leading=14, spaceAfter=8))
    s.add(ParagraphStyle(
        name="Mono", parent=s["Code"], fontName="Courier", fontSize=7.2,
        leading=9.2, spaceAfter=6))
    return s


def load_run_sections():
    text = open(RUN, encoding="utf-8", errors="replace").read()
    # Strip PowerShell noise if present
    text = text.replace("\r\n", "\n")
    parts = re.split(r"\n={10,}\n", text)
    # parts alternate: before, title, body, title, body...
    sections = {}
    i = 1
    while i + 1 < len(parts):
        title = parts[i].strip()
        body = parts[i + 1].strip()
        sections[title] = body
        i += 2
    return text, sections


def mono_block(text, style, max_lines=40):
    lines = text.strip().splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]
    return Preformatted("\n".join(lines), style)


def main():
    s = styles()
    full, sections = load_run_sections()

    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
    )
    story = []

    story.append(Paragraph(
        "Tutorial 6: AI-Powered Fake News Detection<br/>"
        "and Intelligent News Retrieval System", s["TitleCenter"]))
    story.append(Paragraph(
        "Information Retrieval · SVNIT · Roll No. U23AI053", s["SubCenter"]))

    story.append(Paragraph("1. Objective", s["H"]))
    story.append(Paragraph(
        "Build a system that collects news articles, flags potentially "
        "misleading content with a confidence score, groups related stories, "
        "ranks credible and relevant results, and supports semantic search. "
        "Model predictions alone cannot establish whether a news story is true; "
        "outputs are triage signals for human review.", s["BodyJust"]))

    story.append(Paragraph("2. Syllabus coverage", s["H"]))
    data = [
        ["Syllabus topic", "Implementation"],
        ["Word embeddings", "GloVe SIF article vectors"],
        ["Clustering-based retrieval", "K-means on embeddings (k=6)"],
        ["Topic modeling", "LDA + NMF (5 topics)"],
        ["Dimensionality reduction", "PCA and t-SNE plots"],
        ["Anomaly detection", "Low max pairwise cosine"],
        ["Classification-based retrieval", "NB / LogReg / RF / Style features"],
        ["Learning to Rank", "Pointwise GradientBoosting on IR features"],
        ["Feature engineering", "Sensational lexicon, source, TLD, anonymity"],
        ["Ensemble methods", "Weighted soft-voting of 4 classifiers"],
        ["Neural Information Retrieval", "Semantic cosine over GloVe SIF"],
    ]
    t = Table(data, colWidths=[2.3 * inch, 4.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f3f6fa")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)

    story.append(Paragraph("3. Dataset", s["H"]))
    story.append(Paragraph(
        "32 curated news articles across Technology, Health, Politics, Climate, "
        "Security and Business, including related-story pairs (Hub/Wycoff, Oregon "
        "crash lawsuit, California climate case) and two unrelated anomalies "
        "(gardening, sports). 23 labelled credible, 9 potentially misleading.",
        s["BodyJust"]))

    story.append(Paragraph("4. Classification results (held-out)", s["H"]))
    clf = sections.get(
        "Step 2: Classification (credible vs potentially misleading)", "")
    # Keep the metrics table portion
    m = re.search(
        r"(Model\s+Acc.*?)(?:Ensemble predictions|\Z)", clf, re.S)
    story.append(mono_block(m.group(1) if m else clf, s["Mono"], 20))
    story.append(Paragraph(
        "Random Forest (TF-IDF SVD + style features) reached F1 = 1.00 on the "
        "held-out split. The weighted ensemble (NB + LogReg + RF + Style-LogReg) "
        "achieved F1 = 0.857 with full recall of misleading articles "
        "(1 false positive).", s["BodyJust"]))

    story.append(Paragraph("5. Demo: viral vaccine claim", s["H"]))
    demo = sections.get("Step 3b: Demo analysis of a viral claim", "")
    story.append(mono_block(demo, s["Mono"], 12))

    story.append(PageBreak())
    story.append(Paragraph("6. Clustering, topics and anomalies", s["H"]))
    for key in ("Step 5: Clustering-based retrieval (related stories)",
                "Step 6: Topic modeling (LDA + NMF)",
                "Step 7: Anomaly detection (articles far from all others)"):
        body = sections.get(key, "")
        story.append(Paragraph(key, s["Heading3"] if "Heading3" in s else s["H"]))
        story.append(mono_block(body, s["Mono"], 22))

    story.append(Paragraph("7. Visualisations", s["H"]))
    for name, caption in [
        ("classifier_comparison.png", "Classifier accuracy / F1 comparison"),
        ("pca_articles.png", "PCA of GloVe SIF embeddings (X = potentially misleading)"),
        ("tsne_articles.png", "t-SNE of article embeddings"),
    ]:
        path = os.path.join(HERE, name)
        if os.path.exists(path):
            story.append(Paragraph(caption, s["BodyJust"]))
            story.append(Image(path, width=6.2 * inch, height=4.0 * inch))
            story.append(Spacer(1, 8))

    story.append(PageBreak())
    story.append(Paragraph("8. Retrieval and Learning to Rank", s["H"]))
    search = sections.get(
        'Step 10: Search — "electric vehicle battery fires and safety"', "")
    story.append(mono_block(search, s["Mono"], 28))
    evals = sections.get("Step 11: Retrieval evaluation (top 5)", "")
    # Keep summary table at end
    story.append(Paragraph("Evaluation summary", s["Heading3"] if "Heading3" in s else s["H"]))
    m = re.search(r"(Method\s+P@5.*)", evals, re.S)
    story.append(mono_block(m.group(1) if m else evals, s["Mono"], 12))
    story.append(Paragraph(
        "Learning to Rank obtained the best MAP (0.943), ahead of semantic "
        "search (0.859) and keyword TF-IDF (0.843), by combining semantic "
        "similarity with source credibility and a low misleading probability.",
        s["BodyJust"]))

    story.append(Paragraph("9. How to run", s["H"]))
    story.append(mono_block(
        "python main.py                 # full report + plots\n"
        "python main.py --interactive   # query / analyse loop\n"
        "streamlit run app.py           # interactive dashboard",
        s["Mono"], 10))

    story.append(Paragraph("10. Conclusion", s["H"]))
    story.append(Paragraph(
        "The system covers supervised classification with engineered features "
        "and an ensemble, neural/semantic retrieval with GloVe embeddings, "
        "clustering of related stories, topic discovery, anomaly detection, "
        "dimensionality-reduction visualisations, and Learning-to-Rank. "
        "All credibility outputs are phrased as <i>potentially misleading</i> "
        "rather than definitive claims of falsity.", s["BodyJust"]))

    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
