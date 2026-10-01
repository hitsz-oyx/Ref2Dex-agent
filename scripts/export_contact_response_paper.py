#!/usr/bin/env python3
"""Render the actual LaTeX draft as a PDF review copy without installing TeX."""
from __future__ import annotations

import hashlib
from html import escape
import json
from pathlib import Path
import re
import shutil

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'src/task/CmResidual/research/contact_response/output'
PHYSICAL = BASE / 'P-20261001-contact-response-resolution-r1/analysis-v2'
LEARNING = BASE / 'P-20261001-differential-response-learning-r1'
PAPER = ROOT / 'paper'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(text):
    text = re.sub(r'\\cite\{([^}]+)\}', lambda m: '[' + m[1] + ']', text)
    text = re.sub(r'\\(?:texttt|textbf|url|emph)\{([^}]+)\}', r'\1', text)
    text = text.replace('\\%', '%').replace('\\ge', '>=').replace('\\le', '<=')
    text = text.replace('\\tau', 'tau').replace('\\qquad', ' ').replace('\\quad', ' ')
    text = text.replace('\\in', ' in ').replace('\\|', '|').replace('\\{', '{').replace('\\}', '}')
    text = text.replace('$', '').replace('~', ' ').replace('--', '-').replace('\\noindent', '')
    text = re.sub(r'\\(?:sqrt|sum|text|left|right)', '', text)
    text = re.sub(r'\\[A-Za-z]+', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def main():
    destination = PAPER / 'manuscript.pdf'
    if destination.exists():
        raise FileExistsError('retain the existing review copy; export a new version explicitly')
    tex = PAPER / 'manuscript.tex'
    source = tex.read_text()
    physical = json.loads((PHYSICAL / 'analysis.json').read_text())
    learned = json.loads((LEARNING / 'results.json').read_text())
    # The tracked LaTeX table must match independently archived measurements.
    for row in physical['pooled']:
        for key in ('signal_rms_mm', 'noise_rms_mm'):
            if f"{row[key]:.3f}" not in source:
                raise ValueError('physical table differs from recorded JSON')
    for method in ('factual', 'differential', 'no_pulse'):
        for horizon in ('5', '10'):
            expected = f"{learned['averaged'][method][horizon]['contrast_rmse_mm']:.3f}"
            if expected not in source:
                raise ValueError('learned table differs from recorded JSON')
    figures = PAPER / 'figures'
    figures.mkdir(exist_ok=True)
    target_figure = figures / 'contact_response.pdf'
    if target_figure.exists():
        if sha(target_figure) != sha(PHYSICAL / 'contact_response.pdf'):
            raise ValueError('different existing figure')
    else:
        shutil.copyfile(PHYSICAL / 'contact_response.pdf', target_figure)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='PaperBody', fontName='Times-Roman', fontSize=10.3,
                              leading=13.2, spaceAfter=7))
    styles.add(ParagraphStyle(name='PaperTitle', fontName='Times-Bold', fontSize=17,
                              leading=20, spaceAfter=12, alignment=1))
    styles.add(ParagraphStyle(name='PaperCaption', fontName='Times-Italic', fontSize=9,
                              leading=11, spaceAfter=10))
    story = []
    title = re.search(r'\\title\{(.*?)\}\s*\\author', source, re.S)[1].replace('\\\\', ' ')
    story.append(Paragraph(escape(plain(title)), styles['PaperTitle']))
    story.append(Paragraph('Working manuscript - 1 October 2026', styles['PaperCaption']))
    story.append(Paragraph('<b>Exploratory review copy.</b> Rendered from the LaTeX draft using ReportLab. '
                           'The source contains the publication equations; this copy displays equivalent plain-text equations. '
                           'Both pilot gates failed. No policy or journal-readiness claim is made.', styles['PaperBody']))
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', source, re.S)[1]
    story.append(Paragraph('Abstract', styles['Heading2']))
    story.append(Paragraph(escape(plain(abstract)), styles['PaperBody']))
    body = source.split('\\end{abstract}', 1)[1].split('\\begin{thebibliography}', 1)[0]
    tokens = re.split(r'(\\section\*?\{[^}]+\}|\\begin\{(?:table|figure|equation)\}.*?\\end\{(?:table|figure|equation)\})', body, flags=re.S)
    section_number = 0
    table_number = 0
    for token in tokens:
        if token.startswith('\\section'):
            heading = re.search(r'\{([^}]+)\}', token)[1]
            if '\\section*' not in token:
                section_number += 1
                heading = f'{section_number}. {heading}'
            story.append(Paragraph(escape(heading), styles['Heading2']))
        elif token.startswith('\\begin{equation}'):
            if 'R_{i,H}' in token:
                equation = 'R(i,H,u) = p(i,tau+H,u) - p(i,tau,u); D(i,H) = R(i,H,+) - R(i,H,-).'
            else:
                equation = 'S(H) = sqrt(mean_i ||D(i,H)||^2); B(H) = sqrt(mean_i ||N(i,H)||^2).'
            story.append(Paragraph(escape(equation), styles['PaperCaption']))
        elif token.startswith('\\begin{figure}'):
            image = Image(str(PHYSICAL / 'contact_response.png'), width=6.4 * inch, height=2.42 * inch)
            story.append(image)
            caption = re.search(r'\\caption\{(.*?)\}', token, re.S)[1]
            story.append(Paragraph('Figure 1. ' + escape(plain(caption)), styles['PaperCaption']))
        elif token.startswith('\\begin{table}'):
            table_number += 1
            if table_number == 1:
                data = [['Steps', 'Seconds', 'Pulse RMS (mm)', 'Repeat RMS (mm)', 'Ratio']]
                data += [[str(r['horizon']), f"{r['seconds']:.3f}", f"{r['signal_rms_mm']:.3f}",
                          f"{r['noise_rms_mm']:.3f}", 'undefined' if r['contrast_ratio'] is None else f"{r['contrast_ratio']:.2f}"]
                         for r in physical['pooled']]
            else:
                data = [['Model', 'Five steps (mm)', 'Ten steps (mm)']]
                for name, key in [('Factual', 'factual'), ('Differential', 'differential'), ('No-pulse input', 'no_pulse')]:
                    data.append([name, *(f"{learned['averaged'][key][str(h)]['contrast_rmse_mm']:.3f}" for h in (5, 10))])
                for name, key in [('Global fit mean', 'global_fit_mean'), ('Per-motion fit mean', 'motion_fit_mean')]:
                    data.append([name, *(f"{learned['baselines'][key][str(h)]['contrast_rmse_mm']:.3f}" for h in (5, 10))])
            table = Table(data, hAlign='CENTER', repeatRows=1)
            table.setStyle(TableStyle([('FONT', (0, 0), (-1, -1), 'Times-Roman', 9),
                                       ('FONT', (0, 0), (-1, 0), 'Times-Bold', 9),
                                       ('LINEABOVE', (0, 0), (-1, 0), .8, colors.black),
                                       ('LINEBELOW', (0, 0), (-1, 0), .5, colors.black),
                                       ('LINEBELOW', (0, -1), (-1, -1), .8, colors.black),
                                       ('TOPPADDING', (0, 0), (-1, -1), 5),
                                       ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                                       ('ALIGN', (1, 0), (-1, -1), 'RIGHT')]))
            story.append(table)
            story.append(Spacer(1, 6))
            caption = re.search(r'\\caption\{(.*?)\}', token, re.S)[1]
            story.append(Paragraph(f'Table {table_number}. ' + escape(plain(caption)), styles['PaperCaption']))
        else:
            for paragraph in re.split(r'\n\s*\n', token):
                text = plain(paragraph)
                if text:
                    story.append(Paragraph(escape(text), styles['PaperBody']))
    story.append(Paragraph('References', styles['Heading2']))
    references = source.split('\\begin{thebibliography}{9}', 1)[1].split('\\end{thebibliography}', 1)[0]
    for key, reference in re.findall(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|$)', references, re.S):
        story.append(Paragraph(escape('[' + key + '] ' + plain(reference)), styles['PaperBody']))
    def footer(canvas, document):
        canvas.setFont('Times-Roman', 8)
        canvas.drawString(.75 * inch, .42 * inch, 'EXPLORATORY WORKING DRAFT | Physical: 466e80d | Learning: 44f22ea')
        canvas.drawRightString(7.75 * inch, .42 * inch, str(document.page))
    document = SimpleDocTemplate(str(destination), pagesize=(8.5 * inch, 11 * inch),
                                 leftMargin=.75 * inch, rightMargin=.75 * inch,
                                 topMargin=.7 * inch, bottomMargin=.7 * inch,
                                 title=plain(title), author='Working manuscript')
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    inputs = [tex, PHYSICAL / 'analysis.json', PHYSICAL / 'contact_response.png',
              PHYSICAL / 'contact_response.pdf', LEARNING / 'results.json', Path(__file__).resolve()]
    manifest = dict(run_status='COMPLETED', rendering='ReportLab review copy; not native TeX compilation',
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs},
                    output_sha256=sha(destination), physical_label=physical['label'],
                    learning_label=learned['label'], tables_verified_against_json=True)
    (PAPER / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(pdf=str(destination), bytes=destination.stat().st_size, tables_verified=True)))


if __name__ == '__main__':
    main()
