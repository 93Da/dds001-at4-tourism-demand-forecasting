"""Build a Chinese Word/PDF report from the completed simplified replication outputs."""
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
DOCX = ROOT / "技术复现报告_简化版.docx"
PDF = ROOT / "技术复现报告_简化版.pdf"

metrics = pd.read_csv(RESULTS / "test_metrics.csv")
focus = ["Seasonal naive", "Equal-weight experts", "Early-fusion LightGBM", "Context-gated fusion"]
metric_map = {(r.model, int(r.horizon_months)): r for r in metrics.itertuples(index=False)}

# -------- DOCX --------
def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill)
    tcPr.append(shd)

def set_run_font(run, name='Microsoft YaHei', size=10, bold=None, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn('w:eastAsia'), name)
    run.font.size = Pt(size)
    if bold is not None: run.bold = bold
    if color: run.font.color.rgb = RGBColor(*color)

def add_cell_text(cell, text, bold=False, color=None, size=9):
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(str(text))
    set_run_font(r, size=size, bold=bold, color=color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'
    for i,h in enumerate(headers):
        add_cell_text(table.rows[0].cells[i], h, bold=True, color=(255,255,255), size=9)
        set_cell_shading(table.rows[0].cells[i], '006B5E')
    for row in rows:
        cells = table.add_row().cells
        for i,val in enumerate(row):
            add_cell_text(cells[i], val, size=9)
    if widths:
        for row in table.rows:
            for i,w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph()
    return table

def add_para(doc, text, size=10, bold=False, italic=False, color=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    set_run_font(r, size=size, bold=bold, color=color)
    r.italic = italic
    return p

def add_heading(doc, text, level=1):
    p = doc.add_heading('', level=level)
    r = p.add_run(text)
    set_run_font(r, size=16 if level==1 else 12, bold=True, color=(0,104,91))
    return p

doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(0.65); sec.bottom_margin = Inches(0.65)
sec.left_margin = Inches(0.72); sec.right_margin = Inches(0.72)
normal = doc.styles['Normal']
normal.font.name = 'Microsoft YaHei'; normal._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei'); normal.font.size = Pt(10)

p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('AT3 技术复现报告（简化版）'); set_run_font(r, size=22, bold=True, color=(0,91,81))
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('Interpretable Multi-Source Feature Fusion for Tourism Demand Forecasting with Destination Competitiveness Diagnostics'); set_run_font(r, size=11, bold=True)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('基于既有 Eurostat 数据的专家模型与门控融合架构试验 · 2026-10-28'); set_run_font(r, size=9, color=(90,90,90))
doc.add_paragraph()

add_heading(doc, '一、研究目标与复现边界', 1)
add_para(doc, '本项目以 Zou 等人发表于 Systems and Soft Computing 的 2026 年论文为方法参考。原文提出 SG-MSF：六个 LightGBM 专家分别建模历史需求、数字吸引力、航空连接、价格竞争力、住宿容量与气候，再由动态门控网络组合预测。论文报告的测试期为 2024—2025 年，预测期限为 1、3、6 个月。')
add_para(doc, '本次实现的是“架构简化复现”，不是完整复现：我们复用上周下载的 Eurostat tour_occ_nim 工作簿，只构建历史需求、酒店子序列、其他住宿子序列三个特征族专家。后两者来自同一 Eurostat 表，并非独立外部数据源；未接入 Google Trends、航空、HICP、住宿容量或 ERA5-Land。', bold=True)

add_heading(doc, '二、数据与质量检查', 1)
add_para(doc, '数据筛选：EU27、外国居民（FOR）、数量单位（NR）、NACE I551-I553；目标期为 2015 年 1 月至 2025 年 12 月。原始工作簿已覆盖 2000 年 1 月至 2026 年 8 月，因此可使用 2015 年以前的数据构造历史滞后特征。没有重新下载数据。')
add_table(doc, ['数据项', '结果'], [
    ['国家数量', '27'], ['目标期理论观测数', '3,564（27 × 132 个月）'], ['目标期原始缺失值', '69（约 1.94%）'], ['训练期 2015–2021 缺失', '50'], ['2022 年缺失', '9'], ['2023 年缺失', '10'], ['测试期 2024–2025 缺失', '0'],
], widths=[2.8, 3.5])
add_para(doc, '缺失值没有被替换为零。模型拟合时剔除目标值缺失的训练样本；构造特征时，只使用对应预测截点之前的数据，并且只在该历史前缀内部插值，避免用截点之后的数据构造特征。这与原文的目标缺失值插补处理不同，是本次简化复现的一个明确偏差。')

add_heading(doc, '三、模型设计与实验流程', 1)
add_table(doc, ['模块', '简化实现'], [
    ['目标变量', 'EU27 非居民月度住宿夜数（I551-I553）'],
    ['专家 1', '历史需求滞后、滚动均值/波动、季节性滞后与日历特征'],
    ['专家 2', '历史需求特征 + 酒店住宿子序列（I551）的滞后特征'],
    ['专家 3', '历史需求特征 + 其他住宿子序列（I552_I553）的滞后特征'],
    ['早期融合基线', '将所有特征族拼接后训练一个 LightGBM'],
    ['门控融合', '32/16 隐藏层的 MLP 输出 3 个专家的 softmax 权重'],
    ['损失函数', '为简化实现使用 LightGBM L1 损失；原文使用 Huber 损失'],
    ['训练与测试', '逐年扩展窗口重训；门控用 2022 年训练、2023 年早停；最终测试 2024–2025'],
], widths=[1.7, 4.6])
add_para(doc, '每个预测期限 H=1、3、6 月独立训练专家。每个预测年份只重训一次，训练标签截点设为该年 1 月减去预测期限和 2 个月发布滞后。这比原文逐月更新及 expanding-window OOF 门控训练更简单，目的是先建立可复现、无明显未来信息泄漏的代码骨架。')

add_heading(doc, '四、实验结果', 1)
rows=[]
for name in focus:
    vals=[]
    for h in [1,3,6]:
        r=metric_map[(name,h)]
        vals.append(f'{r.median_country_MASE:.3f}')
    rows.append([name,*vals])
add_table(doc, ['模型', 'H=1 MASE', 'H=3 MASE', 'H=6 MASE'], rows, widths=[2.8,1.1,1.1,1.1])
add_para(doc, 'MASE 越低越好。该表使用同一测试样本进行模型间比较；H=6 为 647 条记录，因为一个国家的季节朴素滞后在预测截点不可用。')
for path, caption in [
    (FIGS/'mase_model_comparison.png', '图 1. 不同模型在三个预测期限上的国家级中位数 MASE。'),
    (FIGS/'gate_weights_by_horizon.png', '图 2. 测试期各预测期限的平均门控权重。'),
    (FIGS/'actual_vs_prediction_selected_countries.png', '图 3. 德国、法国、意大利的 H=1 实际值与模型预测。'),
]:
    if path.exists():
        doc.add_picture(str(path), width=Inches(6.25))
        p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        r=p.add_run(caption); set_run_font(r,size=9,color=(80,80,80))

add_heading(doc, '五、结果解读', 1)
add_para(doc, '（1）H=1：等权专家组合的中位数 MASE 最低，为 0.396；早期融合为 0.406，门控融合为 0.439。门控并未在短期预测上稳定胜出。')
add_para(doc, '（2）H=3：门控融合的中位数 MASE 为 0.398，是本次简化实验中最好的结果，略优于等权专家组合（0.412）和季节朴素基线（0.433）。')
add_para(doc, '（3）H=6：季节朴素基线的中位数 MASE 为 0.433，优于早期融合（0.514）、等权组合（0.624）和门控融合（0.658）。这表明在较长预测期限下，当前简化模型没有稳定利用额外特征族提升预测。')
add_para(doc, '（4）指标差异值得关注：季节朴素基线的 WAPE 约为 7.38%，而融合模型约为 8.2%–11.4%。中位数 MASE 按国家汇总，而 WAPE 更受大体量目的地影响；因此不能只凭一个汇总指标判断模型优劣。')
add_para(doc, '原论文报告的 SG-MSF MASE 为 H=1/3/6 的 0.504/0.566/0.627，早期融合 LightGBM 为 0.563/0.539/0.568，季节朴素基线为 0.640。由于本项目的数据源、缺失值处理、损失函数和重训频率均不同，这些数值只作为文献参考，不应声称与原文结果可直接比较。')

add_heading(doc, '六、局限与下一步', 1)
add_para(doc, '1. 当前只有同一 Eurostat 表内的住宿类别子序列，没有真正独立的外部数据源，因此还不能验证原文六源融合的完整贡献。')
add_para(doc, '2. 本次使用 L1 损失代替原文 Huber 损失；采用逐年重训代替逐月重训；门控训练使用 2022 年预测、2023 年早停，未复现原文 2017–2021 年 expanding-window OOF 门控训练。')
add_para(doc, '3. 没有实现 Google Trends、航空旅客/航班、HICP、住宿容量、ERA5-Land、预测区间、目的地聚类、SHAP 竞争力诊断及显著性检验。')
add_para(doc, '建议下一步只增加一个独立且可获取的数据源，例如航空客运或价格指数，并记录其真实发布时间与滞后；在此基础上增加独立专家，再比较早期融合与门控融合。只有完成这一步，才更接近原论文的“多源”主张。')

add_heading(doc, '参考资料', 1)
add_para(doc, 'Zou, L., Zhai, M., Zhang, J., & Liu, L. (2026). Interpretable Multi-Source Feature Fusion for Tourism Demand Forecasting with Destination Competitiveness Diagnostics. Systems and Soft Computing, Article 200665. https://doi.org/10.1016/j.sasc.2026.200665')
add_para(doc, 'Eurostat. Nights spent at tourist accommodation establishments – monthly data (tour_occ_nim). Dataset DOI: https://doi.org/10.2908/tour_occ_nim')

# footer
for section in doc.sections:
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = footer.add_run('AT3 技术复现（简化版） | Eurostat tour_occ_nim | 2026-10-28')
    set_run_font(rr, size=8, color=(100,100,100))

doc.save(DOCX)

# -------- PDF via ReportLab --------
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='CNTitle', parent=styles['Title'], fontName='STSong-Light', fontSize=19, leading=25, alignment=1, textColor=colors.HexColor('#005B51'), spaceAfter=8))
styles.add(ParagraphStyle(name='CNSub', parent=styles['Normal'], fontName='STSong-Light', fontSize=9, leading=13, alignment=1, textColor=colors.HexColor('#555555'), spaceAfter=12))
styles.add(ParagraphStyle(name='CNH1', parent=styles['Heading1'], fontName='STSong-Light', fontSize=13, leading=18, textColor=colors.HexColor('#00685B'), spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle(name='CNBody', parent=styles['BodyText'], fontName='STSong-Light', fontSize=8.8, leading=13, spaceAfter=5))
styles.add(ParagraphStyle(name='CNSmall', parent=styles['BodyText'], fontName='STSong-Light', fontSize=7.5, leading=10, textColor=colors.HexColor('#555555'), spaceAfter=4))
styles.add(ParagraphStyle(name='CNTable', parent=styles['BodyText'], fontName='STSong-Light', fontSize=7.4, leading=9))

def P(text, style='CNBody'):
    return Paragraph(text.replace('&','&amp;'), styles[style])

def pdf_table(headers, rows, widths):
    data=[[P(str(x),'CNTable') for x in headers]] + [[P(str(x),'CNTable') for x in row] for row in rows]
    t=Table(data, colWidths=widths, repeatRows=1, hAlign='CENTER')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#006B5E')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#9BA7A5')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
        ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
    ]))
    return t

def add_page_num(canvas, doc_template):
    canvas.saveState()
    canvas.setFont('STSong-Light', 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(A4[0]/2, 0.35*inch, f'AT3 技术复现（简化版） · 第 {doc_template.page} 页')
    canvas.restoreState()

story=[]
story += [P('AT3 技术复现报告（简化版）','CNTitle'), P('旅游需求预测中的专家模型与门控融合 · 2026-10-28','CNSub')]
story += [P('一、研究目标与复现边界','CNH1'), P('本项目以 Zou 等人发表于 Systems and Soft Computing 的 2026 年论文为方法参考。原文提出 SG-MSF：六个 LightGBM 专家分别建模历史需求、数字吸引力、航空连接、价格竞争力、住宿容量与气候，再由动态门控网络组合预测。'), P('<b>复现边界：</b>本次是架构简化复现，不是完整复现。只使用既有 Eurostat 数据和三个特征族专家；酒店与其他住宿子序列来自同一表，不是独立外部来源。')]
story += [P('二、数据与质量检查','CNH1'), P('目标数据为 EU27、外国居民（FOR）、数量单位（NR）、NACE I551-I553；目标期 2015 年 1 月至 2025 年 12 月。原始工作簿覆盖 2000 年 1 月至 2026 年 8 月，无需重新下载。')]
story.append(pdf_table(['数据项','结果'], [['国家数量','27'],['目标期理论观测数','3,564'],['目标期原始缺失值','69（1.94%）'],['训练期 2015–2021 缺失','50'],['2022 年缺失','9'],['2023 年缺失','10'],['测试期 2024–2025 缺失','0']], [2.2*inch,3.8*inch]))
story += [Spacer(1,6), P('缺失值没有被替换为零；目标缺失行从拟合与评估中排除。构造每一行特征时，先按发布滞后截断历史数据，再在截点内插值，避免读取截点之后的值。该处理与原文的目标插补不同。')]
story += [P('三、模型设计与实验流程','CNH1'), pdf_table(['模块','简化实现'], [
    ['专家 1','历史需求滞后、滚动统计、季节性滞后与日历特征'],
    ['专家 2','历史需求 + 酒店子序列 I551 的滞后特征'],
    ['专家 3','历史需求 + 其他住宿子序列 I552_I553 的滞后特征'],
    ['基线','季节朴素、早期融合 LightGBM、等权专家组合'],
    ['门控','32/16 隐藏层 MLP，输出 3 个专家的 softmax 权重'],
    ['损失','使用 L1 损失替代原文 Huber'],
    ['切分','2022 门控训练、2023 早停、2024–2025 最终测试；专家按预测年份扩展重训'],
], [1.4*inch,4.6*inch])]
story += [P('四、结果：国家级中位数 MASE（越低越好）','CNH1')]
rows=[]
for name in focus:
    rows.append([name]+[f'{metric_map[(name,h)].median_country_MASE:.3f}' for h in [1,3,6]])
story.append(pdf_table(['模型','H=1','H=3','H=6'], rows, [2.8*inch,1.05*inch,1.05*inch,1.05*inch]))
story += [Spacer(1,6), P('H=1 最佳为等权专家组合（0.396）；H=3 最佳为门控融合（0.398）；H=6 最佳为季节朴素（0.433）。门控并非所有期限都胜出。季节朴素 WAPE 为 7.38%，而融合模型约为 8.2%–11.4%，说明按国家汇总的 MASE 与按总量加权的 WAPE 会呈现不同视角。')]
for path, caption in [
    (FIGS/'mase_model_comparison.png','图 1. 不同模型的中位数 MASE。'),
    (FIGS/'gate_weights_by_horizon.png','图 2. 测试期平均门控权重。'),
    (FIGS/'actual_vs_prediction_selected_countries.png','图 3. 德国、法国、意大利 H=1 实际值与预测。'),
]:
    if path.exists():
        img=Image(str(path), hAlign='CENTER')
        img.drawWidth, img.drawHeight = img._restrictSize(6.2*inch, 4.0*inch)
        story += [Spacer(1,6), img, P(caption,'CNSmall')]
story += [P('五、局限与下一步','CNH1'), P('1. 只有同一 Eurostat 表内的住宿类别子序列，没有真正独立的外部数据源。2. 使用 L1 而非原文 Huber；逐年而非逐月重训；门控使用 2022 年预测训练、2023 年早停。3. 未接入 Google Trends、航空、HICP、住宿容量、ERA5-Land，也未复现 SHAP 竞争力诊断、区间预测或显著性检验。'), P('因此应将本项目称为“简化架构复现/可行性试验”，不应称为原论文完整复现。下一步建议先增加一个有明确发布时间的独立来源（如航空客运或 HICP），再训练独立来源专家。')]
story += [P('参考资料','CNH1'), P('Zou, L., Zhai, M., Zhang, J., & Liu, L. (2026). Interpretable Multi-Source Feature Fusion for Tourism Demand Forecasting with Destination Competitiveness Diagnostics. Systems and Soft Computing, Article 200665. DOI: 10.1016/j.sasc.2026.200665.'), P('Eurostat. Nights spent at tourist accommodation establishments – monthly data (tour_occ_nim). DOI: 10.2908/tour_occ_nim.')]
pdf = SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=0.55*inch, leftMargin=0.55*inch, topMargin=0.55*inch, bottomMargin=0.6*inch, title='AT3 技术复现报告（简化版）', author='OpenAI')
pdf.build(story, onFirstPage=add_page_num, onLaterPages=add_page_num)
print('Created', DOCX, DOCX.stat().st_size, 'bytes')
print('Created', PDF, PDF.stat().st_size, 'bytes')
