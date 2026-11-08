"""Build the capstone final report (Word) in the course format: python -m src.report.build_capstone_report

Writes docs/capstone_report.docx. Numbers are read from reports/results/ and data/ so the text matches the artefacts.
Yellow-highlighted text marks details the team must fill in or confirm before submission.
Run scripts/finalize_report.ps1 (Word) afterwards to refresh the table of contents and export the PDF.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from matplotlib.patches import FancyBboxPatch

from src.models import compare as cmp

ROOT = Path(__file__).resolve().parents[2]
RES, FIG, DOCS = ROOT / "reports" / "results", ROOT / "figures", ROOT / "docs"
OUT = DOCS / "capstone_report.docx"
REPFIG = FIG / "report"
REPFIG.mkdir(parents=True, exist_ok=True)
NAVY, TEAL, RED, GREY = "#142D4E", "#157A80", "#C44E52", "#8C99A6"
FONT = "Times New Roman"
rd = lambda f: pd.read_csv(RES / f)

# ============================================================================== figures made for the report
def make_workflow() -> Path:
    fig, ax = plt.subplots(figsize=(10, 3.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    def box(x, y, w, h, text, fc, tc="white", size=8.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec="none"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=size, fontweight="bold" if bold else "normal", linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))

    box(1, 24, 17, 13, "Data collection\n(own scrapers)\nStore, reviews,\nSteamSpy, news\n1,861 games\n1.65M reviews", NAVY)
    box(22, 24, 17, 13, "Cleaning and\nvalidation\n32 logged rules\n21 + 21 checks\nSplit by game\n(80/20, 5 folds)", NAVY)
    box(43, 24, 17, 13, "UNIT 1\nExploratory analysis\nrefund window, price,\nfeatures, tag PCA", TEAL, bold=True)
    box(64, 24, 17, 13, "UNIT 2\nPredictive models\nreview level and\ngame level; grid\nsearch; SHAP", TEAL, bold=True)
    box(85, 24, 14, 13, "Baselines\nDummy vs 4\nmodel families", GREY)
    box(43, 4, 17, 14, "UNIT 3, method 1\nText mining\ncomplaint themes,\nnegative-review\ndetector", RED, bold=True)
    box(64, 4, 17, 14, "UNIT 3, method 2\nAssociation rules\nFP-Growth, FDR,\nheld-out check", RED, bold=True)
    box(85, 4, 14, 14, "Extra:\npatch impact\nwith matched\ncontrols", GREY)
    box(1, 4, 17, 14, "Combined insights\nand\nrecommendations", NAVY, bold=True)
    box(22, 4, 17, 14, "Interactive\ndashboard\n(Streamlit, 5 pages)\nindieguard.streamlit.app", NAVY, bold=True)
    arrow(18.3, 30.5, 21.7, 30.5); arrow(39.3, 30.5, 42.7, 30.5); arrow(60.3, 30.5, 63.7, 30.5); arrow(81.3, 30.5, 84.7, 30.5)
    arrow(30, 23.7, 30, 18.5); arrow(51.5, 23.7, 51.5, 18.5); arrow(72.5, 23.7, 72.5, 18.5)
    arrow(42.7, 11, 39.3, 11); arrow(63.7, 11, 60.3, 11); arrow(21.7, 11, 18.3, 11)
    ax.text(50, 39.2, "Cleaned data and the shared split feed every unit; results flow into the combined insights and the dashboard", ha="center", fontsize=8.5, color=NAVY)
    p = REPFIG / "workflow.png"; fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); return p


def make_rules_figure() -> Path:
    r = rd("association_rules_strong.csv").sort_values("lift", ascending=False).head(8)
    names = [s.replace("tag:", "").replace("price:", "price ") for s in r.antecedents_str]
    fig, ax = plt.subplots(figsize=(7.4, 3.3)); y = range(len(r))
    ax.barh([i + 0.2 for i in y], 100 * r.confidence, height=0.38, color=NAVY, label="training games (1,191)")
    ax.barh([i - 0.2 for i in y], 100 * r.test_confidence, height=0.38, color=RED, label="test games (299)")
    for i, (a, b, n) in enumerate(zip(r.confidence, r.test_confidence, r.test_n_antecedent)):
        ax.text(100 * a + 1, i + 0.2, f"{100 * a:.0f}%", va="center", fontsize=7.5); ax.text(100 * b + 1, i - 0.2, f"{100 * b:.0f}% (n={int(n)})", va="center", fontsize=7.5)
    ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=7.8); ax.invert_yaxis(); ax.set_xlim(0, 105)
    ax.set_xlabel("share of games with the combination that reached the Strong tier (%)", fontsize=8); ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    p = REPFIG / "rules_train_vs_test.png"; fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); return p


def crop(src: Path, box: tuple[int, int, int, int], name: str) -> Path:
    from PIL import Image
    im = Image.open(src).crop(box); p = REPFIG / name; im.save(p); return p


# ============================================================================== document helpers
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Inches(0.9))
HL_RE = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`|⟦.+?⟧)")


def set_font(style_or_run, size=None, bold=None, italic=None, color=None, name=FONT):
    f = style_or_run.font
    f.name = name
    rpr = style_or_run.element.get_or_add_rPr() if hasattr(style_or_run, "element") and hasattr(style_or_run.element, "get_or_add_rPr") else style_or_run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts"); rpr.append(rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(a), name)
    if size: f.size = Pt(size)
    if bold is not None: f.bold = bold
    if italic is not None: f.italic = italic
    if color: f.color.rgb = RGBColor.from_string(color)


st = doc.styles
set_font(st["Normal"], 12)
st["Normal"].paragraph_format.line_spacing = 1.5
st["Normal"].paragraph_format.space_after = Pt(3)
for name, size in (("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 12)):
    h = st[name]; set_font(h, size, bold=True, color="000000")
    h.paragraph_format.space_before = Pt(10 if name == "Heading 1" else 8); h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True; h.paragraph_format.line_spacing = 1.15
set_font(st["Title"], 24, bold=True, color="142D4E"); set_font(st["Subtitle"], 14, italic=True, color="404040")
set_font(st["Caption"], 10, bold=False, italic=False, color="000000")
st["Caption"].paragraph_format.line_spacing = 1.0; st["Caption"].paragraph_format.space_after = Pt(4)
set_font(st["List Bullet"], 12); st["List Bullet"].paragraph_format.line_spacing = 1.5


def add_runs(p, text, size=None, color=None):
    for tok in HL_RE.split(text):
        if not tok:
            continue
        if tok.startswith("**"):
            r = p.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith("⟦"):
            r = p.add_run("[" + tok[1:-1] + "]"); r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        elif tok.startswith("`"):
            r = p.add_run(tok[1:-1]); r.font.name = "Courier New"
            r._element.get_or_add_rPr().find(qn("w:rFonts")).set(qn("w:hAnsi"), "Courier New")
            r.font.size = Pt((size or 12) - 1.5)
            continue
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = p.add_run(tok[1:-1]); r.italic = True
        else:
            r = p.add_run(tok)
        set_font(r, size)
        if color: r.font.color.rgb = RGBColor.from_string(color)
    return p


def para(text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, space_after=None, keep=False):
    p = doc.add_paragraph(); p.alignment = align; add_runs(p, text, size)
    if space_after is not None: p.paragraph_format.space_after = Pt(space_after)
    if keep: p.paragraph_format.keep_with_next = True
    return p


def bullets(items, size=None):
    for t in items:
        p = doc.add_paragraph(style="List Bullet"); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; add_runs(p, t, size)


def heading(text, level=1, page_break=False):
    h = doc.add_heading(level=level); h.add_run(text); set_font(h.runs[0], 14 if level == 1 else 12, bold=True, color="000000")
    if page_break: h.paragraph_format.page_break_before = True
    return h


def field(p, instr, placeholder=""):
    for t, txt in (("begin", None), ("instr", instr), ("separate", None), ("text", placeholder), ("end", None)):
        r = p.add_run()
        if t in ("begin", "separate", "end"):
            fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), {"begin": "begin", "separate": "separate", "end": "end"}[t]); r._r.append(fc)
        elif t == "instr":
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = f" {txt} "; r._r.append(it)
        else:
            r.text = txt
        set_font(r, 10)


def caption(kind, text):
    p = doc.add_paragraph(style="Caption"); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"{kind} "); r.bold = True; set_font(r, 10, bold=True)
    field(p, f"SEQ {kind} \\* ARABIC", "1")
    for r in p.runs[-5:]:
        r.bold = True
    r = p.add_run(". " + text.replace("`", "")); set_font(r, 10)
    return p


FIGN, TABN = [0], [0]


def figure(path: Path, caption_text: str, width=6.0):
    FIGN[0] += 1
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_with_next = True; p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0
    p.add_run().add_picture(str(path), width=Inches(width))
    caption("Figure", caption_text)
    return FIGN[0]


def figure_pair(p1: Path, c1: str, p2: Path, c2: str, width=2.9):
    t = doc.add_table(rows=2, cols=2); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, (pth, c) in enumerate(((p1, c1), (p2, c2))):
        FIGN[0] += 1
        cell = t.cell(0, j); cp = cell.paragraphs[0]; cp.alignment = WD_ALIGN_PARAGRAPH.CENTER; cp.paragraph_format.line_spacing = 1.0
        cp.add_run().add_picture(str(pth), width=Inches(width)); cp.paragraph_format.keep_with_next = True
        cap = t.cell(1, j).paragraphs[0]; cap.style = doc.styles["Caption"]; cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run("Figure "); r.bold = True; set_font(r, 10, bold=True)
        field(cap, "SEQ Figure \\* ARABIC", str(FIGN[0]))
        r = cap.add_run(". " + c.replace("`", "")); set_font(r, 10)
    for rr in t.rows:
        trpr = rr._tr.get_or_add_trPr(); trpr.append(OxmlElement('w:cantSplit'))
    sp_ = doc.add_paragraph(); sp_.paragraph_format.space_after = Pt(0); sp_.paragraph_format.line_spacing = 1.0


def shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), fill); tcpr.append(sh)


def table(rows, header, widths, caption_text, size=9.5, align_right_from=None):
    if caption_text:
        TABN[0] += 1
        cp = caption("Table", caption_text); cp.paragraph_format.keep_with_next = True; cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t = doc.add_table(rows=1 + len(rows), cols=len(header)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, h in enumerate(header):
        c = t.cell(0, j); c.text = ""; add_runs(c.paragraphs[0], f"**{h}**", size); shade(c, "D9E2EC")
        if align_right_from is not None and j >= align_right_from:
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = ""; add_runs(c.paragraphs[0], str(v), size)
            if align_right_from is not None and j >= align_right_from:
                c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for r in t.rows:
        for j, w in enumerate(widths):
            r.cells[j].width = Inches(w)
        for c in r.cells:
            for p in c.paragraphs:
                p.paragraph_format.line_spacing = 1.0; p.paragraph_format.space_after = Pt(1); p.paragraph_format.space_before = Pt(1)
        trpr = r._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); trpr.append(cs)
    th = t.rows[0]._tr.get_or_add_trPr(); h_ = OxmlElement("w:tblHeader"); th.append(h_)
    sp = doc.add_paragraph(); sp.paragraph_format.space_after = Pt(4); sp.paragraph_format.line_spacing = 1.0
    return TABN[0]


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def footer_page_number(section, fmt: str, start: int | None):
    f = section.footer; f.is_linked_to_previous = False
    p = f.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in list(p.runs): r._r.getparent().remove(r._r)
    field(p, "PAGE", "1")
    sp = section._sectPr; pg = sp.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType"); sp.append(pg)
    pg.set(qn("w:fmt"), fmt)
    if start is not None: pg.set(qn("w:start"), str(start))


# ============================================================================== data used in the text
cr, cg = cmp.load("review"), cmp.load("game")
ct = cmp.load("review", "reviewtext_{key}.csv", cmp.TEXT_ORDER, cmp.TEXT_NAMES)
rc = rd("association_rule_configuration_comparison.csv")
rules_s = __import__("json").load(open(RES / "association_rules_summary.json"))
det = rd("reviewtext_tfidf_lr_detail.csv").iloc[0]
tun = rd("comparison_tuning.csv")
games_s = pd.read_csv(ROOT / "data" / "sample" / "games_sample.csv")
rev_s = pd.read_csv(ROOT / "data" / "sample" / "reviews_sample.csv")
SHARE_EARLY, NEG_ALL, NEG_EARLY, NEG_CHEAP = 0.140, 0.1037, 0.2117, 0.1138     # computed from reviews_clean (see 6.3)

# ============================================================================== cover page
sec.different_first_page_header_footer = True
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before = Pt(30)
add_runs(p, "CAPSTONE PROJECT: FINAL REPORT", size=13, color="404040")
t = doc.add_paragraph(style="Title"); t.alignment = WD_ALIGN_PARAGRAPH.CENTER; _r = t.add_run("IndieGuard: Indie Game Launch Analytics"); set_font(_r, 24, bold=True, color="142D4E")
s = doc.add_paragraph(style="Subtitle"); s.alignment = WD_ALIGN_PARAGRAPH.CENTER
_r2 = s.add_run("Predicting negative-review risk and identifying winning feature combinations on Steam"); set_font(_r2, 14, italic=True, color="404040")
meta = [("Course", "23CSE452 Business Analytics (Units 1, 2 and 3)"), ("Programme / section", "⟦programme and section⟧"),
        ("Team number", "Group 1"), ("Guide", "Rohini S"), ("Academic year", "2026-2027"), ("Submission date", "⟦submission date⟧"),
        ("Repository and board", "github.com/Shan713/IndieGuard; github.com/users/Shan713/projects/1"),
        ("Live dashboard", "https://indieguard.streamlit.app")]
tb = doc.add_table(rows=len(meta), cols=2); tb.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, (k, v) in enumerate(meta):
    for j, txt in enumerate((f"**{k}**", v)):
        c = tb.cell(i, j); c.text = ""; add_runs(c.paragraphs[0], txt, 11); c.paragraphs[0].paragraph_format.line_spacing = 1.15
        c.width = Inches(1.9 if j == 0 else 4.4)
doc.add_paragraph().paragraph_format.space_after = Pt(4)
tm = doc.add_table(rows=6, cols=2); tm.style = "Table Grid"; tm.alignment = WD_TABLE_ALIGNMENT.CENTER
members = [("Team member", "Register number"), ("Thotakura Srihitha", "CB.SC.U4CSE23450"), ("Gayas Sheik", "CB.SC.U4CSE23118"), ("Shantharam P", "CB.SC.U4CSE23146"),
           ("Pradeep Sasidharan", "CB.SC.U4CSE23232"), ("Valikala Tejaswini", "CB.SC.U4CSE23752")]
for i, (a, b) in enumerate(members):
    for j, txt in enumerate((a, b)):
        c = tm.cell(i, j); c.text = ""; add_runs(c.paragraphs[0], f"**{txt}**" if i == 0 else txt, 11); c.width = Inches(3.1)
        c.paragraphs[0].paragraph_format.line_spacing = 1.1
        if i == 0: shade(c, "D9E2EC")

# ============================================================================== declaration, acknowledgement
heading("Declaration and acknowledgement", 1, page_break=True)
para("We declare that this report and the project it describes are our own work as a team. The data were collected by our own scrapers from public Steam sources; "
     "no pre-built dataset was used. Prior studies, algorithms and software are cited in the reference list. Our use of generative-AI tools, "
     "what each tool supported, the code and content it affected, and how we verified and corrected it are disclosed in full in Section 8.3. "
     "Tools assisted the work; the project's research design, data collection runs, analytical decisions and final conclusions were made, run and checked by the team.")
para("We thank our guide, Rohini S, for guidance and feedback during the capstone, and our course faculty for the structure and rubric that shaped the work.")
sg = doc.add_table(rows=6, cols=3); sg.style = "Table Grid"
for j, h in enumerate(("Team member", "Signature", "Date")):
    c = sg.cell(0, j); c.text = ""; add_runs(c.paragraphs[0], f"**{h}**", 10.5); shade(c, "D9E2EC")
for i, (n, _) in enumerate(members[1:], start=1):
    sg.cell(i, 0).text = ""; add_runs(sg.cell(i, 0).paragraphs[0], n, 10.5)
    for j in (1, 2): sg.cell(i, j).text = ""
for r in sg.rows:
    r.height = Inches(0.38)
    for j, w in enumerate((2.6, 2.4, 1.5)): r.cells[j].width = Inches(w)

# ============================================================================== abstract and executive summary
heading("Abstract", 1, page_break=True)
para("A small indie studio has little room for a bad launch. Steam lets a player refund a game bought in the last 14 days if they have played it for under two hours, "
     "so the reviews written in that early window show whether the first hour convinced players. IndieGuard asks what drives negative reviews, which feature combinations go with strong reception, and whether patches help. "
     "We collected our own data with throttled, resumable scrapers on 1,861 indie roguelike and roguelite games (2022 to 2025), 1.65 million reviews and 29,918 patch and sale events, "
     "cleaned them under 32 logged rules and split them by game. Exploratory analysis (Unit 1), five model families at review and game level with grid-search tuning (Unit 2), and two Unit 3 methods, "
     "text mining and association rule mining, were evaluated on held-out games. Reviews written inside the refund window are 21.2% negative against 8.8% later. A text model detects negative reviews "
     "with a PR-AUC of 0.84 and ROC-AUC of 0.97, but store metadata predicts a game's success tier only modestly (macro-F1 about 0.42 to 0.44 against 0.22 for guessing). "
     "Ninety-five tag combinations pass false-discovery control and about two-thirds of the testable ones hold on unseen games; patches show no clear effect against matched controls. "
     "A deployed Streamlit dashboard integrates the results and a launch-risk scorer. All findings are associations in observational data.")

heading("Executive summary", 1, page_break=True)
para("Small studios have to decide what to fix before launch, which feature combinations to back and when a patch is worth the effort, but public information gives them little to go on. "
     "Reviews written in a game's first two hours deserve special attention: the player can still ask for a refund at that point, so a negative review there usually means the first hour did not work for them.")
para("To study this we built our own dataset from Steam (1,861 indie roguelike games and 1.65 million reviews) and checked each result on games that were held back from the models. The main findings are:")
bullets(["**The first two hours matter.** 21.2% of reviews written under two hours of play are negative, against 8.8% later; they make up 29% of all negative reviews. For games priced $20 or more the early figure is 31.8%.",
         "**Complaints change.** Early negative reviews lean to crashes, multiplayer bugs and controls; later ones to boss design, balance and repetition.",
         "**Store data alone say little.** A model that sees only store information finds a game's success tier about twice as well as guessing, which is useful for comparing launch plans but not a forecast.",
         "**Reading the review text works.** A text model flags negative reviews far better than store data (ROC-AUC 0.97 against 0.76 on the same reviews). It detects complaints; it does not forecast them.",
         "**Focused 2D, pixel, retro and procedural combinations** reach the Strong tier about two-thirds of the time (35% overall); no combination passed our thresholds for struggling launches.",
         "**Patches do not lift ratings by themselves.** Against matched games with no patch, the change is +0.27 points (interval -0.82 to +1.36)."])
para("From these results we suggest five actions for a studio to test, in priority order: (1) treat the first hour as a standalone product and blind-test it; (2) match the price to what the first hour delivers; "
     "(3) try a focused scope; (4) complete the store page; (5) patch for specific problems rather than to lift sentiment in general. In the illustrative scenario in Section 6.3, a first-hour fix that "
     "brings early negativity down to the level of cheap games would avoid about 14 negative reviews per 1,000; this is an estimate under stated assumptions, not a measured result. "
     "The live dashboard (https://indieguard.streamlit.app) lets a studio explore the market, the tag rules, the complaint themes, the patch results and score a planned game.")

# ============================================================================== contents lists
heading("Table of contents", 1, page_break=True)
p = doc.add_paragraph(); field(p, 'TOC \\o "1-2" \\h \\z \\u', "Right-click and choose Update field to build the table of contents.")
heading("List of figures", 1, page_break=True)
p = doc.add_paragraph(); field(p, 'TOC \\h \\z \\c "Figure"', "Update field to build the list of figures.")
heading("List of tables", 1)
p = doc.add_paragraph(); field(p, 'TOC \\h \\z \\c "Table"', "Update field to build the list of tables.")
heading("Abbreviations", 1)
abbr = [("AUC", "Area under the curve (ROC-AUC: receiver operating characteristic; PR-AUC: precision-recall)"), ("CV", "Cross-validation"), ("EDA", "Exploratory data analysis"),
        ("FDR", "False discovery rate"), ("FP-Growth", "Frequent-pattern growth (itemset mining)"), ("LDA", "Latent Dirichlet allocation"), ("NMF", "Non-negative matrix factorization"),
        ("PCA", "Principal component analysis"), ("SHAP", "Shapley additive explanations"), ("SVD", "Singular value decomposition"), ("TF-IDF", "Term frequency, inverse document frequency")]
table(abbr, ["Abbreviation", "Meaning"], [1.5, 5.0], None, size=10)

# ============================================================================== BODY: new section, arabic page numbers
body = doc.add_section(WD_SECTION.NEW_PAGE)
body.different_first_page_header_footer = False
footer_page_number(doc.sections[0], "lowerRoman", 1)
footer_page_number(body, "decimal", 1)

# ---------------------------------------------------------------- 1
heading("1. Introduction, Business Objectives and Literature Review", 1)
heading("1.1 Context and scope", 2)
para('Steam lets players refund a game within 14 days of purchase if they have played it for less than two hours [4]. Refunds themselves are not public, but reviews are, and a negative review written in that window is an early sign that the first hour did not convince the player. Our stakeholders are indie studios and small publishers who must decide what to build, what to fix before launch and when a patch is worth it. We kept the scope narrow: indie roguelike and roguelite games released between 2022 and 2025 with at least 10 reviews (1,861 games). All findings are associations, and revenue and refund figures are out of scope because Steam does not publish them.')
para('**Overall objective:** help a studio decide which feature combinations to back, what to fix in the first hours of play, and when to patch. We ask five questions: will a review be negative and how does playtime affect this (Q1); which success tier will a game reach (Q2); which tag, price and Early Access combinations go with strong reception (Q3); what do players complain about (Q4); and how do ratings change after a patch (Q5). Table 1 lists the KPIs and Figure 1 shows how the units fit together.')
table([["O1. Quantify the refund-window effect (Q1)", "Negative share inside vs outside the window (percentage points)", "Baseline: 10.4% overall. Target: report the gap with a 95% interval", "U1"],
       ["O2. Detect negative reviews (Q1)", "PR-AUC on held-out games", "Baseline: guessing, 0.115. Target: clearly above baseline, CV spread reported", "U2, U3 M1"],
       ["O3. Predict the game tier (Q2)", "Macro-F1 and recall per tier on held-out games", "Baseline: 0.22. Target: clearly above baseline", "U2"],
       ["O4. Find tag combinations tied to Strong reception (Q3)", "Rules passing FDR control (q < 0.10, lift at least 1.2) and holding on test games", "Baseline: none. Target: rules with a held-out check", "U3 M2"],
       ["O5. Find complaint themes (Q4)", "Topic diversity, coherence, stability and agreement with a manual check", "Baseline: LDA topic model. Target: interpretable, stable themes", "U3 M1"],
       ["O6. Measure patch impact (Q5)", "Change in positive-review share against matched no-patch controls (points)", "Baseline: 0 points. Target: interval around the difference", "U1, extra"],
       ["O7. Deliver an interactive application", "Public dashboard with filters that update results", "Target: working link, 5 pages, live scenario", "App"]],
      ["Business objective", "KPI / unit", "Baseline / target", "Unit / method"], [1.9, 1.9, 1.9, 0.8], "Business objectives, KPIs and success criteria.", size=9)
figure(make_workflow(), "End-to-end workflow: how data collection, Units 1 to 3 and the dashboard connect.", width=5.0)
heading("1.2 Related analytics work and research gap", 2)
para("Lin et al. studied about 11 million reviews of 6,224 Steam games and found that negative reviews arrive early and that players criticise design more than bugs [1]. Their work is descriptive and covers the whole platform. De Luisa et al. related price, size, languages, release date and genre to a game's player numbers using Bayesian models [2]; they predict popularity rather than review sentiment and do not use review text. Jorge and Pardo built SteamBR, a corpus of 2.8 million Brazilian-Portuguese reviews, to predict which reviews are helpful [3]. Steam's own review labels and SteamSpy's owner estimates [4, 5] describe games only after the fact. None of these connects early-playtime negativity to what a studio should fix inside the refund window, or links review risk to launch features and patch timing with held-out testing. Text mining and association rules fit this gap: complaints live in free text, and launch decisions are combinations of tags, price and scope.")
# ---------------------------------------------------------------- 2
heading("2. Dataset Collection, Validation and Preparation", 1)
heading("2.1 Provenance, coverage and challenges", 2)
para("We collected all data ourselves on 1 and 2 October 2026 from four public Steam sources, using scrapers we wrote: store search (Indie plus five roguelike tags, 2022 to 2025), store details and tags, SteamSpy owner estimates, and the news feed (patch and sale announcements). Reviews were paged with Steam's cursor. The dataset is not taken from Kaggle or any other repository, and it contains no synthetic records. The endpoints are public, and we used them for non-commercial academic analysis at a low request rate. The raw files are kept only on team machines (they are excluded from the public repository), frozen with row counts and a hash, and are not redistributed; the repository holds the cleaned, pseudonymised tables.")
para('The raw data hold 1,861 games (58 variables), 1,652,999 reviews in 31 languages and 29,918 news events. After cleaning, 1,648,387 reviews (34 variables) remain. The data are observed at three levels: review, game and announcement. There are two targets. At review level the target is whether a review is negative (about 10% are). At game level it is a three-level tier, defined for the 1,490 games with at least 20 reviews: Struggling (30% or more negative), Solid, and Strong (10% or less). Table 2 shows sample rows and Appendix A gives the data dictionary.')
rows = []
for r in games_s[games_s.tier.notna()].sample(5, random_state=5).itertuples():
    rows.append([r.name[:30], f"{int(r.release_year)}", f"${r.price_usd:.2f}", f"{int(r.steam_total_reviews):,}", f"{r.neg_share:.0%}", r.tier])
table(rows, ["Game", "Release year", "Price", "Steam reviews", "Negative share", "Tier"], [2.4, 0.8, 0.8, 0.9, 0.9, 0.8],
      "Representative sample of the final data: five games (top) and five reviews (bottom); full samples are in data/sample/.", size=9.5)
doc.paragraphs[-1].paragraph_format.space_after = Pt(0)
rows = []
for r in rev_s.sample(5, random_state=2).itertuples():
    txt = str(r.review).replace("\n", " ")
    rows.append([txt[:46] + ("..." if len(txt) > 46 else ""), r.language, f"{r.playtime_hours:g} h", "negative" if not r.voted_up else "positive"])
table(rows, ["Review text (shortened)", "Language", "Playtime at review", "Target"], [3.6, 0.9, 1.1, 1.0], None, size=9.5)
para('Collection was slower than planned. Steam started refusing requests (HTTP 429) after roughly 150 calls, so we changed the scraper to wait between requests, back off when refused and save its place so that a run could resume after a failure. SteamSpy was blocked by our network provider and had to be reached through another route. For the five largest games we kept only the newest 15,000 reviews; for every other game we collected almost all the reviews Steam reports.')
heading("2.2 Data quality, privacy and ethics", 2)
para('Cleaning followed 32 logged rules. We dropped only unusable rows, mainly 4,612 empty reviews (0.28% of the total), corrected 318 wrong launch dates and found no duplicate IDs. Suspicious records (very short or copy-pasted text, reviews written before launch, extreme playtimes, free copies) were flagged rather than deleted, so each analysis could choose its own rule; the counts are in Appendix A. Missing values were few in the fields we use: playtime at review is missing for 12 reviews, and Steam leaves the achievement count empty when a game has none (22% of games), which we read as zero. The Metacritic score exists for only 40 games, so we did not use it. Because only about 10% of reviews are negative, plain accuracy is misleading: always guessing positive already scores about 89%.')
para("We collected only public data. Reviewer IDs are replaced by a salted hash, and names, avatars and profile links are never stored, in line with the data-minimisation and pseudonymisation principles of India's Digital Personal Data Protection Act, 2023 [6]; all reported outputs are aggregated or at game level. The data carry some known biases: only about 40% of reviews are in English, games with fewer than 10 reviews are excluded, reviewers decide for themselves whether to write a review, store attributes are an October 2026 snapshot, and the five largest games are capped.")
heading("2.3 Transformations and usable dataset", 2)
para('Cleaning is kept separate from model-specific preprocessing. The main transformations were UTC timestamps, a winsorised copy of playtime, corrected launch dates, negation handling in text (for example *not_fun*), a PCA of the tag columns (Section 4) and TF-IDF with SVD for review text. Data are split by game, 80/20, with five cross-validation folds inside the training games, so no game appears on both sides. Everything learned from data is fitted on training games only, and an automated script runs 21 leakage and consistency checks. Table 2 shows a sample of the final data and Table 3 maps the main fields to each unit.')
table([["`voted_up`", "bool: recommended or not", "Steam review field; target = not voted_up", "U1, U2, U3 M1"],
       ["`author_playtime_at_review`", "number (minutes)", "Playtime when written; refund window = under 120 min", "U1, U2, U3 M1"],
       ["`review`, `language`", "text, category", "As written; English subset flagged `text_mining_ok`", "U3 M1"],
       ["`created`", "datetime (UTC)", "Converted from Unix time; days since launch, patch recency", "U1, U2"],
       ["`price_usd`, `is_free`, `n_dlc`", "number, bool", "Store details; price tier", "U1, U2, U3 M2"],
       ["`tags`, `genres`, `categories`", "text lists", "User tags (top 20); 68 PCA components; 137 rule items", "U2, U3 M2"],
       ["`steam_total_reviews/negative`", "number", "Steam totals; negative share and game tier (outcome, never a feature)", "U1, U2 target"],
       ["`events` (patch, sale)", "event table", "Steam news, classified; 29,918 rows", "U1, extra"]],
      ["Variable", "Type / meaning", "Origin / preparation", "Used in"], [1.8, 1.4, 2.5, 0.9], "Mapping of key final fields to units and methods.", size=9)

# ---------------------------------------------------------------- 3
heading("3. Exploratory Data Analysis and Visualization (Unit 1)", 1)
gc = pd.read_parquet(ROOT / "data" / "processed" / "games_clean.parquet")
gc["neg_share"] = gc.steam_total_negative / gc.steam_total_reviews
rv_ = pd.concat([pd.read_parquet(p, columns=["author_playtime_at_review", "review"]) for p in sorted((ROOT / "data" / "processed" / "reviews_clean").glob("*.parquet"))])
def _ds(x, name, fmt):
    x = x.dropna(); return [name, f"{len(x):,}", fmt(x.mean()), fmt(x.median()), fmt(x.min()), fmt(x.max())]
desc = [_ds(gc.price_usd, "Game price (USD)", lambda v: f"{v:,.2f}"),
        _ds(gc.steam_total_reviews, "Steam reviews per game", lambda v: f"{v:,.0f}"),
        _ds(100 * gc.neg_share, "Negative share per game (%)", lambda v: f"{v:.1f}"),
        _ds(gc.n_supported_languages, "Supported languages", lambda v: f"{v:.1f}" if v % 1 else f"{v:.0f}"),
        _ds(gc.n_achievements.fillna(0), "Achievements", lambda v: f"{v:.1f}" if v % 1 else f"{v:.0f}"),
        _ds(rv_.author_playtime_at_review / 60, "Playtime at review (hours)", lambda v: f"{v:,.1f}"),
        _ds(rv_.review.str.split().str.len(), "Review length (words)", lambda v: f"{v:,.1f}" if v % 1 else f"{v:,.0f}")]
del rv_
para('The exploratory work started from the two questions that the later units depend on: when negative reviews appear (Q1) and how games differ at launch (Q3). Table 4 summarises the main variables, and the figures were made in Python with Matplotlib [20] and Seaborn [21]. Most of the variables are strongly skewed. The median game has 50 Steam reviews, but the largest has more than 266,000; the median review is five words long and is written after about nine hours of play. We therefore report medians and shares rather than means, and we use logarithms or bands for counts.')
table(desc, ["Variable", "n", "Mean", "Median", "Min", "Max"], [2.4, 1.0, 0.8, 0.8, 0.7, 0.9], "Descriptive statistics of the main game- and review-level variables (cleaned data).", size=9.5, align_right_from=1)
para('About 10.4% of reviews are negative, but the share depends strongly on playtime. Reviews written in the first two hours are 21.2% negative, against roughly 8 to 10% later (Figure 2, left), and they make up 29% of all negative reviews; the same pattern appears in 94% of the games with enough reviews. Inside this window, negativity also rises with price, from about 11% for games under $5 to 23% at $5 to $10 and 32% at $20 or more (Figure 2, right). At game level, 14.5% of games are Struggling and 35% Strong.')
figure(FIG / "findings" / "refund_window.png", "Negative-review share by playtime at review and by price inside the 2-hour window (Python/Matplotlib; `src/analysis`, notebook `02_EDA_Visualizations.ipynb`).", width=5.0)
para("Store attributes shift a game's risk by a few points each, but none decides it on its own (Figure 14, Appendix B). Games with more languages, DLC or achievements are less often Struggling; for example, 12.5% of games that list Steam Achievements as a store feature are Struggling, against 19.6% of those that do not. Review counts, owner estimates and refund shares are produced by a game's reception rather than known at launch, so we kept them out of the models.")
para("These observations shaped Unit 2. Since only about 10% of reviews and 14.5% of games fall in the class we care most about, a model could reach high accuracy by ignoring them, so we evaluate with PR-AUC and macro-F1. Reviews of the same game share vocabulary and settings, so a random split would let a model recognise games instead of learning anything general; we split by game. Finally, the playtime pattern suggested that the review text itself carries the complaint, which led to the separate text model in Section 5.1.")
# ---------------------------------------------------------------- 4
heading("4. Feature Engineering and Predictive Modelling (Unit 2)", 1)
para('At review level we predict whether a review is negative from playtime, refund-window and launch-week flags, days since launch, recent patches and sales, price and text length. At game level we predict the tier from 83 features that a studio knows at launch: price, launch year, DLC, achievements, platforms, controller support and 68 tag components. Columns that describe the outcome are never used as features. The tags needed compressing: after rare tags were dropped, 234 tag columns remained for 1,191 training games, most of them sparse and many strongly related (Pixel Graphics and Retro often appear together). A PCA fitted on the training games had its elbow at 68 components, which keep 75% of the tag variance while reducing the number of columns by more than two-thirds.')
para('Each task has a Dummy baseline and four models: Logistic Regression, Random Forest, LightGBM and XGBoost [7, 8, 9]. All models share the same split and folds. We tuned each by grid search on the folds, keeping the default setting as one of the candidates (Appendix B, Table 19), and chose thresholds and class scales from out-of-fold predictions; the test set was scored once. For leakage, we checked that preprocessing is fitted on training folds only, that outcome columns are excluded, and that no game appears on both sides of the split; 21 automated checks cover these points and all pass.')
rv = {m: cr[cr.Model == m].iloc[0] for m in cr.Model}; gv = {m: cg[cg.Model == m].iloc[0] for m in cg.Model}
rows = []
for m in ["Dummy (prior)", "Logistic Regression", "Random Forest", "LightGBM", "XGBoost"]:
    a, g = rv[m], gv[m]
    rows.append([m, f"{a['CV mean']:.3f} ± {a['CV sd']:.3f}", f"{a.Test:.3f}", f"{a['Test ROC-AUC']:.3f}", f"{g['CV mean']:.3f} ± {g['CV sd']:.3f}", f"{g.Test:.3f} [{g['CI low']:.2f}, {g['CI high']:.2f}]", f"{g['Recall Struggling']:.2f}"])
table(rows, ["Model", "Review CV PR-AUC", "Test PR-AUC", "Test ROC-AUC", "Game CV macro-F1", "Test macro-F1 [95% CI]", "Recall Struggling"], [1.35, 1.0, 0.65, 0.7, 1.0, 1.1, 0.7],
      "Model comparison on the shared held-out test games (metadata only; guessing PR-AUC 0.106, macro-F1 0.22). Validation: 5-fold CV by game; test scored once.", size=9, align_right_from=1)
para('At review level (Table 5, Figure 3) the three tree models are roughly tied at a test PR-AUC of 0.25 to 0.28, about 2.5 times guessing, while Logistic Regression is weaker (0.19). At game level every model beats guessing, but the intervals overlap because the test set has only 299 games. Tuning helped XGBoost (0.437) and Logistic Regression (0.423) on test, while Random Forest and LightGBM fell back to about 0.37 even though their cross-validation scores improved. With only 1,191 training games, the grid search probably fitted noise in the validation folds for these two models, so we kept their test results as they are. Struggling games are the hardest class to find (Figure 4). We selected XGBoost at game level for its test score and Struggling recall, although it is not clearly better than Logistic Regression.')
figure(FIG / "models" / "model_comparison.png", "Cross-validation and test scores for all models at review level, game level and with review text (Python/Matplotlib; `python -m src.models.compare`).", width=5.9)
figure_pair(FIG / "models" / "game_xgb_confusion.png", "Game-level XGBoost confusion matrices on the 299 test games (default and tuned class scales).",
            FIG / "models" / "game_xgb_shap_bar.png", "SHAP importance of the game-level XGBoost model (top features).", width=2.9)
para('We explain the tree models with SHAP values [15]. At review level, the model leans on review length, Chinese-language reviews and playtime; at game level, on a few tag components and the number of achievements (Figure 5). These are associations, not causes, and the review-level models use reviewer history collected at scrape time, so they explain reviews rather than predict a launch. The results are not driven by a few large games: the within-game ROC-AUC (0.75) is slightly above the pooled value.')
gsub = rd("game_subgroup_check.csv"); tsub = rd("reviewtext_subgroup_check.csv")
def _g(grp, lv): return gsub[(gsub.group == grp) & (gsub.level == lv)].iloc[0]
def _t(grp, lv): return tsub[(tsub.group == grp) & (tsub.level == lv)].iloc[0]
para(f"To check for uneven errors, we sliced the test predictions by subgroup (Appendix B, Table 20). For the deployed game-level scorer (macro-F1 {_g('all', 'all test games').macro_f1:.2f} overall), macro-F1 ranges from {_g('early access', 'Early Access').macro_f1:.2f} for Early Access launches and {_g('year', '2022-2023').macro_f1:.2f} for 2022 to 2023 games to {_g('year', '2024-2025').macro_f1:.2f} for 2024 to 2025 games, and it finds few Struggling games among those with 100 or more reviews. These groups hold only 37 to 239 games, so the differences are uncertain, but the scorer should be used with more caution for Early Access plans. "
     f"The text detector is consistent across game price bands (PR-AUC {tsub[tsub.group == 'price'].pr_auc.min():.2f} to {tsub[tsub.group == 'price'].pr_auc.max():.2f}) and works best inside the refund window ({_t('window', 'inside 2 h').pr_auc:.2f}). It is weakest on very short reviews (under 10 words: PR-AUC {_t('length', 'under 10 words').pr_auc:.2f}, recall {_t('length', 'under 10 words').recall:.2f}) and raises more false alarms on free games (precision {_t('price', 'free').precision:.2f}), so short reviews and free games need a manual look. The text model covers English only, so we could not compare languages.")
# ---------------------------------------------------------------- 5
heading("5. Advanced Analytics (Unit 3): Two Selected Methods", 1)
para('The EDA left two questions that the Unit 2 models could not answer. Negative reviews cluster in the first hours, but the playtime and store fields do not say what players are unhappy about; that information is in the review text, so we chose text mining. Store data also predicted the game tier only modestly, and a launch plan is a combination of tags, price and Early Access status rather than a single feature, so we chose association rule mining to look for combinations. We did not choose time-series analysis because most games have too few reviews per week for stable series; patch impact is reported as an extra analysis in Section 6.1.')
heading("5.1 Method 1: Text mining", 2)
para('This method asks what players complain about, and whether the review text can flag a negative review. We used 557,202 English reviews (436,715 for training, 120,487 for testing). The text was cleaned and negations were joined to the next word, so that *not fun* becomes one token instead of counting as *fun*. We represented each review with TF-IDF [10], which gives more weight to words that are frequent in a review but rare across reviews; it suits short, informal text and works well with linear models. The vocabulary and weights were learned from training reviews only.')
para('We built two things on this representation. The first is a detector: a Logistic Regression on TF-IDF plus the metadata features, tuned by grid search, with its threshold set on out-of-fold predictions. To see how much of the text signal survives compression, we also trained a model on 78 SVD components of the text (the elbow of the variance curve). The second is a topic model, NMF [11] with eight topics on 40,000 negative training reviews, compared with LDA [12]. We fixed eight topics so that each could be read and labelled by hand; we did not search formally over other numbers of topics.')
trows = []
for k in ("dummy", "meta", "meta_svd", "tfidf_lr"):
    r = ct[ct.key == k].iloc[0]; trows.append([r.Model, f"{r['CV mean']:.3f} ± {r['CV sd']:.3f}", f"{r.Test:.3f}", f"{r['Test ROC-AUC']:.3f}", f"{r.Precision:.2f} / {r.Recall:.2f}", f"{r.F1:.2f}"])
table(trows, ["Approach", "CV PR-AUC", "Test PR-AUC", "Test ROC-AUC", "Precision / recall", "F1"], [2.6, 1.0, 0.8, 0.8, 0.9, 0.5],
      "Text evaluation on 120,487 held-out English reviews (guessing PR-AUC 0.115).", size=9, align_right_from=1)
para(f'Compressed text features lift PR-AUC from 0.36 to 0.59, and the full TF-IDF model reaches 0.84 (ROC-AUC 0.97). On the test reviews it catches {int(det.tp):,} negative reviews and misses {int(det.fn):,}, with {int(det.fp):,} false alarms (Figure 6); accuracy is {det.accuracy:.1%} against {det.baseline_accuracy_always_positive:.1%} for always guessing positive. For the topics, NMF scored lower than LDA on UMass coherence (-2.38 against -1.73). UMass rewards words that often appear together, and on short reviews LDA reached a higher score by repeating the same general gaming words across several topics. NMF topics overlapped less (90% of their top words were unique, against 86% for LDA), and when we refitted NMF on 80% resamples, the top words of most topics stayed similar (mean Jaccard 0.75). We therefore used NMF because its topics were easier to tell apart. Early negative reviews lean towards crashes, multiplayer bugs and controls, later ones towards boss design and balance (Figure 7); a word such as *boring* appears in 9.1% of negative reviews against 0.8% of positive ones.')
figure_pair(FIG / "models" / "review_text_confusion.png", "Confusion matrix of the text detector on the held-out reviews (row-normalised).",
            FIG / "findings" / "complaint_themes.png", "Complaint themes among negative reviews by playtime (NMF, eight topics).", width=2.9)
para("In a blind check of 200 reviews, labelled by an AI assistant without seeing the model's topic, the labels matched the model on 35.5% (kappa 0.30). The Early Access, repetition and crash topics were recovered well; the boss and balance topics were not, so we treat those as exploratory. An earlier figure of 96% was withdrawn because those labels were not independent; a human re-label would make the check stronger. The detector identifies complaints in existing reviews; it does not forecast them.")
heading("5.2 Method 2: Association rule mining", 2)
para('This method asks which tag, price and Early Access combinations go with the Strong tier. Each of the 1,191 training games became a transaction of up to 137 possible items: its user tags, a price band and Early Access. We left out opinion tags such as *Great Soundtrack*, because players add them after playing and they would make the rules circular. We mined frequent itemsets with FP-Growth [13] using mlxtend [16], keeping itemsets of up to three items that appear in at least 25 games (about 2% of the training games). Below that support, a rule rests on so few games that it rarely holds on new ones (Table 7).')
para('Three measures describe a rule. Take the strongest rule, *Pixel Graphics + Retro + Shooter leads to Strong*. Its support is the 25 training games that have all three tags. Its confidence is the share of those games that are Strong, 20 of 25 or 80%. Its lift compares that with the base rate: 80% against 35% of all training games, a lift of 2.26. From 4,715 candidate rules we kept those with a lift of at least 1.2 that also passed a false-discovery control (Benjamini-Hochberg, q below 0.10) [14], which removes rules likely to appear by chance among so many tests. For the held-out check, we took the test games that contain all of a rule\'s items. A rule holds if their Strong share is still above the test base rate (lift above 1) and its training confidence lies inside the 95% interval of the test confidence. Rules that match fewer than 10 test games could not be tested.')
crow = []
nm = {"FP-Growth, min support 25 games, up to 3 items (chosen)": "FP-Growth, support 25, up to 3 items (chosen)", "Apriori, same thresholds": "Apriori, same thresholds",
      "FP-Growth, min support 15 games (looser)": "FP-Growth, support 15 (looser)", "FP-Growth, min support 40 games (stricter)": "FP-Growth, support 40 (stricter)",
      "FP-Growth, pairs only (max 2 items)": "FP-Growth, pairs only", "FP-Growth, keeping evaluative tags": "FP-Growth, keeping after-play tags"}
for r in rc.itertuples():
    crow.append([nm[r.configuration], f"{int(r.strong_rules)}", f"{r.mean_confidence:.2f} / {r.mean_lift:.2f}", f"{r.share_holding_on_test:.0%} of {int(r.rules_with_enough_test_games)}"])
table(crow, ["Configuration", "Strong rules (FDR)", "Mean confidence / lift", "Hold on test games"], [3.0, 1.1, 1.3, 1.1],
      "Association-rule configuration comparison (same 1,191 training and 299 test games for every row).", size=9, align_right_from=1)
para(f"Of {rules_s['candidate_rules']:,} candidate rules, {rules_s['strong_rules_count']} pass for the Strong tier and none for Struggling: under these thresholds, no tag combination is clearly associated with a struggling launch. About two-thirds of the rules that could be tested (45 of 66) still hold on unseen games (Figure 8). FP-Growth and Apriori return the same rules, and a lower support finds more rules that hold less often (Table 7).")
figure(make_rules_figure(), "The eight Strong rules with the highest lift: share of Strong games on training and test games (Python/Matplotlib; `reports/results/association_rules_strong.csv`).", width=4.3)
para('Each rule rests on only 25 to 60 training games and a handful of test games, and rules involving a price under $5 are often unstable. They are hypotheses to test, not recipes.')
# ---------------------------------------------------------------- 6
heading("6. Comparative Analysis and Business Impact", 1)
heading("6.1 Comparison of the two Unit 3 methods", 2)
table([["Question", "What do players complain about; can text flag a negative review?", "Which launch combinations go with Strong reception?"],
       ["Input and output", "557,202 English reviews; detector scores and eight themes", "1,191 training games as itemsets; 95 Strong rules"],
       ["Validation", "Held-out reviews from unseen games; PR-AUC, ROC-AUC; topic diversity and stability; blind check", "FDR control; each rule checked once on 299 test games; stability"],
       ["Strengths", "Strong, well-validated detector (ROC-AUC 0.97); themes explain complaints", "Transparent rules a studio can read; launch-time information only"],
       ["Limits", "Detects, does not forecast; weak topic agreement (35.5%); English only", "Small supports; no Struggling rule; associations only"]],
      ["Criterion", "Method 1: text mining", "Method 2: association rules"], [1.2, 2.7, 2.6], "Comparison of the two selected Unit 3 methods (their metrics are not interchangeable).", size=9)
para('The two methods answer different questions, so their metrics cannot be compared directly: PR-AUC measures how well the detector separates negative from positive reviews, while lift and the held-out share measure how far a tag combination raises the chance of a Strong game. Text mining describes the complaints players make and when they make them; the rules describe launch choices that are associated with good reception. The findings line up across units: complaints cluster in the first two hours and at higher prices (Unit 1; crashes and controls in the text), while store data and tags only partly separate good launches from weak ones (Unit 2 and the rules). As an extra analysis, positive reviews fall by about 2 points after a patch, but matched games without a patch fall by almost the same amount (difference +0.27 points, interval -0.82 to +1.36). We therefore find no evidence of a patch effect of its own; the drop is consistent with ratings settling after launch (Appendix B, Figure 13).')
heading("6.2 Comparison with existing studies", 2)
table([["Data and setting", "About 11M reviews of 6,224 Steam games (2016 snapshot), whole platform", "Steam games; popularity as player numbers", "2.8M Brazilian-Portuguese Steam reviews", "1,861 indie roguelikes, 1.65M reviews, Oct 2026"],
       ["Methods and validation", "Descriptive analysis; no model scored on unseen games", "Bayesian hierarchical models; coefficients", "Text classifiers for review helpfulness", "Models scored once on held-out games; baselines; FDR-controlled rules; topics"],
       ["Comparable findings", "Negative reviews come early (we: 21.2% inside 2 h); design criticised more than bugs (we: crashes early, balance later)", "Price, languages and release date matter for popularity; our tier macro-F1 is modest (0.42 to 0.44)", "Review text can be modelled; ours detects negativity (PR-AUC 0.84), a different target", "Adds refund window, patch timing, text and tags in one niche"],
       ["Why not numeric", "No predictive metric", "Different target (players, not tiers)", "Helpfulness, other language", "Gaps: causal claims, revenue, other genres"]],
      ["Criterion", "Lin et al. [1]", "De Luisa et al. [2]", "SteamBR [3]", "Current project"], [1.0, 1.5, 1.35, 1.35, 1.5], "Comparison with existing studies (direct numerical comparison is not valid: different data, targets and protocols).", size=9)
para('None of the three studies can be compared with ours on a single number, because each uses different data, a different target and a different evaluation. The closest is Lin et al., whose finding that negative reviews arrive early matches our refund-window result in direction. SteamBR shows that Steam review text can be modelled reliably with text classifiers, which supports our text-based approach, although it predicts helpfulness in Portuguese rather than negativity in English. Our contribution is a whole niche collected from scratch, the refund-window view, held-out testing that also reports weak results, and an application a studio can use.')
heading("6.3 KPI-based business impact and scenarios", 2)
HOURS, RATE = 40, 30
para(f'We turned the refund-window finding into an illustrative scenario for one game. Three numbers are measured: about 14% of reviews are written in the first two hours, 21.2% of those are negative, and the figure is 11.4% for the cheapest games. The rest is assumed: a game with 1,000 reviews; a first-hour fix (blind testing, crash and control fixes) taking about {HOURS} person-hours at an illustrative ${RATE} per hour, or about ${HOURS * RATE:,}; and a drop in early negativity of 0, 5 or 9.8 points (the last matches the cheapest tier). Sales and refunds are not public, so we cannot put a money value on a review. Instead, the table shows what each avoided negative review would cost, which a studio can compare with what a review is worth to it.')
rows = []
for name, d in (("Worst", 0.0), ("Base", 0.05), ("Best", NEG_EARLY - NEG_CHEAP)):
    av = {s: 1000 * s * d for s in (0.10, SHARE_EARLY, 0.20)}
    new = (1000 * NEG_ALL - av[SHARE_EARLY]) / 10
    cost = "no benefit" if av[SHARE_EARLY] == 0 else f"${HOURS * RATE / av[SHARE_EARLY]:,.0f}"
    rows.append([name, f"-{100 * d:.1f} points", f"{av[0.10]:.1f} / {av[SHARE_EARLY]:.1f} / {av[0.20]:.1f}", f"{NEG_ALL * 100:.1f}% to {new:.1f}%", cost])
table(rows, ["Scenario", "Change in early negativity", "Avoided negatives per 1,000 reviews (window share 10% / 14% measured / 20%)", "Overall negative share (14%)", f"Cost per avoided negative (${HOURS * RATE:,})"],
      [0.8, 1.1, 2.3, 1.3, 1.1], "Illustrative scenarios for a first-hour fix (hypothetical effect sizes and cost; sensitivity to the share of reviews written inside the window).", size=9)
base_cost = HOURS * RATE / (1000 * SHARE_EARLY * 0.05)
para(f'The gain is modest: about 7 fewer negative reviews per 1,000 in the base case, at roughly ${base_cost:,.0f} each. The net impact is positive only if a negative review avoided early is worth more than that to the studio in kept sales and rating, which our data cannot show. The fix pays off more easily for games that expect many reviews, because the cost is fixed while the benefit grows with the number of reviews. The estimate also depends on how much a fix achieves and how many reviews come early. For patches, we found no general effect, so patching only to lift sentiment is not supported by our data; patching specific crashes and controls still makes sense. Section 9 ranks the recommendations.')
# ---------------------------------------------------------------- 7
heading("7. Interactive Application, Figure Index and Reproducibility", 1)
heading("7.1 Mandatory dashboard", 2)
para('The dashboard is public at https://indieguard.streamlit.app. The free host sleeps when idle, so it should be opened a few minutes before a demo. It can also be run locally with `streamlit run src/dashboard/app.py`. It is built with Streamlit [17], pandas [18] and Altair [19], and reads small tables exported from the analysis and one saved XGBoost model. Unit 1 summaries and the Unit 3 rule, topic and patch results are precomputed; filters, charts and the launch-risk scorer are computed live. It has five pages: market explorer, tag rules and complaints, patch impact, launch-risk scorer and an about page.')
c1 = crop(FIG / "dashboard" / "dash_1_market_default.png", (0, 40, 1465, 640), "dash_market_default.png")
c2 = crop(FIG / "dashboard" / "dash_2_market_filtered_pixel.png", (0, 40, 1465, 640), "dash_market_filtered.png")
figure_pair(c1, "Market explorer, default state (1,490 games).", c2, "Market explorer after selecting the tag Pixel Graphics (550 games; KPIs and charts update).", width=3.05)
c3 = crop(FIG / "dashboard" / "dash_3_rules_default.png", (300, 90, 1465, 760), "dash_rules.png")
c4 = crop(FIG / "dashboard" / "dash_8_scorer_explained_weak.png", (380, 700, 1470, 1750), "dash_scorer.png")
figure_pair(c3, "Tag rules page filtered to stable rules with lift at least 1.5 (37 of 95 rules).", c4, "Scorer explanation for a game predicted Struggling: reasons for and against, matched rules, and single changes that would help.", width=3.05)
para('At Review 2 the panel asked that the scorer explain its output, so we added an explanation step. After scoring, the dashboard shows why a game looks strong or weak. Each launch choice (achievements, controller support, a tag and so on) is replaced by the opposite choice, for example no achievements instead of 30, and the game is scored again; the page then lists the choices that lower and those that raise the risk of a Struggling launch. It also lists the tag rules the game matches and suggests single changes that would most improve the outlook (Figure 12). For testing, we confirmed that the saved feature pipeline reproduces the stored features for all 1,861 games, and opened every page both in the deployed app and in an automated browser run. The main limits are that the scores are class-scaled, not probabilities (about 0.43 macro-F1), the explanations are associations, and the data are an October 2026 snapshot.')
heading("7.2 Figure index and reproduction map", 2)
figs = [["Fig. 1", "End-to-end workflow, Matplotlib", "1.1: how units connect", "`build_capstone_report.py`"],
        ["Fig. 2", "Refund window and price, Matplotlib", "3: core Unit 1 finding", "`figures/findings/refund_window.png`; notebook 02"],
        ["Fig. 3", "Model comparison, Matplotlib", "4: models vs baselines", "`python -m src.models.compare`"],
        ["Fig. 4, 5", "Game confusion; SHAP bars", "4: errors, interpretation", "`figures/models/game_xgb_*.png`"],
        ["Fig. 6, 7", "Text confusion; complaint themes", "5.1: detector and themes", "`review_text_detail.py`; notebook 03"],
        ["Fig. 8", "Top rules train vs test, Matplotlib", "5.2: held-out check", "`association_rules_strong.csv`"],
        ["Fig. 9 to 12", "Dashboard screenshots", "7.1: interactive states", "`figures/dashboard/`"],
        ["Fig. 13, 14", "Patch impact; risk by game features", "App. B: 6.1 extra; 3 store signals", "`patch_impact.ipynb`; `figures/eda/10_*.png`"]]
table(figs, ["Figure", "Title / tool", "Section / purpose", "Source notebook / file"], [0.8, 1.9, 1.7, 2.1], "Figure index and reproduction map (figures are embedded once, in Sections 1 to 7 and Appendix B).", size=9)
heading("7.3 Code and environment", 2)
para("The code is in a public repository, https://github.com/Shan713/IndieGuard. Its README describes the folder layout (`src`, `notebooks`, `data`, `figures`, `reports`, `docs`, `models`, `scripts`). The project runs on Python 3.12, with the package versions pinned in `requirements.txt` and installed with `pip install -r requirements.txt`. The commands that rebuild the features, models, tuning searches, report and dashboard are listed in Appendix C. No secrets are stored; the raw scrape and the hashing salt are left out, and the collection scripts can regenerate the data from Steam's public endpoints within Steam's terms.")
# ---------------------------------------------------------------- 8
heading("8. End-to-End Validation, Traceability and Contributions", 1)
heading("8.1 Completion and validity", 2)
para('Everything in the declared scope is implemented and can be shown: data collection and cleaning, EDA, five model families at review and game level with tuning, two Unit 3 methods, the deployed dashboard, this report and the presentation. The main threat to internal validity is the small number of games available for tuning: with 1,191 training games, the grid search appears to have overfitted the validation folds for two models, and the topic check relied on a single AI annotator. External validity is limited because the data cover one niche at one point in time, only about 40% of reviews are in English, and there is no revenue data, so the results may not carry over to other genres. On construct validity, the review models use reviewer history collected at scrape time, and the game tier is a rule on the negative share rather than a business outcome. Results are reproducible from the scripts, although game-level scores move by about 0.01 between runs. The subgroup checks covered the game scorer and the English text detector only, and discounts were not analysed.')
table([["O1 Refund window", "Section 3, EDA", "21.2% vs 8.8% negative; 29% of negatives", "Fig. 2", "R1, R2"],
       ["O2 Detect negative reviews", "Sections 4, 5.1", "PR-AUC 0.838; ROC-AUC 0.971 (guessing 0.115)", "Fig. 3, 6; Table 6", "R1"],
       ["O3 Predict game tier", "Section 4", "Macro-F1 0.42 to 0.44 vs 0.22", "Fig. 3, 4; Table 5", "R4"],
       ["O4 Tag combinations", "Section 5.2", "95 Strong rules; 68% of 66 testable hold; none for Struggling", "Fig. 8; Table 7", "R3"],
       ["O5 Complaint themes", "Section 5.1", "NMF diversity 90.0%; blind check 35.5%", "Fig. 7", "R1, R5"],
       ["O6 Patch impact", "Section 6.1", "+0.27 points [-0.82, +1.36] vs controls", "Fig. 13", "R5"],
       ["O7 Application", "Section 7", "Public dashboard, 5 pages, filters and scorer", "Fig. 9 to 12", "R1 to R5 (exploration)"]],
      ["Objective / KPI", "Analysis / method", "Result / metric", "Figure / artifact", "Recommendation"], [1.4, 1.1, 2.2, 1.0, 0.9], "Objective-to-evidence traceability matrix.", size=9)
heading("8.2 Project tracking and individual work", 2)
para('Work was tracked on a GitHub Projects board (https://github.com/users/Shan713/projects/1, private, access on request) with the stages Backlog, To Do, In Progress, Review/Testing and Completed; each card has an owner, a due date and evidence. Commits and pull requests are public in the repository. Individual marks may vary with participation and the explanation of assigned work.')
table([["Thotakura Srihitha / CB.SC.U4CSE23450", "Exploratory analysis (29 figures, association tests, tag combinations); dashboard wireframe and downloads; grid-search tuning of the game-level models and review Logistic Regression", "2 to 17 Oct 2026", "PRs #1, #101, #102; notebook 02; cards #20, #21, #37, #40, #41, #86 to #89, #91, #93"],
       ["Gayas Sheik / CB.SC.U4CSE23118", "Feature engineering and tag PCA; patch impact with matched controls; launch-risk scorer; reproducibility", "2 to 14 Oct 2026", "PRs #2, #49, #50, #69, #71, #80; cards #4, #18, #19, #38, #39, #42"],
       ["Shantharam P / CB.SC.U4CSE23146", "Data collection and cleaning; split and targets; grid-search tuning of the game-level models and review Logistic Regression; dashboard and deployment", "1 to 17 Oct 2026", "`src/collect`, `src/clean`; PRs #57, #95, #96, #102; cards #5 to #17, #23 to #27, #43, #83 to #94"],
       ["Pradeep Sasidharan / CB.SC.U4CSE23232", "Model development: review- and game-level models (Logistic Regression, Random Forest, XGBoost) with SHAP; model comparison against baselines; association rule mining", "3 to 11 Oct 2026", "`src/models`; PRs #55, #70; cards #34, #35, #36"],
       ["Valikala Tejaswini / CB.SC.U4CSE23752", "Text mining and topic modelling; combined business insights; blind topic check", "10 to 16 Oct 2026", "PRs #66, #81, #99; cards #31, #32, #33, #44"]],
      ["Team member / register no.", "Assigned work", "Period", "Evidence link / artifact"], [1.5, 2.6, 0.85, 1.75], "Individual contributions, dates and evidence (dates from the pull-request history).", size=9)
heading("8.3 AI use and originality declarations", 2)
para('We list every generative-AI tool we used. The tools assisted the team; they did not replace it. The research question, scope, data collection runs, analytical choices and conclusions were decided by team members, and each member wrote and checked their own modules. By signing the declaration page, each member confirms that this list is complete for their own work.')
table([["ChatGPT (OpenAI)", "Ideation: brainstorming project topics, research questions and feature ideas in the planning stage.", "Early ideas and the shortlist of topics and questions. No data, code or reported results.",
        "The team discussed the ideas, checked feasibility against the available public data, and wrote and agreed the problem statement and KPIs. Nothing from the tool is reported as a result."],
       ["Antigravity (AI coding assistant)", "Minor coding assistance: small code completions and fixes in project scripts.", "Small, scattered completions and fixes; no complete module or analysis.",
        "Each change was read, run and tested by the team member who made it; outputs were checked against expected values and the automated checks (Section 8.1)."],
       ["Claude (Anthropic)", "Coding and debugging help under the team's direction (data collection, modelling, dashboard and report scripts); drafting documentation, this report and the presentation deck; labelling the blind topic-check sample.",
        "Parts of the project scripts, documentation, slides and this report.",
        "The team ran, checked and corrected all outputs: automated checks, results regenerated from scripts, pull-request review, and slides and figures checked against the data. Errors found and fixed included a text-processing bug and an overstated topic-agreement figure, which was withdrawn."]],
      ["Tool", "Tasks supported", "Content / code affected", "Human verification and corrections"], [1.0, 1.8, 1.8, 2.0], "Generative-AI disclosure.", size=9)
para("No AI tool collected data on its own, created or altered records, or decided the conclusions; every figure in this report is produced by code from the project data. Apart from the cited sources, the ideas, analysis and writing are the team's own, and borrowed ideas, figures and code are cited.")
# ---------------------------------------------------------------- 9
heading("9. Conclusion, Recommendations and Roadmap", 1)
para('The traceability matrix (Table 12) shows how each objective was met. We quantified the refund-window effect, built a text detector and complaint themes, predicted game tiers modestly, found and tested tag-combination rules, measured patch impact against controls, and built a public dashboard. Three results were weaker than we hoped: game-tier prediction from store data is modest, no rule was found for struggling games, and patches showed no effect. These still matter to a studio, because they show where store data and patching alone are not enough.')
table([["R1. Treat the first hour as a standalone product: blind-test the first 30 minutes, check controls and camera, enable crash reporting on launch day", "Early negative share (21.2%)", "Studio / QA lead", "High impact, high feasibility"],
       ["R2. Match the price to what the first hour delivers: early negativity doubles from under $5 to $5 to $10, so a higher price needs a first hour that testers rate well", "Early negativity by price band (11.4% to 31.8%)", "Studio / publisher", "Medium impact, high feasibility (test it)"],
       ["R3. Test a focused 2D, pixel, retro or procedural scope before committing to multiplayer or 3D", "Share of Strong games (about 2/3 vs 35%)", "Design lead", "Medium impact, medium feasibility"],
       ["R4. Complete the store page (achievements, controller support)", "Share of Struggling games (12.5% vs 19.6%)", "Marketing / producer", "Low to medium impact, high feasibility"],
       ["R5. Patch for specific problems (crashes, controls, balance), not to lift sentiment", "Change in positive share (+0.27 points vs controls)", "Engineering", "Medium impact, high feasibility"]],
      ["Recommendation (hypothesis)", "KPI effect", "Stakeholder", "Priority: impact and feasibility"], [3.0, 1.4, 1.0, 1.1], "Prioritised recommendations.", size=9)
para("The roadmap below turns the recommendations into steps, each with an owner, the resources it needs and a check that shows whether it worked.", keep=True)
table([["Now (first month)", "Apply R1 to the next build; run the scorer on the planned scope; small price test for R2", "Studio QA lead, producer", "About 40 person-hours; 10 to 20 blind testers; the dashboard", "Early negative share below 21% in the first launch month"],
       ["Next (1 to 6 months)", "Refresh the Steam data; hand-label 200 reviews; re-check R3 and R4 on new releases", "Analytics team", "Scrapers (about 2 days of runtime); 2 people for labelling", "Topic agreement above 35.5%; at least 2/3 of rules still hold"],
       ["Later (6 to 12 months)", "Calibrate the scorer; add discount analysis and other genres", "Analytics team", "New data collection; model retraining", "Scorer macro-F1 at or above 0.43 on new games; calibrated probabilities"]],
      ["Stage", "Steps", "Owner", "Resources", "Success check"], [1.0, 2.1, 1.0, 1.3, 1.3], "Implementation roadmap with resources and success checks.", size=9)
para("Several extensions were outside the scope of this project: the impact of discounts, causal designs such as experiments, revenue analysis, a language comparison for the review models and a human-labelled topic check.")
# ============================================================================== references
heading("References", 1, page_break=True)
refs = [
    "Dayi Lin, Cor-Paul Bezemer, Ying Zou, and Ahmed E. Hassan. 2019. An empirical study of game reviews on the Steam platform. *Empirical Software Engineering* 24, 1 (2019), 170-207. https://doi.org/10.1007/s10664-018-9627-4",
    "Andraž De Luisa, Jan Hartman, David Nabergoj, Samo Pahor, Marko Rus, Bozhidar Stevanoski, Jure Demšar, and Erik Štrumbelj. 2021. Predicting the popularity of games on Steam. *Elektrotehniški vestnik* 88, 4 (2021), 151-162. arXiv:2110.02896.",
    "Gabriel A. Z. Jorge and Thiago A. S. Pardo. 2023. SteamBR: a dataset for game reviews and evaluation of a state-of-the-art method for helpfulness prediction. In *Proceedings of the Brazilian Workshop on Social Network Analysis and Mining (BraSNAM)*.",
    "Valve Corporation. Steam Refunds. Steam Support. https://store.steampowered.com/steam_refunds/ (accessed October 2026); and User Reviews, Steamworks documentation. https://partner.steamgames.com/doc/store/reviews (accessed October 2026).",
    "SteamSpy. Owner estimates and game statistics. https://steamspy.com (accessed October 2026).",
    "Government of India. 2023. The Digital Personal Data Protection Act, 2023 (No. 22 of 2023).",
    "Fabian Pedregosa et al. 2011. Scikit-learn: machine learning in Python. *Journal of Machine Learning Research* 12 (2011), 2825-2830.",
    "Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. 2017. LightGBM: a highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30*.",
    "Tianqi Chen and Carlos Guestrin. 2016. XGBoost: a scalable tree boosting system. In *Proceedings of KDD '16*. ACM, 785-794. https://doi.org/10.1145/2939672.2939785",
    "Gerard Salton and Christopher Buckley. 1988. Term-weighting approaches in automatic text retrieval. *Information Processing & Management* 24, 5 (1988), 513-523.",
    "Daniel D. Lee and H. Sebastian Seung. 1999. Learning the parts of objects by non-negative matrix factorization. *Nature* 401 (1999), 788-791.",
    "David M. Blei, Andrew Y. Ng, and Michael I. Jordan. 2003. Latent Dirichlet allocation. *Journal of Machine Learning Research* 3 (2003), 993-1022.",
    "Jiawei Han, Jian Pei, and Yiwen Yin. 2000. Mining frequent patterns without candidate generation. In *Proceedings of SIGMOD '00*. ACM, 1-12. https://doi.org/10.1145/335191.335372",
    "Yoav Benjamini and Yosef Hochberg. 1995. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B* 57, 1 (1995), 289-300.",
    "Scott M. Lundberg and Su-In Lee. 2017. A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30*.",
    "Sebastian Raschka. 2018. MLxtend: providing machine learning and data science utilities and extensions to Python's scientific computing stack. *Journal of Open Source Software* 3, 24 (2018), 638.",
    "Streamlit Inc. Streamlit: the fastest way to build data apps. https://streamlit.io (version 1.66, accessed October 2026).",
    "Wes McKinney. 2010. Data structures for statistical computing in Python. In *Proceedings of the 9th Python in Science Conference (SciPy 2010)*, 56-61. https://doi.org/10.25080/Majora-92bf1922-00a",
    "Jacob VanderPlas, Brian E. Granger, Jeffrey Heer, Dominik Moritz, Kanit Wongsuphasawat, Arvind Satyanarayan, Eitan Lees, Ilia Timofeev, Ben Welsh, and Scott Sievert. 2018. Altair: interactive statistical visualizations for Python. *Journal of Open Source Software* 3, 32 (2018), 1057. https://doi.org/10.21105/joss.01057",
    "John D. Hunter. 2007. Matplotlib: a 2D graphics environment. *Computing in Science & Engineering* 9, 3 (2007), 90-95. https://doi.org/10.1109/MCSE.2007.55",
    "Michael L. Waskom. 2021. seaborn: statistical data visualization. *Journal of Open Source Software* 6, 60 (2021), 3021. https://doi.org/10.21105/joss.03021",
]
for i, r in enumerate(refs, start=1):
    p = doc.add_paragraph(); p.paragraph_format.left_indent = Inches(0.4); p.paragraph_format.first_line_indent = Inches(-0.4); p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.line_spacing = 1.15; add_runs(p, f"[{i}] {r}", 11)
para("In-text citation numbers follow ACM numeric style.", size=10)

# ============================================================================== appendices
heading("Appendix A. Extended data dictionary and validation records", 1, page_break=True)
para("Table 17 lists the main fields of the three cleaned tables. The full dictionary (all 34 review columns, 58 game columns and the event table) is in `docs/data_documentation.md`, section 5, and the feature catalogue is `docs/feature_engineering/feature_catalog.csv`.", size=11)
table([["Review", "`recommendationid`", "int", "Steam's review ID (primary key)"], ["Review", "`appid`", "int", "Game ID; joins to the game table"],
       ["Review", "`voted_up`", "bool", "Recommended (True) or not; the review target is not voted_up"], ["Review", "`review`, `language`", "text, category", "Review text as written; 31 languages (39.5% English)"],
       ["Review", "`author_playtime_at_review`", "float (minutes)", "Playtime when the review was written (12 missing)"], ["Review", "`created`", "datetime (UTC)", "When the review was written"],
       ["Review", "`steam_purchase`, `received_for_free`", "bool", "Bought on Steam; marked as received for free (2.1%)"], ["Review", "`written_during_early_access`", "bool", "Written while the game was in Early Access (22.7%)"],
       ["Review", "`author_steamid_hash`", "text", "Salted SHA-256 of the reviewer ID (pseudonymised)"], ["Game", "`appid`, `name`", "int, text", "Steam app ID and name"],
       ["Game", "`price_usd`, `is_free`", "float, bool", "Regular price in USD (0 if free); free to play (308 games)"], ["Game", "`launch_date`, `release_date`", "datetime", "Original launch including Early Access; store (1.0) release"],
       ["Game", "`tags`, `genres`, `categories`", "text lists", "Top 20 user tags; Steam genres; store features"], ["Game", "`n_dlc`, `n_achievements`, `n_supported_languages`", "int", "Counts at collection time (missing achievements = none)"],
       ["Game", "`platform_mac`, `platform_linux`, `controller_support`", "bool, text", "Platforms; full or partial controller support"], ["Game", "`steam_total_reviews`, `steam_total_negative`", "int", "Steam's all-language totals; used for the tier, never as features"],
       ["Game", "`spy_owners_low`, `spy_owners_high`", "int", "SteamSpy owner range (outcome; not a feature)"], ["Event", "`appid`, `date`, `event_type`", "int, datetime, text", "Steam news item: patch or sale announcement"]],
      ["Table", "Field", "Type", "Meaning"], [0.6, 2.4, 1.1, 2.6], "Concise data dictionary of the cleaned tables.", size=9)
para("Validation records are in `docs/cleaning_log.md`, `docs/validation_summary.md` and `data/processed/cleaning_log.json`. Key flags added by cleaning:", size=11)
table([["`playtime_outlier`", "bool", "Playtime at review above the 99.5% quantile (363 h); 8,242 reviews"], ["`is_prerelease`", "bool", "Written before the game's launch; 30,018 reviews"],
       ["`low_content`", "bool", "Under 3 letters or 2 words; 208,724 reviews"], ["`is_ascii_art`, `is_copypasta`", "bool", "ASCII art (4,951); same 50+ character text 10+ times (2,285)"],
       ["`text_mining_ok`", "bool", "English and none of the three flags: the text-mining subset"], ["`received_for_free`", "bool", "Reviewer received the game for free; 34,394 reviews"], ["`launch_date` vs `release_date`", "datetime", "Original launch vs 1.0 date; differ for 318 games"],
       ["`reviews_capped`", "bool", "Newest ~15,000 reviews only (5 games)"]], ["Field", "Type", "Meaning and count"], [2.0, 0.8, 3.7], "Key cleaning flags.", size=9)
heading("Appendix B. Tuning, additional figures and supplementary analysis", 1)
trows = [[f"{r.Level}" + (f", {r.Model}" if isinstance(r.Model, str) and r.Model else ""), r.Metric, f"{int(r['Grid points'])}", f"{r['Default CV']:.3f}", f"{r['Tuned CV']:.3f}"] for _, r in tun.iterrows()]
table(trows, ["Search", "Metric", "Grid points", "Default CV", "Tuned CV"], [3.1, 0.9, 0.9, 0.8, 0.8], "Grid-search results: default (grid point 0) against tuned settings on the shared folds.", size=9, align_right_from=2)
srows = [["Game scorer", f"{r.group}: {r.level}", f"{int(r.n)}", f"{r.macro_f1:.2f}", "" if pd.isna(r.recall_struggling) else f"{r.recall_struggling:.2f} ({int(r.n_struggling)})"] for r in gsub.itertuples()]
srows += [["Text detector", f"{r.group}: {r.level}", f"{int(r.n):,}", f"{r.pr_auc:.2f}", f"{r.precision:.2f} / {r.recall:.2f}"] for r in tsub.itertuples()]
table(srows, ["Model", "Subgroup", "Test n", "Macro-F1 / PR-AUC", "Struggling recall (n) / precision, recall"], [1.1, 2.2, 0.8, 1.1, 1.6],
      "Subgroup error check on the held-out test data (game scorer: macro-F1; text detector: PR-AUC; `python -m src.models.subgroup_check`).", size=9, align_right_from=2)
figure(FIG / "findings" / "patch_impact_by_launch.png", "Patch impact by launch timing (change in positive-review share, points; Python/Matplotlib; `notebooks/review2/patch_impact.ipynb`).", width=5.2)
figure(FIG / "eda" / "10_risk_by_game_features.png", "Share of Struggling games by price band, release year, languages, DLC, achievements and review volume with 95% intervals (Python/Seaborn; `notebooks/02_EDA_Visualizations.ipynb`).", width=5.6)
heading("Appendix C. Implementation details and dashboard user guide", 1)
para("Requirements: Python 3.12 and the pinned packages in `requirements.txt`. From the repository root:", size=11, keep=True)
for lab, cmd in (("Install", "python -m venv .venv  ;  pip install -r requirements.txt"), ("Features and checks", "python -m src.features"),
                 ("Models", "python -m src.models.review_models  (also game_models, review_xgboost, game_xgboost)"), ("Model tables", "python -m src.models.compare"),
                 ("Tuning", "python -m src.models.grid_search --task game --model xgboost"), ("Subgroup check", "python -m src.models.subgroup_check"),
                 ("Report", "python -m src.report.build_capstone_report"), ("Dashboard", "streamlit run src/dashboard/app.py")):
    p = doc.add_paragraph(style="List Bullet"); p.paragraph_format.line_spacing = 1.15; p.paragraph_format.space_after = Pt(1)
    add_runs(p, f"**{lab}:** `{cmd}`", 11)
para("**User guide:** *Market explorer*: filter by release year, price band, minimum reviews and tags; KPI cards and charts update. *Tag rules and complaints*: set stability, minimum lift and a tag or price to filter rules; open the complaint-themes and refund-window tabs. "
     "*Patch impact*: read the raw and matched-control results. *Launch-risk scorer*: describe a planned game (price, launch date, platforms, tags, categories) or load an existing game and press *Score this game*; read the predicted tier and scores (not probabilities).", size=11)
heading("Appendix D. Tracking-board evidence, contribution records and AI declaration", 1)
para('Project board: https://github.com/users/Shan713/projects/1 (68 cards, all Completed; card history, owners, due dates and evidence). Commit and pull-request history: https://github.com/Shan713/IndieGuard/pulls. The signed originality and AI-use declarations are on the declaration page and in Section 8.3.', size=11)

doc.core_properties.title = "IndieGuard: Indie Game Launch Analytics - Final Report"
doc.core_properties.author = "Group 1, 23CSE452"
doc.save(OUT)
print("wrote", OUT, "figures", FIGN[0], "tables", TABN[0])
