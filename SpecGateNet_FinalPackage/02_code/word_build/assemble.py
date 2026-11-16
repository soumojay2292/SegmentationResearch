# -*- coding: utf-8 -*-
# Assemble the complete IEEE two-column SpecGate-Net Word paper:
# pandoc body (text + OMML equations) + native merged tables + figures + refs.
import json, os, re, copy
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH as AL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

WD = r"C:\Users\Soumojay\AppData\Local\Temp\claude\C--Users-Soumojay-Desktop-New-Sm-P\d9396353-c9a1-46c9-b57a-e755af7cb994\scratchpad\docx_work"
TABLES_DOCX = os.path.join(WD, "SpecGateNet_Tables.docx")
BBL = os.path.join(r"C:\Users\Soumojay\AppData\Local\Temp\claude\C--Users-Soumojay-Desktop-New-Sm-P\d9396353-c9a1-46c9-b57a-e755af7cb994\scratchpad\docx_work", "refs.bbl")
FONT = "Times New Roman"
meta = json.load(open(os.path.join(WD, "meta.json"), encoding="utf-8"))

def W(t): return qn('w:'+t)

def set_run_props(rpr, bold=False, italic=False, size=None, sc=False, color=None, sup=False, sub=False, font=FONT):
    rf = OxmlElement('w:rFonts')
    for a in ('w:ascii','w:hAnsi','w:cs'): rf.set(qn(a), font)
    rpr.append(rf)
    if bold: rpr.append(OxmlElement('w:b'))
    if italic: rpr.append(OxmlElement('w:i'))
    if sc: rpr.append(OxmlElement('w:smallCaps'))
    if sup or sub:
        v = OxmlElement('w:vertAlign'); v.set(qn('w:val'), 'superscript' if sup else 'subscript'); rpr.append(v)
    if size:
        s = OxmlElement('w:sz'); s.set(qn('w:val'), str(size)); rpr.append(s)
    if color:
        c = OxmlElement('w:color'); c.set(qn('w:val'), color); rpr.append(c)

def make_run(text, **kw):
    r = OxmlElement('w:r'); rpr = OxmlElement('w:rPr'); set_run_props(rpr, **kw); r.append(rpr)
    t = OxmlElement('w:t'); t.set(qn('xml:space'), 'preserve'); t.text = text; r.append(t)
    return r

def make_sectpr(cols):
    sp = OxmlElement('w:sectPr')
    t = OxmlElement('w:type'); t.set(qn('w:val'), 'continuous'); sp.append(t)
    sz = OxmlElement('w:pgSz'); sz.set(qn('w:w'), '12240'); sz.set(qn('w:h'), '15840'); sp.append(sz)
    mg = OxmlElement('w:pgMar')
    for k, v in (('top','1008'),('bottom','1008'),('left','893'),('right','893'),
                 ('header','720'),('footer','720'),('gutter','0')):
        mg.set(qn('w:'+k), v)
    sp.append(mg)
    c = OxmlElement('w:cols'); c.set(qn('w:num'), str(cols)); c.set(qn('w:space'), '360')
    if cols > 1: c.set(qn('w:equalWidth'), '1')
    sp.append(c)
    return sp

def make_p(runs, align='both', before=0, after=40, sectpr=None, line=240, style=None, keepnext=False):
    p = OxmlElement('w:p'); ppr = OxmlElement('w:pPr')
    if style:
        st = OxmlElement('w:pStyle'); st.set(qn('w:val'), style); ppr.append(st)
    if keepnext: ppr.append(OxmlElement('w:keepNext'))
    jc = OxmlElement('w:jc'); jc.set(qn('w:val'), align); ppr.append(jc)
    sp = OxmlElement('w:spacing'); sp.set(qn('w:before'), str(before)); sp.set(qn('w:after'), str(after))
    sp.set(qn('w:line'), str(line)); sp.set(qn('w:lineRule'), 'auto'); ppr.append(sp)
    if sectpr is not None: ppr.append(sectpr)
    p.append(ppr)
    for r in runs: p.append(r)
    return p

# ---------------- caption cleaner (LaTeX -> plain runs) ----------------
def clean_caption(s):
    s = s.strip()
    s = s.replace('\\%', '%').replace('\\&', '&')
    s = s.replace('---', '\u2014').replace('--', '\u2013')
    s = re.sub(r'\\times', '\u00d7', s)
    s = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\text(?:it|bf|rm)?\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\emph\{([^}]*)\}', r'\1', s)
    s = re.sub(r'_\{([^}]*)\}', r'_\1', s)      # n_{valid} -> n_valid
    s = re.sub(r'\^\{([^}]*)\}', r'^\1', s)
    s = s.replace('\\,', ' ').replace('~', ' ')
    s = s.replace('$', '')
    s = re.sub(r'\\[a-zA-Z]+', '', s)           # drop leftover commands
    s = s.replace('{', '').replace('}', '')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ---------------- open base doc (pandoc body) ----------------
doc = Document(os.path.join(WD, "body.docx"))
body = doc.element.body

# styles: Normal + headings -> Times New Roman, 10pt
def style_font(name, size_hp, bold=None, italic=None):
    try: st = doc.styles[name]
    except KeyError: return
    st.font.name = FONT; st.font.size = Pt(size_hp/2.0)
    rpr = st.element.get_or_add_rPr()
    rf = rpr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts'); rpr.insert(0, rf)
    for a in ('w:ascii','w:hAnsi','w:cs'): rf.set(qn(a), FONT)
    if bold is not None: st.font.bold = bold
    if italic is not None: st.font.italic = italic
for nm in ('Normal','Body Text','First Paragraph','Compact'):
    style_font(nm, 20)
style_font('Heading 1', 20, bold=False)
style_font('Heading 2', 20, bold=False, italic=True)
style_font('Heading 3', 20, bold=False, italic=True)
# center + smallcaps section headings (Heading 1)
try:
    h1 = doc.styles['Heading 1']; h1.paragraph_format.alignment = AL.CENTER
    h1.paragraph_format.space_before = Pt(10); h1.paragraph_format.space_after = Pt(4)
    sc = OxmlElement('w:smallCaps'); h1.element.get_or_add_rPr().append(sc)
except Exception: pass
# body default justify
try:
    doc.styles['Normal'].paragraph_format.alignment = AL.JUSTIFY
    doc.styles['Normal'].paragraph_format.space_after = Pt(2)
    doc.styles['Normal'].paragraph_format.line_spacing = 1.0
except Exception: pass

# final body sectPr -> two columns
old_sect = body.find(qn('w:sectPr'))
if old_sect is not None: body.remove(old_sect)
body.append(make_sectpr(2))

# ---------------- figures: create image paragraphs via API ----------------
FIGW = {"fig:architecture":("fig1.png",5.6), "fig:freqbranch":("fig2.png",3.0),
        "fig:fafusion":("fig3.png",4.8), "fig:qualitative":("fig4.png",5.3)}
fig_para = {}
for lab,(png,win) in FIGW.items():
    p = doc.add_paragraph(); p.alignment = AL.CENTER
    p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(2)
    r = p.add_run(); r.add_picture(os.path.join(WD, png), width=Inches(win))
    fig_para[lab] = p._p
    body.remove(p._p)   # detach; will relocate at placeholder

# ---------------- tables: extract 5 <w:tbl> from tables docx ----------------
tdoc = Document(TABLES_DOCX); tbody = tdoc.element.body
tbls = tbody.findall(qn('w:tbl'))
TAB_BY_LABEL = {"tab:metrics":tbls[0], "tab:benchmark":tbls[1], "tab:boundary":tbls[2],
                "tab:stats":tbls[3], "tab:ablation":tbls[4]}
TAB_ROMAN = {"tab:metrics":"I","tab:benchmark":"II","tab:boundary":"III","tab:stats":"IV","tab:ablation":"V"}
FIG_NUM = {"fig:architecture":"1","fig:freqbranch":"2","fig:fafusion":"3","fig:qualitative":"4"}

flmeta = {fl["label"]: fl for fl in meta["floats"]}

# ---------------- replace placeholders ----------------
def find_placeholder(k):
    tok = "@@FLOAT_%d@@" % k
    for p in body.findall(qn('w:p')):
        txt = "".join(t.text or "" for t in p.iter(qn('w:t')))
        if txt.strip() == tok: return p
    return None

def attach_sectpr(paragraph_el, cols):
    ppr = paragraph_el.find(qn('w:pPr'))
    if ppr is None:
        ppr = OxmlElement('w:pPr'); paragraph_el.insert(0, ppr)
    for ex in ppr.findall(qn('w:sectPr')): ppr.remove(ex)
    ppr.append(make_sectpr(cols))

for k in range(len(meta["floats"])):
    fl = meta["floats"][k]; lab = fl["label"]; ph = find_placeholder(k)
    if ph is None: continue
    prev = ph.getprevious()
    # end the preceding 2-col text section on the previous paragraph
    if prev is not None and prev.tag == qn('w:p'):
        attach_sectpr(prev, 2)
    else:
        gap = make_p([make_run("")], after=0, sectpr=make_sectpr(2)); ph.addprevious(gap)
    blocks = []
    if fl["kind"] == "table":
        cap = clean_caption(fl["caption"])
        blocks.append(make_p([make_run("TABLE "+TAB_ROMAN[lab], size=16)], align='center', before=60, after=0, keepnext=True))
        blocks.append(make_p([make_run(cap, size=16)], align='center', before=0, after=40, keepnext=True))
        blocks.append(copy.deepcopy(TAB_BY_LABEL[lab]))
        # closing paragraph carries the 1-col section end
        blocks.append(make_p([make_run("")], after=0, sectpr=make_sectpr(1)))
    else:
        cap = clean_caption(fl["caption"])
        blocks.append(fig_para[lab])
        cruns = [make_run("Fig. "+FIG_NUM[lab]+".  ", size=16, bold=False), make_run(cap, size=16)]
        blocks.append(make_p(cruns, align='both', before=2, after=60, sectpr=make_sectpr(1)))
    for b in blocks: ph.addprevious(b)
    ph.getparent().remove(ph)

# ---------------- title / authors / abstract / keywords ----------------
first = body[0]  # first content block (Introduction heading)
def insert_before_first(el): first.addprevious(el)

title = meta["title"].replace("--", "\u2013")
insert_before_first(make_p([make_run(title, bold=True, size=42)], align='center', before=0, after=120))

# author table (2 cols, borderless)
authors = [
    ["Soumojay Karmakar", "Manipal Institute of Technology Bengaluru,", "Manipal Academy of Higher Education,", "Manipal, India", "soumojay.mitblr2025@learner.manipal.edu"],
    ["Lokesh Singh", "Manipal Institute of Technology Bengaluru,", "Manipal Academy of Higher Education,", "Manipal, India", "singh.lokesh@manipal.edu"],
]
tbl = OxmlElement('w:tbl'); tpr = OxmlElement('w:tblPr')
tw = OxmlElement('w:tblW'); tw.set(qn('w:w'),'0'); tw.set(qn('w:type'),'auto'); tpr.append(tw)
jc = OxmlElement('w:jc'); jc.set(qn('w:val'),'center'); tpr.append(jc)
bd = OxmlElement('w:tblBorders')
for e in ('top','left','bottom','right','insideH','insideV'):
    b = OxmlElement('w:'+e); b.set(qn('w:val'),'none'); b.set(qn('w:sz'),'0'); bd.append(b)
tpr.append(bd); tbl.append(tpr)
grid = OxmlElement('w:tblGrid')
for _ in range(2):
    gc = OxmlElement('w:gridCol'); gc.set(qn('w:w'),'4600'); grid.append(gc)
tbl.append(grid)
tr = OxmlElement('w:tr')
for col in authors:
    tc = OxmlElement('w:tc'); tcpr = OxmlElement('w:tcPr')
    ww = OxmlElement('w:tcW'); ww.set(qn('w:w'),'4600'); ww.set(qn('w:type'),'dxa'); tcpr.append(ww)
    tc.append(tcpr)
    for i, line in enumerate(col):
        italic = 1 <= i <= 2
        bold = i == 0
        tc.append(make_p([make_run(line, bold=bold, italic=italic, size=20)], align='center', before=0, after=0, line=220))
    tbl.append_tc = None
    tr.append(tc)
tbl.append(tr)
insert_before_first(tbl)

# section break paragraph ending the single-column title section
insert_before_first(make_p([make_run("")], after=0, sectpr=make_sectpr(1)))

# abstract (2-col region)
ab = clean_caption(meta["abstract"]) if False else meta["abstract"]
ab = re.sub(r'\s+', ' ', ab).strip()
insert_before_first(make_p([make_run("Abstract\u2014", bold=True, italic=True, size=18),
                            make_run(ab, bold=True, size=18)], align='both', before=60, after=40))
kw = re.sub(r'\s+', ' ', meta["keywords"]).strip()
insert_before_first(make_p([make_run("Index Terms\u2014", bold=True, italic=True, size=18),
                            make_run(kw, italic=True, size=18)], align='both', before=0, after=80))

# ---------------- references ----------------
def parse_bbl(path):
    raw = open(path, encoding="utf-8").read()
    items = re.split(r'\\bibitem\{[^}]*\}', raw)[1:]
    # cut last item at \end{thebibliography}
    entries = []
    for it in items:
        it = it.split('\\end{thebibliography}')[0]
        entries.append(it)
    out = []
    for e in entries:
        e = e.strip()
        # split into runs by \emph{...}
        runs = []
        pos = 0
        for m in re.finditer(r'\\emph\{([^}]*)\}', e):
            pre = e[pos:m.start()]; runs.append((pre, False)); runs.append((m.group(1), True)); pos = m.end()
        runs.append((e[pos:], False))
        # clean each run text
        cl = []
        for txt, ital in runs:
            t = txt
            t = t.replace('``','\u201c').replace("''",'\u201d').replace('`',"\u2018")
            t = t.replace('---','\u2014').replace('--','\u2013')
            t = re.sub(r'\\url\{([^}]*)\}', r'\1', t)
            t = t.replace('\\&','&').replace('~',' ').replace('\\,',' ')
            t = re.sub(r'\\[a-zA-Z]+', '', t)
            t = t.replace('{','').replace('}','')
            t = re.sub(r'\s+',' ', t)
            if t: cl.append((t, ital))
        out.append(cl)
    return out

refs = parse_bbl(BBL)
# keep final sectPr last: detach, append refs, re-append sectPr
sect = body.find(qn('w:sectPr')); body.remove(sect)
body.append(make_p([make_run("References", bold=True, sc=True, size=20)], align='center', before=120, after=60))
# Build reference paragraphs
for i, runs in enumerate(refs, start=1):
    rr = [make_run("[%d] " % i, size=18)]
    for txt, ital in runs:
        rr.append(make_run(txt, italic=ital, size=18))
    body.append(make_p(rr, align='both', before=0, after=20, line=210))
body.append(sect)

# ---------------- IEEE section/subsection numbering ----------------
ROMAN2 = ["I","II","III","IV","V","VI","VII","VIII","IX","X"]
sec = sub = ssub = 0; nh1 = 0
for el in list(body):
    if el.tag != qn('w:p'): continue
    ppr = el.find(qn('w:pPr'))
    if ppr is None: continue
    st = ppr.find(qn('w:pStyle'))
    if st is None: continue
    val = (st.get(qn('w:val')) or "").replace(" ", "")
    if val == "Heading1":
        sec += 1; sub = 0; ssub = 0; nh1 += 1; pre = ROMAN2[sec-1] + ". "
    elif val == "Heading2":
        sub += 1; ssub = 0; pre = chr(64+sub) + ". "
    elif val == "Heading3":
        ssub += 1; pre = str(ssub) + ") "
    else:
        continue
    ppr.addnext(make_run(pre))   # prepend number run (inherits heading style)
print("numbered headings -> sections(H1):", nh1)

doc.save(os.path.join(WD, "SpecGateNet_Paper.docx"))
print("SAVED SpecGateNet_Paper.docx ; refs:", len(refs))
