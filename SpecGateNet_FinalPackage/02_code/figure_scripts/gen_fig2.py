# -*- coding: utf-8 -*-
# Rebuild Figure 2 (Frequency Branch) as a narrow/tall layout so that, at IEEE
# single-column width (~251 pt), all text renders at legible size (>=~7 pt).
# Content is preserved verbatim; only the geometry is restructured.
import fitz, html

W = 800
COLW_PT = 251.0                     # IEEE columnwidth
SANS = "Helvetica, Arial, sans-serif"
MONO = "'Courier New', monospace"

parts = []
def esc(s): return html.escape(s, quote=False)
def rect(x,y,w,h,fill,stroke,sw=1.6,rx=10,dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def txt(x,y,s,fs,fill="#1c1c1c",anchor="middle",weight="400",font=SANS,style="normal"):
    parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{font}" '
                 f'font-size="{fs}" font-weight="{weight}" fill="{fill}" font-style="{style}">{esc(s)}</text>')
def arrow(x1,y1,x2,y2,color="#444444",sw=2.2,mid="aD"):
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" marker-end="url(#{mid})"/>')
def poly(pts,color="#6E8091",sw=2.0,mid="aM"):
    p=" ".join(f"{a},{b}" for a,b in pts)
    parts.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" marker-end="url(#{mid})"/>')

# ---- fonts (svg units) ----
F_TITLE=30; F_BTITLE=26; F_SUB=23; F_ANNOT=23; F_MASKH=21; F_MASKT=25; F_MASKS=23; F_NOTE=22

# ---- pipeline boxes (title, subtext, fill, stroke) ----
GREY=("#F2F2F2","#6B6B6B"); OR1=("#FBEEE0","#C6803A"); OR2=("#F6E1C8","#B06A2C")
OR3=("#F3D9BE","#B06A2C")
pipe=[
 ("Input Feature","x  (B x C x H x W)",*GREY),
 ("Cast to FP32","x.float()",*GREY),
 ("2D Real FFT","rfft2 (norm=ortho)",*OR1),
 ("High-pass Filter","spectrum \u2299 mask",*OR2),
 ("2D Inverse FFT","irfft2 (s=(H,W), ortho)",*OR1),
 ("Cast to dtype","out.to(x.dtype)",*GREY),
 ("Freq. Feature","(B x C x H x W)",*OR3),
]
# arrow-side annotations keyed by the gap AFTER box index -> list of lines
annot={1:["single precision","(AMP-stable)"],
       2:["complex spectrum","B x C x H x","(W//2 + 1)"],
       3:["filtered spectrum"],
       4:["real, B x C x H x W"]}

BX_W=340; BX_H=80; GAP=40
bx=(W-BX_W)/2
y=96
title_y=54
box_tops=[]
for i,(t,s,fill,st) in enumerate(pipe):
    box_tops.append(y)
    y+=BX_H+GAP
pipe_bottom=box_tops[-1]+BX_H

# ---- assemble header/title ----
txt(W/2, title_y, "Frequency Branch of SpecGate-Net", F_TITLE, "#111111", weight="700")

# draw boxes + arrows + annotations
for i,(t,s,fill,st) in enumerate(pipe):
    ytop=box_tops[i]
    rect(bx,ytop,BX_W,BX_H,fill,st)
    txt(W/2, ytop+34, t, F_BTITLE, "#1c1c1c", weight="700")
    txt(W/2, ytop+62, s, F_SUB, "#555555", font=MONO)
    if i < len(pipe)-1:
        ay1=ytop+BX_H; ay2=ytop+BX_H+GAP
        arrow(W/2, ay1, W/2, ay2)
        if i in annot:
            lines=annot[i]; amid=(ay1+ay2)/2
            ax=bx+BX_W+20
            y0=amid-(len(lines)-1)*12
            for k,ln in enumerate(lines):
                txt(ax, y0+k*24, ln, F_ANNOT, "#7a7a7a", anchor="start", style="italic", font=SANS)

# ---- mask construction group (below pipeline) ----
gy=pipe_bottom+GAP+14
gh=322
gx=bx-40; gw=BX_W+80
rect(gx, gy, gw, gh, "#FAFBFC", "#7C93A8", sw=1.6, rx=14, dash="5,4")
txt(W/2, gy+30, "Fixed radial high-pass mask", F_MASKH, "#4f6472", weight="700")
txt(W/2, gy+52, "non-learnable, cached per (H, W, device)", F_MASKH-2, "#4f6472", weight="600")
mask=[("Frequency Grid","fftfreq(H), rfftfreq(W)"),
      ("Radial Frequency","r = \u221a(f\u1d67\u00b2 + f\u2093\u00b2)"),
      ("Binary High-pass","mask = (r \u2265 cutoff)")]
mbw=BX_W-30; mbx=(W-mbw)/2; my=gy+74; mbh=46; mgap=24
for j,(t,s) in enumerate(mask):
    yt=my+j*(mbh+mgap)
    rect(mbx,yt,mbw,mbh,"#EEF2F6","#7C93A8",sw=1.5,rx=8)
    txt(W/2, yt+20, t, F_MASKT-2, "#1c1c1c", weight="700")
    txt(W/2, yt+37, s, F_MASKS-1, "#555555", font=MONO)
    if j<len(mask)-1:
        arrow(W/2, yt+mbh, W/2, yt+mbh+mgap, color="#6E8091", sw=1.8, mid="aM")
txt(W/2, my+3*(mbh+mgap)+12, "cutoff = 0.1  (fixed hyperparameter)", F_MASKS, "#4f6472", weight="600")

# mask -> High-pass Filter : routed on the LEFT (clear margin)
hp_top=box_tops[3]; hp_midy=hp_top+BX_H/2
lx=bx-46
poly([(gx, gy+gh/2),(lx, gy+gh/2),(lx, hp_midy),(bx, hp_midy)])
txt(lx-15, (hp_midy+gy+gh/2)/2, "mask", F_ANNOT, "#6E8091", anchor="end", style="italic")

# ---- notes + legend ----
ny=gy+gh+44
txt(30, ny, "Parameter-free module (no learnable weights).", F_NOTE, "#555", anchor="start", weight="700")
notes2=["Output spatial resolution equals input (H x W).  Both transforms use",
        "norm = 'ortho'.  The spectrum itself is not passed through abs()/magnitude."]
for k,ln in enumerate(notes2):
    txt(30, ny+26+k*24, ln, F_NOTE-1, "#666", anchor="start", weight="500")
ly=ny+26+len(notes2)*24+22
arrow(97, ly-5, 137, ly-5); txt(147, ly, "data path", F_NOTE-1, "#555", anchor="start")
arrow(367, ly-5, 407, ly-5, color="#6E8091", sw=1.8, mid="aM"); txt(417, ly, "fixed mask (non-learnable)", F_NOTE-1, "#555", anchor="start")

H = int(ly+30)

defs=('<defs>'
 '<marker id="aD" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0.5,0.8 L9,5 L0.5,9.2 Z" fill="#444444"/></marker>'
 '<marker id="aM" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0.5,0.8 L9,5 L0.5,9.2 Z" fill="#6E8091"/></marker>'
 '</defs>')
svg=(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
     f'font-family="{SANS}">{defs}<rect width="{W}" height="{H}" fill="#FFFFFF"/>'+"".join(parts)+"</svg>")

svgpath=r"C:\Users\Soumojay\Desktop\New_Sm_P\SegmentationResearch\paper2\figures\Figure2_FrequencyBranch.svg"
open(svgpath,"w",encoding="utf-8").write(svg)

# convert to PDF via fitz
pdf=fitz.open(svgpath).convert_to_pdf()
pdfpath=r"C:\Users\Soumojay\Desktop\New_Sm_P\SegmentationResearch\paper2\figures\Figure2_FrequencyBranch.pdf"
open(pdfpath,"wb").write(pdf)

# report + preview render at column scale
scale=COLW_PT/W
print(f"viewBox {W}x{H}  aspect {W/H:.3f}  -> at {COLW_PT:.0f}pt wide, height {COLW_PT/(W/H):.0f}pt = {COLW_PT/(W/H)/72:.2f}in")
for nm,f in [("fig title",F_TITLE),("box title",F_BTITLE),("box subtext",F_SUB),
             ("annotation",F_ANNOT),("mask title",F_MASKT-2),("mask subtext",F_MASKS-1),("notes",F_NOTE-1)]:
    print(f"   {nm:12s} {f}u -> {f*scale:.2f} pt displayed")
prev=r"C:\Users\Soumojay\AppData\Local\Temp\claude\C--Users-Soumojay-Desktop-New-Sm-P\d9396353-c9a1-46c9-b57a-e755af7cb994\scratchpad\fig2_new.png"
fitz.open("pdf",pdf)[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(prev)
print("preview:",prev)
