# -*- coding: utf-8 -*-
# Rebuild Figure 1 (Overall Architecture) with fewer horizontal columns and
# larger fonts so text renders at ~7 pt at IEEE textwidth (figure*).
# All content preserved: Input+Norm merged into one box; per-scale g_i kept as
# labels on the fusion outputs; projected skips kept as labelled dashed arrows.
import fitz, html
W=1458; TW_PT=516.0
SANS="Helvetica, Arial, sans-serif"
parts=[]
def esc(s): return html.escape(s,quote=False)
def rect(x,y,w,h,fill,stroke,sw=1.5,rx=8,dash=None,op=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''; o=f' opacity="{op}"' if op else ''
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')
def txt(x,y,s,fs,fill="#1c1c1c",anchor="middle",weight="500",style="normal"):
    parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{SANS}" font-size="{fs}" font-weight="{weight}" fill="{fill}" font-style="{style}">{esc(s)}</text>')
def box(cx,cy,w,h,lines,fill,stroke,fs=19,weight="600"):
    rect(cx-w/2,cy-h/2,w,h,fill,stroke)
    n=len(lines); step=fs+3; y0=cy-(n-1)*step/2
    for i,ln in enumerate(lines): txt(cx,y0+i*step+fs*0.35,ln,fs,"#1c1c1c",weight=weight)
def arr(x1,y1,x2,y2,color="#444444",sw=1.7,mid="aMain"):
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" marker-end="url(#{mid})"/>')
def pla(pts,color,sw,mid,dash=None):
    p=" ".join(f"{a},{b}" for a,b in pts); d=f' stroke-dasharray="{dash}"' if dash else ''
    parts.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#{mid})"/>')
def lnp(pts,color="#444444",sw=1.7):
    p=" ".join(f"{a},{b}" for a,b in pts)
    parts.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}"/>')
FL=22  # legend / free-annotation font (~8 pt at textwidth)

BLU=("#E8EEF7","#3B6BA5"); GRY=("#F2F2F2","#6B6B6B"); ORA=("#FBEEE0","#C6803A")
ORL=("#FDF5EC","#C6803A"); GRN=("#E7F1EA","#4A8C5E"); PUR=("#EDE8F3","#7A5BA6")
PUF=("#F4F0FA","#7A5BA6"); TEA=("#E3F0F0","#3F8C8C"); RED=("#F7E8E8","#B24A4A")
REDP=("#F3DADA","#A83232"); REC=("#EEEEEE","#8A8A8A")
F_TITLE=25; FB=20; FS=18

# column centres
c_in=64; c_enc=196; c_ef=330; c_fb=492; c_al=666; c_gf=840
c_dec=1035; c_head=1205; c_out=1368
rows=[150,345,540,735]           # 4 encoder scales (wider vertical spacing)

txt(W/2,40,"Overall Architecture of SpecGate-Net",F_TITLE,"#111",weight="700")

# input + normalization (merged) and frozen encoder (spans all rows)
midy=(rows[0]+rows[-1])/2
box(c_in,midy,120,74,["Input Image","(3x384x384)","+ Normalize"],*GRY,fs=FS)
rect(c_enc-60,rows[0]-52,120,rows[-1]-rows[0]+104,*BLU);
for i,t in enumerate(["Frozen","SAM2","Encoder"]): txt(c_enc,midy-20+i*20,t,FB+2,"#1c1c1c",weight="700")
arr(c_in+60,midy,c_enc-60,midy)

# per-scale pipeline
for r,cy in enumerate(rows):
    box(c_ef,cy,128,66,["Encoder","Feature f%d"%(r+1)],*BLU,fs=FB)
    arr(c_enc+60,cy,c_ef-64,cy)
    # spatial align (upper) & frequency branch->freq align (lower)
    uy=cy-42; ly=cy+42
    box(c_al,uy,168,52,["Feature Alignment","(Spatial)"],*GRN,fs=FS)
    box(c_fb,ly,150,52,["Frequency","Branch"],*ORA,fs=FS)
    box(c_al,ly,168,52,["Feature Alignment","(Frequency)"],*ORL,fs=FS)
    # enc feature -> spatial align (up) + freq branch (down): orthogonal split
    sx=c_ef+75
    pla([(c_ef+64,cy),(sx,cy),(sx,uy),(c_al-84,uy)],"#444444",1.7,"aMain")
    pla([(c_ef+64,cy),(sx,cy),(sx,ly),(c_fb-75,ly)],"#444444",1.7,"aMain")
    arr(c_fb+75,ly,c_al-84,ly)
    # aligns -> gated fusion: orthogonal merge via a vertical bus
    box(c_gf,cy,116,72,["Gated","Fusion"],*PUR,fs=FB+1)
    busx=(c_al+84+c_gf-58)/2
    lnp([(c_al+84,uy),(busx,uy),(busx,cy)]); lnp([(c_al+84,ly),(busx,ly),(busx,cy)])
    arr(busx,cy,c_gf-58,cy)
    # gated fusion -> fused g_i (label sits above the fusion->decoder arrow)
    txt(c_gf+80,cy-20,"g%d"%(r+1),22,"#5c4488",weight="700")

# ---- decoder (4 stacked stages, flows upward), heads, outputs ----
decL=["Decoder Stage 1|(finest, no skip)","Decoder Stage 2","Decoder Stage 3","Decoder Stage 4|(bottleneck)"]
for r,cy in enumerate(rows):
    lines=decL[r].split("|")
    box(c_dec,cy,162,68,lines,*TEA,fs=FB)
    # heads + outputs
    box(c_head,cy,138,58,["Deep","Supervision Head"],*RED,fs=FS)
    arr(c_dec+81,cy,c_head-69,cy)
    if r==0:
        box(c_out,cy,156,58,["Primary","Segmentation (p1)"],*REDP,fs=FS,weight="700")
    else:
        box(c_out,cy,156,58,["Aux. Prediction","p%d"%(r+1)],*RED,fs=FS)
    arr(c_head+69,cy,c_out-78,cy)
# deepest fused -> decoder stage 4 (bottleneck)
arr(c_gf+58,rows[3],c_dec-81,rows[3])
# decoder upward chain
for r in range(3,0,-1):
    arr(c_dec,rows[r]-36,c_dec,rows[r-1]+36)
# projected skips: g1,g2,g3 -> decoder stages 1,2,3 (dashed, labelled)
for r in range(3):
    pla([(c_gf+58,rows[r]),(c_dec-81,rows[r])], "#8C8C8C",1.3,"aSkip",dash="5.5,3")
txt((c_gf+c_dec)/2,rows[0]-58,"projected skips",21,"#6E6E6E",weight="600",style="italic")

# ---- auxiliary reconstruction branch (training only) : single box, fed from bottleneck ----
ry=rows[3]+120
txt(c_dec-158,rows[3]+72,"training only",21,"#6a6a6a",style="italic",anchor="start")
pla([(c_dec,rows[3]+34),(c_dec,ry+20-25)], "#9A9A9A",1.5,"aRecon",dash="1.5,3")
box(c_dec,ry+20,190,50,["Auxiliary Reconstruction","Branch (RDB)"],*REC,fs=FS-1)

# ---- legend (8 pt, three wide columns) ----
lx=40; lyb=rows[3]+215
rect(lx-8,lyb-30,W-64,210,"#FCFCFC","#D8D8D8",sw=1,rx=9)
txt(lx,lyb,"Legend",FL+1,"#1c1c1c",anchor="start",weight="700")
r1,r2,r3,r4=lyb+40,lyb+82,lyb+124,lyb+166
def line_leg(x,y,color,sw,mid,dash,label):
    pla([(x,y-5),(x+56,y-5)],color,sw,mid,dash=dash); txt(x+68,y,label,FL,"#1c1c1c",anchor="start")
line_leg(lx,r1,"#444444",1.9,"aMain",None,"Main feature flow")
line_leg(lx,r2,"#8C8C8C",1.4,"aSkip","5.5,3","Projected skip connection")
line_leg(lx,r3,"#9A9A9A",1.6,"aRecon","1.5,3","Reconstruction flow (training only)")
def sw(x,y,fill,stroke,label):
    rect(x,y-15,20,18,fill,stroke,sw=1.2,rx=3); txt(x+30,y,label,FL,"#1c1c1c",anchor="start")
sw(520,r1,*BLU,"Frozen SAM2 Encoder"); sw(520,r2,*GRN,"Spatial path (Feature Alignment)")
sw(520,r3,*ORA,"Frequency path (Branch + Alignment)")
sw(980,r1,*PUR,"Gated Fusion"); sw(980,r2,*TEA,"Decoder")
sw(980,r3,*RED,"Segmentation heads / outputs"); sw(980,r4,*REC,"Auxiliary reconstruction")

H=int(lyb+200)
defs=('<defs>'
 '<marker id="aMain" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0.5,0.8 L9,5 L0.5,9.2 Z" fill="#444444"/></marker>'
 '<marker id="aSkip" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0.5,0.8 L9,5 L0.5,9.2 Z" fill="#8C8C8C"/></marker>'
 '<marker id="aRecon" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0.5,0.8 L9,5 L0.5,9.2 Z" fill="#9A9A9A"/></marker></defs>')
svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{SANS}">{defs}<rect width="{W}" height="{H}" fill="#FFFFFF"/>'+"".join(parts)+"</svg>"
sp=r"C:\Users\Soumojay\Desktop\New_Sm_P\SegmentationResearch\paper2\figures\Figure1_Architecture.svg"
open(sp,"w",encoding="utf-8").write(svg)
open(r"C:\Users\Soumojay\Desktop\New_Sm_P\SegmentationResearch\paper2\figures\Figure1_Architecture.pdf","wb").write(fitz.open(sp).convert_to_pdf())
sc=TW_PT/W
print(f"viewBox {W}x{H} aspect {W/H:.3f} -> at {TW_PT:.0f}pt, height {TW_PT/(W/H):.0f}pt = {TW_PT/(W/H)/72:.2f}in")
for nm,f in [("title",F_TITLE),("box",FB),("small",FS)]: print(f"  {nm:6s}{f}u -> {f*sc:.2f}pt")
fitz.open("pdf",fitz.open(sp).convert_to_pdf())[0].get_pixmap(matrix=fitz.Matrix(1.15,1.15)).save(r"C:\Users\Soumojay\AppData\Local\Temp\claude\C--Users-Soumojay-Desktop-New-Sm-P\d9396353-c9a1-46c9-b57a-e755af7cb994\scratchpad\fig1_new.png")
print("ok")
