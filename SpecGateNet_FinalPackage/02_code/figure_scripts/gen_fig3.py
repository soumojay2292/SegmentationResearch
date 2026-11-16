# -*- coding: utf-8 -*-
# Rebuild Figure 3 (Feature Alignment + Gated Fusion) as a full-width figure*
# with three stacked pipeline-rows, so text renders at ~7-9 pt at IEEE textwidth.
# All content preserved verbatim; geometry restructured.
import fitz, html

W = 1240
TW_PT = 516.0                       # IEEE textwidth (figure*)
SANS = "Helvetica, Arial, sans-serif"
MONO = "'Courier New', monospace"
parts=[]
def esc(s): return html.escape(s, quote=False)
def rect(x,y,w,h,fill,stroke,sw=1.6,rx=9,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def txt(x,y,s,fs,fill="#1c1c1c",anchor="middle",weight="400",font=SANS,style="normal"):
    parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{font}" font-size="{fs}" font-weight="{weight}" fill="{fill}" font-style="{style}">{esc(s)}</text>')
def circ(cx,cy,r,sym,fs=22):
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#FFFFFF" stroke="#7A5BA6" stroke-width="1.8"/>')
    parts.append(f'<text x="{cx}" y="{cy+1}" text-anchor="middle" dominant-baseline="central" font-family="{SANS}" font-size="{fs}" fill="#5c4488">{esc(sym)}</text>')
def arr(x1,y1,x2,y2,color="#444",sw=1.9):
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" marker-end="url(#a)"/>')

F_TITLE=26; F_BT=21; F_SUB=17; F_HEAD=18; F_ANN=18; F_FORM=18; F_LEG=17

# box heights/fonts
BH=58; IBH=52
def cbox(x,y,w,title,sub,fill,stroke,bt=F_BT,st=F_SUB,h=BH,subfont=MONO):
    rect(x,y,w,h,fill,stroke)
    if sub:
        txt(x+w/2, y+h*0.40, title, bt, "#1c1c1c", weight="700")
        txt(x+w/2, y+h*0.74, sub, st, "#555", font=subfont)
    else:
        txt(x+w/2, y+h/2+bt*0.34, title, bt, "#1c1c1c", weight="700")
    return x+w

# ---------- title ----------
txt(W/2, 40, "Feature Alignment and Gated Fusion of SpecGate-Net", F_TITLE, "#111", weight="700")

# ---------- align pipeline (reused for spatial & frequency) ----------
def align_row(cy, pal, in_t, in_s, out_t, group_label):
    # pal: (dark, light, convfill, convstroke, inoutfill)
    dark,light,cvf,cvs,iof = pal
    x=40
    xw=cbox(x, cy-BH/2, 210, in_t, in_s, iof, dark);
    gx0=xw+34
    # dashed group around norm/conv/relu
    # boxes:
    bx=gx0+14
    y=cy-IBH/2
    p1=cbox(bx, y, 190, "InstanceNorm2d", "affine=False", light, dark, st=F_SUB, h=IBH)
    p2=cbox(p1+22, y, 150, "1x1 Conv *", "C -> C", cvf, cvs, h=IBH)
    p3=cbox(p2+22, y, 110, "ReLU", "", light, dark, h=IBH)
    gx1=p3+14
    rect(gx0, cy-IBH/2-26, gx1-gx0, IBH+40, "none", dark, sw=1.3, rx=12, dash="5,4")
    txt(gx0+10, cy-IBH/2-32, group_label, F_HEAD, dark, anchor="start", weight="700")
    ox=gx1+34
    oxw=cbox(ox, cy-BH/2, 200, out_t, "B x C x H x W", iof, dark)
    # arrows
    arr(40+210, cy, gx0+14, cy)
    arr(bx+190, cy, bx+190+22, cy); arr(p1+22+150, cy, p2+22, cy); arr(p2+22+110, cy, ox, cy)
    return ox+200, cy   # right edge of aligned box, its center-y

GREEN=("#4A8C5E","#EAF3EC","#DDEEE1","#3f7c50","#E7F1EA")
ORANGE=("#C6803A","#FDF4EA","#F7E6D2","#B06A2C","#FBEEE0")
sp_y=150; fq_y=330
sp_right,_=align_row(sp_y, GREEN, "Spatial Feature","F_s  (B x C x H x W)","Aligned F_s","Feature Alignment (Spatial)")
fq_right,_=align_row(fq_y, ORANGE,"Frequency Feature","F_f  (B x C x H x W)","Aligned F_f","Feature Alignment (Frequency)")

# ---------- gated fusion (centered, symmetric, orthogonal connectors) ----------
PUR="#7A5BA6"; PLIGHT="#EDE8F3"
def ln(pts,color="#444",sw=1.9):
    p=" ".join(f"{a},{b}" for a,b in pts)
    parts.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}"/>')
def pla(pts,color="#444",sw=1.9):
    p=" ".join(f"{a},{b}" for a,b in pts)
    parts.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" marker-end="url(#a)"/>')
afs_cx=sp_right-100; aff_cx=fq_right-100
afs_by=sp_y+BH/2;   aff_by=fq_y+BH/2
CX=W/2                                   # gate computation centred
gsw=186; gsx=CX-gsw/2; gh2=54
gcY=452; gvY=524; gsY=596
cbox(gsx,gcY,gsw,"Concatenate","[F_s || F_f]  (2C)",PLIGHT,PUR,h=gh2)
cbox(gsx,gvY,gsw,"1x1 Conv *","2C -> C",PLIGHT,PUR,h=gh2)
cbox(gsx,gsY,gsw,"Sigmoid","g",PLIGHT,PUR,h=gh2)
arr(CX,gcY+gh2,CX,gvY); arr(CX,gvY+gh2,CX,gsY)
mLx,mRx,mY,mr=CX-162,CX+162,726,21
circ(mLx,mY,mr,"\u2299"); circ(mRx,mY,mr,"\u2299")
rect(mRx-36,640,72,34,PLIGHT,PUR,rx=7); txt(mRx,657,"1 - g",F_ANN,"#1c1c1c",weight="700")
adx,ady=CX,814; circ(adx,ady,22,"\u2295")
cbox(CX-96,882,192,"Fused Feature","B x C x H x W","#F0E7FA","#6C46A6",h=58)
arr(CX,ady+22,CX,882)
# --- feature rails (orthogonal, routed down the clear gaps beside the
#     frequency row so no connector crosses a node; symmetric concat taps) ---
sR=250; fR=398; tapY=430
xLc=267; xRc=mRx+42            # left corridor (freq input|group gap); right corridor (after freq group)
# F_s: down the left corridor -> left multiplier, with a branch into concatenate
ln([(afs_cx,afs_by),(afs_cx,sR),(xLc,sR),(xLc,mY)]); pla([(xLc,mY),(mLx-mr,mY)])
pla([(xLc,tapY),(CX-60,tapY),(CX-60,gcY)])                                                   # F_s -> concat (left)
txt(afs_cx+10,afs_by+24,"F_s",F_ANN,"#3f7c50",anchor="start",style="italic",weight="700")
# F_f: down the right corridor -> right multiplier, with a branch into concatenate
ln([(aff_cx,aff_by),(aff_cx,fR),(xRc,fR),(xRc,mY)]); pla([(xRc,mY),(mRx+mr,mY)])
pla([(xRc,tapY),(CX+60,tapY),(CX+60,gcY)])                                                   # F_f -> concat (right)
txt(aff_cx+10,aff_by+24,"F_f",F_ANN,"#B06A2C",anchor="start",style="italic",weight="700")
# --- gate g -> left mul (top) ; g -> 1-g -> right mul (top) ---
gbY=668
ln([(CX,gsY+gh2),(CX,gbY)])
pla([(CX,gbY),(mLx,gbY),(mLx,mY-mr)]); txt(mLx+9,gbY-7,"g",F_ANN,"#6C46A6",anchor="start",style="italic",weight="700")
ln([(CX,gbY),(mRx-36,gbY)]); pla([(mRx,674),(mRx,mY-mr)])
# --- products -> adder ---
pla([(mLx,mY+mr),(mLx,ady),(adx-22,ady)]); pla([(mRx,mY+mr),(mRx,ady),(adx+22,ady)])

# ---------- formula box ----------
fbY=958
rect(40, fbY, W-80, 94, "#FCFCFC", "#DADADA", sw=1.1, rx=10)
txt(60, fbY+31, "Gate:  g = \u03c3( 1x1 Conv( [ F_s || F_f ] ) )        Fusion:  F_fused = g \u2299 F_s + (1 \u2212 g) \u2299 F_f", F_FORM, "#333", anchor="start", font=MONO)
txt(60, fbY+58, "*  learnable (trainable parameters).  InstanceNorm2d uses affine=False (parameter-free);", F_LEG, "#444", anchor="start", weight="600")
txt(60, fbY+79, "ReLU, Sigmoid, concatenation, \u2299 (element-wise multiply) and \u2295 (element-wise add) are parameter-free.", F_LEG, "#444", anchor="start", weight="600")

# ---------- legend ----------
ly=1094
def sw(x,fill,stroke,label):
    rect(x,ly-13,19,16,fill,stroke,sw=1.2,rx=3); txt(x+27,ly,label,F_LEG,"#555",anchor="start")
    return x+27+len(label)*8.8+42
x=40
x=sw(x,"#DDEEE1","#3f7c50","Spatial path")
x=sw(x,"#F7E6D2","#B06A2C","Frequency path")
x=sw(x,"#EDE8F3","#7A5BA6","Gated Fusion")
txt(x, ly, "\u2299  element-wise multiply     \u2295  element-wise add     [ \u00b7 || \u00b7 ]  channel concatenation", F_LEG, "#555", anchor="start")

H=1130
defs='<defs><marker id="a" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0.5,0.8 L9,5 L0.5,9.2 Z" fill="#444"/></marker></defs>'
svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{SANS}">{defs}<rect width="{W}" height="{H}" fill="#FFFFFF"/>'+"".join(parts)+"</svg>"
sp=r"C:\Users\Soumojay\Desktop\New_Sm_P\SegmentationResearch\paper2\figures\Figure3_FeatureAlignment_GatedFusion.svg"
open(sp,"w",encoding="utf-8").write(svg)
pdf=fitz.open(sp).convert_to_pdf()
open(r"C:\Users\Soumojay\Desktop\New_Sm_P\SegmentationResearch\paper2\figures\Figure3_FeatureAlignment_GatedFusion.pdf","wb").write(pdf)
scale=TW_PT/W
print(f"viewBox {W}x{H} aspect {W/H:.3f} -> at {TW_PT:.0f}pt wide, height {TW_PT/(W/H):.0f}pt = {TW_PT/(W/H)/72:.2f}in")
for nm,f in [("title",F_TITLE),("box title",F_BT),("subtext",F_SUB),("group head",F_HEAD),("annot",F_ANN),("formula",F_FORM),("legend",F_LEG)]:
    print(f"   {nm:11s} {f}u -> {f*scale:.2f} pt")
prev=r"C:\Users\Soumojay\AppData\Local\Temp\claude\C--Users-Soumojay-Desktop-New-Sm-P\d9396353-c9a1-46c9-b57a-e755af7cb994\scratchpad\fig3_new.png"
fitz.open("pdf",pdf)[0].get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(prev)
print("preview:",prev)
