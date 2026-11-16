# -*- coding: utf-8 -*-
# Figure 4 redesign: multi-dataset qualitative comparison (ISIC 2016 + 2018 now;
# 2017 to be merged later once matched U-Net preds are regenerated).
# Columns fixed: Input | Ground Truth | U-Net | Attention U-Net | UNet++ |
#                TransUNet | Restored MAFF-Net | Proposed MAFF-Net
# Model->dir mapping dice-verified against Tables I/II (authoritative):
#   Restored MAFF-Net = maffnet_ISIC<yr>_final ; Proposed MAFF-Net = stabilized_maffnet_ISIC<yr>
import fitz, html, base64, os
EXP="experiments"
def P(*a): return os.path.join(EXP,*a)

def sset(yr, unet_dir, attn_dir, gtin_dir):
    return {
     "in":P(gtin_dir,"samples","input_{i}.png"), "gt":P(gtin_dir,"samples","gt_{i}.png"),
     "U-Net":P(unet_dir,"samples","pred_{i}.png"), "Attn":P(attn_dir,"samples","pred_{i}.png"),
     "UNet++":P(f"unetpp_ISIC_{yr}","samples","pred_{i}.png"),
     "TransUNet":P(f"transunet_ISIC_{yr}","samples","pred_{i}.png"),
     "Restored":P(f"maffnet_ISIC{yr}_final","samples","pred_{i}.png"),
     "Proposed":P(f"stabilized_maffnet_ISIC{yr}","samples","pred_{i}.png"),
    }
S2016=sset("2016","unet_ISIC_2016","attn_unet_ISIC_2016","maffnet_ISIC2016_final")
S2017=sset("2017","unet_ISIC2017_qual_matched","attn_unet_ISIC_2017","maffnet_ISIC2017_final")
S2018=sset("2018","unet_ISIC_2018","attention_unet_ISIC2018_final","maffnet_ISIC2018_final")

sections=[("ISIC 2016",[(S2016,1,"small lesion"),(S2016,2,"hair, fuzzy")]),
          ("ISIC 2017",[(S2017,0,"irregular border"),(S2017,3,"ink markings")]),
          ("ISIC 2018",[(S2018,1,"complex shape"),(S2018,4,"vignette")])]
cols=[("Input","in"),("Ground Truth","gt"),("U-Net","U-Net"),("Attention U-Net","Attn"),
      ("UNet++","UNet++"),("TransUNet","TransUNet"),("MAFF-Net [15]","Restored"),
      ("SpecGate-Net","Proposed")]
NPROP=len(cols)-1  # proposed column index (last)

CELL=132; G=3; LEFT=54; HH=48; DB=26; RG=3
x0=LEFT+6
def colx(c): return x0+c*(CELL+G)
W=colx(len(cols)-1)+CELL+14
SANS="Helvetica, Arial, sans-serif"
parts=[]
def esc(s): return html.escape(s,quote=False)
def txt(x,y,s,fs,fill="#1c1c1c",anchor="middle",weight="600",rot=None):
    tr=f' transform="rotate({rot} {x} {y})"' if rot is not None else ''
    parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{SANS}" font-size="{fs}" font-weight="{weight}" fill="{fill}"{tr}>{esc(s)}</text>')
def imgcell(x,y,path):
    with open(path,"rb") as f: b=base64.b64encode(f.read()).decode()
    parts.append(f'<image x="{x}" y="{y}" width="{CELL}" height="{CELL}" preserveAspectRatio="none" xlink:href="data:image/png;base64,{b}"/>')
    parts.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" fill="none" stroke="#9a9a9a" stroke-width="1"/>')

# ---- column headers (2-line where needed) ----
hdr2={"Ground Truth":("Ground","Truth"),"Attention U-Net":("Attention","U-Net"),
      "MAFF-Net [15]":("MAFF-Net","[15]"),"SpecGate-Net":("SpecGate-","Net")}
for c,(name,_) in enumerate(cols):
    cx=colx(c)+CELL/2; red=(c==NPROP)
    fill="#B00000" if red else "#1c1c1c"
    if name in hdr2:
        a,b=hdr2[name]; txt(cx,26,a,22,fill,weight="700"); txt(cx,46,b,22,fill,weight="700")
    else:
        txt(cx,38,name,22,fill,weight="700")

# ---- sections + rows ----
y=HH+6
row_ys=[]
for sec,rows in sections:
    # dataset band
    parts.append(f'<rect x="{LEFT}" y="{y}" width="{W-LEFT-8}" height="{DB}" rx="5" fill="#EEF1F5" stroke="#C9D2DC" stroke-width="1"/>')
    txt(LEFT+12,y+18,sec,20,"#33475b",anchor="start",weight="700")
    y+=DB
    grp_top=y
    for src,idx,tag in rows:
        for c,(name,key) in enumerate(cols):
            imgcell(colx(c),y,src[key].format(i=idx))
        txt(27,y+CELL/2,tag,16,"#555",anchor="middle",weight="600",rot=-90)  # rotated row tag
        y+=CELL+RG
    # red outline around Proposed column for this dataset group
    px=colx(NPROP)
    parts.append(f'<rect x="{px-2}" y="{grp_top-2}" width="{CELL+4}" height="{y-RG-grp_top+4}" fill="none" stroke="#B00000" stroke-width="2.6"/>')
    y+=6
H=int(y+2)

defs='<defs></defs>'
svg=(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
     f'viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{SANS}">{defs}'
     f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>'+"".join(parts)+"</svg>")
base=r"C:\Users\Soumojay\Desktop\New_Sm_P\SegmentationResearch\paper2\figures"
open(base+r"\Figure4_MultiDataset.svg","w",encoding="utf-8").write(svg)
pdf=fitz.open(base+r"\Figure4_MultiDataset.svg").convert_to_pdf()
open(base+r"\Figure4_MultiDataset.pdf","wb").write(pdf)
print(f"viewBox {W}x{H} aspect {W/H:.3f} -> at 516pt: {516/(W/H):.0f}pt = {516/(W/H)/72:.2f}in tall")
prev=r"C:\Users\Soumojay\AppData\Local\Temp\claude\C--Users-Soumojay-Desktop-New-Sm-P\d9396353-c9a1-46c9-b57a-e755af7cb994\scratchpad\fig4_new.png"
fitz.open("pdf",pdf)[0].get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(prev)
print("preview:",prev)
