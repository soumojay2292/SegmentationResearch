# -*- coding: utf-8 -*-
# Preprocess the SpecGate-Net LaTeX body for pandoc: resolve \cite -> [n],
# strip float environments to placeholder tokens (capturing captions/labels),
# and extract title/authors/abstract/keywords. Does NOT modify the manuscript.
import re, json, os

LX = r"C:\Users\Soumojay\Desktop\New_Sm_P\SegmentationResearch\paper2\latex"
SEC = os.path.join(LX, "sections")
OUT = r"C:\Users\Soumojay\AppData\Local\Temp\claude\C--Users-Soumojay-Desktop-New-Sm-P\d9396353-c9a1-46c9-b57a-e755af7cb994\scratchpad\docx_work"

# citation key -> number (from main.bbl order)
BBL_ORDER = ["ref1","ref2","ref3","ref4","ref5","ref6","ref7","ref8","ref9","ref10",
"ref11","ref12","ref13","ref14","ref15","ref16","ref17","ref18","ref24","ref25","ref26",
"ref27","ref19","ref28","ref29","ref31","ref32","ref20","ref21","ref22","ref33","ref23","ref30"]
KEY2NUM = {k:i+1 for i,k in enumerate(BBL_ORDER)}

def read(p): return open(p, encoding="utf-8").read()

def brace_match(s, i):
    # s[i]=='{'; return index after matching '}'
    d=0
    for j in range(i,len(s)):
        if s[j]=='{': d+=1
        elif s[j]=='}':
            d-=1
            if d==0: return j+1
    return len(s)

ROMAN=["I","II","III","IV","V","VI","VII","VIII"]
SEC_FILE_ORDER=["introduction","related_work","methodology","experiments","results","discussion","conclusion"]
FIG_NUM={"fig:architecture":"1","fig:freqbranch":"2","fig:fafusion":"3","fig:qualitative":"4"}
TAB_NUM={"tab:metrics":"I","tab:benchmark":"II","tab:boundary":"III","tab:stats":"IV","tab:ablation":"V"}
EQ_NUM={"eq:1":"1","eq:2":"2","eq:3":"3","eq:4a":"4a","eq:4b":"4b","eq:5":"5","eq:6":"6",
        "eq:7":"7","eq:12":"8","eq:13":"9"}

def build_refmap():
    m={}
    m.update(FIG_NUM); m.update(TAB_NUM); m.update(EQ_NUM)
    for idx,fn in enumerate(SEC_FILE_ORDER):
        roman=ROMAN[idx]
        txt=read(os.path.join(SEC,fn+".tex"))
        # walk lines, track current subsection letter; associate labels with most recent heading
        letter=None; li=0
        lines=txt.splitlines()
        cur="sec"   # after \section, before any \subsection
        for ln in lines:
            if re.search(r'\\section\*?\{', ln): cur="sec";
            if re.search(r'\\subsection\*?\{', ln):
                li+=1; letter=chr(ord('A')+li-1); cur="sub"
            for lm in re.finditer(r'\\label\{(sec:[^}]*|subsec:[^}]*)\}', ln):
                key=lm.group(1)
                if key.startswith("sec:"): m[key]=roman
                else: m[key]=roman+"-"+(letter or "A")
    return m

REFMAP=build_refmap()

def repl_ref(text):
    def one(mch):
        key=mch.group(1); return REFMAP.get(key, "?")
    # \eqref -> (n)  (parenthesised); handle tilde spacing
    text=re.sub(r'~\\eqref\{([^}]*)\}', lambda m: " ("+REFMAP.get(m.group(1),"?")+")", text)
    text=re.sub(r'\\eqref\{([^}]*)\}', lambda m: "("+REFMAP.get(m.group(1),"?")+")", text)
    # tilde-before-ref: keep a normal space
    text=re.sub(r'~\\ref\{([^}]*)\}', lambda m: " "+REFMAP.get(m.group(1),"?"), text)
    text=re.sub(r'\\ref\{([^}]*)\}', one, text)
    return text

def repl_cite(text):
    out=[]; i=0
    while True:
        m=re.search(r'\\cite\{', text[i:])
        if not m: out.append(text[i:]); break
        s=i+m.start(); b=i+m.end()-1  # position of '{'
        end=brace_match(text,b)
        keys=text[b+1:end-1]
        nums=[KEY2NUM.get(k.strip(),'?') for k in keys.split(',')]
        rep=", ".join(f"[{n}]" for n in nums)
        out.append(text[i:s]); out.append(rep); i=end
    return "".join(out)

def extract_caption(block):
    m=re.search(r'\\caption\{', block)
    if not m: return ""
    b=m.end()-1; end=brace_match(block,b)
    cap=block[b+1:end-1]
    return cap

def strip_floats(text, floats):
    # remove table*/table/figure*/figure envs, insert placeholder token
    envs=["table\\*","figure\\*","table","figure"]
    for env in envs:
        pat=re.compile(r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}', re.DOTALL)
        while True:
            m=pat.search(text)
            if not m: break
            block=m.group(0)
            cap=extract_caption(block)
            lab=""
            lm=re.search(r'\\label\{([^}]*)\}', block)
            if lm: lab=lm.group(1)
            img=""
            im=re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}', block)
            if im: img=im.group(1)
            kind = 'figure' if 'figure' in env else 'table'
            tok=f"@@FLOAT_{len(floats)}@@"
            floats.append({"i":len(floats),"kind":kind,"label":lab,"caption":cap.strip(),"img":img})
            text=text[:m.start()]+"\n\n"+tok+"\n\n"+text[m.end():]
    return text

def main():
    floats=[]
    # ---- abstract ----
    ab=read(os.path.join(SEC,"abstract.tex"))
    am=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', ab, re.DOTALL)
    abstract=repl_cite(am.group(1).strip()) if am else ""
    # ---- keywords (from main.tex) ----
    mt=read(os.path.join(LX,"main.tex"))
    km=re.search(r'\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}', mt, re.DOTALL)
    keywords=" ".join(km.group(1).split()) if km else ""
    # ---- title & authors (from main.tex) ----
    tm=re.search(r'\\title\{(.*?)\}\s*\n', mt, re.DOTALL)
    title=tm.group(1).replace("\\\\","").strip() if tm else ""
    title=re.sub(r'\s+',' ',title)
    # authors block: capture between \author{ ... } (best effort)
    au=""
    am2=re.search(r'\\author\{', mt)
    if am2:
        end=brace_match(mt, am2.end()-1)
        au=mt[am2.end()-1:end]

    # ---- body sections in order ----
    order=["introduction","related_work","methodology","experiments","results","discussion","conclusion"]
    body=""
    for s in order:
        body += "\n\n" + read(os.path.join(SEC,s+".tex"))
    body = strip_floats(body, floats)
    body = repl_cite(body)
    body = repl_ref(body)
    body = re.sub(r'\\label\{[^}]*\}', '', body)
    body = re.sub(r'\\IEEE[a-zA-Z]+', '', body)
    # clean captions too
    for fl in floats:
        fl["caption"]=repl_ref(repl_cite(fl["caption"]))
    abstract2=repl_ref(abstract)

    with open(os.path.join(OUT,"body.tex"),"w",encoding="utf-8") as f:
        f.write(body)
    meta={"title":title,"authors_raw":au,"abstract":abstract2,"keywords":keywords,"floats":floats}
    with open(os.path.join(OUT,"meta.json"),"w",encoding="utf-8") as f:
        json.dump(meta,f,indent=1,ensure_ascii=False)
    print("title:",title[:80])
    print("keywords:",keywords[:80])
    print("abstract chars:",len(abstract))
    print("floats:")
    for fl in floats: print("  ",fl["i"],fl["kind"],fl["label"],"| img=",fl["img"],"| cap:",fl["caption"][:60])
    # citation sanity: any unresolved '?'
    allt=body+abstract
    print("unresolved cites:", allt.count("[?]"))

main()
