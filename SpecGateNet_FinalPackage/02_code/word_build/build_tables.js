// Build a standalone .docx with native Word Tables I–V for the SpecGate-Net paper.
// Table I: editable Word equations (fractions). Tables II–V: native tables with
// merged multi-level headers, dataset groupings, bold, units, footnotes.
const docx = require("docx");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, BorderStyle, VerticalAlign, Math, MathRun, MathFraction, PageOrientation
} = docx;
const fs = require("fs");

const FONT = "Times New Roman";
const SZ = 16;      // 8pt table body
const SZN = 15;     // 7.5pt notes
const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const RULE = { style: BorderStyle.SINGLE, size: 8, color: "000000" };   // top/bottom rules
const THIN = { style: BorderStyle.SINGLE, size: 6, color: "000000" };   // header + group rules

// ---- text run ----
function run(text, o = {}) {
  return new TextRun({ text, font: FONT, size: o.size || SZ, bold: !!o.bold,
    italics: !!o.italic, superScript: !!o.sup, subScript: !!o.sub });
}
// paragraph of runs
function P(children, align = AlignmentType.LEFT, o = {}) {
  return new Paragraph({ children, alignment: align,
    spacing: { before: o.before ?? 10, after: o.after ?? 10, line: 240 } });
}
// simple text paragraph
function PT(text, align = AlignmentType.LEFT, o = {}) { return P([run(text, o)], align, o); }

// ---- cell ----
function CELL(children, o = {}) {
  const b = { top: o.top || NB, bottom: o.bottom || NB, left: NB, right: NB };
  return new TableCell({
    children, width: { size: o.w, type: WidthType.DXA },
    columnSpan: o.span, rowSpan: o.vspan,
    verticalAlign: VerticalAlign.CENTER, borders: b,
    margins: { top: 20, bottom: 20, left: 60, right: 60 },
  });
}
// header/data helpers -> single paragraph cell
function hc(text, w, o = {}) {   // header cell (bold, centered)
  return CELL([P([run(text, { bold: true })], o.align || AlignmentType.CENTER)],
    { w, span: o.span, vspan: o.vspan, top: o.top, bottom: o.bottom });
}
function dc(spec, w, o = {}) {   // data cell: spec = string | {t,b} | run-array
  let kids;
  if (Array.isArray(spec)) kids = [P(spec, o.align || AlignmentType.CENTER)];
  else if (spec && typeof spec === "object") kids = [P([run(spec.t, { bold: spec.b })], o.align || AlignmentType.CENTER)];
  else kids = [P([run(String(spec))], o.align || AlignmentType.CENTER)];
  return CELL(kids, { w, span: o.span, vspan: o.vspan, top: o.top, bottom: o.bottom, align: o.align });
}
const B = (t) => ({ t: String(t), b: true });   // bold value

// caption (IEEE style): "TABLE N" then title, centered
function caption(n, title) {
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240, after: 0 },
      children: [run("TABLE " + n, { size: SZ })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 60 },
      children: [run(title, { size: SZ })] }),
  ];
}
function note(children) {
  return new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 40, after: 160, line: 220 },
    children });
}
// "< 10^-3 *" runs
function lt3() { return [run("< 10"), run("−3", { sup: true }), run("*", { sup: true })]; }
// p value possibly ending with "*" (significance)
function pval(s) {
  if (s === "LT3*") return lt3();
  if (s.endsWith("*")) return [run(s.slice(0, -1)), run("*", { sup: true })];
  return [run(s)];
}

const TWIDTH = 9360; // full text width (Letter, 1in margins)

// ============================ TABLE I ============================
function eqFrac(numT, denT) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 20, after: 20 },
    children: [ new Math({ children: [ new MathFraction({
      numerator: [ new MathRun(numT) ], denominator: [ new MathRun(denT) ] }) ] }) ] });
}
function tableI() {
  const w = [1900, 2700, 2400]; // sums 7000, centered
  const rows = [];
  rows.push(new TableRow({ children: [
    hc("Metric", w[0], { align: AlignmentType.LEFT, top: RULE, bottom: THIN }),
    hc("Expression", w[1], { top: RULE, bottom: THIN }),
    hc("Purpose", w[2], { align: AlignmentType.LEFT, top: RULE, bottom: THIN }),
  ]}));
  const data = [
    ["Accuracy", ["TP+TN", "TP+FP+TN+FN"], "overall pixel correctness"],
    ["Dice", ["2 TP", "2 TP+FP+FN"], "region overlap (primary)"],
    ["IoU (Jaccard)", ["TP", "TP+FP+FN"], "stricter region overlap"],
    ["Sensitivity", ["TP", "TP+FN"], "recall of lesion pixels"],
    ["Specificity", ["TN", "TN+FP"], "background correctness"],
  ];
  data.forEach((r, i) => {
    const last = i === data.length - 1;
    const bb = last ? RULE : undefined;
    rows.push(new TableRow({ children: [
      CELL([P([run(r[0])], AlignmentType.LEFT)], { w: w[0], bottom: bb }),
      CELL([eqFrac(r[1][0], r[1][1])], { w: w[1], bottom: bb }),
      CELL([P([run(r[2])], AlignmentType.LEFT)], { w: w[2], bottom: bb }),
    ]}));
  });
  return new Table({ columnWidths: w, width: { size: 7000, type: WidthType.DXA },
    alignment: AlignmentType.CENTER, borders: {
      top: NB, bottom: NB, left: NB, right: NB, insideHorizontal: NB, insideVertical: NB }, rows });
}

// ===================== TABLES II / III (grouped) =====================
function groupedTable(headers, widths, groups, opts) {
  // headers: array of header strings; widths: col widths; groups: [{name, rows:[[method, v1..]]}]
  const rows = [];
  // header row
  rows.push(new TableRow({ children: headers.map((h, i) =>
    hc(h.t, widths[i], { align: h.align, top: RULE, bottom: THIN })) }));
  groups.forEach((g, gi) => {
    const topb = gi === 0 ? undefined : THIN;   // group separator rule
    g.rows.forEach((r, ri) => {
      const isLastRow = gi === groups.length - 1 && ri === g.rows.length - 1;
      const bb = isLastRow ? RULE : undefined;
      const cells = [];
      if (ri === 0) cells.push(hc(g.name, widths[0], { vspan: g.rows.length, top: topb, bottom: bb }));
      // method
      cells.push(dc(r[0], widths[1], { align: AlignmentType.LEFT, top: ri === 0 ? topb : undefined, bottom: bb }));
      for (let c = 1; c < r.length; c++) {
        cells.push(dc(r[c], widths[c + 1], { top: ri === 0 ? topb : undefined, bottom: bb }));
      }
      rows.push(new TableRow({ children: cells }));
    });
  });
  return new Table({ columnWidths: widths, width: { size: TWIDTH, type: WidthType.DXA },
    alignment: AlignmentType.CENTER, borders: {
      top: NB, bottom: NB, left: NB, right: NB, insideHorizontal: NB, insideVertical: NB }, rows });
}

function tableII() {
  const widths = [1150, 2450, 1152, 1152, 1152, 1152, 1152];
  const headers = [
    { t: "Dataset", align: AlignmentType.LEFT }, { t: "Method", align: AlignmentType.LEFT },
    { t: "Dice" }, { t: "IoU" }, { t: "Accuracy" }, { t: "Sensitivity" }, { t: "Specificity" }];
  const groups = [
    { name: "ISIC 2016", rows: [
      ["U-Net","0.7874","0.6808","0.8951","0.7649","0.9609"],
      ["Attention U-Net","0.8466","0.7443","0.9178","0.8780","0.9495"],
      ["UNet++","0.8141","0.7140","0.9055","0.7996","0.9647"],
      ["TransUNet","0.8365","0.7349","0.9161","0.8077",B("0.9700")],
      ["MAFF-Net [15]",B("0.9183"),B("0.8555"),"0.9541",B("0.9414"),"0.9545"],
      ["SpecGate-Net","0.9169","0.8517",B("0.9548"),"0.9220","0.9689"] ]},
    { name: "ISIC 2017", rows: [
      ["U-Net","0.7229","0.6152","0.9031","0.7211","0.9562"],
      ["Attention U-Net","0.8097","0.7159","0.9331","0.8223","0.9737"],
      ["UNet++","0.7884","0.6959","0.9291","0.7776","0.9784"],
      ["TransUNet","0.8683","0.7843","0.9561","0.8418",B("0.9870")],
      ["MAFF-Net [15]",B("0.9116"),B("0.8447"),B("0.9700"),B("0.9171"),"0.9807"],
      ["SpecGate-Net","0.9096","0.8421","0.9682","0.9069","0.9842"] ]},
    { name: "ISIC 2018", rows: [
      ["U-Net","0.6866","0.5714","0.8732","0.7766","0.8948"],
      ["Attention U-Net","0.7662","0.6641","0.9207","0.8352","0.9497"],
      ["UNet++","0.6978","0.5878","0.8912","0.7692","0.9298"],
      ["TransUNet","0.8259","0.7265","0.9259","0.8929","0.9315"],
      ["MAFF-Net [15]",B("0.8844"),B("0.8080"),B("0.9584"),B("0.8973"),"0.9730"],
      ["SpecGate-Net","0.8701","0.7900","0.9493","0.8733",B("0.9768")] ]},
  ];
  return groupedTable(headers, widths, groups);
}

function tableIII() {
  const widths = [1150, 2450, 1440, 1440, 1440, 1440];
  const headers = [
    { t: "Dataset", align: AlignmentType.LEFT }, { t: "Method", align: AlignmentType.LEFT },
    { t: "HD95 (px)" }, { t: "ASSD (px)" }, { t: "Boundary IoU" }, { t: "n_valid / n" }];
  const groups = [
    { name: "ISIC 2016", rows: [
      ["U-Net","61.976","22.527","0.3195","192 / 192"],
      ["Attention U-Net","50.465","18.221","0.2436","192 / 192"],
      ["UNet++","48.244","18.341","0.3283","190 / 192"],
      ["TransUNet","43.779","16.368","0.2898","192 / 192"],
      ["MAFF-Net [15]",B("21.979"),B("8.336"),B("0.4518"),"192 / 192"],
      ["SpecGate-Net","23.329","8.510","0.4514","192 / 192"] ]},
    { name: "ISIC 2017", rows: [
      ["U-Net","60.572","23.027","0.3560","194 / 200"],
      ["Attention U-Net","40.745","17.004","0.3749","198 / 200"],
      ["UNet++","42.561","15.603","0.4244","197 / 200"],
      ["TransUNet","26.750","9.710","0.4585","200 / 200"],
      ["MAFF-Net [15]",B("16.788"),B("6.565"),"0.5255","200 / 200"],
      ["SpecGate-Net","16.869","6.645",B("0.5277"),"200 / 200"] ]},
    { name: "ISIC 2018", rows: [
      ["U-Net","72.867","29.454","0.2958","388 / 392"],
      ["Attention U-Net","47.746","19.645","0.3035","385 / 392"],
      ["UNet++","63.710","25.348","0.2672","383 / 392"],
      ["TransUNet","41.836","15.092","0.3406","392 / 392"],
      ["MAFF-Net [15]",B("20.570"),B("8.199"),B("0.4701"),"392 / 392"],
      ["SpecGate-Net","22.554","9.240","0.4611","391 / 392"] ]},
  ];
  return groupedTable(headers, widths, groups);
}

// ===================== TABLE IV (multi-level header) =====================
function tableIV() {
  const w = [1300, 1200, 2000, 1860, 3000];
  const rows = [];
  rows.push(new TableRow({ children: [
    hc("Dataset", w[0], { align: AlignmentType.LEFT, vspan: 2, top: RULE, bottom: THIN }),
    hc("Metric", w[1], { align: AlignmentType.LEFT, vspan: 2, top: RULE, bottom: THIN }),
    CELL([P([run("vs. MAFF-Net [15]", { bold: true })], AlignmentType.CENTER)],
      { w: w[2] + w[3] + w[4], span: 3, top: RULE, bottom: THIN }),
  ]}));
  const h2 = (runs, ww) => CELL([P(runs, AlignmentType.CENTER)], { w: ww, bottom: THIN });
  rows.push(new TableRow({ children: [
    h2([run("median ", { bold: true }), run("\u0394", { bold: true })], w[2]),
    h2([run("r", { bold: true, italic: true })], w[3]),
    h2([run("p", { bold: true, italic: true }), run("Holm", { bold: true, sub: true })], w[4]),
  ]}));
  const groups = [
    { name: "ISIC 2016", rows: [
      ["Dice","+0.0009","0.00","0.962"],
      ["IoU","+0.0017","0.00","0.968"],
      ["HD95","\u22120.0598","\u22120.02","0.789"],
      ["ASSD","\u22120.1204","\u22120.03","0.721"],
      ["B-IoU","+0.0100","+0.08","0.357"] ]},
    { name: "ISIC 2017", rows: [
      ["Dice","+0.0007","+0.05","0.580"],
      ["IoU","+0.0011","+0.05","0.550"],
      ["HD95","\u22120.0711","+0.04","0.631"],
      ["ASSD","\u22120.0465","\u22120.04","0.613"],
      ["B-IoU","+0.0055","+0.10","0.238"] ]},
    { name: "ISIC 2018", rows: [
      ["Dice","\u22120.0029","\u22120.14","0.015*"],
      ["IoU","\u22120.0053","\u22120.14","0.017*"],
      ["HD95","+0.4843","+0.23","LT3*"],
      ["ASSD","+0.1473","+0.16","0.006*"],
      ["B-IoU","\u22120.0104","\u22120.10","0.089"] ]},
  ];
  groups.forEach((g, gi) => {
    const topb = gi === 0 ? undefined : THIN;
    g.rows.forEach((r, ri) => {
      const isLast = gi === groups.length - 1 && ri === g.rows.length - 1;
      const bb = isLast ? RULE : undefined;
      const cells = [];
      if (ri === 0) cells.push(hc(g.name, w[0], { align: AlignmentType.LEFT, vspan: 5, top: topb, bottom: bb }));
      cells.push(dc(r[0], w[1], { align: AlignmentType.LEFT, top: ri === 0 ? topb : undefined, bottom: bb }));
      cells.push(dc([run(r[1])], w[2], { top: ri === 0 ? topb : undefined, bottom: bb }));
      cells.push(dc([run(r[2])], w[3], { top: ri === 0 ? topb : undefined, bottom: bb }));
      cells.push(dc(pval(r[3]), w[4], { top: ri === 0 ? topb : undefined, bottom: bb }));
      rows.push(new TableRow({ children: cells }));
    });
  });
  return new Table({ columnWidths: w, width: { size: TWIDTH, type: WidthType.DXA },
    alignment: AlignmentType.CENTER, borders: {
      top: NB, bottom: NB, left: NB, right: NB, insideHorizontal: NB, insideVertical: NB }, rows });
}

// ===================== TABLE V (ablation) =====================
function tableV() {
  const w = [2760, 1100, 1100, 1100, 1100, 1100, 1100];
  const headers = ["Configuration","Dice","IoU","Accuracy","HD95 (px)","ASSD (px)","Boundary IoU"];
  const rows = [];
  rows.push(new TableRow({ children: headers.map((h, i) =>
    hc(h, w[i], { align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, top: RULE, bottom: THIN })) }));
  const data = [
    ["Full SpecGate-Net","0.8701","0.7900","0.9493",B("22.554"),B("9.240"),B("0.4611")],
    ["without Frequency Branch","0.8708","0.7868","0.9496","23.574","9.546","0.4416"],
    ["without Feature Alignment Module","0.8649","0.7811","0.9502","23.676","9.869","0.4326"],
    ["without Gated Fusion Module",B("0.8725"),B("0.7907"),B("0.9520"),"25.546","9.488","0.4508"],
  ];
  data.forEach((r, i) => {
    const bb = i === data.length - 1 ? RULE : undefined;
    const cells = [ dc(r[0], w[0], { align: AlignmentType.LEFT, bottom: bb }) ];
    for (let c = 1; c < r.length; c++) cells.push(dc(r[c], w[c], { bottom: bb }));
    rows.push(new TableRow({ children: cells }));
  });
  return new Table({ columnWidths: w, width: { size: TWIDTH, type: WidthType.DXA },
    alignment: AlignmentType.CENTER, borders: {
      top: NB, bottom: NB, left: NB, right: NB, insideHorizontal: NB, insideVertical: NB }, rows });
}

// ============================ assemble ============================
const children = [];
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
  children: [ new TextRun({ text: "SpecGate-Net — Editable Tables (I–V)", font: FONT, size: 22, bold: true }) ] }));

children.push(...caption("I", "Region-Based Evaluation Metrics"));
children.push(tableI());
children.push(note([ run("TP, FP, TN, FN denote per-pixel true/false positive/negative counts. Standard metric definitions follow [30].", { size: SZN }) ]));

children.push(...caption("II", "Region-Based Segmentation Performance on the Held-Out Test Subsets (ISIC 2016/2017/2018)"));
children.push(tableII());
children.push(note([ run("The highest value in each column, within each dataset, is shown in bold. The MAFF-Net [15] rows report a reproduction of the published architecture trained and evaluated under the protocol of this study, not the results reported in [15].", { size: SZN }) ]));

children.push(...caption("III", "Boundary-Based Segmentation Performance on the Held-Out Test Subsets"));
children.push(tableIII());
children.push(note([ run("HD95 and ASSD are in pixels of the 384×384 evaluation grid (lower is better); higher Boundary IoU is better. n", { size: SZN }), run("valid", { size: SZN, sub: true }), run(" gives the number of images contributing to the distance measures. The best value in each column, within each dataset, is in bold. The MAFF-Net [15] rows are a reproduction, as in Table II.", { size: SZN }) ]));

children.push(...caption("IV", "Paired Statistical Comparison of SpecGate-Net Against the Two Prior-Work Reproductions"));
children.push(tableIV());
children.push(note([ run("Each entry gives the matched-pairs rank-biserial effect size ", { size: SZN }), run("r", { size: SZN, italic: true }), run(" and the Holm-adjusted Wilcoxon signed-rank ", { size: SZN }), run("p", { size: SZN, italic: true }), run("-value (*: significant at α = 0.05). median Δ = SpecGate-Net − MAFF-Net [15]; a positive Δ favours SpecGate-Net for Dice, IoU, and Boundary IoU, and MAFF-Net [15] for HD95 and ASSD. The Friedman omnibus test rejects equality for every dataset and metric (all ", { size: SZN }), run("p", { size: SZN, italic: true }), run(" < 10", { size: SZN }), run("−39", { size: SZN, sup: true }), run(").", { size: SZN }) ]));

children.push(...caption("V", "Leave-One-Out Architecture Ablation of SpecGate-Net (ISIC 2018 Test Subset)"));
children.push(tableV());
children.push(note([ run("HD95 and ASSD are in pixels of the 384×384 evaluation grid (lower is better); higher Dice, IoU, accuracy, and Boundary IoU are better. The best value in each column is shown in bold.", { size: SZN }) ]));

const doc = new Document({
  creator: "SpecGate-Net", title: "SpecGate-Net Tables",
  styles: { default: { document: { run: { font: FONT, size: SZ } } } },
  sections: [{
    properties: { page: {
      size: { width: 12240, height: 15840 },
      margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("SpecGateNet_Tables.docx", buf);
  console.log("WROTE SpecGateNet_Tables.docx", buf.length, "bytes");
});
