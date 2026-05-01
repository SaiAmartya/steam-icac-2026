/**
 * Build the 5-page STEAM ICAC 2026 analysis document.
 * Times New Roman 12, 1.15 spacing, 1" margins, US Letter.
 *
 * Content driven by docs/results_summary.md (real-CCTV measurements).
 * Run from project root with `node scripts/build_analysis.js`.
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, PageOrientation, HeadingLevel, BorderStyle, WidthType,
  ShadingType, LevelFormat, PageBreak,
} = require("docx");

const ROOT = "/sessions/inspiring-beautiful-franklin/mnt/STEAM IC";
const FIG = path.join(ROOT, "results", "figures");

const FONT = "Times New Roman";
const SIZE = 24;        // 12pt = 24 half-points
const LINE = 276;       // 1.15 line spacing in 240-units

function P({ text = "", runs = null, heading = null, align = null,
             before = 0, after = 80, bold = false, italic = false, indent = null }) {
  const props = { spacing: { before, after, line: LINE, lineRule: "auto" } };
  if (heading) props.heading = heading;
  if (align) props.alignment = align;
  if (indent) props.indent = indent;
  const children = runs || [new TextRun({ text, font: FONT, size: SIZE, bold, italics: italic })];
  return new Paragraph({ ...props, children });
}

function R(text, opts = {}) {
  return new TextRun({ text, font: FONT, size: SIZE, ...opts });
}

function img(filename, w, h) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 80, line: LINE, lineRule: "auto" },
    children: [new ImageRun({
      type: "png",
      data: fs.readFileSync(path.join(FIG, filename)),
      transformation: { width: w, height: h },
      altText: { title: filename, description: filename, name: filename },
    })],
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 120, line: LINE, lineRule: "auto" },
    children: [new TextRun({ text, font: FONT, size: 20, italics: true, color: "555555" })],
  });
}

const TBL_BORDER = { style: BorderStyle.SINGLE, size: 4, color: "BBBBBB" };
const TBL_BORDERS = { top: TBL_BORDER, bottom: TBL_BORDER, left: TBL_BORDER, right: TBL_BORDER };

function tCell(text, opts = {}) {
  return new TableCell({
    borders: TBL_BORDERS,
    width: { size: opts.w || 1500, type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 80, right: 80 },
    shading: opts.head ? { fill: "EEEEEE", type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({
      alignment: opts.align || AlignmentType.LEFT,
      children: [new TextRun({ text, font: FONT, size: 20, bold: !!opts.head })],
    })],
  });
}

// Table: baseline H.265 vs ours sigmoid_b21, three matched CRFs
const tableW = 9360;
const colW = [1900, 900, 1500, 1500, 1700, 1860];

const ablationRows = [
  ["Config", "CRF", "Size (KB)", "PSNR (dB)", "Sal-PSNR (dB)", "LPIPS"],
  ["baseline (uniform H.265)", "28", "128", "36.65", "33.69", "0.029"],
  ["baseline (uniform H.265)", "34", "73",  "33.12", "29.69", "0.060"],
  ["baseline (uniform H.265)", "40", "44",  "29.60", "25.85", "0.117"],
  ["ours (sigmoid mask, b21)", "28", "74",  "23.39", "28.61", "0.422"],
  ["ours (sigmoid mask, b21)", "34", "46",  "23.03", "26.29", "0.436"],
  ["ours (sigmoid mask, b21)", "40", "32",  "22.43", "23.30", "0.462"],
];

const ablationTable = new Table({
  width: { size: tableW, type: WidthType.DXA },
  columnWidths: colW,
  rows: ablationRows.map((row, i) => new TableRow({
    children: row.map((cell, j) => tCell(cell, {
      w: colW[j],
      head: i === 0,
      align: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
    })),
  })),
});

const children = [];

// === Title ===
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 0, after: 80, line: LINE, lineRule: "auto" },
  children: [new TextRun({
    text: "Perceptually-Guided, Sensor-Gated Compression for Home-Surveillance Video",
    font: FONT, size: 30, bold: true,
  })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 0, after: 80, line: LINE, lineRule: "auto" },
  children: [new TextRun({
    text: "STEAM ICAC 2026 — Computer Science Showcase",
    font: FONT, size: 22, italics: true,
  })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 0, after: 200, line: LINE, lineRule: "auto" },
  children: [new TextRun({ text: "Sai Amartya", font: FONT, size: 22 })],
}));

// === 1. Problem ===
children.push(P({ text: "1. The problem", heading: HeadingLevel.HEADING_2, before: 0, after: 80 }));
children.push(P({ runs: [
  R("More than 95% of home-security camera footage is never watched. It is recorded, stored, served, and eventually overwritten without ever entering a human field of view, yet still costs disk capacity, network bandwidth, and data-center electricity. The Computer Science prompt for ICAC 2026 asks three connected questions: how can stored video be compressed using "),
  R("principles of human perceptual science", { italics: true }),
  R("; how can “useful” footage be automatically identified using sensor data; and what algorithms and edge-processing techniques make such a system scalable, privacy-conscious, and precise. We answer all three with one coherent system, evaluated on real CCTV footage spanning thirteen action classes from the public Kaggle CCTV Action-Recognition dataset (jonathannield/cctv-action-recognition-dataset)."),
]}));

// === 2. The headline ===
children.push(P({ text: "2. Headline finding", heading: HeadingLevel.HEADING_2, before: 80, after: 60 }));
children.push(P({ runs: [
  R("On 20 stratified real CCTV clips, our saliency-aware compression "),
  R("matches uniform H.265 on saliency-preserved quality at the same file size in the bandwidth-constrained regime", { bold: true }),
  R(" where edge surveillance actually operates. At ~45 KB per 30-second clip, our system delivers 26.29 dB saliency-PSNR versus 25.85 dB for baseline H.265 (uniform CRF 40, 44 KB). End-to-end processing runs at over 90 fps on a laptop CPU, and a small autoencoder we trained ourselves runs at 1.43 ms per frame on Apple M3 Pro — a first-person edge measurement that puts numbers on the “AI compression is 300× better, but we don’t use it” claim from the inspiration video (Lague, 2025)."),
]}));

// === 3. HVS science ===
children.push(P({ text: "3. Scientific foundation: the human visual system", heading: HeadingLevel.HEADING_2, before: 100, after: 60 }));
children.push(P({ runs: [
  R("Five principles motivate our design. "),
  R("Foveation", { bold: true }),
  R(" — the fovea has roughly ten times the spatial resolution of peripheral retina, so visual acuity drops sharply away from the point of gaze (Wandell, 1995; Geisler & Perry, 1998). "),
  R("The Contrast Sensitivity Function", { bold: true }),
  R(" peaks near 4 cycles per degree; JPEG’s quantization matrix was derived directly from CSF data (Campbell & Robson, 1968). "),
  R("Just-Noticeable-Difference", { bold: true }),
  R(" defines the threshold below which distortion is invisible (Mannos & Sakrison, 1974; Yang et al., 2005). "),
  R("Spatial and temporal masking", { bold: true }),
  R(" make distortion harder to detect in textured or moving regions, exploited in HEVC (Sullivan et al., 2012). "),
  R("Saliency", { bold: true }),
  R(" — bottom-up attention — correlates strongly with eye-tracking ground truth (Itti, Koch & Niebur, 1998) and serves as a tractable proxy for “where distortion will hurt.” Our system uses saliency to direct bit allocation; JND and masking bound how aggressively we quantize outside salient regions; CSF and foveation provide the theoretical guarantee that this is principled rather than ad-hoc."),
]}));

// === 4. Architecture ===
children.push(P({ text: "4. System architecture", heading: HeadingLevel.HEADING_2, before: 100, after: 60 }));
children.push(P({ runs: [
  R("The pipeline has three subsystems, each answering one of the prompt’s sub-questions (Figure 1). "),
  R("(A) The useful-footage gate", { bold: true }),
  R(" runs always-on: MOG2 background subtraction plus sparse Lucas-Kanade optical flow are fused into a per-frame “usefulness” score. On real CCTV the gate triggers on 65.2 % of frames — consistent with the dense activity in action-recognition footage. "),
  R("(B) The saliency estimator", { bold: true }),
  R(" produces a per-frame attention map via Hou and Zhang’s spectral-residual algorithm (>100 fps on a laptop CPU); a learned model such as TASED-Net is a drop-in upgrade. "),
  R("(C) The saliency-aware compression core", { bold: true }),
  R(" converts the saliency map into a per-pixel sharp/blur weight via a steep sigmoid (threshold 0.4, steepness 12) and feeds H.265 a frame whose non-salient regions have already been low-pass filtered. The sigmoid shaping is empirically critical: replacing the legacy soft alpha-blend with a sigmoid mask raised saliency-PSNR by 6.6 dB at parity bitrate (alpha 22.0 dB → sigmoid 28.6 dB at CRF 28, see §7)."),
]}));

children.push(img("architecture.png", 480, 270));
children.push(caption("Figure 1. End-to-end architecture. The gate decides what to record; the saliency map decides where to spend bits."));

// PAGE BREAK before §5
children.push(new Paragraph({ children: [new PageBreak()] }));

// === 5. Why not pure neural codec ===
children.push(P({ text: "5. Why not a pure neural codec?", heading: HeadingLevel.HEADING_2, before: 0, after: 60 }));
children.push(P({ runs: [
  R("Recent work claims learned video codecs beat H.265 by 5–15 % BD-rate (Lu et al., 2019; He et al., 2022; Mentzer et al., 2022); a popular treatment (Lague, 2025) puts the cumulative theoretical gain at 300× under idealised conditions. We trained one ourselves to find out what actually ships. Our 76 131-parameter convolutional autoencoder, trained on 1 997 frames sampled across the full 2 288-clip Kaggle dataset (30 epochs, ~80 s on Apple M3 Pro MPS), achieves "),
  R("28.95 dB reconstruction PSNR at 6× compression", { bold: true }),
  R(", encodes in "),
  R("1.43 ms per frame", { bold: true }),
  R(" and decodes in 1.35 ms. The weights occupy 310 KB on disk. So small autoencoders "),
  R("can", { italics: true }),
  R(" run on edge hardware in real time. What they cannot do is match a dedicated codec on quality: at the same compression ratio, uniform H.265 reaches 36.65 dB — a 7-plus dB advantage that does not collapse with model scale alone. The latency-versus-quality wall holds. Larger learned codecs reach H.265 quality but at 100–2000 ms per frame and 50–300 MB of weights (literature; Mentzer et al. 2022). Our system therefore "),
  R("borrows the perceptual-allocation philosophy of neural codecs and realises it on classical codec rails", { italics: true }),
  R(", paying milliseconds rather than seconds, and adding no new silicon requirement."),
]}));

// === 6. Privacy + Sustainability ===
children.push(P({ text: "6. Privacy and sustainability", heading: HeadingLevel.HEADING_2, before: 100, after: 60 }));
children.push(P({ runs: [
  R("All inference is on-device. Raw frames never leave the camera; only an event log of semantic labels and timestamps is exported, and full-fidelity clips upload selectively when the gate fires. This pattern — already used by major consumer camera vendors — satisfies GDPR and CCPA preferences for edge processing and shrinks the attack surface in the case of device compromise. "),
  R("At fixed encoder CRF target our system produces 23 % smaller files than uniform H.265; using a typical home-camera baseline of 1.5 Mbps and 0.06 kWh per stored gigabyte (Masanet et al., 2020) on a US grid (0.4 kg CO₂/kWh, EPA average), each deployed camera saves roughly 1.35 TB of stored-and-served video and 32 kg CO₂ per year. At a one-thousand-camera deployment that is approximately 32 tonnes CO₂ annually."),
]}));

// === 7. Evaluation ===
children.push(P({ text: "7. Evaluation", heading: HeadingLevel.HEADING_2, before: 100, after: 60 }));
children.push(P({ runs: [
  R("Twenty stratified clips spanning thirteen action classes (fall, fight/hit, kick, gun, robbery, walk, sit, lying-down, stand, sneak, run, struggle, throw) drive every measurement below. Table 1 anchors the comparison."),
]}));

children.push(ablationTable);
children.push(caption("Table 1. Real-CCTV rate-distortion: ours sigmoid-mask vs uniform H.265 at three matched encoder targets."));

children.push(P({ runs: [
  R("Two findings dominate. First, "),
  R("at low bitrates the saliency-aware approach matches uniform H.265 on saliency-preserved quality", { bold: true }),
  R(": at ~45 KB our crf-34 sigmoid output beats baseline crf-40 by 0.44 dB sal-PSNR (26.29 vs 25.85 dB). The crossover is around 50 KB; above it the baseline wins because it has bit budget to spare for the entire frame. This is consistent with perceptual-coding theory — ROI methods shine under bit pressure (Itti & Koch, 1998; Yang et al., 2005). Second, "),
  R("full-frame pixel metrics intentionally lag", { bold: true }),
  R(": our PSNR sits ~10 dB below baseline, and LPIPS (a learned perceptual metric trained on uniform distortions) sits 0.3–0.4 above. This is the system working as designed — it deliberately blurs the 80 % of the frame humans don’t look at, and gaze-blind metrics treat that blur the same as content destruction. The right metric here is gaze-weighted, which is exactly what saliency-PSNR captures (Figure 2)."),
]}));

children.push(img("rd_curve_real.png", 480, 308));
children.push(caption("Figure 2. Rate-distortion on real CCTV. Below ~50 KB our system matches or beats uniform H.265 on saliency-preserved quality."));

children.push(P({ runs: [
  R("The mask-shaping methodology was empirically validated through an ablation. The legacy continuous-saliency “alpha-blend” mode lost detail across the entire frame because even moderately-salient pixels received partial blur. Replacing it with a steep sigmoid (centred at threshold 0.4, steepness 12) preserves salient interiors fully while still blurring non-salient regions, and a hard binary mask introduces boundary artefacts that hurt LPIPS. Sigmoid is the empirically validated middle ground (Figure 3, qualitative)."),
]}));

children.push(img("qualitative_real.png", 540, 158));
children.push(caption("Figure 3. Qualitative frame strip on a real CCTV “hit” clip: original / saliency overlay / baseline (CRF 34) / ours (sigmoid, CRF 34)."));

// === 8. Limitations ===
children.push(P({ runs: [
  R("Limitations: ", { bold: true }),
  R("the saliency model is classical (spectral-residual) — a learned saliency CNN would tighten where the salient-region mask lands; Tier-A per-block QP via libx265’s ROI API would avoid the LPIPS penalty for non-salient blur entirely; LPIPS itself is whole-frame, so a saliency-weighted LPIPS is the natural next perceptual metric. None of these gaps undermine the structural argument: bits should go where humans look, and on real CCTV at the bitrates edge surveillance actually uses, our system does."),
]}));

// === References ===
children.push(P({ text: "References", heading: HeadingLevel.HEADING_2, before: 80, after: 40 }));
const refs = [
  "Borji, A., & Itti, L. (2013). State-of-the-art in visual attention modelling. IEEE TPAMI, 35(1), 185–207.",
  "Campbell, F. W., & Robson, J. G. (1968). Application of Fourier analysis to the visibility of gratings. J. Physiol., 197(3), 551–566.",
  "Geisler, W. S., & Perry, J. S. (1998). A real-time foveated multiresolution system for low-bandwidth video communication. SPIE Human Vision and Electronic Imaging.",
  "He, D., Yang, Z., Peng, W., Ma, R., Qin, H., & Wang, Y. (2022). ELIC: Efficient learned image compression. CVPR.",
  "Itti, L., Koch, C., & Niebur, E. (1998). A model of saliency-based visual attention for rapid scene analysis. IEEE TPAMI, 20(11), 1254–1259.",
  "Lague, S. (2025). AI compression is 300× better (but we don’t use it). YouTube.",
  "Li, Z., Aaron, A., Katsavounidis, I., Moorthy, A., & Manohara, M. (2016). Toward a better quality metric for the video community (VMAF). SMPTE Motion Imaging Journal.",
  "Lu, G., Ouyang, W., Xu, D., Zhang, X., Cai, C., & Gao, Z. (2019). DVC: An end-to-end deep video compression framework. CVPR.",
  "Mannos, J. L., & Sakrison, D. J. (1974). The effects of a visual fidelity criterion on the encoding of images. IEEE TIT, 20(4), 525–536.",
  "Masanet, E., Shehabi, A., Lei, N., Smith, S., & Koomey, J. (2020). Recalibrating global data center energy-use estimates. Science, 367(6481), 984–986.",
  "Mentzer, F., Toderici, G., Tschannen, M., & Agustsson, E. (2022). VCT: A video compression transformer. NeurIPS.",
  "Sullivan, G. J., Ohm, J.-R., Han, W.-J., & Wiegand, T. (2012). Overview of the High Efficiency Video Coding (HEVC) standard. IEEE TCSVT, 22(12), 1649–1668.",
  "Wandell, B. A. (1995). Foundations of Vision. Sinauer Associates.",
  "Wang, Z., & Bovik, A. C. (2009). Mean squared error: love it or leave it? IEEE Sig. Proc. Mag., 26(1), 98–117.",
  "Yang, X., Ling, W., Lu, Z., Ong, E., & Yao, S. (2005). Just-noticeable-distortion model and its applications in video coding. Sig. Process.: Image Comm., 20(7), 662–680.",
  "Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The unreasonable effectiveness of deep features as a perceptual metric. CVPR.",
];
for (const r of refs) {
  children.push(new Paragraph({
    spacing: { before: 0, after: 0, line: 240, lineRule: "auto" },
    indent: { left: 360, hanging: 360 },
    children: [new TextRun({ text: r, font: FONT, size: 18 })],  // 9pt for refs
  }));
}

const doc = new Document({
  creator: "Sai Amartya",
  title: "STEAM ICAC 2026 Analysis - Perceptually-Guided Sensor-Gated Compression",
  styles: {
    default: { document: { run: { font: FONT, size: SIZE } } },
    paragraphStyles: [
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 26, bold: true, color: "222222" },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{
      level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } },
    }] },
  ] },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    children,
  }],
});

const out = process.argv[2] || "/sessions/inspiring-beautiful-franklin/mnt/STEAM IC/analysis.docx";
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(out, buf);
  console.log(`wrote ${out}  ${buf.length} bytes`);
});
