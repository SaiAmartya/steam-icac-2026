"""
Build a multi-page PDF catalog of qualitative comparisons:
    original / saliency overlay / baseline H.265 / ours sigmoid

Sweeps every clip × CRF × multiple frames so we can pick the cleanest
example for the showcase deck.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/real"
BASE_DIR = ROOT / "results/ablation_real/baselines"
OURS_DIR = ROOT / "results/ablation_sigmoid"
OUT = ROOT / "results/figures/qualitative_catalog.pdf"

CRFS = [22, 34, 40]                # CRF 28 has no matching ours_sigmoid run
FRAME_FRACS = [0.20, 0.45, 0.70]   # sample early / middle / late action
BITRATE_B = 21                     # matches existing "ours_sigmoid_b21_*" runs


def load_manifest() -> dict[str, str]:
    items = json.loads((DATA / "manifest.json").read_text())
    return {it["clip_id"]: it["action_class"] for it in items}


def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def get_frame(path: Path, idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def saliency_overlay(rgb_frame: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    sal_obj = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok, sal = sal_obj.computeSaliency(bgr)
    if not ok:
        return rgb_frame
    sal = (sal * 255).clip(0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(sal, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(bgr, 0.55, heat, 0.45, 0.0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def kb(path: Path) -> float:
    return path.stat().st_size / 1024 if path.exists() else 0.0


def render_page(
    pdf: PdfPages,
    clip_id: str,
    action_class: str,
    frame_idx: int,
    crf: int,
    orig_path: Path,
    base_path: Path,
    ours_path: Path,
) -> bool:
    orig = get_frame(orig_path, frame_idx)
    base = get_frame(base_path, frame_idx)
    ours = get_frame(ours_path, frame_idx)
    if orig is None or base is None or ours is None:
        return False

    sal = saliency_overlay(orig)

    base_kb = kb(base_path)
    ours_kb = kb(ours_path)
    saved_pct = (1 - ours_kb / base_kb) * 100 if base_kb > 0 else 0.0

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6))
    panels = [
        (orig, "original"),
        (sal, "saliency overlay"),
        (base, f"baseline H.265 (CRF {crf})\n{base_kb:.0f} KB"),
        (ours, f"ours sigmoid (CRF {crf})\n{ours_kb:.0f} KB  ({saved_pct:+.0f}%)"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f'{clip_id} — action class "{action_class}", frame {frame_idx} (of {frame_count(orig_path)})',
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def main() -> None:
    manifest = load_manifest()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    rendered = 0
    skipped = 0
    with PdfPages(OUT) as pdf:
        # cover page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(
            0.5, 0.7, "Qualitative comparison catalog",
            ha="center", va="center", fontsize=24, fontweight="bold",
        )
        ax.text(
            0.5, 0.55,
            "STEAM IC — saliency-aware compression vs H.265 baseline",
            ha="center", va="center", fontsize=14,
        )
        ax.text(
            0.5, 0.40,
            f"{len(manifest)} clips × {len(CRFS)} CRFs × {len(FRAME_FRACS)} frames\n"
            "Panels: original | saliency overlay | baseline H.265 | ours sigmoid\n"
            "Size in KB and % saved vs baseline shown under each ours panel.",
            ha="center", va="center", fontsize=11,
        )
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        for clip_id, action_class in sorted(manifest.items()):
            orig_path = DATA / f"{clip_id}.mp4"
            if not orig_path.exists():
                continue
            n = frame_count(orig_path)
            frames = [max(0, min(n - 1, int(n * f))) for f in FRAME_FRACS]
            for frame_idx in frames:
                for crf in CRFS:
                    base_path = BASE_DIR / f"baseline_crf{crf}_{clip_id}.mp4"
                    ours_path = OURS_DIR / f"ours_sigmoid_b{BITRATE_B}_crf{crf}_{clip_id}.mp4"
                    if not base_path.exists() or not ours_path.exists():
                        skipped += 1
                        continue
                    ok = render_page(
                        pdf, clip_id, action_class, frame_idx, crf,
                        orig_path, base_path, ours_path,
                    )
                    if ok:
                        rendered += 1
                    else:
                        skipped += 1
                    if rendered % 10 == 0 and rendered > 0:
                        print(f"  rendered {rendered} pages...")
    print(f"done: {rendered} pages rendered, {skipped} skipped")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
