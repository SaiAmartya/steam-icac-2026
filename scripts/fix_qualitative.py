"""
Replace results/figures/qualitative_real.png with the proper 4-panel comparison
the analysis was designed around: original / saliency overlay / baseline / ours.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"

CLIP_ID = "clip_04"
TARGET_FRAME = 60

ORIGINAL = ROOT / f"data/real/{CLIP_ID}.mp4"
BASELINE = ROOT / f"results/ablation_real/baselines/baseline_crf34_{CLIP_ID}.mp4"
OURS = ROOT / f"results/ablation_sigmoid/ours_sigmoid_b21_crf34_{CLIP_ID}.mp4"


def get_frame(path: Path, idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


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


def get_size_kb(path: Path) -> float:
    return path.stat().st_size / 1024 if path.exists() else 0.0


def main() -> None:
    # Fail loudly if any of the three required files are missing
    for p, name in [(ORIGINAL, "original"), (BASELINE, "baseline_crf34"), (OURS, "ours_sigmoid")]:
        if not p.exists():
            print(f"MISSING {name}: {p}")
            sys.exit(1)

    orig = get_frame(ORIGINAL, TARGET_FRAME)
    base = get_frame(BASELINE, TARGET_FRAME)
    ours = get_frame(OURS, TARGET_FRAME)
    if orig is None or base is None or ours is None:
        print(f"Failed to read frame {TARGET_FRAME} from one of the videos.")
        sys.exit(1)

    sal = saliency_overlay(orig)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6))
    panels = [
        (orig, "original"),
        (sal, "saliency overlay"),
        (base, f"baseline H.265 (CRF 34)\n{get_size_kb(BASELINE):.0f} KB"),
        (ours, f"ours sigmoid (CRF 34)\n{get_size_kb(OURS):.0f} KB"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Read action class from the manifest if available
    try:
        manifest = json.loads((ROOT / "data/real/manifest.json").read_text())
        cls = next(
            (item["action_class"] for item in manifest if item["clip_id"] == CLIP_ID),
            "?",
        )
    except Exception:
        cls = "?"

    fig.suptitle(
        f"Qualitative comparison — clip_04, action class “{cls}”, frame {TARGET_FRAME}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out = FIG / "qualitative_real.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
