# Research brief: useful-footage detection on the edge

> **Note:** Background research for the gate subsystem. Numbers are typical figures from secondary sources — verify on your own hardware before claiming specific performance in the analysis.

## Classical motion detection (always-on, cheap)

| Technique | OpenCV API | Cost (480p, CPU) | Strengths |
|---|---|---|---|
| **MOG2** background subtraction | `cv2.createBackgroundSubtractorMOG2()` | <10 ms / frame | Adapts to gradual lighting. Tunable history / learning rate. |
| **KNN** background model | `cv2.createBackgroundSubtractorKNN()` | similar | Memory-efficient on embedded; faster than MOG2. |
| **Sparse Lucas-Kanade optical flow** | `cv2.goodFeaturesToTrack` + `cv2.calcOpticalFlowPyrLK` | 5–10 ms | Distinguishes real motion from lighting flicker. |
| **PIR sensor** (hardware) | n/a | ~0 ms | Triggers on warm-body thermal change; near-zero false positives if calibrated. |

In a home-security context the failure modes are:
- waving curtains and changing shadows (mitigated by MOG2 learning rate);
- flickering lights (sparse optical flow median filter helps);
- pets (size + person-detector confidence filter).

## Lightweight object detection

| Model | Size | CPU speed (640p) | mAP@50 | Notes |
|---|---|---|---|---|
| **YOLOv8n** | ~3.2 MB | 40–60 ms | ~67% | Ultralytics; INT8 quantization brings it to ~15 fps on a Pi 4. |
| **YOLOv5s (INT8)** | ~15 MB | 30–50 ms | ~65% | Older but proven. |
| **MobileNet-SSD v2** | ~27 MB | 50–100 ms | ~63% | Stable but heavier. |
| **EfficientDet-Lite0** | ~11 MB | 30–50 ms | ~61% | Good battery profile. |

For "person / package / vehicle" we want YOLOv8n; trigger inference only on motion or PIR rather than every frame to save compute.

## Sound-event detection

| Model | Size | CPU latency | What it catches |
|---|---|---|---|
| **YAMNet** (Google, AudioSet) | ~3.7 MB TFLite | 50–100 ms / 1s window | 521 classes incl. glass break, dog bark, doorbell, alarm, speech. |
| **PANNs** (Cnn6/Cnn10) | 5–30 MB | similar | Higher accuracy, heavier. |
| **Edge Impulse custom** | <1 MB | 50–200 ms | Per-home transfer learning. |
| **Silero VAD** | ~2 MB | ~10 ms | Speech vs non-speech only. |

Practical recipe: YAMNet for broad event detection; threshold confidence on `glass_breaking > 0.5`, `bark > 0.6`, etc.; combine with audio energy threshold (e.g., > −40 dB) to ignore ambient.

## Sensor fusion

Baseline (what `src/gate.py` ships):
```
usefulness = 0.35*motion + 0.25*min(flow/10, 1) + 0.40*person_conf
```
With audio added:
```
usefulness = 0.25*motion + 0.20*min(flow/10, 1) + 0.30*person_conf + 0.25*audio_event
```

Stretch goal: a 5-feature logistic regression trained on hand-labelled clips. Lightweight, learnable, ~2 hours' work to label and fit.

## Privacy-conscious design patterns

- **On-device inference.** No raw frames leave the device. Industry baseline (Wyze, Eufy, Amcrest, Reolink).
- **Selective upload.** Only "useful" clips are ever transmitted; silent hours stay local.
- **Face blurring on decode.** Apply during playback or compression. ~5–10 ms / frame with OpenCV's face cascade.
- **Semantic event log only.** `{"t": 1716000000, "event": "person", "conf": 0.82}` — no pixel data persists in the log.
- **Federated-ready.** Tiny local detectors are compatible with federated fine-tuning; we cite this as future work.

GDPR / CCPA explicitly favour edge processing. The privacy story in the analysis should lean on this — "on-device by design" is a stronger claim than "encrypted in transit."

## Repos to know
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- YAMNet: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- Silero VAD: https://github.com/snakers4/silero-vad
- EfficientDet-Lite: https://github.com/google/automl/tree/master/efficientdet
