#!/usr/bin/env python3
"""
train_autoencoder.py — Train TinyAutoencoder on real surveillance frames.

Usage:
  python scripts/train_autoencoder.py \
    --data data/real \
    --out results/neural_codec \
    --epochs 30 \
    --lr 1e-3 \
    --batch-size 16 \
    --frames 1000 \
    --device auto
"""

import sys
import os
import json
import argparse
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.neural_codec import NeuralCodec

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from skimage.metrics import peak_signal_noise_ratio
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_device(device_arg: str) -> tuple[str, str]:
    """Resolve device string to actual device, return (device_str, resolved_device_name)."""
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
            name = "Apple Metal Performance Shaders (MPS)"
        elif torch.cuda.is_available():
            device = "cuda"
            name = "CUDA (GPU)"
        else:
            device = "cpu"
            name = "CPU"
    else:
        device = device_arg
        name = device.upper()
    return device, name


def extract_frames_stratified(
    data_dir: Path, max_frames: int = 1000, target_size: tuple[int, int] = (256, 256)
) -> tuple[np.ndarray, int]:
    """
    Extract frames from all .mp4 files in data_dir, stratified across clips.

    Returns: (frames_NHWC_uint8, n_frames_extracted)
    """
    # Find all .mp4 files
    mp4_files = sorted(data_dir.glob("**/*.mp4"))

    if not mp4_files:
        print(f"ERROR: No .mp4 files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(mp4_files)} mp4 file(s).")

    # Estimate frames per clip for stratification
    frames_per_clip = max(1, max_frames // len(mp4_files))

    all_frames = []
    total_frames_available = 0

    iterator = enumerate(mp4_files)
    if HAS_TQDM:
        iterator = tqdm(iterator, total=len(mp4_files), desc="Extracting frames", leave=False)

    for clip_idx, mp4_path in iterator:
        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            print(f"WARNING: Could not open {mp4_path}")
            continue

        total_frames_in_clip = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_available += total_frames_in_clip

        if total_frames_in_clip == 0:
            cap.release()
            continue

        # Evenly sample frames_per_clip from this clip
        indices = np.linspace(0, total_frames_in_clip - 1, frames_per_clip, dtype=int)

        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            # Resize to target_size
            frame_resized = cv2.resize(frame, target_size)
            all_frames.append(frame_resized)

        cap.release()

        if len(all_frames) >= max_frames:
            break

    # Trim to exactly max_frames if we have more
    all_frames = all_frames[:max_frames]

    if not all_frames:
        print("ERROR: No frames extracted")
        sys.exit(1)

    frames_array = np.stack(all_frames, axis=0)

    print(f"Extracted {len(all_frames)} frames (from ~{total_frames_available} total available)")
    print(f"Frame shape: {frames_array.shape}, dtype: {frames_array.dtype}")

    return frames_array, len(all_frames)


def train_codec(
    frames: np.ndarray,
    device: str,
    epochs: int,
    lr: float,
    batch_size: int,
    output_dir: Path,
) -> tuple[NeuralCodec, dict, float]:
    """
    Train codec and return (codec, training_log, final_loss).
    """
    print(f"\nInitializing NeuralCodec on {device}...")

    try:
        codec = NeuralCodec(device=device)
        # Verify model is on device
        _ = next(codec.model.parameters()).device
        print(f"  Model loaded on {codec.device}")
    except Exception as e:
        print(f"ERROR initializing codec on {device}: {e}")
        print("Attempting fallback to CPU...")
        codec = NeuralCodec(device="cpu")

    print(f"\nTraining on {frames.shape[0]} frames for {epochs} epochs...")
    print(f"  Learning rate: {lr}, Batch size: {batch_size}")

    torch.manual_seed(42)
    np.random.seed(42)

    loss_history = codec.train_on_frames(
        frames, epochs=epochs, lr=lr, batch_size=batch_size, log_every=20
    )

    # Extract final loss
    final_loss = list(loss_history.values())[-1] if loss_history else None

    print(f"\nTraining complete. Final loss: {final_loss:.6f}")

    # Save weights
    weights_path = output_dir / "weights.pt"
    codec.save(str(weights_path))
    print(f"Weights saved to {weights_path}")

    # Save training log
    training_log = {
        "per_epoch_loss": [float(loss_history[i]) for i in range(epochs)],
        "final_loss": float(final_loss) if final_loss else None,
        "device": str(codec.device),
        "n_frames": frames.shape[0],
        "epochs": epochs,
    }

    return codec, training_log, final_loss


def benchmark_codec(codec: NeuralCodec, test_frames: np.ndarray, n_iters: int = 20) -> dict:
    """
    Benchmark codec on test frames (last 3), return aggregated metrics.
    """
    print(f"\nBenchmarking on {len(test_frames)} held-out frames ({n_iters} iterations each)...")

    all_encode_ms = []
    all_decode_ms = []
    all_bytes = []
    all_psnrs = []

    iterator = enumerate(test_frames)
    if HAS_TQDM:
        iterator = tqdm(iterator, total=len(test_frames), desc="Benchmark", leave=False)

    for frame_idx, test_frame in iterator:
        # Benchmark
        bench = codec.benchmark(test_frame, n_iters=n_iters)
        all_encode_ms.append(bench["encode_ms"])
        all_decode_ms.append(bench["decode_ms"])
        all_bytes.append(bench["bytes_per_frame"])

        # Encode/decode and compute PSNR
        if HAS_SKIMAGE:
            latent_bytes = codec.encode(test_frame)
            reconstructed = codec.decode(latent_bytes, test_frame.shape[:2])
            psnr = peak_signal_noise_ratio(test_frame, reconstructed, data_range=255)
            all_psnrs.append(psnr)

    # Compute medians and means
    encode_ms_median = float(np.median(all_encode_ms))
    decode_ms_median = float(np.median(all_decode_ms))
    bytes_per_frame = int(np.mean(all_bytes))

    # Compression ratio: raw BGR 256x256 vs latent bytes
    raw_bytes_256x256 = 256 * 256 * 3  # 196608
    compression_ratio = float(raw_bytes_256x256 / bytes_per_frame)

    # Reconstruction PSNR
    psnr_mean = float(np.mean(all_psnrs)) if all_psnrs else None

    result = {
        "encode_ms_median": encode_ms_median,
        "decode_ms_median": decode_ms_median,
        "bytes_per_frame": bytes_per_frame,
        "compression_ratio_vs_raw_BGR_256x256": compression_ratio,
        "reconstruction_psnr_mean": psnr_mean,
    }

    print(f"  Encode (median):  {encode_ms_median:.2f} ms")
    print(f"  Decode (median):  {decode_ms_median:.2f} ms")
    print(f"  Bytes/frame:      {bytes_per_frame}")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    if psnr_mean:
        print(f"  PSNR (mean):      {psnr_mean:.2f} dB")

    return result


def get_model_stats(codec: NeuralCodec) -> dict:
    """Compute parameter count and size."""
    param_count = sum(p.numel() for p in codec.model.parameters())
    return {"param_count": int(param_count)}


def main():
    parser = argparse.ArgumentParser(
        description="Train TinyAutoencoder on surveillance frames."
    )
    parser.add_argument("--data", type=str, default="data/real",
                       help="Directory of .mp4 clips")
    parser.add_argument("--out", type=str, default="results/neural_codec",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--frames", type=int, default=1000,
                       help="Max frames to extract")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: auto, cpu, cuda, mps")

    args = parser.parse_args()

    # Setup
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_str, device_name = get_device(args.device)

    print("=" * 70)
    print("Neural Codec Training")
    print("=" * 70)
    print(f"Device:       {device_name} ({device_str})")
    print(f"Data dir:     {data_dir}")
    print(f"Output dir:   {out_dir}")
    print(f"Max frames:   {args.frames}")
    print(f"Epochs:       {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size:   {args.batch_size}")

    # Extract frames
    frames, n_frames = extract_frames_stratified(data_dir, max_frames=args.frames)

    # Reserve last 3 for testing
    if n_frames > 3:
        train_frames = frames[:-3]
        test_frames = frames[-3:]
    else:
        train_frames = frames
        test_frames = frames[-1:]

    print(f"Train: {train_frames.shape[0]}, Test: {test_frames.shape[0]}")

    # Train
    codec, training_log, final_loss = train_codec(
        train_frames,
        device=device_str,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        output_dir=out_dir,
    )

    # Get model stats
    weights_path = out_dir / "weights.pt"
    weights_size = os.path.getsize(weights_path)
    model_stats = get_model_stats(codec)

    # Benchmark
    bench_dict = benchmark_codec(codec, test_frames, n_iters=20)

    # Assemble final benchmark JSON
    benchmark_json = {
        "device": str(device_str),
        "encode_ms_median": bench_dict["encode_ms_median"],
        "decode_ms_median": bench_dict["decode_ms_median"],
        "bytes_per_frame": bench_dict["bytes_per_frame"],
        "compression_ratio_vs_raw_BGR_256x256": bench_dict["compression_ratio_vs_raw_BGR_256x256"],
        "reconstruction_psnr_mean": bench_dict["reconstruction_psnr_mean"],
        "weights_size_bytes": weights_size,
        "param_count": model_stats["param_count"],
        "training_final_loss": final_loss,
        "n_training_frames": train_frames.shape[0],
        "epochs": args.epochs,
    }

    # Save benchmark JSON
    bench_path = out_dir / "benchmark.json"
    with open(bench_path, "w") as f:
        json.dump(benchmark_json, f, indent=2)
    print(f"\nBenchmark JSON saved to {bench_path}")

    # Save training log
    train_log_path = out_dir / "training.json"
    with open(train_log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Device:                               {benchmark_json['device']}")
    print(f"Encode time (median):                 {benchmark_json['encode_ms_median']:.2f} ms")
    print(f"Decode time (median):                 {benchmark_json['decode_ms_median']:.2f} ms")
    print(f"Bytes per frame:                      {benchmark_json['bytes_per_frame']}")
    print(f"Compression ratio (vs raw BGR):       {benchmark_json['compression_ratio_vs_raw_BGR_256x256']:.2f}x")
    if benchmark_json['reconstruction_psnr_mean']:
        print(f"Reconstruction PSNR (mean):           {benchmark_json['reconstruction_psnr_mean']:.2f} dB")
    print(f"Weights size:                         {benchmark_json['weights_size_bytes']} bytes")
    print(f"Parameter count:                      {benchmark_json['param_count']:,}")
    print(f"Training final loss:                  {benchmark_json['training_final_loss']:.6f}")
    print(f"Training frames:                      {benchmark_json['n_training_frames']}")
    print(f"Epochs:                               {benchmark_json['epochs']}")
    print("=" * 70)
    print(f"\nAll results written to: {out_dir}")
    print(f"  - weights.pt")
    print(f"  - training.json")
    print(f"  - benchmark.json")


if __name__ == "__main__":
    main()
