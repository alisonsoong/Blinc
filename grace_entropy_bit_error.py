#!/usr/bin/env python3
"""
grace_entropy_packet_error.py

- Uses GRACE with real torchac entropy coding (via GraceAdapter + EntropyCodec)
- Encodes mv / z / res with torchac
- Uses the CDFs to estimate per-symbol bit contributions
- Builds a cumulative bit-range mapping over mv+z+res for each frame
- Samples bit errors (Poisson number, uniformly over the frame bitstream)
- Maps bit positions -> symbol indices -> tensor entries
- Corrupts those entries by forcing their quantized symbol to decode to 0
- Decodes corrupted frames and measures SSIM vs the original

NOTE: We never decode the corrupted entropy stream; we simulate the effect
      of bit errors at the symbol level using an approximate bit allocation.
"""

import os
import sys
import math
import copy
import argparse

import numpy as np
from tqdm import tqdm

import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image

from pytorch_msssim import ssim as ssim_fn

# Make sure we can import experimental interfaces if they live in ./experimental
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(THIS_DIR, "experimental")
if EXP_DIR not in sys.path:
    sys.path.append(EXP_DIR)

from autoencoder_interface import GraceAdapter, AECode  # type: ignore
from streaming_interface import EntropyCodec           # type: ignore


CHUNK_SIZE_BITS = 256

# -------------------------
# Basic metrics
# -------------------------

def PSNR(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    x, y: tensors in [0,1], shape (C,H,W) on same device
    """
    y = y.to(x.device)
    mse = torch.mean((x - y) ** 2)
    if mse.item() == 0:
        return 100.0
    return float(10.0 * torch.log10(1.0 / mse))


def SSIM(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    x, y: tensors in [0,1], shape (C,H,W) on GPU
    uses pytorch_msssim.ssim
    """
    return float(
        ssim_fn(
            x.float().unsqueeze(0),
            y.float().unsqueeze(0),
            data_range=1.0,
            size_average=False,
        ).cpu().detach()
    )


# -------------------------
# Video loading
# -------------------------

def read_video_into_frames(video_path: str, nframes: int = 1000):
    """
    Extract frames with ffmpeg, pad to multiples of 128 for GRACE.
    Returns: list of PIL.Image
    """
    import subprocess

    def create_temp_path():
        path = f"/tmp/yihua_frames-{np.random.randint(0, 1000)}/"
        while os.path.isdir(path):
            path = f"/tmp/yihua_frames-{np.random.randint(0, 1000)}/"
        os.makedirs(path, exist_ok=True)
        return path

    def remove_temp_path(tmp_path):
        try:
            subprocess.run(["rm", "-rf", tmp_path], check=False)
        except Exception:
            pass

    frame_path = create_temp_path()

    cmd = [
        "ffmpeg",
        "-i", video_path,
        os.path.join(frame_path, "%03d.png"),
        "2>/dev/null",
        "1>/dev/null",
    ]
    # join into shell string to keep redirections
    os.system(" ".join(cmd))

    image_names = sorted(os.listdir(frame_path))
    frames = []
    for img_name in image_names[:nframes]:
        frame = Image.open(os.path.join(frame_path, img_name))

        # pad to nearest 128 for GRACE
        padsz = 128
        w, h = frame.size
        pad_w = int(math.ceil(w / padsz) * padsz)
        pad_h = int(math.ceil(h / padsz) * padsz)
        frames.append(frame.resize((pad_w, pad_h)))

    print(f"[INFO] frame path: {frame_path}")
    print(f"[INFO] Got {len(image_names)} image files, using {len(frames)} frames")
    if len(frames) > 0:
        print(f"[INFO] Resized frames to {frames[0].size}")
    remove_temp_path(frame_path)
    return frames


# -------------------------
# Bit estimation helpers
# -------------------------

def estimate_bits_per_symbol_from_cdf(
    tensor_q: torch.Tensor,
    cdf_int: torch.Tensor,
) -> torch.Tensor:
    """
    Approximate per-symbol bit cost using the integer CDF (like EntropyCodec.calculate_size).

    tensor_q: N,C,H,W (values in 0..L-2)
    cdf_int: N,C,H,W,L (int16)
    Returns: bits_per_symbol, shape N,C,H,W (float32 on CPU)
    """
    # Move to CPU for indexing
    sym = tensor_q.long().cpu()
    cdf = cdf_int.cpu()

    N, C, H, W, Lp = cdf.shape

    # Build index grids N,C,H,W
    idx_n, idx_c, idx_h, idx_w = torch.meshgrid(
        torch.arange(N),
        torch.arange(C),
        torch.arange(H),
        torch.arange(W),
        indexing="ij",
    )

    # sym is N,C,H,W in [0, Lp-2] (torchac expects that)
    sym_p1 = sym + 1

    final_indices = (idx_n, idx_c, idx_h, idx_w, sym)
    final_indices_p1 = (idx_n, idx_c, idx_h, idx_w, sym_p1)

    # CDF difference → prob
    probs_raw = cdf[final_indices_p1] - cdf[final_indices]
    # Correct wrap-around from int16
    probs = (probs_raw.float() + (probs_raw < 0).float() * 65536.0) / 65536.0

    eps = 1e-6
    bits = torch.clamp(-torch.log2(probs + eps), 0.0, 50.0)

    return bits  # N,C,H,W float on CPU


def build_frame_bit_mapping(
    code: AECode,
    entropy_codec: EntropyCodec,
):
    """
    For a single frame:
      - entropy-encode mv, z, res with torchac (real streams)
      - estimate per-symbol bits from distributions
      - build flattened mapping (concatenated mv+z+res) from bit positions
        to symbol index.

    Returns:
      info: dict with:
        'code'             : AECode (original, not modified)
        'streams'          : dict name -> (byte_stream, size_bytes)
        'total_bits'       : int
        'symbol_bits'      : np.array [num_symbols]
        'boundaries'       : np.array [num_symbols+1], cumulative bits
        'sizes'            : dict name -> num_symbols_for_that_tensor
        'shapes'           : dict name -> original tensor shape (N,C,H,W)
        'mxranges'         : dict name -> mxrange for that tensor
    """
    names = ["mv", "z", "res"]
    streams = {}
    symbol_bits_list = []
    sizes = {}
    shapes = {}
    mxranges = {}

    total_bits_true = 0

    for name in names:
        ctx = code.get_tensor(name)
        q = ctx.quantized_code  # N,C,H,W on cuda
        dist = ctx.distribution  # N,C,H,W,L int16 on CPU

        # Real entropy encode
        bs, size_bytes = entropy_codec.entropy_encode(q, dist)
        streams[name] = (bs, size_bytes)
        bits_true = size_bytes * 8
        total_bits_true += bits_true

        # Estimate per-symbol bits
        bits_tensor = estimate_bits_per_symbol_from_cdf(q, dist)  # N,C,H,W
        bits_flat = bits_tensor.reshape(-1)

        # Normalize so that sum(bits_est) == bits_true
        est_sum = float(bits_flat.sum().item())
        if est_sum <= 0:
            # Fallback to uniform
            num = bits_flat.numel()
            if num == 0:
                per_sym = torch.zeros(0)
            else:
                per_sym = torch.full((num,), float(bits_true) / float(num))
        else:
            scale = bits_true / est_sum
            per_sym = bits_flat * scale

        symbol_bits_list.append(per_sym)
        sizes[name] = per_sym.numel()
        shapes[name] = q.shape
        mxranges[name] = ctx.mxrange

    if len(symbol_bits_list) > 0:
        symbol_bits = torch.cat(symbol_bits_list, dim=0)
    else:
        symbol_bits = torch.zeros(0)

    symbol_bits_np = symbol_bits.numpy()
    # boundaries[i] is start of symbol i; boundaries[-1] is total
    boundaries = np.zeros(symbol_bits_np.shape[0] + 1, dtype=np.float64)
    boundaries[1:] = np.cumsum(symbol_bits_np)

    info = {
        "code": code,
        "streams": streams,
        "total_bits": int(round(boundaries[-1])),
        "symbol_bits": symbol_bits_np,
        "boundaries": boundaries,
        "sizes": sizes,
        "shapes": shapes,
        "mxranges": mxranges,
    }
    return info


def sample_bit_errors(total_bits: int, bit_error_rate: float, poisson_scale: float = 1.0):
    """
    total_bits: length of frame stream
    bit_error_rate: per-bit BER
    poisson_scale: multiply expected errors (for tuning)

    Returns: list of bit positions (ints in [0, total_bits-1])
    """
    if total_bits <= 0 or bit_error_rate <= 0:
        return []

    lam = total_bits * bit_error_rate * poisson_scale
    k = np.random.poisson(lam)
    if k <= 0:
        return []
    # Allow duplicates; we'll deduplicate later at symbol level anyway.
    bit_positions = np.random.randint(0, max(total_bits, 1), size=k)
    return bit_positions.tolist()


CHUNK_SIZE_BITS = 256  # your block size

def bits_to_symbol_indices(boundaries, bit_positions):
    """
    boundaries: np.array of length (num_symbols+1)
        boundaries[i]   = bit-start of symbol i
        boundaries[i+1] = bit-end   of symbol i
    
    bit_positions: list of bit indices where flips occurred

    Returns:
        corrupted_symbols: set of symbol indices
        corrupted_chunks:  set of chunk indices (avoid duplicates)
    """
    import numpy as np
    import math

    boundaries = np.asarray(boundaries, dtype=float)
    total_bits = float(boundaries[-1])
    num_symbols = len(boundaries) - 1

    if total_bits <= 0 or num_symbols <= 0:
        return set(), set()

    # ---- 1) Identify which chunks are corrupted ----
    num_chunks = int(math.ceil(total_bits / CHUNK_SIZE_BITS))
    corrupted_chunks = np.zeros(num_chunks, dtype=bool)

    for b in bit_positions:
        if 0 <= b < total_bits:
            chunk_id = int(b // CHUNK_SIZE_BITS)
            if 0 <= chunk_id < num_chunks:
                corrupted_chunks[chunk_id] = True

    # Convert to set of chunk IDs
    corrupted_chunk_ids = set(np.where(corrupted_chunks)[0])

    if len(corrupted_chunk_ids) == 0:
        return set(), corrupted_chunk_ids

    # ---- 2) Identify ALL symbols whose bit-range intersects ANY corrupted chunk ----
    corrupted_symbols = set()

    for i in range(num_symbols):
        s_start = boundaries[i]
        s_end   = boundaries[i+1]
        if s_end <= s_start:
            continue

        # compute symbol's chunk range
        first_chunk = int(s_start // CHUNK_SIZE_BITS)
        last_chunk  = int((s_end - 1) // CHUNK_SIZE_BITS)

        first_chunk = max(first_chunk, 0)
        last_chunk  = min(last_chunk, num_chunks - 1)

        # if ANY chunk in this interval is corrupted -> zero entire block
        if np.any(corrupted_chunks[first_chunk:last_chunk + 1]):
            corrupted_symbols.add(i)

    return corrupted_symbols, corrupted_chunk_ids



def apply_symbol_corruption_to_code(
    code: AECode,
    frame_info,
    corrupted_symbols: set,
):
    """
    Mutate AECode.quantized_code for mv / z / res for the given set of symbol indices.

    Strategy:
      - Flatten mv, z, res in that order
      - For any symbol index in corrupted_symbols, set the corresponding
        quantized_code entry to mxrange, so that (qc - mxrange) = 0 in decode.
    """
    sizes = frame_info["sizes"]
    shapes = frame_info["shapes"]
    mxranges = frame_info["mxranges"]

    names = ["mv", "z", "res"]
    # prefix sizes to know which symbol index lives where
    prefix = {}
    running = 0
    for n in names:
        prefix[n] = running
        running += sizes[n]

    # Build per-tensor flattened masks
    masks_flat = {n: np.zeros(sizes[n], dtype=bool) for n in names}

    for sym_idx in corrupted_symbols:
        # Find which tensor this symbol index belongs to
        for n in names:
            start = prefix[n]
            end = start + sizes[n]
            if start <= sym_idx < end:
                local_idx = sym_idx - start
                masks_flat[n][local_idx] = True
                break

    # Apply to AECode tensors
    for n in names:
        if sizes[n] == 0:
            continue
        ctx = code.get_tensor(n)
        q = ctx.quantized_code  # N,C,H,W
        mask_flat = masks_flat[n]
        if not mask_flat.any():
            continue
        mask = torch.from_numpy(mask_flat.reshape(q.numel())).to(q.device)
        mask = mask.view_as(q)

        # Set symbol to mxrange to represent "zero residual/mv" after de- shift
        mxr = ctx.mxrange
        with torch.no_grad():
            q[mask] = mxr


# -------------------------
# Main experiment
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="GRACE entropy packet error simulation (torchac-based)")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--model_path", type=str, required=True, help="GRACE model path, e.g. models/grace/4096_freeze.model")
    parser.add_argument("--nframes", type=int, default=16, help="Number of frames to process")
    parser.add_argument("--bit_error_rate", type=float, default=0.0, help="Per-bit error probability")
    parser.add_argument("--poisson_scale", type=float, default=1.0, help="Scale factor for Poisson mean (for tuning)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 1. Load video frames
    frames_pil = read_video_into_frames(args.video, nframes=args.nframes)
    if len(frames_pil) == 0:
        print("[ERROR] No frames extracted, exiting.")
        return

    # 2. Init GRACE interfaces
    print("[INFO] Initializing GRACE adapter and entropy codec...")
    adapter = GraceAdapter({"path": args.model_path, "device": torch.device(device).type}, use_half=True, scale_factor=1.0)
    entropy_codec = EntropyCodec()

    # 3. Convert originals to tensors
    orig_frames = [to_tensor(f).to(device) for f in frames_pil]

    # Storage for results
    per_frame_stats = []   # list of dicts: {'frame', 'true_bits', 'n_bit_errors', 'ssim', 'psnr'}
    ref_decoded = orig_frames[0].detach()  # perfect I-frame
    dec_frames: List[torch.Tensor] = []

    print("[INFO] Encoding and simulating bit errors with GRACE (real torchac)...")
    for idx, frame_pil in enumerate(tqdm(frames_pil, desc="GRACE frames")):
        frame_t = orig_frames[idx]

        if idx == 0:
            # I-frame (no entropy coding)
            rec = ref_decoded.clone()
            s_sim  = SSIM(frame_t, rec)
            s_psnr = PSNR(frame_t, rec)
            per_frame_stats.append({
                "frame": idx,
                "true_bits": 0,
                "n_bit_errors": 0,
                "zeroed_chunks": 0,
                "zeroed_bits": 0,
                "ssim": s_sim,
                "psnr": s_psnr,
            })
            continue

        # Encode P-frame
        with torch.no_grad():
            code, dec_frame = adapter.encode(frame_t, ref_decoded)
        dec_frames.append(dec_frame)
        # Build true entropy streams + per-symbol mapping
        frame_info = build_frame_bit_mapping(code, entropy_codec)
        total_bits = frame_info["total_bits"]

        # Sample bit errors
        bit_positions = sample_bit_errors(
            total_bits=total_bits,
            bit_error_rate=args.bit_error_rate,
            poisson_scale=args.poisson_scale,
        )

        # Map bit positions → symbol indices
        corrupted_symbols, corrupted_chunks = bits_to_symbol_indices(
            frame_info["boundaries"], bit_positions
        )

        zeroed_chunks = len(corrupted_chunks)
        zeroed_bits   = zeroed_chunks * CHUNK_SIZE_BITS

        # Apply corruption to AECode copy
        code_corrupted = copy.deepcopy(frame_info["code"])
        if len(corrupted_symbols) > 0:
            apply_symbol_corruption_to_code(code_corrupted, frame_info, corrupted_symbols)

        # Decode with corrupted latents
        with torch.no_grad():
            rec = adapter.decode(code_corrupted, ref_decoded)

        rec = rec.to(device)
        ref_decoded = rec.detach()

        # Frame metrics
        s_sim  = SSIM(frame_t, rec)
        s_psnr = PSNR(frame_t, rec)

        per_frame_stats.append({
            "frame": idx,
            "true_bits": total_bits,
            "n_bit_errors": len(bit_positions),
            "zeroed_chunks": zeroed_chunks,
            "zeroed_bits": zeroed_bits,
            "ssim": s_sim,
            "psnr": s_psnr,
        })

    # ------------------------------
    # 5. Final summary printing
    # ------------------------------

    print("\n================ FRAME-BY-FRAME RESULTS ================")
    print(f"{'Frame':>5} | {'Bits':>12} | {'Errors':>6} | {'ZeroChunks':>10} | {'ZeroBits':>10} | {'SSIM':>8} | {'PSNR(dB)':>8}")
    print("-" * 90)

    for s in per_frame_stats:
        print(f"{s['frame']:5d} | {s['true_bits']:12d} | {s['n_bit_errors']:6d} | "
            f"{s['zeroed_chunks']:10d} | {s['zeroed_bits']:10d} | "
            f"{s['ssim']:8.4f} | {s['psnr']:8.2f}")


    # Aggregate metrics
    mean_ssim = float(np.mean([x["ssim"] for x in per_frame_stats]))
    mean_psnr = float(np.mean([x["psnr"] for x in per_frame_stats]))
    ssim_db = 10.0 * math.log10(1.0 / max(1e-12, 1.0 - mean_ssim))

    print("\n==================== OVERALL SUMMARY ====================")
    print(f"Mean SSIM: {mean_ssim:.6f}")
    print(f"SSIM (dB): {ssim_db:.3f} dB")
    print(f"Mean PSNR: {mean_psnr:.3f} dB")
    print("========================================================\n")


    # -------------------------
    # Save output video
    # -------------------------
    print("[INFO] Saving output video...")

    out_dir = "recon_frames"
    os.makedirs(out_dir, exist_ok=True)

    # Save frames as PNG images
    import torchvision.utils as vutils
    for i, frame in enumerate(dec_frames):
        # Clamp and convert tensor → PIL-like format
        img = frame.clamp(0,1).cpu()
        vutils.save_image(img, f"{out_dir}/{i:04d}.png")

    # Use FFmpeg to turn PNGs into a video
    output_video = "reconstructed_output.mp4"
    fps = 30  # or whatever your assumed fps is

    cmd = (
        f"ffmpeg -y -framerate {fps} -i {out_dir}/%04d.png "
        f"-pix_fmt yuv420p -crf 18 {output_video}"
    )
    os.system(cmd)

    print(f"[INFO] Output video saved as: {output_video}")

if __name__ == "__main__":
    main()
