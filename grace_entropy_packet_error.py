import argparse
import math
import os
import sys
import json
import subprocess
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
from pytorch_msssim import ssim as ssim_fn

# Make sure we can import experimental interfaces if they live in ./experimental
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(THIS_DIR, "experimental")
if EXP_DIR not in sys.path:
    sys.path.append(EXP_DIR)

from autoencoder_interface import GraceAdapter, AECode
from streaming_interface import EntropyCodec


# -----------------------------
# Basic metrics
# -----------------------------
def PSNR(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    x, y: (C,H,W), values in [0,1]
    """
    x = x.to(y.device)
    mse = torch.mean((x - y) ** 2)
    if mse.item() == 0:
        return 99.0
    log10 = torch.log(torch.tensor(10.0, device=y.device))
    psnr = 10.0 * torch.log(1.0 / mse) / log10
    return float(psnr)


def SSIM(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    x, y: (C,H,W), values in [0,1]
    """
    x = x.to(y.device).unsqueeze(0).float()
    y = y.to(y.device).unsqueeze(0).float()
    return float(ssim_fn(x, y, data_range=1.0, size_average=True).item())


# -----------------------------
# Video loading
# -----------------------------
def read_video_into_frames(video_path: str, nframes: int = 16) -> List["Image.Image"]:
    """
    Extracts frames with ffmpeg, pads each frame to nearest multiple of 128
    (this is what GRACE expects), returns a list of PIL images.
    """
    from PIL import Image
    import math as _math

    def create_temp_path():
        base = f"/tmp/grace_frames-{np.random.randint(0, 1_000_000)}/"
        while os.path.isdir(base):
            base = f"/tmp/grace_frames-{np.random.randint(0, 1_000_000)}/"
        os.makedirs(base, exist_ok=True)
        return base

    def remove_temp_path(path):
        try:
            subprocess.run(["rm", "-rf", path], check=False)
        except Exception:
            pass

    tmp_dir = create_temp_path()
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        os.path.join(tmp_dir, "%03d.png"),
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    subprocess.run(cmd, check=False)

    img_names = sorted(os.listdir(tmp_dir))
    frames: List[Image.Image] = []
    for name in img_names[:nframes]:
        frame = Image.open(os.path.join(tmp_dir, name)).convert("RGB")
        # pad to nearest 128 for GRACE
        padsz = 128
        w, h = frame.size
        pad_w = int(_math.ceil(w / padsz) * padsz)
        pad_h = int(_math.ceil(h / padsz) * padsz)
        frames.append(frame.resize((pad_w, pad_h)))

    print(f"[INFO] frame path: {tmp_dir}")
    print(f"[INFO] Got {len(img_names)} image files, using {len(frames)} frames")
    if frames:
        print(f"[INFO] Resizing image to {frames[0].size}")

    remove_temp_path(tmp_dir)
    return frames


# -----------------------------
# Torchac-based bit counting
# -----------------------------
def compute_true_bits_from_torchac(code: AECode, entropy_codec: EntropyCodec) -> int:
    """
    Uses torchac (via EntropyCodec.entropy_encode) to compute the true number of bits
    for mv, z, and res for this frame.
    """
    total_bytes = 0
    for name in ["mv", "z", "res"]:
        ctx = code.get_tensor(name)
        _, nbytes = entropy_codec.entropy_encode(
            ctx.quantized_code, ctx.distribution
        )
        total_bytes += nbytes
    return total_bytes * 8


# -----------------------------
# Virtual packetization
# -----------------------------
def build_packets(total_bits: int, packet_bytes: int = 1440):
    """
    Split a frame's bitstream into packets of up to `packet_bytes` bytes.
    Returns a list of (bit_start, bit_end) for each packet, [start, end).
    The *last* packet may be shorter than packet_bytes.
    """
    packet_bits = packet_bytes * 8
    packets = []
    bit_start = 0
    while bit_start < total_bits:
        bit_end = min(bit_start + packet_bits, total_bits)
        packets.append((bit_start, bit_end))  # [bit_start, bit_end)
        bit_start = bit_end
    return packets


# -----------------------------
# Virtual chunk mapping (latent blocks)
# -----------------------------
def build_virtual_mapping(
    code: AECode,
    total_bits: int,
    chunk_bits: int,
) -> Dict:
    """
    Build a virtual mapping from "chunks" to ranges in the concatenated
    latent tensors (mv, z, res).

    Now closely matches the old packet-level script:
      - Compute bits_per_symbol = total_bits / total_symbols
      - Chunks live in bit-space: n_chunks = ceil(total_bits / chunk_bits)
      - To map a chunk to symbols, we convert its bit-range to symbol-range
        using bits_per_symbol.
    """
    tensor_ranges: Dict[str, Tuple[int, int]] = {}
    offset = 0
    for name in ["mv", "z", "res"]:
        ctx = code.get_tensor(name)
        flat_len = int(ctx.quantized_code.numel())
        tensor_ranges[name] = (offset, offset + flat_len)
        offset += flat_len
    total_symbols = offset

    if total_symbols <= 0 or total_bits <= 0 or chunk_bits <= 0:
        return {
            "code": code,
            "total_bits": int(total_bits),
            "chunk_bits": int(chunk_bits),
            "n_chunks": 0,
            "total_symbols": int(total_symbols),
            "bits_per_symbol": 0.0,
            "tensor_ranges": tensor_ranges,
        }

    bits_per_symbol = float(total_bits) / float(total_symbols)
    n_chunks = int(math.ceil(total_bits / float(chunk_bits)))

    return {
        "code": code,
        "total_bits": int(total_bits),
        "chunk_bits": int(chunk_bits),
        "n_chunks": int(n_chunks),
        "total_symbols": int(total_symbols),
        "bits_per_symbol": float(bits_per_symbol),
        "tensor_ranges": tensor_ranges,  # name -> (start_symbol, end_symbol)
    }


def apply_chunk_corruption_to_code(frame_info: Dict, bad_chunks: List[int]) -> float:
    """
    Given a list of corrupted chunk indices, zero out the corresponding ranges
    in the latent tensors (mv, z, res) in frame_info["code"].

    This uses the same logic as the old packet-level script:
      - bits_per_symbol = total_bits / total_symbols
      - each chunk c has bit range [c*chunk_bits, min((c+1)*chunk_bits,total_bits)-1]
      - bit range -> symbol indices [first_sym, last_sym] via bits_per_symbol
      - those symbol indices are then set to mxrange (encoded zero).

    Returns:
        approx_zero_bits: approximate number of bits covered by zeroed symbols
                          (n_zeroed_symbols * bits_per_symbol).
    """
    code: AECode = frame_info["code"]
    total_bits = frame_info["total_bits"]
    chunk_bits = frame_info["chunk_bits"]
    bits_per_symbol = frame_info["bits_per_symbol"]
    total_symbols = frame_info["total_symbols"]
    tensor_ranges = frame_info["tensor_ranges"]

    if (
        total_symbols <= 0
        or total_bits <= 0
        or chunk_bits <= 0
        or bits_per_symbol <= 0.0
    ):
        return 0.0

    unique_chunks = sorted(set([c for c in bad_chunks if 0 <= c < frame_info["n_chunks"]]))
    if not unique_chunks:
        return 0.0

    # Global mask over the concatenated symbol axis
    drop_mask = np.zeros(total_symbols, dtype=bool)

    for c in unique_chunks:
        start_bit = c * chunk_bits
        end_bit = min(total_bits, (c + 1) * chunk_bits) - 1
        if start_bit > end_bit:
            continue

        first_sym = int(math.floor(start_bit / bits_per_symbol))
        last_sym = int(math.ceil((end_bit + 1) / bits_per_symbol) - 1)

        first_sym = max(0, min(first_sym, total_symbols - 1))
        last_sym = max(0, min(last_sym, total_symbols - 1))
        if last_sym < first_sym:
            continue

        drop_mask[first_sym:last_sym + 1] = True

    n_zeroed = int(drop_mask.sum())
    if n_zeroed == 0:
        return 0.0

    approx_zero_bits = n_zeroed * bits_per_symbol

    # Apply mask to each tensor (mv, z, res)
    for name in ["mv", "z", "res"]:
        ctx = code.get_tensor(name)
        q = ctx.quantized_code
        start_global, end_global = tensor_ranges[name]
        if end_global <= start_global:
            continue

        local_mask_np = drop_mask[start_global:end_global]
        if not local_mask_np.any():
            continue

        # reshape to tensor shape
        local_mask = torch.from_numpy(local_mask_np.reshape(-1)).to(q.device)
        local_mask = local_mask.view_as(q)

        mxr = ctx.mxrange
        with torch.no_grad():
            q[local_mask] = mxr

    return float(approx_zero_bits)


def bits_to_chunks(bit_positions: np.ndarray, chunk_bits: int, n_chunks: int) -> List[int]:
    """
    Map raw bit positions [0, total_bits) to chunk indices, where each chunk
    spans chunk_bits bits. This matches how we define chunks in bit-space.
    """
    if len(bit_positions) == 0 or chunk_bits <= 0 or n_chunks <= 0:
        return []

    chunk_indices = (bit_positions // chunk_bits).astype(np.int64)
    unique = sorted(
        set(
            int(c)
            for c in chunk_indices
            if 0 <= int(c) < n_chunks
        )
    )
    return unique


# -----------------------------
# Bit error sampling
# -----------------------------
def sample_bit_errors(total_bits: int, bit_error_rate: float, poisson_scale: float):
    """
    Sample random bit-error positions in [0, total_bits) using a Poisson model.

    Expected #errors ~ bit_error_rate * total_bits * poisson_scale
    """
    if total_bits <= 0 or bit_error_rate <= 0.0 or poisson_scale <= 0.0:
        return np.array([], dtype=np.int64)

    lam = bit_error_rate * float(total_bits) * float(poisson_scale)
    if lam <= 0:
        return np.array([], dtype=np.int64)

    n_err = np.random.poisson(lam)
    if n_err <= 0:
        return np.array([], dtype=np.int64)

    return np.random.randint(0, total_bits, size=n_err, dtype=np.int64)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GRACE entropy packet-error simulation with true torchac bits"
    )
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="GRACE model path, e.g. models/grace/4096_freeze.model",
    )
    parser.add_argument("--nframes", type=int, default=16, help="Number of frames")
    parser.add_argument(
        "--bit_error_rate",
        type=float,
        default=0.0,
        help="Per-bit error probability (for Poisson mean)",
    )
    parser.add_argument(
        "--poisson_scale",
        type=float,
        default=1.0,
        help="Scale factor for Poisson mean (tuning knob)",
    )
    parser.add_argument(
        "--chunk_bytes",
        type=int,
        default=32,
        help="Virtual chunk size in bytes (each chunk ≈ this many bits in the stream)",
    )
    parser.add_argument(
        "--packet_bytes",
        type=int,
        default=1440,
        help="Max packet size in bytes (last packet may be smaller)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 1. Load video
    frames_pil = read_video_into_frames(args.video, nframes=args.nframes)
    if len(frames_pil) == 0:
        print("[ERROR] No frames extracted, exiting.")
        return

    # 2. Prepare GRACE + entropy codec
    print("[INFO] Initializing GRACE adapter and entropy codec...")
    adapter = GraceAdapter(
        {"path": args.model_path, "device": torch.device(device).type},
        use_half=False,
        scale_factor=1.0
    )
    entropy_codec = EntropyCodec()

    # 3. Convert original frames to tensors
    orig_frames = [to_tensor(f).to(device) for f in frames_pil]

    # Logs for JSON
    packets_log = {
        "video": args.video,
        "packet_bytes_max": args.packet_bytes,
        "chunk_bytes": args.chunk_bytes,
        "frames": []
    }
    bitflips_log = {
        "video": args.video,
        "frames": []
    }

    # 4. Simulation loop
    ref_decoded = orig_frames[0].detach()  # perfect I-frame
    rec_frames: List[torch.Tensor] = []

    frame_stats = []  # collect all stats for final table

    chunk_bits = args.chunk_bytes * 8
    fps_assumed = 30.0  # assumed frame rate for send_time

    print("[INFO] Encoding + corrupting with true torchac bits (packet → chunks)...")
    for idx, frame_pil in enumerate(tqdm(frames_pil, desc="GRACE frames")):
        frame_t = orig_frames[idx]

        if idx == 0:
            # I-frame: no entropy coding, no bit errors
            rec = ref_decoded.clone()
            rec_frames.append(rec)
            ssim_val = SSIM(frame_t, rec)
            psnr_val = PSNR(frame_t, rec)

            packets_log["frames"].append({
                "frame": idx,
                "type": "I",
                "packets": []
            })
            bitflips_log["frames"].append({
                "frame": idx,
                "type": "I",
                "bit_errors": []
            })

            frame_stats.append({
                "frame": idx,
                "type": "I",
                "bits": 0,
                "bit_errs": 0,
                "chunks": 0,
                "bad_chunks": 0,
                "approx_zero_bits": 0,
                "ssim": ssim_val,
                "psnr": psnr_val,
            })
            continue

        # -------------------------
        # Encode P-frame with GRACE
        # -------------------------
        with torch.no_grad():
            code, dec_frame = adapter.encode(frame_t, ref_decoded)

        # Compute true bits from torchac
        true_bits = compute_true_bits_from_torchac(code, entropy_codec)

        # Build virtual mapping (latent blocks tied to chunk_bits via bits/symbol)
        frame_info = build_virtual_mapping(code, total_bits=true_bits, chunk_bits=chunk_bits)

        # Build per-frame packet layout (<= packet_bytes each)
        packets = build_packets(true_bits, packet_bytes=args.packet_bytes)

        # Build packet log entry
        send_time = idx / fps_assumed
        packet_records = []
        for pid, (bs, be) in enumerate(packets):
            size_bits = be - bs
            size_bytes = int(math.ceil(size_bits / 8.0)) if size_bits > 0 else 0
            packet_records.append({
                "packet_id": pid,
                "bit_start": int(bs),
                "bit_end": int(be),
                "size_bytes": size_bytes,
                "send_time": float(send_time),
            })
        packets_log["frames"].append({
            "frame": idx,
            "type": "P",
            "packets": packet_records
        })

        # -------------------------
        # Sample bit errors (global bit indices)
        # -------------------------
        bit_positions = sample_bit_errors(
            total_bits=true_bits,
            bit_error_rate=args.bit_error_rate,
            poisson_scale=args.poisson_scale,
        )

        # Convert those into (packet_id, bit_offset) and log them
        frame_bit_errors = []
        for b in bit_positions:
            b_int = int(b)
            for pid, (bs, be) in enumerate(packets):
                if bs <= b_int < be:
                    offset = b_int - bs
                    frame_bit_errors.append({
                        "packet_id": pid,
                        "bit_offset": int(offset),
                    })
                    break
        bitflips_log["frames"].append({
            "frame": idx,
            "type": "P",
            "bit_errors": frame_bit_errors
        })

        # -------------------------
        # Now pretend we only know (packet_id, bit_offset) from JSON
        # Reconstruct global bit positions from logs
        # -------------------------
        packet_record_for_frame = packets_log["frames"][-1]
        bitflip_record_for_frame = bitflips_log["frames"][-1]

        packet_dict = {p["packet_id"]: p for p in packet_record_for_frame["packets"]}

        global_bits_from_json: List[int] = []
        for err in bitflip_record_for_frame["bit_errors"]:
            pid = err["packet_id"]
            offset = err["bit_offset"]
            if pid not in packet_dict:
                continue
            p = packet_dict[pid]
            bs = p["bit_start"]
            be = p["bit_end"]
            gb = bs + offset
            if gb < be:
                global_bits_from_json.append(gb)

        global_bits_arr = np.array(global_bits_from_json, dtype=np.int64)

        # Map bit errors → chunk indices (chunks are  chunk_bits in bit-space)
        unique_bad_chunks = bits_to_chunks(
            bit_positions=global_bits_arr,
            chunk_bits=chunk_bits,
            n_chunks=frame_info["n_chunks"],
        )

        # Apply corruption to code (in-place) based solely on chunk indices
        approx_zero_bits = apply_chunk_corruption_to_code(frame_info, unique_bad_chunks)

        # Decode corrupted frame
        with torch.no_grad():
            rec = adapter.decode(frame_info["code"], ref_decoded)
        rec = rec.to(device)
        rec_frames.append(rec)
        ref_decoded = rec.detach()  # error propagation

        ssim_val = SSIM(frame_t, rec)
        psnr_val = PSNR(frame_t, rec)

        frame_stats.append({
            "frame": idx,
            "type": "P",
            "bits": int(true_bits),
            "bit_errs": int(len(global_bits_arr)),
            "chunks": int(frame_info["n_chunks"]),
            "bad_chunks": int(len(unique_bad_chunks)),
            "approx_zero_bits": int(round(approx_zero_bits)),
            "ssim": ssim_val,
            "psnr": psnr_val,
        })

    # -------------------------
    # Write JSON logs
    # -------------------------
    with open("packets.json", "w") as f:
        json.dump(packets_log, f, indent=2)
    with open("bitflips.json", "w") as f:
        json.dump(bitflips_log, f, indent=2)

    print("[INFO] Wrote packets.json and bitflips.json")

    # 5. Print summary table
    print("\n========== FRAME STATS ==========")
    header = (
        f"{'frame':>5}  {'type':>4}  {'bits':>10}  {'bit_errs':>8}  "
        f"{'chunks':>6}  {'bad_chunks':>10}  {'approx_zero_bits':>17}  "
        f"{'SSIM':>8}  {'PSNR(dB)':>9}"
    )
    print(header)
    print("-" * len(header))
    for st in frame_stats:
        print(
            f"{st['frame']:5d}  {st['type']:>4}  "
            f"{st['bits']:10d}  {st['bit_errs']:8d}  "
            f"{st['chunks']:6d}  {st['bad_chunks']:10d}  "
            f"{st['approx_zero_bits']:17d}  "
            f"{st['ssim']:8.4f}  {st['psnr']:9.3f}"
        )

    # Overall stats
    ssim_vals = [s["ssim"] for s in frame_stats]
    mean_ssim = float(np.mean(ssim_vals))
    ssim_db = 10.0 * math.log10(1.0 / max(1e-12, 1.0 - mean_ssim))
    psnr_vals = [s["psnr"] for s in frame_stats]
    mean_psnr = float(np.mean(psnr_vals))

    print("\n[INFO] Overall:")
    print(f"  Mean SSIM: {mean_ssim:.6f}")
    print(f"  SSIM (dB): {ssim_db:.3f} dB")
    print(f"  Mean PSNR: {mean_psnr:.3f} dB")

    # -------------------------
    # Save output video
    # -------------------------
    print("[INFO] Saving output video...")

    out_dir = "recon_frames"
    os.makedirs(out_dir, exist_ok=True)

    # Save frames as PNG images
    import torchvision.utils as vutils
    for i, frame in enumerate(rec_frames):
        img = frame.clamp(0, 1).cpu()
        vutils.save_image(img, f"{out_dir}/{i:04d}.png")

    output_video = "reconstructed_output.mp4"
    fps = 30  # assumed fps

    cmd = (
        f"ffmpeg -y -framerate {fps} -i {out_dir}/%04d.png "
        f"-pix_fmt yuv420p -crf 18 {output_video}"
    )
    os.system(cmd)

    print(f"[INFO] Output video saved as: {output_video}")


if __name__ == "__main__":
    main()
