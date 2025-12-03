import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

def read_df(path):
    if os.path.exists(path):
        return pd.read_csv(path).query("frame_id < 16")
    else:
        print("Skip the file because", path, "does not exist!")
        return None

def quick_plot_size(df, label):
    if df is None:
        print("Skip", label, "during size plot because dataframe is None")
        return

    df = df.sort_values("size").reset_index(drop=True)
    if "ssim_db" not in df.columns:
        df["ssim_db"] = -10 * np.log10(1 - df["ssim"])
    plt.plot(df["size"], df["ssim_db"], label=label)

def quick_plot_loss(df, label):
    if df is None:
        print("Skip", label, "during loss plot because dataframe is None")
        return
    df = df.sort_values("loss").reset_index(drop=True)
    if "ssim_db" not in df.columns:
        df["ssim_db"] = -10 * np.log10(1 - df["ssim"])
    plt.plot(df["loss"], df["ssim_db"], label=label)

def interpolate_quality(df, target_size):
    """
    assume input df has loss, size, ssim
    """
    if df is None:
        print("Skip", label, "during quality interpolation because dataframe is None")
        return
    df = df.sort_values(['loss', 'size'])
    def group_interpolate(group):
        return pd.Series({'ssim': np.interp(target_size, group['size'], group['ssim'])})

    result = df.groupby(["loss"]).apply(group_interpolate).reset_index()
    result["ssim_db"] = -10 * np.log10(1 - result["ssim"])
    return result

def quick_plot_fec(df, target_size, fec_ratio):
    if df is None:
        print("Skip", label, "during fec plot because dataframe is None")
        return
    idf = interpolate_quality(df.groupby(["loss", "model"]).mean().reset_index(), size * (1-fec_ratio))
    quality = float(idf["ssim_db"])
    x = [0, fec_ratio - 0.01, fec_ratio + 0.01]
    y = [quality, quality, 8]
    plt.plot(x, y, label=f"{fec_ratio*100:.1f}% FEC")

# Optional video filter - set to None to include all videos
# Examples: "video.str.contains('game')" or "video == 'video-0.mp4'"
video_filter = None  # Use all videos

df_grace = read_df("grace/all.csv")
if df_grace is not None:
    if video_filter is not None:
        df_grace = df_grace.query(video_filter)
    print("Grace videos:", df_grace["video"].unique() if df_grace is not None and len(df_grace) > 0 else "No data")
    
df_265 = read_df("h265/all.csv")
if df_265 is not None and video_filter is not None:
    df_265 = df_265.query(video_filter)
    
df_264 = read_df("h264/all.csv")
if df_264 is not None and video_filter is not None:
    df_264 = df_264.query(video_filter)
    
df_pretrain = read_df("pretrained/all.csv")
if df_pretrain is not None and video_filter is not None:
    df_pretrain = df_pretrain.query(video_filter)
    
df_error = None #read_df("error_concealment/all.csv").query(video_filter)

''' QUALITY VS SIZE CURVE '''
fig = plt.figure()
if df_grace is not None:
    quick_plot_size(df_grace.query("nframes == 0").groupby("model_id").mean().reset_index(), "grace")
if df_pretrain is not None:
    quick_plot_size(df_pretrain.query("nframes == 0").groupby("model_id").mean().reset_index(), "pretrained")
if df_265 is not None:
    quick_plot_size(df_265.groupby("model").mean().reset_index(), "265")
if df_264 is not None:
    quick_plot_size(df_264.groupby("model").mean().reset_index(), "264")
plt.xlim(0, 30000)
plt.xlabel('Size (bytes)', fontsize=12)
plt.ylabel('SSIM (dB)', fontsize=12)
plt.title('Quality vs Size', fontsize=14)
plt.grid()
plt.legend()
fig.savefig("ssim_size.png")


''' QUALITY VS LOSS CURVE '''
if df_grace is not None:
    try:
        # Get reference size from model 4096 with nframes=1
        size_query = df_grace.query("nframes == 1 and model_id == 4096")
        if len(size_query) > 0:
            size = float(size_query.mean()["size"])
            for nframes in [1,3,5]:
                fig = plt.figure()
                if df_grace is not None:
                    grace_data = df_grace.query("nframes == @nframes").groupby(["loss", "model_id"], as_index=False).mean()
                    quick_plot_loss(interpolate_quality(grace_data, size), "grace")
                if df_error is not None:
                    error_data = df_error.query("nframes == @nframes").groupby(["loss", "model_id"], as_index=False).mean()
                    quick_plot_loss(interpolate_quality(error_data, size), "error concealment")
                if df_pretrain is not None:
                    pretrain_data = df_pretrain.query("nframes == @nframes").groupby(["loss", "model_id"], as_index=False).mean()
                    quick_plot_loss(interpolate_quality(pretrain_data, size), "pretrained")
                if df_265 is not None:
                    quick_plot_fec(df_265, size, 0.2)
                    quick_plot_fec(df_265, size, 0.5)
                if df_264 is not None:
                    quick_plot_fec(df_264, size, 0.2)
                    quick_plot_fec(df_264, size, 0.5)
                plt.xlabel('Loss Rate', fontsize=12)
                plt.ylabel('SSIM (dB)', fontsize=12)
                plt.title(f'Quality vs Loss Rate (nframes={nframes})', fontsize=14)
                plt.grid()
                plt.legend()
                fig.savefig(f"ssim_loss-{nframes}.png")
        else:
            print("Skipping loss curve plots: no data found for model_id='4096' with nframes=1")
    except Exception as e:
        print(f"Skipping loss curve plots: {e}")
else:
    print("Skipping loss curve plots because grace data is not available")
