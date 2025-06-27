# compare_videos.py

# Start the streamlit Environment

import os
import random
import tempfile
import subprocess
import hashlib
import json
import base64
import shutil
import imageio.v2 as iio

import streamlit as st
st.set_page_config(page_title="Video Comparison",
                   layout="wide",
                   page_icon="⚗️")

import streamlit.components.v1 as components
import textwrap

import torch
import torch.nn.functional as F
import cv2
import numpy as np    
import importlib.util

from torchmetrics import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio
from head_tracking import BlobHeadTracker, rightmost_x
from pathlib import Path
from torchmetrics.image import StructuralSimilarityIndexMeasure
import pandas as pd
from skimage.metrics import structural_similarity as ski_ssim

import tempfile
import imageio.v2 as iio
from skimage.metrics import structural_similarity as ski_ssim


if "metrics_ready" not in st.session_state:
    st.session_state.metrics_ready = False


PAGES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PAGES_DIR.parent

CUSTOM_METRICS_FILE = PROJECT_ROOT / "metrics" / "custom_metrics.py"

# Holt die Visualisierungs-Konfiguration aus custom_metrics.py
try:
    spec_cfg = importlib.util.spec_from_file_location("custom_metrics_cfg", CUSTOM_METRICS_FILE)
    mod_cfg  = importlib.util.module_from_spec(spec_cfg)
    spec_cfg.loader.exec_module(mod_cfg)
    _METRIC_CFG = mod_cfg._METRIC_CFG
except Exception as e:
    st.error(f"🚨 Konnte _METRIC_CFG nicht laden: {e}")
    _METRIC_CFG = {}

# --- Konfiguration: Pfade & Extensions ---
GT_FOLDER  = "/home/azureuser/cloudfiles/code/Users/dieter.holstein/runs/DataPreparation/CogVideo/Split/TestReference/test/videos"
GEN_FOLDER = PROJECT_ROOT / "output"
VIDEO_EXT  = {".mp4", ".avi", ".mov", ".mkv"}

# --- Helferfunktionen ---
def list_basenames(folder):
    return {os.path.splitext(f)[0] for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in VIDEO_EXT}

def random_choice(names):
    return random.choice(sorted(names))

def find_file(folder, base):
    for f in os.listdir(folder):
        n, ext = os.path.splitext(f)
        if n == base and ext.lower() in VIDEO_EXT:
            return os.path.join(folder, f)
    return None

def ffprobe_streams(path: str) -> dict:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        st.warning("⚠️ ffprobe nicht gefunden - angenommen wird H.264/AAC.")
        return {"v_codec": "h264", "a_codec": "aac"}
    cmd = [ffprobe, "-v", "error", "-show_streams", "-of", "json", path]
    info = json.loads(subprocess.check_output(cmd))
    v = next((s["codec_name"] for s in info["streams"] if s["codec_type"]=="video"), None)
    a = next((s["codec_name"] for s in info["streams"] if s["codec_type"]=="audio"), None)
    return {"v_codec": v, "a_codec": a}

@st.cache_data(show_spinner=False)
def ensure_h264(path: str) -> str:
    probe = ffprobe_streams(path)
    if probe["v_codec"] == "h264" and probe["a_codec"] in ("aac", None):
        return path
    with open(path, "rb") as f:
        digest = hashlib.md5(f.read()).hexdigest()
    cache_dir = os.path.join(tempfile.gettempdir(), "st_video_cache")
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, f"{digest}.mp4")
    if os.path.exists(out):
        return out
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        st.error("❌ ffmpeg nicht gefunden - bitte installiere es.")
        st.stop()
    subprocess.run([
        ffmpeg, "-i", path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "faststart", "-pix_fmt", "yuv420p",
        out, "-y"
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out

def read_video_frames(path: str) -> torch.Tensor:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    if not frames:
        return torch.empty((0,3,0,0), dtype=torch.float32)
    arr = torch.from_numpy(np.stack(frames, axis=0))
    return arr.permute(0,3,1,2).float() / 255.0

# ----- Funktion für die erstellung der intrusionsvideos -------

def _write_annot_video(frames: torch.Tensor, xs: np.ndarray, fps: int = 25) -> str:
    if frames.numel()==0 or len(xs)==0:
        return ""
    T, _, H, W = frames.shape
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with iio.get_writer(tmp.name,
                        format="ffmpeg",
                        mode="I",
                        fps=fps,
                        codec="libx264",
                        pixelformat="yuv420p",
                        bitrate="2M") as w:
        for t in range(T):
            rgb = (frames[t].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.line(bgr, (int(xs[t]),0), (int(xs[t]),H-1), (0,0,255), 2)  # Rot
            w.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return tmp.name



@st.cache_data(show_spinner=False)
def compute_metrics(gt_path: str, gen_path: str) -> dict:
    """
    Pre-Processing (Resize + Padding) bleibt hier.
    Die eigentlichen Kennzahlen kommen aus custom_metrics.py.
    """
    # ------------------------------------------------------------------
    # 1) Frames laden
    gt  = read_video_frames(gt_path)   # [T1,3,H1,W1] float in [0,1]
    gen = read_video_frames(gen_path)  # [T2,3,H2,W2]
    if gt.numel() == 0 or gen.numel() == 0:
        st.error("Videos konnten nicht geladen werden.")
        # immer drei Werte zurückgeben, damit das Unpacking nicht scheitert
        return {}, gt, gen

    # ------------------------------------------------------------------
    # 2) Räumliches Alignment
    _, _, H1, W1 = gt.shape
    _, _, H2, W2 = gen.shape
    H, W = min(H1, H2), min(W1, W2)
    gt  = F.interpolate(gt,  size=(H, W), mode="bilinear", align_corners=False)
    gen = F.interpolate(gen, size=(H, W), mode="bilinear", align_corners=False)

    # ------------------------------------------------------------------
    # 3) Temporales Padding
    T = max(gt.shape[0], gen.shape[0])
    if gt.shape[0] < T:
        gt  = torch.cat([gt,  gt[-1:].repeat(T-gt.shape[0],1,1,1)], 0)
    if gen.shape[0] < T:
        gen = torch.cat([gen, gen[-1:].repeat(T-gen.shape[0],1,1,1)], 0)

    # ------------------------------------------------------------------
    # 4) custom_metrics.py dynamisch laden
    try:
        spec = importlib.util.spec_from_file_location("custom_metrics", CUSTOM_METRICS_FILE)
        metrics_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics_mod)
    except FileNotFoundError:
        st.error(f"❌ {CUSTOM_METRICS_FILE} nicht gefunden.")
        return {}
    except Exception as e:
        st.error(f"❌ Fehler beim Laden von {CUSTOM_METRICS_FILE}: {e}")
        return {}

    # ------------------------------------------------------------------
    # 5) Metriken berechnen
    try:
        metrics = metrics_mod.compute(gt.contiguous(), gen.contiguous())
    except Exception as e:
        st.error(f"❌ Fehler in custom_metrics.compute(): {e}")
        # auch hier drei Werte zurückgeben
        return {}, gt, gen

    return metrics, gt, gen

def metric_bar(name: str, value: float, help_txt: str):
    """Zeigt Wert + farbigen Balken + Tooltip."""
    vmin, vmax, direction = _METRIC_CFG[name]
    # clamp & normalisieren ---------------------------------------------------
    value_clamped = max(vmin, min(value, vmax))
    if direction == "higher":
        frac = (value_clamped - vmin) / (vmax - vmin)          # 0 = rot, 1 = grün
    else:  # lower-is-better ⇒ invertieren
        frac = (vmax - value_clamped) / (vmax - vmin)

    # HTML-Balken mit Verlauf -------------------------------------------------
    bar_html = textwrap.dedent(f"""
        <div style="position:relative;height:14px;border-radius:3px;
                    background:linear-gradient(to right, #d73027, #fdae61, #a6d96a, #1a9850);">
            <div style="position:absolute;left:calc({frac*100:.1f}% - 6px);top:-4px;
                        width:0;height:0;border-left:6px solid transparent;
                        border-right:6px solid transparent;border-bottom:6px solid #000;">
            </div>
        </div>
    """)

    st.markdown(f"**{name.upper()} = {value:.4f}**", help=help_txt)
    st.markdown(bar_html, unsafe_allow_html=True)

def _annotate_frame(frame_tensor: torch.Tensor, x_pos: int) -> np.ndarray:
    """
    Malt einen roten Vertikalstrich in das Frame und gibt ein RGB-uint8-Bild zurück.
    """
    rgb = (frame_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if x_pos >= 0:
        cv2.line(bgr, (int(x_pos), 0), (int(x_pos), bgr.shape[0] - 1), (0, 0, 255), 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# ------------------------------------------------------------------
#   Einheitliche Intrusion-Helpers via HeadTracker
# ------------------------------------------------------------------

def _tensor_hash(t: torch.Tensor):
    return hash((t.shape, float(t.sum())))

@st.cache_data(show_spinner=False, hash_funcs={torch.Tensor: _tensor_hash})

def _intrusion_track(frames: torch.Tensor):
    tracker = BlobHeadTracker(dE_thr=18, area_min=150, max_shift=140)
    masks, _ = tracker.track(frames)
    xs   = np.array([ rightmost_x(m) for m in masks ], dtype=int)
    W    = frames.shape[-1]
    d    = xs / (W - 1)
    idx  = int(np.nanargmax(d)) if xs.size else -1
    return xs, float(d[idx]) if idx>=0 else 0.0, idx, masks



# ------------------------------------------------------------------
#  Masken-Overlay-Video (Debugging)
# ------------------------------------------------------------------

def _write_mask_overlay_video(frames: torch.Tensor,
                              masks: list[np.ndarray],
                              fps: int = 25) -> str:
    """
    Erzeugt ein MP4-Video, in dem die Head-Maske halbtransparent
    (pink) über das Originalframe gelegt wird.
    """
    if frames.numel() == 0 or not masks:
        return ""

    T, _, H, W = frames.shape
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    # pinke Farbe (RGB) und Alpha
    color = np.array([255,  0, 0], dtype=np.uint8)
    alpha = 0.45                                          # Transparenz

    with iio.get_writer(tmp.name, format="ffmpeg", mode="I",
                        fps=fps, codec="libx264",
                        pixelformat="yuv420p",
                        bitrate="2M") as w:
        for t in range(T):
            rgb = (frames[t].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)

            m = masks[t]
            if m is not None:
                # ---------------------------------------
                # Form-Check  → ggf. transponieren
                # ---------------------------------------
                if m.shape != rgb.shape[:2]:
                    if m.T.shape == rgb.shape[:2]:
                        m = m.T
                    else:                      # Fallback: Größen-Resample
                        m = cv2.resize(m.astype(np.uint8),
                                       (rgb.shape[1], rgb.shape[0]),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                overlay = rgb.copy()
                overlay[m] = color
                rgb = cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)

            w.append_data(rgb)

    return tmp.name


# --- Streamlit UI ---

st.title("Video Comparison")

common = sorted(list_basenames(GT_FOLDER) & list_basenames(GEN_FOLDER))
if not common:
    st.error("Keine gemeinsamen Videos gefunden.")
    st.stop()

if "sel" not in st.session_state:
    st.session_state.sel = common[0]

st.sidebar.button("🔀 Zufällig", on_click=lambda: st.session_state.update(sel=random_choice(common)))
st.sidebar.selectbox("Datei wählen:", common, key="sel")
basename = st.session_state.sel

gt_src  = ensure_h264(find_file(GT_FOLDER,  basename))
gen_src = ensure_h264(find_file(GEN_FOLDER, basename))

with open(gt_src,  "rb") as f: gt_b = base64.b64encode(f.read()).decode()
with open(gen_src, "rb") as f: gen_b = base64.b64encode(f.read()).decode()

# HTML-CSS
html = f"""
<style>
  .play-btn-container {{
    display: flex;
    justify-content: center;
    gap: 24px;               /* Abstand zwischen Buttons */
    margin-bottom: 12px;
  }}
  .videos {{
    display: flex;
    gap: 16px;
    margin-bottom: 8px;
  }}
  .video-wrapper {{
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
  }}
  .video-wrapper strong {{ margin-bottom:4px; }}
  video {{ width:100%; max-height:480px; }}
  .slider-container {{ text-align:center; margin-top:8px; }}
  input[type=range] {{ width:80%; }}

  /* Gestylte Buttons im Streamlit-Look mit dünnem schwarzen Rand */
  .streamlit-btn {{
    all: unset;
    box-sizing: border-box;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: "Inter", sans-serif;  /* Streamlit-Default */
    font-size: 0.875rem;
    padding: 0.5em 1em;
    border-radius: 4px;
    border: 1px solid #d6d6d9;      /* dünner schwarzer Rand */
    background-color: var(--secondary-background-color);
    color: var(--text-color);
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    cursor: pointer;
    transition: background-color 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease, color 0.15s ease;
    margin: 0 8px;
  }}
  /* Hover: Rand und Text in Pastell-Rot */
  .streamlit-btn:hover {{
    border-color: #ff4b4b;
    color: #ff4b4b;
  }}
  /* Active: Hintergrund Pastell-Rot, Text weiß */
  .streamlit-btn:active {{
    background-color: #ff4b4b;
    color: #fff;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.15);
    border-color: #ff4b4b;
  }}
</style>

<div class="play-btn-container">
  <button class="streamlit-btn" id="playBoth">▶ Play Both</button>
  <button class="streamlit-btn" id="playLoop">Loop</button>
</div>
<div class="videos">
  <div class="video-wrapper">
    <strong>GroundTruth</strong>
    <video id="v1" controls src="data:video/mp4;base64,{gt_b}"></video>
  </div>
  <div class="video-wrapper">
    <strong>GeneratedVideo</strong>
    <video id="v2" controls src="data:video/mp4;base64,{gen_b}"></video>
  </div>
</div>
<div class="slider-container">
  <input type="range" id="syncSlider" min="0" max="0" step="any" value="0">
</div>
<script>
  const v1 = document.getElementById("v1");
  const v2 = document.getElementById("v2");
  const btn = document.getElementById("playBoth");
  const loopBtn = document.getElementById("playLoop");
  const slider = document.getElementById("syncSlider");
  let looping = false;

  Promise.all([
    new Promise(r => v1.onloadedmetadata = r),
    new Promise(r => v2.onloadedmetadata = r)
  ]).then(() => {{
    const duration = Math.min(v1.duration, v2.duration);
    slider.max = duration;
  }});

  btn.onclick = () => {{
    looping = false;
    v1.loop = false;
    v2.loop = false;
    loopBtn.innerText = 'Loop';
    v1.pause(); v2.pause();
    v1.currentTime = 0; v2.currentTime = 0;
    slider.value = 0;
    v1.play(); v2.play();
  }};

  loopBtn.onclick = () => {{
    looping = !looping;
    v1.loop = looping;
    v2.loop = looping;
    loopBtn.innerText = looping ? '⏹ Stop Loop' : 'Loop';
    if (looping) {{
      v1.currentTime = 0; v2.currentTime = 0;
      slider.value = 0;
      v1.play(); v2.play();
    }}
  }};

  slider.oninput = () => {{
    const t = parseFloat(slider.value);
    v1.currentTime = t;
    v2.currentTime = t;
  }};

  [v1, v2].forEach(v => {{ v.ontimeupdate = () => {{ slider.value = v.currentTime; }} }});
</script>
"""



components.html(html, height=720)

st.divider()
st.header("Metriken")

if st.button("Metriken berechnen"):
    metrics, frames_gt, frames_gen = compute_metrics(gt_src, gen_src)

    # in session_state ablegen ------------
    st.session_state.metrics       = metrics
    st.session_state.frames_gt     = frames_gt
    st.session_state.frames_gen    = frames_gen
    st.session_state.metrics_ready = True


# ------------------------------------------------------------------------------
#  ⬇️  NEU: Darstellung – nur wenn schon berechnet ----------------------------
# ------------------------------------------------------------------------------
if st.session_state.metrics_ready:
    metrics     = st.session_state.metrics
    frames_gt   = st.session_state.frames_gt
    frames_gen  = st.session_state.frames_gen

    # ----------  KPI-Kacheln --------------------------------------------------
    keys = list(metrics.keys())
    for i in range(0, len(keys), 2):
        cols = st.columns(2)
        for j, key in enumerate(keys[i:i+2]):
            val, help_text = metrics[key]
            with cols[j]:
                metric_bar(key, val, help_text)
                st.write("")

    # ——— SSIM-Verlauf pro Frame ———
    st.subheader("SSIM-Verlauf pro Frame")
    # 1) SSIM pro Frame berechnen
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim_values = []
    T = frames_gt.shape[0]
    for t in range(T):
        gt_f  = frames_gt[t:t+1]    # Batch-Dimension 1×3×H×W
        gen_f = frames_gen[t:t+1]
        ssim_values.append(ssim_metric(gt_f, gen_f).item())
        ssim_metric.reset()

    # ——— SSIM-Heatmap als Videosequenz ———
    st.subheader("SSIM-Heatmap Video")

    # 1) win_size berechnen wie gehabt
    gt0 = (frames_gt[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    h, w, _ = gt0.shape
    win = min(h, w, 7)
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)

    # 2) Temp-Datei für das MP4
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = iio.get_writer(
        out_file, format="ffmpeg", mode="I",
        fps=25, codec="libx264", pixelformat="yuv420p"
    )

    # 3) Alle Frames durchlaufen, SSIM-Map berechnen und als Heatmap ins Video
    T = frames_gt.shape[0]
    for t in range(T):
        # a) RGB-Frames als uint8
        gt_img  = (frames_gt[t].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        gen_img = (frames_gen[t].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

        # b) in Graustufen umwandeln
        gt_gray  = cv2.cvtColor(gt_img,  cv2.COLOR_RGB2GRAY)
        gen_gray = cv2.cvtColor(gen_img, cv2.COLOR_RGB2GRAY)

        # c) SSIM-Map auf Graustufen berechnen
        _, ssim_map = ski_ssim(
            gt_gray, gen_gray,
            win_size=win,
            data_range=255,
            full=True
        )

        # d) normalisieren & Colormap anwenden
        ssim_map = np.clip(ssim_map, 0, 1)
        ssim_u8  = (ssim_map * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(ssim_u8, cv2.COLORMAP_JET)

        # e) in RGB konvertieren und ins Video schreiben
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(heatmap_rgb)

    writer.close()

    # 4) Video abspielen
    st.video(out_file)



    # ——— MSE-Heatmap als Videosequenz ———

    st.subheader("MSE-Heatmap Video")

    # win_size-Berechnung entfällt bei MSE
    # 1) Temp-Datei für Video
    out_mse = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer_mse = iio.get_writer(
        out_mse, format="ffmpeg", mode="I",
        fps=25, codec="libx264", pixelformat="yuv420p"
    )

    T = frames_gt.shape[0]
    for t in range(T):
        # 2) Frames in uint8-Arrays
        gt_img  = (frames_gt[t].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        gen_img = (frames_gen[t].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

        # 3) MSE-Map berechnen (über alle Kanäle mitteln)
        err = (gt_img.astype(np.float32) - gen_img.astype(np.float32))**2
        mse_map = err.mean(axis=2)  # H×W

        # 4) Normierung auf [0,1] – hier als Beispiel linear über maximalen MSE-Wert 65025
        #    (255²=65025). Du kannst stattdessen max(mse_map) pro Video nutzen.
        max_err = mse_map.max()            # höchster MSE-Wert in diesem Frame/Video
        norm = mse_map / (max_err + 1e-8)  # vermeidet Division durch Null
        norm = np.clip(norm, 0, 1)

        #norm = np.clip(mse_map / 65025.0, 0, 1)

        # 5) Colormap und in RGB konvertieren
        mse_u8 = (norm * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(mse_u8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        writer_mse.append_data(heatmap_rgb)

    writer_mse.close()

    # 6) Ausgabe
    st.video(out_mse)



    # ----------  Intrusion-Frames anzeigen ----------
    if "intrusion" in metrics:
        st.subheader("Maximale Intrusion - visuelle Kontrolle")

        xs_gt,  depth_gt,  idx_gt, masks_gt    = _intrusion_track(frames_gt)
        xs_gen, depth_gen, idx_gen, masks_gen  = _intrusion_track(frames_gen)

        x_gt  = xs_gt[idx_gt]  if idx_gt  >= 0 else -1
        x_gen = xs_gen[idx_gen] if idx_gen >= 0 else -1

        if idx_gt == -1 or idx_gen == -1:
            st.warning("Das lila Kopf-Objekt konnte nicht zuverlässig erkannt werden.")
        else:
            img_gt  = _annotate_frame(frames_gt[idx_gt],  x_gt)
            img_gen = _annotate_frame(frames_gen[idx_gen], x_gen)

            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    img_gt,
                    caption=f"Ground Truth – Frame {idx_gt}  (Tiefe {depth_gt:.3f})",
                    use_container_width=True,
                )
            with col2:
                st.image(
                    img_gen,
                    caption=f"Generiert – Frame {idx_gen}  (Tiefe {depth_gen:.3f})",
                    use_container_width=True,
                )

            st.caption(
                "Der rote Strich markiert die am weitesten rechts liegende Position "
                "des lila Kopf-Objekts in der jeweiligen Sequenz."
            )


            # ----------  Annotierte Videos unten ----------
            st.markdown("### Sequenzen mit Live-Intrusion-Marker")

            # 1) Clips erzeugen ------------------------------------------------
            xs_gt  = xs_gt        # schon berechnet
            xs_gen = xs_gen

            vid_gt_tmp  = _write_annot_video(frames_gt,  xs_gt)
            vid_gen_tmp = _write_annot_video(frames_gen, xs_gen)

            vid_gt  = ensure_h264(vid_gt_tmp)
            vid_gen = ensure_h264(vid_gen_tmp)

            # 2) Base64 einlesen ----------------------------------------------
            with open(vid_gt,  "rb") as f: vid_gt_b  = base64.b64encode(f.read()).decode()
            with open(vid_gen, "rb") as f: vid_gen_b = base64.b64encode(f.read()).decode()

            # 3) HTML-Player (seitliches Layout + gemeinsamer Slider) ---------
            html_annot = f"""
            <style>
              .videos-small {{ display:flex; gap:16px; margin-bottom:6px; }}
              .videos-small video{{ width:100%; max-height:300px; }}
              .slider-small input{{ width:80%; display:block; margin:6px auto; }}
            </style>

            <div class="videos-small">
              <div style="flex:1;text-align:center">
                <strong>GT annotiert</strong><br>
                <video id="av1" controls src="data:video/mp4;base64,{vid_gt_b}"></video>
              </div>
              <div style="flex:1;text-align:center">
                <strong>Gen annotiert</strong><br>
                <video id="av2" controls src="data:video/mp4;base64,{vid_gen_b}"></video>
              </div>
            </div>
            <div class="slider-small">
              <input type="range" id="syncSlider2" min="0" max="0" step="any" value="0">
            </div>

            <script>
              const av1=document.getElementById('av1'),
                    av2=document.getElementById('av2'),
                    s2 =document.getElementById('syncSlider2');

              Promise.all([new Promise(r=>av1.onloadedmetadata=r),
                           new Promise(r=>av2.onloadedmetadata=r)]).then(()=>{{
                   s2.max=Math.min(av1.duration,av2.duration);}});

              [av1,av2].forEach(v=>v.ontimeupdate=()=>{{ s2.value=v.currentTime; }});
              s2.oninput=()=>{{ const t=parseFloat(s2.value); av1.currentTime=t; av2.currentTime=t; }};
            </script>
            """
            components.html(html_annot, height=350)

            # ----------  Masken-Overlay erzeugen ----------
            st.markdown("### 🔍 Masken-Overlay-Videos")

            if st.button("Overlay-Clips rendern"):
                xs_gt,  _, _, masks_gt  = _intrusion_track(frames_gt)   # Masks holen
                xs_gen, _, _, masks_gen = _intrusion_track(frames_gen)

                ov_gt  = _write_mask_overlay_video(frames_gt,  masks_gt)
                ov_gen = _write_mask_overlay_video(frames_gen, masks_gen)

                vid_gt_ov  = ensure_h264(ov_gt)
                vid_gen_ov = ensure_h264(ov_gen)

                with open(vid_gt_ov,  "rb") as f: ov_gt_b  = base64.b64encode(f.read()).decode()
                with open(vid_gen_ov, "rb") as f: ov_gen_b = base64.b64encode(f.read()).decode()

                html_ov = f"""
                <style>.videos-mask{{display:flex;gap:16px}} video{{width:100%;max-height:300px}}</style>
                <div class="videos-mask">
                <div style="flex:1;text-align:center">
                    <strong>GT-Maske</strong><br>
                    <video controls src="data:video/mp4;base64,{ov_gt_b}"></video>
                </div>
                <div style="flex:1;text-align:center">
                    <strong>Gen-Maske</strong><br>
                    <video controls src="data:video/mp4;base64,{ov_gen_b}"></video>
                </div>
                </div>
                """
                components.html(html_ov, height=320)


