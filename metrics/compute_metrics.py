
#!/usr/bin/env python3
"""batch_metrics.py – **Referenz‑Klone** der Streamlit‑Funktionen
===============================================================================
Dieses Skript übernimmt *wortwörtlich* den Pipeline‑Ablauf aus
`compare_videos.py` ‑ inklusive

1. **enable H.264‑Reencode** via *ffmpeg* (≙ `ensure_h264()`),
2. identischem *pre‑processing* (Resize → Padding),
3. Aufruf von `custom_metrics.compute()` aus deiner Originaldatei,
4. Neu‑Import pro Videopaar → keine globalen Akkumulator‑Leaks,
5. TXT‑Export (Tab‑getrennt) mit MSE, SSIM, Intrusion.

Damit sind alle Zahlen nun *bit‑genau* mit dem Streamlit‑UI identisch.

--------------------------------------------------------------------------------
🔧 **CONFIG** – Pfade & Device anpassen
--------------------------------------------------------------------------------
```python
ROOT_DIR          = Path(__file__).resolve().parent
GT_DIR            = ROOT_DIR / "data" / "GroundTruth"
GEN_DIR           = ROOT_DIR / "data" / "Generated"
OUT_FILE          = ROOT_DIR / "metrics.txt"
CUSTOM_METRICS_PY = ROOT_DIR / "metrics" / "custom_metrics.py"
VIDEO_EXTS        = {".mp4", ".avi", ".mov", ".mkv"}
DEVICE            = "cpu"        # "cuda" für GPU
FFPROBE           = shutil.which("ffprobe") or "ffprobe"
FFMPEG            = shutil.which("ffmpeg")  or "ffmpeg"
```

Aufrufen → `python batch_metrics.py`
-------------------------------------------------------------------------------
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import cv2  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG --------------------------------------------------------------------
# ---------------------------------------------------------------------------
ROOT_DIR          = Path(__file__).resolve().parent
GT_DIR = Path(
    "/home/azureuser/cloudfiles/code/Users/dieter.holstein/runs/DataPreparation/CogVideo/Split/TestReference/test/videos"
)
GEN_DIR = Path(
    "/home/azureuser/cloudfiles/code/Users/dieter.holstein/runs/HuggingFace/streamlit/data/Generated"
)
OUT_FILE = Path(
    "/home/azureuser/cloudfiles/code/Users/dieter.holstein/runs/HuggingFace/CogKitFix/metrics/metrics.txt"
)
CUSTOM_METRICS_PY = Path(
    "/home/azureuser/cloudfiles/code/Users/dieter.holstein/runs/HuggingFace/CogKitFix/metrics/custom_metrics.py"
)
VIDEO_EXTS        = {".mp4", ".avi", ".mov", ".mkv"}
DEVICE            = "cpu"  # oder "cuda"
FFPROBE           = shutil.which("ffprobe") or "ffprobe"
FFMPEG            = shutil.which("ffmpeg")  or "ffmpeg"

# ---------------------------------------------------------------------------
# 1️⃣  Hilfsfunktionen aus compare_videos.py  (unverändert kopiert)
# ---------------------------------------------------------------------------

def _ffprobe_streams(path: str) -> dict:
    """Liest Codec‑Infos via *ffprobe* (fallback = h264/aac)."""
    if not FFMPEG:
        return {"v_codec": "h264", "a_codec": "aac"}
    try:
        cmd = [FFPROBE, "-v", "error", "-show_streams", "-of", "json", path]
        info = json.loads(subprocess.check_output(cmd))
        v = next((s["codec_name"] for s in info["streams"] if s["codec_type"] == "video"), None)
        a = next((s["codec_name"] for s in info["streams"] if s["codec_type"] == "audio"), None)
        return {"v_codec": v, "a_codec": a}
    except Exception:
        return {"v_codec": "h264", "a_codec": "aac"}


def _ensure_h264(path: Path) -> Path:
    """Gibt denselben oder re‑encodierten MP4‑Pfad (H.264/AAC) zurück."""
    probe = _ffprobe_streams(str(path))
    if probe["v_codec"] == "h264" and probe["a_codec"] in ("aac", None):
        return path

    # Cache‑Datei anhand MD5
    with open(path, "rb") as f:
        import hashlib
        digest = hashlib.md5(f.read()).hexdigest()
    cache_dir = Path(tempfile.gettempdir()) / "st_video_cache"
    cache_dir.mkdir(exist_ok=True)
    out = cache_dir / f"{digest}.mp4"
    if out.exists():
        return out

    # Re‑Encode mit ffmpeg (CRF 23 wie im UI)
    cmd = [
        FFMPEG, "-i", str(path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "faststart", "-pix_fmt", "yuv420p",
        str(out), "-y",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out


# ---------------------------------------------------------------------------
# 2️⃣  Video‑IO & Alignment (identisch zur Streamlit‑Version)
# ---------------------------------------------------------------------------

def _read_video_frames(path: Path) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()
    if not frames:
        return torch.empty((0, 3, 0, 0), dtype=torch.float32)
    arr = torch.from_numpy(np.stack(frames, axis=0))
    return (arr.permute(0, 3, 1, 2).float() / 255.0).to(DEVICE)


def _align(gt: torch.Tensor, gen: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # --- Räumlich ---
    _, _, H1, W1 = gt.shape
    _, _, H2, W2 = gen.shape
    H, W = min(H1, H2), min(W1, W2)
    if (H1, W1) != (H, W):
        gt = F.interpolate(gt, size=(H, W), mode="bilinear", align_corners=False)
    if (H2, W2) != (H, W):
        gen = F.interpolate(gen, size=(H, W), mode="bilinear", align_corners=False)

    # --- Temporales Padding ---
    T = max(gt.shape[0], gen.shape[0])
    if gt.shape[0] < T:
        gt = torch.cat([gt, gt[-1:].repeat(T - gt.shape[0], 1, 1, 1)], 0)
    if gen.shape[0] < T:
        gen = torch.cat([gen, gen[-1:].repeat(T - gen.shape[0], 1, 1, 1)], 0)

    return gt.contiguous(), gen.contiguous()


# ---------------------------------------------------------------------------
# 3️⃣  custom_metrics‑Modul dynamisch & isoliert laden
# ---------------------------------------------------------------------------

def _load_metrics_module(path: Path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"custom_metrics.py nicht gefunden: {p}")
    spec = importlib.util.spec_from_file_location("custom_metrics_tmp", str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Spec konnte nicht erzeugt werden für {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.pop("custom_metrics_tmp", None)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# 4️⃣  Batch‑Durchlauf
# ---------------------------------------------------------------------------

def _list_basenames(folder: Path) -> set[str]:
    return {p.stem for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS}


def _find_video(folder: Path, basename: str) -> Path | None:
    for ext in VIDEO_EXTS:
        p = folder / f"{basename}{ext}"
        if p.exists():
            return p
    return None


def main() -> None:
    if not (GT_DIR.exists() and GEN_DIR.exists()):
        sys.exit("❌ GroundTruth- oder Generated-Ordner nicht gefunden.")
    common = _list_basenames(GT_DIR) & _list_basenames(GEN_DIR)
    if not common:
        sys.exit("❌ Keine gemeinsamen Videodateien gefunden.")

    lines = ["MSE\t\t\tSSIM\t\tINTRUSION\tfile"]
    for base in tqdm(sorted(common), desc="Berechne Metriken"):
        p_gt = _find_video(GT_DIR, base)
        p_gen = _find_video(GEN_DIR, base)
        if p_gt is None or p_gen is None:
            print(f"⚠️  Überspringe {base}: Datei fehlt")
            continue

        try:
            # -- Re‑Encode falls nötig (exakt wie im UI) --
            p_gt_h264  = _ensure_h264(p_gt)
            p_gen_h264 = _ensure_h264(p_gen)

            # -- Frames laden --
            gt = _read_video_frames(p_gt_h264)
            gen = _read_video_frames(p_gen_h264)
            if gt.numel() == 0 or gen.numel() == 0:
                raise RuntimeError("Leeres Video")

            # -- Alignment --
            gt, gen = _align(gt, gen)

            # -- Metriken --
            metrics_mod = _load_metrics_module(CUSTOM_METRICS_PY)
            res: Dict[str, Tuple[float, str]] = metrics_mod.compute(gt, gen)  # type: ignore[arg-type]

            mse       = res["mse"][0]
            ssim      = res["ssim"][0]
            intrusion = res["intrusion"][0]
            lines.append(f"{mse:.6f}\t{ssim:.6f}\t{intrusion:.6f}\t{base}")

        except Exception as e:
            print(f"❌ Fehler bei {base}: {e}")
            continue

    # -- Schreiben --
    OUT_FILE.write_text("\n".join(lines))
    print(f"\n✅ Fertig → {OUT_FILE}")


if __name__ == "__main__":
    main()
