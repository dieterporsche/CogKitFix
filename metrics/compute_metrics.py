
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
ROOT_DIR   = Path(__file__).resolve().parent
REPO_ROOT  = ROOT_DIR.parent
GT_DIR     = Path("/path/to/GroundTruth")  # bleibt absolut
OUTPUT_ROOT = REPO_ROOT / "output"
CUSTOM_METRICS_PY = REPO_ROOT / "metrics" / "custom_metrics.py"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
DEVICE     = "cpu"  # "cuda" für GPU
FFPROBE    = shutil.which("ffprobe") or "ffprobe"
FFMPEG     = shutil.which("ffmpeg") or "ffmpeg"
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
from typing import Dict, Tuple, List, Tuple as Tup
import re
import cv2  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG --------------------------------------------------------------------
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent

GT_DIR = Path(
    "/home/azureuser/cloudfiles/code/Users/dieter.holstein/runs/DataPreparation/CogVideo/Split/TestReference/test/videos"
)
# Ordner mit Trainingsausgaben (Output*/validation_Epoch_*)
OUTPUT_ROOT = REPO_ROOT / "output"

# Dynamisch geladene custom_metrics.py aus diesem Repository
CUSTOM_METRICS_PY = REPO_ROOT / "metrics" / "custom_metrics.py"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
DEVICE = "cpu"  # oder "cuda"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


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

def _strip_suffix(name: str) -> str:
    """Remove trailing `_XX00` like pattern from basename."""
    return re.sub(r"_[A-Za-z]{2}\d{2}$", "", name)

def _list_basenames(folder: Path) -> set[str]:
    names: set[str] = set()
    for p in folder.iterdir():
        if p.suffix.lower() in VIDEO_EXTS:
            names.add(_strip_suffix(p.stem))
    return names


def _find_video(folder: Path, basename: str) -> Path | None:
    for p in folder.iterdir():
        if p.suffix.lower() in VIDEO_EXTS and _strip_suffix(p.stem) == basename:
            return p
    return None


def _compute_for_dir(gen_dir: Path) -> Tup[List[str], Tup[float, float, float]]:
    if not gen_dir.exists():
        raise FileNotFoundError(gen_dir)

    common = _list_basenames(GT_DIR) & _list_basenames(gen_dir)

    lines = ["MSE\t\t\tSSIM\t\tINTRUSION\tfile"]
    mses, ssims, intrs = [], [], []

    for base in tqdm(sorted(common), desc=f"{gen_dir.name}"):
        p_gt = _find_video(GT_DIR, base)
        p_gen = _find_video(gen_dir, base)
        if p_gt is None or p_gen is None:
            print(f"⚠️  Überspringe {base}: Datei fehlt")
            continue

        try:
            # -- Re‑Encode falls nötig (exakt wie im UI) --
            p_gt_h264 = _ensure_h264(p_gt)
            p_gen_h264 = _ensure_h264(p_gen)

            gt = _read_video_frames(p_gt_h264)
            gen = _read_video_frames(p_gen_h264)
            if gt.numel() == 0 or gen.numel() == 0:
                raise RuntimeError("Leeres Video")

            gt, gen = _align(gt, gen)

            metrics_mod = _load_metrics_module(CUSTOM_METRICS_PY)
            res: Dict[str, Tuple[float, str]] = metrics_mod.compute(gt, gen)  # type: ignore[arg-type]

            mse = res["mse"][0]
            ssim = res["ssim"][0]
            intrusion = res["intrusion"][0]
            mses.append(mse)
            ssims.append(ssim)
            intrs.append(intrusion)
            lines.append(f"{mse:.6f}\t{ssim:.6f}\t{intrusion:.6f}\t{base}")

        except Exception as e:
            print(f"❌ Fehler bei {base}: {e}")
            continue

    # -- Schreiben --
    from statistics import mean

    avg = (
        float(mean(mses)) if mses else float("nan"),
        float(mean(ssims)) if ssims else float("nan"),
        float(mean(intrs)) if intrs else float("nan"),
    )
    return lines, avg


def main() -> None:
    if not GT_DIR.exists():
        sys.exit("❌ GroundTruth-Ordner nicht gefunden.")

    for out_dir in sorted(OUTPUT_ROOT.glob("Output*")):
        epoch_dirs = sorted(out_dir.glob("validation_Epoch_*"))
        if not epoch_dirs:
            continue

        all_lines = ["------------------------------------------------------------",
                     f"###### {out_dir.name} ######",
                     "------------------------------------------------------------"]
        averages: Dict[str, Tup[float, float, float]] = {}

        for e_dir in epoch_dirs:
            epoch_label = e_dir.name.replace("validation_", "")
            lines, avg = _compute_for_dir(e_dir)
            averages[epoch_label] = avg
            all_lines.append("")
            all_lines.append(f"## {epoch_label}")
            all_lines.extend(lines)

        all_lines.append("------------------------------------------------------------")
        all_lines.append("------------------------------------------------------------")
        all_lines.append("#### Average ALL: ####\n")
        for epoch_label, (m_mse, m_ssim, m_intr) in averages.items():
            all_lines.append(f"{epoch_label}:")
            all_lines.append("MSE\t\t\tSSIM\t\tINTRUSION")
            all_lines.append(f"{m_mse:.6f}\t{m_ssim:.6f}\t{m_intr:.6f}\n")

        out_file = OUTPUT_ROOT / f"metrics_{out_dir.name}.txt"
        out_file.write_text("\n".join(all_lines))
        print(f"\n✅ Fertig → {out_file}")


if __name__ == "__main__":
    main()
