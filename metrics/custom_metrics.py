"""
custom_metrics.py

Alle Qualitäts­metriken zentral an einem Ort.
Die compute(frames_gt, frames_gen) gibt Dict[label] -> (value, help_text) zurück.
"""
from typing import Dict, Tuple
import torch
import numpy as np
import cv2
from head_tracking import BlobHeadTracker

# ----------------------------------------------------------------------
# TorchMetrics-Import
# ----------------------------------------------------------------------
from torchmetrics import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import DeepImageStructureAndTextureSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore

# Instanzen (global)
_mse    = MeanSquaredError()
_psnr   = PeakSignalNoiseRatio(data_range=1.0)
_ssim   = StructuralSimilarityIndexMeasure(data_range=1.0)
_ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
_lpips  = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
_dists  = DeepImageStructureAndTextureSimilarity()
_fid    = FrechetInceptionDistance(feature=2048, normalize=True)
_kid    = KernelInceptionDistance(subset_size=50)
_iscore = InceptionScore()



###########################################################################
#  Neue generische Tracking-Utilities
###########################################################################
def _track(frames: torch.Tensor):
    tracker = BlobHeadTracker(dE_thr=18, area_min=150, max_shift=120)
    masks, centers = tracker.track(frames)
    angles = np.full(frames.shape[0], np.nan)
    return masks, centers, angles



def _intrusion_depth(frames):
    masks, _, _ = _track(frames)
    W = frames.shape[-1]
    xs = [int(np.where(m)[1].max()) for m in masks if m is not None and m.any()]
    return (max(xs) / (W - 1)) if xs else 0.0


# Optionale neue Metriken – noch nicht im UI verkabelt:
# def _rotation_rms(frames_gt: torch.Tensor, frames_gen: torch.Tensor) -> float:
#     """RMS-Differenz der Kopf-Rotationswinkel (Grad)."""
#     _, _, ang_gt  = _track(frames_gt)
#     _, _, ang_gen = _track(frames_gen)
#     if len(ang_gt) == 0: return float("nan")
#     diff = ang_gt - ang_gen
#     diff = diff[~np.isnan(diff)]
#     return float(np.sqrt((diff ** 2).mean())) if diff.size else float("nan")

# def _accel_mae(frames_gt: torch.Tensor, frames_gen: torch.Tensor) -> float:
#     """MAE der x-Beschleunigung (Pixel/F²) basierend auf Schwerpunkten."""
#     _, c_gt, _  = _track(frames_gt)
#     _, c_gen, _ = _track(frames_gen)
#     # 2. Ableitung (Finite Differences)
#     acc_gt  = np.diff(c_gt[:,0], n=2)
#     acc_gen = np.diff(c_gen[:,0], n=2)
#     n = min(len(acc_gt), len(acc_gen))
#     if n <= 0: return float("nan")
#     return float(np.abs(acc_gt[:n] - acc_gen[:n]).mean())



# Tooltip-Texte ---------------------------------------------------------------
_HELP: Dict[str, str] = {
    "mse"   : "Mittlere quadratische Abweichung: Durchschnitt der quadrierten Pixelfehler zwischen Ground-Truth- und generiertem Frame. 0 bedeutet perfekte Übereinstimmung.",
    "psnr"  : "Peak Signal-to-Noise Ratio in dB - höher ist besser (≥ 30 dB ist gut).",
    "ssim"  : "Structural Similarity Index in [0,1] - höher ist besser.",
    "ms-ssim": "Multi-Scale SSIM in [0,1] - höher ist besser.",
    "lpips" : "Learned Perceptual Image Patch Similarity - kleiner ist besser.",
    "dists" : "Deep Image Structure & Texture Similarity - kleiner ist besser.",
    "fid"   : "Fréchet Inception Distance - kleiner ist besser.",
    "kid"   : "Kernel Inception Distance - kleiner ist besser.",
    "is"    : "Inception Score - höher ist besser.",
    "intrusion": "Differenz der maximalen Intrusion-Tiefe des lila Kopf-Objekts\nzwischen Ground-Truth- und generierter Sequenz (0 = identisch, 1 = maximale Abweichung)."
}

# Zusatz-Konfiguration für die Visualisierung ---------------------------------
_METRIC_CFG = {
    # name        min   max   direction
    "mse"      : (0.0, 0.10, "lower"),
    "psnr"     : (10.0, 40.0, "higher"),
    "ssim"     : (0.0, 1.0,  "higher"),
    "ms-ssim"  : (0.0, 1.0,  "higher"),
    "lpips"    : (0.0, 0.50, "lower"),
    "dists"    : (0.0, 0.50, "lower"),
    "fid"      : (0.0, 100.0,"lower"),
    "kid"      : (0.0, 0.30, "lower"),
    "is"       : (1.0, 9.0,  "higher"),
    "intrusion": (0.0, 1.0,  "lower"),   # 0 = perfekt, 1 = völlig daneben
}

# ----------------------------------------------------------------------
# Haupt­funktion
# ----------------------------------------------------------------------
def compute(frames_gt: torch.Tensor,
            frames_gen: torch.Tensor) -> Dict[str, Tuple[float, str]]:
    """
    Erwartet Tensoren [T,C,H,W] mit Werten in [0,1] und gleicher Größe.
    Berechnet verschiedene TorchMetrics-Metriken und gibt ein Dict[label] → (value, help_text).
    """

    # ---------------- Basismetriken -----------------------------------
    mse_val      = _mse(frames_gt,   frames_gen).item()
    psnr_val     = _psnr(frames_gt,  frames_gen).item()
    ssim_val     = _ssim(frames_gt,  frames_gen).item()
    ms_ssim_val  = _ms_ssim(frames_gt, frames_gen).item()
    lpips_val    = _lpips(frames_gt, frames_gen).item()
    dists_val    = _dists(frames_gt, frames_gen).item()

    # -------------- Verteilungsmetriken (FID / KID / IS) --------------
    real_imgs = frames_gt.reshape(-1, *frames_gt.shape[-3:])
    fake_imgs = frames_gen.reshape(-1, *frames_gen.shape[-3:])
    real_u8   = (real_imgs * 255).round().clamp(0,255).to(torch.uint8)
    fake_u8   = (fake_imgs * 255).round().clamp(0,255).to(torch.uint8)

    _fid.reset();  _fid.update(real_u8, real=True);  _fid.update(fake_u8, real=False)
    fid_val = _fid.compute().item()

    n = real_u8.shape[0]
    if n > 1:
        kid_metric = KernelInceptionDistance(subset_size=min(50, n-1))
        kid_metric.update(real_u8, real=True); kid_metric.update(fake_u8, real=False)
        kid_val, _ = kid_metric.compute(); kid_val = kid_val.item()
    else:
        kid_val = float("nan")

    _iscore.reset(); _iscore.update(fake_u8)
    is_val, _ = _iscore.compute(); is_val = is_val.item()

    # ----------------- NEUE Intrusion-Metrik --------------------------
    depth_gt  = _intrusion_depth(frames_gt)
    depth_gen = _intrusion_depth(frames_gen)
    intrusion_diff = abs(depth_gen - depth_gt)

    # ---------------- Ergebnis-Dictionary ----------------------------
    return {
        "mse"      : (mse_val,      _HELP["mse"]),
        "psnr"     : (psnr_val,     _HELP["psnr"]),
        "ssim"     : (ssim_val,     _HELP["ssim"]),
        "ms-ssim"  : (ms_ssim_val,  _HELP["ms-ssim"]),
        "lpips"    : (lpips_val,    _HELP["lpips"]),
        "dists"    : (dists_val,    _HELP["dists"]),
        "fid"      : (fid_val,      _HELP["fid"]),
        "kid"      : (kid_val,      _HELP["kid"]),
        "is"       : (is_val,       _HELP["is"]),
        "intrusion": (intrusion_diff,_HELP["intrusion"]),
        #"rot_rms"  : (_rotation_rms(frames_gt, frames_gen), "RMS-Winkelabweichung (°)"),
        #"acc_mae"  : (_accel_mae(frames_gt, frames_gen),    "MAE der x-Beschleunigung"),
    }
