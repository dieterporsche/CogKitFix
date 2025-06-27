import cv2
import numpy as np
import torch

def violet_mask(rgb_u8: np.ndarray) -> np.ndarray:
    hsv  = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (120,40,40), (175,255,255))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))


# helper for intrusion
def rightmost_x(mask: np.ndarray) -> int:
    return int(np.where(mask)[1].max()) if mask is not None and mask.any() else -1

class BlobHeadTracker:
    """
    Blob-Tracker mit
    • Overlap-Fallback            (Rotation / große Verschiebungen)
    • Growth- & Shrink-Guard      (Kollisions-Artefakte)
    • Adaptiver Farbmaske        (gegen Helligkeits-/Farb-Drift)
    """
    def __init__(self,
                 dE_thr: float = 18.0,         
                 dE_margin: float = 30.0,      
                 area_min: int = 150,
                 max_shift: int = 220,
                 overlap_radius: int = 12,
                 max_growth: float = 0.15,
                 max_shrink: float = 0.15):
        self.dE_thr         = dE_thr
        self.dE_margin      = dE_margin      
        self.area_min       = area_min
        self.max_shift      = max_shift
        self.overlap_radius = overlap_radius
        self.max_growth     = max_growth
        self.max_shrink     = max_shrink

        # Laufende Referenzen
        self.ref_lab  = None
        self.ref_area = None
        self.cx = self.cy = None
        self.prev_mask = None




    # ---------- Hilfsroutinen ----------------------------------------
    @staticmethod
    def _components(mask):
        num, lbl, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
        comps = [dict(id=i,
                      area=int(stats[i, cv2.CC_STAT_AREA]),
                      cx=float(cent[i][0]),
                      cy=float(cent[i][1]),
                      mask=(lbl == i))
                 for i in range(1, num)]
        return comps

                       # ------------------------------------------------------------------
    #   verarbeitet 1 RGB-Frame (uint8) → (bool-Maske, (cx, cy))
    # ------------------------------------------------------------------
    def update(self, rgb_u8: np.ndarray):
        H, W = rgb_u8.shape[:2]
        lab  = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)

        # -------------------------------------------------------------- #
        # 1)  ADAPTIVE FARB-MASKE
        # -------------------------------------------------------------- #
        if self.ref_lab is None:
            # erster Frame → grober HSV-Threshold reicht
            hsv      = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
            fullmask = cv2.inRange(hsv, (120, 40, 40), (175, 255, 255))
        else:
            # ΔE  - Entfernung jedes Pixels zur Referenzfarbe
            diff     = np.linalg.norm(lab.astype(np.int16) - self.ref_lab, axis=2)
            tol      = self.dE_thr + self.dE_margin       # etwas großzügiger
            fullmask = (diff < tol).astype(np.uint8) * 255

        fullmask = cv2.morphologyEx(fullmask, cv2.MORPH_OPEN,
                                    np.ones((3, 3), np.uint8))

        # -------------------------------------------------------------- #
        # 2)  OVERLAP-CLIP  (falls wir bereits eine Maske haben)
        # -------------------------------------------------------------- #
        if self.prev_mask is not None and self.overlap_radius > 0:
            k      = 2 * self.overlap_radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            near   = cv2.dilate(self.prev_mask.astype(np.uint8), kernel)
            clipmask = cv2.bitwise_and(fullmask, near)
        else:
            clipmask = fullmask

        # Hilfs-Funktion – CC-Liste aus Maske holen
        def comps_from(mask):
            return [c for c in self._components(mask)
                    if c["area"] >= self.area_min]

        comps = comps_from(clipmask) or comps_from(fullmask)
        if not comps:                       # kein Blob → behalte alte Maske
            return self.prev_mask, (self.cx, self.cy)

        # -------------------------------------------------------------- #
        # 3)  ERSTER FRAME: linkester Blob
        # -------------------------------------------------------------- #
        if self.ref_lab is None:
            best          = min(comps, key=lambda c: c["cx"])
            self.ref_lab  = lab[best["mask"]].mean(0)
            self.ref_area = best["area"]

        # -------------------------------------------------------------- #
        # 4)  WEITERE FRAMES
        # -------------------------------------------------------------- #
        else:
            prev_area = self.prev_mask.sum() if self.prev_mask is not None else self.ref_area
            upper     = prev_area * (1 + self.max_growth)
            lower     = prev_area * (1 - self.max_shrink)

            comps = [c for c in comps if lower <= c["area"] <= upper]
            if not comps:
                return self.prev_mask, (self.cx, self.cy)

            def iou(c_mask):
                inter = np.logical_and(c_mask, self.prev_mask).sum()
                union = np.logical_or(c_mask,  self.prev_mask).sum()
                return inter / union if union else 0.0

            scored = []
            for c in comps:
                if abs(c["cx"] - self.cx) > self.max_shift:
                    continue
                scored.append((
                    -iou(c["mask"]),
                    abs(c["area"] - self.ref_area) / self.ref_area,
                    np.linalg.norm(lab[c["mask"]] - self.ref_lab, axis=1).mean(),
                    np.hypot(c["cx"] - self.cx, c["cy"] - self.cy),
                    c,
                ))

            if not scored:
                return self.prev_mask, (self.cx, self.cy)

            best = min(scored)[4]

            # Referenzen sanft nachführen
            self.ref_lab  = 0.9 * self.ref_lab  + 0.1 * lab[best["mask"]].mean(0)
            self.ref_area = 0.9 * self.ref_area + 0.1 * best["area"]

        # -------------------------------------------------------------- #
        # 5)  FINALE MASKE + MITTELPUNKT
        # -------------------------------------------------------------- #
        m = np.zeros((H, W), bool)
        m[best["mask"]] = True
        self.cx, self.cy = int(best["cx"]), int(best["cy"])
        self.prev_mask   = m
        return m, (self.cx, self.cy)





    # ---------- Sequenz-Wrapper -------------------------------------
    def track(self, frames: torch.Tensor):
        masks, centers = [], []
        for t in range(frames.shape[0]):
            rgb = (frames[t].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            m, (cx, cy) = self.update(rgb)
            masks.append(m)
            centers.append((cx, cy))
        return masks, np.array(centers)


#HeadTracker      = HeadTracker
rightmost_x      = rightmost_x