import cv2
import numpy as np
import io
from PIL import Image
from quality_validator import ImageQualityValidator


class SkinVisionEngine:
    """
    Real Computer Vision engine that extracts facial skin metrics from photos.

    HOW THE RESULTS ARE COMPUTED
    ──────────────────────────────
    Each photo goes through these steps:

    1. QUALITY GATE  – ImageQualityValidator checks sharpness, lighting, and
       whether a face is detectable. Photos that fail are skipped.

    2. COLOR-SPACE CONVERSION
       • BGR  → LAB  (isolates color channel from luminance)
       • BGR  → HSV  (isolates Value/brightness channel)
       • BGR  → GRAY (single-channel for edge / texture work)

    3. GLOBAL FEATURE EXTRACTION (whole face):
       oiliness   = fraction of pixels with V > 215 (specular highlight pixels)
                    multiplied by a calibration factor → 0-1 range.
       redness    = mean of the LAB a* channel (positive = reddish) → normalised.
       texture    = std-dev of gray pixel intensities / 60.
       dryness    = inverse of oiliness, boosted by rough texture.
       pores      = density of high-frequency edges (Laplacian residual).
       blemishes  = fraction of dark-spot pixels found by adaptive threshold.

    4. ZONE-LEVEL ANALYSIS – the image is divided into 4 ROIs:
       Forehead   top  15-35 %  × centre 25-75 %
       Nose       mid  35-65 %  × centre 35-65 %
       Cheeks     mid  35-70 %  × full  10-90 %
       Chin       low  70-90 %  × centre 30-70 %
       Each ROI computes its own oiliness / dryness / redness / pore score.

    5. MULTI-PHOTO AVERAGING – if front + left + right are all usable, their
       feature vectors are averaged (weighted: front x2, profiles x1 each).

    6. The resulting dict is returned to Laravel which feeds it into
       ClassificationEngine.php (score-based skin type determination).
    """

    # Calibration constants (tuned for typical smartphone / webcam captures)
    OIL_SPECULAR_THRESHOLD = 210   # V-channel pixel value considered "shiny"
    OIL_SCALE              = 14.0  # Multiplier to map highlight fraction → 0-1
    OIL_BASE               = 0.18  # Minimum baseline (even dry skin has some)
    PORE_EDGE_THRESHOLD    = 15    # High-freq residual threshold for pore detection
    PORE_SCALE             = 9.0
    PORE_BASE              = 0.12
    BLEMISH_BLOCK_SIZE     = 13    # Adaptive threshold block size for spot detection
    BLEMISH_SCALE          = 4.0

    def __init__(self):
        self.validator = ImageQualityValidator()

    # ─────────────────────────────────────────────────────────────────────────
    def process_photos(self, photos_dict: dict) -> dict:
        """
        Entry point. photos_dict = {"front": bytes, "left": bytes, "right": bytes}
        Returns a structured dict with features, zones, and quality_reports.
        """
        all_results: list = []     # (weight, feature_dict, zone_dict) per usable photo
        quality_reports: dict = {}
        skipped_reasons: dict = {}

        # Process each photo
        for pose, image_bytes in photos_dict.items():
            if not image_bytes:
                continue

            # Quality gate
            q_res = self.validator.validate_image_bytes(image_bytes, pose_type=pose)
            quality_reports[pose] = q_res

            if not q_res["usable"]:
                reason = q_res.get("reason", "Quality check failed")
                skipped_reasons[pose] = reason
                print(f"[Vision] Skipping {pose}: {reason}")
                continue

            try:
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                # Normalize size: resize to a standard 640×480 for consistent ROI math
                pil_img = pil_img.resize((640, 480), Image.LANCZOS)
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                feat, zones = self._analyze_single_image(cv_img, pose)
                # Front photo is twice as important as profiles
                weight = 2.0 if pose == "front" else 1.0
                all_results.append((weight, feat, zones))
                print(f"[Vision] {pose} → oil={feat['oiliness']:.2f} dry={feat['dryness']:.2f} "
                      f"red={feat['redness']:.2f} pores={feat['visible_pores']:.2f} "
                      f"blm={feat['blemishes']:.2f} tex={feat['texture']:.2f}")
            except Exception as e:
                print(f"[Vision] Error analyzing {pose}: {e}")
                import traceback
                traceback.print_exc()

        # ── No usable photos ─────────────────────────────────────────────────
        if not all_results:
            print("[Vision] No usable photos. Returning partial status.")
            return {
                "status": "partial",
                "error": "No usable photos were processed. " + str(skipped_reasons),
                "quality_reports": quality_reports,
                "features": None,
                "zones": None
            }

        # ── Weighted average across photos ────────────────────────────────────
        total_weight = sum(w for w, _, _ in all_results)

        def wavg(key: str) -> float:
            return round(sum(w * f[key] for w, f, _ in all_results) / total_weight, 3)

        avg_features = {
            "oiliness":      wavg("oiliness"),
            "dryness":       wavg("dryness"),
            "redness":       wavg("redness"),
            "visible_pores": wavg("visible_pores"),
            "blemishes":     wavg("blemishes"),
            "texture":       wavg("texture"),
        }

        # ── Zone averages ────────────────────────────────────────────────────
        zone_names = ["forehead", "nose", "cheeks", "chin"]
        zone_metrics = ["oiliness", "dryness", "redness", "pores"]
        final_zones: dict = {}
        for z in zone_names:
            vals = {m: [] for m in zone_metrics}
            for w, _, zones in all_results:
                if z in zones:
                    for m in zone_metrics:
                        vals[m].append(zones[z].get(m, 0.0))
            final_zones[z] = {
                m: round(float(np.mean(v)) if v else 0.0, 3)
                for m, v in vals.items()
            }

        print(f"[Vision] Final features: {avg_features}")
        return {
            "status": "success",
            "quality_reports": quality_reports,
            "face": {"detected": True, "usable": True},
            "photos_used": len(all_results),
            "features": avg_features,
            "zones": final_zones,
        }

    # ─────────────────────────────────────────────────────────────────────────
    def _analyze_single_image(self, img_bgr: np.ndarray, pose: str) -> tuple:
        """
        Analyze one image. Returns (features_dict, zones_dict).

        Feature computation explained:
        ─────────────────────────────
        oiliness:      High-intensity specular pixels (V > 210 in HSV).
                       Sebum reflects light brightly. More bright pixels = oilier.
        redness:       LAB a* channel mean. In LAB space, a* > 128 means reddish.
                       Normalised and clipped to 0.05-0.90.
        texture:       Std-dev of gray pixel values divided by 60.
                       High variance = rough / uneven texture.
        dryness:       Inverse heuristic: low oiliness + high texture → dry.
        visible_pores: Density of edge pixels using Laplacian residual method.
                       High-freq edges correspond to pore openings & texture.
        blemishes:     Adaptive threshold detects localized dark spots vs.
                       surrounding skin (pimples, post-inflammatory marks).
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

        # ── 1. Redness via LAB a* channel ────────────────────────────────────
        _, a_chan, _ = cv2.split(lab)
        # a* mean is ~128 for neutral, >128 = reddish
        redness_score = float(np.clip((np.mean(a_chan) - 118.0) / 30.0, 0.05, 0.90))

        # ── 2. Oiliness via specular highlights (V channel) ──────────────────
        v_chan = hsv[:, :, 2]
        specular_pixels = np.sum(v_chan > self.OIL_SPECULAR_THRESHOLD)
        specular_ratio  = specular_pixels / (w * h)
        oiliness_score  = float(np.clip(
            specular_ratio * self.OIL_SCALE + self.OIL_BASE, 0.10, 0.95))

        # ── 3. Texture (std-dev of pixel intensity) ──────────────────────────
        texture_score = float(np.clip(np.std(gray) / 60.0, 0.10, 0.90))

        # ── 4. Dryness heuristic ─────────────────────────────────────────────
        dryness_score = float(np.clip(
            (1.0 - oiliness_score) * 0.65 + texture_score * 0.35, 0.10, 0.90))

        # ── 5. Pore density (Laplacian high-freq residual) ───────────────────
        blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = cv2.absdiff(gray, blurred)
        pore_density = np.sum(residual > self.PORE_EDGE_THRESHOLD) / (w * h)
        pore_score   = float(np.clip(
            pore_density * self.PORE_SCALE + self.PORE_BASE, 0.10, 0.90))

        # ── 6. Blemish detection (adaptive threshold dark spots) ─────────────
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.BLEMISH_BLOCK_SIZE, 3)
        # Morphological open to remove noise, keep real spots
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        blemish_ratio = np.sum(thresh > 0) / (w * h)
        blemish_score = float(np.clip(blemish_ratio * self.BLEMISH_SCALE, 0.05, 0.85))

        # ── Zone ROI analysis ─────────────────────────────────────────────────
        # ROI percentages designed for a 640x480 standardized image.
        # For profile photos (left/right), use the visible half.
        if pose == "left":
            x_start, x_end = 0.50, 1.00   # right half of frame is the face
        elif pose == "right":
            x_start, x_end = 0.00, 0.50   # left half of frame is the face
        else:
            x_start, x_end = 0.00, 1.00   # full width for front

        xs = int(w * x_start)
        xe = int(w * x_end)

        rois = {
            "forehead": img_bgr[int(h * 0.10):int(h * 0.32), xs:xe],
            "nose":     img_bgr[int(h * 0.35):int(h * 0.62), int(w * 0.30):int(w * 0.70)],
            "cheeks":   img_bgr[int(h * 0.35):int(h * 0.70), xs:xe],
            "chin":     img_bgr[int(h * 0.68):int(h * 0.88), int(w * 0.25):int(w * 0.75)],
        }

        zones: dict = {}
        for zone_name, crop in rois.items():
            zones[zone_name] = self._analyze_roi(
                crop, oiliness_score, dryness_score, redness_score, pore_score)

        features = {
            "oiliness":      round(oiliness_score, 3),
            "dryness":       round(dryness_score, 3),
            "redness":       round(redness_score, 3),
            "visible_pores": round(pore_score, 3),
            "blemishes":     round(blemish_score, 3),
            "texture":       round(texture_score, 3),
        }
        return features, zones

    # ─────────────────────────────────────────────────────────────────────────
    def _analyze_roi(self, crop: np.ndarray,
                     base_oil: float, base_dry: float,
                     base_red: float, base_pore: float) -> dict:
        """Compute skin metrics for a single facial zone crop."""
        if crop is None or crop.size == 0:
            return {
                "oiliness": round(base_oil, 3),
                "dryness":  round(base_dry, 3),
                "redness":  round(base_red, 3),
                "pores":    round(base_pore, 3),
            }

        gray_c = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv_c  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab_c  = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

        v_c      = hsv_c[:, :, 2]
        _, a_c, _ = cv2.split(lab_c)
        total    = max(1, gray_c.size)

        oil  = float(np.clip(
            (np.sum(v_c > self.OIL_SPECULAR_THRESHOLD) / total) * self.OIL_SCALE + base_oil * 0.35,
            0.10, 0.95))
        red  = float(np.clip((np.mean(a_c) - 118.0) / 30.0, 0.05, 0.90))
        tex  = float(np.clip(np.std(gray_c) / 60.0, 0.10, 0.90))
        dry  = float(np.clip((1.0 - oil) * 0.65 + tex * 0.35, 0.10, 0.90))

        blur_c  = cv2.GaussianBlur(gray_c, (5, 5), 0)
        res_c   = cv2.absdiff(gray_c, blur_c)
        pore = float(np.clip(
            (np.sum(res_c > self.PORE_EDGE_THRESHOLD) / total) * self.PORE_SCALE + self.PORE_BASE,
            0.10, 0.90))

        return {
            "oiliness": round(oil, 3),
            "dryness":  round(dry, 3),
            "redness":  round(red, 3),
            "pores":    round(pore, 3),
        }
