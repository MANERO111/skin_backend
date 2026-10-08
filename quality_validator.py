import cv2
import numpy as np
import io
from PIL import Image


def _to_py_type(val):
    """Converts numpy scalars/booleans into native Python types for JSON serialization."""
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, dict):
        return {k: _to_py_type(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_to_py_type(i) for i in val]
    return val


class ImageQualityValidator:
    """
    Validates image quality before skin analysis.
    Checks: sharpness, brightness, face presence, face distance, and face pose (front vs profile left/right).
    """

    # Try to load frontal and profile face cascades at import time
    try:
        _frontal_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
        
        _frontal_cascade = cv2.CascadeClassifier(_frontal_path)
        _profile_cascade = cv2.CascadeClassifier(_profile_path)
        
        _frontal_available = bool(_frontal_cascade and not _frontal_cascade.empty())
        _profile_available = bool(_profile_cascade and not _profile_cascade.empty())
    except Exception:
        _frontal_cascade = None
        _profile_cascade = None
        _frontal_available = False
        _profile_available = False

    def validate_image_bytes(self, image_bytes: bytes, pose_type: str = "front") -> dict:
        # ── Parse image ──────────────────────────────────────────────────────
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            return _to_py_type({
                "usable": False, "valid": False, "score": 0.0,
                "reason": f"Impossible de lire l'image : {str(e)}",
                "issues": [f"Impossible de lire l'image : {str(e)}"],
                "metrics": {"sharpness_val": 0, "brightness_val": 0, "face_coverage_pct": 0},
                "checks": {
                    "readable": False, "sharpness": False,
                    "lighting": False, "face_visible": False,
                    "single_face": False, "face_centered": False,
                    "pose_correct": False, "distance_good": False
                }
            })

        height, width, _ = cv_img.shape
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # ── 1. Sharpness (Laplacian variance) ────────────────────────────────
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_sharp = bool(laplacian_var > 28.0)  # >28 = acceptable for webcam/phone

        # ── 2. Brightness ─────────────────────────────────────────────────────
        avg_brightness = float(np.mean(gray))
        is_lighting_good = bool(35.0 <= avg_brightness <= 235.0)

        # ── 3. Face detection & Pose validation ──────────────────────────────
        face_detected = False
        single_face = True
        face_area_ratio = 0.0
        distance_good = False
        face_centered = False
        pose_correct = True
        
        frontal_faces = []
        profile_faces = []

        if self._frontal_available and self._frontal_cascade is not None:
            try:
                frontal_faces = self._frontal_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
            except Exception:
                pass

        if self._profile_available and self._profile_cascade is not None:
            try:
                # Profile cascade facing left/right (flip gray to detect both profile directions)
                p1 = self._profile_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
                gray_flipped = cv2.flip(gray, 1)
                p2 = self._profile_cascade.detectMultiScale(
                    gray_flipped, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
                
                profile_faces = list(p1) + list(p2)
            except Exception:
                pass

        all_detected = list(frontal_faces) + list(profile_faces)
        face_detected = bool(len(all_detected) > 0)
        single_face = bool(len(frontal_faces) <= 1 and len(profile_faces) <= 1)

        if face_detected:
            # Pick best box
            best_box = frontal_faces[0] if len(frontal_faces) > 0 else profile_faces[0]
            (x, y, w, h) = best_box
            face_area_ratio = float((w * h) / (width * height))
            
            # Face should be nicely filling frame (between 4.5% and 55% of full frame)
            distance_good = bool(0.045 <= face_area_ratio <= 0.55)
            
            cx = x + w / 2
            offset = abs(cx - width / 2) / width
            face_centered = bool(offset < 0.45)

            # Check Pose
            if pose_type == "front":
                # Front step requires frontal face cascade match or centered face
                if len(frontal_faces) == 0 and len(profile_faces) > 0:
                    pose_correct = False
            elif pose_type in ("left", "right"):
                # Profile step requires turning face (profile match OR off-center / asymmetric box)
                # If only frontal face is found and it's perfectly centered, user didn't turn head!
                if len(frontal_faces) > 0 and len(profile_faces) == 0:
                    # If user is strictly facing front during left/right step
                    (fx, fy, fw, fh) = frontal_faces[0]
                    fcx = fx + fw / 2
                    if abs(fcx - width / 2) / width < 0.15:
                        pose_correct = False
        else:
            # Fallback if cascades fail completely: assume visible if lighting & sharp
            if not self._frontal_available:
                face_detected = True
                single_face = True
                distance_good = True
                face_centered = True
                pose_correct = True
                face_area_ratio = 0.15

        # ── 4. Quality score ──────────────────────────────────────────────────
        quality_score = float(min(1.0, max(0.1, (
            (0.25 if is_sharp else 0.05) +
            (0.25 if is_lighting_good else 0.05) +
            (0.25 if face_detected else 0.05) +
            (0.15 if distance_good else 0.00) +
            (0.10 if pose_correct else 0.00)
        ))))

        # ── 5. Collect failure reasons ────────────────────────────────────────
        reasons = []
        if not is_lighting_good:
            if avg_brightness < 35:
                reasons.append("L'image est trop sombre. Éclairez mieux votre visage.")
            else:
                reasons.append("L'image est surexposée / trop lumineuse.")
        if not is_sharp:
            reasons.append("L'image est floue. Assurez-vous d'être bien immobile.")
        if self._frontal_available or self._profile_available:
            if not face_detected:
                reasons.append("Aucun visage clairement détecté.")
            elif not pose_correct:
                if pose_type == "front":
                    reasons.append("Regardez bien en face de la caméra.")
                elif pose_type == "left":
                    reasons.append("Tournez votre visage vers la gauche (~45°).")
                elif pose_type == "right":
                    reasons.append("Tournez votre visage vers la droite (~45°).")
            elif not distance_good:
                if face_area_ratio < 0.045:
                    reasons.append("Rapprochez-vous de la caméra (visage trop éloigné).")
                else:
                    reasons.append("Reculez légèrement de la caméra (visage trop près).")
            elif not single_face:
                reasons.append("Plusieurs visages détectés. Restez seul(e) à l'écran.")

        # ── 6. Usability decision ─────────────────────────────────────────────
        if self._frontal_available or self._profile_available:
            is_usable = bool(is_sharp and is_lighting_good and face_detected and distance_good and pose_correct)
        else:
            is_usable = bool(is_sharp and is_lighting_good)

        res = {
            "usable": is_usable,
            "valid": bool(len(reasons) == 0),
            "score": round(quality_score, 2),
            "pose_type": pose_type,
            "reason": reasons[0] if reasons else "Qualité d'image excellente.",
            "issues": reasons,
            "metrics": {
                "sharpness_val": round(laplacian_var, 1),
                "brightness_val": round(avg_brightness, 1),
                "face_coverage_pct": round(face_area_ratio * 100, 1),
            },
            "checks": {
                "readable": True,
                "sharpness": is_sharp,
                "lighting": is_lighting_good,
                "face_visible": face_detected,
                "single_face": single_face,
                "face_centered": face_centered,
                "pose_correct": pose_correct,
                "distance_good": distance_good
            }
        }

        return _to_py_type(res)
