import math
from typing import Dict, Any, List


class ClassificationEngine:
    """
    Classifies skin type from Image Vision Observations only.
    Questionnaire argument is accepted for API compatibility but ignored.

    Primary Skin Types: Dry, Normal, Oily, Combination
    Sensitivity: Low, Medium, High

    Weight: 100% Vision (photos only).
    """

    def classify(self, vision_features: Dict[str, float], zone_features: Dict[str, Any], questionnaire: Dict[str, str]) -> Dict[str, Any]:
        # Extract vision metrics (normalized 0.0 to 1.0)
        oiliness = vision_features.get('oiliness', 0.5)
        dryness = vision_features.get('dryness', 0.4)
        redness = vision_features.get('redness', 0.2)
        pores = vision_features.get('visible_pores', 0.3)
        blemishes = vision_features.get('blemishes', 0.15)
        texture = vision_features.get('texture', 0.3)

        # Zone metrics
        forehead_oil = zone_features.get('forehead', {}).get('oiliness', oiliness)
        nose_oil = zone_features.get('nose', {}).get('oiliness', oiliness)
        cheeks_oil = zone_features.get('cheeks', {}).get('oiliness', oiliness)
        cheeks_dry = zone_features.get('cheeks', {}).get('dryness', dryness)

        # Questionnaire signals
        wash_feel = questionnaire.get('wash_feel', 'comfortable')
        midday_shine = questionnaire.get('midday_shine', 'sometimes')
        cheek_dryness = questionnaire.get('cheek_dryness', 'sometimes')
        irritation_freq = questionnaire.get('irritation_frequency', 'sometimes')
        stinging_freq = questionnaire.get('product_stinging', 'never')
        sun_reaction = questionnaire.get('sun_reaction', 'tan_sometimes_burn')

        # ── 1. PHOTO VISION SCORING (100% Weight) ─────────────────────────────
        v_score_combo = 0.0
        v_score_oily  = 0.0
        v_score_dry   = 0.0
        v_score_normal = 0.0

        t_zone_oil = (forehead_oil + nose_oil) / 2.0
        t_zone_diff = t_zone_oil - cheeks_oil

        # Combination: high T-zone vs cheeks
        if t_zone_diff >= 0.20 or (t_zone_oil >= 0.55 and cheeks_dry >= 0.40):
            v_score_combo += 100.0
        elif t_zone_diff >= 0.10:
            v_score_combo += 60.0

        # Oily: high overall oil & cheek oil
        if oiliness >= 0.65 and cheeks_oil >= 0.55:
            v_score_oily += 100.0
        elif oiliness >= 0.55:
            v_score_oily += 60.0

        # Dry: high dryness, low T-zone oil
        if dryness >= 0.60 and t_zone_oil < 0.45:
            v_score_dry += 100.0
        elif dryness >= 0.50:
            v_score_dry += 60.0

        # Normal: balanced oil (0.35 - 0.55), low dryness, low contrast
        if 0.35 <= oiliness <= 0.55 and dryness <= 0.45 and t_zone_diff < 0.15:
            v_score_normal += 100.0
        elif 0.30 <= oiliness <= 0.60:
            v_score_normal += 50.0

        # ── 2. FINAL SCORES (100% Vision) ─────────────────────────────────────
        score_combo  = round(v_score_combo, 1)
        score_oily   = round(v_score_oily, 1)
        score_dry    = round(v_score_dry, 1)
        score_normal = round(v_score_normal, 1)

        primary_scores = {
            'Combination': score_combo,
            'Oily':        score_oily,
            'Dry':         score_dry,
            'Normal':      score_normal
        }

        # Select highest score
        primary_type = max(primary_scores, key=primary_scores.get)

        # Sensitivity scoring (100% Vision Redness)
        vision_sensitivity = 100.0 if redness >= 0.60 else (60.0 if redness >= 0.40 else 20.0)
        sensitivity_score = vision_sensitivity

        if sensitivity_score >= 60.0:
            sensitivity = 'High'
        elif sensitivity_score >= 35.0:
            sensitivity = 'Medium'
        else:
            sensitivity = 'Low'

        # Region breakdowns & explanation logic
        reasons = [
            "Résultat calculé à 100% sur l'analyse visuelle par ordinateur (OpenCV) des 3 photos faciales.",
            f"La zone T (Front & Nez) a présenté {'un sébum et une brillance élevés' if t_zone_oil >= 0.55 else 'un niveau de sébum modéré'}.",
            f"Les joues ont présenté {'une sécheresse relative par rapport à la zone T' if cheeks_dry >= 0.40 else 'une hydratation équilibrée'}.",
        ]
        if sensitivity in ['High', 'Medium']:
            reasons.append(f"Sensibilité évaluée à '{sensitivity}' d'après la rougeur cutanée visuelle détectée sur les photos.")

        return {
            'primary_skin_type': primary_type,
            'sensitivity': sensitivity,
            'scores': primary_scores,
            'sensitivity_score': round(sensitivity_score, 2),
            'visual_summary': {
                'oiliness': self._get_qualitative_rating(oiliness),
                'dryness': self._get_qualitative_rating(dryness),
                'redness': self._get_qualitative_rating(redness),
                'visible_pores': self._get_qualitative_rating(pores),
                'blemishes': self._get_qualitative_rating(blemishes),
                'texture': self._get_qualitative_rating(texture),
            },
            'numeric_metrics': {
                'oiliness': round(oiliness * 100),
                'dryness': round(dryness * 100),
                'redness': round(redness * 100),
                'visible_pores': round(pores * 100),
                'blemishes': round(blemishes * 100),
                'texture': round(texture * 100),
            },
            'reasoning': reasons,
            'zones': {
                'forehead': {'shine': self._get_qualitative_rating(forehead_oil), 'description': 'Forehead zone evaluation'},
                'nose': {'shine': self._get_qualitative_rating(nose_oil), 'description': 'Nose & T-zone pore visibility'},
                'cheeks': {'shine': self._get_qualitative_rating(cheeks_oil), 'description': 'Cheek hydration level'},
            }
        }

    def _get_qualitative_rating(self, val: float) -> str:
        if val < 0.25:
            return 'Low'
        if val < 0.50:
            return 'Low–Medium'
        if val < 0.70:
            return 'Medium–High'
        return 'High'
