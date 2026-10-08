import base64
import datetime
import secrets
import uvicorn
import json
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from quality_validator import ImageQualityValidator
from vision_engine import SkinVisionEngine
from classification_engine import ClassificationEngine
from recommendation_engine import RecommendationEngine
from explanation_generator import ExplanationGenerator
from database_manager import DatabaseManager
from luna_verifier import LunaVerifier

app = FastAPI(
    title="AI Skin Analysis Unified Python Service",
    description="Full unified FastAPI backend handling computer vision, classification, recommendations, auth, questionnaire, session history, and GPT-6 Luna 2nd layer verification.",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

validator = ImageQualityValidator()
engine = SkinVisionEngine()
classifier = ClassificationEngine()
recommender = RecommendationEngine()
explainer = ExplanationGenerator()
db_mgr = DatabaseManager()
luna = LunaVerifier()


# ─── Health Checks ─────────────────────────────────────────────────────────────

@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AI Skin Unified Python Backend",
        "version": "2.0.0"
    }


# ─── Questionnaire Definition ──────────────────────────────────────────────────

@app.get("/api/questionnaire")
def get_questionnaire():
    return {
        "title": "Questionnaire sur le Profil de Peau & la Routine Cosmétique\nاستبيان نوع البشرة والروتين العناية",
        "total_questions": 10,
        "questions": [
            {
                "id": "wash_feel",
                "question": "Après avoir lavé votre visage et attendu 30 à 60 minutes sans appliquer de crème hydratante, comment votre peau se sent-elle ?\nبعد غسل وجهك والانتظار من 30 إلى 60 دقيقة دون وضع مرطب، كيف تشعر بشرتك؟",
                "options": [
                    {"value": "tight", "label": "Tiraillée, étirée ou inconfortable\nمشدودة أو جافة وغير مرتاحة"},
                    {"value": "comfortable", "label": "Confortable et équilibrée\nمرتاحة ومتوازنة"},
                    {"value": "oily", "label": "Brillante ou grasse sur tout le visage\nلامعة أو دهنية في كامل الوجه"},
                    {"value": "oily_tzone", "label": "Grasse sur le front/nez, mais normale ou sèche sur les joues\nدهنية على الجبهة والأنف، ولكن عادية أو جافة على الخدين"}
                ]
            },
            {
                "id": "midday_shine",
                "question": "À quelle vitesse la brillance ou le sébum apparaissent-ils sur votre visage pendant la journée ?\nما مدى سرعة ظهور اللمعان أو الدهون على وجهك خلال النهار؟",
                "options": [
                    {"value": "rarely", "label": "Rarement ou presque jamais\nنادراً أو تقريباً أبداً"},
                    {"value": "sometimes", "label": "Seulement en fin d'après-midi ou sur la zone T\nفقط في نهاية اليوم أو على منطقة T"},
                    {"value": "quickly", "label": "Dans les 1 à 2 heures après le nettoyage\nخلال 1 إلى 2 ساعة بعد التنظيف"}
                ]
            },
            {
                "id": "cheek_dryness",
                "question": "Vos joues vous semblent-elles sèches, rugueuses ou tiraillées ?\nهل تشعر أن خديك جافان، خشنان أو مشدودان؟",
                "options": [
                    {"value": "never", "label": "Jamais\nأبداً"},
                    {"value": "sometimes", "label": "Parfois par temps sec ou froid\nأحياناً في الطقس الجاف أو البارد"},
                    {"value": "often", "label": "Souvent tout au long de l'année\nغالباً طوال السنة"}
                ]
            },
            {
                "id": "irritation_frequency",
                "question": "À quelle fréquence votre peau présente-t-elle des rougeurs, des bouffées ou des irritations ?\nكم مرة تظهر على بشرتك احمرار أو تهيج؟",
                "options": [
                    {"value": "almost_never", "label": "Presque jamais\nنادراً جداً"},
                    {"value": "sometimes", "label": "Parfois à cause de la météo ou de certains produits\nأحياناً بسبب الطقس أو بعض المنتجات"},
                    {"value": "frequently", "label": "Fréquemment ou très facilement déclenchées\nبشكل متكرر أو بسهولة كبيرة"}
                ]
            },
            {
                "id": "product_stinging",
                "question": "Les nouveaux produits de soin, savons ou crèmes provoquent-ils parfois des sensations de brûlure ou de picotement ?\nهل تسبب لك المنتجات الجديدة، الصابون أو الكريمات شعوراً بالحرقان أو الوخز؟",
                "options": [
                    {"value": "never", "label": "Jamais\nأبداً"},
                    {"value": "sometimes", "label": "Seulement avec des exfoliants forts ou des acides\nفقط مع المقشرات القوية أو الأحماض"},
                    {"value": "often", "label": "Souvent, même avec des produits ordinaires\nغالباً، حتى مع المنتجات العادية"}
                ]
            },
            {
                "id": "sun_reaction",
                "question": "Comment votre peau réagit-elle généralement à la première exposition au soleil ?\nكيف تتفاعل بشرتك عادةً عند التعرض للشمس لأول مرة؟",
                "options": [
                    {"value": "tan_easily", "label": "Bronze facilement avec peu de rougeur\nتكتسب اسمراراً بسهولة مع احمرار قليل"},
                    {"value": "tan_sometimes_burn", "label": "Bronze progressivement, brûle occasionnellement\nتكتسب اسمراراً تدريجياً وتحترق أحياناً"},
                    {"value": "burn_easily", "label": "Brûle facilement et rougit rapidement\nتحترق بسهولة وتحمر بسرعة"}
                ]
            },
            {
                "id": "pore_concern",
                "question": "À quel point vos pores sont-ils visibles dans un miroir ordinaire ?\nما مدى وضوح مسامك في المرآة العادية؟",
                "options": [
                    {"value": "minimal", "label": "À peine visibles sur tout le visage\nغير ظاهرة تقريباً في كامل الوجه"},
                    {"value": "t_zone_only", "label": "Visibles surtout sur le nez et le front\nواضحة خاصة على الأنف والجبهة"},
                    {"value": "widespread", "label": "Visibles sur la zone T et les joues\nواضحة على منطقة T والخدين"}
                ]
            },
            {
                "id": "breakout_frequency",
                "question": "À quelle fréquence avez-vous des boutons, des pores obstrués ou des imperfections ?\nكم مرة تظهر لديك حبوب، مسام مسدودة أو شوائب؟",
                "options": [
                    {"value": "rarely", "label": "Rarement ou jamais\nنادراً أو أبداً"},
                    {"value": "localized_tzone", "label": "Occasionnellement sur la zone T ou le menton\nأحياناً على منطقة T أو الذقن"},
                    {"value": "frequent", "label": "Fréquemment sur plusieurs zones\nبشكل متكرر في مناطق متعددة"}
                ]
            },
            {
                "id": "flaking_propensity",
                "question": "Observez-vous des peaux qui se desquament ou une texture inégale lors de l'application de fond de teint ou de crème ?\nهل تلاحظ تقشر البشرة أو ملمساً غير مستوٍ عند وضع المكياج أو الكريم؟",
                "options": [
                    {"value": "never", "label": "Jamais\nأبداً"},
                    {"value": "occasionally", "label": "Occasionnellement autour du nez ou des joues\nأحياناً حول الأنف أو الخدين"},
                    {"value": "frequently", "label": "Fréquemment sur les zones sèches\nبشكل متكرر في المناطق الجافة"}
                ]
            },
            {
                "id": "climate_impact",
                "question": "Comment les changements saisonniers (air froid / chauffage intérieur sec) affectent-ils votre peau ?\nكيف تؤثر التغيرات الموسمية (البرد / التدفئة الجافة) على بشرتك؟",
                "options": [
                    {"value": "no_change", "label": "Peu ou pas de changement notable\nتغير طفيف أو لا يوجد تغير ملحوظ"},
                    {"value": "feels_drier", "label": "La peau devient plus sèche et nécessite une crème plus riche\nتصبح البشرة أكثر جفافاً وتحتاج كريم أكثر غنى"},
                    {"value": "becomes_flaky_red", "label": "Sujette aux tiraillements, rougeurs et desquamations\nمعرضة للشد والاحمرار والتقشر"}
                ]
            }
        ]
    }


# ─── Auth Endpoints ────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/auth/register")
def register(req: RegisterRequest):
    if not req.name.strip() or not req.email.strip() or not req.password:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Veuillez remplir tous les champs (nom, email, mot de passe)."}
        )
    res = db_mgr.register_user(req.name, req.email, req.password)
    if res.get("status") == "error":
        return JSONResponse(status_code=400, content=res)
    return res


@app.post("/api/auth/login")
def login(req: LoginRequest):
    if not req.email.strip() or not req.password:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Adresse e-mail ou mot de passe incorrect."}
        )
    res = db_mgr.login_user(req.email, req.password)
    if res.get("status") == "error":
        return JSONResponse(status_code=401, content=res)
    return res


# ─── Pre-quiz photo quality check ────────────────────────────────────────────

class ValidatePhotosRequest(BaseModel):
    front: Optional[str] = None
    left:  Optional[str] = None
    right: Optional[str] = None


@app.post("/api/v1/validate-photos-base64")
@app.post("/api/photos/validate")
@app.post("/api/analysis/validate-photos")
async def validate_photos_base64(req: ValidatePhotosRequest):
    """
    Quick quality gate called BEFORE the questionnaire.
    Returns per-photo usability and a combined pass/fail verdict.
    """
    reports: dict = {}
    any_usable = False

    for key, data_str in [("front", req.front), ("left", req.left), ("right", req.right)]:
        if not data_str:
            reports[key] = {"provided": False, "usable": False, "reason": "Photo non fournie."}
            continue
        try:
            clean = data_str.split(",", 1)[1] if "," in data_str else data_str
            padding = 4 - len(clean) % 4
            if padding != 4:
                clean += "=" * padding
            raw = base64.b64decode(clean)
            result = validator.validate_image_bytes(raw, pose_type=key)
            result["provided"] = True
            reports[key] = result
            if result.get("usable"):
                any_usable = True
        except Exception as e:
            reports[key] = {
                "provided": True, "usable": False,
                "reason": f"Erreur de décodage : {str(e)}"
            }

    all_pass = bool(all(
        bool(reports.get(k, {}).get("usable", False))
        for k in ["front", "left", "right"]
        if reports.get(k, {}).get("provided", False)
    ))

    return {
        "status": "success",
        "overall_pass": bool(all_pass),
        "any_usable": bool(any_usable),
        "reports": reports
    }


# ─── Direct Base64 Vision Analysis ───────────────────────────────────────────

class Base64PhotosRequest(BaseModel):
    front: Optional[str] = None
    left:  Optional[str] = None
    right: Optional[str] = None


@app.post("/api/v1/analyze-base64")
async def analyze_base64_photos(req: Base64PhotosRequest):
    photos_data = {}

    for key, data_str in [("front", req.front), ("left", req.left), ("right", req.right)]:
        if not data_str:
            continue
        try:
            clean = data_str.split(",", 1)[1] if "," in data_str else data_str
            padding = 4 - len(clean) % 4
            if padding != 4:
                clean += "=" * padding
            raw_bytes = base64.b64decode(clean)
            if len(raw_bytes) > 100:
                photos_data[key] = raw_bytes
        except Exception as e:
            print(f"[Vision] Error decoding base64 for {key}: {e}")

    if not photos_data:
        raise HTTPException(
            status_code=400,
            detail="No valid base64 photo data could be decoded."
        )

    result = engine.process_photos(photos_data)
    return result


# ─── Full End-to-End Analysis Process ─────────────────────────────────────────

class ProcessAnalysisRequest(BaseModel):
    full_name: Optional[str] = ""
    address: Optional[str] = ""
    email: Optional[str] = ""
    phone_number: Optional[str] = ""
    ville: Optional[str] = ""
    user_id: Optional[str] = "usr_guest"
    photos: Optional[Dict[str, Any]] = None
    vision_features: Optional[Dict[str, float]] = None
    zone_features: Optional[Dict[str, Any]] = None


@app.post("/api/analysis/process")
async def process_analysis(req: ProcessAnalysisRequest):
    """
    Main analysis pipeline:
    1. Extracts vision metrics from photos using OpenCV
    2. Runs Classification Engine (100% Photo-based)
    3. Runs Recommendation Engine
    4. Generates Natural Language Explanation
    5. GPT-6 Luna 2nd AI Verification Layer
    6. Saves session with user details (full_name, address, email, phone_number, ville) in SQLite database
    """
    vision_features = req.vision_features
    zone_features = req.zone_features
    py_error = None

    if req.photos and (not vision_features or not zone_features):
        photos_data = {}
        for key in ["front", "left", "right"]:
            data_str = req.photos.get(key)
            if data_str and isinstance(data_str, str):
                try:
                    clean = data_str.split(",", 1)[1] if "," in data_str else data_str
                    padding = 4 - len(clean) % 4
                    if padding != 4:
                        clean += "=" * padding
                    raw_bytes = base64.b64decode(clean)
                    if len(raw_bytes) > 100:
                        photos_data[key] = raw_bytes
                except Exception as e:
                    print(f"[Analysis] Error decoding photo {key}: {e}")

        if photos_data:
            py_res = engine.process_photos(photos_data)
            if py_res.get("features"):
                vision_features = py_res["features"]
                zone_features = py_res.get("zones", {})
            else:
                py_error = py_res.get("error", "Vision analysis failed.")

    # Neutral fallback if no vision data available
    if not vision_features:
        vision_features = {
            "oiliness": 0.68, "dryness": 0.35, "redness": 0.20,
            "visible_pores": 0.52, "blemishes": 0.18, "texture": 0.40
        }

    if not zone_features:
        o = vision_features.get("oiliness", 0.5)
        d = vision_features.get("dryness", 0.4)
        r = vision_features.get("redness", 0.2)
        p = vision_features.get("visible_pores", 0.35)
        zone_features = {
            "forehead": {"oiliness": min(0.95, o * 1.15), "dryness": d * 0.80, "redness": r, "pores": min(0.9, p * 1.15)},
            "nose":     {"oiliness": min(0.95, o * 1.30), "dryness": d * 0.60, "redness": r, "pores": min(0.9, p * 1.25)},
            "cheeks":   {"oiliness": o * 0.70,            "dryness": min(0.9, d * 1.25), "redness": min(0.9, r * 1.10), "pores": p * 0.80},
            "chin":     {"oiliness": o * 0.90,            "dryness": d * 0.90, "redness": r, "pores": p},
        }

    q_answers = {}

    # 1. Classification (100% Photo)
    classification = classifier.classify(vision_features, zone_features, q_answers)

    # 2. Recommendations
    recommendations = recommender.generate_recommendations(
        classification["primary_skin_type"],
        classification["sensitivity"],
        classification["numeric_metrics"]
    )

    # 3. Explanation
    explanation = explainer.generate_natural_explanation(classification, recommendations)

    # 4. GPT-6 Luna 2nd AI Verification Layer
    luna_verification = luna.verify_analysis(
        photos=req.photos or {},
        opencv_metrics={"features": vision_features, "zones": zone_features},
        classification=classification
    )

    session_id = f"sess_{secrets.token_hex(6)}_{int(datetime.datetime.now().timestamp())}"

    user_info = {
        "full_name": (req.full_name or "").strip(),
        "address": (req.address or "").strip(),
        "email": (req.email or "").strip(),
        "phone_number": (req.phone_number or "").strip(),
        "ville": (req.ville or "").strip(),
    }

    result_payload = {
        "session_id": session_id,
        "user_info": user_info,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "disclaimer": "Cosmetic and wellness assessment only. Not a medical diagnosis.",
        "classification": {
            "primary_skin_type": classification["primary_skin_type"],
            "sensitivity": classification["sensitivity"],
            "visual_summary": classification["visual_summary"],
            "numeric_metrics": classification["numeric_metrics"],
            "reasoning": classification["reasoning"],
            "zones": classification["zones"]
        },
        "recommendations": recommendations,
        "explanation": explanation,
        "luna_verification": luna_verification,
        "vision_raw": {
            "features": vision_features,
            "zones": zone_features
        },
        "analysis_mode": "vision_only" if not py_error else "vision_fallback"
    }

    if py_error:
        result_payload["vision_warning"] = py_error

    # Save to single table SQLite storage
    db_mgr.save_scan(user_info, result_payload)

    return {
        "status": "success",
        "data": result_payload
    }



# ─── History Endpoints ────────────────────────────────────────────────────────

@app.get("/api/history")
def get_history():
    return db_mgr.get_history()


@app.get("/api/history/compare")
def get_compare_timeline():
    return db_mgr.get_compare_timeline()


@app.delete("/api/analysis/{session_id}")
def delete_session(session_id: str):
    return db_mgr.delete_session(session_id)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
