"""
database_manager.py
--------------------
SQLite database with a single table `user_scans` storing user details
(full_name, address, email, phone_number, ville) and skin scan analysis.
"""

import os
import json
import uuid
import datetime
from typing import Dict, Any, List, Optional

from sqlalchemy import (
    create_engine, Column, String, Text, DateTime
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from dotenv import load_dotenv

load_dotenv()

# ── Database URL ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(STORAGE_DIR, 'ai_skin.db')}"
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Single Database Table Model ───────────────────────────────────────────────

class UserScan(Base):
    """
    Single primary table storing user information and scan results.
    """
    __tablename__ = "user_scans"

    id           = Column(String(64),  primary_key=True)
    full_name    = Column(String(255), nullable=False, default="")
    address      = Column(String(255), nullable=True,  default="")
    email        = Column(String(255), nullable=True,  default="")
    phone_number = Column(String(100), nullable=True,  default="")
    ville        = Column(String(100), nullable=True,  default="")
    created_at   = Column(DateTime,    default=datetime.datetime.utcnow)
    payload      = Column(Text,        nullable=False)   # Full JSON blob of analysis result


# Create single table on startup
Base.metadata.create_all(bind=engine)


# ── DatabaseManager Class ─────────────────────────────────────────────────────

class DatabaseManager:
    """
    Unified single-table data layer for storing user details & scan results.
    """

    def _get_db(self) -> Session:
        return SessionLocal()

    def save_scan(self, user_info: Dict[str, Any], session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Saves user information and scan payload into the single `user_scans` table.
        """
        db = self._get_db()
        try:
            sid = session_data.get("session_id", f"sess_{uuid.uuid4().hex[:12]}")
            created_str = session_data.get("created_at", datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
            try:
                created_dt = datetime.datetime.strptime(str(created_str)[:19], "%Y-%m-%d %H:%M:%S")
            except Exception:
                created_dt = datetime.datetime.utcnow()

            # Attach user_info to payload dictionary
            session_data["user_info"] = user_info

            # Clean raw photos from payload to keep DB lean
            clean_data = {k: v for k, v in session_data.items() if k != "photos"}

            scan_record = UserScan(
                id=sid,
                full_name=user_info.get("full_name", "").strip(),
                address=user_info.get("address", "").strip(),
                email=user_info.get("email", "").strip().lower(),
                phone_number=user_info.get("phone_number", "").strip(),
                ville=user_info.get("ville", "").strip(),
                created_at=created_dt,
                payload=json.dumps(clean_data, ensure_ascii=False),
            )

            db.add(scan_record)
            db.commit()
            return {"status": "success", "data": clean_data}
        except Exception as e:
            db.rollback()
            print(f"[DB] save_scan error: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            db.close()

    def get_history(self) -> Dict[str, Any]:
        """
        Retrieve all scans from single user_scans table.
        """
        db = self._get_db()
        try:
            rows = db.query(UserScan).order_by(UserScan.created_at.desc()).all()
            sessions = []
            for row in rows:
                try:
                    p = json.loads(row.payload)
                    # Ensure user details are attached
                    p["user_info"] = {
                        "full_name": row.full_name,
                        "address": row.address,
                        "email": row.email,
                        "phone_number": row.phone_number,
                        "ville": row.ville,
                    }
                    sessions.append(p)
                except Exception:
                    pass

            return {"status": "success", "count": len(sessions), "history": sessions}
        except Exception as e:
            print(f"[DB] get_history error: {e}")
            return {"status": "success", "count": 0, "history": []}
        finally:
            db.close()

    def get_compare_timeline(self) -> Dict[str, Any]:
        """
        Retrieve timeline data for historical comparison.
        """
        db = self._get_db()
        try:
            rows = db.query(UserScan).order_by(UserScan.created_at.asc()).all()
            sessions = []
            for row in rows:
                try:
                    sessions.append({
                        "id": row.id,
                        "full_name": row.full_name,
                        "address": row.address,
                        "email": row.email,
                        "phone_number": row.phone_number,
                        "ville": row.ville,
                        "created_at": row.created_at,
                        "payload": json.loads(row.payload)
                    })
                except Exception:
                    pass

            timeline = []
            for s in sessions:
                p = s.get("payload", {})
                created_at = s.get("created_at", datetime.date.today().strftime("%Y-%m-%d"))
                classification = p.get("classification", {})
                numeric = classification.get("numeric_metrics", {})
                timeline.append({
                    "date": str(created_at)[:10],
                    "session_id": s.get("id", ""),
                    "full_name": s.get("full_name", ""),
                    "email": s.get("email", ""),
                    "phone_number": s.get("phone_number", ""),
                    "ville": s.get("ville", ""),
                    "primary_type": classification.get("primary_skin_type", "Combination"),
                    "sensitivity": classification.get("sensitivity", "Low"),
                    "oiliness": numeric.get("oiliness", 50),
                    "redness": numeric.get("redness", 20),
                    "pores": numeric.get("visible_pores", 30),
                    "blemishes": numeric.get("blemishes", 15),
                    "texture": numeric.get("texture", 30),
                })

            return {"status": "success", "timeline": timeline}
        except Exception as e:
            print(f"[DB] get_compare_timeline error: {e}")
            return {"status": "success", "timeline": []}
        finally:
            db.close()

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        db = self._get_db()
        try:
            row = db.query(UserScan).filter_by(id=session_id).first()
            if row:
                db.delete(row)
                db.commit()
                deleted = True
            else:
                deleted = False

            return {
                "status": "success",
                "message": "Enregistrement supprimé définitivement.",
                "deleted": deleted
            }
        except Exception as e:
            db.rollback()
            print(f"[DB] delete_session error: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            db.close()

