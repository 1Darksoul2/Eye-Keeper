"""
EyeKeeper v3 — FastAPI backend  (auth + payments edition)
─────────────────────────────────────────────────────────
ADDITIONS on top of original v3 (nothing removed):
  ✅ JWT authentication  (signup / login / me)
  ✅ bcrypt password hashing
  ✅ Per-user file storage  data/users/{id}/screenshots/{persons,targets,crops}
  ✅ Per-user detection log
  ✅ Per-user target-face isolation
  ✅ Per-user session state (video / webcam / CCTV)
  ✅ Streaming endpoints accept ?token= query param
  ✅ Razorpay payment routes  (POST /create_order, POST /verify_payment)
  ✅ Protected /playground route (serves playground.html, auth is client-side)
  ✅ /login HTML route (serves login.html)
"""

from __future__ import annotations

# ── Load .env FIRST so os.getenv picks up keys ────────────────────
from dotenv import load_dotenv
load_dotenv()

import base64
import io
import json
import os
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from ultralytics import YOLO

# ── Auth / DB / Payments ─────────────────────────────────────────
from auth import (
    create_access_token,
    get_current_user,
    get_current_user_stream,
    hash_password,
    verify_password,
)
from database import (
    create_user,
    get_user_by_email,
    get_user_by_id,
    init_db,
)
import payments as _payments_module

# ──────────────────────────────────────────────────────────────────
# Dirs & constants
# ──────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
SCREENSHOTS_DIR  = BASE_DIR / "screenshots"
PERSONS_DIR      = BASE_DIR / "screenshots" / "persons"
TARGETS_DIR      = BASE_DIR / "screenshots" / "targets"
CROPS_DIR        = BASE_DIR / "screenshots" / "crops"
for _d in (SCREENSHOTS_DIR, PERSONS_DIR, TARGETS_DIR, CROPS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Per-user data root
USER_DATA_DIR = BASE_DIR / "data" / "users"
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_IMGSZ     = 1280
DEFAULT_CONF      = 0.15
FACE_MATCH_THRESH = 0.55
FACE_MIN_SIZE     = 40

# ──────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────
app = FastAPI(title="EyeKeeper v3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Razorpay payment routes
app.include_router(_payments_module.router)

# Initialise SQLite on startup
init_db()

print(f"[EK] Device: {DEVICE}")

# ──────────────────────────────────────────────────────────────────
# Per-user directory helper (cached)
# ──────────────────────────────────────────────────────────────────
_user_dirs_cache: Dict[int, dict] = {}

def get_user_dirs(user_id: int) -> dict:
    if user_id in _user_dirs_cache:
        return _user_dirs_cache[user_id]
    base = USER_DATA_DIR / str(user_id) / "screenshots"
    dirs = {
        "persons": base / "persons",
        "targets": base / "targets",
        "crops":   base / "crops",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    _user_dirs_cache[user_id] = dirs
    return dirs

# ──────────────────────────────────────────────────────────────────
# Face recognition engine
# ──────────────────────────────────────────────────────────────────
_FACE_BACKEND = "none"
_cascade = None
try:
    from deepface import DeepFace as _DF
    _DF.build_model("Facenet")
    _FACE_BACKEND = "deepface"
    print("[EK] Face backend: DeepFace/Facenet")
except Exception:
    try:
        _cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        _FACE_BACKEND = "opencv"
        print("[EK] Face backend: OpenCV Haar (fallback)")
    except Exception:
        print("[EK] Face backend: DISABLED")


@dataclass
class TargetFace:
    name:        str
    embedding:   Optional[np.ndarray]
    thumb_b64:   str
    opencv_face: Optional[np.ndarray]


# ── Global target (legacy / unauthenticated fallback) ─────────────
_target_lock = threading.Lock()
_target: Optional[TargetFace] = None

# ── Per-user targets ───────────────────────────────────────────────
_user_targets:      Dict[int, Optional[TargetFace]] = {}
_user_targets_lock  = threading.Lock()


def _get_user_target(user_id: Optional[int]) -> Optional[TargetFace]:
    if user_id is None:
        with _target_lock:
            return _target
    with _user_targets_lock:
        return _user_targets.get(user_id)


def _set_user_target(user_id: Optional[int], tgt: Optional[TargetFace]) -> None:
    if user_id is None:
        global _target
        with _target_lock:
            _target = tgt
        return
    with _user_targets_lock:
        _user_targets[user_id] = tgt


# ──────────────────────────────────────────────────────────────────
# Face helpers (unchanged logic)
# ──────────────────────────────────────────────────────────────────
def _extract_embedding(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    if _FACE_BACKEND == "deepface":
        try:
            import tempfile as _tf
            with _tf.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, img_bgr)
                res = _DF.represent(
                    tmp.name, model_name="Facenet",
                    enforce_detection=True, detector_backend="opencv",
                )
                os.unlink(tmp.name)
            if res:
                return np.array(res[0]["embedding"], dtype=np.float32)
        except Exception as e:
            print(f"[EK] DeepFace embed error: {e}")
    return None


def _compare_faces(target: TargetFace, crop_bgr: np.ndarray) -> Tuple[bool, float]:
    if target is None:
        return False, 0.0
    if _FACE_BACKEND == "deepface" and target.embedding is not None:
        emb = _extract_embedding(crop_bgr)
        if emb is None:
            return False, 0.0
        a   = target.embedding / (np.linalg.norm(target.embedding) + 1e-8)
        b   = emb / (np.linalg.norm(emb) + 1e-8)
        sim = float(np.dot(a, b))
        return sim >= FACE_MATCH_THRESH, sim
    elif _FACE_BACKEND == "opencv" and target.opencv_face is not None:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
        sz   = (64, 64)
        t    = cv2.resize(target.opencv_face, sz).astype(np.float32)
        c    = cv2.resize(gray, sz).astype(np.float32)
        t -= t.mean(); c -= c.mean()
        denom  = np.linalg.norm(t) * np.linalg.norm(c) + 1e-8
        sim    = float(np.sum(t * c) / denom)
        sim_01 = (sim + 1) / 2
        return sim_01 >= 0.65, sim_01
    return False, 0.0


def _detect_faces_in_crop(crop_bgr: np.ndarray) -> List[np.ndarray]:
    if _FACE_BACKEND == "none":
        return [crop_bgr]
    if _FACE_BACKEND in ("deepface", "opencv") and _cascade is not None:
        gray  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        faces = _cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4,
            minSize=(FACE_MIN_SIZE, FACE_MIN_SIZE),
        )
        if len(faces):
            return [crop_bgr[fy: fy + fh, fx: fx + fw] for fx, fy, fw, fh in faces]
    return [crop_bgr]


# ──────────────────────────────────────────────────────────────────
# Per-user detection log
# ──────────────────────────────────────────────────────────────────
_person_log:      deque          = deque(maxlen=1000)   # global fallback
_log_lock         = threading.Lock()

_user_logs:       Dict[int, deque] = {}
_user_logs_lock   = threading.Lock()


def _get_user_log(user_id: Optional[int]) -> deque:
    if user_id is None:
        return _person_log
    with _user_logs_lock:
        if user_id not in _user_logs:
            _user_logs[user_id] = deque(maxlen=1000)
        return _user_logs[user_id]


def _log_person(
    label: str, conf: float, matched_target: bool, target_score: float,
    source: str, video_ts: Optional[float],
    screenshot_path: Optional[str], crop_path: Optional[str],
    user_id: Optional[int] = None,
) -> None:
    entry = {
        "id":             int(time.time() * 1000),
        "wall_ts":        datetime.now().isoformat(timespec="milliseconds"),
        "video_ts":       round(video_ts, 2) if video_ts is not None else None,
        "label":          label,
        "confidence":     round(conf, 4),
        "source":         source,
        "matched_target": matched_target,
        "target_score":   round(target_score, 4),
        "screenshot":     Path(screenshot_path).name if screenshot_path else None,
        "crop":           Path(crop_path).name if crop_path else None,
    }
    log = _get_user_log(user_id)
    if user_id is None:
        with _log_lock:
            log.append(entry)
    else:
        with _user_logs_lock:
            log.append(entry)


# ──────────────────────────────────────────────────────────────────
# YOLO models
# ──────────────────────────────────────────────────────────────────
det_model = YOLO("yoloe-26s-seg.pt").to(DEVICE)
seg_model = YOLO("yoloe-26s-seg.pt").to(DEVICE)
_dummy    = np.zeros((DEFAULT_IMGSZ, DEFAULT_IMGSZ, 3), dtype=np.uint8)
det_model(_dummy, imgsz=DEFAULT_IMGSZ, verbose=False)
seg_model(_dummy, imgsz=DEFAULT_IMGSZ, verbose=False)
print("[EK] Models ready")
_model_lock = threading.Lock()

LIVE_IMGSZ       = 640
LIVE_INFER_EVERY = 2
_last_prompts_set: List[str] = []


def _set_classes_if_changed(prompts: List[str]) -> None:
    global _last_prompts_set
    if prompts != _last_prompts_set:
        det_model.set_classes([p.lower() for p in prompts])
        _last_prompts_set = list(prompts)


# ──────────────────────────────────────────────────────────────────
# Session state (per-user)
# ──────────────────────────────────────────────────────────────────
@dataclass
class SessionState:
    video_path:            Optional[str]  = None
    prompts:               List[str]      = field(default_factory=list)
    use_segmentation:      bool           = False
    conf:                  float          = DEFAULT_CONF
    imgsz:                 int            = DEFAULT_IMGSZ
    tile_overlap:          float          = 0.3
    frame_count:           int            = 0
    auto_screenshot:       bool           = True
    save_crops:            bool           = True
    screenshot_cooldown:   float          = 2.0
    face_recognition:      bool           = True
    _last_saved:           dict           = field(default_factory=dict)
    # ── Per-user fields ───────────────────────────────────────────
    user_id:               Optional[int]  = None

    def should_save(self, key: str) -> bool:
        now  = time.time()
        last = self._last_saved.get(key, 0)
        if now - last >= self.screenshot_cooldown:
            self._last_saved[key] = now
            return True
        return False


# ── Global state (legacy / single-user) ──────────────────────────
_state      = SessionState()
_state_lock = threading.Lock()

# ── Per-user states ───────────────────────────────────────────────
_user_states:      Dict[int, SessionState] = {}
_user_states_lock  = threading.Lock()


def _get_user_state(user_id: int) -> SessionState:
    """Return (or create) the mutable session state for a user."""
    with _user_states_lock:
        if user_id not in _user_states:
            s         = SessionState()
            s.user_id = user_id
            _user_states[user_id] = s
        return _user_states[user_id]


def _snap_user_state(user_id: int) -> SessionState:
    """Create an immutable snapshot of a user's state for use in generator functions."""
    s = _get_user_state(user_id)
    with _user_states_lock:
        snap = SessionState(
            video_path           = s.video_path,
            prompts              = list(s.prompts),
            use_segmentation     = s.use_segmentation,
            conf                 = s.conf,
            imgsz                = s.imgsz,
            tile_overlap         = s.tile_overlap,
            auto_screenshot      = s.auto_screenshot,
            save_crops           = s.save_crops,
            screenshot_cooldown  = s.screenshot_cooldown,
            face_recognition     = s.face_recognition,
            user_id              = user_id,
        )
    return snap


def _snap_state() -> SessionState:
    """Legacy snapshot (used by unauthenticated / fallback code paths)."""
    with _state_lock:
        return SessionState(
            video_path           = _state.video_path,
            prompts              = list(_state.prompts),
            use_segmentation     = _state.use_segmentation,
            conf                 = _state.conf,
            imgsz                = _state.imgsz,
            tile_overlap         = _state.tile_overlap,
            auto_screenshot      = _state.auto_screenshot,
            save_crops           = _state.save_crops,
            screenshot_cooldown  = _state.screenshot_cooldown,
            face_recognition     = _state.face_recognition,
        )


# ──────────────────────────────────────────────────────────────────
# Screenshot helpers  (user-aware)
# ──────────────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _vidts_str(video_ts: Optional[float]) -> str:
    if video_ts is None:
        return ""
    m = int(video_ts // 60)
    s = video_ts % 60
    return f"_at_{m:02d}m{s:05.2f}s".replace(".", "p")


def _stamp(frame: np.ndarray, label: str, conf: float, matched: bool) -> np.ndarray:
    out    = frame.copy()
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    col    = (0, 72, 255) if matched else (0, 229, 128)
    tag    = f"[TARGET ★ {conf:.0%}]" if matched else f"[{label} {conf:.0%}]"
    cv2.putText(
        out, f"{ts_str}  {tag}",
        (8, out.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA,
    )
    return out


def save_person_screenshot(
    frame: np.ndarray, label: str, conf: float,
    matched: bool, source: str,
    video_ts: Optional[float] = None,
    user_id:  Optional[int]   = None,
) -> str:
    dirs     = get_user_dirs(user_id) if user_id else {"persons": PERSONS_DIR, "targets": TARGETS_DIR}
    stamped  = _stamp(frame, label, conf, matched)
    pfx      = "TARGET" if matched else label.replace(" ", "_")
    filename = f"{source}_{pfx}_{conf:.2f}{_vidts_str(video_ts)}_{_ts()}.jpg"
    folder   = dirs["targets"] if matched else dirs["persons"]
    path     = folder / filename
    cv2.imwrite(str(path), stamped, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[SS] {'🎯 TARGET' if matched else '👤 Person'}: {filename}")
    return str(path)


def save_crop(
    crop: np.ndarray, label: str, matched: bool,
    source: str, video_ts: Optional[float] = None,
    user_id: Optional[int] = None,
) -> str:
    dirs     = get_user_dirs(user_id) if user_id else {"crops": CROPS_DIR}
    pfx      = "TARGET_crop" if matched else "person_crop"
    filename = f"{source}_{pfx}{_vidts_str(video_ts)}_{_ts()}.jpg"
    path     = dirs["crops"] / filename
    cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return str(path)


# ──────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────
def preprocess_for_small_objects(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)
    lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    enh   = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.filter2D(enh, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))


# ──────────────────────────────────────────────────────────────────
# Colour / label helpers
# ──────────────────────────────────────────────────────────────────
_COLS       = [(0, 229, 160), (37, 99, 255), (255, 61, 110), (255, 165, 0), (180, 0, 255), (0, 200, 255)]
_TARGET_COL = (0, 72, 255)


def _col(label: str) -> tuple:
    return _COLS[abs(hash(label)) % len(_COLS)]


def _resolve_label(names, cls_t, prompts: List[str]) -> str:
    try:
        idx = int(cls_t.item() if hasattr(cls_t, "item") else int(cls_t))
    except Exception:
        idx = 0
    if isinstance(names, list) and 0 <= idx < len(names):
        return str(names[idx])
    if isinstance(names, dict):
        lbl = names.get(idx) or names.get(str(idx))
        if lbl:
            return str(lbl)
    if prompts and 0 <= idx < len(prompts):
        return prompts[idx]
    return f"obj_{idx}"


def _draw_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
              label: str, conf: float, thick: int = 2) -> np.ndarray:
    col  = _TARGET_COL if label.startswith("TARGET") else _col(label)
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)
    text = label if len(label) > 8 else f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    lx, ly = x1, max(y1 - th - 6, 0)
    cv2.rectangle(frame, (lx, ly), (lx + tw + 8, ly + th + 6), col, -1)
    cv2.putText(
        frame, text, (lx + 4, ly + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255) if col == _TARGET_COL else (0, 0, 0),
        1, cv2.LINE_AA,
    )
    return frame


# ──────────────────────────────────────────────────────────────────
# JPEG helpers
# ──────────────────────────────────────────────────────────────────
_JPEG_Q      = [cv2.IMWRITE_JPEG_QUALITY, 75]
_JPEG_Q_LIVE = [cv2.IMWRITE_JPEG_QUALITY, 60]


def _encode(frame: np.ndarray, live: bool = False) -> bytes:
    q  = _JPEG_Q_LIVE if live else _JPEG_Q
    ok, buf = cv2.imencode(".jpg", frame, q)
    return buf.tobytes() if ok else b""


def _chunk(d: bytes) -> bytes:
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + d + b"\r\n"


def _blank() -> bytes:
    return _chunk(_encode(np.zeros((360, 640, 3), dtype=np.uint8)))


# ──────────────────────────────────────────────────────────────────
# Core inference (user-aware)
# ──────────────────────────────────────────────────────────────────
def _infer(
    frame: np.ndarray, state: SessionState,
    is_video: bool = False, video_ts: Optional[float] = None,
    live: bool = False,
) -> np.ndarray:
    user_id  = state.user_id
    prompts  = list(state.prompts)
    if "person" not in [p.lower() for p in prompts]:
        prompts.insert(0, "person")

    h, w      = frame.shape[:2]
    annotated = frame.copy()
    infer_sz  = LIVE_IMGSZ if live else state.imgsz

    if live:
        state.frame_count += 1
        if state.frame_count % LIVE_INFER_EVERY != 0:
            if hasattr(state, "_cached_boxes"):
                for (x1, y1, x2, y2, lbl, cf) in state._cached_boxes:
                    annotated = _draw_box(annotated, x1, y1, x2, y2, lbl, cf)
            return annotated

    if live and max(h, w) > infer_sz:
        scale       = infer_sz / max(h, w)
        infer_frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        inv_scale   = 1.0 / scale
    else:
        infer_frame = frame
        inv_scale   = 1.0

    with _model_lock:
        _set_classes_if_changed(prompts)
        results = det_model(infer_frame, conf=state.conf, imgsz=infer_sz, verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        if live:
            state._cached_boxes = []
        return annotated

    names      = results.names
    source     = "video" if is_video else "webcam"
    cur_target = _get_user_target(user_id)

    cached = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if inv_scale != 1.0:
            x1 = int(x1 * inv_scale); y1 = int(y1 * inv_scale)
            x2 = int(x2 * inv_scale); y2 = int(y2 * inv_scale)

        conf_val  = float(box.conf[0])
        raw_label = _resolve_label(names, box.cls[0], prompts)
        is_person = "person" in raw_label.lower()

        matched      = False
        target_score = 0.0

        if is_person and state.face_recognition and cur_target is not None:
            cx1 = max(0, x1); cy1 = max(0, y1)
            cx2 = min(w, x2); cy2 = min(h, y2)
            if cx2 > cx1 and cy2 > cy1:
                for fc in _detect_faces_in_crop(frame[cy1:cy2, cx1:cx2]):
                    if fc.shape[0] >= FACE_MIN_SIZE and fc.shape[1] >= FACE_MIN_SIZE:
                        m, sc = _compare_faces(cur_target, fc)
                        if sc > target_score:
                            target_score = sc; matched = m

        display_label = f"TARGET★ {cur_target.name}" if (matched and cur_target) else raw_label
        annotated     = _draw_box(annotated, x1, y1, x2, y2, display_label, conf_val)
        cached.append((x1, y1, x2, y2, display_label, conf_val))

        if is_person and state.auto_screenshot:
            ss_key = "TARGET" if matched else "person"
            if state.should_save(ss_key):
                ss_path   = save_person_screenshot(
                    annotated, display_label, conf_val, matched, source, video_ts, user_id=user_id,
                )
                crop_path = None
                if state.save_crops:
                    cx1 = max(0, x1); cy1 = max(0, y1)
                    cx2 = min(w, x2); cy2 = min(h, y2)
                    crop_path = save_crop(
                        frame[cy1:cy2, cx1:cx2], display_label, matched, source, video_ts, user_id=user_id,
                    )
                _log_person(display_label, conf_val, matched, target_score, source, video_ts, ss_path, crop_path, user_id=user_id)

        elif not is_person and state.auto_screenshot:
            if state.should_save(raw_label):
                ss_path = save_person_screenshot(
                    annotated, raw_label, conf_val, False, source, video_ts, user_id=user_id,
                )
                _log_person(raw_label, conf_val, False, 0.0, source, video_ts, ss_path, None, user_id=user_id)

    if live:
        state._cached_boxes = cached

    return annotated


# ──────────────────────────────────────────────────────────────────
# Streaming generators
# ──────────────────────────────────────────────────────────────────
def _video_stream(state: SessionState):
    path = state.video_path
    if not path or not os.path.exists(path):
        yield _blank(); return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        yield _blank(); return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (1.0 <= fps <= 120.0):
        fps = 25.0
    spf             = 1.0 / fps
    state.frame_count = 0
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            vts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield _chunk(_encode(_infer(frame, state, is_video=True, video_ts=vts)))
            rem = spf - (time.time() - t0)
            if rem > 0:
                time.sleep(rem)
            else:
                for _ in range(min(int(-rem / spf), 6)):
                    if not cap.read()[0]:
                        return
    except GeneratorExit:
        pass
    finally:
        cap.release()


def _webcam_stream(state: SessionState):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        for _ in range(30):
            yield _blank(); time.sleep(0.1)
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          30)
    state.frame_count  = 0
    state._cached_boxes = []
    try:
        while True:
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret:
                ret, frame = cap.read()
            if not ret:
                yield _blank(); time.sleep(0.05); continue
            yield _chunk(_encode(_infer(frame, state, is_video=False, live=True), live=True))
    except GeneratorExit:
        pass
    finally:
        cap.release()


# ──────────────────────────────────────────────────────────────────
# Routes — streams  (token required)
# ──────────────────────────────────────────────────────────────────
@app.get("/video_feed")
def video_feed(current_user: dict = Depends(get_current_user_stream)):
    user_id = int(current_user["sub"])
    snap    = _snap_user_state(user_id)
    if not snap.video_path:
        return JSONResponse({"error": "No video uploaded"}, status_code=400)
    return StreamingResponse(_video_stream(snap), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/webcam_feed")
def webcam_feed(current_user: dict = Depends(get_current_user_stream)):
    user_id = int(current_user["sub"])
    return StreamingResponse(
        _webcam_stream(_snap_user_state(user_id)),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ──────────────────────────────────────────────────────────────────
# CCTV — brand templates & state
# ──────────────────────────────────────────────────────────────────
import socket
import ipaddress
import concurrent.futures

CCTV_BRANDS: Dict[str, Dict] = {
    "cpplus":     {"name": "CP Plus",      "port": 554, "fallbacks": [
        "rtsp://{user}:{pass}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=0",
        "rtsp://{user}:{pass}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=1",
        "rtsp://{user}:{pass}@{ip}:{port}/h264/ch{channel}/main/av_stream",
        "rtsp://{user}:{pass}@{ip}:{port}/Streaming/Channels/{channel}01",
    ]},
    "hikvision":  {"name": "Hikvision",    "port": 554, "fallbacks": [
        "rtsp://{user}:{pass}@{ip}:{port}/Streaming/Channels/{channel}01",
        "rtsp://{user}:{pass}@{ip}:{port}/Streaming/Channels/{channel}02",
        "rtsp://{user}:{pass}@{ip}:{port}/h264/ch{channel}/main/av_stream",
        "rtsp://{user}:{pass}@{ip}:{port}/ISAPI/Streaming/Channels/{channel}01",
    ]},
    "dahua":      {"name": "Dahua",        "port": 554, "fallbacks": [
        "rtsp://{user}:{pass}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=0",
        "rtsp://{user}:{pass}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=1",
        "rtsp://{user}:{pass}@{ip}:{port}/live",
    ]},
    "tvt":        {"name": "TVT / OEM",    "port": 554, "fallbacks": [
        "rtsp://{user}:{pass}@{ip}:{port}/user={user}&password={pass}&channel={channel}&stream=0.sdp",
        "rtsp://{user}:{pass}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=0",
    ]},
    "prama":      {"name": "Prama",        "port": 554, "fallbacks": [
        "rtsp://{user}:{pass}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=0",
        "rtsp://{user}:{pass}@{ip}:{port}/Streaming/Channels/{channel}01",
    ]},
    "godrej":     {"name": "Godrej",       "port": 554, "fallbacks": [
        "rtsp://{user}:{pass}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=0",
        "rtsp://{user}:{pass}@{ip}:{port}/Streaming/Channels/{channel}01",
    ]},
    "reolink":    {"name": "Reolink",      "port": 554, "fallbacks": [
        "rtsp://{user}:{pass}@{ip}:{port}/h264Preview_0{channel}_main",
        "rtsp://{user}:{pass}@{ip}:{port}/h264Preview_0{channel}_sub",
        "rtsp://{user}:{pass}@{ip}:{port}/stream{channel}",
    ]},
    "axis":       {"name": "Axis",         "port": 554, "fallbacks": [
        "rtsp://{user}:{pass}@{ip}:{port}/axis-media/media.amp",
        "rtsp://{user}:{pass}@{ip}:{port}/axis-media/media.amp?videocodec=h264",
    ]},
    "amcrest":    {"name": "Amcrest",      "port": 554, "fallbacks": [
        "rtsp://{user}:{pass}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=0",
        "rtsp://{user}:{pass}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=1",
    ]},
    "tapo":       {"name": "TP-Link Tapo", "port": 554, "fallbacks": [
        "rtsp://{user}:{pass}@{ip}:{port}/stream1",
        "rtsp://{user}:{pass}@{ip}:{port}/stream2",
    ]},
    "foscam":     {"name": "Foscam",       "port": 88,  "fallbacks": [
        "rtsp://{user}:{pass}@{ip}:{port}/videoMain",
        "rtsp://{user}:{pass}@{ip}:{port}/videoSub",
    ]},
    "vivotek":    {"name": "Vivotek",      "port": 554, "fallbacks": [
        "rtsp://{user}:{pass}@{ip}:{port}/live.sdp",
        "rtsp://{user}:{pass}@{ip}:{port}/video.mp4",
    ]},
    "uniview":    {"name": "Uniview",      "port": 554, "fallbacks": [
        "rtsp://{user}:{pass}@{ip}:{port}/media/video{channel}",
        "rtsp://{user}:{pass}@{ip}:{port}/unicast/c{channel}/s0/live",
    ]},
    "hanwha":     {"name": "Hanwha",       "port": 554, "fallbacks": [
        "rtsp://{user}:{pass}@{ip}:{port}/profile{channel}/media.smp",
        "rtsp://{user}:{pass}@{ip}:{port}/cam{channel}/h264",
    ]},
    "bosch":      {"name": "Bosch",        "port": 554, "fallbacks": [
        "rtsp://{user}:{pass}@{ip}:{port}/rtsp_tunnel",
        "rtsp://{user}:{pass}@{ip}:{port}/video",
    ]},
    "generic":    {"name": "Generic RTSP", "port": 554, "fallbacks": [
        "rtsp://{user}:{pass}@{ip}:{port}/",
        "rtsp://{user}:{pass}@{ip}:{port}/live",
        "rtsp://{user}:{pass}@{ip}:{port}/stream",
        "rtsp://{user}:{pass}@{ip}:{port}/video",
        "rtsp://{user}:{pass}@{ip}:{port}/live/ch00_0",
    ]},
    "custom":     {"name": "Custom URL",   "port": 554, "fallbacks": []},
}

for _bid, _bv in CCTV_BRANDS.items():
    _bv["template"] = _bv["fallbacks"][0] if _bv.get("fallbacks") else ""


@dataclass
class CCTVState:
    rtsp_url:  Optional[str] = None
    brand:     str           = "generic"
    label:     str           = "CCTV"
    active:    bool          = False
    connected: bool          = False
    error_msg: str           = ""


_cctv_state      = CCTVState()
_cctv_state_lock = threading.Lock()

_RTSP_FFMPEG_FLAGS = (
    "rtsp_transport;tcp|stimeout;5000000|max_delay;500000|"
    "allowed_media_types;video|fflags;nobuffer|flags;low_delay"
)


def _open_cap(url: str) -> Optional[cv2.VideoCapture]:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _RTSP_FFMPEG_FLAGS
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          20)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return cap
        time.sleep(0.3)
    cap.release()
    return None


def _make_info_frame(line1: str, line2: str = "", line3: str = "", color=(0, 80, 255)) -> np.ndarray:
    f = np.zeros((360, 640, 3), dtype=np.uint8)
    for x in range(0, 640, 40): cv2.line(f, (x, 0), (x, 360), (20, 20, 30), 1)
    for y in range(0, 360, 40): cv2.line(f, (0, y), (640, y), (20, 20, 30), 1)
    cv2.putText(f, line1, (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    if line2: cv2.putText(f, line2, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 140), 1, cv2.LINE_AA)
    if line3: cv2.putText(f, line3, (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 100), 1, cv2.LINE_AA)
    return f


def _build_fallback_urls(brand_id: str, ip: str, port: int, user: str, pw: str, channel: int) -> List[str]:
    brand = CCTV_BRANDS.get(brand_id, CCTV_BRANDS["generic"])
    urls  = []
    for tmpl in brand.get("fallbacks", []):
        u = (tmpl
             .replace("{ip}",      ip)
             .replace("{port}",    str(port))
             .replace("{user}",    user)
             .replace("{pass}",    pw)
             .replace("{channel}", str(channel)))
        urls.append(u)
    return urls


def _cctv_stream(rtsp_url: str, brand_id: str, ip: str, port: int, user: str, pw: str, channel: int,
                 state: SessionState):
    urls_to_try = [rtsp_url]
    for u in _build_fallback_urls(brand_id, ip, port, user, pw, channel):
        if u not in urls_to_try:
            urls_to_try.append(u)

    cap   = None
    tried = 0
    total = len(urls_to_try)

    for url in urls_to_try:
        tried += 1
        info = _make_info_frame(
            f"Trying URL {tried}/{total}...", url[:70],
            f"Brand: {CCTV_BRANDS.get(brand_id,{}).get('name','?')}", color=(0, 180, 255),
        )
        for _ in range(6):
            yield _chunk(_encode(info)); time.sleep(0.15)
        with _cctv_state_lock:
            if not _cctv_state.active:
                return
        cap = _open_cap(url)
        if cap is not None:
            with _cctv_state_lock:
                _cctv_state.rtsp_url  = url
                _cctv_state.connected = True
                _cctv_state.error_msg = ""
            print(f"[CCTV] ✅ Connected: {url}")
            break
        else:
            print(f"[CCTV] ❌ Failed: {url}")

    if cap is None:
        err = _make_info_frame(
            "STREAM UNAVAILABLE",
            "All URL paths tried — check IP, port, credentials",
            f"Last tried: {urls_to_try[-1][:60]}", color=(0, 50, 255),
        )
        with _cctv_state_lock:
            _cctv_state.connected = False; _cctv_state.error_msg = "All URLs failed"
        for _ in range(80):
            yield _chunk(_encode(err)); time.sleep(0.1)
        return

    state.frame_count   = 0
    state._cached_boxes = []

    try:
        while True:
            with _cctv_state_lock:
                if not _cctv_state.active:
                    break
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret:
                ret, frame = cap.read()
            if not ret:
                yield _blank(); time.sleep(0.05); continue
            annotated = _infer(frame, state, is_video=True, live=True)
            yield _chunk(_encode(annotated, live=True))
    except GeneratorExit:
        pass
    finally:
        cap.release()
        with _cctv_state_lock:
            _cctv_state.active    = False
            _cctv_state.connected = False


def _probe_rtsp(host: str, port: int = 554, timeout: float = 0.8) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _get_local_subnet() -> Optional[str]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        parts = local_ip.rsplit(".", 1)
        return parts[0] + "." if len(parts) == 2 else None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────
# Routes — CCTV
# ──────────────────────────────────────────────────────────────────
@app.get("/cctv/brands")
def cctv_brands():
    return {"brands": [
        {"id": k, "name": v["name"], "template": v["template"], "default_port": v["port"]}
        for k, v in CCTV_BRANDS.items()
    ]}


@app.post("/cctv/test")
async def cctv_test(
    rtsp_url: str = Form(...), brand_id: str = Form("generic"),
    ip: str = Form(""), port: int = Form(554),
    user: str = Form("admin"), pw: str = Form(""), channel: int = Form(1),
    current_user: dict = Depends(get_current_user),
):
    urls = [rtsp_url]
    for u in _build_fallback_urls(brand_id, ip, port, user, pw, channel):
        if u not in urls:
            urls.append(u)
    for idx, url in enumerate(urls):
        cap = _open_cap(url)
        if cap is not None:
            cap.release()
            return {"success": True, "working_url": url, "tried": idx + 1, "total": len(urls)}
    return JSONResponse(
        {"success": False, "tried": len(urls), "total": len(urls),
         "message": "No URL worked. Check IP, credentials, and camera brand."},
        status_code=422,
    )


@app.post("/cctv/connect")
async def cctv_connect(
    rtsp_url:            str   = Form(...), brand: str = Form("generic"),
    ip:                  str   = Form(""),  port:  int = Form(554),
    user:                str   = Form("admin"), pw: str = Form(""), channel: int = Form(1),
    label:               str   = Form("CCTV"), prompt: str = Form(""),
    conf:                float = Form(DEFAULT_CONF), auto_screenshot: bool = Form(True),
    save_crops:          bool  = Form(True), screenshot_cooldown: float = Form(2.0),
    face_recognition:    bool  = Form(True),
    current_user:        dict  = Depends(get_current_user),
):
    user_id = int(current_user["sub"])
    with _cctv_state_lock:
        _cctv_state.rtsp_url  = rtsp_url; _cctv_state.brand = brand
        _cctv_state.label     = label;    _cctv_state.active = True
        _cctv_state.connected = False;    _cctv_state.error_msg = ""

    s = _get_user_state(user_id)
    with _user_states_lock:
        s.prompts            = [p.strip() for p in prompt.split(",") if p.strip()]
        s.conf               = max(0.01, min(1.0, conf))
        s.auto_screenshot    = auto_screenshot; s.save_crops = save_crops
        s.screenshot_cooldown= max(0.5, screenshot_cooldown)
        s.face_recognition   = face_recognition
        s.frame_count        = 0; s._last_saved = {}

    return {"status": "connecting", "rtsp_url": rtsp_url, "brand": brand,
            "ip": ip, "port": port, "user": user, "channel": channel}


@app.post("/cctv/disconnect")
def cctv_disconnect(current_user: dict = Depends(get_current_user)):
    with _cctv_state_lock:
        _cctv_state.active    = False
        _cctv_state.connected = False
        _cctv_state.rtsp_url  = None
    return {"status": "disconnected"}


@app.get("/cctv/status")
def cctv_status(current_user: dict = Depends(get_current_user)):
    with _cctv_state_lock:
        return {
            "active":    _cctv_state.active,    "connected": _cctv_state.connected,
            "brand":     _cctv_state.brand,     "label":     _cctv_state.label,
            "rtsp_url":  _cctv_state.rtsp_url,  "error":     _cctv_state.error_msg,
        }


@app.get("/cctv_feed")
def cctv_feed(
    ip: str = Query(""), port: int = Query(554),
    user: str = Query("admin"), pw: str = Query(""), channel: int = Query(1),
    current_user: dict = Depends(get_current_user_stream),
):
    with _cctv_state_lock:
        url    = _cctv_state.rtsp_url
        active = _cctv_state.active
        brand  = _cctv_state.brand
    if not url or not active:
        return JSONResponse({"error": "No CCTV stream configured"}, status_code=400)
    uid  = int(current_user["sub"])
    snap = _snap_user_state(uid)
    return StreamingResponse(
        _cctv_stream(url, brand, ip, port, user, pw, channel, snap),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/cctv/scan")
def cctv_scan(
    subnet: Optional[str] = None, ports: str = "554,8554", timeout: float = 0.6,
    current_user: dict = Depends(get_current_user),
):
    port_list = [int(p) for p in ports.split(",") if p.strip().isdigit()] or [554]
    base      = subnet or _get_local_subnet()
    if not base:
        return {"error": "Could not determine local subnet. Provide ?subnet=192.168.1.", "found": []}
    if not base.endswith("."):
        base += "."
    hosts = [f"{base}{i}" for i in range(1, 255)]

    def probe(h):
        for p in port_list:
            if _probe_rtsp(h, p, timeout):
                return {"ip": h, "port": p}
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=80) as ex:
        results = ex.map(probe, hosts)

    return {"subnet": base, "scanned": len(hosts), "found": [r for r in results if r]}


# ──────────────────────────────────────────────────────────────────
# Routes — face target (per-user)
# ──────────────────────────────────────────────────────────────────
@app.post("/upload_face")
async def upload_face(
    file: UploadFile = File(...), name: str = Form("Target"),
    current_user: dict = Depends(get_current_user),
):
    user_id = int(current_user["sub"])
    raw     = await file.read()
    arr     = np.frombuffer(raw, np.uint8)
    img     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Cannot decode image"}, status_code=400)

    th = cv2.resize(img, (120, 120))
    ok, buf   = cv2.imencode(".jpg", th, [cv2.IMWRITE_JPEG_QUALITY, 85])
    thumb_b64 = base64.b64encode(buf.tobytes()).decode() if ok else ""

    embedding   = None
    opencv_face = None
    face_backend= _FACE_BACKEND

    if face_backend == "deepface":
        embedding = _extract_embedding(img)
        if embedding is None:
            return JSONResponse({"error": "No face detected — try a clearer front-facing photo"}, status_code=422)
    elif face_backend == "opencv" and _cascade is not None:
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        if len(faces) == 0:
            return JSONResponse({"error": "No face detected (OpenCV Haar)"}, status_code=422)
        fx, fy, fw, fh = faces[0]
        opencv_face = gray[fy:fy + fh, fx:fx + fw]

    new_target = TargetFace(name=name, embedding=embedding, thumb_b64=thumb_b64, opencv_face=opencv_face)
    _set_user_target(user_id, new_target)

    # Save copy to user's targets dir
    dirs    = get_user_dirs(user_id)
    ts_path = dirs["targets"] / f"target_{name.replace(' ', '_')}_{_ts()}.jpg"
    cv2.imwrite(str(ts_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[EK] Target face set for user {user_id}: {name} ({face_backend})")

    return {
        "status": "ok", "name": name, "backend": face_backend,
        "has_embedding": embedding is not None or opencv_face is not None,
        "thumb_b64": thumb_b64,
    }


@app.get("/target_face")
def get_target_face(current_user: dict = Depends(get_current_user)):
    user_id = int(current_user["sub"])
    t       = _get_user_target(user_id)
    if t is None:
        return JSONResponse({"set": False})
    return {"set": True, "name": t.name, "thumb_b64": t.thumb_b64, "backend": _FACE_BACKEND}


@app.delete("/target_face")
def clear_target_face(current_user: dict = Depends(get_current_user)):
    user_id = int(current_user["sub"])
    _set_user_target(user_id, None)
    return {"status": "cleared"}


# ──────────────────────────────────────────────────────────────────
# Routes — config / upload (per-user)
# ──────────────────────────────────────────────────────────────────
@app.post("/upload_video")
async def upload_video(
    file:                UploadFile = File(...),
    prompt:              str        = Form(""),
    conf:                float      = Form(DEFAULT_CONF),
    imgsz:               int        = Form(DEFAULT_IMGSZ),
    auto_screenshot:     bool       = Form(True),
    save_crops:          bool       = Form(True),
    screenshot_cooldown: float      = Form(2.0),
    face_recognition:    bool       = Form(True),
    current_user:        dict       = Depends(get_current_user),
):
    user_id = int(current_user["sub"])
    s       = _get_user_state(user_id)
    with _user_states_lock:
        if s.video_path and os.path.exists(s.video_path):
            try: os.remove(s.video_path)
            except OSError: pass
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(await file.read()); tmp.close()
        s.video_path          = tmp.name
        s.prompts             = [p.strip() for p in prompt.split(",") if p.strip()]
        s.conf                = max(0.01, min(1.0, conf))
        s.imgsz               = min(1920, max(320, imgsz))
        s.auto_screenshot     = auto_screenshot; s.save_crops = save_crops
        s.screenshot_cooldown = max(0.5, screenshot_cooldown)
        s.face_recognition    = face_recognition
        s.frame_count         = 0; s._last_saved = {}
    return {"status": "uploaded"}


@app.post("/set_webcam_config")
async def set_webcam_config(
    prompt:              str   = Form(""),
    conf:                float = Form(DEFAULT_CONF),
    imgsz:               int   = Form(DEFAULT_IMGSZ),
    auto_screenshot:     bool  = Form(True),
    save_crops:          bool  = Form(True),
    screenshot_cooldown: float = Form(2.0),
    face_recognition:    bool  = Form(True),
    current_user:        dict  = Depends(get_current_user),
):
    user_id = int(current_user["sub"])
    s       = _get_user_state(user_id)
    with _user_states_lock:
        s.prompts             = [p.strip() for p in prompt.split(",") if p.strip()]
        s.conf                = max(0.01, min(1.0, conf))
        s.imgsz               = min(1920, max(320, imgsz))
        s.auto_screenshot     = auto_screenshot; s.save_crops = save_crops
        s.screenshot_cooldown = max(0.5, screenshot_cooldown)
        s.face_recognition    = face_recognition
        s.frame_count         = 0; s._last_saved = {}
    return {"status": "ok"}


# ──────────────────────────────────────────────────────────────────
# Routes — detect (single image, user-aware)
# ──────────────────────────────────────────────────────────────────
@app.post("/detect")
async def detect(
    file:             UploadFile = File(...),
    prompt:           str        = Form(""),
    confidence:       float      = Form(DEFAULT_CONF),
    max_det:          int        = Form(20),
    imgsz:            int        = Form(DEFAULT_IMGSZ),
    face_recognition: bool       = Form(True),
    save_crops:       bool       = Form(True),
    current_user:     dict       = Depends(get_current_user),
):
    user_id = int(current_user["sub"])
    raw     = await file.read()
    arr     = np.frombuffer(raw, np.uint8)
    frame   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "Cannot decode image"}, status_code=400)

    h, w    = frame.shape[:2]
    prompts = [p.strip() for p in prompt.split(",") if p.strip()]
    if "person" not in [p.lower() for p in prompts]:
        prompts.insert(0, "person")
    if max(h, w) > 800:
        frame = preprocess_for_small_objects(frame)

    with _model_lock:
        det_model.set_classes([p.lower() for p in prompts])
        results = det_model(frame, conf=max(0.01, min(1.0, confidence)), imgsz=imgsz, verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        return JSONResponse({"detections": [], "image_size": {"width": w, "height": h}})

    names       = results.names
    detections  = []
    annotated   = frame.copy()
    ss_count    = 0
    cur_target  = _get_user_target(user_id)

    for i, box in enumerate(results.boxes):
        if i >= max_det:
            break
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_val  = round(float(box.conf[0]), 4)
        raw_label = _resolve_label(names, box.cls[0], prompts)
        is_person = "person" in raw_label.lower()

        matched      = False; target_score = 0.0
        if is_person and face_recognition and cur_target is not None:
            cx1, cy1 = max(0, x1), max(0, y1); cx2, cy2 = min(w, x2), min(h, y2)
            if cx2 > cx1 and cy2 > cy1:
                for fc in _detect_faces_in_crop(frame[cy1:cy2, cx1:cx2]):
                    if fc.shape[0] >= FACE_MIN_SIZE and fc.shape[1] >= FACE_MIN_SIZE:
                        m, sc = _compare_faces(cur_target, fc)
                        if sc > target_score:
                            target_score = sc; matched = m

        display_label = f"TARGET★ {cur_target.name}" if matched else raw_label
        annotated     = _draw_box(annotated, x1, y1, x2, y2, display_label, conf_val)
        detections.append({
            "label": display_label, "confidence": conf_val,
            "is_person": is_person, "matched_target": matched,
            "target_score": round(target_score, 4),
            "bbox_norm": [round(x1/w, 6), round(y1/h, 6), round(x2/w, 6), round(y2/h, 6)],
        })

        ss_path = save_person_screenshot(annotated, display_label, conf_val, matched, "image", user_id=user_id)
        ss_count += 1
        crop_path = None
        if is_person and save_crops:
            cx1, cy1 = max(0, x1), max(0, y1); cx2, cy2 = min(w, x2), min(h, y2)
            crop_path = save_crop(frame[cy1:cy2, cx1:cx2], display_label, matched, "image", user_id=user_id)
        _log_person(display_label, conf_val, matched, target_score, "image", None, ss_path, crop_path, user_id=user_id)

    return JSONResponse({
        "detections":       detections,
        "image_size":       {"width": w, "height": h},
        "screenshots_saved": ss_count,
        "targets_found":    sum(1 for d in detections if d["matched_target"]),
    })


# ──────────────────────────────────────────────────────────────────
# Routes — person log (per-user)
# ──────────────────────────────────────────────────────────────────
@app.get("/person_log")
def person_log(
    limit:        int  = Query(100),
    targets_only: bool = Query(False),
    current_user: dict = Depends(get_current_user),
):
    user_id = int(current_user["sub"])
    log     = _get_user_log(user_id)
    with _user_logs_lock:
        entries = list(log)
    entries.reverse()
    if targets_only:
        entries = [e for e in entries if e["matched_target"]]
    return {"entries": entries[:limit], "total": len(entries)}


@app.delete("/person_log")
def clear_person_log(current_user: dict = Depends(get_current_user)):
    user_id = int(current_user["sub"])
    log     = _get_user_log(user_id)
    with _user_logs_lock:
        log.clear()
    return {"status": "cleared"}


# ──────────────────────────────────────────────────────────────────
# Routes — screenshots (per-user)
# ──────────────────────────────────────────────────────────────────
def _list_dir(folder: Path, limit: int) -> List[dict]:
    files = sorted(folder.glob("*.jpg"), key=lambda f: f.stat().st_mtime, reverse=True)
    out   = []
    for f in files[:limit]:
        out.append({
            "filename": f.name,
            "folder":   folder.name,
            "size_kb":  round(f.stat().st_size / 1024, 1),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        })
    return out


@app.get("/screenshots")
def list_screenshots(
    limit:        int  = Query(200),
    folder:       str  = Query("all"),
    current_user: dict = Depends(get_current_user),
):
    user_id = int(current_user["sub"])
    dirs    = get_user_dirs(user_id)
    if folder == "persons":
        items = _list_dir(dirs["persons"], limit)
    elif folder == "targets":
        items = _list_dir(dirs["targets"], limit)
    elif folder == "crops":
        items = _list_dir(dirs["crops"],   limit)
    else:
        items = (
            _list_dir(dirs["targets"], limit) +
            _list_dir(dirs["persons"], limit) +
            _list_dir(dirs["crops"],   limit)
        )
        items.sort(key=lambda x: x["modified"], reverse=True)
        items = items[:limit]
    return {"screenshots": items, "total": len(items)}


@app.get("/screenshots/{folder}/{filename}")
def get_screenshot(
    folder:       str,
    filename:     str,
    current_user: dict = Depends(get_current_user),
):
    user_id    = int(current_user["sub"])
    dirs       = get_user_dirs(user_id)
    folder_map = {"persons": dirs["persons"], "targets": dirs["targets"], "crops": dirs["crops"]}
    d = folder_map.get(folder)
    if not d:
        raise HTTPException(404, "Unknown folder")
    p = d / filename
    if not p.exists():
        raise HTTPException(404, "Not found")
    return Response(p.read_bytes(), media_type="image/jpeg")


@app.delete("/screenshots/{folder}/{filename}")
def delete_screenshot(
    folder:       str,
    filename:     str,
    current_user: dict = Depends(get_current_user),
):
    user_id    = int(current_user["sub"])
    dirs       = get_user_dirs(user_id)
    folder_map = {"persons": dirs["persons"], "targets": dirs["targets"], "crops": dirs["crops"]}
    d = folder_map.get(folder)
    if not d:
        raise HTTPException(404, "Unknown folder")
    p = d / filename
    if not p.exists():
        raise HTTPException(404, "Not found")
    p.unlink()
    return {"deleted": filename}


# ──────────────────────────────────────────────────────────────────
# Routes — auth
# ──────────────────────────────────────────────────────────────────
from pydantic import BaseModel

class SignupRequest(BaseModel):
    email:    str
    password: str

class LoginRequest(BaseModel):
    email:    str
    password: str


@app.post("/auth/signup")
def signup(body: SignupRequest):
    email = body.email.strip().lower()
    if not email or len(body.password) < 6:
        raise HTTPException(400, "Email and password (≥6 chars) required")
    if get_user_by_email(email):
        raise HTTPException(409, "Email already registered")
    user_id = create_user(email, hash_password(body.password))
    token   = create_access_token(user_id, email, "free")
    return {"token": token, "user": {"id": user_id, "email": email, "plan": "free"}}


@app.post("/auth/login")
def login(body: LoginRequest):
    email = body.email.strip().lower()
    user  = get_user_by_email(email)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")
    token = create_access_token(user["id"], user["email"], user["plan"])
    return {"token": token, "user": {"id": user["id"], "email": user["email"], "plan": user["plan"]}}


@app.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    user_id = int(current_user["sub"])
    user    = get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return {"id": user["id"], "email": user["email"], "plan": user["plan"], "credits": user["credits"]}


# ──────────────────────────────────────────────────────────────────
# Routes — misc
# ──────────────────────────────────────────────────────────────────
@app.get("/info")
def info(current_user: dict = Depends(get_current_user)):
    return {"status": "ok", "device": DEVICE, "face_backend": _FACE_BACKEND}


@app.get("/health")
def health(current_user: dict = Depends(get_current_user)):
    user_id = int(current_user["sub"])
    dirs    = get_user_dirs(user_id)
    return {
        "status":       "ok",
        "device":       DEVICE,
        "face_backend": _FACE_BACKEND,
        "target_set":   _get_user_target(user_id) is not None,
        "screenshots": {
            "persons": len(list(dirs["persons"].glob("*.jpg"))),
            "targets": len(list(dirs["targets"].glob("*.jpg"))),
            "crops":   len(list(dirs["crops"].glob("*.jpg"))),
        },
    }


# ──────────────────────────────────────────────────────────────────
# Routes — HTML pages
# ──────────────────────────────────────────────────────────────────
@app.get("/playground", response_class=HTMLResponse)
@app.get("/playground.html", response_class=HTMLResponse)
def serve_playground():
    p = BASE_DIR / "playground.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>playground.html not found</h2>", status_code=404)


@app.get("/login", response_class=HTMLResponse)
@app.get("/login.html", response_class=HTMLResponse)
def serve_login():
    p = BASE_DIR / "login.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>login.html not found</h2>", status_code=404)


@app.get("/", response_class=HTMLResponse)
def serve_index():
    p = BASE_DIR / "index.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found</h2>", status_code=404)


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)