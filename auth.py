"""
EyeKeeper — auth.py
JWT authentication + bcrypt password hashing.
FastAPI dependencies: get_current_user (strict) and get_current_user_stream (accepts query-param token).
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

# ── Config ─────────────────────────────────────────────────────────
SECRET_KEY                 = os.getenv("SECRET_KEY", "eyekeeper-change-me-in-production-2025")
ALGORITHM                  = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS  = 24 * 7   # 1 week

_bearer = HTTPBearer(auto_error=False)


# ──────────────────────────────────────────────────────────────────
# Password helpers
# ──────────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


# ──────────────────────────────────────────────────────────────────
# JWT helpers
# ──────────────────────────────────────────────────────────────────
def create_access_token(user_id: int, email: str, plan: str = "free") -> str:
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {
        "sub":   str(user_id),
        "email": email,
        "plan":  plan,
        "exp":   expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


# ──────────────────────────────────────────────────────────────────
# FastAPI dependencies
# ──────────────────────────────────────────────────────────────────
def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """
    Strict dependency — raises 401 if token is missing or invalid.
    Use on every protected endpoint that receives the token via the
    Authorization: Bearer <token> header.
    """
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


def get_current_user_stream(
    token:       Optional[str]                            = Query(None),
    credentials: Optional[HTTPAuthorizationCredentials]  = Depends(_bearer),
) -> dict:
    """
    Dependency for MJPEG streaming endpoints.
    Accepts the JWT either as:
      • Authorization: Bearer <token>  header, OR
      • ?token=<token>                 query parameter
    (Browser <img src="...?token=xxx"> cannot set headers, so query-param is needed.)
    """
    tok = None
    if credentials:
        tok = credentials.credentials
    elif token:
        tok = token

    if not tok:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(tok)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload