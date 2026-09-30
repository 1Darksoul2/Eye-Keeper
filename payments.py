"""
EyeKeeper — payments.py
Razorpay payment integration.
Exposes an APIRouter that is included in app.py.

Routes:
  GET  /plans          → list available plans (public)
  POST /create_order   → create a Razorpay order   (auth required)
  POST /verify_payment → verify payment & upgrade   (auth required)
"""

import hashlib
import hmac
import os

import razorpay
from fastapi import APIRouter, Depends, Form, HTTPException

from auth import get_current_user
from database import get_user_by_id, update_user_plan

router = APIRouter()

# ── Keys — loaded from .env ─────────────────────────────────────────
RAZORPAY_KEY_ID     = os.getenv("RAZORPAY_KEY_ID",     "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")

# ── Plan catalogue ─────────────────────────────────────────────────
PLANS: dict = {
    "starter": {
        "name":     "Starter",
        "price":    49900,       # paise  (₹499)
        "credits":  100,
        "features": ["Image detection", "Video detection", "100 screenshots/month"],
    },
    "pro": {
        "name":     "Pro",
        "price":    99900,       # ₹999
        "credits":  300,
        "features": ["Everything in Starter", "Live webcam", "CCTV streams", "Face recognition", "300 screenshots/month"],
    },
    "enterprise": {
        "name":     "Enterprise",
        "price":    199900,      # ₹1999
        "credits":  1000,
        "features": ["Everything in Pro", "Unlimited screenshots", "Priority support", "API access"],
    },
}


# ── Routes ─────────────────────────────────────────────────────────
@router.get("/plans")
def list_plans():
    """Return the plan catalogue. Public endpoint."""
    return {
        "plans": [
            {
                "id":       plan_id,
                "name":     info["name"],
                "price":    info["price"],
                "credits":  info["credits"],
                "features": info["features"],
            }
            for plan_id, info in PLANS.items()
        ],
        "key_id": RAZORPAY_KEY_ID,
    }


@router.post("/create_order")
async def create_order(
    plan_id:      str  = Form(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Create a Razorpay order for the requested plan.
    Returns order details the frontend passes to Razorpay checkout.
    """
    if plan_id not in PLANS:
        raise HTTPException(status_code=400, detail="Invalid plan ID")
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise HTTPException(status_code=503, detail="Razorpay keys not configured — check .env")

    plan = PLANS[plan_id]
    try:
        client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
        order  = client.order.create({
            "amount":   plan["price"],
            "currency": "INR",
            "receipt":  f"ek_{current_user['sub']}_{plan_id}",
            "notes": {
                "user_id": str(current_user["sub"]),
                "email":   current_user.get("email", ""),
                "plan":    plan_id,
            },
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Razorpay error: {exc}")

    return {
        "order_id":   order["id"],
        "amount":     order["amount"],
        "currency":   order["currency"],
        "key_id":     RAZORPAY_KEY_ID,
        "plan_id":    plan_id,
        "plan_name":  plan["name"],
    }


@router.post("/verify_payment")
async def verify_payment(
    razorpay_order_id:   str  = Form(...),
    razorpay_payment_id: str  = Form(...),
    razorpay_signature:  str  = Form(...),
    plan_id:             str  = Form(...),
    current_user:        dict = Depends(get_current_user),
):
    """
    Verify Razorpay HMAC signature and upgrade the user's plan.
    Called by the frontend after a successful payment.
    """
    if not RAZORPAY_KEY_SECRET:
        raise HTTPException(status_code=503, detail="Razorpay keys not configured")

    # HMAC-SHA256 verification
    body     = f"{razorpay_order_id}|{razorpay_payment_id}"
    expected = hmac.new(
        RAZORPAY_KEY_SECRET.encode("utf-8"),
        body.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if expected != razorpay_signature:
        raise HTTPException(status_code=400, detail="Payment signature mismatch — verification failed")

    # Upgrade user plan
    user_id = int(current_user["sub"])
    update_user_plan(user_id, plan_id)
    user = get_user_by_id(user_id)

    return {
        "status":   "success",
        "plan":     plan_id,
        "user": {
            "email": user["email"] if user else "",
            "plan":  plan_id,
        },
    }