# EyeKeeper v4 — Setup & Run Guide

## What Was Added

| Feature | File(s) |
|---|---|
| User auth (signup / login / JWT) | `auth.py`, `database.py` |
| SQLite database | `database.py` → `eyekeeper.db` |
| Per-user data isolation | `app.py` → `data/users/user_<id>/` |
| Login page | `templates/login.html` |
| Protected playground | `templates/playground.html` |
| Razorpay payments | `payments.py` |
| Updated landing page with pricing | `templates/index.html` |

---

## 1. Install New Dependencies

```bash
pip install python-jose[cryptography] passlib[bcrypt] razorpay
```

Your existing requirements (fastapi, uvicorn, opencv-python, ultralytics, etc.) are unchanged.

---

## 2. File Structure After Update

```
eyekeeper/
├── app.py                  ← updated (auth + user isolation)
├── auth.py                 ← NEW
├── database.py             ← NEW
├── payments.py             ← NEW
├── eyekeeper.db            ← auto-created on first run
├── yoloe-26s-seg.pt        ← your existing model (unchanged)
│
├── templates/              ← put all HTML files here
│   ├── index.html
│   ├── login.html
│   └── playground.html
│
└── data/
    └── users/
        ├── user_1/
        │   ├── screenshots/
        │   │   ├── persons/
        │   │   ├── targets/
        │   │   └── crops/
        │   └── logs/
        └── user_2/
            └── ...
```

---

## 3. Configure Razorpay (Test Mode)

1. Sign up free at **https://razorpay.com**
2. Go to **Settings → API Keys → Generate Test Key**
3. Set environment variables (or edit `payments.py` directly for dev):

```bash
export RAZORPAY_KEY_ID="rzp_test_XXXXXXXXXXXX"
export RAZORPAY_KEY_SECRET="XXXXXXXXXXXXXXXXXXXX"
```

Or on Windows:
```cmd
set RAZORPAY_KEY_ID=rzp_test_XXXXXXXXXXXX
set RAZORPAY_KEY_SECRET=XXXXXXXXXXXXXXXXXXXX
```

> **Test card:** `4111 1111 1111 1111` · Expiry: any future date · CVV: any 3 digits

---

## 4. (Optional) Change JWT Secret

In production, set a strong secret:

```bash
export EK_SECRET_KEY="your-very-long-random-secret-here"
```

---

## 5. Run the Server

```bash
python app.py
```

Or with uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then open: **http://localhost:8000*

---

## 6. First Use Flow
*
1. Open **http://localhost:8000** — landing page with pricing
2. Click **🎮 Open Playground** or **Sign In**
3. Create an account on `/login`
4. You're redirected to `/playground` — fully functional with your data isolated
5. To test payments: click **⚡ Upgrade Plan** in the sidebar → choose a plan → use Razorpay test card

---

## 7. API Endpoints Summary

### Auth (public)
| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/signup` | Create account |
| POST | `/auth/login` | Login, get JWT |
| GET  | `/auth/me` | Get current user info |

### Detection (requires JWT)
All original endpoints work exactly as before — just add `Authorization: Bearer <token>` header.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/video_feed` | Video stream |
| GET | `/webcam_feed` | Webcam stream |
| POST | `/detect` | Single image detection |
| POST | `/upload_video` | Upload + stream video |
| POST | `/upload_face` | Set target face |
| GET/DELETE | `/target_face` | Get/clear target face |
| GET/DELETE | `/person_log` | Detection log |
| GET | `/screenshots` | List user screenshots |
| GET | `/screenshots/{folder}/{file}` | Serve screenshot |
| DELETE | `/screenshots/{folder}/{file}` | Delete screenshot |

### Payments (requires JWT)
| Method | Endpoint | Description |
|---|---|---|
| GET | `/payments/plans` | List all plans |
| POST | `/payments/create_order` | Create Razorpay order |
| POST | `/payments/verify_payment` | Verify & activate plan |

---

## 8. Security Notes

- Passwords are hashed with **bcrypt** (cost factor 12) — never stored in plain text
- JWT tokens expire after **7 days** — change `ACCESS_TOKEN_EXPIRE_MINUTES` in `auth.py`
- Payment signatures verified with **HMAC-SHA256** before any plan upgrade
- Path traversal protection on all screenshot endpoints
- Each user can only access their own data folders
- Stream endpoints accept token via `?token=` query param (required for `<img src>` elements)

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: jose` | `pip install python-jose[cryptography]` |
| `ModuleNotFoundError: passlib` | `pip install passlib[bcrypt]` |
| `ModuleNotFoundError: razorpay` | `pip install razorpay` |
| Login redirects back to login | Token expired or invalid — sign out and back in |
| Razorpay button does nothing | Check browser console — likely key ID not set |
| Stream shows 401 | Token missing from stream URL — check `streamUrl()` helper |
| Old screenshots missing | Old data was in `screenshots/` — new data goes in `data/users/user_<id>/` |
