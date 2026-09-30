# EyeKeeper — Auth + Payments Edition

A full-stack SaaS upgrade on top of the original EyeKeeper AI detection system.
**Zero existing functionality was removed or broken.**

---

## What was added

| Feature | Files |
|---|---|
| JWT authentication (signup / login / logout) | `auth.py`, `database.py` |
| bcrypt password hashing | `auth.py` |
| SQLite user database | `database.py` |
| Per-user file isolation | `app.py` (user dirs) |
| Per-user detection log | `app.py` |
| Per-user target face | `app.py` |
| Per-user session state | `app.py` |
| Razorpay payment plans | `payments.py` |
| Login / signup page | `login.html` |
| Pricing section on landing page | `index.html` |
| User profile widget in playground sidebar | `playground.html` |
| Auth guard on playground (client-side redirect) | `playground.html` |
| Token injected on all API calls + stream URLs | `playground.html` |

---

## File structure after update

```
eyekeeper/
├── app.py               ← Updated backend (auth + payments wired in)
├── auth.py              ← NEW  JWT + bcrypt
├── database.py          ← NEW  SQLite helpers
├── payments.py          ← NEW  Razorpay routes
├── .env                 ← NEW  Secret keys (from .env.example)
├── .env.example         ← Template
├── eyekeeper.db         ← Created automatically on first run
│
├── index.html           ← Updated (pricing section + auth-aware nav)
├── login.html           ← NEW  Dark login/signup page
├── playground.html      ← Updated (auth guard + user widget + token injection)
│
└── data/
    └── users/
        ├── 1/
        │   └── screenshots/
        │       ├── persons/
        │       ├── targets/
        │       └── crops/
        └── 2/
            └── screenshots/ ...
```

---

## Installation

### 1. Install new Python dependencies

```bash
pip install python-jose[cryptography] bcrypt python-dotenv razorpay
```

Full requirements for the new modules:

```
python-jose[cryptography]>=3.3.0
bcrypt>=4.0.1
python-dotenv>=1.0.0
razorpay>=1.4.1
pydantic>=2.0.0    # already required by FastAPI
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
SECRET_KEY=your-32-char-random-secret-here
RAZORPAY_KEY_ID=rzp_test_XXXXXXXXXXXXXXXXXX
RAZORPAY_KEY_SECRET=XXXXXXXXXXXXXXXXXXXXXXXX
```

**Generate a secure SECRET_KEY:**
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

**Get Razorpay test keys:**
1. Sign up at https://dashboard.razorpay.com
2. Go to Settings → API Keys
3. Generate Test Key — copy both Key ID and Key Secret

### 3. Place the files

Copy all 8 files into your project root (same folder as the original `app.py`):

```
app.py          ← replace the original
auth.py         ← new
database.py     ← new
payments.py     ← new
.env            ← new (from .env.example)
login.html      ← new
playground.html ← replace the original
index.html      ← replace the original
```

### 4. Run

```bash
python app.py
```

Open http://localhost:8000 in your browser.

---

## How authentication works

### Flow

```
User visits /playground
    ↓
playground.html checks localStorage for ek_token
    ↓ (no token)
Redirect to /login
    ↓
User fills signup/login form → POST /auth/signup or /auth/login
    ↓
Server returns JWT token + user object
    ↓
Stored in localStorage as ek_token + ek_user
    ↓
Redirect to /playground
    ↓
All API calls include: Authorization: Bearer <token>
All stream URLs include: ?token=<token>
```

### API endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/auth/signup` | Public | Create account |
| POST | `/auth/login`  | Public | Login |
| GET  | `/me`          | Required | Get current user |
| GET  | `/plans`       | Public | List payment plans |
| POST | `/create_order`| Required | Create Razorpay order |
| POST | `/verify_payment` | Required | Verify + upgrade plan |

All other endpoints (`/detect`, `/upload_face`, `/video_feed`, etc.) now require a valid JWT token.

---

## How Razorpay integration works

### Flow

```
User clicks "Upgrade" in playground sidebar
    ↓
POST /create_order { plan_id: "pro" }
    ↓
Razorpay order created → order_id returned
    ↓
Frontend opens Razorpay checkout modal
    ↓
User pays with test card: 4111 1111 1111 1111, CVV: any, Expiry: any future
    ↓
Razorpay calls handler with payment_id + signature
    ↓
POST /verify_payment (HMAC-SHA256 signature check)
    ↓
user.plan updated in SQLite database
    ↓
Plan pill in sidebar updates to new plan
```

### Plans

| Plan ID | Name | Price |
|---|---|---|
| `starter` | Starter | ₹499/month |
| `pro` | Pro | ₹999/month |
| `enterprise` | Enterprise | ₹1,999/month |

---

## Per-user data isolation

Each user's screenshots are saved to:
```
data/users/{user_id}/screenshots/persons/
data/users/{user_id}/screenshots/targets/
data/users/{user_id}/screenshots/crops/
```

Each user also has their own:
- Detection log (in-memory per-user deque)
- Target face (in-memory per-user TargetFace object)
- Session state (video path, prompts, config)

---

## Database schema

```sql
CREATE TABLE users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT    UNIQUE NOT NULL,
    password_hash TEXT    NOT NULL,
    created_at    TEXT    NOT NULL,
    plan          TEXT    NOT NULL DEFAULT 'free',
    credits       INTEGER NOT NULL DEFAULT 0
);
```

---

## Security notes

- Passwords are hashed with bcrypt (cost factor default, ~12 rounds)
- JWT tokens expire after 7 days (configurable in `auth.py`)
- Razorpay payment verification uses HMAC-SHA256 — never trust client-side payment data
- Change `SECRET_KEY` in `.env` before deploying to production
- For production, replace `allow_origins=["*"]` in CORS config with your actual domain
- Stream endpoints accept `?token=` query param because browser `<img>` tags cannot set Authorization headers

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'jose'`**
```bash
pip install python-jose[cryptography]
```

**`ModuleNotFoundError: No module named 'bcrypt'`**
```bash
pip install bcrypt
```

**`ModuleNotFoundError: No module named 'razorpay'`**
```bash
pip install razorpay
```

**Login returns 401 "Invalid email or password"**
Make sure you signed up first. Check `eyekeeper.db` exists in the project root.

**Razorpay "keys not configured" error**
Check your `.env` file has valid `RAZORPAY_KEY_ID` and `RAZORPAY_KEY_SECRET` values and that `python-dotenv` is installed.

**Stream shows 401 Unauthorized**
The playground now passes `?token=<jwt>` in all stream URLs. Make sure you are logged in (have a valid token in localStorage).
