# Soul Mate ML Recommendation Service

A dedicated FastAPI microservice for AI-powered match recommendations. Uses the weighted attribute algorithm (marital status 30%, personality traits 25%, caste 20%, sect 15%, age 5%, hobbies 5%) to generate compatibility scores.

## Setup

```bash
# Navigate to the ml-service directory
cd ml-service

# Create virtual environment
python -m venv venv

# Activate venv
# On Windows PowerShell:
# venv\Scripts\Activate.ps1
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env` and set your Supabase credentials:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
API_PORT=8000
DEBUG=false
```

**Get your service key:**
- Go to Supabase Dashboard → Project Settings (gear icon) → **API**
- Under "Service Role Key" click "Show" and copy the key (starts with `eyJ...`)
- ⚠️ This key has admin privileges. Never expose it in client-side Flutter code. It stays **server-side only** in the ML service.

**Optional: Database URL** (needed only for direct Postgres connection if Supabase client hits rate limits):
- Supabase Dashboard → Project Settings → Database → Connection String → "URI format". Click "Show password" and replace `[YOUR-PASSWORD]`.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs will be available at `http://localhost:8000/docs`.

## Endpoints

- `GET /recommend?user_id=<uuid>&top_n=10&min_score=0.5` — Get top-N AI-matched partners with compatibility scores and reasons.
- `GET /health` — Health check.

## Deployment (Render / Railway / Fly.io)

### Quick Deploy to Render (free tier available)

1. Push the `ml-service/` folder to GitHub.
2. Go to [Render Dashboard](https://dashboard.render.com) → New → Web Service.
3. Connect your repository.
4. Configure:
   - **Name**: `soulmate-ml-api` (or your choice)
   - **Environment**: Python 3.11+
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add Environment Variables (from `.env`):
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_KEY`
   - `API_PORT=8000`
   - `DEBUG=false`
6. Click **Create Web Service**.
7. After deployment, your URL will be `https://soulmate-ml-api.onrender.com`.

### Deploy to Railway

1. Push to GitHub.
2. Go to [Railway Dashboard](https://railway.app) → New Project → Deploy from GitHub.
3. Set service type → Python.
4. Set Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add variables under Settings → Variables.
6. Deploy.

### Update Flutter App

Once deployed, edit `lib/core/constants/app_constants.dart`:

```dart
static const String recommendationApiUrl =
    'https://soulmate-ml-api.onrender.com'; // Your deployed URL
```

**Important:** In production, consider restricting CORS `allow_origins` in `main.py` to your actual app domain instead of `["*"]`.

## API Testing

### Local run with Swagger UI
```bash
uvicorn main:app --reload
```
Visit `http://localhost:8000/docs` to test `/recommend` endpoint interactively.

### CLI test
```bash
curl "http://localhost:8000/recommend?user_id=YOUR_USER_UUID&top_n=5"
```

### Response format
```json
[
  {
    "user_id": "uuid-here",
    "score": 92.5,
    "reasons": ["Both Single", "Shares kind, creative traits", "Same Barelvi", "Barelvi caste", "Enjoys reading, travel"]
  }
]
```

## Troubleshooting

**"ML API not configured"** in Flutter app → Set `recommendationApiUrl` in `AppConstants.dart` to your deployed URL and hot-restart.

**401 Unauthorized from Supabase** → Verify `SUPABASE_SERVICE_KEY` is correct and has `service_role` permissions.

**Empty results** → Ensure your test user has a complete profile (age, caste, sect, marital_status, hobbies, personality_traits filled). The algorithm requires these fields.

**Incorrect sect values** → Make sure your database allows the `maslak` values you are sending (see `database/fix_maslak_constraint.sql`).

**CORS errors** → Update `app.add_middleware(CORSMiddleware...)` origins to your actual domain in production.

## Algorithm Reference

| Attribute             | Weight | Scoring Method                                |
|-----------------------|--------|-----------------------------------------------|
| Marital Status        | 0.30   | Exact match = 1.0, else 0.0                   |
| Personality Traits    | 0.25   | Jaccard similarity (intersection/union)       |
| Caste                 | 0.20   | Exact = 1.0; same region = 0.7; else 0.0     |
| Sect / Maslak         | 0.15   | Exact = 1.0; Sunni/Shia alignment = 0.8      |
| Age                   | 0.05   | ±3y = 1.0, ±5y = 0.8, ±8y = 0.5, else 0.2   |
| Hobbies / Interests   | 0.05   | Jaccard similarity                            |

Scores are weighted sum (0–1) and then normalized to 0–100 relative to the top match.
