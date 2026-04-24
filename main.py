import os
import traceback
from typing import List, Optional
from datetime import datetime

# Clear proxy environment variables BEFORE importing supabase
# HuggingFace Spaces injects these which cause httpx to fail
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('HTTP_PROXIES', None)
os.environ.pop('HTTPS_PROXIES', None)
os.environ.pop('NO_PROXY', None)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Supabase client (service role for full read access)
SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY', '')

def get_supabase():
    """Create Supabase client. Proxy env vars are cleared globally before import."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return None
    
    try:
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except Exception as e:
        print(f"❌ Failed to initialize Supabase client: {e}")
        traceback.print_exc()
        return None


supabase = get_supabase()

app = FastAPI(
    title="Soul Mate ML Match API",
    description="AI-powered matchmaking recommendations using weighted attribute scoring",
    version="1.0.0"
)

# CORS middleware for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    # Check Supabase connection
    supabase_status = "connected" if supabase is not None else "not_configured"
    return {
        "status": "healthy",
        "supabase": supabase_status,
        "timestamp": datetime.utcnow().isoformat()
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    print("🚀 Soul Mate ML API starting...")
    print(f"   Supabase URL: {SUPABASE_URL[:30]}..." if SUPABASE_URL else "   Supabase URL: NOT SET")
    print(f"   Supabase Key: {'SET' if SUPABASE_SERVICE_KEY else 'NOT SET'}")
    print(f"   Port: {os.getenv('PORT', 8000)}")

    if supabase is None:
        print("❌ Supabase client NOT initialized. API will return 500 on /recommend.")
        print("   → Go to Space Settings → Repository secrets and add:")
        print("     - SUPABASE_URL")
        print("     - SUPABASE_SERVICE_KEY")
        return

    # Test connection with a simple query
    try:
        test = supabase.from_('profiles').select('id').limit(1).execute()
        print("✅ Supabase connection verified — ready to serve recommendations")
    except Exception as e:
        print(f"❌ Supabase connection test FAILED: {e}")
        print("   Check your SUPABASE_URL and SUPABASE_SERVICE_KEY")


@app.get("/recommend", response_model=List[MatchResponse])
async def get_recommendations(
    user_id: str = Query(..., description="UUID of the user to match for"),
    top_n: int = Query(10, ge=1, le=50, description="Number of recommendations to return"),
    min_score: float = Query(0.2, ge=0.0, le=1.0, description="Minimum raw score threshold"),
    save: bool = Query(False, description="Whether to persist matches to database")
):
    """Get top-N AI-matched partner recommendations."""
    try:
        # Check Supabase client
        if supabase is None:
            raise HTTPException(
                status_code=500,
                detail="Supabase client not initialized. Check SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables."
            )

        # Fetch current user profile
        try:
            user_resp = supabase.from_('profiles').select('*').eq('id', user_id).maybe_single().execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase query failed (user fetch): {str(e)}")

        if not user_resp.data:
            raise HTTPException(status_code=404, detail=f"User profile not found for ID: {user_id}")

        current = user_resp.data

        # Determine opposite gender
        opposite_gender = 'female' if current.get('gender') == 'male' else (
            'male' if current.get('gender') == 'female' else None
        )
        if not opposite_gender:
            raise HTTPException(status_code=400, detail=f"User gender must be male or female, got: {current.get('gender')}")

        # Fetch candidates
        try:
            candidates_resp = (supabase.from_('profiles')
                               .select('*')
                               .eq('gender', opposite_gender)
                               .neq('id', user_id)
                               .eq('is_active', True)
                               .eq('is_blocked', False)
                               .limit(200)
                               .execute())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase query failed (candidates fetch): {str(e)}")

        if not candidates_resp.data:
            return []

        candidates = candidates_resp.data

        # Rank candidates
        scored = []
        for idx, cand in enumerate(candidates):
            try:
                # Compute weighted score
                components = {
                    'marital': calculate_marital_score(current, cand),
                    'personality': calculate_personality_score(current, cand),
                    'caste': calculate_caste_score(current, cand),
                    'maslak': calculate_maslak_score(current, cand),
                    'age': calculate_age_score(current, cand),
                    'hobbies': calculate_hobbies_score(current, cand),
                }

                raw_score = (
                    components['marital'] * 0.30 +
                    components['personality'] * 0.25 +
                    components['caste'] * 0.20 +
                    components['maslak'] * 0.15 +
                    components['age'] * 0.05 +
                    components['hobbies'] * 0.05
                )

                if raw_score >= min_score:
                    reasons = extract_match_reasons(current, cand, components)
                    scored.append({
                        'candidate': cand,
                        'score': raw_score,
                        'reasons': reasons
                    })
            except Exception as e:
                # Log error but continue with other candidates
                print(f"⚠️  Error scoring candidate {cand.get('id', 'unknown')}: {e}")
                continue

        if not scored:
            return []

        # Sort by score descending
        scored.sort(key=lambda x: x['score'], reverse=True)
        top_matches = scored[:top_n]

        # Build response - return raw score (0-1)
        results = []
        for m in top_matches:
            c = m['candidate']
            results.append(MatchResponse(
                user_id=c['id'],
                score=m['score'],  # raw score 0-1
                reasons=m['reasons']
            ))

        return results

    except HTTPException:
        raise
    except Exception as e:
        # Print full traceback to HuggingFace logs
        print(f"❌ Unexpected error in /recommend:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/debug")
async def debug_ranking(user_id: str = Query(...)):
    """Debug endpoint showing score breakdown for all candidates."""
    try:
        if supabase is None:
            raise HTTPException(status_code=500, detail="Supabase client not configured")

        try:
            user_resp = supabase.from_('profiles').select('*').eq('id', user_id).maybe_single().execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase query failed: {str(e)}")

        if not user_resp.data:
            raise HTTPException(status_code=404, detail="User not found")
        current = user_resp.data

        try:
            candidates_resp = supabase.from_('profiles').select('*').neq('id', user_id).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase query failed: {str(e)}")

        candidates = candidates_resp.data or []

        scored = []
        for cand in candidates:
            try:
                components = {
                    'marital': calculate_marital_score(current, cand),
                    'personality': calculate_personality_score(current, cand),
                    'caste': calculate_caste_score(current, cand),
                    'maslak': calculate_maslak_score(current, cand),
                    'age': calculate_age_score(current, cand),
                    'hobbies': calculate_hobbies_score(current, cand),
                }
                raw = (
                    components['marital'] * 0.30 +
                    components['personality'] * 0.25 +
                    components['caste'] * 0.20 +
                    components['maslak'] * 0.15 +
                    components['age'] * 0.05 +
                    components['hobbies'] * 0.05
                )
                scored.append({
                    'candidate_id': cand['id'],
                    'full_name': cand.get('full_name'),
                    'raw_score': round(raw, 4),
                    **{f'{k}_score': round(v, 3) for k, v in components.items()}
                })
            except Exception as e:
                print(f"⚠️  Error scoring candidate {cand.get('id', 'unknown')}: {e}")
                continue

        scored.sort(key=lambda x: x['raw_score'], reverse=True)
        return {'user': current.get('full_name'), 'candidates': scored[:20]}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error in /debug:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('API_PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
