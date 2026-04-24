import os
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Supabase client (service role for full read access)
supabase: Client = create_client(
    os.getenv('SUPABASE_URL', ''),
    os.getenv('SUPABASE_SERVICE_KEY', '')
)

app = FastAPI(
    title="Soul Mate ML Match API",
    description="AI-powered matchmaking recommendations using weighted attribute scoring",
    version="1.0.0"
)

# CORS middleware for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchRecommendation(BaseModel):
    user_id: str = Field(..., alias="user_id")
    full_name: str = Field(..., alias="full_name")
    age: Optional[int] = None
    caste: Optional[str] = None
    sect: Optional[List[str]] = None
    maslak: Optional[str] = None
    marital_status: Optional[str] = None
    personality_traits: Optional[List[str]] = None
    hobbies: Optional[List[str]] = None
    city: Optional[str] = None
    compatibility_score: float = Field(..., ge=0, le=100)
    match_reasons: List[str] = []

    class Config:
        populate_by_name = True
        from_attributes = True


class MatchResponse(BaseModel):
    user_id: str
    score: float = Field(..., ge=0, le=100, description="Normalized 0-100 compatibility score")
    reasons: List[str]


def calculate_marital_score(current, candidate) -> float:
    """Very Important (0.30): Exact match = 1.0, else 0.0"""
    cs = current.get('marital_status')
    ds = candidate.get('marital_status')
    if cs is None or ds is None:
        return 0.5  # Neutral if missing
    return 1.0 if cs == ds else 0.0


def jaccard_similarity(a: List[str], b: List[str]) -> float:
    """Jaccard index for set similarity"""
    set_a, set_b = set(a or []), set(b or [])
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def calculate_personality_score(current, candidate) -> float:
    """Very Important (0.25): Jaccard similarity of personality traits"""
    return jaccard_similarity(
        current.get('personality_traits', []) or [],
        candidate.get('personality_traits', []) or []
    )


def calculate_caste_score(current, candidate) -> float:
    """Important (0.20): Exact caste match = 1.0; same region = 0.7; else 0.0"""
    current_caste = current.get('caste')
    candidate_caste = candidate.get('caste')
    if current_caste is None or candidate_caste is None:
        return 0.5
    if current_caste == candidate_caste:
        return 1.0

    # Partial match via region_caste array first element
    current_rc = current.get('region_caste')
    candidate_rc = candidate.get('region_caste')
    if (current_rc and candidate_rc and
        len(current_rc) > 0 and len(candidate_rc) > 0 and
        current_rc[0] == candidate_rc[0]):
        return 0.7
    return 0.0


def calculate_maslak_score(current, candidate) -> float:
    """Important (0.15): Exact sect match = 1.0; same broad sect = 0.8; else 0.0"""
    # Extract maslak: use maslak field or first element of sect array
    cm = current.get('maslak')
    if not cm:
        sect_list = current.get('sect') or []
        cm = sect_list[0] if len(sect_list) > 0 else None

    dm = candidate.get('maslak')
    if not dm:
        sect_list = candidate.get('sect') or []
        dm = sect_list[0] if len(sect_list) > 0 else None

    if cm is None or dm is None:
        return 0.5
    if cm == dm:
        return 1.0

    # Broad sect grouping
    sunni = {'Barelvi', 'Deobandi', 'Ahle Hadith', 'Salafi', 'Others', 'Sunni'}
    shia = {'Ithna Ashari (12 Imami)', 'Ismaili', 'Bohra', 'Zaidi', 'Others', 'Shia',
            'Shia (Al Tashi)'}

    is_c_sunni = cm in sunni
    is_d_sunni = dm in sunni
    is_c_shia = cm in shia
    is_d_shia = dm in shia

    if (is_c_sunni and is_d_sunni) or (is_c_shia and is_d_shia):
        return 0.8
    return 0.0


def calculate_age_score(current, candidate) -> float:
    """Moderate (0.05): ±3 years = 1.0; ±5 years = 0.8; ±8 years = 0.5; else 0.2"""
    ca = current.get('age', 25)
    da = candidate.get('age', 25)
    diff = abs(ca - da)
    if diff <= 3:
        return 1.0
    if diff <= 5:
        return 0.8
    if diff <= 8:
        return 0.5
    return 0.2


def calculate_hobbies_score(current, candidate) -> float:
    """Moderate (0.05): Jaccard similarity of hobbies"""
    return jaccard_similarity(
        current.get('hobbies', []) or [],
        candidate.get('hobbies', []) or []
    )


def extract_match_reasons(current, candidate, score_components) -> List[str]:
    """Generate human-readable reasons for the match"""
    reasons = []

    # Marital status exact match
    if current.get('marital_status') == candidate.get('marital_status'):
        status = current.get('marital_status', '').capitalize()
        reasons.append(f'Both {status}')

    # Personality traits - top 2
    curr_traits = set(current.get('personality_traits', []) or [])
    cand_traits = set(candidate.get('personality_traits', []) or [])
    common = curr_traits & cand_traits
    if common:
        reasons.append(f'Shares {", ".join(sorted(common)[:2])}')

    # Sect alignment
    cm = current.get('maslak') or (current.get('sect', [None])[0] if current.get('sect') else None)
    dm = candidate.get('maslak') or (candidate.get('sect', [None])[0] if candidate.get('sect') else None)
    if cm and cm == dm:
        reasons.append(f'Same {cm}')

    # Caste
    if current.get('caste') == candidate.get('caste'):
        reasons.append(f'{current.get("caste")} caste')

    # Common hobbies - top 2
    curr_h = set(current.get('hobbies', []) or [])
    cand_h = set(candidate.get('hobbies', []) or [])
    common_h = curr_h & cand_h
    if common_h:
        reasons.append(f'Enjoys {", ".join(sorted(common_h)[:2])}')

    # Age proximity
    ca, da = current.get('age'), candidate.get('age')
    if ca is not None and da is not None:
        diff = abs(ca - da)
        if diff <= 3:
            reasons.append(f'Similar age ({ca} & {da})')
        elif diff <= 5:
            reasons.append('Close in age')

    # City fallback
    if current.get('city') == candidate.get('city') and current.get('city'):
        reasons.append(f'From {current.get("city")}')

    return reasons[:5] if reasons else ['Compatible profile']


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/recommend", response_model=List[MatchResponse])
async def get_recommendations(
    user_id: str = Query(..., description="UUID of the user to match for"),
    top_n: int = Query(10, ge=1, le=50, description="Number of recommendations to return"),
    min_score: float = Query(0.2, ge=0.0, le=1.0, description="Minimum raw score threshold"),
    save: bool = Query(False, description="Whether to persist matches to database")
):
    """Get top-N AI-matched partner recommendations."""
    try:
        # Fetch current user profile
        user_resp = supabase.from_('profiles').select('*').eq('id', user_id).maybe_single().execute()
        if not user_resp.data:
            raise HTTPException(status_code=404, detail="User profile not found")

        current = user_resp.data

        # Determine opposite gender and fetch candidates
        opposite_gender = 'female' if current.get('gender') == 'male' else (
            'male' if current.get('gender') == 'female' else None
        )
        if not opposite_gender:
            raise HTTPException(status_code=400, detail="User gender must be male or female")

        candidates_resp = (supabase.from_('profiles')
                           .select('*')
                           .eq('gender', opposite_gender)
                           .neq('id', user_id)
                           .eq('is_active', True)
                           .eq('is_blocked', False)
                           .limit(200)  # Fetch enough to rank
                           .execute())

        if not candidates_resp.data:
            return []

        candidates = candidates_resp.data

        # Rank candidates
        scored = []
        for cand in candidates:
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

        # Sort by score descending
        scored.sort(key=lambda x: x['score'], reverse=True)
        top_matches = scored[:top_n]

        # Normalize scores to 0-100 range for display
        if top_matches:
            max_score = top_matches[0]['score'] if top_matches[0]['score'] > 0 else 1.0
            for m in top_matches:
                m['norm_score'] = round((m['score'] / max_score) * 100, 1)

        # Optionally persist matches to DB
        if save and top_matches:
            # In production, batch insert into matches table
            pass

        # Build response
        results = []
        for m in top_matches:
            c = m['candidate']
            results.append(MatchResponse(
                user_id=c['id'],
                score=m.get('norm_score', m['score'] * 100),
                reasons=m['reasons']
            ))

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug")
async def debug_ranking(user_id: str = Query(...)):
    """Debug endpoint showing score breakdown for all candidates."""
    try:
        user_resp = supabase.from_('profiles').select('*').eq('id', user_id).maybe_single().execute()
        if not user_resp.data:
            raise HTTPException(status_code=404, detail="User not found")
        current = user_resp.data

        candidates_resp = supabase.from_('profiles').select('*').neq('id', user_id).execute()
        candidates = candidates_resp.data or []

        scored = []
        for cand in candidates:
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

        scored.sort(key=lambda x: x['raw_score'], reverse=True)
        return {'user': current.get('full_name'), 'candidates': scored[:20]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('API_PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
