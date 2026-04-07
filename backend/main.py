"""
Yatrika FastAPI Backend
Exposes the RAG travel recommendation engine as REST endpoints.
Run: uvicorn main:app --reload --port 8000
"""

import os, io, math, time, requests, json, asyncio, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [YATRIKA] %(message)s')
log = logging.getLogger('yatrika')
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()
OWM_API_KEY = os.getenv("OWM_API_KEY")
print("Loaded OWM KEY:", OWM_API_KEY)
# ── Config ────────────────────────────────────────────────────────────────
EMBED_MODEL  = "all-MiniLM-L6-v2"
FAISS_PATH   = "./faiss_yatrika_v3"
OWM_URL      = "https://api.openweathermap.org/data/2.5/forecast"
GEOCODE_URL  = "https://nominatim.openstreetmap.org/search"
DATASET_PATH = "./destinations.csv"
TOP_K        = 5
FAISS_FETCH  = 30   # candidates passed to LLM reranker (wider pool → better recall)
_last_llm_success = {"time": None, "type": None}  # tracks last successful LLM call

from contextlib import asynccontextmanager

async def _keep_hf_warm():
    """Ping HuggingFace every 4 minutes to prevent model cold-starts."""
    await asyncio.sleep(10)          # wait for server to fully start
    while True:
        if HF_API_KEY:
            try:
                requests.post(
                    HF_API_URL,
                    headers={"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"},
                    json={"model": HF_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                    timeout=10
                )
            except Exception:
                pass  # silently ignore — just a warmup ping
        await asyncio.sleep(240)     # ping every 4 minutes

@asynccontextmanager
async def lifespan(app):
    if HF_API_KEY:
        log.info(f"HF_API_KEY loaded: {HF_API_KEY[:8]}... model={HF_MODEL}")
        asyncio.create_task(_keep_hf_warm())
    else:
        log.warning("HF_API_KEY not set — LLM disabled, using static fallback")
    yield

app = FastAPI(title="Yatrika API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Radius table ──────────────────────────────────────────────────────────
# All distances are ONE-WAY maximums.
# 1-day trips are round-trip constrained: user must travel there AND back same day.
RADIUS = {
    (1,"any"):   (250,  "day-trip ≤250 km one-way (round-trip)",   "car/bus"),
    (1,"drive"): (250,  "day-trip ≤250 km one-way (round-trip)",   "car"),
    (1,"fly"):   (450,  "day-trip flight ≤450 km one-way",         "flight"),
    (2,"any"):   (500,  "weekend ≤500 km one-way",                 "train/bus"),
    (2,"drive"): (450,  "weekend drive ≤450 km one-way",           "car"),
    (2,"fly"):   (1500, "short-haul flight ≤1500 km",              "flight"),
    (3,"any"):   (900,  "3-day trip ≤900 km one-way",              "train/flight"),
    (3,"drive"): (650,  "road-trip ≤650 km one-way",               "car"),
    (3,"fly"):   (2500, "domestic flight ≤2500 km",                "flight"),
    (4,"any"):   (1500, "≤1500 km one-way",                        "train/flight"),
    (4,"drive"): (1000, "long road-trip ≤1000 km",                 "car"),
    (4,"fly"):   (99999,"all India",                               "flight"),
    (5,"any"):   (99999,"all India",                               "any"),
    (5,"drive"): (1500, "extended road-trip ≤1500 km",             "car"),
    (5,"fly"):   (99999,"all India",                               "flight"),
}
def _bucket(n): return 1 if n<=1 else 2 if n==2 else 3 if n==3 else 4 if n<=5 else 5
def get_radius(days, transport="any"):
    k = (_bucket(days), transport)
    return RADIUS.get(k, RADIUS[(_bucket(days),"any")])

# ── State coords fallback ─────────────────────────────────────────────────
STATE_COORDS = {
    "andhra pradesh":              (15.9129, 79.7400),
    "arunachal pradesh":           (28.2180, 94.7278),
    "assam":                       (26.2006, 92.9376),
    "bihar":                       (25.0961, 85.3131),
    "chhattisgarh":                (21.2787, 81.8661),
    "goa":                         (15.2993, 74.1240),
    "gujarat":                     (22.2587, 71.1924),
    "haryana":                     (29.0588, 76.0856),
    "himachal pradesh":            (31.1048, 77.1734),
    "jharkhand":                   (23.6102, 85.2799),
    "jammu and kashmir":           (33.7782, 76.5762),
    "karnataka":                   (15.3173, 75.7139),
    "kerala":                      ( 9.9312, 76.2673),
    "madhya pradesh":              (22.9734, 78.6569),
    "maharashtra":                 (19.7515, 75.7139),
    "manipur":                     (24.6637, 93.9063),
    "meghalaya":                   (25.4670, 91.3662),
    "mizoram":                     (23.1645, 92.9376),
    "nagaland":                    (26.1584, 94.5624),
    "odisha":                      (20.9517, 85.0985),
    "punjab":                      (31.1471, 75.3412),
    "rajasthan":                   (27.0238, 74.2179),
    "sikkim":                      (27.5330, 88.5122),
    "tamil nadu":                  (11.1271, 78.6569),
    "telangana":                   (18.1124, 79.0193),
    "tripura":                     (23.9408, 91.9882),
    "uttar pradesh":               (26.8467, 80.9462),
    "uttarakhand":                 (30.0668, 79.0193),
    "west bengal":                 (22.9868, 87.8550),
    "delhi":                       (28.6139, 77.2090),
    "chandigarh":                  (30.7333, 76.7794),
    "andaman and nicobar islands": (11.7401, 92.6586),
    "puducherry":                  (11.9416, 79.8083),
    "ladakh":                      (34.1526, 77.5771),
}

# ── HuggingFace Router API config (OpenAI-compatible) ───────────────────
HF_API_KEY  = os.environ.get("HF_API_KEY", "")
HF_MODEL    = "Qwen/Qwen2.5-72B-Instruct:novita"
HF_API_URL  = "https://router.huggingface.co/v1/chat/completions"

def _hf_call(system: str, user: str, max_tokens: int = 250) -> str:
    """
    Call HuggingFace Router API (OpenAI-compatible format).
    Uses Qwen2.5-72B — free, powerful, excellent instruction following.
    """
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    log.info(f"HF call attempt | model={HF_MODEL} | tokens={max_tokens}")
    for attempt in range(3):
        try:
            r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=45)
            if r.status_code == 503:
                log.warning(f"HF 503 model loading — retrying (attempt {attempt+1}/3)")
                time.sleep(20)
                continue
            r.raise_for_status()
            data = r.json()
            result = data["choices"][0]["message"]["content"].strip()
            log.info(f"HF call SUCCESS | preview: {result[:80]}")
            return result
        except Exception as e:
            log.warning(f"HF call attempt {attempt+1} failed: {e}")
            if attempt == 2:
                raise
            time.sleep(5)
    raise ValueError("HF call failed after 3 retries")

# ── Emotion expansions (static fallback) ────────────────────────────────
EMOTION_EXPANSIONS = {
    "burned out":   "exhausted burnout rest healing peace quiet solitude nature recovery",
    "burnout":      "exhausted rest healing peace quiet solitude nature recovery",
    "stressed":     "calm tranquil peaceful serene quiet escape retreat healing",
    "anxious":      "calm safe grounded peaceful tranquil simple slow",
    "sad":          "healing comfort warmth gentle beauty uplifting soothing",
    "lonely":       "community warmth human connection culture vibrant festive",
    "bored":        "adventure exciting new discovery thrill stimulating explore",
    "excited":      "adventure thrill adrenaline vibrant energetic explore dynamic",
    "romantic":     "romance intimate love couples scenic beautiful serene mist",
    "adventurous":  "thrill adrenaline trekking extreme outdoor sport challenge",
    "spiritual":    "meditation temple monastery sacred divine pilgrimage inner peace",
    "creative":     "artistic culture heritage beauty photography scenic inspiring",
    "nostalgic":    "heritage history old charm vintage colonial classic timeless",
    "curious":      "discovery culture history unique offbeat tribal authentic",
    "peaceful":     "serene quiet calm tranquil slow nature solitude",
    "happy":        "joyful vibrant beach sunshine tropical warm blissful",
    "free":         "open vast sky nature solitude wild untamed",
    "disconnected": "remote isolated offbeat no crowd solitude wilderness",
    "overwhelmed":  "slow simple quiet minimal peace healing nature retreat",
}

def _static_expand(raw: str, avoid: str = "", activities: str = "") -> str:
    """Fallback: dictionary-based emotion expansion (no API needed)."""
    q = raw.lower()
    extras = [exp for kw, exp in EMOTION_EXPANSIONS.items() if kw in q]
    enriched = f"I am feeling: {raw}. I need a place that is: {raw}. Emotional resonance: {raw}"
    if extras:
        enriched += ". " + " ".join(extras)
    if activities and activities.lower().strip() not in ("skip", "none", "no", "-", ""):
        enriched += f". I want: {activities}"
    if avoid and avoid.lower().strip() not in ("skip", "none", "no", "-", ""):
        enriched += f". NOT: {avoid}"
    return enriched

def expand_emotion_query(raw: str, avoid: str = "", activities: str = "") -> str:
    """
    Keyword-based query enrichment using the LLM.
    Passes avoid and activities directly into the prompt so the LLM
    steers the FAISS embedding AWAY from unwanted landscape types and
    TOWARD desired ones — fixing it at the vector search level.
    Falls back to static dictionary if HF_API_KEY is not set or call fails.
    """
    if not HF_API_KEY:
        log.warning("expand_emotion_query: no HF_API_KEY — using static fallback")
        return _static_expand(raw, avoid, activities)
    log.info(f"expand_emotion_query: calling LLM for: {raw[:50]}")
    try:
        avoid_instruction = ""
        if avoid and avoid.lower().strip() not in ("skip", "none", "no", "-", ""):
            avoid_instruction = (
                f"\nCRITICAL: The user wants to AVOID: '{avoid}'. "
                f"Do NOT output any keywords related to this. "
                f"For example if they avoid hills, do not output: hill, hills, mountain, mountains, "
                f"trekking, altitude, ghats, valley, peak, highland, scenic viewpoint."
            )
        activities_instruction = ""
        if activities and activities.lower().strip() not in ("skip", "none", "no", "-", ""):
            activities_instruction = (
                f"\nThe user WANTS activities like: '{activities}'. "
                f"Include keywords that reflect these preferences."
            )
        system = (
            "You are a travel keyword extractor for India destinations. "
            "Given an emotional state, output 8-12 single travel-related keywords "
            "that describe the ideal destination mood, landscape, and atmosphere. "
            "Use words like: peaceful, solitude, beach, forest, backwaters, "
            "healing, vibrant, quiet, spiritual, romantic, heritage, offbeat. "
            f"{avoid_instruction}"
            f"{activities_instruction}"
            "\nOutput ONLY the keywords separated by spaces. No sentences, no punctuation."
        )
        keywords = _hf_call(system, raw, max_tokens=60)
        keywords = keywords.replace("[INST]", "").replace("[/INST]", "")
        keywords = " ".join(w.strip('.,;:!?"\'') for w in keywords.split() if len(w) > 2)
        static = _static_expand(raw, avoid, activities)
        _last_llm_success["time"] = datetime.now().strftime("%H:%M:%S")
        _last_llm_success["type"] = "keyword_extraction"
        return f"{static} {keywords}"
    except Exception as e:
        log.warning(f"expand_emotion_query FAILED: {type(e).__name__}: {e}")
        return _static_expand(raw, avoid, activities)

def narrate_results(emotion: str, results: list) -> list:
    """
    HuggingFace-powered result narration.
    Mistral-7B writes a personalised why-this-matches-you sentence per destination.
    Falls back gracefully (no why_match field) if API key missing or call fails.
    """
    if not HF_API_KEY or not results:
        return results
    try:
        dest_lines = "\n".join(
            f"{i+1}. {r['destination']}, {r['state']} — {r['vibe']} — {r['description'][:100]}"
            for i, r in enumerate(results)
        )
        system = (
            "You are Yatrika, an India travel concierge. "
            "You will be given a numbered list of destinations and a user's emotional state. "
            "Write exactly one sentence per destination (max 20 words) explaining why THAT SPECIFIC destination suits the user. "
            "STRICT RULES: "
            "1. Only mention the destination name that is given to you for that number. Do NOT invent or substitute other place names. "
            "2. Base your sentence only on the vibe and description provided — do not add facts not in the input. "
            "3. Output ONLY a valid JSON array of strings with exactly as many items as destinations given. "
            "4. No preamble, no explanation, no markdown — just the raw JSON array."
        )
        user = (
            f"User feeling: {emotion}\n\n"
            f"Destinations (write one sentence for each, using ONLY these place names):\n{dest_lines}\n\n"
            f"Output a JSON array with exactly {len(results)} strings, one per destination above."
        )
        raw_text = _hf_call(system, user, max_tokens=400)
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        # Find JSON array in response
        start = raw_text.find("[")
        end   = raw_text.rfind("]") + 1
        if start != -1 and end > start:
            narrations = json.loads(raw_text[start:end])
            other_names = [r['destination'] for r in results]
            for i, r in enumerate(results):
                if i < len(narrations):
                    narration = str(narrations[i])
                    # If it mentions another destination's name instead of this one,
                    # discard it — better to show nothing than wrong info
                    others = [n for j, n in enumerate(other_names) if j != i]
                    mentions_wrong = any(o.lower() in narration.lower() for o in others)
                    if mentions_wrong and r['destination'].lower() not in narration.lower():
                        log.warning(f"narrate_results: hallucination in slot {i+1} — discarding")
                    else:
                        r["why_match"] = narration
        _last_llm_success["time"] = datetime.now().strftime("%H:%M:%S")
        _last_llm_success["type"] = "narration"
        return results
    except Exception as e:
        log.warning(f"narrate_results FAILED: {type(e).__name__}: {e}")
        return results

# ── Startup: load ML models once ─────────────────────────────────────────
_emb = None
_vs  = None

def get_emb():
    global _emb
    if _emb is None:
        _emb = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _emb

def get_vs():
    global _vs
    if _vs is None:
        emb = get_emb()
        idx = os.path.join(FAISS_PATH, "index.faiss")
        if os.path.exists(idx):
            _vs = FAISS.load_local(FAISS_PATH, emb, allow_dangerous_deserialization=True)
        else:
            df = pd.read_csv(DATASET_PATH)
            docs = []
            for _, row in df.iterrows():
                content = (
                    f"A destination for someone feeling {row['emotion_tags']}. "
                    f"If you feel {row['vibe']}, visit {row['destination']} in {row['state']}. "
                    f"Emotional Vibe: {row['vibe']}. "
                    f"Emotional Tags: {row['emotion_tags']}. "
                    f"Perfect for someone feeling: {row['emotion_tags']}. "
                    f"This place resonates with: {row['vibe']}. "
                    f"Destination: {row['destination']}, {row['state']}. "
                    f"Landscape: {row['landscape']}. "
                    f"Best Season: {row['best_season']}. "
                    f"Activities: {row['activities']}. "
                    f"Description: {row['description']}"
                )
                docs.append(Document(page_content=content, metadata={
                    "destination": str(row["destination"]),
                    "state":       str(row["state"]),
                    "region":      str(row.get("region", "India")),
                    "lat":         float(row["lat"]),
                    "lon":         float(row["lon"]),
                    "vibe":        str(row["vibe"]),
                    "best_season": str(row["best_season"]),
                    "activities":  str(row["activities"]),
                    "nearby":      str(row["nearby_attractions"]),
                    "landscape":   str(row["landscape"]),
                }))
            _vs = FAISS.from_documents(docs, emb)
            os.makedirs(FAISS_PATH, exist_ok=True)
            _vs.save_local(FAISS_PATH)
    return _vs

_df = None
def load_df():
    global _df
    if _df is None:
        _df = pd.read_csv(DATASET_PATH)
    return _df

# ── Core functions ────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def _tokenize(text: str) -> list:
    cleaned = []
    for ch in (text or "").lower():
        cleaned.append(ch if ch.isalnum() else " ")
    return [tok for tok in "".join(cleaned).split() if len(tok) >= 3]


def _doc_text(doc) -> str:
    md = getattr(doc, "metadata", {}) or {}
    parts = [
        md.get("destination", ""),
        md.get("state", ""),
        md.get("vibe", ""),
        md.get("landscape", ""),
        md.get("activities", ""),
        getattr(doc, "page_content", ""),
    ]
    return " ".join(str(p) for p in parts if p)


def _keyword_overlap_score(text: str, tokens: list, phrase_bonus_terms: list = None) -> float:
    if not tokens:
        return 0.0
    hay = text.lower()
    token_hits = sum(1 for tok in set(tokens) if tok in hay)
    score = token_hits / max(len(set(tokens)), 1)
    if phrase_bonus_terms:
        phrase_hits = sum(1 for term in phrase_bonus_terms if term and term.lower() in hay)
        score += 0.12 * phrase_hits
    return score


def heuristic_rerank(candidates: list, query: str, enriched_query: str,
                     olat: float, olon: float, max_km: float,
                     origin_state: str = None, transport: str = "any",
                     avoid: str = "", activities: str = "", k: int = TOP_K) -> list:
    """
    Deterministic reranker used before/without the LLM.
    It balances semantic fit, distance, home-state fit, and explicit preferences
    so results stay sensible even when the external reranker is unavailable.
    """
    if not candidates:
        return []

    query_tokens = _tokenize(f"{query} {enriched_query}")
    activity_tokens = _tokenize(activities)
    avoid_tokens = _tokenize(avoid)
    same_state_name = (origin_state or "").lower().strip()
    scored = []

    for idx, (doc, faiss_score) in enumerate(candidates):
        md = doc.metadata
        dist_km = haversine(olat, olon, float(md["lat"]), float(md["lon"]))
        text = _doc_text(doc)

        semantic_score = 1.0 / (1.0 + max(float(faiss_score), 0.0))
        lexical_score = _keyword_overlap_score(text, query_tokens)
        activity_score = _keyword_overlap_score(text, activity_tokens, phrase_bonus_terms=activity_tokens[:4])
        avoid_penalty = _keyword_overlap_score(text, avoid_tokens) if avoid_tokens else 0.0

        within_cap = max(max_km, 1.0)
        distance_score = max(0.0, 1.0 - min(dist_km, within_cap) / within_cap)

        state = str(md.get("state", "")).lower().strip()
        same_state_bonus = 0.0
        if same_state_name:
            if transport == "fly":
                same_state_bonus = -0.18 if state == same_state_name else 0.08
            else:
                same_state_bonus = 0.18 if state == same_state_name else 0.0

        locality_bonus = 0.0
        if transport == "drive":
            locality_bonus = 0.25 * distance_score
        elif transport != "fly":
            locality_bonus = 0.15 * distance_score

        score = (
            1.25 * semantic_score +
            0.90 * lexical_score +
            0.55 * activity_score +
            locality_bonus +
            same_state_bonus -
            0.45 * avoid_penalty -
            0.015 * idx
        )

        scored.append((score, (doc, faiss_score)))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in scored[:max(len(candidates), k)]]



def llm_rerank(emotion: str, candidates: list, k: int = TOP_K,
               avoid: str = "", activities: str = "") -> list:
    """
    LLM Re-ranking: FAISS returns top-20 by vector distance.
    Mistral reads all candidates and re-orders them based on true
    emotional and contextual fit — not just cosine similarity.
    User preferences (avoid / activities) are baked into the prompt
    so the model handles them with full linguistic understanding.
    Falls back to original FAISS order if API unavailable.
    """
    if not HF_API_KEY:
        log.warning(f"llm_rerank: skipped — no API key")
        return candidates[:k]
    if len(candidates) <= 1:
        log.info(f"llm_rerank: only {len(candidates)} candidate(s) — skipping rerank")
        return candidates[:k]
    log.info(f"llm_rerank: re-ranking {len(candidates)} candidates for: {emotion[:50]}")
    try:
        candidate_lines = "\n".join(
            f"{i+1}. {doc.metadata['destination']}, {doc.metadata['state']} "
            f"| Vibe: {doc.metadata['vibe']} "
            f"| Landscape: {doc.metadata['landscape']} "
            f"| Activities: {doc.metadata['activities'][:80]}"
            for i, (doc, _) in enumerate(candidates)
        )

        # Build preference instructions for the prompt
        pref_lines = []
        if avoid and avoid.lower() not in ("skip", "none", "no", "-"):
            pref_lines.append(
                f"HARD RULE — EXCLUDE any destination whose landscape, activities, or vibe "
                f"matches: {avoid}. These must NOT appear in your output regardless of emotional fit."
            )
        if activities and activities.lower() not in ("skip", "none", "no", "-"):
            pref_lines.append(f"PREFER destinations that offer: {activities}")
        pref_block = ("\n".join(pref_lines) + "\n") if pref_lines else ""

        system = (
            f"You are a travel expert ranking India destinations by emotional and personal fit. "
            f"You MUST respect all HARD RULEs — they override emotional fit entirely. "
            f"You will receive up to {FAISS_FETCH} candidate destinations. "
            f"Return ONLY a JSON array of {k} integers (1-based indices) ranking the best "
            f"destinations for the user. Example: [3, 1, 2]. No explanation."
        )
        user = (
            f"User feeling: {emotion}\n"
            f"{pref_block}"
            f"\nDestinations:\n{candidate_lines}"
        )
        raw = _hf_call(system, user, max_tokens=40)
        log.info(f"llm_rerank raw response: {raw[:100]}")
        raw = raw.replace("```json","").replace("```","").strip()
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end <= start:
            log.warning(f"llm_rerank: no JSON array found in response")
            return candidates[:k]
        indices = json.loads(raw[start:end])
        log.info(f"llm_rerank: got indices {indices}")
        # Validate and convert to 0-based
        valid = [i-1 for i in indices if isinstance(i, int) and 1 <= i <= len(candidates)]
        if len(valid) < k:
            # Fill missing slots with remaining candidates in FAISS order
            used = set(valid)
            for i in range(len(candidates)):
                if i not in used:
                    valid.append(i)
                if len(valid) >= k:
                    break
        reranked = [candidates[i] for i in valid[:k]]
        _last_llm_success["time"] = datetime.now().strftime("%H:%M:%S")
        _last_llm_success["type"] = "reranking"
        return reranked
    except Exception:
        return candidates[:k]


def llm_narrate_itinerary(emotion: str, itin: list, weather: dict, days: int) -> list:
    """
    LLM Itinerary Narration: replaces dry activity lists with a warm,
    emotionally resonant day-by-day narrative written by Mistral.
    Adds a 'narrative' field to each stop. Falls back gracefully.
    """
    if not HF_API_KEY or not itin:
        log.warning(f"llm_narrate_itinerary: skipped (key={bool(HF_API_KEY)})")
        return itin
    log.info(f"llm_narrate_itinerary: writing narrative for {len(itin)} stops")
    try:
        stops = "\n".join(
            f"Stop {i+1}: {s['destination']}, {s['state']} — {s['days']} day(s) "
            f"| {s['landscape']} | Activities: {s['activities'][:100]}"
            for i, s in enumerate(itin)
        )
        weather_note = (
            f"Weather: {weather.get('temp_avg',25)}°C avg, "
            f"{weather.get('conditions',['pleasant'])[0]}, "
            f"humidity {weather.get('humidity',60)}%"
        )
        system = (
            "You are Yatrika, an India travel writer. "
            "You will be given a list of itinerary stops with their destination name, landscape, and activities. "
            "Write a short paragraph (3-4 sentences) for EACH stop. "
            "STRICT RULES: "
            "1. Only write about the exact destination name given for each stop. NEVER mention any other place. "
            "2. Only reference landscape and activities that are explicitly listed in the stop data. Do not invent attractions, landmarks, or experiences. "
            "3. Weave in the user's emotional state naturally. "
            "4. Output ONLY a valid JSON array of strings, one paragraph per stop, in the same order as the stops. "
            "5. No preamble, no explanation, no markdown — just the raw JSON array."
        )
        user = (
            f"User feeling: {emotion}\n"
            f"Trip: {days} days | {weather_note}\n\n"
            f"Stops (write ONLY about these places, in this order):\n{stops}\n\n"
            f"Output a JSON array with exactly {len(itin)} paragraph strings."
        )
        raw = _hf_call(system, user, max_tokens=500)
        log.info(f"llm_narrate_itinerary raw response: {raw[:200]}")
        raw = raw.replace("```json","").replace("```","").strip()
        # Try JSON array first
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start != -1 and end > start:
            try:
                narratives = json.loads(raw[start:end])
                for i, stop in enumerate(itin):
                    if i < len(narratives):
                        narrative = str(narratives[i])
                        # Sanity check: if the narrative doesn't mention this stop's
                        # destination at all but mentions a different stop's destination,
                        # it's hallucinated — fall back to the description field
                        other_stops = [s['destination'] for j, s in enumerate(itin) if j != i]
                        mentions_wrong = any(
                            other.lower() in narrative.lower() for other in other_stops
                        )
                        if mentions_wrong and stop['destination'].lower() not in narrative.lower():
                            log.warning(f"llm_narrate_itinerary: hallucination detected for stop {i+1} ({stop['destination']}) — using description fallback")
                            stop["narrative"] = stop.get("description", "")
                        else:
                            stop["narrative"] = narrative
                _last_llm_success["time"] = datetime.now().strftime("%H:%M:%S")
                _last_llm_success["type"] = "itinerary_narration"
                log.info(f"llm_narrate_itinerary: SUCCESS — {len(narratives)} narratives")
                return itin
            except json.JSONDecodeError:
                log.warning("llm_narrate_itinerary: JSON parse failed — using raw text as single narrative")
        # Fallback: use the whole raw text as narrative for the first stop
        if raw and len(raw) > 20:
            itin[0]["narrative"] = raw[:400]
            _last_llm_success["time"] = datetime.now().strftime("%H:%M:%S")
            _last_llm_success["type"] = "itinerary_narration_raw"
            log.info("llm_narrate_itinerary: used raw text fallback")
        return itin
    except Exception as e:
        log.warning(f"llm_narrate_itinerary FAILED: {type(e).__name__}: {e}")
        return itin

def geo_search(query, olat, olon, max_km, k=TOP_K, origin_state=None, transport="any",
               avoid="", activities=""):
    vs       = get_vs()
    emb      = get_emb()
    enriched = expand_emotion_query(query, avoid=avoid, activities=activities)

    # ── FAISS search — fetch ALL docs so nearby ones aren't knocked out ────
    # With 356 destinations, k=50 misses most nearby spots because distant
    # adventure/emotion matches score higher globally. Fetch the full index.
    total_docs = vs.index.ntotal
    pool = vs.similarity_search_with_score(enriched, k=total_docs)

    filtered = [(d, s) for d, s in pool
                if haversine(olat, olon, float(d.metadata["lat"]), float(d.metadata["lon"])) <= max_km]
    widened = False

    # Widen carefully if not enough nearby candidates.
    # Road trips should stay geographically plausible; only flight searches
    # are allowed to expand aggressively.
    if transport == "fly":
        hard_cap = 99999
        widen_steps = [1.5, 2.0, 3.0]
    elif transport == "drive":
        hard_cap = min(max_km * 1.25, max_km + 250)
        widen_steps = [1.1, 1.25]
    elif max_km <= 500:
        hard_cap = min(max_km * 1.35, max_km + 250)
        widen_steps = [1.15, 1.35]
    else:
        hard_cap = min(max_km * 1.5, max_km + 400)
        widen_steps = [1.15, 1.35, 1.5]

    for multiplier in widen_steps:
        if len(filtered) >= 5:
            break
        wider    = min(max_km * multiplier, hard_cap)
        filtered = [(d, s) for d, s in pool
                    if haversine(olat, olon, float(d.metadata["lat"]), float(d.metadata["lon"])) <= wider]
        widened  = True
        max_km   = wider

    # Penalise same-state only for fly searches (user wants to leave home state).
    # Never penalise for drive/any — nearby Kerala results are exactly right.
    if origin_state:
        same = origin_state.lower()
        if transport == "fly":
            filtered = [
                (d, s * 1.6 if d.metadata.get("state","").lower() == same else s * 0.7)
                for d, s in filtered
            ]

    filtered.sort(key=lambda x: x[1])

    def _place_type(doc):
        lnd = doc.metadata.get("landscape","")
        return lnd.split(" in ")[0].strip() if " in " in lnd else lnd.strip()

    # ── Pool building strategy ────────────────────────────────────────────
    # For fly / long-distance trips (>1200 km radius): enforce diversity so
    # results aren't all from one state/landscape type.
    # For 1/2/3-day local trips: NO limits at all — just take the top 20
    # nearest+best-scoring candidates and let the LLM pick the best 5.
    # There are plenty of Kerala destinations; we must not filter them out.

    is_local_trip = (transport != "fly" and max_km <= 1200)

    if is_local_trip:
        # No type or state caps — pass all filtered candidates straight to reranker
        pool_for_rerank = filtered[:FAISS_FETCH]
    else:
        def _pick(pool, max_type, max_state):
            tc, sc, chosen, rest = {}, {}, [], []
            for doc, score in pool:
                pt  = _place_type(doc)
                st  = doc.metadata.get("state","")
                if tc.get(pt,0) < max_type and sc.get(st,0) < max_state:
                    tc[pt] = tc.get(pt,0)+1
                    sc[st] = sc.get(st,0)+1
                    chosen.append((doc, score))
                else:
                    rest.append((doc, score))
                if len(chosen) >= FAISS_FETCH: break
            return chosen, rest

        pool_for_rerank, fallback = _pick(filtered, 2, 2)
        if len(pool_for_rerank) < FAISS_FETCH:
            extra, _ = _pick(fallback, 4, 4)
            pool_for_rerank += extra
        pool_for_rerank += fallback[:max(0, FAISS_FETCH - len(pool_for_rerank))]

    # Deterministic rerank first so candidate quality is strong even if the
    # external LLM reranker is unavailable or returns unstable rankings.
    pool_for_rerank = heuristic_rerank(
        pool_for_rerank,
        query=query,
        enriched_query=enriched,
        olat=olat,
        olon=olon,
        max_km=max_km,
        origin_state=origin_state,
        transport=transport,
        avoid=avoid,
        activities=activities,
        k=k,
    )

    # ── Hard avoid pre-filter ─────────────────────────────────────────────
    # Filter on the landscape TYPE (the part before " in ") extracted directly
    # from doc.metadata["landscape"]. This is exact — no fuzzy text matching,
    # no synonym misses. Maps user avoid keywords → exact landscape type sets
    # derived from the actual destinations.csv landscape column.
    AVOID_LANDSCAPE_TYPES = {
        "hill":        {"Hill", "Hill Station", "Ghat", "Gravity Hill", "Mountain", "Mountain Peak",
                        "Valley", "Viewpoint", "Scenic Point", "Trekking"},
        "mountain":    {"Mountain", "Mountain Peak", "Hill", "Hill Station", "Valley",
                        "Trekking", "Ski Resort"},
        "valley":      {"Valley"},
        "trek":        {"Trekking", "Adventure Sport"},
        "adventure":   {"Trekking", "Adventure Sport", "Ski Resort"},
        "beach":       {"Beach", "Island Beach", "Cliff Beach", "Wellness Beach"},
        "coast":       {"Beach", "Island Beach", "Cliff Beach", "Wellness Beach", "Port", "Promenade"},
        "sea":         {"Beach", "Island Beach", "Cliff Beach", "Wellness Beach", "Port"},
        "island":      {"Island", "Island Beach", "River Island"},
        "desert":      {"Sand Dunes", "Desert"},
        "forest":      {"Forest", "Wildlife Sanctuary", "National Park", "Tiger Reserve",
                        "Bird Sanctuary", "Ecotourism"},
        "wildlife":    {"Wildlife Sanctuary", "Tiger Reserve", "Bird Sanctuary", "National Park",
                        "Zoo", "Ecotourism"},
        "temple":      {"Temple", "Temples", "Shrine", "Religious Shrine", "Religious Site",
                        "Religious Complex", "Spiritual Center"},
        "religious":   {"Temple", "Temples", "Shrine", "Religious Shrine", "Religious Site",
                        "Religious Complex", "Spiritual Center", "Mosque", "Gurudwara", "Monastery"},
        "city":        {"Cultural City", "Heritage City", "Heritage Town", "Urban Development Project"},
        "cold":        {"Ski Resort", "Glacier"},
        "snow":        {"Ski Resort"},
        "cave":        {"Cave"},
        "waterfall":   {"Waterfall"},
        "lake":        {"Lake"},
        "fort":        {"Fort"},
        "palace":      {"Palace"},
        "monument":    {"Monument", "Memorial", "Mausoleum", "Tomb", "Tombs"},
        "museum":      {"Museum"},
        "backwater":   {"Backwaters"},
    }

    def _landscape_type(doc):
        """Extract the type part from 'Hill Station in Cool climate' → 'Hill Station'."""
        lnd = doc.metadata.get("landscape", "")
        return lnd.split(" in ")[0].strip() if " in " in lnd else lnd.strip()

    def _matches_avoid(doc, avoid_str):
        """Return True if this doc's landscape type matches something the user wants to avoid."""
        if not avoid_str or avoid_str.lower().strip() in ("skip", "none", "no", "-", ""):
            return False
        avoid_lower = avoid_str.lower()
        ltype = _landscape_type(doc)  # e.g. "Hill Station"

        # Check each avoid keyword against the landscape type map
        for key, bad_types in AVOID_LANDSCAPE_TYPES.items():
            if key in avoid_lower:
                if ltype in bad_types:
                    log.info(f"avoid filter: '{doc.metadata.get('destination')}' blocked — landscape '{ltype}' matches avoid key '{key}'")
                    return True
        return False

    # ── Activities preference boost ───────────────────────────────────────
    # Unlike avoid (hard exclusion), activities is a SOFT BOOST — destinations
    # matching preferred activities get their FAISS score lowered (better rank)
    # so the LLM sees them at the top of the candidate list.
    ACTIVITIES_LANDSCAPE_TYPES = {
        "beach":       {"Beach", "Island Beach", "Cliff Beach", "Wellness Beach"},
        "swim":        {"Beach", "Island Beach", "Cliff Beach", "Lake", "Backwaters"},
        "trek":        {"Trekking", "Mountain", "Mountain Peak", "Hill", "Hill Station",
                        "Valley", "Forest", "National Park", "Adventure Sport"},
        "hike":        {"Trekking", "Mountain", "Hill", "Forest", "National Park", "Valley"},
        "water":       {"Backwaters", "Lake", "River Island", "Island", "Waterfall", "Beach"},
        "houseboat":   {"Backwaters", "Lake", "River Island"},
        "backwater":   {"Backwaters", "River Island", "Island"},
        "wildlife":    {"Wildlife Sanctuary", "Tiger Reserve", "National Park",
                        "Bird Sanctuary", "Ecotourism", "Forest"},
        "bird":        {"Bird Sanctuary", "Wildlife Sanctuary", "National Park",
                        "Ecotourism", "Backwaters"},
        "temple":      {"Temple", "Temples", "Shrine", "Religious Site",
                        "Religious Complex", "Spiritual Center", "Monastery"},
        "spiritual":   {"Temple", "Temples", "Shrine", "Monastery", "Spiritual Center",
                        "Gurudwara", "Mosque", "Religious Complex"},
        "heritage":    {"Heritage City", "Heritage Town", "Fort", "Palace", "Historical",
                        "Monument", "Mausoleum", "Tomb", "Tombs", "Museum"},
        "fort":        {"Fort", "Heritage Town", "Heritage City", "Historical"},
        "relax":       {"Wellness Beach", "Wellness Centre", "Backwaters", "Lake",
                        "Beach", "Island", "Ecotourism"},
        "ayurveda":    {"Wellness Centre", "Wellness Beach", "Ecotourism"},
        "waterfall":   {"Waterfall"},
        "cave":        {"Cave"},
        "island":      {"Island", "Island Beach", "River Island"},
        "photography": {"Viewpoint", "Scenic Point", "Sunrise Point", "Waterfall",
                        "Fort", "Heritage Town", "Natural Feature"},
        "camping":     {"Forest", "National Park", "Valley", "Mountain", "Trekking"},
        "surf":        {"Beach", "Cliff Beach", "Island Beach"},
        "dive":        {"Beach", "Island Beach", "Island"},
        "snorkel":     {"Beach", "Island Beach", "Island"},
        "culture":     {"Cultural City", "Heritage City", "Heritage Town", "Museum",
                        "Festival", "Cultural"},
        "food":        {"Heritage City", "Heritage Town", "Cultural City", "Market", "Festival"},
        "adventure":   {"Adventure Sport", "Trekking", "Mountain", "Ski Resort",
                        "Valley", "Suspension Bridge"},
        "ski":         {"Ski Resort"},
        "vineyard":    {"Vineyard", "Orchard"},
        "garden":      {"Botanical Garden", "Park", "Orchard"},
        "family":      {"Amusement Park", "Theme Park", "Zoo", "Aquarium",
                        "Park", "Beach", "Lake"},
        "amusement":   {"Amusement Park", "Theme Park", "Entertainment"},
        "park":        {"Amusement Park", "Theme Park", "National Park",
                        "Park", "Botanical Garden"},
        "festival":    {"Festival", "Cultural City", "Heritage City"},
        "pilgrimage":  {"Temple", "Temples", "Shrine", "Monastery", "Gurudwara",
                        "Mosque", "Religious Site", "Spiritual Center"},
    }

    def _matches_activities(doc, activities_str):
        """Return True if this doc's landscape type matches a preferred activity."""
        if not activities_str or activities_str.lower().strip() in ("skip", "none", "no", "-", ""):
            return False
        acts_lower = activities_str.lower()
        ltype = _landscape_type(doc)
        for key, good_types in ACTIVITIES_LANDSCAPE_TYPES.items():
            if key in acts_lower:
                if ltype in good_types:
                    return True
        return False

    if activities and activities.lower().strip() not in ("skip", "none", "no", "-", ""):
        # Boost matching docs by halving their FAISS distance score (lower = better)
        boosted = []
        for doc, score in pool_for_rerank:
            if _matches_activities(doc, activities):
                boosted.append((doc, score * 0.5))
                log.info(f"activities boost: '{doc.metadata.get('destination')}' boosted for activities='{activities}'")
            else:
                boosted.append((doc, score))
        # Re-sort so boosted candidates rise to the top of the LLM's input list
        pool_for_rerank = sorted(boosted, key=lambda x: x[1])

    allowed = pool_for_rerank  # default: no filtering
    if avoid:
        pre_filter_count = len(pool_for_rerank)
        allowed   = [(d, s) for d, s in pool_for_rerank if not _matches_avoid(d, avoid)]
        removed   = pre_filter_count - len(allowed)
        log.info(f"avoid pre-filter: removed {removed}/{pre_filter_count} candidates for avoid='{avoid}'")
        # Only apply if enough candidates remain — never strand the user with < k results
        if len(allowed) >= k:
            pool_for_rerank = allowed
        else:
            log.warning(f"avoid pre-filter: only {len(allowed)} left after filter — keeping full pool")

    # LLM re-ranks the pool — picks best k based on emotional fit + user preferences
    reranked = llm_rerank(query, pool_for_rerank[:FAISS_FETCH], k, avoid=avoid, activities=activities)

    # Pad if LLM returned fewer than k — but NEVER pad with avoided landscape types
    if len(reranked) < k:
        used = set(id(d) for d, _ in reranked)
        # Use allowed pool for padding, not the full pool_for_rerank
        pad_source = allowed if (avoid and len(allowed) >= k) else pool_for_rerank
        for item in pad_source:
            if id(item[0]) not in used and not _matches_avoid(item[0], avoid):
                reranked.append(item)
            if len(reranked) >= k:
                break

    # Final hard pass — strip any avoided types that snuck through the LLM reranker
    if avoid:
        reranked = [item for item in reranked if not _matches_avoid(item[0], avoid)]

    # ── Post-rerank state diversity cap ──────────────────────────────────
    # Local trips: no cap — all 5 results can (and should) be from Kerala.
    # Fly / long trips: cap at 2 per state to force geographic spread.
    if is_local_trip:
        return reranked[:k], widened

    max_per_state_final = 2
    state_counts: dict = {}
    diverse: list = []
    overflow: list = []
    for item in reranked:
        st = item[0].metadata.get("state", "")
        if state_counts.get(st, 0) < max_per_state_final:
            state_counts[st] = state_counts.get(st, 0) + 1
            diverse.append(item)
        else:
            overflow.append(item)
    if len(diverse) < k:
        diverse += overflow[:k - len(diverse)]

    return diverse[:k], widened

def find_sec(primary, df, n=3, olat=None, olon=None, max_km=None):
    """Find secondary stops for an itinerary.
    Only includes destinations within max_km of the origin — never pulls
    far-off places (e.g. Delhi) as side trips on a local drive itinerary.
    """
    plat = float(primary.metadata["lat"])
    plon = float(primary.metadata["lon"])

    def _within_range(r):
        """True if this row is reachable given the trip radius."""
        if max_km is None or olat is None:
            return True
        return haversine(olat, olon, float(r["lat"]), float(r["lon"])) <= max_km

    nearby = [x.strip() for x in primary.metadata.get("nearby","").split(",") if x.strip()]
    out = []
    for name in nearby:
        row = df[df["destination"].str.contains(name, case=False, na=False)]
        if not row.empty:
            r = row.iloc[0]
            if _within_range(r):
                out.append({
                    "destination": r["destination"], "state": r["state"],
                    "vibe": r["vibe"], "activities": r["activities"],
                    "landscape": r["landscape"], "description": r["description"],
                    "lat": float(r["lat"]), "lon": float(r["lon"])
                })
        if len(out) >= n: break

    if len(out) < n:
        # Fallback: find closest destinations to the primary stop, same state preferred
        pn = primary.metadata["destination"]
        ps = primary.metadata.get("state", "")
        candidates = df[df["destination"] != pn].copy()
        candidates["_dist"] = candidates.apply(
            lambda r: haversine(plat, plon, float(r["lat"]), float(r["lon"])), axis=1
        )
        # Filter by max_km from origin if provided
        if max_km is not None and olat is not None:
            candidates = candidates[
                candidates.apply(lambda r: haversine(olat, olon, float(r["lat"]), float(r["lon"])) <= max_km, axis=1)
            ]
        # Sort: same-state first, then by distance from primary
        candidates["_same_state"] = (candidates["state"] == ps).astype(int)
        candidates = candidates.sort_values(["_same_state", "_dist"], ascending=[False, True])
        for _, r in candidates.iterrows():
            if len(out) >= n: break
            if not any(x["destination"] == r["destination"] for x in out):
                out.append({
                    "destination": r["destination"], "state": r["state"],
                    "vibe": r["vibe"], "activities": r["activities"],
                    "landscape": r["landscape"], "description": r["description"],
                    "lat": float(r["lat"]), "lon": float(r["lon"])
                })
    return out[:n]

def build_itin(primary, secs, days, start_date=None):
    if days == 1:
        primary_days, leftover = 1, 0
    else:
        primary_days = max(1, round(days * 0.6))
        leftover     = days - primary_days

    # Always parse as pure date to avoid timezone-induced off-by-one
    if start_date:
        date_part = start_date.split("T")[0]  # strip any time component
        start = datetime.strptime(date_part, "%Y-%m-%d").replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        start = None

    def dr(o, d):
        if not start: return ""
        s = start + timedelta(days=o-1)
        e = s + timedelta(days=d-1)
        return s.strftime("%d %b %Y") if d==1 else f"{s.strftime('%d %b')} → {e.strftime('%d %b %Y')}"

    def day_label(sd, d):
        return f"Day {sd}" if d==1 else f"Day {sd}–{sd+d-1}"

    m    = primary.metadata
    desc = primary.page_content.split("Description:")[-1].strip()
    itin = [{
        "type": "primary", "destination": str(m["destination"]), "state": str(m["state"]),
        "days": primary_days, "day_range": day_label(1, primary_days),
        "date_range": dr(1, primary_days),
        "activities": str(m["activities"]), "landscape": str(m["landscape"]),
        "description": desc, "lat": float(m["lat"]), "lon": float(m["lon"]),
    }]

    if leftover > 0 and secs:
        usable = secs[:leftover]
        n      = len(usable)
        sd     = [leftover // n] * n
        sd[-1] += leftover - sum(sd)
        cur    = 1 + primary_days
        for i, s in enumerate(usable):
            d = sd[i]
            itin.append({
                "type": "secondary", "destination": s["destination"], "state": s["state"],
                "days": d, "day_range": day_label(cur, d),
                "date_range": dr(cur, d),
                "activities": s["activities"], "landscape": s["landscape"],
                "description": s["description"], "lat": s["lat"], "lon": s["lon"],
            })
            cur += d
    return itin

def fetch_weather(lat, lon, city):
    if OWM_API_KEY == "YOUR_API_KEY":
        return {"city": city, "is_live": False, "temp_min": 19.0, "temp_max": 31.0,
                "temp_avg": 25.0, "humidity": 70, "wind_kmh": 16.0,
                "rain_mm": 1.0, "conditions": ["Partly Cloudy"]}
    try:
        r = requests.get(OWM_URL,
            params={"lat":lat,"lon":lon,"appid":OWM_API_KEY,"units":"metric","cnt":8},
            timeout=8)
        r.raise_for_status()
        items = r.json().get("list", [])
        if not items: raise ValueError
        temps   = [i["main"]["temp"] for i in items]
        humids  = [i["main"]["humidity"] for i in items]
        winds   = [i["wind"]["speed"]*3.6 for i in items]
        rain    = sum(i.get("rain",{}).get("3h",0) for i in items)
        conds   = list(dict.fromkeys(i["weather"][0]["description"].title() for i in items))
        return {"city": city, "is_live": True,
                "temp_min": round(min(temps),1), "temp_max": round(max(temps),1),
                "temp_avg": round(sum(temps)/len(temps),1),
                "humidity": round(sum(humids)/len(humids)),
                "wind_kmh": round(sum(winds)/len(winds),1),
                "rain_mm": round(rain,1), "conditions": conds[:3]}
    except Exception:
        return {"city": city, "is_live": False, "temp_min": 19.0, "temp_max": 31.0,
                "temp_avg": 25.0, "humidity": 70, "wind_kmh": 16.0,
                "rain_mm": 1.0, "conditions": ["Partly Cloudy"]}

def make_packing(weather, itin):
    lands = " ".join(s["landscape"].lower() for s in itin)
    acts  = " ".join(s["activities"].lower() for s in itin)
    stats = " ".join(s["state"].lower() for s in itin)
    t = weather["temp_avg"]; rain = weather["rain_mm"] > 1; cold = t < 15; hot = t > 28
    cl = ["Comfortable walking shoes", "Casual day clothes (3-4 sets)"]
    if cold:   cl += ["Heavy fleece / down jacket","Thermal layers","Warm hat & gloves"]
    elif hot:  cl += ["Lightweight breathable shirts","Shorts / linen trousers","Wide-brim sun hat","Sandals"]
    else:      cl += ["Light jacket or hoodie","Mix of t-shirts & long sleeves"]
    if rain:   cl += ["Waterproof rain jacket","Quick-dry trousers"]
    if "trekking" in acts:  cl += ["Sturdy trekking boots","Moisture-wicking socks x3"]
    if "beach" in lands:    cl += ["Swimwear x2","Rash guard","Water shoes"]
    if "temple" in acts:    cl += ["Full-sleeve tops","Trousers covering knees"]
    ge = ["Daypack 25-30L","Reusable water bottle","Power bank 10000mAh","Universal adapter","Headlamp"]
    if "mountains" in lands or "trekking" in acts: ge += ["Trekking poles","Thermal flask"]
    if "photography" in acts: ge += ["Camera + extra SD cards","Portable tripod"]
    if "beach" in lands:    ge += ["Dry bag 10L"]
    he = ["Personal meds","First-aid kit","Insect repellent DEET 30%+","Hand sanitiser","Sunscreen SPF 50+"]
    if cold:   he += ["Altitude meds (if high elevation)","Lip balm SPF"]
    if hot:    he += ["Aloe vera gel","Antihistamine"]
    if rain:   he += ["Anti-diarrhoeal medicine"]
    if "kerala" in stats: he += ["Mosquito net (backwaters)","Light cotton cloth (humidity)"]
    do = ["Photo ID / Passport + photocopies","Travel insurance","Booking confirmations","Cash INR + 2 cards"]
    ex = []
    if "desert" in lands:  ex += ["Scarf / shemagh","Goggles"]
    if "kerala" in stats:  ex += ["Coconut oil — Kerala cure-all"]
    if "rajasthan" in stats: ex += ["Handicraft shopping bag"]
    return {"clothing": cl, "gear": ge, "health": he, "documents": do,
            "extras": ex or ["You're all set!"]}



def apply_preferences(results, avoid, activities):
    """
    LLM-first preference filtering: avoid/activities are passed to the
    reranker prompt so the model understands context and nuance — not just
    substring matches. This function is now a lightweight annotator that
    adds a preference_score based on the LLM's own reranking decisions.
    It no longer re-sorts — the LLM already ordered by best fit.
    """
    for r in results:
        r.setdefault("preference_score", 0)
    return results


# ── Request/Response models ────────────────────────────────────────────────
class GeocodeRequest(BaseModel):
    city: str
    state: Optional[str] = None

class SearchRequest(BaseModel):
    emotion: str
    lat: float
    lon: float
    max_km: float
    days: int
    transport: str = "any"
    origin_state: Optional[str] = None
    avoid: Optional[str] = ""
    activities: Optional[str] = ""

class ItineraryRequest(BaseModel):
    destination_idx: int
    results: list
    days: int
    start_date: Optional[str] = None

class ItineraryBuildRequest(BaseModel):
    emotion: str
    lat: float
    lon: float
    max_km: float
    days: int
    transport: str = "any"
    origin_state: Optional[str] = None
    destination: str
    start_date: Optional[str] = None

# ── API Endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": EMBED_MODEL}

@app.post("/geocode")
def geocode_endpoint(req: GeocodeRequest):
    """Geocode a city name. Falls back to state centre if city not found."""
    coords = None
    source = "default"
    try:
        r = requests.get(GEOCODE_URL,
            params={"q": f"{req.city}, India", "format": "json", "limit": 1},
            headers={"User-Agent": "Yatrika/4.0"}, timeout=6)
        r.raise_for_status()
        d = r.json()
        if d:
            coords = {"lat": float(d[0]["lat"]), "lon": float(d[0]["lon"])}
            source = "city"
    except Exception:
        pass

    if not coords and req.state:
        fb = STATE_COORDS.get(req.state.lower())
        if fb:
            coords = {"lat": fb[0], "lon": fb[1]}
            source = "state"

    if not coords:
        coords = {"lat": 20.5937, "lon": 78.9629}
        source = "default"

    return {"coords": coords, "source": source, "city": req.city, "state": req.state}

@app.get("/radius")
def radius_endpoint(days: int, transport: str = "any"):
    max_km, label, hint = get_radius(days, transport)
    return {"max_km": max_km, "label": label, "hint": hint}

@app.post("/search")
def search_endpoint(req: SearchRequest):
    """Run emotion + geo search, return top-k destinations."""
    try:
        results, widened = geo_search(
            req.emotion, req.lat, req.lon,
            req.max_km, TOP_K, req.origin_state, req.transport,
            avoid=req.avoid or "", activities=req.activities or ""
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    out = []
    for doc, score in results:
        m = doc.metadata
        lat  = float(m["lat"])
        lon  = float(m["lon"])
        sc   = float(score)
        dist_km   = haversine(req.lat, req.lon, lat, lon)
        match_pct = max(0, min(100, int((1 - sc / 2.0) * 100)))
        out.append({
            "destination":   str(m["destination"]),
            "state":         str(m["state"]),
            "region":        str(m.get("region", "")),
            "lat":           lat,
            "lon":           lon,
            "vibe":          str(m["vibe"]),
            "landscape":     str(m["landscape"]),
            "best_season":   str(m["best_season"]),
            "activities":    str(m["activities"]),
            "description":   doc.page_content.split("Description:")[-1].strip(),
            "dist_km":       round(dist_km, 1),
            "emotion_score": round(sc, 4),
            "match_pct":     match_pct,
        })

    out = apply_preferences(out, req.avoid, req.activities)  # preferences already handled by LLM reranker
    # Claude narrates WHY each destination matches the user's emotional state
    out = narrate_results(req.emotion, out)
    return {"results": out, "widened": bool(widened), "count": len(out)}

@app.post("/itinerary")
def itinerary_endpoint(req: ItineraryBuildRequest):
    """Build full itinerary + weather + packing for a chosen destination."""
    try:
        # Look up the chosen destination directly from the dataset — never
        # re-run geo_search here, because a second LLM rerank pass can return
        # a different order and the fallback (results[0][0]) would silently
        # serve the wrong place.
        df = load_df()
        row = df[df["destination"].str.lower() == req.destination.lower()]
        if row.empty:
            # Fuzzy fallback: partial match
            row = df[df["destination"].str.contains(req.destination, case=False, na=False)]
        if row.empty:
            raise HTTPException(404, f"Destination '{req.destination}' not found in dataset")

        r = row.iloc[0]
        content = (
            f"A destination for someone feeling {r['emotion_tags']}. "
            f"If you feel {r['vibe']}, visit {r['destination']} in {r['state']}. "
            f"Emotional Vibe: {r['vibe']}. "
            f"Emotional Tags: {r['emotion_tags']}. "
            f"Perfect for someone feeling: {r['emotion_tags']}. "
            f"This place resonates with: {r['vibe']}. "
            f"Destination: {r['destination']}, {r['state']}. "
            f"Landscape: {r['landscape']}. "
            f"Best Season: {r['best_season']}. "
            f"Activities: {r['activities']}. "
            f"Description: {r['description']}"
        )
        chosen = Document(page_content=content, metadata={
            "destination": str(r["destination"]),
            "state":       str(r["state"]),
            "region":      str(r.get("region", "India")),
            "lat":         float(r["lat"]),
            "lon":         float(r["lon"]),
            "vibe":        str(r["vibe"]),
            "best_season": str(r["best_season"]),
            "activities":  str(r["activities"]),
            "nearby":      str(r["nearby_attractions"]),
            "landscape":   str(r["landscape"]),
        })

        secs = find_sec(chosen, df, olat=req.lat, olon=req.lon, max_km=req.max_km)
        itin = build_itin(chosen, secs, req.days, req.start_date)
        wea  = fetch_weather(chosen.metadata["lat"], chosen.metadata["lon"], req.destination)
        pack = make_packing(wea, itin)

        # LLM narrates each stop with emotionally resonant prose
        itin = llm_narrate_itinerary(req.emotion, itin, wea, req.days)
        return {"itinerary": itin, "weather": wea, "packing": pack}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/warmup")
def warmup():
    """Pre-load the embedding model and FAISS index."""
    try:
        get_vs()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/test-llm")
def test_llm():
    """
    Diagnostic endpoint — checks if HuggingFace LLM is working.
    Visit http://localhost:8000/test-llm in your browser.
    """
    status = {
        "hf_api_key_set": bool(HF_API_KEY),
        "hf_api_key_preview": f"{HF_API_KEY[:8]}..." if HF_API_KEY else "NOT SET",
        "model": HF_MODEL,
        "llm_active": False,
        "last_successful_llm_call": _last_llm_success,
        "keyword_extraction": None,
        "narration_test": None,
        "error": None,
    }

    if not HF_API_KEY:
        status["error"] = "HF_API_KEY not set — using static dictionary fallback"
        return status

    # Test 1: keyword extraction
    try:
        test_emotion = "burned out and need peace"
        raw_query    = expand_emotion_query(test_emotion)
        static_base = _static_expand(test_emotion)
        # LLM output = static_base + " " + keywords, so longer = LLM ran
        llm_ran = len(raw_query.strip()) > len(static_base.strip()) + 5
        status["keyword_extraction"] = {
            "input":    test_emotion,
            "output":   raw_query,
            "used_llm": llm_ran,
            "note":     "LLM adds keywords after static base — check output length" if not llm_ran else "LLM keywords appended successfully",
        }
        status["llm_active"] = llm_ran
    except Exception as e:
        status["error"] = f"Keyword extraction failed: {str(e)}"

    # Test 2: narration (single destination mock)
    try:
        mock_results = [{
            "destination": "Munnar",
            "state": "Kerala",
            "vibe": "Serene misty tea gardens",
            "description": "Rolling green hills covered in tea estates with cool misty mornings.",
        }]
        narrated = narrate_results("burned out and need peace", mock_results)
        status["narration_test"] = {
            "destination": "Munnar",
            "why_match": narrated[0].get("why_match", "NOT GENERATED — LLM narration did not run"),
        }
    except Exception as e:
        status["narration_test"] = {"error": str(e)}

    return status
