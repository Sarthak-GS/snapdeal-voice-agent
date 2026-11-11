"""
main.py — FastAPI application and REST endpoints for Snapdeal AI Virtual KAM.
"""

import copy
import uuid
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import (
    HTML_FILE,
    AGENTROUTER_MODEL,
    DEFAULT_VOICE,
    MAX_TOKENS,
    SUMMARY_MAX_TOKENS,
    logger,
)
from app.data_loader import BASE_KBs
from app.diagnostic import compute_health_score
from app.voice import transcribe_audio, generate_tts
from app.agent import build_kiran_system, call_llm_with_tools
from app.tools import execute_tool
import generate_data

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP & MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="Snapdeal AI Virtual KAM", version="4.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-Memory Active Sessions Store
sessions: dict[str, dict] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════
class StartRequest(BaseModel):
    seller_id: Optional[str] = generate_data.DEFAULT_SELLER_ID


class StartResponse(BaseModel):
    session_id: str
    greeting: str
    provider: str
    health_score: int
    health_band: str
    seller_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    provider: str
    turn: int
    is_final: bool
    actions_executed: list[str] = []
    tools_called: list[str] = []
    blueprint: Optional[dict] = None


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = DEFAULT_VOICE


# ═══════════════════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/")
async def serve_index():
    """Serves the WebAudio VAD frontend."""
    return FileResponse(HTML_FILE)


@app.get("/favicon.ico")
async def favicon():
    """Silence browser favicon 404s."""
    return Response(status_code=204)


@app.get("/api/health")
async def health_check():
    """System readiness check."""
    return {
        "status": "ready",
        "version": "4.1.0",
        "model": AGENTROUTER_MODEL,
        "active_sessions": len(sessions),
        "sellers": generate_data.SELLER_IDS,
    }


@app.get("/api/test_llm")
async def test_llm():
    """Diagnostic endpoint to inspect LLM errors on cloud hosts."""
    import anthropic
    from app.config import AGENTROUTER_API_KEY, AGENTROUTER_BASE_URL, AGENTROUTER_MODEL
    try:
        client = anthropic.Anthropic(
            api_key=AGENTROUTER_API_KEY,
            base_url=AGENTROUTER_BASE_URL,
            timeout=15.0,
        )
        resp = client.messages.create(
            model=AGENTROUTER_MODEL,
            max_tokens=50,
            messages=[{"role": "user", "content": "ping"}],
        )
        return {"status": "ok", "key_prefix": AGENTROUTER_API_KEY[:10], "reply": resp.content[0].text}
    except Exception as e:
        import traceback
        return {"status": "error", "key_prefix": AGENTROUTER_API_KEY[:10], "error": str(e), "traceback": traceback.format_exc()}


@app.get("/api/sellers")
async def list_sellers():
    """Returns summaries and health scores for all available sellers."""
    result = []
    for sid in generate_data.SELLER_IDS:
        kb = BASE_KBs[sid]
        health = compute_health_score(kb)
        prof = kb["seller_profile"]
        lm = kb["performance"]["last_month"]
        tm = kb["performance"]["this_month"]
        rev_drop_pct = (lm["revenue"] - tm["revenue"]) / lm["revenue"] * 100
        result.append({
            "seller_id": sid,
            "name": prof["name"],
            "store": prof["store"],
            "category": prof["category"],
            "city": prof["city"],
            "tier": prof["tier"],
            "rating": prof["rating"],
            "health_score": health["score"],
            "health_band": health["band"],
            "rev_drop_pct": round(rev_drop_pct, 1),
            "this_month_revenue": tm["revenue"],
            "last_month_revenue": lm["revenue"],
        })
    return result


@app.get("/api/seller/data")
async def get_seller_data(seller_id: Optional[str] = None, session_id: Optional[str] = None):
    """Returns full catalog, unit economics, and hypotheses for a seller."""
    if session_id and session_id in sessions:
        kb = sessions[session_id]["kb"]
    elif seller_id and seller_id in BASE_KBs:
        kb = BASE_KBs[seller_id]
    else:
        kb = BASE_KBs[generate_data.DEFAULT_SELLER_ID]

    return {
        "profile": kb["seller_profile"],
        "performance": kb["performance"],
        "products": kb["products"],
        "cancellation_breakdown": kb["cancellation_breakdown"],
        "market_context": kb["market_context"],
        "health": compute_health_score(kb),
        "hypotheses": kb["causal_hypotheses"],
    }


@app.post("/api/session/start", response_model=StartResponse)
async def start_session(req: StartRequest):
    """Initializes a new live voice KAM session."""
    seller_id = req.seller_id or generate_data.DEFAULT_SELLER_ID
    if seller_id not in BASE_KBs:
        raise HTTPException(400, f"Unknown seller_id: {seller_id!r}")

    session_id = str(uuid.uuid4())
    session_kb = copy.deepcopy(BASE_KBs[seller_id])
    health = compute_health_score(session_kb)
    system_prompt = build_kiran_system(session_kb)

    init_msg = {
        "role": "user",
        "content": f"Hello, this is {session_kb['seller_profile']['name']} calling from {session_kb['seller_profile']['store']}.",
    }

    sess_obj = {
        "seller_id": seller_id,
        "kb": session_kb,
        "history": [],
        "turn": 0,
        "executed_actions": [],
        "summary_cache": None,
        "blueprint": None,
    }

    try:
        greeting, provider, tools_called = await call_llm_with_tools(
            [init_msg], system_prompt, sess_obj, max_tokens=MAX_TOKENS
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"[start_session error] {tb}")
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}\n{tb}")

    sess_obj["history"] = [init_msg, {"role": "assistant", "content": greeting}]
    sessions[session_id] = sess_obj

    return StartResponse(
        session_id=session_id,
        greeting=greeting,
        provider=provider,
        health_score=health["score"],
        health_band=health["band"],
        seller_id=seller_id,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Processes conversational turns and executes live actions via ReAct agentic loop."""
    if not req.session_id or req.session_id not in sessions:
        raise HTTPException(404, "Session not found. Please click 'Start Call'.")

    cleaned_msg = req.message.strip()
    if not cleaned_msg:
        raise HTTPException(400, "Empty message.")

    sess = sessions[req.session_id]
    sess["turn"] += 1
    turn = sess["turn"]

    sess["history"].append({"role": "user", "content": cleaned_msg})

    farewell_phrases = ["thank you", "thanks kiran", "bye", "goodbye", "that's all", "that is all", "wrap up", "close"]
    is_closing = any(p in cleaned_msg.lower() for p in farewell_phrases) and turn >= 2

    system_prompt = build_kiran_system(sess["kb"])

    trimmed = sess["history"][-10:]
    if trimmed and trimmed[0]["role"] == "assistant":
        trimmed = trimmed[1:]

    response, provider, tools_called = await call_llm_with_tools(
        trimmed, system_prompt, sess, max_tokens=MAX_TOKENS
    )

    is_final = "---WRITTEN---" in response or sess.get("blueprint") is not None
    blueprint = sess.get("blueprint")
    if is_closing and not is_final:
        is_final = True

    sess["history"].append({"role": "assistant", "content": response})

    return ChatResponse(
        response=response,
        provider=provider,
        turn=turn,
        is_final=is_final,
        actions_executed=sess.get("executed_actions", []),
        tools_called=tools_called,
        blueprint=blueprint,
    )


@app.post("/api/session/{session_id}/summarize", response_model=ChatResponse)
async def summarize_session(session_id: str):
    """Guaranteed 7-Day Pilot Blueprint generation upon call termination."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    sess = sessions[session_id]

    if sess.get("summary_cache"):
        response, provider = sess["summary_cache"]
        return ChatResponse(
            response=response,
            provider=provider,
            turn=sess["turn"],
            is_final=True,
            actions_executed=sess.get("executed_actions", []),
            tools_called=["generate_call_summary"],
            blueprint=sess.get("blueprint"),
        )

    logger.info(f"[session:{session_id[:8]}] Generating on-demand blueprint via /summarize")

    kb = sess["kb"]
    hypotheses = kb.get("causal_hypotheses", [])
    primary_issues = [h["title"] for h in hypotheses[:3]]
    actions = [h["action"] for h in hypotheses[:3]]

    blueprint_json = execute_tool(
        "generate_call_summary",
        {"primary_issues": primary_issues, "actions_agreed": actions},
        sess,
    )

    system_prompt = build_kiran_system(sess["kb"])
    summary_instructions = (
        f"\n\nThe call is now ending. Here is the 7-Day Blueprint data:\n{blueprint_json}\n\n"
        f"Give a 2-3 sentence warm spoken closing: state the primary issue with numbers, "
        f"confirm the top action to take today, and commit to a 3-day follow-up. "
        f"Then on a new line write exactly '---WRITTEN---' and nothing more (the UI renders the blueprint separately)."
    )

    close_msg = [{"role": "user", "content": "KIRAN, please close the call and give me the summary."}]
    response, provider, _ = await call_llm_with_tools(
        close_msg, system_prompt + summary_instructions, sess, max_tokens=SUMMARY_MAX_TOKENS
    )

    sess["summary_cache"] = (response, provider)
    sess["history"].append({"role": "assistant", "content": response})

    return ChatResponse(
        response=response,
        provider=provider,
        turn=sess["turn"],
        is_final=True,
        actions_executed=sess.get("executed_actions", []),
        tools_called=["generate_call_summary"],
        blueprint=sess.get("blueprint"),
    )


@app.post("/api/session/{session_id}/apply_day1")
async def apply_day1_actions(session_id: str):
    """Applies Day 1 recovery fixes live on seller's account."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    sess = sessions[session_id]
    kb = sess["kb"]
    mkt = kb["market_context"]
    mkt["oos_ads_paused"] = True
    mkt["seller_enrolled_festive"] = True
    kb["performance"]["this_month"]["cancellation_rate"] = max(0.05, kb["performance"]["this_month"]["cancellation_rate"] - 0.06)
    for p in kb["products"]:
        if p["price"] > p["competitor_avg"] * 1.05:
            p["price"] = p["competitor_avg"]
    health = compute_health_score(kb)
    return {"ok": True, "health": health}


@app.post("/api/stt")
async def handle_stt(file: UploadFile = File(...)):
    """Transcribes uploaded audio via faster-whisper."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio payload")
    text = await transcribe_audio(audio_bytes)
    return {"text": text}


@app.post("/api/tts")
async def handle_tts(req: TTSRequest):
    """Synthesizes Indian English neural audio via edge-tts."""
    audio_bytes = await generate_tts(req.text, req.voice or DEFAULT_VOICE)
    if not audio_bytes:
        raise HTTPException(400, "Empty text for TTS")
    return Response(content=audio_bytes, media_type="audio/mpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
