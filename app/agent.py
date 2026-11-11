"""
agent.py — KIRAN system prompt, ReAct agentic loop, and multi-provider LLM cascade.
"""

import re
import json
import time
import asyncio
import httpx
import anthropic

from app.config import (
    logger,
    AGENTROUTER_API_KEY,
    AGENTROUTER_BASE_URL,
    AGENTROUTER_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_TOKENS,
)
from app.diagnostic import build_account_snapshot
from app.tools import execute_tool, TOOL_DESCRIPTIONS

TOOL_CALL_REGEX = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# Also catches bare JSON tool calls the LLM sometimes outputs without tags,
# e.g.  {"name": "get_recommendations", "parameters": {...}}
_BARE_JSON_TOOL_REGEX = re.compile(
    r'\{[\s]*"name"\s*:\s*"(?:get_seller_metrics|analyze_issue|get_recommendations'
    r'|generate_call_summary|execute_account_action)".*?\}',
    re.DOTALL,
)
# Catch trailing noise the LLM adds after a JSON leak ("Result?", "But format exact.", etc.)
_TOOL_LEAK_NOISE_REGEX = re.compile(
    r"(?:Result\??|But format exact\.?|No markdown\.?|Since this is tool call.*?answer\.?)\s*",
    re.IGNORECASE,
)


def _strip_markdown(text: str) -> str:
    """Removes markdown symbols for spoken TTS cleanliness."""
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"^[ \t]*[-•][ \t]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*\d+\.[ \t]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"`[^`]*`", lambda m: m.group()[1:-1], text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def sanitize_response(text: str) -> str:
    """
    Hard validation gate: strips any tool-call artifacts that leaked into the
    spoken response before it is returned to the user or fed to TTS.

    Handles three leak patterns observed in production:
      1. <tool_call>{...}</tool_call> tags (should have been stripped by loop)
      2. Bare JSON  {"name": "get_recommendations", "parameters": {...}}
      3. Commentary noise after a JSON leak ("Result?", "But format exact.", etc.)
    Also strips <tool_result> blocks that might be in working_messages context.
    """
    # Strip tagged tool calls
    text = TOOL_CALL_REGEX.sub("", text)
    # Strip bare JSON tool calls (without tags)
    text = _BARE_JSON_TOOL_REGEX.sub("", text)
    # Strip <tool_result> blocks
    text = re.sub(r"<tool_result>.*?</tool_result>", "", text, flags=re.DOTALL)
    # Strip trailing noise left after JSON removal
    text = _TOOL_LEAK_NOISE_REGEX.sub("", text)
    # Collapse excess whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    cleaned = text.strip()
    if cleaned:
        return cleaned
    # If sanitization emptied the response, return a safe fallback
    return "I have the data pulled up. Could you tell me more about the specific issue you're seeing?"


def build_kiran_system(kb: dict) -> str:
    """Builds the phone conversation system prompt with live account diagnostic context."""
    prof = kb["seller_profile"]
    snapshot = build_account_snapshot(kb)
    return f"""You are KIRAN, an AI-powered Virtual Key Account Manager at Snapdeal.
You are on a live voice call with {prof['name']} from {prof['store']} ({prof['city']}).

VOICE RULES (this is a phone call — critical):
- Respond in 2-3 sentences max per turn. Never longer.
- Never use bullet points, asterisks, dashes, numbered lists, or markdown formatting.
- Speak naturally, like an experienced business advisor on a phone call.
- Always cite exact numbers from the data when answering seller questions.
- Ask at most ONE question per turn to keep the conversation flowing.
- Speak natural Indian business English.
- Do NOT follow a rigid script. Address the seller's actual concern first.

{TOOL_DESCRIPTIONS}

HOW TO CALL A TOOL:
Output:
<tool_call>{{"name": "tool_name", "parameters": {{"param": "val"}}}}</tool_call>
You will receive <tool_result> with data, then provide your spoken answer to the seller.
If the snapshot below already gives you sufficient information, answer directly without calling a tool.

CRITICAL — NEVER DO THIS:
- Never output raw JSON like {{"name": "get_recommendations", ...}} in your spoken reply
- Never say "Result?" or add commentary about the tool call in your reply
- The seller must only ever hear natural spoken language — tool calls are invisible to them
- After receiving a <tool_result>, always synthesise the data into a natural spoken answer

{snapshot}"""


# ═══════════════════════════════════════════════════════════════════════════════
# LLM PROVIDER CASCADE (AgentRouter -> Ollama -> Gemini -> Groq)
# ═══════════════════════════════════════════════════════════════════════════════
async def _call_agentrouter(messages: list, system: str, max_tokens: int = MAX_TOKENS) -> str:
    client = anthropic.Anthropic(
        api_key=AGENTROUTER_API_KEY,
        base_url=AGENTROUTER_BASE_URL,
        timeout=35.0,
    )
    loop = asyncio.get_running_loop()

    def _sync_create():
        return client.messages.create(
            model=AGENTROUTER_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )

    last_ex = None
    for attempt in range(2):
        try:
            resp = await loop.run_in_executor(None, _sync_create)
            text_blocks = [
                getattr(b, "text", "")
                for b in resp.content
                if hasattr(b, "text") or getattr(b, "type", "") == "text"
            ]
            text = "".join(t for t in text_blocks if t).strip()
            if text:
                return text
        except Exception as e:
            last_ex = e
            logger.warning(f"[llm:agentrouter] Attempt {attempt+1} failed: {e}")
            await asyncio.sleep(1.0)
    raise last_ex or ValueError("AgentRouter returned empty text")


async def _call_ollama(messages: list, system: str, max_tokens: int = MAX_TOKENS) -> str:
    ollama_msgs = [{"role": "system", "content": system}] + messages
    async with httpx.AsyncClient(timeout=40.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": ollama_msgs, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


async def _call_gemini(messages: list, system: str, max_tokens: int = MAX_TOKENS) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")
    contents = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7},
    }
    async with httpx.AsyncClient(timeout=25.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


async def _call_groq(messages: list, system: str, max_tokens: int = MAX_TOKENS) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")
    all_msgs = [{"role": "system", "content": system}] + messages
    async with httpx.AsyncClient(timeout=25.0) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": GROQ_MODEL, "messages": all_msgs, "max_tokens": max_tokens, "temperature": 0.7},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


async def call_llm_plain(messages: list, system_prompt: str, max_tokens: int = MAX_TOKENS) -> tuple[str, str]:
    """Plain LLM cascade fallback without tool loops."""
    providers = [
        ("agentrouter", _call_agentrouter),
        ("ollama", _call_ollama),
        ("gemini", _call_gemini),
        ("groq", _call_groq),
    ]
    last_error = None
    for name, fn in providers:
        try:
            text = await fn(messages, system_prompt, max_tokens)
            text = _strip_markdown(text)
            if text:
                return text, name
        except Exception as e:
            logger.warning(f"[llm:cascade] {name} failed: {str(e)[:100]}")
            last_error = e
    raise RuntimeError(f"All LLM providers failed. Last: {last_error}")


# ═══════════════════════════════════════════════════════════════════════════════
# AGENTIC REACT TOOL LOOP
# ═══════════════════════════════════════════════════════════════════════════════
async def call_llm_with_tools(
    messages: list,
    system_prompt: str,
    session: dict,
    max_tokens: int = MAX_TOKENS,
) -> tuple[str, str, list[str]]:
    """
    Runs the ReAct agentic loop: parses <tool_call>, invokes handlers against session,
    injects <tool_result>, and falls back safely if output is empty.
    """
    tools_called = []
    loop = asyncio.get_running_loop()

    client = anthropic.Anthropic(
        api_key=AGENTROUTER_API_KEY,
        base_url=AGENTROUTER_BASE_URL,
        timeout=35.0,
    )

    working_messages = list(messages)
    final_text = ""
    max_iterations = 4

    for iteration in range(max_iterations):
        current_messages = list(working_messages)

        def _sync_create():
            return client.messages.create(
                model=AGENTROUTER_MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=current_messages,
            )

        try:
            resp = await loop.run_in_executor(None, _sync_create)
            raw_text = "".join(
                getattr(b, "text", "")
                for b in resp.content
                if hasattr(b, "text") or getattr(b, "type", "") == "text"
            ).strip()

            if not raw_text:
                logger.warning(f"[tool_loop] Iteration {iteration} returned empty text (stop_reason={resp.stop_reason}). Falling back to plain cascade.")
                text, provider = await call_llm_plain(messages, system_prompt, max_tokens)
                return sanitize_response(_strip_markdown(text)), provider, tools_called
        except Exception as e:
            logger.warning(f"[tool_loop] AgentRouter iteration {iteration} failed: {e}. Falling back to plain cascade.")
            text, provider = await call_llm_plain(messages, system_prompt, max_tokens)
            return sanitize_response(_strip_markdown(text)), provider, tools_called

        # Parse potential tool call
        match = TOOL_CALL_REGEX.search(raw_text)
        if match:
            tool_json_str = match.group(1).strip()
            try:
                tool_data = json.loads(tool_json_str)
                tool_name = tool_data.get("name")
                params = tool_data.get("parameters", {})
                logger.info(f"[tool_loop] Tool call: {tool_name} with params={params}")

                tool_result = execute_tool(tool_name, params, session)
                tools_called.append(tool_name)

                working_messages.append({"role": "assistant", "content": raw_text})
                working_messages.append({"role": "user", "content": f"<tool_result>\n{tool_result}\n</tool_result>"})
                continue
            except json.JSONDecodeError as err:
                logger.warning(f"[tool_loop] JSON decode error in tool_call: {err}")
                final_text = TOOL_CALL_REGEX.sub("", raw_text).strip()
                break
        else:
            # Detect bare JSON tool call (LLM forgot the <tool_call> tags)
            bare_match = _BARE_JSON_TOOL_REGEX.search(raw_text)
            if bare_match:
                try:
                    tool_data = json.loads(bare_match.group(0))
                    tool_name = tool_data.get("name")
                    params = tool_data.get("parameters", {})
                    logger.info(f"[tool_loop] Bare JSON tool call (no tags): {tool_name} params={params}")
                    tool_result = execute_tool(tool_name, params, session)
                    tools_called.append(tool_name)
                    working_messages.append({"role": "assistant", "content": raw_text})
                    working_messages.append({
                        "role": "user",
                        "content": (
                            f"<tool_result>\n{tool_result}\n</tool_result>\n"
                            "Now give the seller a natural spoken answer using this data. "
                            "No JSON, no tool syntax — only conversational speech."
                        ),
                    })
                    continue
                except (json.JSONDecodeError, ValueError) as err:
                    logger.warning(f"[tool_loop] Bare JSON parse failed: {err}")

            final_text = raw_text
            break

    # If max_iterations exhausted with no spoken answer, run plain fallback
    if not final_text:
        logger.warning("[tool_loop] max_iterations hit without final answer; running plain fallback")
        try:
            text, provider = await call_llm_plain(messages, system_prompt, max_tokens)
            return sanitize_response(_strip_markdown(text)), provider, tools_called
        except Exception:
            pass
        return (
            "I have the data pulled up. Could you share more about what you're seeing with your sales?",
            "fallback",
            tools_called,
        )

    return sanitize_response(_strip_markdown(final_text)), "agentrouter+react_tools", tools_called
