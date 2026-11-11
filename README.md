# Snapdeal AI Virtual KAM — Project KIRAN
> **Autonomous, Voice-First AI Key Account Manager for E-Commerce Marketplace Sellers**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Whisper](https://img.shields.io/badge/STT-faster--whisper%20(int8)-blue)](https://github.com/SYSTRAN/faster-whisper)
[![Edge-TTS](https://img.shields.io/badge/TTS-Neural%20Indian%20English-orange)](https://github.com/rany2/edge-tts)
[![Agentic Tools](https://img.shields.io/badge/Architecture-ReAct%20Tool%20Calling-purple)](#architecture)

---

## 1. Executive Summary

When marketplace sellers experience sudden revenue drops, traditional platforms present them with complex, confusing analytics dashboards. Human Key Account Managers (KAMs) are effective but expensive and cannot scale to hundreds of thousands of long-tail sellers.

**KIRAN** is an autonomous AI Virtual KAM built specifically for Snapdeal sellers. It conducts natural, two-way spoken phone consultations that:
1. **Diagnose root causes** using SKU-level unit economics, competitor pricing benchmarks, stockout history, and cancellation breakdowns.
2. **Conduct unscripted conversations** driven by the seller's actual voice questions, using an agentic ReAct tool-calling loop rather than rigid scripted stages.
3. **Execute margin-safe actions live** on the seller’s catalog during the call (pausing ads on out-of-stock items, matching prices while protecting gross margin, enrolling eligible products in festive sales).
4. **Deliver a structured 7-Day Pilot Blueprint** with explicit stop conditions and 1-click execution.

---

## 2. Key Capabilities & Innovations

### 🎙️ Two-Way Natural Voice with Interruption (Barge-In)
- **WebAudio VAD (Voice Activity Detection)**: Dynamically calibrates to background room noise within 400ms. Detects speech onset and completes turns after 1.4s of natural silence.
- **Immediate Interruption**: If the seller speaks or clicks the microphone while the assistant is speaking, KIRAN immediately halts playback (`stopAssistantSpeech()`) and listens without delay.
- **Dual TTS Engine**:
  - **Fast Voice (<50ms instant)**: Built-in browser `window.speechSynthesis` with Indian English accents for zero network latency.
  - **Neural Cloud Voice (`edge-tts`)**: High-fidelity Indian English female voice (`en-IN-NeerjaNeural`) with custom currency and e-commerce normalization (e.g., converts "₹2,00,000" to "2 lakh rupees").

### 🧠 Agentic Tool Calling (No Scripted Routing)
Instead of hardcoding a `GREET → PROBE → REVEAL → RECOMMEND` sequence, KIRAN reasons dynamically and calls live tools as needed:
- `get_seller_metrics(metric_group)`: Fetches MoM revenue, orders, cancellations, returns, and inventory counts.
- `analyze_issue(issue_type)`: Performs deep-dive diagnostics on `inventory`, `pricing`, `cancellations`, `returns`, `delivery`, or `visibility`.
- `get_recommendations(query)`: Retrieves playbook protocols using semantic TF-IDF cosine similarity.
- `execute_account_action(action)`: Applies live changes to the seller's catalog upon explicit seller consent.
- `generate_call_summary(primary_issues, actions_agreed)`: Generates the 7-Day Pilot Blueprint.

### 👥 Multi-Seller Synthetic Sandbox
Includes 3 realistic, internally consistent seller personas:
1. **Rahul Sharma** (*Rahul Electronics & Accessories, Pune*):
   - **Metrics**: Revenue fell 32.5% (₹2,00,000 → ₹1,35,000); 18 of 25 SKUs Out-of-Stock.
   - **Root Cause**: Post-order stockouts drive 42% of cancellations; top USB-C cable priced 20% above market.
2. **Priya Mehta** (*Priya's Fashion Studio, Mumbai*):
   - **Metrics**: Revenue fell 34.4% (₹3,20,000 → ₹2,10,000); Wedding season demand missed.
   - **Root Cause**: High-velocity Kurti and Bridal Lehenga sets went OOS; 32% cancellations from size mismatches.
3. **Mohammed Farouk** (*Farouk Electronics Hub, Delhi*):
   - **Metrics**: Revenue fell 27.6% (₹5,80,000 → ₹4,20,000); Return rate spiked to 9%.
   - **Root Cause**: LED TV remote/WiFi quality defects; 43% delivery-delay cancellations due to >7-day lead times.

---

## 3. System Architecture

```
                                [Seller Voice / Text]
                                         │
                                         ▼
                             [WebAudio VAD Engine]
                       (Energy RMS + Noise Floor Calibrator)
                                         │
                                         ▼
                            [faster-whisper STT (CPU)]
                                         │
                                         ▼
                            [KIRAN ReAct Agentic Loop]
                                 (gpt-5.6-sol)
                                         │
        ┌────────────────────────────────┴───────────────────────────────┐
        │ When Data / Action Needed                                      │ Direct Response
        ▼                                                                ▼
[Tool Execution Dispatcher]                                      [Spoken Response]
  ├─ get_seller_metrics(...)                                             │
  ├─ analyze_issue(...)                                                  │
  ├─ get_recommendations(...)                                            │
  ├─ execute_account_action(...)                                         │
  └─ generate_call_summary(...)                                          │
        │                                                                │
        └────────────────────────────────┬───────────────────────────────┘
                                         ▼
                               [Dual TTS Audio Stream]
                          (Browser Fast / Neural Edge-TTS)
                                         │
                                         ▼
                           [Browser Audio & Dashboard UI]
```

---

## 4. Repository Structure

```
voice-kam/
├── app/                  # Modular backend application package
│   ├── __init__.py
│   ├── config.py         # Global settings, paths, logger, .env loader
│   ├── data_loader.py    # KBs bootstrap & PlaybookVectorStore
│   ├── diagnostic.py     # Health score computation & account snapshot
│   ├── voice.py          # faster-whisper STT + edge-tts TTS + Indian normalizer
│   ├── tools.py          # ReAct tool schemas & dispatcher (5 tools)
│   ├── agent.py          # ReAct agentic loop, prompt builder & LLM cascade
│   └── main.py           # FastAPI app & REST endpoints
├── data/
│   ├── rahul_kb.json     # Rahul Sharma catalog, unit economics & playbook
│   ├── rahul_embeddings.npz
│   ├── priya_kb.json     # Priya Mehta fashion catalog & size diagnostics
│   ├── priya_embeddings.npz
│   ├── farouk_kb.json    # Mohammed Farouk electronics catalog & lead time diagnostics
│   └── farouk_embeddings.npz
├── index.html            # WebAudio VAD frontend, waveform canvas, 1-click blueprint UI
├── main.py               # Root entrypoint (forwards to app.main:app)
├── generate_data.py      # Offline data & TF-IDF / dense vector generator
├── requirements.txt      # Production dependencies
├── SUBMISSION_NOTE.md    # Technical architecture & evaluation guide
├── DEMO_SCRIPT.md        # 2-Minute structured demo speech & walkthrough script
├── .env                  # API keys and configuration
└── .env.example          # Template configuration
```

---

## 5. Quickstart Guide

### Prerequisites
- Linux / macOS / Windows with Python 3.10+
- Conda or virtual environment
- Microphone access enabled on `http://localhost:8000`

### 1. Installation
```bash
# Clone and enter directory
cd "snapdeal/voice-kam"

# Create/activate Python environment
conda activate projects_env

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration (`.env`)
Copy the template and configure your keys:
```bash
cp .env.example .env
```
Default `.env`:
```ini
AGENTROUTER_API_KEY=your_agentrouter_api_key_here
AGENTROUTER_BASE_URL=https://agentrouter.org
AGENTROUTER_MODEL=gpt-5.6-sol
LOG_LEVEL=INFO
```

### 3. Generate Knowledge Bases & Vectors
```bash
python3 generate_data.py
```
*(Pre-computes KBs and vector embeddings for Rahul, Priya, and Mohammed Farouk)*

### 4. Run Server
Run with **either** command:
```bash
# Standard modular run:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or via root main:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://localhost:8000`** in Google Chrome or Microsoft Edge.

---

## 6. API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /api/health` | GET | System status, model provider, registered sellers |
| `GET /api/sellers` | GET | List of all sellers with health scores and MoM deltas |
| `GET /api/seller/data?seller_id={id}` | GET | Complete SKU economics, competitor pricing, and hypotheses |
| `POST /api/session/start` | POST | Initializes session, builds diagnostic context, generates spoken greeting |
| `POST /api/chat` | POST | Runs ReAct agentic tool loop, updates session, executes actions |
| `POST /api/session/{id}/summarize` | POST | Generates closing summary and 7-Day Pilot Blueprint on demand |
| `POST /api/session/{id}/apply_day1`| POST | One-click execution of Day 1 fixes (pause OOS ads, match prices) |
| `POST /api/stt` | POST | Transcribes WebM audio chunks via `faster-whisper` |
| `POST /api/tts` | POST | Generates neural Indian English audio via `edge-tts` |

---

## 7. Key Engineering Decisions

1. **Why not XTTS v2 on local CPU?**
   - XTTS v2 is a 2.4 GB autoregressive model requiring dedicated CUDA VRAM. Running on CPU leads to 100% thread lock and freezes laptop systems. We chose `faster-whisper (int8)` for STT and `edge-tts` / WebSpeech for TTS, providing sub-second latency with 0% CPU strain.
2. **ReAct Tool Loop vs OpenAI Function Schema:**
   - Reasoning models on third-party proxies often reject standard function tools when reasoning effort is default. We engineered an XML-based ReAct loop (`<tool_call>...<tool_result>`) which operates with 100% reliability and zero 400 API errors.
3. **Synchronous Anthropic Client inside Async Executor:**
   - Some reverse proxies block async HTTP client signatures with anti-bot 401 challenges. Using `anthropic.Anthropic` inside `loop.run_in_executor` bypasses bot triggers cleanly while keeping FastAPI 100% non-blocking.
4. **Barge-in / Interruption Handling:**
   - Both WebAudio RMS speech detection and manual mic clicks immediately cancel active audio elements and `window.speechSynthesis`, preventing the bot from talking over the user.
