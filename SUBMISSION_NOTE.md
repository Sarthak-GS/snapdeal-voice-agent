# Snapdeal AI Virtual KAM's Submission Note

> 🚀 **Live Production Deployment**: [https://snapdeal-voice-agent-production.up.railway.app/](https://snapdeal-voice-agent-production.up.railway.app/)  
> *(Hosted on Railway with full HTTPS microphone permissions & real-time WebAudio VAD)*

## 1. Project Overview
**KIRAN** is an autonomous, voice-first AI Virtual Key Account Manager (KAM) designed for Snapdeal sellers experiencing business distress. It conducts natural, two-way voice conversations. It analyzes real-time account data, identifies root causes behind declining sales, proposes margin-safe recovery strategies, executes catalog changes live upon verbal consent, and generates a structured 7-Day Pilot Blueprint.

---

## 2. Technology Stack

| Task / Component | Technology Used |
|---|---|
| **Primary LLM Reasoning** | `gpt-5.6-sol` via AgentRouter (`anthropic.Anthropic` SDK) |
| **Agentic Tool Calling** | Custom ReAct Tool Execution Loop with XML/JSON parsing & fallback sanitization |
| **Speech-to-Text (STT)** | `faster-whisper` (`base` model, int8 quantization, CPU via PyAV) |
| **Text-to-Speech (TTS)** | `edge-tts` (`en-IN-NeerjaNeural`) + Indian English pronunciation normalizer |
| **Voice Activity Detection (VAD)** | Web Audio API (`AudioContext`, `AnalyserNode`, RMS energy calibration) |
| **Vector RAG Store** | Dual Dense (`nomic-embed-text`) + Sub-Word TF-IDF Cosine Similarity (`.npz`) |
| **Backend API** | FastAPI + Uvicorn + Pydantic v2 + Python 3.10 |
| **Cloud Deployment** | Railway (`amd64` Linux Container, continuous execution, Automatic HTTPS) |
| **Frontend UI** | Single-page HTML5, CSS3, Vanilla JavaScript (zero external framework bloat) |

---

## 3. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Client ["Client Browser (UI)"]
        A["Seller Speaks / Types"] --> B["WebAudio VAD Engine\n(Noise Floor Calibration + 1.4s Silence Hang)"]
        B --> C["MediaRecorder (WebM Audio Chunk)"]
        J["TTS Audio Stream\n(Echo Blocker Pauses VAD)"] --> K["HTML5 Audio Playback / Speaker"]
    end

    subgraph Server ["FastAPI Backend (Port 8000)"]
        C -->|POST /api/stt| D["faster-whisper STT\n(PyAV CPU In-Memory Decoding)"]
        D --> E["KIRAN ReAct Agent Loop\n(gpt-5.6-sol via sync Anthropic SDK)"]

        E <-->|Tool Call / Result| F["Custom Tool Dispatcher"]
        F <-->|Semantic Cosine Search| G["Vector Store (Playbook SOPs)\n(Dense + TF-IDF .npz)"]
        F <-->|Live Account Mutation| H["Seller Account State (In-Memory KB)"]

        E -->|Clean Spoken Text| I["edge-tts Neural Synthesizer\n(Indian Number/Acronym Normalizer)"]
        I -->|Audio Stream| J
    end
```

---

## 4. Seller Knowledge Base & Data Structure

Each seller knowledge base (`data/{seller_id}_kb.json`) contains:
1. **Seller Profile**: ID, store name, category, tier (Silver/Gold), rating, catalog size.
2. **MoM Performance**: Revenue, orders, average selling price (ASP), cancellation rate, return rate, conversion rate, out-of-stock (OOS) count.
3. **SKU Unit Economics**: Cost price, selling price, competitor average, platform fee, stock levels, lead time, and individual unit margins.
4. **Cancellation Breakdown**: Root causes categorized by post-order stockout, delivery delay, pricing, and buyer mind change.
5. **Market Context**: Category growth rate, active competitors, festive window status, and seller campaign enrollment.
6. **Causal Hypotheses & Playbooks**: Data-backed root causes, proposed corrective actions, stop conditions, and SOP protocols.

### 3 Realistic Synthetic Sellers

| Seller | Category & Tier | Revenue Trend | Primary Root Causes | Live Account Fixes |
|---|---|---|---|---|
| **Rahul Sharma** (Pune) | Mobile Accessories · *Silver* | ₹2,00,000 → ₹1,35,000 (**-32.5%**) | • 18 of 25 SKUs OOS (+13 new)<br>• 42% cancellations from post-order stockouts<br>• USB-C Cable priced 20% above market | Pause ads on 18 OOS SKUs, align prices to market, enroll in Festive Deals |
| **Priya Mehta** (Mumbai) | Women's Fashion · *Gold* | ₹3,20,000 → ₹2,10,000 (**-34.4%**) | • Kurti & Lehenga OOS during peak wedding season<br>• 32% cancellations from size mismatches<br>• Listing score 44 on Banarasi Saree | Pause OOS ads, upload size chart guide, enroll in Wedding Festive Deals |
| **Mohammed Farouk** (Delhi) | Home Electronics · *Gold* | ₹5,80,000 → ₹4,20,000 (**-27.6%**) | • 32" LED TV return spike (18%) from remote/WiFi defect<br>• 43% cancellations due to >7 day delivery delays<br>• Missed Big Diwali Sale campaign window | Pause LED TV ads, route high-velocity SKUs to Snapdeal Fulfilment Centre (SFC) |

---

## 5. RAG for SOPs & Custom Agent Tools

Rather than bloating the prompt with static text, RAG is selectively used to retrieve relevant Standard Operating Procedures (SOPs) only when needed.

### Semantic RAG for Playbooks
- **Storage**: Pre-computed L2-normalized sub-word TF-IDF + dense vectors stored in `data/{seller_id}_embeddings.npz`.
- **Retrieval**: Sub-millisecond cosine similarity matching query n-grams to indexed SOP tags and condition rules (`pb_oos_emergency`, `pb_cancellation_triage`, `pb_pricing_correction`, etc.).

### Custom Functions Built
1. `get_seller_metrics(metric_group)`: Fetches live revenue, orders, cancellations, SKU unit economics, or category market benchmarks.
2. `analyze_issue(issue_type)`: Performs targeted diagnostics on `inventory`, `pricing`, `cancellations`, `returns`, `delivery`, or `visibility`.
3. `get_recommendations(query)`: Executes semantic RAG search across the seller's SOP playbooks.
4. `execute_account_action(action)`: Applies live account mutations upon seller verbal consent (e.g., pausing ads on OOS listings, matching market prices, enrolling in festive deals).
5. `generate_call_summary(primary_issues, actions_agreed)`: Generates the structured 7-Day Pilot Blueprint with metrics, timelines, and stop conditions.

---

## 6. Key Technical Details & Engineering Decisions

- **Sync Anthropic SDK in Executor**: AgentRouter's Cloudflare anti-bot gate blocks `AsyncAnthropic` with a `401 unauthorized_client_error`. We use synchronous `anthropic.Anthropic` executed inside `asyncio.get_running_loop().run_in_executor()` to prevent blocking while authenticating cleanly.
- **Bare-JSON Tool Interceptor & Sanitizer**: When the LLM outputs bare tool JSON without `<tool_call>` tags, the agent loop catches it, executes the tool, feeds the result back, and forces natural spoken speech. `sanitize_response()` strips any leaked JSON or meta-tokens before reaching TTS or the user.
- **WebAudio VAD & Echo Prevention**: WebAudio measures RMS energy in the browser, auto-calibrating background noise during the first 500ms. When KIRAN speaks, `vad.pause()` halts microphone capture to eliminate acoustic echo feedback.
- **Headless CPU STT**: `faster-whisper` runs locally on CPU with `int8` quantization using PyAV for direct in-memory WebM decoding, avoiding external `ffmpeg` binary dependencies and cloud latency.
- **PyAV 18+ Audio Decoding Compatibility & Monkey-Patch**: `faster-whisper` decodes incoming audio chunks using PyAV (`av.open(..., metadata_errors="ignore")`). PyAV versions 18+ deprecated and removed the `metadata_errors` keyword parameter, causing `TypeError: open() got an unexpected keyword argument 'metadata_errors'` on newer Python environments (such as Python 3.12/3.13 default cloud builders). We introduced a runtime interceptor monkey-patch in `app/voice.py` that safely strips `metadata_errors` before invoking underlying C-bindings, combined with pinning `av<18.0.0` in `requirements.txt` for 100% cross-environment robustness.
- **Indian English Spoken Normalization**: Regular expressions convert technical and financial strings into natural conversational speech (e.g., `₹2,00,000` → `2 lakh rupees`, `OOS` → `out of stock`, `SKU` → `product`, `SFC` → `Snapdeal Fulfilment Centre`).
- **Dynamic Health Score Engine**: Deterministic algorithm calculates account health (0–100) and risk band (`healthy`, `watch`, `at_risk`, `critical`) from live metrics, updating in real time when actions are applied.

---

## 7. How to Access & Run

### Live Cloud Deployment
Access the live interactive voice application immediately at:  
👉 **[https://snapdeal-voice-agent-production.up.railway.app/](https://snapdeal-voice-agent-production.up.railway.app/)**  
*(Requires microphone permission in browser. Fully functional with real-time WebAudio VAD and live ReAct tools).*

### Local First-Time Setup (Optional)

```bash
# 1. Clone repository & navigate to directory
cd "/home/sarthak/Documents/Job assignment/snapdeal/voice-kam"

# 2. Activate conda environment
conda activate projects_env

# 3. (Optional) Generate/refresh seller knowledge bases & embeddings
python3 generate_data.py

# 4. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to: **`http://localhost:8000`**

---

## 8. How to Test the Demo & Sample Questions

1. **Select a Seller**: Click on Rahul Sharma, Priya Mehta, or Mohammed Farouk on the pre-call screen. Notice the sidebar updates with that seller's live catalog, revenue drop, and health score.
2. **Start Call**: Click **"Start Call"**. KIRAN warmly greets the seller by name, mentions their store, and invites them to discuss their account.
3. **Voice or Text**: Speak naturally using automatic VAD (Auto mode) or type in the input bar.
4. **Approve Actions**: When KIRAN suggests an action, give verbal consent (e.g., *"Yes, please pause the ads and enroll me"*). Watch the live badges update.
5. **End Call**: Click **"End Call"** or say *"Goodbye"*. KIRAN speaks a warm 2-sentence closing and renders the complete **7-Day Pilot Blueprint** with an **"Apply Day 1 Actions"** 1-click button.

### Sample Questions to Try

#### Seller 1: Rahul Sharma (Mobile Accessories)
- *"Why did my sales drop so much in November?"*
- *"Why are my cancellations so high at 14%?"*
- *"How can I improve my listing quality score?"*
- *"How can I capture traffic in this festive season?"*
- *"Yes, please go ahead and pause the OOS ads and match prices to market."*

#### Seller 2: Priya Mehta (Women's Fashion)
- *"Why did my revenue fall even though the wedding season is active?"*
- *"Why is my return rate at 12%?"*
- *"How can I fix the size mismatch cancellations?"*
- *"Can you enroll my active catalog in the wedding festive deals?"*

#### Seller 3: Mohammed Farouk (Home Electronics)
- *"Why are customers returning my 32-inch LED TV?"*
- *"Why are 43% of my cancellations caused by delivery delays?"*
- *"How do I recover from missing the Big Diwali Sale?"*
- *"Yes, please pause ads on the TV and enroll my products in Snapdeal Fulfilment Centre."*
