"""
voice.py — Speech-to-Text (STT) and Text-to-Speech (TTS) audio pipeline.
"""

import os
import re
import asyncio
import tempfile
from app.config import logger, DEFAULT_VOICE

# Monkey-patch PyAV: newer versions (>=18) removed metadata_errors kwarg,
# which breaks faster-whisper's decode_audio().
try:
    import av
    _orig_av_open = av.open
    def _safe_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)
    av.open = _safe_av_open
except Exception:
    pass

_whisper_model = None

def get_whisper_model():
    """Lazy loader for faster-whisper base model (int8 CPU)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        logger.info("[stt] Loading faster-whisper 'base' model (int8 CPU)...")
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


async def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribes raw WebM/WAV audio payload to text using faster-whisper."""
    loop = asyncio.get_running_loop()

    def _sync_transcribe():
        if not audio_bytes or len(audio_bytes) < 100:
            return ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
            f.write(audio_bytes)
            input_path = f.name
        try:
            model = get_whisper_model()
            segments, _ = model.transcribe(input_path, beam_size=1, vad_filter=True, language="en")
            text = " ".join(seg.text for seg in segments).strip()
            logger.info(f"[stt] Transcribed ({len(audio_bytes)} bytes) -> '{text}'")
            return text
        finally:
            if os.path.exists(input_path):
                try:
                    os.remove(input_path)
                except OSError:
                    pass

    return await loop.run_in_executor(None, _sync_transcribe)


def normalize_for_tts(text: str) -> str:
    """Formats Indian currency and e-commerce abbreviations for natural spoken audio."""
    def rupees_spoken(m):
        amt_str = m.group(1).replace(",", "")
        try:
            amt = int(amt_str)
        except ValueError:
            return m.group(0)
        if amt >= 100000:
            lakhs = amt / 100000
            if lakhs == int(lakhs):
                return f"{int(lakhs)} lakh rupees"
            else:
                return f"{lakhs:.1f} lakh rupees"
        elif amt >= 1000:
            thousands = amt / 1000
            if thousands == int(thousands):
                return f"{int(thousands)} thousand rupees"
            else:
                return f"{thousands:.1f} thousand rupees"
        else:
            return f"{amt} rupees"

    text = re.sub(r"₹([\d,]+)", rupees_spoken, text)
    text = text.replace("≤", "under ").replace("→", " to ").replace("->", " to ")
    text = re.sub(r"\bOOS\b", "out of stock", text)
    text = re.sub(r"\bSKUs\b", "products", text)
    text = re.sub(r"\bSKU\b", "product", text)
    text = re.sub(r"\bSFC\b", "Snapdeal Fulfilment Centre", text)
    text = re.sub(r"\bKAM\b", "Key Account Manager", text)
    return text


async def generate_tts(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Synthesizes Indian English neural audio using edge-tts."""
    import edge_tts
    if not text or not text.strip():
        return b""
    spoken_text = normalize_for_tts(text.strip())
    communicate = edge_tts.Communicate(spoken_text, voice=voice, rate="+6%")
    chunks = [chunk["data"] async for chunk in communicate.stream() if chunk["type"] == "audio"]
    return b"".join(chunks)
