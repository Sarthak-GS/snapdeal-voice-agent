"""
config.py — Global configuration, environment settings, and logger setup.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
HTML_FILE = ROOT_DIR / "index.html"
LOG_FILE = "/tmp/kam_server.log"

# Add ROOT_DIR to sys.path so imports like `generate_data` resolve
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load .env file
load_dotenv(ROOT_DIR / ".env")

# LLM Providers Configuration (Loaded from .env with fallback)
AGENTROUTER_API_KEY = os.getenv(
    "AGENTROUTER_API_KEY",
    "sk-u9zoqj81bgaHXMEzXjXe1egezoDLdT15olwWi7OFFeb0fF5N"
)
AGENTROUTER_BASE_URL = os.getenv("AGENTROUTER_BASE_URL", "https://agentrouter.org")
AGENTROUTER_MODEL = os.getenv("AGENTROUTER_MODEL", "gpt-5.6-sol")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"

# Token Headroom Limits
MAX_TOKENS = 4096
SUMMARY_MAX_TOKENS = 4096

# Voice Settings
DEFAULT_VOICE = "en-IN-NeerjaNeural"

# Logging Setup
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler()],
)
logger = logging.getLogger("kam")
