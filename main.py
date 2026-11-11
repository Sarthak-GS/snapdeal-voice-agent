"""
main.py — Root application entrypoint.

Run via Uvicorn:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
or:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
or directly:
    python main.py
"""

from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
