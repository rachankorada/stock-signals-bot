"""Start the app: API, website and live engine in one process.

Run:  python -m backend.run
"""
import logging
import os

import uvicorn

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run("backend.api.main:app", host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")), log_level="info")
