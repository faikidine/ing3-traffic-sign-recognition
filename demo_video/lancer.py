"""Lance le lecteur vidéo sur http://127.0.0.1:8501."""

import uvicorn
from app import app


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8501, access_log=False)
