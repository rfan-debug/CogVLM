import os

import uvicorn


RANK = int(os.environ.get('RANK', 0))

if __name__ == "__main__":
    uvicorn.run("api_demo:app", host="0.0.0.0", port=7860 + RANK)

