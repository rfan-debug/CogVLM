import os

import uvicorn


RANK = int(os.environ.get('RANK', 0))

if __name__ == "__main__":
    uvicorn.run("your_app_module:app", host="0.0.0.0", port=5000 + RANK)

