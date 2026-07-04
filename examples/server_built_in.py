"""examples/server_built_in.py -- run the built-in SSE server (zero code).

Path A (built-in / config-only). Requires the api extra::

    pip install koboi-agent[api]
    python examples/server_built_in.py

Then stream a turn::

    curl -N -X POST http://127.0.0.1:8000/v1/chat/stream \\
      -H 'Content-Type: application/json' \\
      -d '{"message": "hello"}'
"""

import uvicorn

from koboi.config import Config
from koboi.server import create_app

if __name__ == "__main__":
    config = Config.from_yaml("configs/server_simple.yaml")
    app = create_app(config)
    uvicorn.run(app, host="127.0.0.1", port=8000)
