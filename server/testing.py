"""Test helpers: run the app on a real socket in a background thread (used by the WebSocket and load tests)."""

import socket
import threading
import time
from types import TracebackType

import uvicorn
from fastapi import FastAPI


class LiveServer:
    """``with LiveServer(app) as srv: connect(srv.ws_url)``. Binds an ephemeral port on 127.0.0.1."""

    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self.port: int = self._sock.getsockname()[1]
        self._server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
        self._thread = threading.Thread(target=lambda: self._server.run(sockets=[self._sock]), daemon=True)

    @property
    def http_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/ws"

    def __enter__(self) -> "LiveServer":
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(0.01)
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)
        self._sock.close()
