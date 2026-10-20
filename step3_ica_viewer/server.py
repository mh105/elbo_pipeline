"""Small loopback-only viewer service with revisioned subject-local autosaves."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit
import webbrowser

from .storage import ReviewConflict, ensure_cache, read_review, save_review


_SERVERS = {}
_SERVERS_LOCK = threading.Lock()


class _ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, viewer_dir, cache_dir, **kwargs):
        self.viewer_dir = viewer_dir
        self.cache_dir = cache_dir
        super().__init__(*args, directory=str(cache_dir), **kwargs)

    def log_message(self, format, *args):
        pass

    def _allowed(self):
        expected = f"127.0.0.1:{self.server.server_port}"
        if self.headers.get("Host") != expected:
            self._json(403, {"error": "This service accepts its own loopback address only."})
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin != "http://" + expected:
            self._json(403, {"error": "Cross-origin review access is not allowed."})
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self._json(403, {"error": "Cross-site review access is not allowed."})
            return False
        return True

    def _json(self, status, value):
        content = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if not self._allowed():
            return
        path = urlsplit(self.path).path
        if path == "/api/review":
            try:
                self._json(200, read_review(self.viewer_dir))
            except (ValueError, OSError, KeyError, TypeError) as error:
                self._json(400, {"error": str(error)})
        elif path.startswith("/api/"):
            self._json(404, {"error": "Unknown review API route."})
        else:
            target = Path(self.translate_path(self.path)).resolve()
            if not target.is_relative_to(self.cache_dir):
                self._json(403, {"error": "Only viewer cache assets can be served."})
                return
            super().do_GET()

    def do_HEAD(self):
        if not self._allowed():
            return
        target = Path(self.translate_path(self.path)).resolve()
        if not target.is_relative_to(self.cache_dir):
            self._json(403, {"error": "Only viewer cache assets can be served."})
            return
        super().do_HEAD()

    def do_PUT(self):
        if not self._allowed():
            return
        if urlsplit(self.path).path != "/api/review":
            self._json(404, {"error": "Unknown review API route."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("Invalid review request length.")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict) or set(request) != {"exclude", "expected_revision"}:
                raise ValueError("Send exclude and expected_revision only.")
            review = save_review(self.viewer_dir, request["exclude"], request["expected_revision"])
        except ReviewConflict as error:
            self._json(409, {"error": str(error), "review": read_review(self.viewer_dir)})
        except (ValueError, OSError, KeyError, TypeError) as error:
            self._json(400, {"error": str(error)})
        else:
            self._json(200, review)


def open_viewer(viewer_dir, *, open_browser=True):
    """Serve one bundle, returning its URL; Python must remain running for saving."""
    viewer_dir = Path(viewer_dir).expanduser().resolve(strict=True)
    cache_dir = ensure_cache(viewer_dir).resolve()
    with _SERVERS_LOCK:
        existing = _SERVERS.get(viewer_dir)
        if existing is not None and existing[2] != cache_dir:
            existing[0].shutdown()
            existing[0].server_close()
            existing[1].join(timeout=2)
            existing = None
        if existing is None:
            handler = partial(_ViewerHandler, viewer_dir=viewer_dir, cache_dir=cache_dir)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            server.daemon_threads = True
            worker = threading.Thread(target=server.serve_forever, daemon=True,
                                      name="ICA viewer")
            worker.start()
            existing = (server, worker, cache_dir)
            _SERVERS[viewer_dir] = existing
        server, _, _ = existing
        url = f"http://127.0.0.1:{server.server_port}/index.html"
    if open_browser:
        webbrowser.open(url)
    return url


def close_viewer(viewer_dir):
    """Stop this process's server for a bundle without changing saved decisions."""
    viewer_dir = Path(viewer_dir).expanduser().resolve()
    with _SERVERS_LOCK:
        existing = _SERVERS.pop(viewer_dir, None)
    if existing is not None:
        server, worker, _ = existing
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
