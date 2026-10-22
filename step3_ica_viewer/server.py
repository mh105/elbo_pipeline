"""Small loopback-only viewer service with revisioned subject-local autosaves."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit
import webbrowser

from .storage import (ReviewConflict, create_review_version, ensure_cache,
                      read_review, save_review, set_stable_review_version)


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
                query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                if set(query) - {"version"} or len(query.get("version", [])) > 1:
                    raise ValueError("Send a single labeling version only.")
                version = query["version"][0] if "version" in query else None
                self._json(200, read_review(self.viewer_dir, version=version))
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
        self._mutate_review()

    def do_POST(self):
        if urlsplit(self.path).path == "/api/envelope":
            try:
                self._calculate_envelope()
            except (BrokenPipeError, ConnectionResetError):
                # A new scroll request can cancel the previous read-only view.
                pass
        else:
            self._mutate_review()

    def _calculate_envelope(self):
        if not self._allowed():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("Invalid envelope request length.")
            request = json.loads(self.rfile.read(length))
            if (not isinstance(request, dict) or
                    set(request) != {"start_s", "end_s", "exclude", "before_exclude"}):
                raise ValueError("Send start_s, end_s, exclude, and before_exclude only.")
            with self.server.envelope_lock:
                if self.server.envelope_data is None:
                    from .envelope import EnvelopeData
                    self.server.envelope_data = EnvelopeData(self.cache_dir)
                envelope = self.server.envelope_data
            result = envelope.calculate(**request)
        except (ValueError, OSError, KeyError, TypeError) as error:
            self._json(400, {"error": str(error)})
        else:
            self._json(200, result)

    def _mutate_review(self):
        if not self._allowed():
            return
        path = urlsplit(self.path).path
        routes = {("PUT", "/api/review"), ("PUT", "/api/review/stable"),
                  ("POST", "/api/review/versions")}
        if (self.command, path) not in routes:
            self._json(404, {"error": "Unknown review API route."})
            return
        version = None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("Invalid review request length.")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("Send a JSON review request object.")
            version = request.get("version")
            if "version" in request and not isinstance(version, str):
                raise ValueError("Send a labeling version ID string.")
            if path == "/api/review":
                if set(request) not in ({"exclude", "expected_revision"},
                                        {"exclude", "expected_revision", "version"}):
                    raise ValueError("Send exclude, expected_revision, and optionally version only.")
                review = save_review(self.viewer_dir, request["exclude"], request["expected_revision"],
                                     version=version)
            elif path == "/api/review/versions":
                if set(request) != {"expected_revision"}:
                    raise ValueError("Send expected_revision only.")
                review = create_review_version(self.viewer_dir, request["expected_revision"])
            else:
                if set(request) != {"version", "expected_revision"}:
                    raise ValueError("Send version and expected_revision only.")
                review = set_stable_review_version(self.viewer_dir, version, request["expected_revision"])
        except ReviewConflict as error:
            self._json(409, {"error": str(error), "review": read_review(self.viewer_dir, version=version)})
        except (ValueError, OSError, KeyError, TypeError) as error:
            self._json(400, {"error": str(error)})
        else:
            self._json(200, review)


def open_viewer(viewer_dir, *, open_browser=True):
    """Serve one bundle, returning its URL; Python must remain running for saving."""
    viewer_dir = Path(viewer_dir).expanduser().resolve(strict=True)
    cache_dir = ensure_cache(viewer_dir).resolve()
    complete = (cache_dir / ".complete.json").stat()
    cache_stamp = (complete.st_ino, complete.st_mtime_ns)
    with _SERVERS_LOCK:
        existing = _SERVERS.get(viewer_dir)
        if existing is not None and (existing[2] != cache_dir or
                                     getattr(existing[0], "cache_stamp", None) != cache_stamp):
            existing[0].shutdown()
            existing[0].server_close()
            existing[1].join(timeout=2)
            existing = None
        if existing is None:
            handler = partial(_ViewerHandler, viewer_dir=viewer_dir, cache_dir=cache_dir)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            server.daemon_threads = True
            server.envelope_lock = threading.Lock()
            server.envelope_data = None
            server.cache_stamp = cache_stamp
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
