"""Request logging: one line per request with a request ID, status, duration, body and error.

The middleware and the body dependency are registered once in app/main.py, so every endpoint,
including ones added later, is logged without any per-route code.

Usage: setup_logging() at startup; FastAPI(dependencies=[Depends(capture_request_body)]);
app.add_middleware(RequestLoggingMiddleware).
"""
import json
import logging
import re
import time
import uuid
from contextvars import ContextVar
from typing import Any, Optional

from fastapi import Request
from starlette.datastructures import UploadFile

from app.config import settings
from app.utils.generic import preview_text

logger = logging.getLogger(__name__)

# ID of the request being served; log lines written outside a request show "-"
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# A caller-sent ID is reused only if it is short and safe, so it can't inject fake log lines
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
_REQUEST_ID_HEADER = b"x-request-id"

# Error bodies are small JSON; the cap stops a 422 echoing a huge input from flooding the log
ERROR_BODY_LOG_LIMIT = 2000
# Upload/update form values (summary, metadata) can be long, so each is cut to this many characters
FORM_FIELD_PREVIEW_CHARS = 50

LOG_FILE = "app.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s"

# Key in request.state (scope["state"]) where the dependency leaves the body for the middleware
_BODY_LOG_KEY = "body_log"


class RequestIdFilter(logging.Filter):
    """Adds the current request ID to every log record as record.request_id"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging() -> None:
    """Configure root logging once: app.log + console at INFO, with the request ID on each line"""
    root = logging.getLogger()
    # Safe to call twice (uvicorn --reload, tests): our handlers are only added the first time
    if any(isinstance(f, RequestIdFilter) for handler in root.handlers for f in handler.filters):
        return

    formatter = logging.Formatter(LOG_FORMAT)
    for handler in (logging.FileHandler(LOG_FILE), logging.StreamHandler()):
        handler.setFormatter(formatter)
        handler.addFilter(RequestIdFilter())
        root.addHandler(handler)
    root.setLevel(logging.INFO)


def _single_line(text: str) -> str:
    """Escape line breaks so a logged value can never start a fake log line"""
    return text.replace("\r", "\\r").replace("\n", "\\n")


def _format_form_field(key: str, value: Any) -> str:
    # A file is logged by name and type only; its contents never reach the log
    if isinstance(value, UploadFile):
        return f"{key}=<{_single_line(value.filename or '')} ({value.content_type})>"
    return f"{key}={_single_line(preview_text(value, FORM_FIELD_PREVIEW_CHARS))}"


def _format_json_body(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")
    # Re-serialize compactly so a pretty-printed body still logs as one line
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":"))
    except ValueError:
        return _single_line(text)


async def capture_request_body(request: Request) -> None:
    """App-wide dependency: format the request body for the middleware's log line

    FastAPI parses the form/JSON before it runs dependencies and Starlette caches it on the
    request, so reading it here is not a second read, even for large uploads.
    """
    if not settings.LOG_REQUEST_BODY:
        return

    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        body = await request.body()
        if body:
            setattr(request.state, _BODY_LOG_KEY, _format_json_body(body))
    elif content_type.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
        form = await request.form()
        fields = [_format_form_field(key, value) for key, value in form.multi_items()]
        if fields:
            setattr(request.state, _BODY_LOG_KEY, ", ".join(fields))


def _request_id_from(scope: dict) -> str:
    # Reuse the caller's X-Request-ID (e.g. from Commons) so one ID traces a call across services
    for name, value in scope.get("headers", []):
        if name == _REQUEST_ID_HEADER:
            candidate = value.decode("latin-1")
            if _VALID_REQUEST_ID.fullmatch(candidate):
                return candidate
            break
    return uuid.uuid4().hex


def _request_line(scope: dict, status: int, started: float, detail: Optional[str] = None) -> str:
    target = scope.get("path", "")
    query = scope.get("query_string", b"").decode("latin-1")
    if query:
        target = f"{target}?{query}"
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    line = f"{scope.get('method', '')} {_single_line(target)} -> {status} ({elapsed_ms} ms)"
    body_log = scope.get("state", {}).get(_BODY_LOG_KEY)
    if body_log:
        line += f" | body: {body_log}"
    if detail:
        line += f" | {detail}"
    return line


class RequestLoggingMiddleware:
    """Plain ASGI middleware: sets the request ID and writes one log line per HTTP request"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # Lifespan and websocket traffic pass straight through
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id_from(scope)
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_code: Optional[int] = None
        error_body = bytearray()

        async def send_with_logging(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                # Return the ID so a caller can quote it when reporting a problem
                headers = list(message.get("headers", []))
                headers.append((_REQUEST_ID_HEADER, request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            elif message["type"] == "http.response.body" and status_code is not None and status_code >= 400:
                # Only error bodies are kept (first ERROR_BODY_LOG_LIMIT bytes); success bodies are never read
                room = ERROR_BODY_LOG_LIMIT - len(error_body)
                if room > 0:
                    error_body.extend(message.get("body", b"")[:room])
            await send(message)

        try:
            await self.app(scope, receive, send_with_logging)
        except Exception:
            # Unhandled error: log it with the traceback, then re-raise so FastAPI still returns 500
            logger.exception(_request_line(scope, status_code or 500, started, "unhandled error"))
            raise
        else:
            status = status_code or 500
            if status < 400:
                logger.info(_request_line(scope, status, started))
            else:
                error_text = _single_line(error_body.decode("utf-8", errors="replace"))
                level = logging.ERROR if status >= 500 else logging.WARNING
                logger.log(level, _request_line(scope, status, started, f"error: {error_text}"))
        finally:
            request_id_var.reset(token)
