""
================================================================================
AI Agent Lambda — Product-level Overview
================================================================================

What this Lambda does (in plain language):

1) Customer DTO (account summary data)
   - Routes calls to the customer DTO endpoint.
   - Adds helpful computed fields: average monthly per line, active line count.
   - Optionally builds a compact "snapshot" (tickets, next payments, flags).
   - Normalizes dates (epoch → ISO) and trims payloads to stay within token limits.
   - Caches DTOs briefly to cut latency and backend load.

2) Goodwill Credits (ex: add a data credit for a line)
   - Accepts light inputs (msisdn, size, reason) or uses sensible defaults.
   - Optionally resolves the correct subscription/billing account from an msisdn.
   - Builds and submits a productOrder to create the goodwill credit.
   - Can verify after the goodwill by re-fetching the DTO and snapshotting it.

3) Zendesk conversation by email
   - Finds a Zendesk user by email.
   - Picks the most recent ticket (or a requested ticket) and fetches comments.
   - Maps comments to a compact conversation model (public by default).
   - Applies date normalization and token/budget trimming just like DTO.

Performance & resilience highlights:
- Deadline-aware HTTP timeouts (adapts to Lambda's remaining time).
- Tiny in-memory DTO cache for faster GETs.
- Token-aware trimming to ensure responses fit LLM budgets.
- Global sanitization to drop heavy/binary-ish fields safely.
- Optional requests.Session pooling + retries (when available).
- Gzip decoding & compact JSON serialization (orjson if present).

Security & tenancy:
- Injects channel/brand headers.
- Can use a static JWT or the incoming one depending on env flags.
- Redacts sensitive response headers before returning.

This module includes both "Bedrock-style" event handling and a "console/direct"
mode so you can test it outside the agent via direct Lambda invokes.

================================================================================
""

import os
import re
import ssl
import json as _json
import math
import time
import datetime
import hashlib
import gzip
import io
import urllib.parse
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple, List, Optional

# ---------------- Fast JSON (optional) ----------------
# Uses orjson if available for speed & smaller payloads; falls back to std json.
try:
    import orjson as _fastjson
    def _dumps(obj) -> str:
        return _fastjson.dumps(obj, option=_fastjson.OPT_OMIT_MICROSECONDS).decode("utf-8")
    def _loads(s: str):
        return _fastjson.loads(s)
except Exception:
    _fastjson = None
    def _dumps(obj) -> str:
        return _json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    def _loads(s: str):
        return _json.loads(s)

# ---------------- Try to use requests (optional) ----------------
# If 'requests' exists, we benefit from connection pooling & small retries; else use urllib.
_USE_REQUESTS = True
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except Exception:
    _USE_REQUESTS = False
    requests = None

# ========= Config (triPica) =========
# BASE_URL + header presets; timeouts tuned for short Lambda budgets.
BASE_URL     = os.environ.get("BASE_URL", "https://api-demo.tparici.com")
AUTH_PREFIX  = os.environ.get("AUTH_PREFIX", "Bearer ")

# *** Default short timeouts to fit tight Lambda budgets; will be adjusted dynamically per invocation ***
TIMEOUT_CONN_DEFAULT = float(os.environ.get("TIMEOUT_CONNECT_SEC", "0.8"))
TIMEOUT_READ_DEFAULT = float(os.environ.get("TIMEOUT_READ_SEC",    "2.0"))

# Keep a minimal slack so we don't overrun Lambda timeout; abort gracefully if too late.
TIME_BUDGET_SLACK_MS = int(os.environ.get("TIME_BUDGET_SLACK_MS", "500"))
MIN_CALL_BUDGET_MS   = int(os.environ.get("MIN_CALL_BUDGET_MS",   "600"))

X_BRAND_DEFAULT   = os.environ.get("X_BRAND", "DEMO-DEMO")
X_CHANNEL_DEFAULT = os.environ.get("X_CHANNEL", "agent-tool")

# Auth — can always use env token (for demos) or prefer per-request token.
STATIC_JWT        = os.environ.get("STATIC_JWT", "")
ALWAYS_USE_ENV_TOKEN = os.environ.get("ALWAYS_USE_ENV_TOKEN", "true").lower() == "true"

# Default customer when none is provided; keeps demos frictionless.
DEFAULT_CUSTOMER_OUID = os.environ.get("DEFAULT_CUSTOMER_OUID", "").strip()

# Optional safety net: after goodwill, re-fetch DTO to confirm/attach snapshot.
VERIFY_AFTER_GOODWILL = os.environ.get("VERIFY_AFTER_GOODWILL", "false").lower() == "true"

# ========= Config (Zendesk) =========
# To read conversations, either OAuth token OR email+API token must be configured.
ZD_SUBDOMAIN_RAW = os.environ.get("ZD_SUBDOMAIN", "").strip()
ZENDESK_EMAIL     = os.environ.get("ZD_EMAIL", "").strip()
ZENDESK_API_TOKEN = os.environ.get("ZD_API_TOKEN", "").strip()
ZENDESK_OAUTH     = os.environ.get("ZD_OAUTH_TOKEN", "").strip()
ZD_DEFAULT_LIMIT  = int(os.environ.get("ZD_DEFAULT_LIMIT", "25"))

# ========= Goodwill defaults (demo) =========
# Sensible defaults let you trigger a goodwill with minimal inputs.
GOODWILL_CUSTOMER_OUID = os.environ.get("GOODWILL_CUSTOMER_OUID", "07D9EF423C516BFA96E89E654D197A6E")
GOODWILL_BA_OUID       = os.environ.get("GOODWILL_BA_OUID", "EF585B06263135C51CA1B2D991C09B99")
GOODWILL_PARENT_OUID   = os.environ.get("GOODWILL_PARENT_OUID", "81462040F9AE84EA538A6A8920D6A056")
GOODWILL_OFFERING_OUID = os.environ.get("GOODWILL_OFFERING_OUID", "518A3D53BE867A62C9F5BB07B4EC84C7")
GOODWILL_SPEC_OUID     = os.environ.get("GOODWILL_SPEC_OUID", "71FDF809A7495CACDDBC3F614331DD2F")

# ========= Token control / epoch replace =========
# "Token" here = LLM token estimate; we trim payloads to fit.
TOKEN_LIMIT = int(os.environ.get("TOKEN_LIMIT", "60000"))
TOKEN_BYTES_PER_TOKEN_DENOM = float(os.environ.get("TOKEN_BYTES_PER_TOKEN_DENOM", "2.0"))

# Convert epoch numbers into ISO timestamps so answers are readable.
EPOCH_REPLACE_ENABLED = os.environ.get("EPOCH_REPLACE_ENABLED", "true").lower() == "true"
EPOCH_KEYS_REGEX       = os.environ.get(
    "EPOCH_KEYS_REGEX",
    r"(date|time|timestamp|created|updated|renewal|expiry|expiration|due|validUntil|activated|deactivated|at)$"
)
EPOCH_REPLACE_MAX      = int(os.environ.get("EPOCH_REPLACE_MAX", "3000"))

# "Pinned" notification types are never dropped when trimming (e.g., TICKET_CREATED).
ALWAYS_KEEP_NOTIFICATION_TYPES = set(
    t.strip().upper()
    for t in os.environ.get("ALWAYS_KEEP_NOTIFICATION_TYPES", "TICKET_CREATED").split(",")
    if t.strip()
)
PINNED_STRIP_HEAVY_KEYS = os.environ.get("PINNED_STRIP_HEAVY_KEYS", "true").lower() == "true"

# ========= Average value flags =========
# Fine-grained control over how monthly averages are computed from DTO.
AVG_INCLUDE_PLAN = os.environ.get("AVG_INCLUDE_PLAN", "false").lower() == "true"
AVG_INCLUDE_TERMINATED_OPTIONS = os.environ.get("AVG_INCLUDE_TERMINATED_OPTIONS", "false").lower() == "true"
AVG_USE_TAX_INCLUDED = os.environ.get("AVG_USE_TAX_INCLUDED", "true").lower() == "true"
CURRENCY_SYMBOL = os.environ.get("CURRENCY_SYMBOL", "$")

# ========= Snapshot / Global budget flags (new) =========
# Snapshot keeps the "essentials" from a DTO for quick summaries; global trim enforces size.
SNAPSHOT_ENABLED = os.environ.get("SNAPSHOT_ENABLED", "true").lower() == "true"
SNAPSHOT_MAX_TICKETS = int(os.environ.get("SNAPSHOT_MAX_TICKETS", "5"))
SNAPSHOT_MAX_PAYMENTS = int(os.environ.get("SNAPSHOT_MAX_PAYMENTS", "10"))

GLOBAL_TRIM_ENABLED = os.environ.get("GLOBAL_TRIM_ENABLED", "true").lower() == "true"
GLOBAL_MAX_ARRAY_LEN = int(os.environ.get("GLOBAL_MAX_ARRAY_LEN", "20"))
GLOBAL_MAX_FIELD_LEN = int(os.environ.get("GLOBAL_MAX_FIELD_LEN", "160"))

# Keys that are never useful for the assistant and can be dropped safely.
DROP_KEYS = set((
    "TrpcCtx","TrpcCtxOuid","authorities","sandBoxUuid","jwt","bpmInstanceOuid","idTrx","loginOuid",
    "paymentMeanOuid","orderOuid","networkEntityOuid","paymentRequestOuid",
    "createOrder_IdResult","cleanSandBox_IdResult","updateProductsStatus_PendingActiveResult",
    "updateProductsStatus_Active_1Result","getDueAmount_IdResult","createNotification_successResult",
    "computeNextRenewalDate_1Result","renewProducts_1Result","updateNetworkEntityStatus_toActiveResult",
))

# ========= Deadline-aware timeout handling =========
# These helpers adjust HTTP timeouts based on remaining Lambda time.
_REQUEST_TIMEOUT = (TIMEOUT_CONN_DEFAULT, TIMEOUT_READ_DEFAULT)
_REMAINING_MS = None

def _set_request_timeout(connect_read_tuple: Tuple[float, float]):
    global _REQUEST_TIMEOUT
    _REQUEST_TIMEOUT = connect_read_tuple

def _get_request_timeout() -> Tuple[float, float]:
    return _REQUEST_TIMEOUT

def _update_deadline_from_context(context):
    """Recompute per-request timeouts from Lambda's remaining time.
    - Prevents slow backends from causing function timeouts.
    - Uses only a fraction of remaining time for HTTP calls (60% read / half for connect)."""
    global _REMAINING_MS
    try:
        rem = int(context.get_remaining_time_in_millis())
    except Exception:
        rem = 900000  # default huge for local tests
    _REMAINING_MS = rem
    budget = max(100, rem - TIME_BUDGET_SLACK_MS)
    read = min(TIMEOUT_READ_DEFAULT, max(0.4, (budget / 1000.0) * 0.6))
    conn = min(TIMEOUT_CONN_DEFAULT, max(0.2, read / 2.0))
    _set_request_timeout((conn, read))

def _ensure_budget_or_fail():
    """If there isn't enough time left for a safe network call, bail out early."""
    rem = _REMAINING_MS if _REMAINING_MS is not None else 0
    return (rem >= MIN_CALL_BUDGET_MS), rem

# ========= Optional pooled HTTP client (requests) =========
# If available, we set up a shared session with small, dynamic retries.
if _USE_REQUESTS:
    _session = requests.Session()
    _session.headers.update({
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "triPica-AgentLambda/0.4"
    })
    def _dynamic_retries():
        # Disable retries when time is almost up; otherwise use small backoff.
        try:
            rem = _REMAINING_MS or 0
        except Exception:
            rem = 0
        total = 0 if rem and rem < 2500 else int(os.environ.get("HTTP_RETRIES", "1"))
        return Retry(
            total=total,
            backoff_factor=float(os.environ.get("HTTP_BACKOFF", "0.05")),
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
        )
    _adapter = HTTPAdapter(
        pool_connections=int(os.environ.get("HTTP_POOL_CONN", "32")),
        pool_maxsize=int(os.environ.get("HTTP_POOL_MAX", "32")),
        max_retries=_dynamic_retries(),
    )
    _session.mount("https://", _adapter)
    _session.mount("http://", _adapter)

# ========= Tiny in-memory cache for DTOs =========
# Cache successful DTO GETs briefly to lower latency & backend load during a chat.
_DTO_CACHE_TTL = int(os.environ.get("DTO_CACHE_TTL_SEC", "600"))   # 10 min
_DTO_CACHE_MAX = int(os.environ.get("DTO_CACHE_MAX_ENTRIES", "64"))
_DTO_CACHE: Dict[str, Tuple[float, Any]] = {}  # key -> (expires_at, payload)

def _cache_get(key: str):
    ent = _DTO_CACHE.get(key)
    if not ent:
        return None
    expires, val = ent
    if expires < time.time():
        _DTO_CACHE.pop(key, None)
        return None
    return val

def _cache_set(key: str, val: Any):
    if len(_DTO_CACHE) >= _DTO_CACHE_MAX:
        try:
            _DTO_CACHE.pop(next(iter(_DTO_CACHE)))
        except Exception:
            _DTO_CACHE.clear()
    _DTO_CACHE[key] = (time.time() + _DTO_CACHE_TTL, val)

# ========= HTTP helpers (triPica & Zendesk) =========
def _mk_url(path: str, query: dict | None) -> str:
    url = BASE_URL.rstrip("/") + path
    if query:
        url += "?" + urllib.parse.urlencode(query)
    return url

def _decode_response_bytes(raw: bytes, headers: Dict[str, str]) -> str:
    # Transparently handle gzip; always decode as UTF-8 for safety.
    enc = (headers.get("Content-Encoding") or headers.get("content-encoding") or "").lower()
    if enc == "gzip":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                return gz.read().decode("utf-8", errors="replace")
        except Exception:
            return raw.decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")

def _do_request_requests(method: str, url: str, headers: dict | None, body: dict | str | None):
    # Pooled client path.
    try:
        _adapter.max_retries = _dynamic_retries()  # type: ignore[attr-defined]
    except Exception:
        pass
    data = None
    json_body = None
    if isinstance(body, (dict, list)):
        json_body = body
    elif isinstance(body, str):
        data = body.encode("utf-8")

    timeout = _get_request_timeout()
    resp = _session.request(
        method=method.upper(),
        url=url,
        headers=headers or {},
        json=json_body,
        data=data,
        timeout=timeout,
    )
    text = resp.text or ""
    try:
        parsed = _loads(text) if text and text.strip().startswith(("{", "[")) else text
    except Exception:
        parsed = text
    resp_headers = {k: v for k, v in resp.headers.items()}
    return int(resp.status_code), parsed, resp_headers

def _do_request_urllib(method: str, url: str, headers: dict | None, body: dict | str | None):
    # Lightweight fallback when 'requests' is unavailable.
    data = None
    if isinstance(body, (dict, list)):
        data = _dumps(body).encode("utf-8")
    elif isinstance(body, str):
        data = body.encode("utf-8")
    req = urllib.request.Request(url=url, method=method.upper())
    base_headers = {
        "Accept": "application/json",
        "Connection": "keep-alive",
        "Accept-Encoding": "gzip",
        "User-Agent": "triPica-AgentLambda/0.4"
    }
    for k, v in {**base_headers, **(headers or {})}.items():
        req.add_header(k, v)
    ctx = ssl.create_default_context()
    timeout = max(_get_request_timeout())
    with urllib.request.urlopen(req, data=data, timeout=timeout, context=ctx) as resp:
        raw = resp.read() or b""
        resp_headers = {k: v for k, v in resp.getheaders()}
        text = _decode_response_bytes(raw, resp_headers)
        try:
            parsed = _loads(text) if text and text.strip().startswith(("{", "[")) else text
        except Exception:
            parsed = text
        return int(resp.getcode()), parsed, resp_headers

def _do_request(method: str, url: str, headers: dict | None, body: dict | str | None):
    # Respect the time budget before making any outbound call.
    ok, rem = _ensure_budget_or_fail()
    if not ok:
        return 504, {"error": "GatewayTimeout", "message": f"Not enough time budget ({rem}ms) to call {url}"}, {"X-FastFail": "true"}
    if _USE_REQUESTS:
        return _do_request_requests(method, url, headers, body)
    return _do_request_urllib(method, url, headers, body)

# ---------------- Zendesk plumbing ----------------
def _zd_base_url() -> str:
    if not ZD_SUBDOMAIN_RAW:
        raise RuntimeError("ZD_SUBDOMAIN missing")
    if "://" in ZD_SUBDOMAIN_RAW:
        u = ZD_SUBDOMAIN_RAW.rstrip("/")
        if u.endswith(".json"):
            u = u.split("/api/")[0]
        return u
    return f"https://{ZD_SUBDOMAIN_RAW}.zendesk.com"

def _zd_auth_headers() -> Dict[str, str]:
    # Supports either OAuth bearer or Basic (email+API token).
    h = {"Accept": "application/json", "Accept-Encoding": "gzip"}
    if ZENDESK_OAUTH:
        h["Authorization"] = f"Bearer {ZENDESK_OAUTH}"
        return h
    if not (ZENDESK_EMAIL and ZENDESK_API_TOKEN):
        raise RuntimeError("Zendesk API auth missing: set ZD_OAUTH_TOKEN or ZD_EMAIL + ZD_API_TOKEN")
    import base64 as _b64
    raw = f"{ZENDESK_EMAIL}/token:{ZENDESK_API_TOKEN}".encode("utf-8")
    h["Authorization"] = "Basic " + _b64.b64encode(raw).decode("ascii")
    return h

def _zd_request(method: str, path: str, query: Dict[str, Any] | None = None) -> tuple[int, Any, Dict[str,str]]:
    url = _zd_base_url() + path
    if query:
        qs = urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        url += ("?" + qs)
    return _do_request(method, url, _zd_auth_headers(), None)

# ========= Token & epoch helpers =========
def _estimate_tokens_from_str(s: str) -> int:
    # Coarse estimate of LLM tokens from bytes; good enough for trimming decisions.
    b = len(s.encode("utf-8", errors="ignore"))
    if b <= 0:
        return 0
    return int(math.ceil(b / max(1e-6, TOKEN_BYTES_PER_TOKEN_DENOM)))

def _to_body_str(payload: Any) -> str:
    if isinstance(payload, (dict, list)):
        return _dumps(payload)
    return "" if payload is None else str(payload)

_EPOCH_KEYS_RE  = re.compile(EPOCH_KEYS_REGEX, re.IGNORECASE)
_SKIP_KEYS_RE   = re.compile(r"(?:^|_)(?:pdf|image|content|html|body|blob|binary|base64)$", re.IGNORECASE)

def _epoch_unit(num: float | int | None) -> Optional[Tuple[str,int]]:
    # Detect seconds vs milliseconds for epoch timestamps.
    if num is None:
        return None
    try:
        n = float(num)
    except Exception:
        return None
    if 1_000_000_000 <= n < 100_000_000_000:
        return ("s", int(n))
    if 1_000_000_000_000 <= n < 100_000_000_000_000:
        return ("ms", int(n))
    return None

def _to_iso(epoch: int, unit: str) -> Optional[str]:
    # Turn raw epoch numbers into ISO dates so they read well in answers.
    try:
        ts = epoch / 1000.0 if unit == "ms" else epoch
        return datetime.datetime.utcfromtimestamp(ts).replace(microsecond=0).isoformat() + "Z"
    except Exception:
        return None

def _replace_epoch_dates(obj, *, max_replacements=EPOCH_REPLACE_MAX):
    # Walk the payload and convert any field that *looks* like a date key carrying an epoch value.
    replaced = 0
    def walk(x, parent_key=None):
        nonlocal replaced
        if replaced >= max_replacements:
            return x
        if isinstance(x, dict):
            if parent_key and _SKIP_KEYS_RE.search(parent_key):
                return x
            for k, v in list(x.items()):
                if replaced >= max_replacements:
                    break
                nv = walk(v, k)
                if nv is not v:
                    x[k] = nv
                if replaced >= max_replacements:
                    break
                if k and _EPOCH_KEYS_RE.search(k):
                    candidate = None
                    if isinstance(x[k], (int, float)):
                        candidate = x[k]
                    elif isinstance(x[k], str) and x[k].isdigit():
                        candidate = int(x[k])
                    unit_epoch = _epoch_unit(candidate) if candidate is not None else None
                    if unit_epoch:
                        unit, ep = unit_epoch
                        iso = _to_iso(ep, unit)
                        if iso:
                            x[k] = iso
                            replaced += 1
            return x
        if isinstance(x, list):
            for i in range(len(x)):
                if replaced >= max_replacements:
                    break
                x[i] = walk(x[i], parent_key)
            return x
        return x
    return walk(obj), replaced

def _locate_notifications_array(obj: Any):
    # Identify the array (notifications/tickets/comments) we can trim first if needed.
    if not isinstance(obj, dict):
        return None, None, None, None
    for k in ("notifications", "notification", "tickets", "ticketEvents", "comments", "conversation"):
        v = obj.get(k)
        if isinstance(v, list) and len(v) > 0:
            return False, k, v, k
    data = obj.get("data")
    if isinstance(data, dict):
        for k in ("notifications", "notification", "tickets", "ticketEvents", "comments", "conversation"):
            v = data.get(k)
            if isinstance(v, list) and len(v) > 0:
                return True, k, v, f"data.{k}"
    return None, None, None, None

def _enforce_token_limit(payload: Any, token_limit: int = TOKEN_LIMIT):
    # Trim from the tail, but always keep "pinned" types (e.g. TICKET_CREATED).
    def _is_pinned(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        t = str(item.get("type", "")).upper()
        return t in ALWAYS_KEEP_NOTIFICATION_TYPES

    def _strip_heavy_keys_inplace(item: dict):
        # If pinned items alone exceed the limit, remove heavy/binary-like fields.
        if not isinstance(item, dict):
            return
        for k in list(item.keys()):
            if _SKIP_KEYS_RE.search(k or ""):
                item.pop(k, None)

    body_str = _to_body_str(payload)
    t_before = _estimate_tokens_from_str(body_str)
    if t_before <= token_limit:
        return payload, False, t_before, t_before, None, 0

    obj = None
    if isinstance(payload, (dict, list)):
        obj = payload
    else:
        try:
            obj = _loads(body_str)
        except Exception:
            obj = None
    if not isinstance(obj, dict):
        return payload, False, t_before, t_before, None, 0

    in_data, key, arr, path = _locate_notifications_array(obj)
    if key is None or not isinstance(arr, list) or not arr:
        return payload, False, t_before, t_before, None, 0

    pinned_flags = [_is_pinned(it) for it in arr]
    non_pinned_total = sum(1 for f in pinned_flags if not f)

    def build_keep_k_non_pinned(k: int):
        kept = []
        keep_np = k
        for it, is_p in zip(arr, pinned_flags):
            if is_p:
                kept.append(it)
            else:
                if keep_np > 0:
                    kept.append(it)
                    keep_np -= 1
        if in_data:
            new_data = dict(obj["data"])
            new_data[key] = kept
            new_obj = dict(obj)
            new_obj["data"] = new_data
        else:
            new_obj = dict(obj)
            new_obj[key] = kept
        return new_obj, kept

    # Binary search the largest "k" of non-pinned we can keep under the token limit.
    lo, hi, best_k = 0, non_pinned_total, -1
    best_obj = obj
    while lo <= hi:
        mid = (lo + hi) // 2
        cand_obj, _ = build_keep_k_non_pinned(mid)
        t_mid = _estimate_tokens_from_str(_to_body_str(cand_obj))
        if t_mid <= token_limit:
            best_k = mid
            best_obj = cand_obj
            lo = mid + 1
        else:
            hi = mid - 1

    if best_k < 0:
        # Even pinned alone exceed limit → keep only pinned and strip heavy keys.
        best_k = 0
        best_obj, kept = build_keep_k_non_pinned(0)
        if PINNED_STRIP_HEAVY_KEYS:
            kept = (best_obj["data"][key] if in_data else best_obj[key])
            for it in kept:
                if isinstance(it, dict) and _is_pinned(it):
                    _strip_heavy_keys_inplace(it)

    t_after = _estimate_tokens_from_str(_to_body_str(best_obj))

    kept_list = (best_obj["data"][key] if in_data else best_obj[key])
    trimmed_non_pinned = non_pinned_total - best_k
    pinned_kept_count = sum(1 for it in kept_list if isinstance(it, dict) and _is_pinned(it))

    meta_note = f"Kept all pinned notifications ({pinned_kept_count}); removed {trimmed_non_pinned} non-pinned from the tail to fit token limit."
    new_obj = dict(best_obj)
    meta = dict(new_obj.get("_meta", {}))
    meta.update({
        "trimmedNotifications": int(trimmed_non_pinned),
        "trimmedFrom": "end",
        "trimmedPath": path,
        "pinnedTypes": sorted(list(ALWAYS_KEEP_NOTIFICATION_TYPES)),
        "pinnedKeptCount": int(pinned_kept_count),
        "nonPinnedKeptCount": int(best_k),
        "pinnedExceedsLimit": _estimate_tokens_from_str(_to_body_str(build_keep_k_non_pinned(0)[0])) > token_limit
    })
    new_obj["_meta"] = meta
    if "_note" not in new_obj:
        new_obj["_note"] = meta_note

    t_final = _estimate_tokens_from_str(_to_body_str(new_obj))
    return new_obj, (trimmed_non_pinned > 0), t_before, t_final, f"{path}: removed {trimmed_non_pinned} non-pinned item(s) from tail; kept all pinned.", trimmed_non_pinned

def _append_token_counters(payload: Any, t_before: int, t_after: int, token_limit: int) -> Any:
    # Attach simple diagnostics so you can see trimming impact in responses.
    if isinstance(payload, dict):
        out = dict(payload)
        out["_tokenEstimateBefore"] = int(t_before)
        out["_tokenEstimateAfter"]  = int(t_after)
        out["_tokenLimit"]          = int(token_limit)
        return out
    return payload

# ========= Average value computation (robust over DTO) =========
# These helpers read product prices from subscriptions to compute average monthly line value.
def _money_from_product(p: dict) -> int:
    if not isinstance(p, dict):
        return 0
    key = "taxIncludedAmount" if AVG_USE_TAX_INCLUDED else "dutyFreeAmount"
    v = p.get(key, 0)
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return 0

def _looks_like_subscription(node: dict) -> bool:
    if not isinstance(node, dict):
        return False
    prods = node.get("products")
    if not isinstance(prods, dict):
        return False
    has_option = isinstance(prods.get("OPTION"), list)
    has_plan = isinstance(prods.get("PLAN"), list)
    return ("status" in node) and (has_option or has_plan)

def _collect_subscriptions(root: Any) -> list[dict]:
    out: list[dict] = []
    def walk(x):
        if isinstance(x, dict):
            if _looks_like_subscription(x):
                out.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)
    walk(root)
    return out

def _extract_msisdn_from_sub(sub: dict) -> str | None:
    # Find msisdn from either products.RESOURCE or legacy resources[].
    prods = (sub.get("products") or {})
    res = prods.get("RESOURCE") or []
    for r in res:
        s = (r or {}).get("productSerialNumber") or (r or {}).get("msisdn")
        if isinstance(s, str) and any(c.isdigit() for c in s):
            return s
    for r in (sub.get("resources") or []):
        s = (r or {}).get("productSerialNumber") or (r or {}).get("msisdn")
        if isinstance(s, str) and any(c.isdigit() for c in s):
            return s
    return None

def _sum_monthly_cents_for_sub(sub: dict) -> int:
    # Sum OPTION prices (and PLAN if enabled). Include terminated options if configured.
    if not isinstance(sub, dict):
        return 0
    prods = (sub.get("products") or {})
    total = 0
    for opt in (prods.get("OPTION") or []):
        status = str((opt or {}).get("status", "")).upper()
        if status == "ACTIVE" or (AVG_INCLUDE_TERMINATED_OPTIONS and status == "TERMINATED"):
            total += _money_from_product(opt)
    if AVG_INCLUDE_PLAN:
        for pl in (prods.get("PLAN") or []):
            status = str((pl or {}).get("status", "")).upper()
            if status == "ACTIVE":
                total += _money_from_product(pl)
    return total

def _compute_avg_line_value(payload: Any) -> dict:
    # Returns active line count and average monthly spend per active line.
    subs = _collect_subscriptions(payload)
    active_subs = [s for s in subs if str(s.get("status", "")).upper() == "ACTIVE"]
    per_line = []
    total_cents = 0
    for s in active_subs:
        cents = _sum_monthly_cents_for_sub(s)
        total_cents += cents
        s["_monthlyTotalCents"] = int(cents)
        s["_monthlyTotal"] = round(cents / 100.0, 2)
        msisdn = _extract_msisdn_from_sub(s)
        if msisdn:
            s["_msisdn"] = msisdn
        per_line.append({
            "subscriptionOuid": s.get("ouid"),
            "msisdn": msisdn,
            "monthlyCents": int(cents),
            "monthly": round(cents / 100.0, 2),
        })
    count = len(active_subs)
    avg_cents = int(round(total_cents / count)) if count else 0
    avg = round(avg_cents / 100.0, 2)
    return {
        "activeLineCount": count,
        "perLineMonthly": per_line,
        "avgMonthlyCents": avg_cents,
        "avgMonthly": avg,
        "avgMonthlyFormatted": f"{CURRENCY_SYMBOL}{avg:.2f}"
    }

def _annotate_dto_avg_inplace(dto: dict) -> dict:
    # Adds a compact "_computed" section in the DTO with summary metrics.
    summary = _compute_avg_line_value(dto)
    comp = dict(dto.get("_computed", {}))
    comp.update({
        "activeLineCount": summary["activeLineCount"],
        "avgMonthlyCents": summary["avgMonthlyCents"],
        "avgMonthly": summary["avgMonthly"],
        "currency": CURRENCY_SYMBOL,
    })
    dto["_computed"] = comp
    return summary

# ========= NEW: Snapshot & Global budget helpers =========
def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _count_goodwills(notif: Any) -> int:
    # Quick heuristic flag if many goodwill credits exist.
    if not isinstance(notif, list):
        return 0
    cnt = 0
    for n in notif:
        p = (n or {}).get("parameters") or {}
        if str(p.get("typeNotification","")).upper() == "GOODWILL_CREDIT_OK":
            cnt += 1
    return cnt

def _build_state_snapshot(dto: dict) -> dict:
    # Small, stable "snapshot" the assistant can reason about quickly.
    if not isinstance(dto, dict):
        return {}
    subs = _collect_subscriptions(dto)
    active = [s for s in subs if str(s.get("status","")).upper() == "ACTIVE"]
    next_payments = []
    for s in active:
        msisdn = _extract_msisdn_from_sub(s) or ""
        amount_cents = int(s.get("_monthlyTotalCents") or 0)
        due = s.get("nextRenewalDate") or s.get("renewalDate") or ""
        next_payments.append({
            "msisdn": msisdn,
            "amountCents": amount_cents,
            "dueDate": due
        })
    tickets = []
    for n in (dto.get("notification") or dto.get("notifications") or []):
        if str(n.get("type","")).upper() == "TICKET_CREATED":
            p = n.get("parameters") or {}
            tickets.append({
                "id": n.get("ouid"),
                "title": p.get("ticketTitle") or "ticket",
                "createdAt": n.get("notificationDate"),
                "status": p.get("status")
            })
            if len(tickets) >= SNAPSHOT_MAX_TICKETS:
                break
    party = dto.get("party") or dto.get("customer") or {}
    id_invalid = str(((party.get("attributes") or {}).get("identificationStatus") or "")).upper() == "INVALID"
    task = dto.get("taskInfo") or {}
    flags = {
        "idInvalid": id_invalid,
        "hasTaskErrors": int(task.get("errors") or 0) > 0 or int(task.get("blockedProcesses") or 0) > 0,
        "manyGoodwills": _count_goodwills(dto.get("notification") or dto.get("notifications")) >= 3
    }
    comp = dto.get("_computed") or {}
    snap = {
        "_computed": {k: comp.get(k) for k in ("activeLineCount","avgMonthlyCents","avgMonthly","currency") if k in comp},
        "nbActiveSubscription": dto.get("nbActiveSubscription") or comp.get("activeLineCount"),
        "tickets": tickets[:SNAPSHOT_MAX_TICKETS],
        "nextPayments": next_payments[:SNAPSHOT_MAX_PAYMENTS],
        "flags": flags
    }
    return snap

def _sanitize_for_budget(x: Any, *, max_array_len=GLOBAL_MAX_ARRAY_LEN, max_field_len=GLOBAL_MAX_FIELD_LEN) -> Any:
    # Second-layer size control — trims arrays, long strings, and binary-ish fields globally.
    def looks_soup(s: str) -> bool:
        return bool(re.search(r"[A-F0-9]{80,}|[A-Za-z0-9+/=]{120,}", s))
    def walk(v):
        if isinstance(v, dict):
            out = {}
            for k, val in v.items():
                if k in DROP_KEYS:
                    continue
                if _SKIP_KEYS_RE.search(k or ""):
                    continue
                out[k] = walk(val)
            return out
        if isinstance(v, list):
            return [walk(it) for it in v[:max_array_len]]
        if isinstance(v, str):
            if looks_soup(v):
                return v[:40] + "…(redacted)…"
            if len(v) > max_field_len:
                return v[:max_field_len] + "…"
            return v
        return v
    try:
        return walk(x)
    except Exception:
        return x

def _enforce_global_budget(payload: Any, token_limit: int = TOKEN_LIMIT):
    # First try targeted trim (notifications). If still too big, apply global sanitization.
    p1, trimmed, t_before, t_after, info, trimmed_count = _enforce_token_limit(payload, token_limit)
    if t_after <= token_limit or not GLOBAL_TRIM_ENABLED:
        return p1, trimmed, t_before, t_after, info, trimmed_count
    sanitized = _sanitize_for_budget(p1)
    t_after2 = _estimate_tokens_from_str(_to_body_str(sanitized))
    info2 = (info or "trim") + "; global-sanitize applied"
    return sanitized, True, t_before, t_after2, info2, int(trimmed_count)

# ========= DTO utilities =========
def _apply_headers(h: Dict[str, str], token: Optional[str]) -> Dict[str, str]:
    # Inject tenant/channel headers and authorization strategy.
    out = {**(h or {})}
    out["X-Brand"]   = X_BRAND_DEFAULT
    out["X-Channel"] = X_CHANNEL_DEFAULT
    out["Accept"]    = "application/json"
    use_token = STATIC_JWT if ALWAYS_USE_ENV_TOKEN else (token or STATIC_JWT)
    if use_token:
        out["Authorization"] = use_token if use_token.startswith("Bearer ") else f"{AUTH_PREFIX}{use_token}"
    return out

def _resolve_default_customer_ouid(session_attrs: Dict[str, Any] | None) -> str:
    # Choose customer OUID (session > env default > demo goodwill default).
    sa = session_attrs or {}
    return (str(sa.get("customerOuid") or "").strip()
            or DEFAULT_CUSTOMER_OUID
            or GOODWILL_CUSTOMER_OUID)

def _normalize_customer_dto_path(path: str, session_attrs: Dict[str, Any] | None) -> tuple[str, Optional[str]]:
    # If the base DTO path is called without an id, automatically plug a default OUID.
    base = "/api/private/v1/agent/customer/customerDto"
    if path.rstrip("/") == base:
        ouid = _resolve_default_customer_ouid(session_attrs)
        return f"{base}/{ouid}", ouid
    return path, None

# ========= Goodwill builders =========
def _merge_goodwill_overrides(body: Any, session_attrs: Dict[str, Any] | None) -> dict:
    # Compose the goodwill inputs from body or session, with safe defaults.
    sa = session_attrs or {}
    b = body if isinstance(body, dict) else {}
    def pick_str(*keys, default=""):
        for k in keys:
            v = (b.get(k) if isinstance(b, dict) else None) if k in b else sa.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
        return default
    def pick_int(*keys, default=2):
        for k in keys:
            v = (b.get(k) if isinstance(b, dict) else None) if k in b else sa.get(k)
            try:
                if v is None:
                    continue
                return int(float(v))
            except Exception:
                continue
        return int(default)
    gw = {
        "customerOuid":       pick_str("customerOuid", default=GOODWILL_CUSTOMER_OUID),
        "billingAccountOuid": pick_str("billingAccountOuid", default=GOODWILL_BA_OUID),
        "parentOuid":         pick_str("parentOuid", default=GOODWILL_PARENT_OUID),
        "offeringOuid":       pick_str("offeringOuid", default=GOODWILL_OFFERING_OUID),
        "specOuid":           pick_str("specOuid", default=GOODWILL_SPEC_OUID),
        "sizeGb":             pick_int("sizeGb", "goodwillSizeGb", default=2),
        "reason":             pick_str("reason", "goodwillReason", default="boosterOrPassRefund"),
        "msisdn":             pick_str("msisdn", "lineMsisdn", default="")
    }
    if gw["sizeGb"] < 1:
        gw["sizeGb"] = 1
    return gw

def _build_goodwill_payload(now_ms: int, gw: dict) -> dict:
    # Translate inputs into the productOrder payload required by the billing/catalog layer.
    size_units = int(gw.get("sizeGb", 2)) * 1024 * 1024
    return {
        "category": "AGENT",
        "customerOuid": gw.get("customerOuid", GOODWILL_CUSTOMER_OUID),
        "characteristics": {},
        "description": "AGENT",
        "externalId": "TO_GENERATE",
        "notificationContact": "unused",
        "orderDate": now_ms,
        "orderItems": [
            {
                "action": "ADD",
                "billingAccountOuid": gw.get("billingAccountOuid", GOODWILL_BA_OUID),
                "product": {
                    "agreements": [],
                    "billingAccountOuid": gw.get("billingAccountOuid", GOODWILL_BA_OUID),
                    "characteristics": {
                        "DATA_CREDIT": str(size_units),
                        "REASON": gw.get("reason", "boosterOrPassRefund")
                    },
                    "customerOuid": gw.get("customerOuid", GOODWILL_CUSTOMER_OUID),
                    "name": "AGENT_GOODWILL_CREDIT",
                    "orderDate": now_ms,
                    "productOfferingOuid": gw.get("offeringOuid", GOODWILL_OFFERING_OUID),
                    "productPrices": [
                        {
                            "name": "AGENT_GOODWILL_CREDIT",
                            "price": {
                                "currencyCode": "USD",
                                "dutyFreeAmount": 0,
                                "percentage": 0,
                                "priceType": "RECURRING",
                                "taxIncludedAmount": 0,
                                "taxRate": 0
                            },
                            "priceType": "RECURRING",
                            "recurringChargePeriod": "MONTH",
                            "recurringChargePeriodNumber": 1,
                            "startDateTime": 1606089600000
                        }
                    ],
                    "productRelationships": [
                        { "targetProductOuid": gw.get("parentOuid", GOODWILL_PARENT_OUID), "type": "PARENT" }
                    ],
                    "productSpecificationOuid": gw.get("specOuid", GOODWILL_SPEC_OUID),
                    "startDateTime": now_ms,
                    "status": "CREATED"
                },
                "productOfferingOuid": gw.get("offeringOuid", GOODWILL_OFFERING_OUID)
            }
        ],
        "state": "ACKNOWLEDGED"
    }

def _call_product_order_for_goodwill(headers: Dict[str, str], gw: dict) -> tuple[int, Any, Dict[str, str]]:
    # Submit the product order and bubble up status + correlation id when available.
    h = dict(headers or {})
    h["Content-Type"] = "application/json"
    h["X-Channel"] = "mobile-app"
    url = _mk_url("/api/private/v1/agent/productOrder", None)
    now_ms = int(datetime.datetime.utcnow().timestamp() * 1000)
    payload = _build_goodwill_payload(now_ms, gw)
    status, _payload, resp_headers = _do_request("POST", url, h, payload)
    rh = dict(resp_headers or {})
    for cid in ("X-Correlation-Id", "X-Request-Id", "X-Request-ID"):
        if cid in rh:
            rh["X-Backend-Correlation"] = rh[cid]
            break
    if status == 204:
        return 204, {"message": "No Content from backend", "composedSizeGb": int(gw.get("sizeGb", 2))}, rh
    return int(status), _payload, rh

# ========= Event normalization =========
# We support two calling modes:
#     - "Bedrock-style" events (tool/action invocation)
#     - Direct console/testing invoke (pass method/path/body yourself)
def _extract_call_console(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mode": "console",
        "method": event.get("method", "GET"),
        "path": event.get("path", "/api/private/v1/agent/customer/customerDto"),
        "query": event.get("query") or {},
        "headers": event.get("headers") or {},
        "body": event.get("body"),
        "token": event.get("token"),
        "bedrock_meta": None,
        "raw_api_path": event.get("path"),
        "raw_action_group": None,
        "tool_use_id": None,
    }

def _extract_call_bedrock_v1(event: Dict[str, Any]) -> Dict[str, Any]:
    # Legacy v1 shape — extract method/path/query/headers/body consistently.
    tool = event.get("tool", {}) or {}
    hdrs = {}
    for h in tool.get("headers") or []:
        if isinstance(h, dict) and "name" in h and "value" in h:
            hdrs[h["name"]] = h["value"]
    query: Dict[str, Any] = {}
    for p in tool.get("parameters") or []:
        if isinstance(p, dict) and p.get("name") is not None:
            query[p["name"]] = p.get("value")
    body = None
    rb = tool.get("requestBody") or {}
    content = rb.get("content")
    if isinstance(content, (dict, list)):
        body = content
    elif isinstance(content, str) and content.strip():
        try:
            body = _loads(content)
        except Exception:
            body = content
    token = (event.get("sessionState", {}) or {}).get("sessionAttributes", {}).get("jwt")
    return {
        "mode": "bedrock_v1",
        "method": (tool.get("httpMethod") or "GET").upper(),
        "path": tool.get("apiPath") or "/api/private/v1/agent/customer/customerDto",
        "query": query,
        "headers": hdrs,
        "body": body,
        "token": token,
        "bedrock_meta": {"actionGroup": tool.get("actionGroup")},
        "raw_api_path": tool.get("apiPath"),
        "raw_action_group": tool.get("actionGroup"),
        "tool_use_id": tool.get("toolUseId"),
    }

def _extract_call_bedrock_v2(event: Dict[str, Any]) -> Dict[str, Any]:
    # Newer v2 shape — similar extraction logic.
    hdrs = {}
    for h in event.get("headers") or []:
        if isinstance(h, dict) and "name" in h and "value" in h:
            hdrs[h["name"]] = h["value"]
    query: Dict[str, Any] = {}
    for p in event.get("parameters") or []:
        if isinstance(p, dict) and p.get("name") is not None:
            query[p["name"]] = p.get("value")
    body = None
    rb = event.get("requestBody") or {}
    content = rb.get("content")
    if isinstance(content, (dict, list)):
        body = content
    elif isinstance(content, str) and content.strip():
        try:
            body = _loads(content)
        except Exception:
            body = content
    token = (event.get("sessionAttributes") or {}).get("jwt")
    return {
        "mode": "bedrock_v2",
        "method": (event.get("httpMethod") or "GET").upper(),
        "path": event.get("apiPath") or "/api/private/v1/agent/customer/customerDto",
        "query": query,
        "headers": hdrs,
        "body": body,
        "token": token,
        "bedrock_meta": {"actionGroup": event.get("actionGroup")},
        "raw_api_path": event.get("apiPath"),
        "raw_action_group": event.get("actionGroup"),
        "tool_use_id": event.get("toolUseId"),
    }

def _is_bedrock_event(event: Dict[str, Any]) -> bool:
    return isinstance(event, dict) and ("messageVersion" in event)

# Replace /path/{id} placeholders using provided parameters.
_PATH_PARAM_RE = re.compile(r"\{([^{}]+)\}")
def _substitute_path_params(path: str, params: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    if "{" not in path:
        return path, params
    remaining = dict(params or {})
    def repl(match):
        key = match.group(1)
        if key in remaining:
            val = "" if remaining[key] is None else str(remaining[key])
            remaining.pop(key, None)
            return urllib.parse.quote(val, safe="")
        return match.group(0)
    new_path = _PATH_PARAM_RE.sub(repl, path)
    return new_path, remaining

# ========= DTO inline fetch (with cache) =========
def _fetch_dto_for_customer(customer_ouid: str, headers: Dict[str, str]) -> tuple[int, Any]:
    # Cached DTO GET for snappy experiences during chat sessions.
    hdrs = dict(headers or {})
    hdrs["X-Channel"] = "mobile-app"
    path = f"/api/private/v1/agent/customer/customerDto/{customer_ouid}"
    url = _mk_url(path, None)
    auth = hdrs.get("Authorization", "")
    key = f"DTO::{customer_ouid}::{hashlib.sha1(auth.encode()).hexdigest()}"
    cached = _cache_get(key)
    if cached is not None:
        return 200, cached
    status, payload, _ = _do_request("GET", url, hdrs, None)
    if 200 <= status < 300 and isinstance(payload, (dict, list)):
        _cache_set(key, payload)
    return status, payload

def _try_resolve_subscription_from_msisdn(msisdn: str, customer_ouid: str, headers: Dict[str, str]) -> tuple[Optional[str], Optional[str]]:
    # Convenience — from a phone number, guess the correct subscription and BA OUID.
    try:
        st, dto = _fetch_dto_for_customer(customer_ouid, headers)
        if not (200 <= st < 300) or not isinstance(dto, (dict, list, str)):
            return None, None
        if isinstance(dto, str):
            try: dto = _loads(dto)
            except Exception: return None, None
        subs = _collect_subscriptions(dto)
        ms = str(msisdn).strip()
        best_sub = None
        for s in subs:
            if str(s.get("status","")).upper() != "ACTIVE":
                continue
            m = _extract_msisdn_from_sub(s) or ""
            if ms and ms in m:
                best_sub = s
                break
        if not best_sub:
            return None, None
        sub_ouid = best_sub.get("ouid")
        ba = best_sub.get("billingAccountOuid")
        if not ba:
            prods = (best_sub.get("products") or {})
            for arr_key in ("PLAN","OPTION"):
                for it in (prods.get(arr_key) or []):
                    ba = ba or it.get("billingAccountOuid")
            ba = ba or GOODWILL_BA_OUID
        return sub_ouid, ba
    except Exception:
        return None, None

# ========= Bedrock response =========
def _redact_headers_for_bedrock(resp_headers: Dict[str, str]) -> list[dict]:
    # Do not leak secrets (auth/cookies) back to the caller UI.
    redacted = []
    for k, v in (resp_headers or {}).items():
        if k.lower() in {"authorization", "set-cookie"}:
            continue
        redacted.append({"name": k, "value": str(v)})
    return redacted

def _make_bedrock_response(raw_api_path: str | None, action_group: str | None,
                           tool_use_id: str | None, method: str,
                           status: int, payload: Any,
                           resp_headers: Dict[str, str]) -> Dict[str, Any]:
    # Standardizes the envelope back to the chat runtime.
    body_str = _dumps(payload) if isinstance(payload, (dict, list)) else (str(payload) if payload is not None else "")
    resp = {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": action_group,
            "apiPath": raw_api_path,
            "httpMethod": method.upper(),
            "httpStatusCode": int(status),
            "responseHeaders": _redact_headers_for_bedrock(resp_headers),
            "responseBody": { "application/json": { "body": body_str } }
        }
    }
    if tool_use_id:
        resp["response"]["toolUseId"] = tool_use_id
    return resp

# ========= Zendesk: user, tickets, conversation =========
def _zd_find_user_by_email(email: str) -> Optional[dict]:
    st, payload, _ = _zd_request("GET", "/api/v2/users/search.json", {"query": email})
    if 200 <= st < 300 and isinstance(payload, dict):
        for u in payload.get("users") or []:
            if str(u.get("email","")).lower() == email.lower():
                return u
    return None

def _zd_list_user_tickets(user_id: int, role: str = "requester", limit: int = 25) -> List[dict]:
    # Fetch recent tickets for this user; role can be requester/cc/assigned.
    role = (role or "requester").lower()
    path_by_role = {
        "requester": f"/api/v2/users/{user_id}/tickets/requested.json",
        "cc":        f"/api/v2/users/{user_id}/tickets/ccd.json",
        "assigned":  f"/api/v2/users/{user_id}/tickets/assigned.json",
    }
    tickets = []
    path = path_by_role.get(role) or path_by_role["requester"]
    page = 1
    got = 0
    while got < limit:
        st, payload, _ = _zd_request("GET", path, {"page": page, "per_page": min(100, max(1, limit - got))})
        if not (200 <= st < 300) or not isinstance(payload, dict):
            break
        for t in (payload.get("tickets") or []):
            tickets.append(t)
            got += 1
            if got >= limit:
                break
        if not payload.get("next_page"):
            break
        page += 1
    def _parse_dt(x):
        try:
            from dateutil import parser as dp
            return dp.isoparse(x)
        except Exception:
            return datetime.datetime.min
    tickets.sort(key=lambda t: _parse_dt(t.get("created_at") or t.get("updated_at") or ""), reverse=True)
    return tickets

def _zd_get_ticket_comments(ticket_id: int) -> List[dict]:
    # Get the conversation (comments) for a specific ticket id.
    st, payload, _ = _zd_request("GET", f"/api/v2/tickets/{ticket_id}/comments.json", None)
    if 200 <= st < 300 and isinstance(payload, dict):
        return payload.get("comments") or []
    return []

def _map_comment(c: dict, requester_email: str) -> dict:
    # Map raw Zendesk comment → compact, assistant-friendly structure.
    author = c.get("author_id")
    public = bool(c.get("public", True))
    body = c.get("plain_body") or c.get("body") or ""
    attachments = []
    for att in (c.get("attachments") or []):
        attachments.append({
            "file_name": att.get("file_name"),
            "size": att.get("size"),
            "content_url": att.get("content_url"),
            "content_type": att.get("content_type")
        })
    return {
        "type": "COMMENT",
        "id": c.get("id"),
        "public": public,
        "created_at": c.get("created_at"),
        "author_id": author,
        "via": (c.get("via") or {}).get("channel"),
        "body": body,
        "attachments": attachments
    }

# ========= Lambda entry =========
def lambda_handler(event, context):
    # Entry point — normalizes incoming event, routes to Zendesk / Goodwill / DTO, and applies
    #     date normalization + budget trimming before returning.
    _update_deadline_from_context(context)

    print("Incoming event keys:", list(event.keys()))
    try:
        if _is_bedrock_event(event):
            # Normalize Bedrock-style event to a common "call" shape.
            call = _extract_call_bedrock_v1(event) if "tool" in event else _extract_call_bedrock_v2(event)
            path_for_http, remaining = _substitute_path_params(call["path"], call.get("query") or {})
            call["path"], call["query"] = path_for_http, remaining
            sa = (event.get("sessionState", {}) or {}).get("sessionAttributes") if "tool" in event else (event.get("sessionAttributes") or {})

            # If DTO path has no id, inject a default customer OUID for convenience.
            call["path"], used_ouid = _normalize_customer_dto_path(call["path"], sa)

            headers = _apply_headers(call.get("headers") or {}, call.get("token"))
            print("Mode:", call["mode"], "| Call (no token shown):", {k: v for k, v in call.items() if k not in ("token","bedrock_meta","mode","raw_api_path","raw_action_group","tool_use_id")})
            print("Auth style:", "Bearer" if headers.get("Authorization","").startswith("Bearer ") else "None/Raw")

            # --- Zendesk: full conversation by email ---
            # GET /api/private/v1/zendesk/email/conversation?email=...&ticketId=...&includeInternal=false
            if call["path"] == "/api/private/v1/zendesk/email/conversation" and call["method"].upper() == "GET":
                q = call.get("query") or {}
                email = (q.get("email") or "").strip()
                if not email:
                    return _make_bedrock_response(call.get("raw_api_path") or call["path"], call.get("raw_action_group"), call.get("tool_use_id"),
                                                  call["method"], 400, {"error":"email is required"}, {})
                include_internal = str(q.get("includeInternal","false")).lower() == "true"
                role = (q.get("role") or "requester").lower()
                limit = int(q.get("limit") or ZD_DEFAULT_LIMIT)
                ticket_id = q.get("ticketId")

                user = _zd_find_user_by_email(email)
                if not user:
                    return _make_bedrock_response(call.get("raw_api_path") or call["path"], call.get("raw_action_group"), call.get("tool_use_id"),
                                                  call["method"], 404, {"error": "Zendesk user not found", "email": email}, {})

                # Choose a ticket (newest by default) then fetch comments.
                if not ticket_id:
                    tickets = _zd_list_user_tickets(int(user["id"]), role=role, limit=max(5, limit))
                    if not tickets:
                        out = {"user": {"id": user["id"], "name": user.get("name"), "email": user.get("email")},
                               "conversation": [], "count": 0}
                        return _make_bedrock_response(call.get("raw_api_path") or call["path"], call.get("raw_action_group"),
                                                      call.get("tool_use_id"), call["method"], 200, out, {
                                                          "X-ZD-Subdomain": _zd_base_url().split("//",1)[1],
                                                          "X-ZD-Role": role,
                                                          "X-ZD-Limit": str(limit)
                                                      })
                    ticket_id = tickets[0].get("id")

                comments = _zd_get_ticket_comments(int(ticket_id)) or []
                convo = []
                for c in comments:
                    if not include_internal and not c.get("public", True):
                        continue
                    convo.append(_map_comment(c, user.get("email","")))

                out = {
                    "user": {"id": user["id"], "name": user.get("name"), "email": user.get("email")},
                    "ticket": {"id": int(ticket_id)},
                    "conversation": convo,
                    "count": len(convo)
                }

                # Make dates readable, then enforce token budget.
                if EPOCH_REPLACE_ENABLED:
                    out, _ = _replace_epoch_dates(out)

                out, trimmed, t_before, t_after, info, trimmed_count = _enforce_global_budget(out, TOKEN_LIMIT)
                out = _append_token_counters(out, t_before, t_after, TOKEN_LIMIT)
                rh = {"X-ZD-Subdomain": _zd_base_url().split("//",1)[1], "X-ZD-Role": role, "X-ZD-Limit": str(limit)}
                if trimmed:
                    rh["X-Content-Trimmed"] = info or "conversation"
                    rh["X-Trimmed-Count"] = str(trimmed_count)

                return _make_bedrock_response(call.get("raw_api_path") or call["path"], call.get("raw_action_group"),
                                              call["tool_use_id"], call["method"], 200, out, rh)

            # --- Goodwill entrypoints ---
            # Either explicit test endpoint or a convenience POST to productOrder with empty body.
            if (
                (call["path"] == "/api/private/v1/agent/test/addDataGoodwill" and call["method"].upper() == "POST") or
                (call["path"] == "/api/private/v1/agent/productOrder" and call["method"].upper() == "POST" and not call.get("body"))
            ):
                gw = _merge_goodwill_overrides(call.get("body"), sa)

                # If we only have msisdn, try to resolve the right subscription & BA automatically.
                if gw.get("msisdn") and (not gw.get("parentOuid") or gw.get("parentOuid") == GOODWILL_PARENT_OUID):
                    sub_ouid, ba_ouid = _try_resolve_subscription_from_msisdn(gw["msisdn"], gw["customerOuid"], headers)
                    if sub_ouid: gw["parentOuid"] = sub_ouid
                    if ba_ouid:  gw["billingAccountOuid"] = ba_ouid

                status, msg, rh = _call_product_order_for_goodwill(headers, gw)
                rh = dict(rh or {})
                rh["X-Goodwill-SizeGb"] = str(gw.get("sizeGb", 2))
                rh["X-Goodwill-Reason"] = gw.get("reason", "boosterOrPassRefund")
                rh["X-Goodwill-CustomerOuid"] = gw.get("customerOuid", "")
                if gw.get("msisdn"):
                    rh["X-Goodwill-Msisdn"] = gw["msisdn"]

                payload_out = {"productOrderResult": msg, "status": int(status)}

                # Optional sanity check — fetch DTO after goodwill and attach snapshot (~"before/after" feel).
                try:
                    if VERIFY_AFTER_GOODWILL and 200 <= int(status) < 300:
                        dto_headers = dict(headers)
                        dto_headers["X-Channel"] = "mobile-app"
                        dto_path = f"/api/private/v1/agent/customer/customerDto/{gw.get('customerOuid') or _resolve_default_customer_ouid(sa)}"
                        dto_status, dto_payload, _ = _do_request("GET", _mk_url(dto_path, None), dto_headers, None)
                        if isinstance(dto_payload, dict):
                            _annotate_dto_avg_inplace(dto_payload)
                            if SNAPSHOT_ENABLED:
                                snap = _build_state_snapshot(dto_payload)
                                dto_payload["_snapshot"] = snap
                                dto_payload["_snapshotSha1"] = _sha1(_dumps(snap))
                        payload_out["postVerify"] = {"status": int(dto_status), "dto": dto_payload}
                        rh["X-PostVerify-Status"] = str(dto_status)
                except Exception as _e:
                    rh["X-PostVerify-Error"] = str(_e)[:200]

                return _make_bedrock_response(call.get("raw_api_path") or call["path"],
                                              call.get("raw_action_group") or (call.get("bedrock_meta") or {}).get("actionGroup"),
                                              call.get("tool_use_id"),
                                              call["method"], status, payload_out, rh)

            # --- DTO passthrough (deadline aware) ---
            if call["path"].startswith("/api/private/v1/agent/customer/customerDto"):
                # Enforce channel header expected by some setups.
                headers["X-Channel"] = "mobile-app"
                url = _mk_url(call["path"], call.get("query") or {})
                m = re.match(r"^/api/private/v1/agent/customer/customerDto/([A-F0-9]+)$", call["path"])

                # GET with id → use cache. Others → passthrough.
                if call["method"].upper() == "GET" and m:
                    status, payload = _fetch_dto_for_customer(m.group(1), headers)
                    resp_headers = {}
                else:
                    status, payload, resp_headers = _do_request(call["method"], url, headers, call.get("body"))

                # Dates → ISO
                if EPOCH_REPLACE_ENABLED and isinstance(payload, (dict, list)):
                    payload, _repl_cnt = _replace_epoch_dates(payload)
                    resp_headers = {**(resp_headers or {}), "X-Epoch-Replaced": str(_repl_cnt)}

                # Add summary stats + snapshot so the assistant can do "one-look" reasoning.
                if isinstance(payload, dict):
                    try:
                        summary = _annotate_dto_avg_inplace(payload)
                        if SNAPSHOT_ENABLED:
                            snap = _build_state_snapshot(payload)
                            payload["_snapshot"] = snap
                            payload["_snapshotSha1"] = _sha1(_dumps(snap))
                        resp_headers = dict(resp_headers or {})
                        resp_headers["X-Active-Lines"] = str(summary.get("activeLineCount", 0))
                        resp_headers["X-Avg-Line-Monthly-Cents"] = str(summary.get("avgMonthlyCents", 0))
                        resp_headers["X-Avg-Line-Monthly"] = f"{summary.get('avgMonthly', 0.0):.2f}"
                    except Exception as _e:
                        resp_headers = dict(resp_headers or {})
                        resp_headers["X-Avg-Compute-Error"] = str(_e)[:200]

                # Keep response within token budget & attach diagnostics.
                payload, trimmed, t_before, t_after, info, trimmed_count = _enforce_global_budget(payload, TOKEN_LIMIT)
                payload = _append_token_counters(payload, t_before, t_after, TOKEN_LIMIT)
                if trimmed:
                    resp_headers = dict(resp_headers or {})
                    resp_headers["X-Content-Trimmed"] = info or "notifications"
                    resp_headers["X-Token-Estimate-Before"] = str(t_before)
                    resp_headers["X-Token-Estimate-After"]  = str(t_after)
                    resp_headers["X-Trimmed-Count"] = str(trimmed_count)
                if used_ouid:
                    resp_headers = dict(resp_headers or {})
                    resp_headers["X-Used-CustomerOuid"] = used_ouid

                return _make_bedrock_response(call.get("raw_api_path") or call["path"],
                                              call.get("raw_action_group") or (call.get("bedrock_meta") or {}).get("actionGroup"),
                                              call["tool_use_id"],
                                              call["method"], status, payload, resp_headers)

            # --- Default passthrough ---
            # For any other backend route, just forward with the time/budget safety nets.
            url = _mk_url(call["path"], call.get("query") or {})
            status, payload, resp_headers = _do_request(call["method"], url, headers, call.get("body"))
            payload, trimmed, t_before, t_after, info, trimmed_count = _enforce_global_budget(payload, TOKEN_LIMIT)
            payload = _append_token_counters(payload, t_before, t_after, TOKEN_LIMIT)
            if trimmed:
                resp_headers = dict(resp_headers or {})
                resp_headers["X-Content-Trimmed"] = info or "payload"
                resp_headers["X-Trimmed-Count"] = str(trimmed_count)

            return _make_bedrock_response(call.get("raw_api_path") or call["path"],
                                          call.get("raw_action_group") or (call.get("bedrock_meta") or {}).get("actionGroup"),
                                          call.get("tool_use_id"),
                                          call["method"], status, payload, resp_headers)

        # ===== Console / direct invoke =====
        else:
            # Same business rules as above, but returns a simple JSON envelope for local tests.
            _update_deadline_from_context(context)  # ensure timeouts set for console path too
            call = _extract_call_console(event)
            path_for_http, remaining = _substitute_path_params(call["path"], call.get("query") or {})
            call["path"], call["query"] = path_for_http, remaining
            call["path"], _ = _normalize_customer_dto_path(call["path"], None)

            headers = _apply_headers(call.get("headers") or {}, call.get("token"))

            # Goodwill (console)
            if call["path"] == "/api/private/v1/agent/test/addDataGoodwill" and call["method"].upper() == "POST":
                gw = _merge_goodwill_overrides(call.get("body"), None)
                status, msg, rh = _call_product_order_for_goodwill(headers, gw)
                return {"statusCode": int(status),
                        "body": _dumps({"productOrderResult": msg, "status": int(status)}),
                        "headers": rh}

            # DTO (console): ensure mobile-app channel & cache GET
            if call["path"].startswith("/api/private/v1/agent/customer/customerDto/"):
                headers["X-Channel"] = "mobile-app"

            url = _mk_url(call["path"], call.get("query") or {})

            m = re.match(r"^/api/private/v1/agent/customer/customerDto/([A-F0-9]+)$", call["path"])
            if call["method"].upper() == "GET" and m:
                status, payload = _fetch_dto_for_customer(m.group(1), headers)
                resp_headers = {}
            else:
                status, payload, resp_headers = _do_request(call["method"], url, headers, call.get("body"))

            if EPOCH_REPLACE_ENABLED and isinstance(payload, (dict, list)):
                payload, _repl_cnt = _replace_epoch_dates(payload)

            if isinstance(payload, dict) and call["path"].startswith("/api/private/v1/agent/customer/customerDto/"):
                try:
                    _annotate_dto_avg_inplace(payload)
                    if SNAPSHOT_ENABLED:
                        snap = _build_state_snapshot(payload)
                        payload["_snapshot"] = snap
                        payload["_snapshotSha1"] = _sha1(_dumps(snap))
                except Exception:
                    pass

            payload, trimmed, t_before, t_after, info, trimmed_count = _enforce_global_budget(payload, TOKEN_LIMIT)
            payload = _append_token_counters(payload, t_before, t_after, TOKEN_LIMIT)

            result = {
                "ok": 200 <= status < 300,
                "status": status,
                "url": url,
                "responseHeaders": resp_headers if 200 <= status < 300 else {},
                "data": payload if 200 <= status < 300 else None,
                "error": None if 200 <= status < 300 else "HTTPError",
                "message": None if 200 <= status < 300 else (payload if isinstance(payload, str) else _dumps(payload)),
                "trimmed": trimmed,
                "tokenEstimateBefore": t_before,
                "tokenEstimateAfter": t_after,
                "tokenLimit": TOKEN_LIMIT,
                "trimmedInfo": info,
                "trimmedCount": trimmed_count,
            }
            return {"statusCode": 200, "body": _dumps(result), "headers": {"Content-Type": "application/json"}}

    # ===== Error handling with helpful payloads =====
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        if _is_bedrock_event(event):
            call = _extract_call_bedrock_v1(event) if "tool" in event else _extract_call_bedrock_v2(event)
            return _make_bedrock_response(
                raw_api_path = call.get("raw_api_path") or call["path"],
                action_group = call.get("raw_action_group") or (call.get("bedrock_meta") or {}).get("actionGroup"),
                tool_use_id  = call.get("tool_use_id"),
                method       = call["method"],
                status       = getattr(e, "code", 500),
                payload      = {"error": "HTTPError", "message": msg, "_tokenLimit": TOKEN_LIMIT},
                resp_headers = {}
            )
        return {"statusCode": 200, "body": _dumps({
                "ok": False, "status": getattr(e, "code", 0),
                "url": event.get("path"), "error": "HTTPError", "message": msg}),
                "headers": {"Content-Type": "application/json"}}

    except Exception as e:
        msg = repr(e)
        if _is_bedrock_event(event):
            call = _extract_call_bedrock_v1(event) if "tool" in event else _extract_call_bedrock_v2(event)
            return _make_bedrock_response(
                raw_api_path = call.get("raw_api_path") or call["path"],
                action_group = call.get("raw_action_group") or (call.get("bedrock_meta") or {}).get("actionGroup"),
                tool_use_id  = call.get("tool_use_id"),
                method       = call["method"],
                status       = 500,
                payload      = {"error": msg, "_tokenLimit": TOKEN_LIMIT},
                resp_headers = {}
            )
        return {"statusCode": 200, "body": _dumps({
                "ok": False, "status": 0,
                "url": event.get("path"), "error": "Exception", "message": msg}),
                "headers": {"Content-Type": "application/json"}}
