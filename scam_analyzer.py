import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests


def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ[key.strip()] = value.strip()

_load_dotenv()

LLM_API_BASE_URL = os.environ.get("LLM_API_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "google/gemma-4-31b-it")


_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
with open(_HTML_PATH, encoding="utf-8") as _f:
    WEB_PAGE = _f.read()





class ScamAnalyzerHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        if self.path not in ("/", "/index.html"):
            self.send_error(404, "Not found")
            return
        body = WEB_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_error(404, "Not found")
            return
        body = WEB_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/analyze":
            self.send_error(404, "Not found")
            return
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            self._send_json(411, {"error": "Missing Content-Length header."})
            return
        try:
            body_size = int(content_length)
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length header."})
            return
        if body_size <= 0:
            self._send_json(400, {"error": "Request body is empty."})
            return

        body = self.rfile.read(body_size)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Request body must be valid JSON."})
            return

        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            self._send_json(400, {"error": "A non-empty 'message' field is required."})
            return

        try:
            result = analyze_with_llm(message.strip())
        except RuntimeError as exc:
            self._send_json(502, {"error": str(exc)})
            return
        except ValueError as exc:
            self._send_json(502, {"error": f"Invalid model response: {exc}"})
            return

        self._send_json(200, result)


REQUIRED_SIGNALS = ("urgency", "threat", "financial_request", "link_present", "impersonation")
ANALYSIS_PROMPT = (
    "You are a scam detection API. Analyze the user's message for scam indicators.\n"
    "You MUST respond with ONLY a JSON object. No explanation, no markdown, no text before or after.\n"
    "Use this EXACT schema:\n"
    '{"score": <0-100>, "risk": "<low|medium|high>", "confidence": <0-100>, '
    '"reasons": ["<reason1>", "<reason2>", "<reason3>"], '
    '"advice": "<one sentence recommendation>", '
    '"signals": {"urgency": <true|false>, "threat": <true|false>, "financial_request": <true|false>, '
    '"link_present": <true|false>, "impersonation": <true|false>}}\n'
    "Example — if the message is 'You won a free iPhone, click here now!':\n"
    '{"score": 85, "risk": "high", "confidence": 90, '
    '"reasons": ["Creates false urgency", "Promises unrealistic free prize", "Contains suspicious link request"], '
    '"advice": "Do not click any links or share personal information.", '
    '"signals": {"urgency": true, "threat": false, "financial_request": false, '
    '"link_present": true, "impersonation": false}}\n'
    "IMPORTANT: Output ONLY the raw JSON object. No markdown, no code fences, no extra text."
)


def clamp_int(value, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def normalize_analysis(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Model output is not an object.")

    try:
        score = clamp_int(raw.get("score", 0), 0, 100)
    except (TypeError, ValueError):
        raise ValueError("score must be an integer.")
    risk = raw.get("risk")
    if not isinstance(risk, str) or risk not in ("low", "medium", "high"):
        risk = "high" if score >= 70 else "medium" if score >= 30 else "low"
    try:
        confidence = clamp_int(raw.get("confidence", 50), 0, 100)
    except (TypeError, ValueError):
        raise ValueError("confidence must be an integer.")

    reasons = raw.get("reasons")
    if not isinstance(reasons, list):
        raise ValueError("reasons must be an array.")
    reason_text = [str(item).strip() for item in reasons if str(item).strip()]
    reason_text = reason_text[:3]
    while len(reason_text) < 3:
        reason_text.append("No additional risk signals were returned")

    advice = raw.get("advice")
    if not isinstance(advice, str) or not advice.strip():
        raise ValueError("advice must be a non-empty string.")

    raw_signals = raw.get("signals")
    if not isinstance(raw_signals, dict):
        raise ValueError("signals must be an object.")
    signals = {}
    for signal in REQUIRED_SIGNALS:
        signals[signal] = bool(raw_signals.get(signal, False))

    return {
        "score": score,
        "risk": risk,
        "confidence": confidence,
        "reasons": reason_text,
        "advice": advice.strip(),
        "signals": signals,
    }


def extract_json_payload(content: str) -> dict:
    text = content.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # Try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # Find the first balanced { ... } block
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            if in_string:
                escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    break
    # Last resort: greedy regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON is not an object.")
    return parsed


def analyze_with_llm(message: str) -> dict:
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set. Add it to the .env file.")

    payload = {
        "model": LLM_MODEL,
        "temperature": 0.1,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": ANALYSIS_PROMPT},
            {"role": "user", "content": message},
        ],
    }
    try:
        request = urllib.request.Request(
            LLM_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=45) as resp:
            decoded = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"API request failed with status {exc.code}: {body}")
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise RuntimeError("API request timed out.")
        raise RuntimeError(f"API connection error: {exc.reason}")

    choices = decoded.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("API response did not include choices.")

    message_payload = choices[0].get("message")
    if not isinstance(message_payload, dict):
        raise RuntimeError("API response is missing message content.")
    content = message_payload.get("content")
    if isinstance(content, list):
        text_chunks = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_chunks.append(part["text"])
        content = "".join(text_chunks)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("API response content is empty.")

    parsed = extract_json_payload(content)
    return normalize_analysis(parsed)


def run_server(port: int = 8000):
    server = HTTPServer(("0.0.0.0", port), ScamAnalyzerHandler)
    print(f"Scam Analyzer running at http://0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Port must be an integer.", file=sys.stderr)
            sys.exit(1)
    run_server(port)
