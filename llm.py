"""Shared LLM client with provider failover.

Every provider here speaks the OpenAI /chat/completions shape, so one client covers
them all and the list is just an ordered fallback chain:

  nim         NVIDIA NIM, free credits, but its free tier drops requests under load
  groq        fast free tier, no card, ~14,400 requests/day, needs GROQ_API_KEY
  openrouter  free models only, ~50 requests/day, needs OPENROUTER_API_KEY

Free tiers only, with no opt-out: an OpenRouter model is dropped from the chain unless
its catalogue price is verified to be zero.

A provider is tried until its retry budget is spent, then the next one takes over.
Runs used to die outright when NIM read-timed out three times in a row; with a second
key configured that becomes a log line instead of a lost slot.

Also handled here, because every caller needs it: reasoning models return a null
content field when the token budget runs out mid-thought, and a wrong model id must
fail fast rather than be retried as though it were transient.
"""
import json, os, random, re, time

import requests

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODELS_URL = "https://integrate.api.nvidia.com/v1/models"
NIM_MODEL = (os.environ.get("NIM_MODEL") or "").strip() or "meta/llama-3.3-70b-instruct"
# The free tiers drop requests under load and reasoning models answer slowly, so the
# per-call ceiling is generous and the retry budget is what actually protects a run.
NIM_TIMEOUT = int(os.environ.get("NIM_TIMEOUT", "180"))
NIM_ATTEMPTS = int(os.environ.get("NIM_ATTEMPTS", "5"))


def log(msg): print(f"[llm] {msg}", flush=True)


class TransientNimError(Exception):
    """Worth retrying. A 4xx other than 408/429 is a config error and never is."""


class Provider:
    def __init__(self, name, url, key, model, attempts, extra_headers=None):
        self.name, self.url, self.key, self.model = name, url, key, model
        self.attempts = attempts
        self.extra_headers = extra_headers or {}

    def __repr__(self):
        return f"{self.name}({self.model})"


# This pipeline runs on free tiers only, with no opt-out. NIM runs on free credits and
# Groq's free tier needs no card, but OpenRouter serves paid and free models side by
# side through one key, so a mistyped model id there is the one way it could start
# spending money. Any OpenRouter model whose price is not zero is refused.
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
# One of the free models available on OpenRouter; verified against their catalogue.
OPENROUTER_DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


def _env(name):
    value = (os.environ.get(name) or "").strip()
    return "" if value.lower() in ("", "xxxx") else value


# Free tiers cap by BOTH per-minute and per-day. The critic loop makes 4+ rapid calls
# per script (write, review, revise, review) which trips per-minute ceilings even when
# daily quota is fine. Space calls to the same provider by at least this many seconds
# to keep 429s out of the log; provider-specific because their limits differ.
_MIN_INTERVAL = {"groq": 2.5, "nim": 0.5, "openrouter": 1.5, "ollama": 0.0}
_last_call = {}


def _throttle(provider):
    """Sleep just enough that the next request stays under the provider's per-minute
    limit. Zero cost when we're already spaced out."""
    gap = _MIN_INTERVAL.get(provider.name, 1.0)
    elapsed = time.time() - _last_call.get(provider.name, 0)
    if elapsed < gap:
        time.sleep(gap - elapsed)


def openrouter_is_free(model):
    """Confirm with OpenRouter that the model really costs nothing. The catalogue is
    public, so this needs no key, and it catches a ':free' id that has been retired or
    repriced rather than trusting the suffix alone."""
    try:
        models = requests.get(OPENROUTER_MODELS_URL, timeout=30).json()["data"]
    except Exception as e:
        # Fail closed. An unverifiable price is not a free price, and skipping one
        # fallback provider costs far less than an unexpected bill.
        log(f"could not verify OpenRouter pricing ({type(e).__name__}); skipping it")
        return False
    for m in models:
        if m["id"] == model:
            pricing = m.get("pricing") or {}
            free = all(float(pricing.get(k, 0) or 0) == 0 for k in ("prompt", "completion"))
            if not free:
                log(f"OpenRouter model {model} is not free: {pricing}")
            return free
    log(f"OpenRouter does not list {model}; free ids currently include "
        + ", ".join(m["id"] for m in models if m["id"].endswith(":free"))[:200])
    return False


def providers():
    """The fallback chain, in order. Groq goes first because its free tier is fast
    (sub-second) and generous (~14,400 req/day, no card); NIM stays as backup for when
    Groq's daily quota drains. NIM's free credits used to run first, but its free-tier
    endpoint routinely stalls past our 180s read timeout, burning ~15 min of wall-clock
    per broken call before the chain rolled over. Fast path is now the default."""
    chain = []
    if _env("GROQ_API_KEY"):
        chain.append(Provider(
            "groq", "https://api.groq.com/openai/v1/chat/completions",
            _env("GROQ_API_KEY"),
            (os.environ.get("GROQ_MODEL") or "").strip() or "llama-3.3-70b-versatile",
            int(os.environ.get("GROQ_ATTEMPTS", "3")),
        ))
    if _env("NIM_API_KEY"):
        chain.append(Provider("nim", NIM_URL, _env("NIM_API_KEY"), NIM_MODEL, NIM_ATTEMPTS))
    # OpenRouter last: ~50 req/day on free models vs Groq's ~14,400. Only useful when
    # both Groq and NIM are down for the same run.
    if _env("OPENROUTER_API_KEY"):
        model = (os.environ.get("OPENROUTER_MODEL") or "").strip() or OPENROUTER_DEFAULT_MODEL
        if openrouter_is_free(model):
            chain.append(Provider(
                "openrouter", "https://openrouter.ai/api/v1/chat/completions",
                _env("OPENROUTER_API_KEY"), model,
                int(os.environ.get("OPENROUTER_ATTEMPTS", "3")),
                # OpenRouter attributes traffic with these; harmless, and it keeps the
                # free-tier rate limits applied per app rather than per IP.
                {"HTTP-Referer": "https://github.com/codeaz-org/mpt",
                 "X-Title": "mpt-autopilot"},
            ))
        else:
            log(f"skipping OpenRouter: {model} is not free. Set OPENROUTER_MODEL to "
                "one of the ':free' ids listed above.")
    # Ollama last. Runs locally (or in the GitHub Actions runner during a workflow),
    # no API key, no quota. Slow on CPU-only hosts, so it only fires when everything
    # hosted is down or quota-drained. Gated on OLLAMA_URL so it doesn't accidentally
    # connect to a dev's laptop during a manual run.
    ollama_url = (os.environ.get("OLLAMA_URL") or "").strip()
    if ollama_url:
        if not ollama_url.rstrip("/").endswith("/chat/completions"):
            ollama_url = ollama_url.rstrip("/") + "/v1/chat/completions"
        chain.append(Provider(
            "ollama", ollama_url,
            # Ollama accepts any Authorization header; a placeholder keeps _ask uniform.
            "ollama",
            (os.environ.get("OLLAMA_MODEL") or "").strip() or "llama3.1:8b",  # matches workflow default
            int(os.environ.get("OLLAMA_ATTEMPTS", "2")),
        ))
    if not chain:
        raise RuntimeError("no LLM provider configured: set GROQ_API_KEY, NIM_API_KEY, "
                           "OPENROUTER_API_KEY, or OLLAMA_URL")
    return chain


def _config_error(provider, status, body):
    """Turn a terse 4xx into something that names the actual mistake."""
    if status == 404:
        near = []
        if provider.name == "nim":
            try:
                available = [m["id"] for m in
                             requests.get(NIM_MODELS_URL, timeout=30).json()["data"]]
                vendor = provider.model.split("/")[0].lower()
                near = [m for m in available if vendor in m.lower()][:6]
            except Exception:
                pass
        hint = f" Models available: {', '.join(near)}." if near else ""
        return RuntimeError(f"{provider.name} does not serve model {provider.model!r} "
                            f"(404). Check the model id.{hint}")
    if status in (401, 403):
        return RuntimeError(f"{provider.name} rejected the API key ({status}).")
    return RuntimeError(f"{provider.name} request failed: {status} {body[:200]}")


def _nim_content(data):
    """Reasoning models (gpt-oss, deepseek) put their thinking in reasoning_content and
    leave content null when the budget runs out mid-thought. Return the answer text plus
    the finish reason so the caller can grow the budget and retry."""
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = (message.get("content") or "").strip()
    reasoned = bool((message.get("reasoning_content") or "").strip())
    return content, choice.get("finish_reason"), reasoned


def _ask(provider, system, user, temperature, max_tokens):
    """One provider, with its own retry budget. Raises on exhaustion or misconfiguration."""
    last = None
    budget = max_tokens
    for i in range(provider.attempts):
        try:
            _throttle(provider)
            r = requests.post(
                provider.url,
                headers={"Authorization": f"Bearer {provider.key}", **provider.extra_headers},
                json={
                    "model": provider.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": temperature,
                    "max_tokens": budget,
                },
                timeout=NIM_TIMEOUT,
            )
            if r.status_code in (408, 429) or r.status_code >= 500:
                raise TransientNimError(f"{r.status_code} {r.text[:200]}")
            if r.status_code >= 400:
                raise _config_error(provider, r.status_code, r.text)
            content, finish, reasoned = _nim_content(r.json())
            if content:
                _last_call[provider.name] = time.time()
                return content
            why = ("spent the whole budget reasoning" if reasoned
                   else f"returned no content ({finish})")
            raise TransientNimError(f"{provider.model} {why} at max_tokens={budget}")
        except (requests.Timeout, requests.ConnectionError, TransientNimError) as e:
            last = e
            empty = isinstance(e, TransientNimError) and "max_tokens" in str(e)
            if empty:
                budget = min(budget * 4, 8000)  # reasoning ate the budget: give it room
            if i < provider.attempts - 1:
                wait = 0 if empty else min(5 * 2 ** i, 45) + random.uniform(0, 3)
                log(f"{provider.name} failed ({type(e).__name__}), attempt "
                    f"{i + 1}/{provider.attempts}, retry"
                    f"{'' if empty else f' in {wait:.0f}s'}: {str(e)[:140]}")
                if wait:
                    time.sleep(wait)
    raise last


def nim_chat(system, user, temperature=0.9, attempts=None, max_tokens=512):
    """Ask each configured provider in turn; the first usable answer wins."""
    chain = providers()
    last = None
    for index, provider in enumerate(chain):
        if attempts:
            provider.attempts = attempts
        try:
            answer = _ask(provider, system, user, temperature, max_tokens)
            if index:
                log(f"answered by fallback provider {provider.name}")
            return answer
        except Exception as e:
            last = e
            remaining = chain[index + 1:]
            log(f"{provider.name} exhausted ({type(e).__name__}: {str(e)[:120]})"
                + (f"; switching to {remaining[0].name}" if remaining else ""))
    raise last


def nim_json(system, user, temperature=0.6, max_tokens=1200):
    """Chat call that must return JSON. Models wrap answers in prose or fences often
    enough that the raw text is rarely parseable on its own."""
    raw = nim_chat(
        system + " Respond ONLY with valid JSON. No prose, no markdown fences.",
        user, temperature=temperature, max_tokens=max_tokens,
    )
    cleaned = re.sub(r"```[a-z]*|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"[{\[].*[}\]]", cleaned, re.S)
        if not match:
            raise RuntimeError(f"model returned no JSON: {cleaned[:200]}")
        return json.loads(match.group(0))
