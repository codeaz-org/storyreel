"""Reddit fetchers, trimmed from mpt for the story pipeline.

Four ways into a subreddit, best first: official OAuth (needs free app creds,
live scores), Arctic Shift (keyless archive, works from CI IPs), anonymous
JSON (403s from datacenter IPs), RSS (titles only). Story channels should set
REDDIT_CLIENT_ID/SECRET -- votes are the whole quality filter.
"""
import html, json, os, random, re, time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
SAAS_FILE = ROOT / "saas_ideas.json"
UA = {"User-Agent": "python:mpt-autopilot:v1.1 (by /u/mpt-autopilot)"}
MIN_POSTS = 5
DIGEST_POSTS = 20   # how many make it into the prompt; the rest only affect ranking


def log(msg): print(f"[research] {msg}", flush=True)


def _get(url, attempts=3, headers=None, **kw):
    """GET with backoff. 429 and 5xx are retried, 403 is not -- it means the source
    is blocking this IP outright and no amount of waiting changes that."""
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, headers=headers or UA, timeout=30, **kw)
            if r.status_code == 429:
                wait = min(float(r.headers.get("Retry-After", 0) or 2 ** i * 5), 60)
                log(f"rate limited, waiting {wait:.0f}s")
                time.sleep(wait)
                last = RuntimeError("429 Too Many Requests")
                continue
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                raise
            last = e
        except requests.RequestException as e:
            last = e
        if i < attempts - 1:
            time.sleep(2 ** i * 3)
    raise last


_oauth_token = None


def reddit_token():
    """App-only OAuth. Reddit serves oauth.reddit.com to datacenter IPs that it refuses
    on www.reddit.com, so this is the only path that works from a CI runner. Free:
    create a 'script' app at reddit.com/prefs/apps and set REDDIT_CLIENT_ID/SECRET."""
    global _oauth_token
    if _oauth_token:
        return _oauth_token
    cid = (os.environ.get("REDDIT_CLIENT_ID") or "").strip()
    secret = (os.environ.get("REDDIT_CLIENT_SECRET") or "").strip()
    if not cid or not secret or cid.lower() == "xxxx":
        raise RuntimeError("REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET not set")
    r = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=(cid, secret), headers=UA, timeout=30,
        data={"grant_type": "client_credentials"},
    )
    r.raise_for_status()
    _oauth_token = r.json()["access_token"]
    return _oauth_token


def _parse_listing(payload, text_chars=400):
    posts = []
    for child in payload["data"]["children"]:
        d = child["data"]
        if d.get("stickied") or d.get("over_18"):
            continue
        posts.append({
            "title": d.get("title", ""),
            "text": (d.get("selftext") or "")[:text_chars],
            "score": d.get("score", 0),
            "num_comments": d.get("num_comments", 0),
            "id": f"reddit:{d.get('id')}",
            "url": "https://reddit.com" + (d.get("permalink") or ""),
            "source": f"r/{d.get('subreddit', '')}",
        })
    return posts


def fetch_subreddit_oauth(sub, listing="hot", limit=15, text_chars=400):
    token = reddit_token()
    r = _get(f"https://oauth.reddit.com/r/{sub}/{listing}?limit={limit}&raw_json=1",
             headers={**UA, "Authorization": f"bearer {token}"})
    return _parse_listing(r.json(), text_chars)


ARCTIC_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"


def fetch_subreddit_arctic(sub, limit=60, hours_back=None, min_age_hours=18, text_chars=400):
    """Arctic Shift -- a free, keyless, open-source Reddit archive API. It answers from
    its own servers, so it works from IPs Reddit refuses, no app registration needed.

    Scores are captured near post creation, so anything newer than min_age_hours reads
    as 1 point. Asking for a window that already accumulated votes gives real ranking
    while staying recent enough to reflect what people are on about this week.

    The window length is randomised: a fixed one returns the same top posts every run,
    which produced the same pain points and, eventually, the same video twice."""
    hours_back = hours_back or random.choice((72, 96, 144, 216))
    now = int(time.time())
    r = _get(ARCTIC_URL, params={
        "subreddit": sub, "limit": limit, "sort": "desc", "sort_type": "created_utc",
        "after": now - hours_back * 3600, "before": now - min_age_hours * 3600,
    })
    posts = []
    for d in r.json().get("data", []):
        if d.get("stickied") or d.get("over_18") or d.get("removed_by_category"):
            continue
        title = (d.get("title") or "").strip()
        if not title:
            continue
        posts.append({
            "title": title,
            "text": (d.get("selftext") or "")[:text_chars],
            "score": d.get("score") or 0,
            "num_comments": d.get("num_comments") or 0,
            "id": f"reddit:{d.get('id')}",
            "url": "https://reddit.com" + (d.get("permalink") or f"/r/{sub}"),
            "source": f"r/{sub}",
        })
    return posts


def fetch_subreddit_json(sub, listing="hot", limit=15, text_chars=400):
    r = _get(f"https://www.reddit.com/r/{sub}/{listing}.json?limit={limit}&raw_json=1")
    return _parse_listing(r.json(), text_chars)


def fetch_subreddit_rss(sub, listing="hot", limit=15):
    """Fallback for IPs Reddit blocks from the JSON endpoints. Titles only: the RSS
    feed carries no score, so ranking falls back to feed order. One retry only --
    a blocked IP stays blocked, and other sources are waiting."""
    r = _get(f"https://www.reddit.com/r/{sub}/{listing}.rss?limit={limit}", attempts=2)
    entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.S)
    posts = []
    for e in entries:
        m = re.search(r"<title>(.*?)</title>", e, re.S)
        if not m:
            continue
        title = html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
        body = re.search(r"<content[^>]*>(.*?)</content>", e, re.S)
        text = html.unescape(re.sub(r"<[^>]+>", " ", body.group(1))) if body else ""
        posts.append({"title": title, "text": re.sub(r"\s+", " ", text)[:400],
                      "score": 0, "num_comments": 0})
    return posts


_reddit_blocked = False


def fetch_subreddit(sub, listing="hot", limit=15, text_chars=400):
    """Four ways in, best first:

      oauth    official API, needs free app credentials, richest and most current
      arctic   Arctic Shift, keyless and unblocked, works from CI runners
      json     www.reddit.com, 403s from datacenter IPs
      rss      same, and rate limits almost immediately

    Reddit blocks by IP range rather than per subreddit, so once anonymous access fails
    it is remembered and later subreddits skip straight past it."""
    global _reddit_blocked
    if os.environ.get("REDDIT_CLIENT_ID"):
        try:
            return fetch_subreddit_oauth(sub, listing, limit, text_chars)
        except Exception as e:
            log(f"r/{sub}/{listing} oauth failed ({type(e).__name__}: {str(e)[:70]})")
    try:
        posts = fetch_subreddit_arctic(sub, text_chars=text_chars)
        if posts:
            return posts
        log(f"r/{sub} arctic returned nothing, trying anonymous")
    except Exception as e:
        log(f"r/{sub} arctic failed ({type(e).__name__}: {str(e)[:70]}), trying anonymous")

    if _reddit_blocked:
        raise RuntimeError("reddit is blocking this IP (skipped)")
    try:
        return fetch_subreddit_json(sub, listing, limit, text_chars)
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        log(f"r/{sub}/{listing} json blocked ({code}), trying rss")
    try:
        return fetch_subreddit_rss(sub, listing, limit)
    except Exception:
        _reddit_blocked = True
        raise


