#!/usr/bin/env python3
"""One-time (per channel) helper: prints a YouTube refresh token.

Run locally on a browser-capable machine:
    pip install google-auth-oauthlib
    NICHE=moneymech python get_youtube_token.py

Sign in with the Google account that owns THAT channel, then save the printed token as
the secret named at the end.

Upload quota is per Google Cloud project (~6 uploads/day), so a channel with its own
Google account and project gets its own ceiling. A refresh token only works with the
OAuth client that minted it, so when a niche has its own project set
YT_CLIENT_ID_<NICHEID> and YT_CLIENT_SECRET_<NICHEID> as well; this script uses them
when present and falls back to the shared pair.
"""
import os, sys
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

_env = Path(__file__).resolve().parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

NICHE = (os.environ.get("NICHE") or (sys.argv[1] if len(sys.argv) > 1 else "")).strip()
if not NICHE:
    sys.exit("Usage: NICHE=<nicheid> python get_youtube_token.py   (e.g. NICHE=moneymech)")
SUFFIX = NICHE.upper()


def pick(name):
    value = (os.environ.get(f"{name}_{SUFFIX}") or "").strip()
    if value:
        print(f"using {name}_{SUFFIX}")
        return value
    value = (os.environ.get(name) or "").strip()
    if not value:
        sys.exit(f"Set {name}_{SUFFIX} (per-project) or {name} (shared)")
    return value


flow = InstalledAppFlow.from_client_config(
    {
        "installed": {
            "client_id": pick("YT_CLIENT_ID"),
            "client_secret": pick("YT_CLIENT_SECRET"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    },
    scopes=["https://www.googleapis.com/auth/youtube.upload"],
)
creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
print(f"\nREFRESH TOKEN for {NICHE}:\n{creds.refresh_token}")
print(f"\nSave it as the secret YT_REFRESH_TOKEN_{SUFFIX}")
print(f"  gh secret set YT_REFRESH_TOKEN_{SUFFIX} --repo codeaz-org/mpt")
if (os.environ.get(f"YT_CLIENT_ID_{SUFFIX}") or "").strip():
    print(f"  gh secret set YT_CLIENT_ID_{SUFFIX} --repo codeaz-org/mpt")
    print(f"  gh secret set YT_CLIENT_SECRET_{SUFFIX} --repo codeaz-org/mpt")
