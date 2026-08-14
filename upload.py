"""YouTube publishing, lifted from mpt and trimmed to what storyreel needs.

One refresh token per channel (YT_REFRESH_TOKEN_<CHANNELID>), optionally one OAuth
client per channel (YT_CLIENT_ID_<CHANNELID>/YT_CLIENT_SECRET_<CHANNELID>) since
YouTube's ~6-uploads/day quota is per Google Cloud project, not per channel.
"""
import json, os, re

from llm import nim_chat


def log(msg): print(f"[upload] {msg}", flush=True)


def make_metadata(topic, channel):
    """Never let a metadata hiccup discard a rendered video -- fall back to the topic."""
    try:
        raw = nim_chat(
            "You write YouTube metadata for a narrated story video. Respond ONLY with "
            'JSON: {"title": "...", "description": "..."}. Title under 80 chars, one '
            "dread-hook line, no ALL CAPS, no clickbait cliches. Description: ONE short "
            "sentence (under 140 chars) that teases the fear without spoiling the turn.",
            f"Video topic: {topic}",
            temperature=0.7,
        )
        meta = json.loads(re.sub(r"```json|```", "", raw).strip())
    except Exception as e:
        log(f"metadata generation failed ({type(e).__name__}), using the topic as-is")
        meta = {"title": topic, "description": topic}
    meta["description"] = f'{meta.get("description", topic)}\n\n{channel["hashtags"]}'
    meta["title"] = meta.get("title", topic)[:95]
    return meta


def credentials(channel_id):
    suffix = channel_id.upper()

    def pick(name):
        value = (os.environ.get(f"{name}_{suffix}") or "").strip()
        return value or (os.environ.get(name) or "").strip()

    refresh = (os.environ.get(f"YT_REFRESH_TOKEN_{suffix}") or "").strip()
    return pick("YT_CLIENT_ID"), pick("YT_CLIENT_SECRET"), (
        "" if refresh.lower() == "xxxx" else refresh)


def upload_youtube(video_path, meta, channel):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    client_id, client_secret, refresh_token = credentials(channel["id"])
    if not refresh_token:
        log(f"No YT_REFRESH_TOKEN_{channel['id'].upper()} set; skipping upload")
        return None
    creds = Credentials(
        None, refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id, client_secret=client_secret,
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    yt = build("youtube", "v3", credentials=creds)
    body = {
        "snippet": {
            "title": meta["title"],
            "description": meta["description"],
            "tags": channel.get("youtube_tags", []),
            "categoryId": str(channel.get("youtube_category", "24")),
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
            # Model-written narration, synthetic voice: YouTube's altered-media
            # disclosure applies, and the label costs nothing.
            "containsSyntheticMedia": True,
        },
    }
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True,
                            mimetype="video/mp4")
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    log(f"YouTube uploaded: https://youtu.be/{resp['id']}")
    return resp["id"]
