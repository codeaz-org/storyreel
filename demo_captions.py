"""One-shot local preview that looks like production.

Fetches one Pexels clip that matches the channel's fallback visual, applies the
channel's colour grade + brand wordmark, burns per-word captions (edge-tts
WordBoundary events -> exact timings), muxes narration. Same code path as
prod's final mp4, just one clip instead of the corpus. Opens the mp4.

    python demo_captions.py            # default line
    python demo_captions.py "your own script here"
"""
import json, os, subprocess, sys
from pathlib import Path

import requests

_env = Path(__file__).with_name(".env")
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import montage

SAMPLE = ("The thing in the hallway had my sister's voice. "
          "It knew the name only she and I ever used.")


def fetch_pexels_clip(query, dest, portrait=True):
    """Grab one Pexels video that matches the query. Portrait if requested."""
    key = os.environ.get("PEXELS_API_KEY", "").strip()
    if not key or key.lower() == "xxxx":
        sys.exit("PEXELS_API_KEY missing from .env -- can't fetch demo footage")
    r = requests.get("https://api.pexels.com/videos/search",
        params={"query": query, "per_page": 5,
                "orientation": "portrait" if portrait else "landscape"},
        headers={"Authorization": key}, timeout=30)
    r.raise_for_status()
    videos = r.json().get("videos") or []
    if not videos: sys.exit(f"pexels returned no videos for {query!r}")
    files = [f for f in videos[0]["video_files"] if f.get("width", 0) >= 720]
    files.sort(key=lambda f: f.get("width", 0))
    url = (files or videos[0]["video_files"])[0]["link"]
    with requests.get(url, stream=True, timeout=60) as g:
        g.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in g.iter_content(1 << 16): fh.write(chunk)

def main():
    script = " ".join(sys.argv[1:]) or SAMPLE
    channel = json.loads(Path("channels.json").read_text())["channels"][0]

    out = Path("work/demo"); out.mkdir(parents=True, exist_ok=True)
    narration, ass, raw, fitted, final = (out / n for n in
        ("narration.mp3", "captions.ass", "pexels.mp4", "bg.mp4", "demo.mp4"))

    if b"subtitles" not in subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"], capture_output=True).stdout:
        sys.exit("this ffmpeg lacks libass -- brew install ffmpeg-full")

    words = montage.tts(script, channel["voice"], narration)
    secs = montage.ffprobe_duration(narration)
    montage.write_ass(words, ass, channel, script=script, audio_secs=secs)

    print(f"\n{len(words)} word timings from edge-tts:")
    for w in words[:8]:
        print(f"  {w['start']:6.3f}s  +{w['duration']:.3f}s  {w['text']!r}")
    if len(words) > 8: print(f"  ... {len(words) - 8} more")

    portrait = channel.get("orientation", "landscape") == "portrait"
    w, h = (1080, 1920) if portrait else (1920, 1080)
    query = channel.get("fallback_visual") or "dark cinematic scene"
    print(f"\nfetching one Pexels clip for {query!r}...")
    fetch_pexels_clip(query, raw, portrait=portrait)
    montage.fit_clip(raw, fitted, secs + 0.5, w, h)
    montage.burn_and_mux(fitted, narration, ass, final, channel, secs)

    print(f"\nASS -> {ass}\nMP4 -> {final}")
    subprocess.run(["open", str(final)])

if __name__ == "__main__":
    main()
