"""Deterministic driver for OpenMontage's documentary-montage tools.

OpenMontage is agent-first: interactively, a coding agent reads its manifests and
skills and improvises the production. That cannot run unattended on a CI runner, so
this module hard-codes ONE recipe -- the story-retell montage -- and calls the same
Python tools the agent would, in a fixed order, with our critic-approved script as
the creative input and no human gates:

  1. scene_plan       LLM splits the narration into 6-10 visual slots
  2. tts              edge-tts narration -> mp3 (per-channel brand voice)
  3. build_corpus     OpenMontage corpus_builder: Pexels/Pixabay (keys) +
                      Archive.org/Wikimedia/NASA (keyless) -> CLIP-indexed corpus
  4. pick clips       OpenMontage clip_search rank_for_slot per scene
  5. fit + stitch     trim/loop each clip to its slot, OpenMontage video_stitch
  6. captions + mux   sentence-timed SRT burned in, narration muxed, ffprobe-checked

The corpus directory persists across runs (Actions cache), so retrieval gets richer
and cheaper over time instead of re-downloading the same clips.

Requires the OpenMontage checkout on disk (OPENMONTAGE_DIR, default ./OpenMontage)
with its deps installed: numpy, Pillow, opencv-python-headless, torch (CPU),
transformers. First run downloads CLIP ViT-B/32 (~350 MB) into ~/.cache/huggingface.
"""
import asyncio, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

from llm import nim_json

ROOT = Path(__file__).resolve().parent
OM_DIR = Path(os.environ.get("OPENMONTAGE_DIR", ROOT / "OpenMontage"))
WORK = ROOT / "work"
CORPUS_ROOT = ROOT / "corpus"

FPS = 30
# Keyless sources always on; keyed ones join automatically when their env key is set
# (OpenMontage's adapters check the same PEXELS_API_KEY/PIXABAY_API_KEY names).
KEYLESS_SOURCES = ["archive_org", "wikimedia", "nasa"]
KEYED_SOURCES = {"pexels": "PEXELS_API_KEY", "pixabay_video": "PIXABAY_API_KEY"}
MIN_CLIP_SCORE = 0.22   # CLIP ViT-B/32: ~0.25 is a decent match; below 0.22 is noise


def log(msg): print(f"[montage] {msg}", flush=True)


def _import_om():
    """Put the OpenMontage checkout on sys.path. Lazy so importing this module
    doesn't require torch -- only render() does."""
    if not OM_DIR.exists():
        raise RuntimeError(f"OpenMontage checkout not found at {OM_DIR}; "
                           "set OPENMONTAGE_DIR or clone it there")
    if str(OM_DIR) not in sys.path:
        sys.path.insert(0, str(OM_DIR))


def _run(cmd, why):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{why} failed:\n{r.stderr[-1500:]}")
    return r


def ffprobe_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


# ---- 1. scene plan -----------------------------------------------------------

SCENE_SYSTEM = """You are an editor breaking a narrated story into visual slots for a
real-footage montage. Split the script into consecutive slots, aiming for roughly
one slot every {words_per_slot} words. For each slot:
  - "text": the EXACT next contiguous chunk of the script, verbatim, in order.
    Every word of the script must appear in exactly one slot. Do not rewrite.
  - "visual": a 3-8 word description of concrete, filmable stock/archive footage
    that fits the mood of that chunk. Places, objects, weather, hands, streets --
    never people's readable faces, never text on screen, never brand logos.
Slots should break at the story's beats. Return JSON only."""

# One slot per ~55 words = one clip roughly every 20-22s at spoken pace.
WORDS_PER_SLOT = 55


def scene_plan(script, channel):
    """[{text, visual}] covering the whole script, or a mechanical fallback."""
    word_count = len(script.split())
    target_slots = max(6, min(50, word_count // WORDS_PER_SLOT))
    try:
        result = nim_json(
            SCENE_SYSTEM.format(words_per_slot=WORDS_PER_SLOT)
            + ' JSON schema: {"slots": [{"text": "...", "visual": "..."}]}',
            f"Style keywords for this channel: {channel.get('style_suffix', '')}\n\n"
            f"Script (aim for ~{target_slots} slots):\n{script}",
            max_tokens=6000, temperature=0.3,
        )
        slots = [s for s in (result.get("slots") or [])
                 if isinstance(s, dict) and s.get("text") and s.get("visual")]
        planned = " ".join(s["text"] for s in slots)
        lo, hi = max(5, target_slots // 2), min(60, target_slots * 2)
        if lo <= len(slots) <= hi and \
                abs(len(planned.split()) - word_count) <= word_count * 0.12:
            return slots
        log(f"scene plan rejected ({len(slots)} slots, "
            f"{len(planned.split())}/{word_count} words); using fallback")
    except Exception as e:
        log(f"scene plan failed ({type(e).__name__}); using fallback")
    return _mechanical_plan(script, channel, target_slots)


def _mechanical_plan(script, channel, target_slots=8):
    """Even sentence-group split with the channel's default footage look."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", script) if s.strip()]
    n = max(1, min(target_slots, len(sentences)))
    per = max(1, len(sentences) // n)
    visual = (channel.get("fallback_visual")
              or "moody city street at dusk, cinematic")
    slots = []
    for i in range(0, len(sentences), per):
        slots.append({"text": " ".join(sentences[i:i + per]), "visual": visual})
    return slots


# ---- 2. narration ------------------------------------------------------------

def tts(script, voice, out_path):
    import edge_tts
    short = re.sub(r"-(Male|Female)$", "", voice)

    async def go():
        await edge_tts.Communicate(script, short, rate="+4%").save(str(out_path))
    asyncio.run(go())
    if not out_path.exists() or out_path.stat().st_size < 1000:
        raise RuntimeError(f"edge-tts produced no audio for {short}")
    log(f"narration -> {out_path.name} ({out_path.stat().st_size // 1024} KB, {short})")


# ---- 3. corpus ---------------------------------------------------------------

def active_sources(channel):
    wanted = channel.get("footage_sources") or (
        KEYLESS_SOURCES + [s for s, env in KEYED_SOURCES.items()
                           if (os.environ.get(env) or "").strip().lower() not in ("", "xxxx")])
    return wanted


def build_corpus(channel, queries, corpus_dir, per_source=8, max_new=80):
    _import_om()
    from tools.video.corpus_builder import CorpusBuilder

    corpus_dir.mkdir(parents=True, exist_ok=True)
    orientation = channel.get("orientation", "landscape")
    payload = {
        "corpus_dir": str(corpus_dir),
        "queries": [{"query": q, "kind": "video", "per_source": per_source}
                    for q in queries],
        "sources": active_sources(channel),
        "filters": {"orientation": orientation, "min_duration": 4},
        "max_new_clips": max_new,
        "skip_existing": True,
    }
    res = CorpusBuilder().execute(payload)
    if not res.success:
        raise RuntimeError(f"corpus_builder: {res.error}")
    data = res.data or {}
    log(f"corpus: +{data.get('added', '?')} new clips "
        f"(size {data.get('corpus_size', '?')}, sources {payload['sources']})"
        + (f", errors: {len(data.get('errors', []))}" if data.get("errors") else ""))


def pick_clip(corpus_dir, query, exclude_ids, min_secs):
    """Best CLIP match for a slot: prefer real motion and enough runtime; a short
    clip is acceptable (it gets looped) but a weak match is not."""
    _import_om()
    from tools.video.clip_search import ClipSearch

    res = ClipSearch().execute({
        "operation": "rank_for_slot", "corpus_dir": str(corpus_dir),
        "query_text": query, "k": 6, "kind": "video",
        "motion_min": 0.8, "exclude_ids": list(exclude_ids),
    })
    if not res.success:
        raise RuntimeError(f"clip_search: {res.error}")
    results = (res.data or {}).get("results") or []
    for hit in results:
        rec, score = hit["record"], hit["score"]
        if score < MIN_CLIP_SCORE:
            break  # ordered best-first: everything after is worse
        if (rec.get("duration") or 0) >= min_secs * 0.5:
            return rec, score
    if results:  # nothing long enough scored well; take the best and loop it
        rec, score = results[0]["record"], results[0]["score"]
        if score >= MIN_CLIP_SCORE:
            return rec, score
    return None, 0.0


# ---- 4/5. fit + stitch --------------------------------------------------------

def fit_clip(src, dest, seconds, width, height):
    """Trim or loop a source clip to exactly `seconds`, cover-cropped to the
    channel's frame, silent (narration owns the audio track)."""
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=increase,"
          f"crop={width}:{height},fps={FPS}")
    _run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(src), "-t", f"{seconds:.3f}",
          "-an", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
          "-pix_fmt", "yuv420p", str(dest)], f"fit {Path(src).name}")


def stitch(clips, out_path, width, height):
    _import_om()
    from tools.video.video_stitch import VideoStitch

    res = VideoStitch().execute({
        "operation": "stitch", "clips": [str(c) for c in clips],
        "output_path": str(out_path), "transition": "cut",
        "auto_normalize": True, "target_resolution": f"{width}x{height}",
        "target_fps": FPS, "preset": "veryfast",
    })
    if not res.success:
        raise RuntimeError(f"video_stitch: {res.error}")


# ---- 6. captions + mux ---------------------------------------------------------

def _ts(seconds):
    ms = int(round(seconds * 1000))
    return f"{ms//3600000:02d}:{ms//60000%60:02d}:{ms//1000%60:02d},{ms%1000:03d}"


CAPTION_MAX_WORDS = 7  # short chunks so libass never wraps off-screen


def _chunk_sentence(sentence, max_words=CAPTION_MAX_WORDS):
    """Break a sentence into ~max_words chunks at natural pauses (commas, dashes)
    when possible, else on word boundaries. Prevents multi-line captions from
    overflowing the safe area."""
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    chunks, current = [], []
    for w in words:
        current.append(w)
        ends_pause = w.endswith((",", ";", "--", "—", ":"))
        if len(current) >= max_words or (ends_pause and len(current) >= 4):
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def write_srt(slots, slot_secs, path):
    """Chunked captions (<= CAPTION_MAX_WORDS per line), timed proportionally by
    word count inside each slot. Chunking keeps burned captions inside the safe
    area on every aspect ratio."""
    idx, t, lines = 1, 0.0, []
    for slot, secs in zip(slots, slot_secs):
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", slot["text"]) if s.strip()]
        chunks = [c for s in sentences for c in _chunk_sentence(s)]
        words_total = sum(len(c.split()) for c in chunks) or 1
        for c in chunks:
            d = max(0.7, secs * len(c.split()) / words_total)
            lines += [str(idx), f"{_ts(t)} --> {_ts(t + d)}", c, ""]
            idx += 1
            t += d
    path.write_text("\n".join(lines))


def _brand_overlay(channel):
    """drawtext filter for a corner wordmark, or empty string if the channel
    doesn't declare a brand. Kept small and semi-transparent so it never fights
    the story."""
    brand = channel.get("brand") or {}
    wordmark = (brand.get("wordmark") or "").strip()
    if not wordmark:
        return ""
    return (f",drawtext=text='{wordmark}':fontcolor=white@0.55:fontsize=28:"
            f"box=0:borderw=1:bordercolor=black@0.4:"
            f"x=w-tw-32:y=32")


def burn_and_mux(video, narration, srt, out_path, channel, total_secs):
    theme = channel.get("caption_style", {})
    style = (f"FontName={theme.get('font', 'DejaVu Sans')},"
             f"FontSize={theme.get('size', 22)},Bold=1,"
             f"PrimaryColour={theme.get('colour', '&H00FFFFFF')},"
             f"OutlineColour=&H00000000,Outline=3,Shadow=1,"
             f"Alignment=2,MarginV={theme.get('margin_v', 80)},"
             f"WrapStyle=0")
    srt_arg = str(srt).replace("'", r"\'")
    vf = f"subtitles='{srt_arg}':force_style='{style}'" + _brand_overlay(channel)
    _run(["ffmpeg", "-y", "-i", str(video), "-i", str(narration),
          "-vf", vf,
          "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast",
          "-crf", "22", "-c:a", "aac", "-b:a", "160k",
          "-t", f"{total_secs + 0.6:.2f}", str(out_path)], "final mux")


# ---- orchestrate ----------------------------------------------------------------

def render(topic, channel, script, story=None):
    """Full recipe. Returns the final mp4 path. Every step logs; any failure
    raises so the caller's retry loop owns recovery."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    project = WORK / f"{channel['id']}-{ts}"
    project.mkdir(parents=True, exist_ok=True)
    w, h = (1080, 1920) if channel.get("orientation", "landscape") == "portrait" \
        else (1920, 1080)

    slots = scene_plan(script, channel)
    log(f"1/6 scene plan -> {len(slots)} slots")

    narration = project / "narration.mp3"
    tts(script, channel["voice"], narration)
    audio_secs = ffprobe_duration(narration)
    total_words = sum(len(s["text"].split()) for s in slots) or 1
    slot_secs = [audio_secs * len(s["text"].split()) / total_words for s in slots]
    log(f"2/6 narration  -> {audio_secs:.0f}s across {len(slots)} slots")

    corpus_dir = CORPUS_ROOT / channel["id"]
    style = channel.get("style_suffix", "")
    queries = list(dict.fromkeys(f"{s['visual']}" for s in slots))
    build_corpus(channel, queries, corpus_dir)
    log("3/6 corpus     -> ready")

    fitted, used = [], set()
    for i, (slot, secs) in enumerate(zip(slots, slot_secs)):
        rec, score = pick_clip(corpus_dir, f"{slot['visual']}, {style}", used, secs)
        if rec is None:
            raise RuntimeError(f"no corpus clip matched slot {i + 1} "
                               f"({slot['visual']!r}); widen queries or sources")
        used.add(rec["clip_id"])
        src = corpus_dir / rec["local_path"]
        dest = project / f"slot-{i:02d}.mp4"
        fit_clip(src, dest, secs, w, h)
        fitted.append(dest)
        log(f"4/6 slot {i + 1}/{len(slots)} -> {rec['clip_id']} "
            f"(score {score:.2f}, {rec.get('source')})")

    stitched = project / "stitched.mp4"
    stitch(fitted, stitched, w, h)
    log(f"5/6 stitch     -> {stitched.name} ({ffprobe_duration(stitched):.0f}s)")

    srt = project / "captions.srt"
    write_srt(slots, slot_secs, srt)
    final = project / f"{channel['id']}-{ts}-final.mp4"
    burn_and_mux(stitched, narration, srt, final, channel, audio_secs)
    log(f"6/6 final      -> {final.name} ({ffprobe_duration(final):.0f}s, {w}x{h})")

    for f in fitted + [stitched]:
        f.unlink(missing_ok=True)  # keep the runner's disk sane
    return str(final)
