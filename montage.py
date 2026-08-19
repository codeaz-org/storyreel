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

# One slot per ~20 words = one clip roughly every 8s at spoken pace.
# Sized for the 60s one-story short: yields ~7 cuts across a 140-word script.
WORDS_PER_SLOT = 20


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
    """Narrate and capture real per-word timestamps from edge-tts. WordBoundary
    events tell us exactly when each word is spoken, so captions never drift.

    Rate is NOT passed as a prosody modifier: the <prosody rate=...> SSML wrapper
    suppresses WordBoundary events in current edge-tts / Azure combinations, and
    zero timings means unsynced captions. Natural rate keeps the events flowing."""
    import edge_tts
    short = re.sub(r"-(Male|Female)$", "", voice)
    words = []

    async def go():
        # edge-tts 7.x defaults to SentenceBoundary; must opt in for per-word events.
        comm = edge_tts.Communicate(script, short, boundary="WordBoundary")
        with open(out_path, "wb") as f:
            async for chunk in comm.stream():
                ct = chunk.get("type")
                if ct == "audio":
                    f.write(chunk["data"])
                elif ct == "WordBoundary":
                    words.append({
                        "text": chunk.get("text") or "",
                        "start": (chunk.get("offset") or 0) / 1e7,
                        "duration": (chunk.get("duration") or 0) / 1e7,
                    })
    asyncio.run(go())
    if not out_path.exists() or out_path.stat().st_size < 1000:
        raise RuntimeError(f"edge-tts produced no audio for {short}")
    log(f"narration -> {out_path.name} ({out_path.stat().st_size // 1024} KB, "
        f"{short}, {len(words)} word timings)")
    return words


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

def _ass_ts(seconds):
    """ASS time format: H:MM:SS.cc (centiseconds)."""
    cs = int(round(seconds * 100))
    return f"{cs//360000}:{cs//6000%60:02d}:{cs//100%60:02d}.{cs%100:02d}"


CAPTION_MAX_WORDS = 1        # one spoken word per cue, timed to edge-tts WordBoundary
CAPTION_PAUSE_BREAK = 0.30   # gap between spoken words that ends a caption chunk early
CAPTION_TAIL_HOLD = 0.05     # tiny hold past the last word so it doesn't blink out
FALLBACK_LEAD = 0.35         # rough TTS initial silence for the no-boundaries fallback


def _chunk_timings(word_timings, max_words=CAPTION_MAX_WORDS):
    """[(start, end, text), ...] from real edge-tts per-word timestamps."""
    out, i = [], 0
    while i < len(word_timings):
        end_i = min(i + max_words, len(word_timings))
        for j in range(i + 1, end_i):
            prev = word_timings[j - 1]
            gap = word_timings[j]["start"] - (prev["start"] + prev["duration"])
            if gap > CAPTION_PAUSE_BREAK and j - i >= 4:
                end_i = j
                break
        chunk = word_timings[i:end_i]
        text = " ".join(w["text"] for w in chunk).strip()
        start = chunk[0]["start"]
        end = chunk[-1]["start"] + chunk[-1]["duration"] + CAPTION_TAIL_HOLD
        if end_i < len(word_timings):  # never overlap the next cue's start
            end = min(end, word_timings[end_i]["start"] - 0.01)
        out.append((start, end, text))
        i = end_i
    return out


def _fallback_chunks(script, audio_secs, max_words=CAPTION_MAX_WORDS):
    """No WordBoundary events -> sentence-proportional split so we always ship
    something usable. Bounded drift (per sentence, not accumulated)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", script) if s.strip()]
    if not sentences or audio_secs <= 0:
        return [(0.0, 1.0, " ")]
    total_words = sum(len(s.split()) for s in sentences) or 1
    speech = max(audio_secs - FALLBACK_LEAD, 0.5)
    t, out = FALLBACK_LEAD, []
    for sent in sentences:
        sent_words = sent.split()
        sent_dur = speech * len(sent_words) / total_words
        chunks = ([sent] if len(sent_words) <= max_words else
                  [" ".join(sent_words[i:i + max_words])
                   for i in range(0, len(sent_words), max_words)])
        for c in chunks:
            cd = sent_dur * len(c.split()) / len(sent_words)
            out.append((t, t + cd, c))
            t += cd
    return out


# ASS_COLOUR_KEYS: caption_style key -> ASS style field. All colours are ASS
# format (&HAABBGGRR); alpha 00 = opaque, 80 = 50% translucent.
_DEFAULTS = {
    "font": "DejaVu Sans", "size": 60, "colour": "&H00FFFFFF",
    "outline_colour": "&H00000000", "back_colour": "&H80000000",
    "outline": 4, "shadow": 2, "alignment": 5, "margin_v": 0,
}


def write_ass(word_timings, path, channel, script="", audio_secs=0.0,
              max_words=CAPTION_MAX_WORDS):
    """Emit an ASS file with PlayRes matched to the output frame, so FontSize
    renders as real pixels and Alignment=5 truly centers. Skips SRT entirely --
    libass's SRT->ASS auto-conversion uses a 384x288 PlayRes that mangled our
    styling before."""
    t = {**_DEFAULTS, **(channel.get("caption_style") or {})}
    pw, ph = (1080, 1920) if channel.get("orientation") == "portrait" else (1920, 1080)
    chunks = (_chunk_timings(word_timings, max_words) if word_timings
              else _fallback_chunks(script, audio_secs, max_words))
    if not word_timings:
        log(f"no word timings; sentence-proportional fallback over {audio_secs:.1f}s")
    style = (
        f"Style: Default,{t['font']},{t['size']},"
        f"{t['colour']},{t['colour']},{t['outline_colour']},{t['back_colour']},"
        f"1,0,0,0,100,100,0,0,1,{t['outline']},{t['shadow']},"
        f"{t['alignment']},40,40,{t['margin_v']},1"
    )
    events = "\n".join(
        f"Dialogue: 0,{_ass_ts(s)},{_ass_ts(e)},Default,,0,0,0,,{txt}"
        for s, e, txt in chunks
    )
    path.write_text(
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {pw}\nPlayResY: {ph}\n"
        f"ScaledBorderAndShadow: yes\nWrapStyle: 2\n\n"
        f"[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
        f"OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,"
        f"Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,"
        f"Encoding\n{style}\n\n"
        f"[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
        f"{events}\n"
    )


def _brand_overlay(channel):
    """drawtext filter for a persistent corner wordmark. Reads brand.wordmark_colour
    (ffmpeg colour syntax) and brand.wordmark_size so each channel can dial its own
    on-brand look. Empty string if the channel declares no brand."""
    brand = channel.get("brand") or {}
    wordmark = (brand.get("wordmark") or "").strip()
    if not wordmark:
        return ""
    colour = brand.get("wordmark_colour", "white@0.55")
    size = brand.get("wordmark_size", 28)
    return (f",drawtext=text='{wordmark}':fontcolor={colour}:fontsize={size}:"
            f"box=0:borderw=2:bordercolor=black@0.6:"
            f"x=w-tw-32:y=48")


def burn_and_mux(video, narration, ass, out_path, channel, total_secs):
    ass_arg = str(ass).replace("'", r"\'")
    look = (channel.get("look") or "").strip()
    vf = ((f"{look}," if look else "")
          + f"ass='{ass_arg}'"
          + _brand_overlay(channel))
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
    word_timings = tts(script, channel["voice"], narration)
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

    ass = project / "captions.ass"
    write_ass(word_timings, ass, channel, script=script, audio_secs=audio_secs)
    final = project / f"{channel['id']}-{ts}-final.mp4"
    burn_and_mux(stitched, narration, ass, final, channel, audio_secs)
    log(f"6/6 final      -> {final.name} ({ffprobe_duration(final):.0f}s, {w}x{h})")

    for f in fitted + [stitched]:
        f.unlink(missing_ok=True)  # keep the runner's disk sane
    return str(final)
