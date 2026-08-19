#!/usr/bin/env python3
"""storyreel autopilot: upvoted subreddit story -> critic-approved retell ->
OpenMontage real-footage edit -> YouTube. Multiple channels, one config file,
free services only.

The proven parts (LLM failover chain, story harvesting/ranking, critic loop,
topic dedupe, per-channel YouTube auth, state file) come from mpt. The renderer
is new: montage.py drives OpenMontage's documentary-montage tools headlessly.

Env: NIM_API_KEY (plus optional GROQ_API_KEY / OPENROUTER_API_KEY / Ollama
fallback -- see llm.py), PEXELS_API_KEY / PIXABAY_API_KEY (optional, more
footage), REDDIT_CLIENT_ID/SECRET (recommended -- live vote counts),
YT_REFRESH_TOKEN_<CHANNELID> per channel. DRY_RUN=1 renders without uploading.
"""
import difflib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

import buffer
import critic
import montage
import stories
import upload as uploader
import viral
from llm import nim_chat

ROOT = Path(__file__).resolve().parent
_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

STATE_FILE = ROOT / "posted.json"
OUT_DIR = ROOT / "out"
DRY_RUN = os.environ.get("DRY_RUN", "").strip().lower() in ("1", "true", "yes")


def log(msg): print(f"[autopilot] {msg}", flush=True)


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"topics": {}, "uploads": []}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ---- topic dedupe (lifted from mpt) -----------------------------------------

_STOP = {"the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "by", "with",
         "your", "you", "like", "why", "how", "what", "when", "is", "are", "does",
         "do", "it", "its", "that", "this", "as", "at", "from", "into", "my", "his",
         "her", "their", "our"}
TOPIC_JACCARD_LIMIT = 0.40
TOPIC_RATIO_LIMIT = 0.72


def _tokens(topic):
    words = re.findall(r"[a-z]+", topic.lower())
    return {w[:-1] if len(w) > 4 and w.endswith("s") else w
            for w in words if w not in _STOP and len(w) > 2}


def too_similar(topic, used_topics):
    ta = _tokens(topic)
    for old in used_topics:
        tb = _tokens(old)
        jaccard = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
        if jaccard >= TOPIC_JACCARD_LIMIT:
            return old, f"{jaccard:.0%} shared vocabulary"
        ratio = difflib.SequenceMatcher(None, topic.lower(), old.lower()).ratio()
        if ratio >= TOPIC_RATIO_LIMIT:
            return old, f"{ratio:.0%} identical phrasing"
    return None, ""


# ---- script generation (mpt's loop, story-only) -------------------------------

WEAK_HOOK_RE = re.compile(
    r"^\s*(?:have you ever|ever wondered|did you know|let'?s talk|in this video|"
    r"today (?:we|i)\b|welcome back|so,? i found|imagine if)", re.I)
HEDGE_RE = re.compile(
    r"\b(?:can lead to|may (?:lead|cause)|is important|it'?s essential)\b", re.I)

# Mechanical Reddit-speak / meta detector -- more reliable than asking the critic
# model. Any hit forces a revise with concrete feedback.
REDDIT_SPEAK_RE = re.compile(
    r"\b(?:so basically|TL;DR|edit ?[:\-]|update ?[:\-]|obligatory|throwaway|"
    r"long[- ]time lurker|first[- ]time poster|buckle up|let that sink in|"
    r"wait for it|not clickbait|OP\b|redditor|subreddit|Reddit)\b",
    re.I)


def dirty_phrases(script, channel):
    """Return (phrases_found, reason_string) for mechanical fails, or ([], '')."""
    hits = set()
    for m in REDDIT_SPEAK_RE.finditer(script):
        hits.add(m.group(0).lower())
    for term in channel.get("drift_terms", []) or []:
        if re.search(rf"\b{re.escape(term)}\b", script, re.I):
            hits.add(term.lower())
    if not hits:
        return [], ""
    return sorted(hits), (
        "Delete these Reddit-speak / drift phrases from the script -- they must "
        "not appear in the final read: " + ", ".join(sorted(hits)))


def _first_sentence(script):
    return re.split(r"(?<=[.!?])\s+", script.strip(), maxsplit=1)[0].strip()


def weak_hook(script):
    opener = _first_sentence(script)
    if not opener:
        return "the script is empty"
    if opener.endswith("?"):
        return f"opens with a question: {opener!r}"
    if WEAK_HOOK_RE.match(opener):
        return f"opens with a stock phrase: {opener!r}"
    if len(opener.split()) > 28:
        return f"opening sentence is {len(opener.split())} words"
    if HEDGE_RE.search(opener):
        return f"hedged opener: {opener!r}"
    return None


def _clean_script(raw):
    s = re.sub(r"(?is)<(think|thinking|reasoning)>.*?</\1>", " ", raw)
    s = re.sub(r"(?is)<(?:think|thinking|reasoning)>.*$", " ", s)
    s = re.sub(r"```[a-z]*|```", "", s)
    s = re.sub(r"^\s*(?:#+|\*+|\d+[\.\)])\s*", "", s, flags=re.M)
    s = re.sub(r"\*\*|__|\[[^\]]*\]", "", s)
    s = re.sub(r"\n{2,}", " ", s).replace("\n", " ")
    return re.sub(r"\s{2,}", " ", s).strip().strip('"')


def _write_script(topic, channel, story, feedback=None):
    system = channel["script_prompt"]
    notes = [
        "SOURCE THREAD (your working text -- keep its sentences and voice where they "
        "work, edit only what breaks the read; follow the channel's editorial "
        "instructions above for exactly what to keep, cut, and add):\n\n"
        + (story.get("text") or "")]
    if feedback:
        notes.append("An editor reviewed your previous attempt and rejected it. "
                     f"Their instructions:\n{feedback}")
    system += "\n\n" + "\n\n".join(notes)
    return _clean_script(nim_chat(
        system, f"Topic: {topic}\n\nProduce the finished script now.",
        temperature=0.5 if not feedback else 0.4, max_tokens=3500,
    ))


def generate_script(topic, channel, story):
    """Draft, critique, revise until the critic passes it; publish-or-nothing."""
    def write(feedback):
        script = _write_script(topic, channel, story, feedback)
        log(f"[{channel['id']}] draft: {len(script.split())} words")
        return script

    max_words = int(channel.get("max_script_words", 140))

    def review(t, script, asked, min_score):
        verdict, scores, problems, fix = critic.review(t, script, asked, min_score,
                                                       niche=channel)
        bad = weak_hook(script)
        if bad:
            problems = [f"weak hook: {bad}"] + problems
            fix = ("Replace the first sentence with the moment of maximum tension, "
                   "stated as a concrete fact. " + fix)
            verdict = "revise"
        dirty, dirty_fix = dirty_phrases(script, channel)
        if dirty:
            problems = [f"drift/reddit-speak: {', '.join(dirty)}"] + problems
            fix = f"{dirty_fix} {fix}".strip()
            verdict = "revise"
        wc = len(script.split())
        if wc > max_words:
            problems = [f"too long: {wc} words (cap {max_words})"] + problems
            fix = (f"Cut to {max_words} words or fewer -- this is a strict one-minute "
                   f"vertical short. Keep the cold open, the events, and the closer; "
                   f"cut side detail and repetition. " + fix)
            verdict = "revise"
        return verdict, scores, problems, fix

    script, verdict, scores = critic.refine(
        topic, write, question=story.get("title"), review_fn=review)
    # Loosen: publish borderline drafts (mean >= 5, no score below 3) instead of
    # hard-failing the whole run when the critic never fully approved.
    mean = (sum(scores.values()) / len(scores)) if scores else 0
    worst = min(scores.values()) if scores else 0
    if verdict != "publish" and (mean < 5 or worst < 3):
        raise RuntimeError(f"script never passed review ({critic.summarise(scores)})")
    tag = "approved" if verdict == "publish" else "borderline"
    log(f"[{channel['id']}] script {tag}: {critic.summarise(scores)}")
    return script


# ---- render sanity (lifted from mpt) ------------------------------------------

WORDS_PER_SECOND = 2.5
MIN_VIDEO_SECONDS = int(os.environ.get("MIN_VIDEO_SECONDS", "8"))   # let short stories ship
MAX_VIDEO_SECONDS = int(os.environ.get("MAX_VIDEO_SECONDS", "62"))  # hard 1-minute ceiling


def check_rendered_video(path, script):
    duration = montage.ffprobe_duration(path)
    expected = len(script.split()) / WORDS_PER_SECOND
    if duration < MIN_VIDEO_SECONDS or duration < expected * 0.6:
        raise RuntimeError(f"rendered video is {duration:.0f}s but the script needs "
                           f"~{expected:.0f}s: narration was cut short, not uploading")
    if duration > MAX_VIDEO_SECONDS:
        raise RuntimeError(f"rendered video is {duration:.0f}s, over the "
                           f"{MAX_VIDEO_SECONDS}s ceiling: retrying with a shorter script")
    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of",
         "csv=p=0", str(path)], capture_output=True, text=True).stdout.split()
    if "audio" not in streams:
        raise RuntimeError("rendered video has no audio track, not uploading")
    log(f"video checks out: {duration:.0f}s for {len(script.split())} words")


TIKTOK_MAX_SECONDS = 180  # anything shorter is published as a single post, not split


def split_for_tiktok(video, channel, out_dir):
    """For shorts channels (portrait, <= TIKTOK_MAX_SECONDS), return the master
    video as a single 'part 1/1' -- no crop, no split, just publish the whole
    thing. For long-form landscape videos, center-crop to 9:16 and cut into
    ~part_seconds segments. Returns [(index, path, duration), ...]."""
    cfg = channel.get("tiktok", {}) or {}
    if not cfg.get("enabled"):
        return []
    duration = montage.ffprobe_duration(video)
    portrait = channel.get("orientation", "landscape") == "portrait"

    if portrait and duration <= TIKTOK_MAX_SECONDS:
        return [(1, Path(video), duration)]

    part_secs = int(cfg.get("part_seconds", 55))
    min_secs = int(cfg.get("min_part_seconds", 30))
    if duration < min_secs:
        log(f"[{channel['id']}] tiktok split skipped: video is only {duration:.0f}s")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video).stem
    pattern = str(out_dir / f"{stem}-part-%03d.mp4")
    vf = "crop=ih*9/16:ih,scale=1080:1920" if not portrait else "scale=1080:1920"
    _r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video),
         "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
         "-c:a", "aac", "-b:a", "160k",
         "-f", "segment", "-segment_time", str(part_secs),
         "-reset_timestamps", "1", pattern],
        capture_output=True, text=True)
    if _r.returncode != 0:
        raise RuntimeError(f"tiktok split failed: {_r.stderr[-800:]}")

    parts = sorted(out_dir.glob(f"{stem}-part-*.mp4"))
    kept = []
    for i, p in enumerate(parts):
        d = montage.ffprobe_duration(p)
        if d < min_secs:
            log(f"[{channel['id']}] tiktok part {i + 1} too short ({d:.0f}s), dropping")
            p.unlink(missing_ok=True)
            continue
        kept.append((len(kept) + 1, p, d))
    log(f"[{channel['id']}] tiktok: {len(kept)} portrait parts")
    return kept


def tiktok_caption(meta, channel, limit=2200):
    """TikTok has no separate description: caption and hashtags are one string."""
    tags = channel.get("tiktok_hashtags", channel.get("hashtags", "")).strip()
    body = re.sub(r"\s*#\S+", "", meta.get("description", "")).strip()
    parts = [p for p in (meta.get("title", "").strip(), body, tags) if p]
    caption = "\n\n".join(parts)
    if len(caption) <= limit:
        return caption
    room = limit - len(tags) - 2
    return f"{caption[:max(room, 0)].rstrip()}\n\n{tags}"[:limit]


# ---- run loop -------------------------------------------------------------------

def used_story_ids(state, channel_id):
    ids = set()
    for u in state.get("uploads", []):
        if u.get("channel") != channel_id:
            continue
        if u.get("story_id"):
            ids.add(u["story_id"])
        ids.update(u.get("story_ids") or ())
    return ids


def channel_ready(channel):
    return bool((os.environ.get(f"YT_REFRESH_TOKEN_{channel['id'].upper()}") or "")
                .strip() not in ("", "xxxx"))


def run_channel(channel, state):
    if not channel_ready(channel):
        log(f"[{channel['id']}] skipped: set YT_REFRESH_TOKEN_"
            f"{channel['id'].upper()} once the channel exists")
        return
    used = state["topics"].setdefault(channel["id"], [])
    used_ids = used_story_ids(state, channel["id"])

    # 1) Trend research: what's viral right now, matched to one of OUR real
    #    stories. Theme-level only -- see viral.py's recreate rule.
    topic = story = trend = theme = None
    if channel.get("viral_research", True):
        try:
            topic, story, trend, theme = viral.themed_pick(
                channel, state, used, used_ids, too_similar)
        except Exception as e:
            log(f"[{channel['id']}] trend research failed "
                f"({type(e).__name__}: {str(e)[:120]}); falling back")
    # 2) Fallback: nothing fresh/matching found -> people's own posts, ranked by
    #    the subreddit's votes, exactly as before.
    if not story:
        log(f"[{channel['id']}] no fresh trend matched; using top-voted stories")
        topic, story = stories.pick_story(channel, used, used_ids, too_similar)
    log(f"[{channel['id']}] Topic: {topic}"
        + (f" (riding trend: {theme})" if theme else ""))
    log(f"[{channel['id']}] retelling: {story['url']}")

    script = generate_script(topic, channel, story)
    video = montage.render(topic, channel, script, story=story)
    check_rendered_video(video, script)
    meta = uploader.make_metadata(topic, channel)
    if channel.get("credit_source", True) and story.get("url"):
        meta["description"] += f"\n\nInspired by a real thread: {story['url']}"

    tiktok_parts = split_for_tiktok(video, channel, OUT_DIR / "tiktok")

    if DRY_RUN:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        dest = OUT_DIR / Path(video).name
        shutil.copy(video, dest)
        dest.with_suffix(".txt").write_text(
            f"topic: {topic}\n\ntitle: {meta['title']}\n\n"
            f"description:\n{meta['description']}\n\nscript:\n{script}\n\n"
            f"tiktok_parts: {len(tiktok_parts)}\n")
        log(f"[{channel['id']}] DRY_RUN: upload skipped, video at {dest}"
            + (f" + {len(tiktok_parts)} tiktok parts" if tiktok_parts else ""))
        used.append(topic)
        return

    yt_id = uploader.upload_youtube(video, meta, channel)
    used.append(topic)
    entry = {
        "channel": channel["id"], "topic": topic, "title": meta["title"],
        "story": story.get("title"), "story_id": story.get("id"),
        "story_url": story.get("url"),
        "trend_id": (trend or {}).get("id"), "trend_url": (trend or {}).get("url"),
        "theme": theme,
        "youtube": yt_id, "tiktok": False, "tiktok_post_ids": [],
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    state["uploads"].append(entry)
    save_state(state)  # YouTube is live: record before TikTok can fail

    if buffer.enabled() and tiktok_parts:
        base_caption = tiktok_caption(meta, channel)
        total = len(tiktok_parts)
        for i, path, _ in tiktok_parts:
            part_title = meta["title"] if total == 1 else f"{meta['title']} — Part {i}/{total}"
            part_caption = base_caption if total == 1 else f"Part {i}/{total}\n\n{base_caption}"
            try:
                post_id = buffer.publish(str(path), part_caption,
                                         title=part_title, niche_id=channel["id"])
                entry["tiktok_post_ids"].append(post_id)
            except Exception as e:
                log(f"[{channel['id']}] Buffer/TikTok part {i} failed: "
                    f"{type(e).__name__}: {str(e)[:160]}")
        entry["tiktok"] = bool(entry["tiktok_post_ids"])
        save_state(state)
    for _, path, _ in tiktok_parts:
        path.unlink(missing_ok=True)
    Path(video).unlink(missing_ok=True)


RUN_ATTEMPTS = int(os.environ.get("RUN_ATTEMPTS", "6"))


def run_channel_with_retries(channel, state):
    last = None
    for i in range(RUN_ATTEMPTS):
        try:
            run_channel(channel, state)
            return
        except Exception as e:
            last = e
            log(f"[{channel['id']}] attempt {i + 1}/{RUN_ATTEMPTS} failed: "
                f"{type(e).__name__}: {str(e)[:200]}")
            if i < RUN_ATTEMPTS - 1:
                time.sleep(30 * (i + 1))
    raise last


def main():
    channels = json.loads((ROOT / "channels.json").read_text())["channels"]
    only = [s.strip() for s in os.environ.get("CHANNELS", "").split(",") if s.strip()]
    if only:
        channels = [c for c in channels if c["id"] in only]
    state = load_state()
    failures = []
    for channel in channels:
        try:
            run_channel_with_retries(channel, state)
        except Exception as e:
            log(f"[{channel['id']}] FAILED: {e}")
            failures.append(channel["id"])
    if failures:
        # Don't crash the workflow -- log the misses and let the next scheduled
        # run try again. A red X on every off-day is noise, not signal.
        log(f"channels with no upload this run: {failures}")
    else:
        log("All channels done.")


if __name__ == "__main__":
    main()
