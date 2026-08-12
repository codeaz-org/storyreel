"""Trend research: what story videos are going viral right now, and which of OUR
harvested stories fits that theme.

The recreate rule, stated once and enforced in every prompt: we recreate the THEME
and FORMAT of a trending video, never its content. A trending video tells us "revenge
stories about landlords are pulling millions of views this week"; we then retell a
DIFFERENT real story -- one from our own subreddit harvest -- that lives in the same
theme, in our own brand and voice. Nothing is transcribed, nothing is paraphrased
from another creator, and the script still goes through the same critic gate as
every other video. Copying a creator's script would be a copyright and reuse-policy
problem; theme-level trend-following is just editorial judgment.

Discovery is keyless via yt-dlp search: `ytsearch` (relevance) plus `ytsearchdate`
(freshness) per channel query, ranked by view count. Every failure here is soft --
the caller falls back to plain top-voted story selection, which is also the path
when nothing new/fresh is found.
"""
import os, random, re, time

import stories
from llm import nim_json

MIN_VIEWS_DEFAULT = 50_000
PER_QUERY = 10
SHORTLIST = 15


def log(msg): print(f"[viral] {msg}", flush=True)


# ---- discovery ------------------------------------------------------------------

def fetch_trending(channel, per_query=PER_QUERY):
    """Candidate viral videos from yt-dlp search, no API key. Flat extraction only:
    titles, views and durations are enough to read a trend; we never download or
    transcribe anyone's video."""
    import yt_dlp

    queries = channel.get("viral_queries") or ["reddit story", "storytime"]
    random.shuffle(queries)
    lo, hi = channel.get("viral_duration_range", [15, 600])
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "extract_flat": "in_playlist"}
    seen, vids = set(), []
    with yt_dlp.YoutubeDL(opts) as ydl:
        for q in queries:
            # ponytail: only `ytsearch` -- `ytsearchdate` was throwing "Unsupported
            # url scheme" on current yt-dlp; relevance is the important ranking anyway.
            try:
                info = ydl.extract_info(f"ytsearch{per_query}:{q}", download=False)
            except Exception as e:
                log(f"search '{q}': {type(e).__name__}: {str(e)[:60]}")
                continue
            for e in (info or {}).get("entries") or []:
                if not e or not e.get("id") or e["id"] in seen:
                    continue
                seen.add(e["id"])
                dur = e.get("duration") or 0
                if dur and not lo <= dur <= hi:
                    continue
                vids.append({
                    "id": e["id"],
                    "title": (e.get("title") or "").strip(),
                    "url": f"https://www.youtube.com/watch?v={e['id']}",
                    "views": e.get("view_count") or 0,
                    "duration": dur,
                    "uploader": e.get("channel") or e.get("uploader") or "",
                    "query": q,
                })
    min_views = int(channel.get("viral_min_views", MIN_VIEWS_DEFAULT))
    vids = [v for v in vids if v["views"] >= min_views and v["title"]]
    vids.sort(key=lambda v: v["views"], reverse=True)
    log(f"{len(vids)} trending candidates >= {min_views:,} views")
    return vids


# ---- theme extraction --------------------------------------------------------------

THEME_SYSTEM = """You are a trend analyst for a story channel. You are given titles and
view counts of story videos currently pulling big numbers. Pick the ONE whose THEME
our channel should ride next, and describe that theme in YOUR OWN WORDS.

A theme is the underlying story shape, not the specific video: "tenant beats a
landlord's bogus damage claim with paperwork" is a theme; the video's title is not.

Rules:
  - Never copy or lightly rephrase the video's title. Describe the situation type.
  - Prefer themes that recur across several of the listed videos -- one outlier view
    count is luck, three similar titles is a trend.
  - The theme must be tellable through an everyday real story (no celebrities, no
    news events, nothing that requires that specific creator's footage or persona).

Return the index of the video that best evidences the trend, the theme (under 20
words), and why."""


def pick_theme(channel, videos, used_trend_ids, limit=SHORTLIST):
    fresh = [v for v in videos if v["id"] not in used_trend_ids][:limit]
    if len(fresh) < 3:
        log(f"only {len(fresh)} unseen trending videos; not enough to read a trend")
        return None, None
    listed = "\n".join(
        f"{i}. [{v['views']:,} views, {v['duration'] or '?'}s] {v['title']}"
        for i, v in enumerate(fresh))
    result = nim_json(
        THEME_SYSTEM + ' JSON schema: {"index": <number>, "theme": "...", "why": "..."}',
        f"Channel: {channel['name']}\n\nTrending now:\n{listed}",
        max_tokens=500,
    )
    index = result.get("index")
    theme = (result.get("theme") or "").strip()
    if not isinstance(index, int) or not 0 <= index < len(fresh) or not theme:
        raise RuntimeError(f"unusable theme selection: {str(result)[:150]}")
    trend = fresh[index]
    log(f"trend: {trend['title'][:70]} ({trend['views']:,} views)")
    log(f"  -> theme: {theme}")
    return trend, theme


# ---- theme -> our own real story ----------------------------------------------------

MATCH_SYSTEM = """You match a trending THEME to a real story from our own harvest. You are
given the theme and a list of real stories a subreddit already upvoted. Pick the ONE
story that genuinely lives in that theme AND has a clear arc (setup, escalation, turn,
landing). If none is a genuine fit, return index -1 -- a forced match makes a worse
video than no trend at all.

Then write a hook TITLE for the video: under 12 words, states the collision at the
heart of OUR story (not the trending video's) without spoiling the turn.

Return the index and the title."""


def match_story(channel, theme, used_topics, used_ids, too_similar, limit=20):
    """(topic, story) for the best theme-matching harvested story, or (None, None)
    when nothing genuinely fits -- the caller then falls back to plain top-voted
    selection, so a themeless day still ships a video."""
    harvested = stories.harvest(channel)
    available = stories.unused(harvested, set(used_ids), used_topics, too_similar)
    if not available:
        return None, None
    shortlist = available[:limit]
    listed = "\n".join(
        f"{i}. [{s.get('score', 0)}pts/{s.get('num_comments', 0)}c {s.get('source', '')}] "
        f"{s['title']}\n   {s.get('text', '')[:250]}"
        for i, s in enumerate(shortlist))
    result = nim_json(
        MATCH_SYSTEM + ' JSON schema: {"index": <number>, "title": "...", "why": "..."}',
        f"Theme to match: {theme}\n\nOur harvested stories:\n{listed}",
        max_tokens=700,
    )
    index = result.get("index")
    title = (result.get("title") or "").strip()
    if not isinstance(index, int) or index < 0 or index >= len(shortlist) or not title:
        log("no harvested story genuinely fits the theme")
        return None, None
    clash, why = too_similar(title, used_topics)
    if clash:
        log(f"themed pick '{title}' rejected -- {why} with '{clash}'")
        return None, None
    story = shortlist[index]
    log(f"themed story: [{story.get('source')}] {story['title'][:70]}")
    log(f"  -> {title}")
    return title, story


# ---- entry point ---------------------------------------------------------------------

def themed_pick(channel, state, used_topics, used_ids, too_similar):
    """Full trend path: discover -> theme -> match to a real story.
    Returns (topic, story, trend, theme) with Nones on any soft failure."""
    used_trends = {u.get("trend_id") for u in state.get("uploads", [])
                   if u.get("channel") == channel["id"] and u.get("trend_id")}
    videos = fetch_trending(channel)
    if not videos:
        return None, None, None, None
    trend, theme = pick_theme(channel, videos, used_trends)
    if not theme:
        return None, None, None, None
    topic, story = match_story(channel, theme, used_topics, used_ids, too_similar)
    if not story:
        return None, None, None, None
    return topic, story, trend, theme
