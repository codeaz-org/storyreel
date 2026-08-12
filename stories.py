"""Harvest full-length stories people actually upvoted, and pick one to retell.

The story counterpart of questions.py. A story niche (mode="story" in niches.json)
does not answer questions -- it retells a story that a subreddit already voted to the
top. Reddit's own signal is the whole quality filter:

  1. it must be a self-post with a real body, long enough to carry a video but short
     enough to retell in one narration
  2. it must have engagement -- upvotes and comments ARE the audience test; we never
     publish a story Reddit itself ignored
  3. it must not repeat a story already used, by post id or by subject
  4. it must clear the safety list -- some stories are not this channel's to tell

The retell is written in the channel's own words (enforced in the niche's
script_prompt): the source thread is inspiration and gets a credit link in the
description, never a verbatim read-aloud. That is both the copyright-safe and the
YouTube-reuse-policy-safe way to run a story channel.
"""
import os, random, re

import research
from llm import nim_json

# How much of the post body the writer sees. Stories need the full arc, not the
# 400-char digest the question pipeline uses.
STORY_TEXT_CHARS = 8000

# Bounds on the SOURCE post, in words. Below the floor there is no story to tell;
# above the ceiling it will not survive a single-narration retell.
MIN_WORDS_DEFAULT = 120
MAX_WORDS_DEFAULT = 1000

# score + 2.5 * comments. Comments weigh more: a story people argue about retains
# viewers better than one they silently upvote.
MIN_ENGAGEMENT_DEFAULT = 150
COMMENT_WEIGHT = 2.5

# Stories this channel does not touch, however well they scored. Platform-safe and
# simply not ours to monetise. A niche can override via "blocked_terms".
BLOCKED_DEFAULT = (
    "suicide", "self harm", "self-harm", "kill myself", "rape", "molest", "incest",
    "underage", "minor", "child abuse", "csa", "grooming", "overdose", "od'd",
    "school shooting", "shooting", "stabbed", "gore", "domestic violence",
    "miscarriage", "stillbirth", "terminal", "hospice",
)

# Updates and meta posts assume context the viewer does not have.
SKIP_TITLE_RE = re.compile(
    r"^\s*(?:\[?\s*(?:final\s+)?update\s*\]?|meta|mod\s?post|megathread|best of)\b", re.I)


def log(msg): print(f"[stories] {msg}", flush=True)


def _blocked_pattern(niche):
    terms = niche.get("blocked_terms")
    if terms is None:
        terms = BLOCKED_DEFAULT
    if not terms:
        return None
    return re.compile(r"(?:" + "|".join(re.escape(t) for t in terms) + r")", re.I)


def weighted(post):
    return (post.get("score") or 0) + COMMENT_WEIGHT * (post.get("num_comments") or 0)


def usable_story(post, niche):
    """Why this post cannot be a video, or None if it can."""
    title = (post.get("title") or "").strip()
    text = (post.get("text") or "").strip()
    if not title or not text:
        return "no body text"
    if SKIP_TITLE_RE.search(title):
        return "update/meta post"
    words = len(text.split())
    if words < niche.get("story_min_words", MIN_WORDS_DEFAULT):
        return f"too short ({words} words)"
    if words > niche.get("story_max_words", MAX_WORDS_DEFAULT):
        return f"too long ({words} words)"
    blocked = _blocked_pattern(niche)
    if blocked:
        hit = blocked.search(title) or blocked.search(text)
        if hit:
            return f"blocked term: {hit.group(0)!r}"
    if weighted(post) < niche.get("story_min_engagement", MIN_ENGAGEMENT_DEFAULT):
        return f"not enough engagement ({post.get('score', 0)}pts/{post.get('num_comments', 0)}c)"
    return None


def harvest(niche):
    """Every configured subreddit, several listings, full body text, ranked by
    Reddit's own votes. Randomised subreddit order plus the randomised time windows
    inside research.fetch_subreddit_arctic keep consecutive runs from seeing the
    same top posts -- that is what makes the supply 'always new'."""
    subs = list(niche.get("subreddits", []))
    random.shuffle(subs)
    # "top" only exists on the official API; Arctic Shift answers one time-windowed
    # query whatever the listing, so asking twice would fetch the same rows.
    listings = ("hot", "top") if os.environ.get("REDDIT_CLIENT_ID") else ("hot",)
    posts = []
    for sub in subs:
        for listing in listings:
            try:
                posts += research.fetch_subreddit(
                    sub, listing, limit=25, text_chars=STORY_TEXT_CHARS)
            except Exception as e:
                log(f"r/{sub}/{listing}: {type(e).__name__}: {str(e)[:70]}")

    stories, seen, rejected = [], set(), {}
    for p in posts:
        key = (p.get("id") or p.get("title", "").strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        why = usable_story(p, niche)
        if why:
            rejected[why.split(" (")[0]] = rejected.get(why.split(" (")[0], 0) + 1
            continue
        stories.append(p)

    stories.sort(key=weighted, reverse=True)
    log(f"{len(stories)} tellable stories from {len(posts)} posts"
        + (f" (rejected: {rejected})" if rejected else ""))
    return stories


def unused(stories, used_ids, used_topics, too_similar):
    """Drop anything already retold, by source post id or by subject."""
    out = []
    for s in stories:
        if s.get("id") and s["id"] in used_ids:
            continue
        clash, _ = too_similar(s["title"], used_topics)
        if clash:
            continue
        out.append(s)
    return out


SELECT_SYSTEM = """You pick which real story to retell in the next video for a story channel.

You are given stories that a subreddit already upvoted, ranked by their votes and
comments. Reddit has done the quality filter; your job is narratability. Pick the ONE
that:
  - has a clear arc: setup, escalation, turn, landing -- not a mood piece or a rant;
  - stands alone: no context from an earlier post needed, no niche in-jokes;
  - a narrator can tell in one take without on-screen images to make sense;
  - punches up or at the teller themselves, never down.

Then write a hook TITLE for the video: under 12 words, states the collision at the
heart of the story without spoiling the turn. "My landlord's 'free' fridge cost me my
deposit" is right. "Crazy landlord story" is wrong.

Return the index of the story you chose and the title."""


def choose(niche, stories, limit=12):
    """The model picks the most tellable story from the top of the ranking and
    phrases the video title. It selects; it does not invent."""
    if not stories:
        raise RuntimeError("no unused stories available")
    shortlist = stories[:limit]
    listed = "\n".join(
        f"{i}. [{s.get('score', 0)}pts/{s.get('num_comments', 0)}c {s.get('source', '')}] "
        f"{s['title']}\n   {s.get('text', '')[:300]}"
        for i, s in enumerate(shortlist)
    )
    guidance = niche.get("select_prompt") or SELECT_SYSTEM
    result = nim_json(
        guidance + ' JSON schema: {"index": <number>, "title": "...", "why": "..."}',
        f"Niche: {niche['name']}\n\nStories:\n{listed}",
        max_tokens=700,
    )
    index = result.get("index")
    title = (result.get("title") or "").strip()
    if not isinstance(index, int) or not 0 <= index < len(shortlist) or not title:
        raise RuntimeError(f"model returned an unusable selection: {str(result)[:160]}")
    story = shortlist[index]
    log(f"chose [{story.get('source')}] {story['title'][:80]} "
        f"({story.get('score', 0)}pts/{story.get('num_comments', 0)}c)")
    log(f"  -> {title}")
    return title, story


def pick_story(niche, used_topics, used_ids, too_similar, attempts=2):
    """(topic, story) grounded in a real upvoted thread. Mirrors autopilot.pick_topic
    so run_niche treats both modes identically -- the story dict carries the same
    id/url/title keys the state file already records for questions."""
    harvested = harvest(niche)
    available = unused(harvested, set(used_ids), used_topics, too_similar)
    log(f"[{niche['id']}] {len(available)} unretold stories after dedupe")
    if not available:
        raise RuntimeError("no unretold stories today: widen the niche's subreddits "
                           "or lower story_min_engagement")
    rejected = []
    for _ in range(attempts):
        topic, story = choose(niche, [s for s in available if s not in rejected])
        clash, why = too_similar(topic, used_topics)
        if clash:
            log(f"[{niche['id']}] rejected '{topic}' -- {why} with '{clash}'")
            rejected.append(story)
            continue
        return topic, story
    raise RuntimeError("every candidate story repeated an existing video")


def pick_stories(niche, used_topics, used_ids, too_similar, count, attempts=2):
    """N distinct stories for one compilation video. One harvest, then repeated
    selection with everything already picked excluded -- by post, by id, and by
    subject, so a compilation never tells the same story twice in one video."""
    harvested = harvest(niche)
    available = unused(harvested, set(used_ids), used_topics, too_similar)
    log(f"[{niche['id']}] {len(available)} unretold stories after dedupe")

    picks, local_topics = [], list(used_topics)
    for _ in range(count):
        pool = [s for s in available if all(s is not p[1] for p in picks)]
        if not pool:
            break
        chosen = None
        rejected = []
        for _ in range(attempts):
            title, story = choose(niche, [s for s in pool if s not in rejected])
            clash, why = too_similar(title, local_topics)
            if clash:
                log(f"[{niche['id']}] rejected '{title}' -- {why} with '{clash}'")
                rejected.append(story)
                if len(rejected) >= len(pool):
                    break
                continue
            chosen = (title, story)
            break
        if not chosen:
            break
        picks.append(chosen)
        local_topics.append(chosen[0])
        available = [s for s in available if s is not chosen[1]]

    if len(picks) < 2:
        raise RuntimeError(f"only {len(picks)} usable stories today; a compilation "
                           "needs at least 2 -- widen subreddits or lower "
                           "story_min_engagement")
    if len(picks) < count:
        log(f"[{niche['id']}] compilation shortened to {len(picks)} stories "
            f"(wanted {count})")
    return picks
