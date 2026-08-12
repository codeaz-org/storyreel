# storyreel

Fully automated multi-channel story pipeline: an upvoted subreddit story is retold
in the channel's own words (critic-gated, publish-or-nothing), edited over **real
footage** using [OpenMontage](https://github.com/calesthio/OpenMontage)'s
documentary-montage tools, and published to YouTube. Runs entirely on GitHub
Actions with free services.

## How a video is made

1. **Story** — `stories.py` harvests full-text self-posts from each channel's
   subreddits; Reddit's own votes (`score + 2.5 × comments`) are the quality
   filter, a safety list blocks stories that aren't ours to tell, and every used
   post id in `posted.json` is excluded forever.
2. **Script** — an LLM retells the story in the channel's voice (never verbatim,
   names invented, source credited in the description). A separate critic call
   scores hook/arc/faithfulness/own-words/pacing/landing and rejects until the
   draft earns a pass. Nothing that fails review is published.
3. **Footage** — `montage.py` drives OpenMontage headlessly: an LLM splits the
   approved narration into 6–10 visual slots; `corpus_builder` downloads and
   CLIP-indexes candidates from Pexels/Pixabay (free keys) and Archive.org/
   Wikimedia/NASA (keyless); `clip_search` ranks the corpus per slot;
   clips are fitted to slot length and stitched with `video_stitch`.
4. **Finish** — edge-tts narration (per-channel brand voice), sentence-timed
   burned captions, ffprobe sanity checks, YouTube upload with the
   synthetic-media disclosure, state committed back to the repo.

OpenMontage itself is agent-first and interactive; this repo deliberately
hard-codes one of its recipes so it can run unattended. The LLM chain is
mpt's: NVIDIA NIM → Groq → OpenRouter free tiers, with a repo-local **Ollama**
(llama3.1:8b, cached across runs) as the last-resort fallback.

## Setup (~30 min)

1. Free keys: NVIDIA NIM (build.nvidia.com), Pexels, Pixabay, a Reddit "script"
   app (reddit.com/prefs/apps — recommended, it's the only source with live votes).
2. One YouTube channel per entry in `channels.json`. Google Cloud project →
   enable YouTube Data API v3 → Desktop OAuth client. Per channel, run
   `python get_youtube_token.py`, sign in as that channel, save the token as the
   `YT_REFRESH_TOKEN_<CHANNELID>` secret.
3. Add every key from `.env.example` as an Actions secret. Done — it runs at
   08:00/20:00 UTC and commits `posted.json` back after each upload.

Test locally or via *Run workflow* with `dry_run` checked; the rendered mp4 is
attached as an artifact instead of uploaded.

## Channels

Edit `channels.json` — each entry is a full brand: voice, orientation
(portrait/landscape), footage sources, subreddits, retell prompt, critic rubric,
caption style, hashtags. Add a channel = add an entry + its
`YT_REFRESH_TOKEN_<ID>` secret + one line in the workflow's env block.

## Notes and honest limits

- **First run is slow**: torch CPU wheel + CLIP weights (~350 MB) + Ollama model
  download. All cached; later runs skip them.
- The clip corpus persists via Actions cache, so retrieval improves over time.
- OpenMontage is checked out at `ref: main`; pin a commit SHA after your first
  verified run.
- OpenMontage is AGPL-3.0 — fine to run for your own channels; don't offer this
  pipeline as a closed hosted service.
- YouTube may limit monetization of mass-produced content. The critic gate,
  real-footage edits, and per-story source credits are the mitigations, not a
  guarantee. Start at 1–2 uploads/day per channel.

## Trend research and TikTok

Before falling back to plain top-voted stories, each run researches what's viral:
`viral.py` searches YouTube keylessly (yt-dlp, relevance + freshness) for the
channel's `viral_queries`, keeps videos above `viral_min_views`, and has the LLM
read the **theme** — the underlying story shape, never the title or transcript.
That theme is then matched against our own subreddit harvest, and the best-fitting
*real* story is retold in our brand and voice. **We recreate themes and formats,
never content** — no transcript is fetched, nothing is paraphrased from another
creator, and the script passes the same critic gate. Used trend video ids are
recorded in `posted.json` so a trend is only ridden once per channel. When nothing
fresh matches (or research fails), selection falls back to the subreddit-votes path.

TikTok posting goes through **Buffer** (`buffer.py`, ported from mpt): Buffer's
TikTok-approved app publishes publicly with caption + hashtags, which an unaudited
first-party app cannot. Set `BUFFER_ACCESS_TOKEN` and per-channel
`BUFFER_CHANNEL_ID_<ID>`; the rendered mp4 is hosted as a rolling GitHub release
asset for Buffer to fetch (public repo required). No token = TikTok is skipped,
YouTube unaffected.
