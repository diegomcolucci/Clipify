# -*- coding: utf-8 -*-
"""Clean LLM -> align -> SRT helpers.
One stable entrypoint: call_llm(...)
"""

import difflib
import json
import logging
import os
import re
import unicodedata
from pathlib import Path as _P
from typing import Any, Dict, List, Optional, Tuple

try:
    from rapidfuzz import fuzz as rf_fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

logger = logging.getLogger("clipify.pipelines.llm_pipeline")


# -------------------------
# Utils
# -------------------------

def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _get_env_key() -> Optional[str]:
    return os.getenv("GOOGLE_API_KEY")


def _get_env_model(default: str = "gemini-1.5-flash") -> str:
    m = (os.getenv("GEMINI_MODEL") or "").strip()
    return m or default


def _json_from_text(txt: str) -> Optional[dict]:
    txt = (txt or "").strip()
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def _highlight_narrative_signature(text: str) -> tuple:
    words = [w for w in re.findall(r"\b\w+\b", text.lower()) if len(w) > 2]
    if not words:
        return (), (), ()
    return tuple(words[:5]), tuple(words[-5:]), tuple(sorted({w for w in words if len(w) >= 5}))[:10]


def _rank_highlight_candidate(excerpt: str, title: str, position: int, min_words: int, max_words: int) -> float:
    if not excerpt:
        return -999.0

    words = excerpt.split()
    wc = len(words)
    score = 0.0

    # Keep word-range as a soft preference, not a hard filter.
    if min_words <= wc <= max_words:
        score += 9.0
    elif wc > max_words:
        over = wc - max_words
        if over <= 10:
            score += 6.0
        elif over <= 24:
            score += 3.0
        else:
            score -= min(3.0, (over - 24) * 0.10)
    else:
        under = min_words - wc
        if under <= 5:
            score += 4.0
        else:
            score -= min(4.0, (under - 5) * 0.25)

    text = f"{title} {excerpt}".lower()
    if any(ch.isdigit() for ch in text):
        score += 3.0
    if "?" in title or "?" in excerpt:
        score += 2.0
    if excerpt and excerpt[-1] in ".!?;:":
        score += 2.0

    hooks = ("how", "why", "what", "when", "who", "did", "can", "could", "would", "should", "i", "you", "we", "my", "our")
    if any(h in text.split()[:4] for h in hooks):
        score += 4.0

    score += max(0.0, 2.5 - (position * 0.35))
    return score


def _normalize_highlights(highlights: List[Dict[str, Any]], k: int, min_words: int, max_words: int) -> List[dict]:
    candidates = []
    seen_texts = set()
    seen_signatures = set()

    for idx, h in enumerate(highlights):
        if not isinstance(h, dict):
            continue
        title = (h.get("title") or "").strip()
        excerpt = (h.get("excerpt") or h.get("text") or "").strip()
        if not excerpt:
            continue

        norm_excerpt = _normalize(excerpt)
        signature = _highlight_narrative_signature(f"{title} {excerpt}")
        if norm_excerpt in seen_texts or signature in seen_signatures:
            continue
        seen_texts.add(norm_excerpt)
        seen_signatures.add(signature)
        candidates.append(
            {
                "title": title[:80] or excerpt[:50],
                "excerpt": excerpt,
                "_score": _rank_highlight_candidate(excerpt, title, idx, min_words, max_words),
                "_signature": signature,
            }
        )

    if not candidates:
        return []

    candidates.sort(key=lambda item: item.get("_score", 0.0), reverse=True)
    selected = []
    selected_signatures = []
    for cand in candidates:
        sig = cand.get("_signature") or ((), (), ())
        if sig in selected_signatures:
            continue
        selected.append({"title": cand["title"], "excerpt": cand["excerpt"]})
        selected_signatures.append(sig)
        if len(selected) >= k:
            break
    return selected


# -------------------------
# Timings & tokens
# -------------------------

def read_timings(timings_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Return transcript and a flat list of word timing dicts."""
    tpath = _P(timings_path)
    if not tpath.exists():
        raise FileNotFoundError(f"Timings JSON not found: {tpath}")

    try:
        data = json.loads(tpath.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Invalid JSON at {tpath}: {e}")

    transcript = ""
    words_raw: List[dict] = []

    if isinstance(data, dict):
        transcript = (data.get("transcript") or "").strip()
        if isinstance(data.get("word_timings"), list):
            words_raw = data["word_timings"]
        elif isinstance(data.get("words"), list):
            words_raw = data["words"]
        elif isinstance(data.get("segments"), list):
            for seg in data["segments"]:
                if isinstance(seg, dict) and isinstance(seg.get("words"), list):
                    words_raw.extend(seg["words"])
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        if {"text", "start", "end"}.issubset(set(data[0].keys())):
            words_raw = data
        else:
            for item in data:
                if isinstance(item, dict) and isinstance(item.get("words"), list):
                    words_raw.extend(item["words"])

    word_timings: List[Dict[str, Any]] = []
    for w in words_raw:
        txt = str(w.get("text") or w.get("word") or "")
        s = w.get("start")
        e = w.get("end")
        if s is None or e is None:
            continue
        try:
            word_timings.append({"text": txt, "start": float(s), "end": float(e)})
        except Exception:
            continue

    if not transcript:
        alt_txt = tpath.parent / "input_transcript.txt"
        if alt_txt.exists():
            transcript = alt_txt.read_text(encoding="utf-8").strip()
        elif word_timings:
            transcript = " ".join(w["text"] for w in word_timings)

    return transcript, word_timings


def tokens_from_words(words: List[dict]) -> List[dict]:
    tokens = []
    for w in words:
        txt = str(w.get("text") or "")
        s = w.get("start")
        e = w.get("end")
        if s is None or e is None:
            continue
        tokens.append({"txt": txt, "n": _normalize(txt), "start": float(s), "end": float(e)})
    return tokens


# -------------------------
# Align excerpt
# -------------------------

def align_excerpt_exact_or_fuzzy(excerpt: str, tokens: List[dict], use_fuzzy: bool = False) -> Optional[Tuple[float, float]]:
    if not excerpt or not tokens:
        return None

    tgt = _normalize(excerpt)
    if not tgt:
        return None

    norms = [t["n"] for t in tokens]
    tgt_words = tgt.split()
    if len(tgt_words) < 3:
        return None

    for L in range(min(12, len(tgt_words)), 4, -1):
        for i in range(0, len(tgt_words) - L + 1):
            phrase = " ".join(tgt_words[i : i + L])
            buf = []
            for idx, w in enumerate(norms):
                buf.append(w)
                if len(buf) > L:
                    buf.pop(0)
                if len(buf) == L and " ".join(buf) == phrase:
                    s_idx = idx - L + 1
                    e_idx = min(len(tokens) - 1, idx + (len(tgt_words) - L) + 1)
                    return (tokens[s_idx]["start"], tokens[e_idx]["end"])

    if not use_fuzzy:
        return None

    best = (0.0, None)
    tgt_join = " ".join(tgt_words[:50])
    n_tokens = len(norms)
    window = min(20, max(8, len(tgt_words)))
    step = max(1, window // 2)

    def score(a, b):
        if HAVE_RAPIDFUZZ:
            return rf_fuzz.token_set_ratio(a, b)
        return int(difflib.SequenceMatcher(None, a, b).ratio() * 100)

    for i in range(0, max(1, n_tokens - window + 1), step):
        candidate = " ".join(norms[i : i + window])
        sc = score(tgt_join, candidate)
        if sc > best[0]:
            s_time = tokens[i]["start"]
            e_time = tokens[min(n_tokens - 1, i + window - 1)]["end"]
            best = (sc, (s_time, e_time))

    if best[1] and best[0] >= 70:
        return best[1]
    return None


# -------------------------
# Build SRTs
# -------------------------

def _to_srt_time(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    t -= h * 3600
    m = int(t // 60)
    t -= m * 60
    s = int(t)
    ms = int(round((t - s) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def build_srts(
    segments: List[dict],
    word_timings: List[dict],
    out_dir: str,
    words_per_caption: int = 8,
    max_chars_per_caption: int = 42,
) -> List[_P]:
    outp = _P(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    tokens = tokens_from_words(word_timings) if word_timings else []
    srts = []
    for idx, seg in enumerate(segments, start=1):
        start = seg.get("start")
        end = seg.get("end")
        title = seg.get("title") or f"segment_{idx}"
        p = outp / f"segment_{idx}.srt"
        cues = []

        if start is None or end is None:
            cues = [(0.0, 2.0, title)]
        else:
            clip_duration = float(end) - float(start)
            if tokens:
                ws = [t for t in tokens if start <= t["start"] <= end]
                if ws:
                    line, lstart = [], None
                    lend = 0.0
                    for w in ws:
                        word_rel_start = max(0.0, w["start"] - float(start))
                        word_rel_end = min(clip_duration, w["end"] - float(start))
                        if not line:
                            lstart = word_rel_start
                        line.append(w["txt"])
                        lend = word_rel_end

                        joined = " ".join(line)
                        ends_sentence = bool(joined) and joined[-1] in ".!?;:"
                        if len(line) >= max(1, int(words_per_caption)) or len(joined) >= max(20, int(max_chars_per_caption)) or ends_sentence:
                            if lstart is not None and lend > lstart:
                                cues.append((lstart, lend, joined))
                            line, lstart = [], None

                    if line and lstart is not None and lend > lstart:
                        cues.append((lstart, lend, " ".join(line)))

            if not cues:
                cues = [(0.0, clip_duration, title)]

        lines = []
        for i, (a, b, txt) in enumerate(cues, start=1):
            lines.append(str(i))
            lines.append(f"{_to_srt_time(a)} --> {_to_srt_time(b)}")
            lines.append(txt or title)
            lines.append("")
        p.write_text("\n".join(lines), encoding="utf-8")
        srts.append(p)
    return srts


# -------------------------
# Fallback highlights
# -------------------------

def _fallback_highlights_from_text(full_text: str, k: int = 6, min_words: int = 5, max_words: int = 20) -> List[dict]:
    if not full_text:
        return []

    words_all = re.findall(r"\S+", full_text)
    if len(words_all) < min_words:
        return []

    candidates = []
    fallback_max_words = max(int(max_words), int(min_words) + 12, 32)
    fallback_max_words = min(fallback_max_words, 90)
    parts = re.split(r"(?<=[\.!?])\s+", full_text.strip())
    parts = [s.strip() for s in parts if s.strip()]

    for idx, s in enumerate(parts):
        w = re.findall(r"\S+", s)
        if len(w) < min_words:
            continue
        # Prefer the full sentence unless it is too long for a fallback candidate.
        excerpt = " ".join(w[:fallback_max_words])
        title = " ".join(w[:8])
        candidates.append({
            "title": title[:80],
            "excerpt": excerpt,
            "_score": _rank_highlight_candidate(excerpt, title, idx, min_words, max_words),
        })

    if len(candidates) < k and words_all:
        step = max(min_words, fallback_max_words // 2, len(words_all) // max(1, k))
        i = 0
        chunk_idx = 0
        while len(candidates) < k and i < len(words_all):
            chunk = words_all[i : i + fallback_max_words]
            if len(chunk) >= min_words:
                seg = " ".join(chunk)
                title = " ".join(chunk[:8])
                candidates.append({
                    "title": title[:80],
                    "excerpt": seg,
                    "_score": _rank_highlight_candidate(seg, title, chunk_idx + len(parts), min_words, max_words),
                })
                chunk_idx += 1
            i += step

    if not candidates:
        return []

    return _normalize_highlights(candidates, k, min_words, max_words)


# -------------------------
# Gemini core and wrapper
# -------------------------

def _call_llm_core(
    full_text: str,
    clips: int = 6,
    min_words: int = 5,
    max_words: int = 20,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    ai_provider_name: str = "gemini",
    provider_api_key: Optional[str] = None,
    **_unused,
) -> dict:
    if not full_text:
        return {"highlights": []}

    try:
        clip_count = max(1, int(clips))
    except Exception:
        clip_count = 6

    try:
        lower_bound = max(1, int(min_words))
    except Exception:
        lower_bound = 5

    try:
        upper_bound = max(lower_bound, int(max_words))
    except Exception:
        upper_bound = max(lower_bound, 20)

    prompt = (
        "You are a senior social-media video editor and viral-content strategist.\n"
        "Given a speech transcript, first understand the content and infer the ideal narrative arc(s) before choosing excerpts. "
        f"Select up to {clip_count} excerpts most likely to perform as viral short-form videos (TikTok/Reels/Shorts).\n\n"
        "Return ONLY valid JSON with a single key \"highlights\": a list where each item has:\n"
        "- \"title\": a punchy, clickable title (<= 70 chars). Prefer hooks (numbers, questions, commands, contrasts), clarity, and immediate benefit. "
        "The title MUST be written in the same language as the transcript. Do not translate or switch languages.\n"
        f"- \"excerpt\": exact words copied verbatim from the transcript. Use about {lower_bound} to {upper_bound} words as a flexible guidance range, but preserve the strongest complete thought for that arc.\n"
        "  Excerpts MUST be an exact contiguous substring of the transcript.\n"
        "  Each excerpt MUST form a complete thought with a natural beginning, middle, and end.\n"
        "  * Beginning: a hook or context that pulls the viewer in\n"
        "  * Middle: the core message, insight, or story development\n"
        "  * End: a resolution, punchline, or takeaway that gives closure\n"
        "  The excerpt should feel like a satisfying mini-narrative, not a fragment that cuts off mid-thought.\n"
        "  Trim incidental filler like 'um'/'uh' unless they add character.\n\n"
        "Prioritize excerpts that:\n"
        "- open with a strong hook, reveal, confession, counterintuitive insight, concrete number, or surprise\n"
        "- contain concrete details, vivid imagery, emotions, or a clear actionable takeaway\n"
        "- have a natural arc or progression (setup -> development -> payoff)\n"
        "- are self-contained and understandable out-of-context\n"
        "- are edit-friendly (easy to pair with quick cuts/graphics)\n"
        f"- provide variety across the {clip_count} picks (mix of emotion, practical tip, story/conflict, provocative question)\n\n"
        "Important: do not blindly optimize for 10-15 second teasers. Infer the ideal clip length from the transcript itself. "
        "Prefer the shortest excerpt that still contains a complete and compelling arc, and allow longer arcs when the content needs them.\n"
        "For most strong clips, target around 25-60 seconds when spoken naturally, but let the transcript complexity and payoff determine the final length. "
        "If a stronger arc needs to exceed the guidance range slightly, do that rather than cutting the thought too early.\n"
        "If the transcript contains many strong arcs, return them all as separate highlights instead of compressing to a minimal set. "
        "Do not economize on viral options; surface all genuinely strong candidates you can find.\n"
        "If the transcript does not contain enough strong material, return fewer highlights rather than weaker short ones. "
        f"The user asked for up to {clip_count} clips, but you should return only as many strong, complete arcs as the transcript actually supports.\n\n"
        "Constraints:\n"
        "- Exactly one JSON object, nothing else.\n"
        "- Only the key \"highlights\" with a list of objects containing \"title\" and \"excerpt\" (extra keys will be ignored).\n"
        "- No invented text in \"excerpt\"; it must match the transcript verbatim.\n"
        "- Do NOT pick excerpts that start or end mid-sentence.\n\n"
        "Avoid duplicates and avoid weak, generic lines. Favor short, surprising, emotional, or useful moments that can immediately hook a viewer.\n"
        "When the transcript supports it, prefer multiple distinct arcs over several near-duplicate variants from the same passage.\n\n"
        "Example:\n"
        "{\"highlights\":[{\"title\":\"I lost $10,000 in one week\",\"excerpt\":\"I lost ten thousand dollars in one week and here's what I learned from it so you don't have to make the same mistake\"}]}\n\n"
        "Transcript:\n"
        f"{full_text[:20000]}"
    )

    try:
        from clipify.core.ai_providers import get_ai_provider
    except Exception:
        get_ai_provider = None

    provider_name = (ai_provider_name or "gemini").lower()
    prov_key = provider_api_key or api_key

    logger.info(
        "LLM core call: provider=%s model=%s clips=%s min_words=%s max_words=%s transcript_chars=%s",
        provider_name,
        model,
        clip_count,
        lower_bound,
        upper_bound,
        len(full_text or ""),
    )

    llm_meta = {}

    if get_ai_provider:
        prov = get_ai_provider(provider_name, prov_key, model)
        resp = prov.get_response(prompt)
        if isinstance(resp, dict):
            llm_meta = dict(resp.get("_llm_meta") or {})
    else:
        import google.generativeai as genai

        key = prov_key or _get_env_key()
        if not key:
            raise RuntimeError("Missing GOOGLE_API_KEY in env/.env")
        genai.configure(api_key=key)
        mdl = model or _get_env_model()
        gmodel = genai.GenerativeModel(mdl)
        resp = gmodel.generate_content(prompt, generation_config={"temperature": 0.4, "max_output_tokens": 1400})
        llm_meta = {"provider": provider_name, "model_used": mdl}

    txt = ""
    if isinstance(resp, dict):
        ch = resp.get("choices") or []
        if ch and isinstance(ch, list):
            first = ch[0]
            if isinstance(first, dict):
                msg = first.get("message") or first
                if isinstance(msg, dict):
                    txt = msg.get("content") or ""
                else:
                    txt = msg
    if not txt:
        txt = getattr(resp, "text", "") or ""

    obj = _json_from_text(txt)
    if not obj or not isinstance(obj, dict) or "highlights" not in obj:
        try:
            cands = getattr(resp, "candidates", []) or []
            blocked = any(getattr(c, "finish_reason", None) == 2 for c in cands)
        except Exception:
            blocked = False
        if blocked:
            raise ValueError("AI provider blocked the output.")
        raise ValueError("AI provider did not return valid JSON.")

    return {
        "highlights": _normalize_highlights(obj.get("highlights") or [], clip_count, lower_bound, upper_bound),
        "_llm_meta": llm_meta,
    }


def call_llm(
    *args,
    full_text: Optional[str] = None,
    transcript: Optional[str] = None,
    word_timings=None,
    clips: int = 6,
    min_words: int = 5,
    max_words: int = 20,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> dict:
    if args and full_text is None and transcript is None:
        a0 = args[0]
        if isinstance(a0, str):
            full_text = a0
        elif isinstance(a0, dict):
            if a0.get("transcript"):
                full_text = str(a0["transcript"])
            elif a0.get("full_text"):
                full_text = str(a0["full_text"])
            elif a0.get("text"):
                full_text = str(a0["text"])

    text = (full_text or transcript or "").strip()
    if not text:
        return {"highlights": []}

    requested_clips = max(1, int(clips))
    candidate_clips = max(requested_clips * 4, requested_clips + 12, 16)
    ai_provider_name = kwargs.get("ai_provider_name", "gemini")
    provider_api_key = kwargs.get("provider_api_key") or api_key
    openai_model = kwargs.get("openai_model")
    effective_model = openai_model if openai_model else model

    logger.info(
        "LLM wrapper call: provider=%s model=%s requested_clips=%s candidate_clips=%s min_words=%s max_words=%s",
        ai_provider_name,
        effective_model,
        clips,
        candidate_clips,
        min_words,
        max_words,
    )

    def _try_llm(req_clips, req_min, req_max):
        return _call_llm_core(
            full_text=text,
            clips=req_clips,
            min_words=req_min,
            max_words=req_max,
            model=effective_model,
            api_key=api_key,
            ai_provider_name=ai_provider_name,
            provider_api_key=provider_api_key,
        )

    try:
        res = _try_llm(candidate_clips, min_words, max_words)
        highs = res.get("highlights") or []
        if highs:
            logger.info("LLM wrapper success: provider=%s highlights=%s", ai_provider_name, len(highs))
            return {
                "highlights": highs[:candidate_clips],
                "_llm_meta": dict(res.get("_llm_meta") or {}),
            }
        raise ValueError("LLM returned empty highlights.")
    except Exception:
        logger.exception("LLM wrapper fallback triggered: provider=%s", ai_provider_name)
        highs = _fallback_highlights_from_text(text, k=requested_clips, min_words=min_words, max_words=max_words)
        return {
            "highlights": highs,
            "_llm_meta": {
                "provider": (ai_provider_name or "gemini").lower(),
                "model_requested": effective_model,
                "fallback": True,
            },
        }


def _call_gemini_core(*args, **kwargs) -> dict:
    """Backward-compatible alias for older call sites."""
    return _call_llm_core(*args, **kwargs)


def call_gemini(*args, **kwargs) -> dict:
    """Backward-compatible alias for older call sites."""
    return call_llm(*args, **kwargs)


# -------------------------
# Write processed segments
# -------------------------

def write_processed_segments(segments, out_path: str = "processed_content/input_processed.json"):
    p = _P(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    norm = []
    for i, s in enumerate(segments, 1):
        if not isinstance(s, dict):
            s = {"title": str(s), "start": None, "end": None}
        title = (s.get("title") or f"segment_{i}").strip()
        start = s.get("start")
        end = s.get("end")

        try:
            start = float(start) if start is not None else None
        except Exception:
            start = None
        try:
            end = float(end) if end is not None else None
        except Exception:
            end = None

        norm.append({"title": title, "start": start, "end": end})

    out = {"segments": norm}
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
