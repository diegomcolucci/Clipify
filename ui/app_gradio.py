import os, json, re, tempfile, uuid, zipfile
from pathlib import Path
import gradio as gr

def _normalize_highlights(res):
    # Accept dict({highlights:[...]}) or raw list
    if isinstance(res, dict):
        highs = res.get('highlights') or []
    elif isinstance(res, list):
        highs = res
    else:
        highs = []
    out=[]
    for h in (highs.get('highlights', highs) if isinstance(highs, dict) else highs):
        if isinstance(h, dict):
            title   = (h.get('title') or '').strip()
            excerpt = (h.get('excerpt') or h.get('text') or '').strip()
        else:
            title   = str(h)[:80]
            excerpt = str(h)
        if excerpt:
            out.append({'title': title or excerpt[:50], 'excerpt': excerpt})
    return {'highlights': out}

from dotenv import load_dotenv
import sys
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


load_dotenv()

# Configure ImageMagick for moviepy (Windows common locations)
import shutil
if not shutil.which('magick'):
    import moviepy.config as c
    possible_paths = [
        r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe",
        r"C:\Program Files\ImageMagick\magick.exe",
        r"C:\Program Files (x86)\ImageMagick\magick.exe",
    ]
    for path in possible_paths:
        if _P(path).exists():
            c.change_settings({'IMAGEMAGICK_BINARY': path})
            break

ENV_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash-001')

from clipify.pipelines.llm_pipeline import (
    read_timings, call_llm, tokens_from_words,
    align_excerpt_exact_or_fuzzy, write_processed_segments, build_srts
)
from clipify.pipelines.ui_helpers import (
    ensure_transcripts, ffmpeg_cut, ffmpeg_burn_subs,
    css_hex_to_ass, safe_name
)

REPO = Path(__file__).resolve().parents[1]

# --- Logging: capture end-to-end UI/server activity into app_gradio.log ---
import logging
logger = logging.getLogger("clipify.ui.app_gradio")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    try:
        log_path = REPO / "app_gradio.log"
        fh = logging.FileHandler(str(log_path), mode='a')
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Also echo logs to the terminal so they are visible while running `python ui/app_gradio.py`.
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        # Route LLM/provider logs to the same file and terminal for end-to-end traceability.
        for lname in ("clipify.pipelines.llm_pipeline", "clipify.ai_providers"):
            lgr = logging.getLogger(lname)
            lgr.setLevel(logging.INFO)
            if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == fh.baseFilename for h in lgr.handlers):
                lgr.addHandler(fh)
            if not any(isinstance(h, logging.StreamHandler) for h in lgr.handlers):
                lgr.addHandler(sh)
            lgr.propagate = False
    except Exception:
        # fallback to basic config if file handler cannot be created
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')


def _collect_file_paths(items):
    paths = []
    for item in (items or []):
        p = None
        if isinstance(item, str):
            p = item
        elif isinstance(item, dict):
            p = item.get("path") or item.get("name")
        elif hasattr(item, "name"):
            p = getattr(item, "name", None)
        if p and os.path.isfile(p):
            paths.append(p)
    return paths


def _create_clips_zip(items):
    paths = _collect_file_paths(items)
    if not paths:
        return None

    out_dir = REPO / "processed_content" / "downloads"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"clipify_clips_{uuid.uuid4().hex[:8]}.zip"

    used_names = set()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, p in enumerate(paths, start=1):
            base = Path(p).name
            arcname = base
            if arcname in used_names:
                arcname = f"{i}_{base}"
            used_names.add(arcname)
            zf.write(p, arcname=arcname)

    return str(zip_path)


def _gather_fonts(repo: Path):
    """Find fonts in repo/fonts, ui/fonts, system locations and bundled captacity assets."""
    try:
        import fontTools.ttLib as ttLib
    except ImportError:
        print("Warning: fontTools not installed. Font metadata will be limited.")
        ttLib = None

    fonts_list = []  # display names
    fonts_map = {}   # name -> path
    fonts_meta = {}  # name -> metadata

    # Locate fonts bundled with captacity_clipify
    try:
        import captacity_clipify
        captacity_fonts_dir = Path(captacity_clipify.__file__).resolve().parent / "assets" / "fonts"
    except Exception:
        captacity_fonts_dir = None

    # Look in standard locations
    search_paths = [
        repo / "fonts",
        repo / "ui" / "fonts",
        captacity_fonts_dir,
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
    ]
    # Add Windows font directories
    if os.name == "nt":
        win_fonts = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
        search_paths.append(win_fonts)
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        if local_appdata:
            search_paths.append(Path(local_appdata) / "Microsoft" / "Windows" / "Fonts")

    # Common font extensions
    FONT_EXTENSIONS = {'.ttf', '.otf', '.ttc', '.dfont'}

    def get_font_info(font_path):
        """Extract font metadata if possible."""
        if not ttLib:
            return {'family': font_path.stem, 'style': 'Regular'}

        try:
            tt = ttLib.TTFont(font_path)
            family = style = ""

            for record in tt['name'].names:
                if record.nameID == 1 and not family:  # Font Family name
                    family = record.toUnicode()
                elif record.nameID == 2 and not style:  # Font Subfamily name
                    style = record.toUnicode()

            return {
                'family': family or font_path.stem,
                'style': style or 'Regular'
            }
        except Exception:
            return {'family': font_path.stem, 'style': 'Regular'}

    # Scan for fonts
    for path in search_paths:
        if not path or not path.exists():
            continue

        for font_file in path.glob('**/*'):
            if not font_file.is_file():
                continue

            if font_file.suffix.lower() in FONT_EXTENSIONS:
                try:
                    font_info = get_font_info(font_file)
                    display_name = f"{font_info['family']} ({font_info['style']})"
                    if display_name not in fonts_map:
                        fonts_list.append(display_name)
                        fonts_map[display_name] = str(font_file)
                        fonts_meta[display_name] = font_info
                except Exception as e:
                    print(f"Warning: Could not process font {font_file}: {e}")

    # Set default font: prefer Bangers-Regular.ttf, then any Regular, then first available
    default_font = None
    for name, path in fonts_map.items():
        if Path(path).name.lower() == "bangers-regular.ttf":
            default_font = name
            break
    if default_font is None:
        for name in fonts_list:
            if "Bangers" in name:
                default_font = name
                break
    if default_font is None:
        for name in fonts_list:
            if "Regular" in name or "regular" in name:
                default_font = name
                break
    if default_font is None and fonts_list:
        default_font = fonts_list[0]

    return fonts_list, fonts_map, default_font

# discover fonts once at module import
FONTS_LIST, FONTS_MAP, FONTS_DEFAULT = _gather_fonts(REPO)

def get_video_duration(path: Path) -> float:
    """Return video duration in seconds using ffprobe, or 0 on failure."""
    try:
        import subprocess
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return float(out)
    except Exception:
        return 0.0


def _get_video_width(path: Path) -> int:
    """Return video width via ffprobe, or a safe default on failure."""
    try:
        import subprocess
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width",
            "-of", "csv=s=x:p=0", str(path)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return int(out)
    except Exception:
        return 1080


def _suggest_params(duration: float, total_words: int) -> dict:
    """Suggest clip parameters based on video duration and transcript word count."""
    if duration <= 0:
        duration = 60.0
    if total_words <= 0:
        total_words = int(duration * 2.5)  # assume ~150 wpm

    # Prefer fewer, stronger clips over many short teaser fragments.
    clips = max(2, min(6, round(duration / 120) + 2))

    # Estimate speaking rate and target excerpt length.
    # These are broad guidance values; the LLM should still infer the ideal arc.
    words_per_second = total_words / max(1.0, duration)
    target_seconds = 45.0
    target_words = max(28, min(160, round(words_per_second * target_seconds)))

    min_words = max(18, target_words - 16)
    max_words = max(min_words + 24, target_words + 24)

    # Pads and minimum duration scale slightly with clip density
    pre_pad = 1.5
    post_pad = 2.8
    min_duration = max(28.0, min(75.0, target_seconds * 1.35))

    return {
        "clips": clips,
        "min_words": min_words,
        "max_words": max_words,
        "pre_pad": pre_pad,
        "post_pad": post_pad,
        "min_duration": min_duration,
    }


def _ui_updates_from_llm_params(params: dict) -> list:
    """Return UI-safe updates that rewrite slider ranges around the LLM suggestion."""
    if not isinstance(params, dict):
        params = {}

    clips_value = int(params.get("clips", 3))
    min_words_value = int(params.get("min_words", 18))
    max_words_value = int(params.get("max_words", 48))
    pre_pad_value = float(params.get("pre_pad", 0.5))
    post_pad_value = float(params.get("post_pad", 1.0))
    min_duration_value = float(params.get("min_duration", 15.0))

    clips_min = max(1, min(3, clips_value))
    clips_max = max(16, clips_value + 6)
    min_words_min = max(1, min(5, min_words_value))
    min_words_max = max(240, min_words_value + 60)
    max_words_min = max(12, min(max_words_value, 12))
    max_words_max = max(400, max_words_value + 80)
    pre_pad_min, pre_pad_max = 0.0, 5.0
    post_pad_min, post_pad_max = 0.0, 5.0
    min_duration_min = 2.0
    min_duration_max = max(30.0, min_duration_value + 12.0)

    return [
        gr.update(minimum=clips_min, maximum=clips_max, value=clips_value),
        gr.update(minimum=min_words_min, maximum=min_words_max, value=min_words_value),
        gr.update(minimum=max_words_min, maximum=max_words_max, value=max_words_value),
        gr.update(minimum=pre_pad_min, maximum=pre_pad_max, value=pre_pad_value),
        gr.update(minimum=post_pad_min, maximum=post_pad_max, value=post_pad_value),
        gr.update(minimum=min_duration_min, maximum=min_duration_max, value=min_duration_value),
    ]


def _effect_phrase_candidates(transcript: str, limit: int = 3) -> list:
    """Return concise, contextual fallback lines when strong viral arcs are scarce."""
    text = (transcript or "").strip()
    if not text:
        return []

    hype_terms = {
        "agora", "nunca", "segredo", "erro", "viral", "choque", "prova", "resultado",
        "how", "why", "secret", "mistake", "proof", "truth", "impacto", "aprendi"
    }
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s and s.strip()]
    scored = []
    for s in sentences:
        words = re.findall(r"\S+", s)
        wc = len(words)
        if wc < 8 or wc > 34:
            continue
        s_low = s.lower()
        score = 0
        if 12 <= wc <= 28:
            score += 4
        if any(ch.isdigit() for ch in s_low):
            score += 2
        if "?" in s:
            score += 2
        if any(t in s_low.split() for t in hype_terms):
            score += 3
        if s.endswith((".", "!", "?")):
            score += 1
        scored.append((score, s))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    seen = set()
    for _, s in scored:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _llm_clip_feedback(requested: int, actual: int, alt_phrases: list = None) -> str:
    requested = max(1, int(requested or 1))
    actual = max(0, int(actual or 0))
    alts = [a for a in (alt_phrases or []) if isinstance(a, str) and a.strip()]

    if actual <= 0:
        msg = (
            "A LLM nao encontrou conteudo suficientemente forte para cortes virais de qualidade neste material."
        )
        if alts:
            msg += "\nAlternativa sugerida (frases de efeito com contexto):"
            for s in alts[:3]:
                msg += f"\n- {s}"
        return msg

    if actual < requested:
        return (
            f"A LLM identificou {actual} clip(s) forte(s) de {requested} solicitados. "
            "O controle foi atualizado para refletir o total recomendado."
        )

    return f"A LLM confirmou {actual} clip(s) com potencial viral dentro da solicitacao atual."


def _score_excerpt(excerpt: str, start: float, end: float, total_duration: float, min_words: int, max_words: int) -> int:
    """Heuristic virality score (0-100) for an excerpt.
    - prefers medium-long excerpts (within min/max words)
    - rewards hook words/questions and numbers
    - slight bonus for earlier placement
    """
    if not excerpt:
        return 0
    txt = excerpt.lower()
    words = txt.split()
    wc = len(words)

    # length score: ideal around max_words, scaled 0..1
    if wc <= min_words:
        length_score = wc / max(1, min_words)
    else:
        length_score = min(1.0, wc / max(1, max_words))

    # hook words
    hooks = ("how", "why", "what", "when", "who", "did", "can", "could", "would", "should", "i", "you", "we", "my", "our")
    hook_bonus = 1.0 if any(h in txt.split()[:4] for h in hooks) else 0.0
    # number bonus
    number_bonus = 1.0 if any(ch.isdigit() for ch in txt) else 0.0

    # position bonus: earlier in video slightly better
    pos_norm = 1.0 - min(1.0, (start / max(1.0, total_duration))) if total_duration > 0 else 0.5

    score = (0.5 * length_score + 0.25 * (0.6 * hook_bonus + 0.4 * number_bonus) + 0.25 * pos_norm)
    return int(max(0, min(100, round(score * 100))))


def _narrative_signature(text: str) -> tuple:
    """Build a coarse narrative signature so near-duplicate arcs can be filtered."""
    if not text:
        return (), (), ()

    words = [w for w in re.findall(r"\b\w+\b", text.lower()) if len(w) > 2]
    if not words:
        return (), (), ()

    hook_words = tuple(words[:6])
    tail_words = tuple(words[-6:])
    key_terms = tuple(sorted({w for w in words if len(w) >= 5}))
    return hook_words, tail_words, key_terms[:12]


STYLE_PRESETS = {
    "Hype Clean": {
        "font_size": 30,
        "primary_hex": "#FFFFFF",
        "outline_w": 4,
        "outline_hex": "#111111",
        "subtitle_style": "Box",
        "words_per_caption": 3,
        "position": "bottom",
        "vertical_offset": 84,
        "horizontal_align": "center",
    },
    "Neon Punch": {
        "font_size": 31,
        "primary_hex": "#F8FF4D",
        "outline_w": 5,
        "outline_hex": "#060606",
        "subtitle_style": "Outline",
        "words_per_caption": 2,
        "position": "center",
        "vertical_offset": 62,
        "horizontal_align": "center",
    },
    "Bold Contrast": {
        "font_size": 29,
        "primary_hex": "#FFFFFF",
        "outline_w": 6,
        "outline_hex": "#000000",
        "subtitle_style": "Box",
        "words_per_caption": 3,
        "position": "bottom",
        "vertical_offset": 82,
        "horizontal_align": "center",
    },
}

# Global parity factor to better match rendered ASS text with UI style preview.
FONT_PARITY_FACTOR = 0.95


def _calibrate_style_for_aspect(font_size: int, outline_w: int, vertical_offset: int, aspect: str) -> tuple:
    """Auto-calibrate font, stroke and vertical offset by aspect ratio for consistency."""
    multipliers = {
        "source": (1.00, 1.00, 1.00),
        "9:16": (1.10, 1.20, 1.00),
        "4:5": (1.05, 1.10, 0.95),
        "1:1": (0.96, 0.95, 0.90),
    }
    f_mul, o_mul, v_mul = multipliers.get(aspect, (1.0, 1.0, 1.0))
    f_size = max(8, int(round(float(font_size) * f_mul)))
    o_size = max(0, int(round(float(outline_w) * o_mul)))
    v_pos = int(max(5, min(95, round(float(vertical_offset) * v_mul))))
    return f_size, o_size, v_pos


def _resolve_font_name_for_ass(font_name: str) -> str:
    """Resolve UI font selection into an ASS-friendly font family name."""
    font_path = font_name or "Bangers-Regular.ttf"
    if isinstance(font_path, str) and font_path in FONTS_MAP:
        font_path = FONTS_MAP.get(font_path)
    else:
        maybe = REPO / "fonts" / str(font_path)
        if maybe.exists():
            font_path = str(maybe)
        else:
            maybe2 = REPO / "ui" / "fonts" / str(font_path)
            if maybe2.exists():
                font_path = str(maybe2)

    try:
        if isinstance(font_name, str) and " (" in font_name:
            return font_name.rsplit(" (", 1)[0]
        if isinstance(font_path, str):
            return _P(font_path).stem
        return str(font_name)
    except Exception:
        return str(font_name or "Bangers")


def _resolve_font_path(font_name: str) -> str:
    """Resolve selected font to a concrete font file path when available."""
    try:
        if isinstance(font_name, str) and font_name in FONTS_MAP:
            return str(FONTS_MAP.get(font_name))
        if isinstance(font_name, str):
            p1 = REPO / "fonts" / font_name
            if p1.exists():
                return str(p1)
            p2 = REPO / "ui" / "fonts" / font_name
            if p2.exists():
                return str(p2)
    except Exception:
        pass
    return ""


def _build_ass_style(font_name: str,
                     font_size: int,
                     primary_hex: str,
                     outline_w: int,
                     outline_hex: str,
                     subtitle_style: str,
                     position: str,
                     vertical_offset: int,
                     horizontal_align: str,
                     aspect: str,
                     target_width: int = 1080,
                     font_path: str = "") -> dict:
    cal_font_size, cal_outline_w, cal_vertical = _calibrate_style_for_aspect(font_size, outline_w, vertical_offset, aspect)
    font_for_ass = _resolve_font_name_for_ass(font_name)
    # Keep this value as desired on-screen size (pixels perceived by user).
    # Conversion to ASS script units is handled in ffmpeg_burn_subs with video height.
    ass_font_size = max(8, int(round(cal_font_size * FONT_PARITY_FACTOR)))
    style = {
        "FontName": font_for_ass,
        "FontSize": ass_font_size,
        "PrimaryColour": (primary_hex or "#FFFFFF"),
        "OutlineColour": (outline_hex or "#000000"),
        "Outline": int(cal_outline_w) if subtitle_style in ("Box", "Outline") else 0,
        "BorderStyle": 3 if subtitle_style == "Box" else 1,
    }
    if font_path:
        style["FontPath"] = font_path
    alignment, margin_v = _position_to_ass_alignment(position, int(cal_vertical), horizontal_align)
    style["Alignment"] = alignment
    style["MarginV"] = margin_v
    return style


def _render_style_preview_clip(video_file: str,
                               font_name: str,
                               font_size: int,
                               primary_hex: str,
                               outline_w: int,
                               outline_hex: str,
                               subtitle_style: str,
                               position: str,
                               aspect: str,
                               vertical_offset: int,
                               horizontal_align: str) -> tuple:
    """Render a real 3-second preview clip using the final ASS subtitle pipeline."""
    if not video_file:
        return None, "Upload a video first to render preview."
    try:
        src = Path(video_file)
        preview_dir = REPO / "processed_content" / "style_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        suffix = uuid.uuid4().hex[:8]
        raw_clip = preview_dir / f"style_preview_{suffix}_raw.mp4"
        out_clip = preview_dir / f"style_preview_{suffix}.mp4"
        srt = preview_dir / f"style_preview_{suffix}.srt"

        ffmpeg_cut(src, 0.0, 3.0, raw_clip)

        # Use real spoken words from the video for style preview (no placeholder text).
        preview_text = ""
        try:
            tjson = REPO / "transcripts" / "input_timings.json"
            if not tjson.exists():
                tjson = ensure_transcripts(REPO, src)
            transcript, words = read_timings(tjson)
            words_3s = [
                str(w.get("text") or "").strip()
                for w in (words or [])
                if 0.15 <= float(w.get("start", 0.0)) <= 2.9
            ]
            preview_text = " ".join([w for w in words_3s if w][:10]).strip()
            if not preview_text:
                preview_text = " ".join((transcript or "").split()[:10]).strip()
        except Exception:
            preview_text = ""

        if not preview_text:
            preview_text = "PREVIA DE LEGENDA"

        srt.write_text(
            f"1\n00:00:00,200 --> 00:00:02,900\n{preview_text}\n",
            encoding="utf-8"
        )
        target_width = 1080 if aspect in ("9:16", "4:5", "1:1") else _get_video_width(raw_clip)
        font_path = _resolve_font_path(font_name)
        style = _build_ass_style(
            font_name=font_name,
            font_size=int(font_size),
            primary_hex=primary_hex,
            outline_w=int(outline_w),
            outline_hex=outline_hex,
            subtitle_style=subtitle_style,
            position=position,
            vertical_offset=int(vertical_offset),
            horizontal_align=horizontal_align,
            aspect=aspect,
            target_width=target_width,
            font_path=font_path,
        )
        ffmpeg_burn_subs(raw_clip, srt, out_clip, aspect=aspect, style=style, fonts_dir=(Path(font_path).parent if font_path else None))
        return str(out_clip), "3s rendered preview ready."
    except Exception as e:
        logger.exception("Failed to render style preview clip")
        return None, f"Preview render failed: {e}"


def _render_first_segment_preview(video_file: str,
                                  segment: dict,
                                  words: list,
                                  font_name: str,
                                  font_size: int,
                                  primary_hex: str,
                                  outline_w: int,
                                  outline_hex: str,
                                  subtitle_style: str,
                                  position: str,
                                  aspect: str,
                                  vertical_offset: int,
                                  horizontal_align: str,
                                  words_per_caption: int) -> str:
    """Render one real preview clip from the first computed segment using final ASS styling."""
    preview_dir = REPO / "processed_content" / "preview_segments"
    preview_dir.mkdir(parents=True, exist_ok=True)
    suffix = uuid.uuid4().hex[:8]
    raw_clip = preview_dir / f"segment_preview_{suffix}_raw.mp4"
    out_clip = preview_dir / f"segment_preview_{suffix}.mp4"
    srt_dir = preview_dir / f"srt_{suffix}"
    srt_dir.mkdir(parents=True, exist_ok=True)

    total_duration = get_video_duration(Path(video_file))
    start = max(0.0, float(segment.get("start", 0.0)))
    end = max(start + 0.3, float(segment.get("end", start + 3.0)))
    if total_duration > 0:
        start = min(start, max(0.0, total_duration - 0.2))
        end = min(end, total_duration)
    if end - start < 0.2:
        raise ValueError(f"Invalid preview segment window: start={start:.3f}, end={end:.3f}")

    ffmpeg_cut(Path(video_file), start, end, raw_clip)
    if not raw_clip.exists() or raw_clip.stat().st_size < 1024:
        raise RuntimeError(f"Preview cut produced empty output: {raw_clip}")

    # Build exactly one SRT using the same word chunking logic as final generation.
    build_srts(
        [{"title": segment.get("title", "preview"), "start": start, "end": end}],
        words,
        srt_dir,
        words_per_caption=int(words_per_caption)
    )
    srt = srt_dir / "segment_1.srt"

    target_width = 1080 if aspect in ("9:16", "4:5", "1:1") else _get_video_width(raw_clip)
    font_path = _resolve_font_path(font_name)
    style = _build_ass_style(
        font_name=font_name,
        font_size=int(font_size),
        primary_hex=primary_hex,
        outline_w=int(outline_w),
        outline_hex=outline_hex,
        subtitle_style=subtitle_style,
        position=position,
        vertical_offset=int(vertical_offset),
        horizontal_align=horizontal_align,
        aspect=aspect,
        target_width=target_width,
        font_path=font_path,
    )
    ffmpeg_burn_subs(raw_clip, srt, out_clip, aspect=aspect, style=style, fonts_dir=(Path(font_path).parent if font_path else None))
    return str(out_clip)


def _refine_clip_candidates(candidates: list, desired_count: int, total_duration: float) -> list:
    """Rank and diversify clip candidates for stronger, less repetitive picks."""
    if not candidates:
        return []

    hype_terms = {
        "agora", "nunca", "segredo", "erro", "viral", "choque", "prova", "resultado",
        "how", "why", "secret", "mistake", "viral", "proof", "result", "truth"
    }

    scored = []
    for c in candidates:
        title = (c.get("title") or "").strip()
        excerpt = (c.get("excerpt") or "").strip()
        content = f"{title} {excerpt}".strip()
        txt = content.lower()
        base = float(c.get("score", 0))
        dur = max(0.0, float(c.get("end", 0.0)) - float(c.get("start", 0.0)))
        hook_words, tail_words, key_terms = _narrative_signature(content)

        # Prefer story-sized clips with enough context to land as a mini-arc.
        if 20.0 <= dur <= 45.0:
            base += 12.0
        elif 16.0 <= dur <= 55.0:
            base += 6.0
        elif dur < 16.0:
            base -= 8.0

        # Narrative completeness and specificity should beat generic teaser fragments.
        if len(key_terms) >= 5:
            base += 8.0
        if len(hook_words) >= 4 and len(tail_words) >= 4:
            base += 4.0
        if hook_words and tail_words and hook_words[0] != tail_words[-1]:
            base += 2.0

        if any(ch.isdigit() for ch in txt):
            base += 3.0
        if "?" in title:
            base += 2.0
        if any(t in txt.split() for t in hype_terms):
            base += 4.0

        c2 = dict(c)
        c2["_signature"] = (hook_words, tail_words, key_terms)
        c2["_rank"] = base
        scored.append(c2)

    scored.sort(key=lambda x: x.get("_rank", 0), reverse=True)

    if not scored:
        return []

    top_rank = float(scored[0].get("_rank", 0))
    # Keep any clip that is strong relative to the best candidate; this allows
    # the system to surface multiple viral cuts instead of artificially stopping
    # at a tiny top-N list.
    quality_cutoff = max(40.0, top_rank * 0.72)

    selected = []
    selected_signatures = []
    # Keep temporal diversity so clips are spread across the source video.
    bucket_size = max(1.0, total_duration / max(3.0, min(6.0, float(desired_count) * 1.2))) if total_duration > 0 else 30.0
    used_buckets = set()
    for c in scored:
        s = float(c.get("start", 0.0))
        e = float(c.get("end", 0.0))
        rank = float(c.get("_rank", 0))
        b = int(s // bucket_size)
        signature = c.get("_signature") or ((), (), ())

        strong_enough = rank >= quality_cutoff

        overlap = False
        for k in selected:
            ks, ke = float(k.get("start", 0.0)), float(k.get("end", 0.0))
            inter = max(0.0, min(e, ke) - max(s, ks))
            union = max(0.001, (e - s) + (ke - ks) - inter)
            if (inter / union) > 0.30 and strong_enough:
                overlap = True
                break
        if overlap:
            continue

        if selected_signatures:
            hook_words, tail_words, key_terms = signature
            for sh, st, sk in selected_signatures:
                shared = len(set(key_terms) & set(sk))
                if shared >= 4 and (hook_words[:3] == sh[:3] or tail_words[-3:] == st[-3:]):
                    overlap = True
                    break
            if overlap:
                continue

        if strong_enough and (b not in used_buckets or len(selected) < 2):
            selected.append(c)
            used_buckets.add(b)
            selected_signatures.append(signature)

    if len(selected) < desired_count:
        for c in scored:
            if c in selected:
                continue
            if float(c.get("_rank", 0)) < max(32.0, quality_cutoff * 0.85):
                continue
            signature = c.get("_signature") or ((), (), ())
            duplicate_arc = False
            for sh, st, sk in selected_signatures:
                if len(set(signature[2]) & set(sk)) >= 5 and (signature[0][:2] == sh[:2] or signature[1][-2:] == st[-2:]):
                    duplicate_arc = True
                    break
            if duplicate_arc:
                continue
            selected.append(c)
            selected_signatures.append(signature)
            if len(selected) >= desired_count:
                break

    # If the transcript supports more than the minimum requested count, keep them.
    if len(selected) < desired_count:
        # Backfill the remainder with the best available candidates, even if they
        # are not strongly above the cutoff, so the pipeline never returns too few.
        for c in scored:
            if len(selected) >= desired_count:
                break
            if c in selected:
                continue
            signature = c.get("_signature") or ((), (), ())
            if any(len(set(signature[2]) & set(sk)) >= 6 for _, _, sk in selected_signatures):
                continue
            selected.append(c)
            selected_signatures.append(signature)

    selected.sort(key=lambda x: x.get("start", 0.0))
    for c in selected:
        c.pop("_rank", None)
        c.pop("_signature", None)
    return selected


def _remux_audio(source_video: Path, target_video: Path) -> bool:
    """Copy audio stream from source_video into target_video and normalize loudness.

    Some captioning tools (e.g. captacity) drop the audio track. This helper
    re-injects the original audio into the captioned output using ffmpeg and
    applies a loudnorm filter for more consistent, broadcast-friendly levels.
    """
    import subprocess
    tmp = target_video.with_suffix(".tmp" + target_video.suffix)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(target_video),
        "-i", str(source_video),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(tmp)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tmp.replace(target_video)
        return True
    except Exception as e:
        logger.warning("Audio remux failed for %s: %s", target_video, e)
        if tmp.exists():
            tmp.unlink()
        return False


def _srt_to_captacity_segments(srt_path: Path) -> list:
    """Convert an SRT file into the segment format expected by captacity_clipify.

    captacity expects segments=[{"words": [{"word": "...", "start": 0.0, "end": 0.5}, ...]}].
    SRT only has cue-level timings, so we split each cue's text into words and
    distribute the cue duration evenly across words.
    """
    import re
    if not srt_path.exists():
        return []

    def _parse_time(t: str) -> float:
        # SRT time: HH:MM:SS,mmm
        m = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", t)
        if not m:
            return 0.0
        h, mn, s, ms = map(int, m.groups())
        return h * 3600 + mn * 60 + s + ms / 1000.0

    content = srt_path.read_text(encoding="utf-8")
    # Split by double newlines to get cues
    cues = re.split(r"\n\s*\n", content.strip())
    segments = []
    for cue in cues:
        lines = [l.strip() for l in cue.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        # Find the timing line (contains -->)
        timing_line = None
        for line in lines:
            if "-->" in line:
                timing_line = line
                break
        if not timing_line:
            continue
        # Text is everything after the timing line
        idx = lines.index(timing_line)
        text = " ".join(lines[idx + 1:])
        if not text:
            continue
        start_str, end_str = timing_line.split("-->")
        start = _parse_time(start_str.strip())
        end = _parse_time(end_str.strip())
        words = text.split()
        if not words:
            continue
        cue_dur = max(0.001, end - start)
        word_dur = cue_dur / len(words)
        word_objs = []
        for i, w in enumerate(words):
            w_start = start + i * word_dur
            w_end = start + (i + 1) * word_dur
            # Always add trailing space so captacity doesn't merge words across cues
            word_objs.append({"word": w + " ", "start": w_start, "end": w_end})
        segments.append({"words": word_objs})
    return segments


def _resolved_llm_model(provider: str, gemini_model: str, openai_model: str, openrouter_model: str, openrouter_strategy: str) -> str:
    provider = (provider or "").lower()
    if provider == "gemini":
        return gemini_model
    if provider == "openai":
        return openai_model
    if provider == "openrouter":
        if openrouter_model == "auto (best quality)":
            return "auto"
        if openrouter_model == "auto (best free)":
            return "auto_free"
        if openrouter_model == "auto (strategy)":
            strategy = (openrouter_strategy or "Quality").strip().lower()
            if strategy == "low cost":
                return "auto_free"
            if strategy == "balanced":
                return "auto_balanced"
            return "auto"
        return openrouter_model
    return gemini_model


def _llm_status_line(provider: str, selected_model: str, openrouter_model: str = "", openrouter_strategy: str = "", actual_model_used: str = "") -> str:
    provider = (provider or "").lower()
    actual = (actual_model_used or "").strip()
    if provider == "openrouter":
        strategy = (openrouter_strategy or "Quality")
        if actual:
            return f"LLM route: provider=openrouter strategy={strategy} requested={openrouter_model} route={selected_model} used={actual}"
        return f"LLM route: provider=openrouter strategy={strategy} requested={openrouter_model} route={selected_model}"
    if actual:
        return f"LLM route: provider={provider} model={selected_model} used={actual}"
    return f"LLM route: provider={provider} model={selected_model}"


def run_pipeline(video_file, clips, min_words, max_words, model, openai_model, openrouter_model, fuzzy,
                 font_name, font_size, primary_hex, outline_w, outline_hex, subtitle_style, position, aspect,
                 ai_provider_name, gemini_api_key, openai_api_key, openrouter_api_key,
                 openrouter_strategy,
                 pre_pad, post_pad, min_duration, auto_size=False,
                 words_per_caption=3, vertical_offset=85, horizontal_align="center"):

    logger.info("run_pipeline called: video=%s clips=%s model=%s provider=%s subtitle_style=%s",
                str(video_file), clips, model, ai_provider_name, subtitle_style)

    # Validate selected provider has a key available either via input or env
    if ai_provider_name == "gemini":
        if not (gemini_api_key or os.getenv("GOOGLE_API_KEY")):
            return "Missing GOOGLE_API_KEY in .env or input", None, []
    elif ai_provider_name == "openai":
        if not (openai_api_key or os.getenv("OPENAI_API_KEY")):
            return "Missing OPENAI_API_KEY in .env or input", None, []
    elif ai_provider_name == "openrouter":
        if not (openrouter_api_key or os.getenv("OPENROUTER_API_KEY")):
            return "Missing OPENROUTER_API_KEY in .env or input", None, []

    if video_file is None:
        return "Please upload a video.", None, []

    src = Path(video_file)
    try:
        tjson = ensure_transcripts(REPO, src)
        transcript, words = read_timings(tjson)
    except Exception as e:
        logger.exception("Transcription/read_timings failed for %s", src)
        return f"Transcription error: {e}", None, []

    # pick API key override if provided
    api_key = None
    if ai_provider_name == "gemini":
        api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
    elif ai_provider_name == "openai":
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    elif ai_provider_name == "openrouter":
        api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")

    # Determine which model to use
    if ai_provider_name == "gemini":
        selected_model = model
    elif ai_provider_name == "openai":
        selected_model = openai_model
    elif ai_provider_name == "openrouter":
        strategy = (openrouter_strategy or "Quality").strip().lower()
        if openrouter_model == "auto (best quality)":
            selected_model = "auto"
        elif openrouter_model == "auto (best free)":
            selected_model = "auto_free"
        elif openrouter_model == "auto (strategy)":
            if strategy == "low cost":
                selected_model = "auto_free"
            elif strategy == "balanced":
                selected_model = "auto_balanced"
            else:
                selected_model = "auto"
        else:
            selected_model = openrouter_model
    else:
        selected_model = model
    
    logger.info("Using AI provider: %s with model: %s (raw openrouter_model=%s, openai_model=%s, gemini_model=%s)",
                ai_provider_name, selected_model, openrouter_model, openai_model, model)

    total_duration = get_video_duration(src)

    # Give the LLM broad narrative guidance derived from the actual video length.
    # This keeps the prompt transcript-aware instead of relying on fixed teaser-era bounds.
    suggested_params = _suggest_params(total_duration, len(words))
    requested_clips = max(1, int(clips))
    selection_min_words = max(int(min_words), int(suggested_params["min_words"]))
    selection_max_words = max(int(max_words), int(suggested_params["max_words"]), selection_min_words + 12)
    # Let the LLM decide how many strong clips the transcript supports, up to the user's request.
    llm_clip_count = requested_clips
    
    highs = call_llm(
        transcript,
        clips=llm_clip_count,
        min_words=selection_min_words,
        max_words=selection_max_words,
        model=selected_model,
        api_key=api_key,
        ai_provider_name=ai_provider_name,
        openai_model=selected_model
    )
    llm_meta = dict(highs.get("_llm_meta") or {}) if isinstance(highs, dict) else {}
    actual_model_used = llm_meta.get("model_used") or ""

    logger.info("Received %s highlights from AI provider", (len(highs.get('highlights')) if isinstance(highs, dict) and highs.get('highlights') else (len(highs) if isinstance(highs, list) else 'unknown')))

    tokens = tokens_from_words(words)
    segments = []
    for h in (highs.get('highlights', highs) if isinstance(highs, dict) else highs):
        excerpt = (h.get("excerpt") if isinstance(h, dict) else str(h))
        hit = align_excerpt_exact_or_fuzzy(excerpt, tokens, use_fuzzy=bool(fuzzy))
        if not hit:
            continue
        # Support either legacy hits of shape (a,b,s,e) or current (s,e).
        # Be defensive: prefer to treat hit as a sequence and take the last two
        # items as (start, end). Avoid unpacking failures for odd iterable types.
        s = e = None
        seq = None
        try:
            if not isinstance(hit, (str, bytes)):
                seq = list(hit)
        except Exception:
            seq = None

        if seq and len(seq) >= 2:
            s, e = seq[-2], seq[-1]
        else:
            # Fallback: try a direct unpack (will raise if not possible)
            try:
                s, e = hit
            except Exception:
                # Can't interpret hit -> skip
                continue
        segments.append({"title": (h.get("title") if isinstance(h, dict) else str(h)[:80])[:80], "excerpt": excerpt, "start": round(float(s),3), "end": round(float(e),3)})

    if not segments:
        alt_phrases = _effect_phrase_candidates(transcript, limit=3)
        return _llm_clip_feedback(requested_clips, 0, alt_phrases), None, []

    proc_path = REPO / "processed_content" / "input_processed.json"
    # Persist segments (pass the list of dicts directly)
    write_processed_segments(segments, proc_path)

    srt_dir = REPO / "segmented_videos" / "input" / "srt"
    # We'll compute adjusted segments first (applying padding / min_duration / auto_size)
    seg_dir = REPO / "segmented_videos" / "input"
    seg_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    adjusted_segments = []
    for idx, seg in enumerate(segments, start=1):
        # compute a heuristic score for each excerpt
        score = _score_excerpt(seg.get("title") or seg.get("excerpt", ""), seg.get("start"), seg.get("end"), total_duration, selection_min_words, selection_max_words)

        # If auto_size enabled, adjust padding and min_duration based on total video length
        pad_pre = float(pre_pad)
        pad_post = float(post_pad)
        min_dur = float(min_duration)
        try:
            if bool(auto_size):
                if total_duration <= 60:
                    pad_pre = max(0.2, pad_pre * 0.5)
                    pad_post = max(0.5, pad_post * 0.75)
                    min_dur = max(6.0, min_dur * 0.8)
                elif total_duration <= 300:
                    pad_pre = max(0.5, pad_pre)
                    pad_post = max(1.0, pad_post)
                    min_dur = max(8.0, min_dur)
                else:
                    pad_pre = max(1.0, pad_pre * 1.5)
                    pad_post = max(2.0, pad_post * 1.5)
                    min_dur = max(10.0, min_dur * 1.25)
        except Exception:
            pass

        # Expand segment by computed pre/post padding, clamp to video bounds, and
        # enforce a minimum duration so clips have context and aren't too short.
        start = max(0.0, float(seg["start"]) - pad_pre)
        end = float(seg["end"]) + pad_post
        if end - start < min_dur:
            mid = (float(seg["start"]) + float(seg["end"])) / 2.0
            half = min_dur / 2.0
            start = max(0.0, mid - half)
            end = mid + half

        if total_duration > 0:
            start = min(start, max(0.0, total_duration - 0.2))
            end = min(end, total_duration)
        if end - start < 0.2:
            logger.warning("Skipping invalid segment window idx=%s start=%.3f end=%.3f total=%.3f", idx, start, end, total_duration)
            continue

        adjusted_segments.append({"idx": idx, "title": seg.get("title"), "start": round(start, 3), "end": round(end, 3), "score": score})
    logger.info("Adjusted segment %s: start=%s end=%s score=%s", idx, start, end, score)

    if not adjusted_segments:
        alt_phrases = _effect_phrase_candidates(transcript, limit=3)
        return _llm_clip_feedback(requested_clips, 0, alt_phrases), None, []

    # Persist adjusted segments so downstream tools see the final timings
    write_processed_segments([{"title": s["title"], "start": s["start"], "end": s["end"]} for s in adjusted_segments], proc_path)

    # Refine and diversify clips for final output quality.
    adjusted_segments = _refine_clip_candidates(adjusted_segments, llm_clip_count, total_duration)
    for new_idx, seg in enumerate(adjusted_segments, start=1):
        seg["idx"] = new_idx

    # Build SRTs for adjusted segments (build_srts uses start/end to produce relative SRT cue times)
    build_srts(
        [{"title": s["title"], "start": s["start"], "end": s["end"]} for s in adjusted_segments],
        words,
        srt_dir,
        words_per_caption=int(words_per_caption)
    )

    # Now run ffmpeg cut + captioning per adjusted segment
    scored_segments = []
    for s in adjusted_segments:
        idx = s["idx"]
        title = safe_name(s["title"])
        raw_out = seg_dir / f"segment_{idx}_{title}.mp4"
        sub_out = seg_dir / f"segment_{idx}_{title}_subtitled.mp4"
        srt = srt_dir / f"segment_{idx}.srt"

        # perform the cut
        logger.info("Cutting segment %s -> %s (%.3f-%.3f)", idx, raw_out, float(s['start']), float(s['end']))
        try:
            ffmpeg_cut(REPO / "input.mp4", float(s["start"]), float(s["end"]), raw_out)
        except Exception:
            logger.exception("ffmpeg_cut failed for segment %s", idx)
            raise

        # Use ASS/ffmpeg rendering as the single source of truth for subtitle style.
        # Normalize numeric and color inputs
        try:
            f_size = int(font_size)
        except Exception:
            f_size = 28
        try:
            s_w = int(outline_w)
        except Exception:
            s_w = 3
        f_primary = primary_hex or "#FFFFFF"
        f_outline = outline_hex or "#000000"

        logger.info("Burning subtitles for segment %s via ffmpeg ASS", idx)
        target_width = 1080 if aspect in ("9:16", "4:5", "1:1") else _get_video_width(raw_out)
        font_path = _resolve_font_path(font_name)
        style = _build_ass_style(
            font_name=font_name,
            font_size=f_size,
            primary_hex=f_primary,
            outline_w=s_w,
            outline_hex=f_outline,
            subtitle_style=subtitle_style,
            position=position,
            vertical_offset=int(vertical_offset),
            horizontal_align=horizontal_align,
            aspect=aspect,
            target_width=target_width,
            font_path=font_path,
        )
        logger.info("Calling ffmpeg_burn_subs for segment %s with style %s", idx, {k: style[k] for k in ('FontName','FontSize','Alignment','MarginV')})
        try:
            ffmpeg_burn_subs(raw_out, srt, sub_out, aspect=aspect, style=style, fonts_dir=(Path(font_path).parent if font_path else None))
        except Exception:
            logger.exception("ffmpeg_burn_subs failed for segment %s", idx)
            raise

        outputs.append(str(sub_out))
        scored_segments.append({"idx": s["idx"], "title": s["title"], "score": s["score"], "path": str(sub_out)})

    # Build a status summary with scores
    status_lines = [
        _llm_status_line(ai_provider_name, selected_model, openrouter_model, openrouter_strategy, actual_model_used),
        f"LLM used: provider={ai_provider_name} model={selected_model} candidate_target={llm_clip_count} aligned={len(segments)}",
        f"Done! Generated {len(outputs)} clips.",
        _llm_clip_feedback(requested_clips, len(outputs)),
    ]
    for s in scored_segments:
        status_lines.append(f"#{s['idx']}: {s['title'][:60]} — score {s['score']}")
    status = "\n".join(status_lines)
    logger.info("Generation complete: %s", status.replace('\n', ' | '))
    return status, (outputs[0] if outputs else None), outputs

def _position_to_ass_alignment(position: str, vertical_offset: int, horizontal_align: str) -> tuple:
    """Map UI position controls to ASS Alignment value and MarginV.

    ASS Alignment values:
      1=bottom-left, 2=bottom-center, 3=bottom-right
      4=middle-left, 5=middle-center, 6=middle-right
      7=top-left,    8=top-center,    9=top-right
    """
    # The selected menu position is the primary source of truth.
    # Vertical offset only nudges the caption away from the edge.
    if position == "top":
        row = 7  # top
    elif position == "center":
        row = 4  # middle
    else:
        row = 1  # bottom

    # Horizontal alignment
    if horizontal_align == "left":
        col = 0
    elif horizontal_align == "right":
        col = 2
    else:
        col = 1

    alignment = row + col

    vo = int(max(0, min(100, vertical_offset)))
    if row in (7, 1):
        # Keep captions close to the edge. Higher offsets increase separation a bit,
        # but never enough to push the subtitle into the middle of the frame.
        margin_v = int(18 + ((100 - vo) / 100.0) * 42)
    else:
        # Center captions should sit around the middle without large edge gaps.
        margin_v = int(12 + (abs(vo - 50) / 100.0) * 18)

    margin_v = max(8, min(margin_v, 80))
    return alignment, margin_v

with gr.Blocks(title="Clipify — AI Highlights") as demo:
    gr.Markdown("## Clipify — AI Highlights (Gemini)")

    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.File(label="Upload video", file_types=[".mp4", ".mov", ".m4v"])

            gr.Markdown("### AI Provider")
            with gr.Group():
                ai_provider_name = gr.Radio(
                    ["gemini", "openai", "openrouter"],
                    value=os.getenv("AI_PROVIDER", "gemini"),
                    label="Provider"
                )
                # API keys (only relevant one shown per provider)
                gemini_api_key = gr.Textbox(value=os.getenv("GOOGLE_API_KEY", ""), label="Gemini API Key", type="password")
                openai_api_key = gr.Textbox(value=os.getenv("OPENAI_API_KEY", ""), label="OpenAI API Key", type="password", visible=False)
                openrouter_api_key = gr.Textbox(value=os.getenv("OPENROUTER_API_KEY", ""), label="OpenRouter API Key", type="password", visible=False)
                # Model dropdowns (only relevant one shown per provider)
                model = gr.Dropdown(
                    ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
                    value=os.getenv('GEMINI_MODEL', 'gemini-1.5-flash'),
                    label="Gemini Model"
                )
                openai_model = gr.Dropdown(
                    ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"],
                    value=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                    label="OpenAI Model",
                    visible=False
                )
                openrouter_model = gr.Dropdown(
                    [
                        "auto (strategy)",
                        "auto (best quality)",
                        "openai/o3",
                        "openai/gpt-4.1",
                        "anthropic/claude-sonnet-4",
                        "google/gemini-2.5-pro",
                        "openai/gpt-4o",
                        "auto (best free)",
                        "openai/gpt-4o-mini",
                        "anthropic/claude-3.7-sonnet",
                        "anthropic/claude-3.5-sonnet",
                        "deepseek/deepseek-chat",
                        "meta-llama/llama-3.1-70b-instruct",
                        "qwen/qwen-2.5-72b-instruct",
                    ],
                    value=os.getenv('OPENROUTER_MODEL', 'auto (strategy)'),
                    label="OpenRouter Model",
                    visible=False
                )
                openrouter_strategy = gr.Dropdown(
                    ["Quality", "Balanced", "Low cost"],
                    value=os.getenv('OPENROUTER_STRATEGY', 'Quality'),
                    label="OpenRouter Strategy",
                    visible=False,
                )

            gr.Markdown("### Clip Settings")
            with gr.Group():
                clips      = gr.Slider(1, 16, value=3, step=1, label="How many clips")
                min_words  = gr.Slider(5, 240, value=18, step=1, label="Min words per excerpt")
                max_words  = gr.Slider(12, 400, value=48, step=1, label="Max words per excerpt")
                pre_pad    = gr.Slider(0.0, 5.0, value=0.5, step=0.1, label="Pre-pad (seconds)")
                post_pad   = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="Post-pad (seconds)")
                min_duration = gr.Slider(2.0, 30.0, value=15.0, step=0.5, label="Min clip duration (s)")

                auto_suggest_btn = gr.Button("✨ Auto-suggest params", size="sm")
                llm_recommended_note = gr.Markdown("LLM-recommended values will appear here after auto-suggest.")

                fuzzy      = gr.Checkbox(value=True, label="Use fuzzy aligner")

            from ui.style_preview import StylePreview
            style_preview = StylePreview(FONTS_MAP, FONTS_DEFAULT)
            
            gr.Markdown("### Subtitle Style")
            with gr.Group():
                with gr.Row():
                    style_preset = gr.Dropdown(
                        choices=list(STYLE_PRESETS.keys()),
                        value="Hype Clean",
                        label="Style Preset"
                    )
                    apply_preset_btn = gr.Button("Apply Preset", size="sm")

                with gr.Row():
                    with gr.Column():
                        font_name = gr.Dropdown(
                            FONTS_LIST or [FONTS_DEFAULT],
                            value=FONTS_DEFAULT,
                            label="Font Family"
                        )
                        font_size = gr.Slider(6, 60, value=28, step=1, label="Font Size")
                        words_per_caption = gr.Slider(1, 10, value=3, step=1, label="Words per caption")
                        primary_hex = gr.ColorPicker(value="#FFFFFF", label="Font Color")
                    with gr.Column():
                        subtitle_style = gr.Radio(
                            ["Box", "Outline", "None"],
                            value="Box",
                            label="Subtitle Style"
                        )
                        outline_w = gr.Slider(0, 12, value=3, step=1, label="Outline/Box Width",
                                            visible=True)
                        outline_hex = gr.ColorPicker(value="#000000", label="Outline/Box Color",
                                               visible=True)
                
                with gr.Row():
                    position = gr.Radio(
                        ["bottom", "center", "top"],
                        value="bottom",
                        label="Position"
                    )
                    aspect = gr.Radio(
                        ["source", "9:16", "4:5", "1:1"],
                        value="source",
                        label="Aspect Ratio"
                    )
                
                with gr.Row():
                    vertical_offset = gr.Slider(
                        0, 100, value=85, step=1,
                        label="Vertical offset (% from top)"
                    )
                    horizontal_align = gr.Radio(
                        ["left", "center", "right"],
                        value="center",
                        label="Horizontal alignment"
                    )
                
                # Get fonts
                fonts_list, fonts_map, fonts_meta = _gather_fonts(REPO)
                default_font = next(iter(fonts_list))
                
                # Preview area
                preview_image = gr.Image(
                    label="Style Preview",
                    height=354,
                    width=200
                )
                preview_style_btn = gr.Button("Render 3s Style Preview", size="sm")
                preview_style_video = gr.Video(label="Rendered 3s Style Preview")
                preview_style_status = gr.Textbox(label="Style Preview Status", interactive=False)
                
                # Initialize StylePreview (avoid name collision with Gradio `preview` component)
                style_preview_instance = StylePreview(fonts_map=fonts_map, fonts_default=default_font)
                
                # Update preview when settings change
                style_inputs = [font_name, font_size, primary_hex, outline_w,
                              outline_hex, subtitle_style]
                
                preview = StylePreview(fonts_map=fonts_map, fonts_default=default_font)

                def apply_style_preset(preset_name):
                    preset = STYLE_PRESETS.get(preset_name or "", STYLE_PRESETS["Hype Clean"])
                    return (
                        gr.update(value=preset["font_size"]),
                        gr.update(value=preset["primary_hex"]),
                        gr.update(value=preset["outline_w"]),
                        gr.update(value=preset["outline_hex"]),
                        gr.update(value=preset["subtitle_style"]),
                        gr.update(value=preset["words_per_caption"]),
                        gr.update(value=preset["position"]),
                        gr.update(value=preset["vertical_offset"]),
                        gr.update(value=preset["horizontal_align"]),
                    )

                def apply_aspect_calibration(aspect_value, current_font_size, current_outline_w, current_vertical_offset):
                    fs, ow, vo = _calibrate_style_for_aspect(
                        int(current_font_size),
                        int(current_outline_w),
                        int(current_vertical_offset),
                        str(aspect_value),
                    )
                    return gr.update(value=fs), gr.update(value=ow), gr.update(value=vo)
                
                def update_preview(f, s, p, w, o, st):
                    try:
                        if not all([f, s, p]):
                            return None

                        # Normalize color inputs: accept '#RRGGBB', 'rgb(...)', 'rgba(...)' or tuples
                        def normalize_color(col):
                            if not col:
                                return "#FFFFFF"
                            if isinstance(col, (list, tuple)) and len(col) >= 3:
                                r, g, b = int(col[0]), int(col[1]), int(col[2])
                                return f"#{r:02X}{g:02X}{b:02X}"
                            if isinstance(col, str):
                                c = col.strip()
                                # hex already
                                if c.startswith('#') and (len(c) == 7 or len(c) == 4):
                                    # Expand short hex like #abc
                                    if len(c) == 4:
                                        r, g, b = c[1], c[2], c[3]
                                        return f"#{r}{r}{g}{g}{b}{b}".upper()
                                    return c.upper()
                                # rgb/rgba formats
                                if c.startswith('rgba') or c.startswith('rgb'):
                                    import re
                                    m = re.search(r"([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)", c)
                                    if m:
                                        r = int(round(float(m.group(1))))
                                        g = int(round(float(m.group(2))))
                                        b = int(round(float(m.group(3))))
                                        return f"#{r:02X}{g:02X}{b:02X}"
                            # fallback
                            return "#FFFFFF"

                        primary = normalize_color(p)
                        outline = normalize_color(o) if o else "#000000"

                        return style_preview_instance.update_subtitle_preview(
                            font_name=f,
                            font_size=int(s) if s else 48,
                            primary_hex=primary,
                            outline_w=float(w) if w else 0,
                            outline_hex=outline,
                            subtitle_style=st
                        )
                    except Exception as e:
                        print(f"Error generating preview: {e}")
                        return None
                
                for input_elem in style_inputs:
                    input_elem.change(
                        fn=update_preview,
                        inputs=style_inputs,
                        outputs=[preview_image]
                    )

                # Also refresh static preview when positioning/aspect changes
                for input_elem in [position, aspect, vertical_offset, horizontal_align]:
                    input_elem.change(
                        fn=update_preview,
                        inputs=style_inputs,
                        outputs=[preview_image]
                    )

                apply_preset_btn.click(
                    fn=apply_style_preset,
                    inputs=[style_preset],
                    outputs=[font_size, primary_hex, outline_w, outline_hex, subtitle_style,
                             words_per_caption, position, vertical_offset, horizontal_align]
                )

                aspect.change(
                    fn=apply_aspect_calibration,
                    inputs=[aspect, font_size, outline_w, vertical_offset],
                    outputs=[font_size, outline_w, vertical_offset]
                )

                preview_style_btn.click(
                    fn=_render_style_preview_clip,
                    inputs=[video_in, font_name, font_size, primary_hex, outline_w, outline_hex,
                            subtitle_style, position, aspect, vertical_offset, horizontal_align],
                    outputs=[preview_style_video, preview_style_status]
                )

                
                # Update control visibility based on style
                subtitle_style.change(
                    fn=style_preview_instance.update_style_controls,
                    inputs=[subtitle_style],
                    outputs=[outline_w, outline_hex]
                )
                


            
            auto_size = gr.Checkbox(value=True, label="Auto-size clips by video length")

            run = gr.Button("Generate")

        with gr.Column(scale=2):
            status = gr.Textbox(label="Status")
            preview = gr.Video(label="Preview (first clip)")
            files = gr.Files(label="All output clips")
            download_all_btn = gr.Button("Download all clips (ZIP)")
            download_zip_file = gr.File(label="Download ZIP")
            preview_table = gr.JSON(label="Computed segments (start/end/score)")

            def download_all_clips(files_list):
                zip_file = _create_clips_zip(files_list)
                if not zip_file:
                    return None, "No generated clips available to bundle."
                return zip_file, f"ZIP ready: {Path(zip_file).name}"

            download_all_btn.click(
                download_all_clips,
                inputs=[files],
                outputs=[download_zip_file, status]
            )

            # Toggle visibility of API key and model fields based on selected provider
            def _provider_visibility(provider):
                provider = (provider or "").lower()
                if provider == "gemini":
                    return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False))
                elif provider == "openai":
                    return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                        gr.update(visible=False))
                elif provider == "openrouter":
                    return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                        gr.update(visible=True))
                else:
                    return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False))

            ai_provider_name.change(_provider_visibility, inputs=[ai_provider_name],
                outputs=[gemini_api_key, openai_api_key, openrouter_api_key, model, openai_model, openrouter_model, openrouter_strategy])

            # Auto-suggest parameters based on uploaded video
            def auto_suggest(video_file):
                if not video_file:
                    return [gr.update()] * 6 + [gr.update(value="LLM-recommended values will appear here after auto-suggest.")]
                try:
                    src = Path(video_file)
                    tjson = ensure_transcripts(REPO, src)
                    _, words = read_timings(tjson)
                    duration = get_video_duration(src)
                    p = _suggest_params(duration, len(words))
                    updates = _ui_updates_from_llm_params(p)
                    updates.append(gr.update(value=(
                        f"LLM-recommended: {p['clips']} clips, "
                        f"{p['min_words']} to {p['max_words']} words per excerpt, "
                        f"{p['min_duration']:.1f}s min clip duration"
                    )))
                    return updates
                except Exception as e:
                    logger.warning("Auto-suggest failed: %s", e)
                    return [gr.update()] * 6 + [gr.update(value="LLM-recommended values could not be computed.")]

            auto_suggest_btn.click(
                auto_suggest,
                inputs=[video_in],
                outputs=[clips, min_words, max_words, pre_pad, post_pad, min_duration, llm_recommended_note]
            )
            # Also trigger when a video is uploaded
            video_in.change(
                auto_suggest,
                inputs=[video_in],
                outputs=[clips, min_words, max_words, pre_pad, post_pad, min_duration, llm_recommended_note]
            )

            # Two-step process: preview first, then generate
            def preview_clicked(video_file, clips, min_words, max_words, model, openai_model, openrouter_model, fuzzy,
                           pre_pad, post_pad, min_duration, auto_size,
                           ai_provider_name, gemini_api_key, openai_api_key, openrouter_api_key,
                           openrouter_strategy,
                           font_name, font_size, primary_hex, outline_w, outline_hex, subtitle_style,
                           position, aspect, words_per_caption, vertical_offset, horizontal_align):
                requested_clips = max(1, int(clips or 1))
                preview_result = preview_segments(video_file, clips, min_words, max_words, model, openai_model, openrouter_model, fuzzy,
                                               pre_pad, post_pad, min_duration, auto_size,
                                               ai_provider_name, gemini_api_key, openai_api_key, openrouter_api_key,
                                               openrouter_strategy, True)

                # Error handling: return status text and keep Generate hidden
                if isinstance(preview_result, dict) and "error" in preview_result:
                    return f"Error: {preview_result['error']}", None, gr.update(visible=False), gr.update(), gr.update()

                # If preview_result is a plain string, show it as status
                if isinstance(preview_result, str):
                    return preview_result, None, gr.update(visible=False), gr.update(), gr.update()

                # Preview table-only mode returns a list of segments.
                preview_segments_list = preview_result
                preview_llm_meta = {}
                preview_alt_phrases = []
                if isinstance(preview_result, dict) and isinstance(preview_result.get("segments"), list):
                    preview_segments_list = preview_result.get("segments")
                    preview_llm_meta = dict(preview_result.get("_llm_meta") or {})
                    preview_alt_phrases = list(preview_result.get("_alt_phrases") or [])

                if isinstance(preview_segments_list, list):
                    recommended = len(preview_segments_list)
                    note_msg = _llm_clip_feedback(requested_clips, recommended, preview_alt_phrases)
                    if not preview_segments_list:
                        return note_msg, None, gr.update(visible=False), gr.update(value=max(1, recommended)), gr.update(value=note_msg)
                    try:
                        src = Path(video_file)
                        tjson = ensure_transcripts(REPO, src)
                        _, words = read_timings(tjson)
                        selected_model = _resolved_llm_model(
                            ai_provider_name,
                            model,
                            openai_model,
                            openrouter_model,
                            openrouter_strategy,
                        )
                        llm_line = _llm_status_line(ai_provider_name, selected_model, openrouter_model, openrouter_strategy)
                        actual_model_used = preview_llm_meta.get("model_used") or ""
                        if actual_model_used:
                            llm_line = _llm_status_line(ai_provider_name, selected_model, openrouter_model, openrouter_strategy, actual_model_used)
                        preview_clip = _render_first_segment_preview(
                            video_file=video_file,
                            segment=preview_segments_list[0],
                            words=words,
                            font_name=font_name,
                            font_size=int(font_size),
                            primary_hex=primary_hex,
                            outline_w=int(outline_w),
                            outline_hex=outline_hex,
                            subtitle_style=subtitle_style,
                            position=position,
                            aspect=aspect,
                            vertical_offset=int(vertical_offset),
                            horizontal_align=horizontal_align,
                            words_per_caption=int(words_per_caption),
                        )
                        status_msg = f"{llm_line}\nPreview ready — {recommended} segments\n{note_msg}"
                        return status_msg, preview_clip, gr.update(visible=True), gr.update(value=max(1, recommended)), gr.update(value=note_msg)
                    except Exception as e:
                        logger.exception("Failed to render main preview clip")
                        return f"Preview failed while rendering subtitle clip: {e}", None, gr.update(visible=False), gr.update(), gr.update()

                # If preview_result is a dict with computed values, show preview and enable Generate
                preview_clip = None
                preview_files = None
                preview_table_data = None
                if isinstance(preview_result, dict):
                    preview_clip = preview_result.get('preview_clip') if preview_result.get('preview_clip') and os.path.isfile(preview_result.get('preview_clip')) else None
                    preview_files = [f for f in preview_result.get('files', []) if os.path.isfile(f)]
                    preview_table_data = preview_result.get('segments')

                selected_model = _resolved_llm_model(
                    ai_provider_name,
                    model,
                    openai_model,
                    openrouter_model,
                    openrouter_strategy,
                )
                llm_line = _llm_status_line(ai_provider_name, selected_model, openrouter_model, openrouter_strategy)
                final_count = len(preview_table_data) if preview_table_data else (len(preview_files) if preview_files else 0)
                note_msg = _llm_clip_feedback(requested_clips, final_count)
                status_text = f"{llm_line}\nPreview ready — {final_count} segments\n{note_msg}"
                return status_text, preview_clip, gr.update(visible=True), gr.update(value=max(1, final_count)), gr.update(value=note_msg)
                
            # Separate generate function that uses the last preview results
            def generate_clicked(video_file, clips, min_words, max_words, model, openai_model, openrouter_model, fuzzy,
                            font_name, font_size, primary_hex, outline_w, outline_hex, subtitle_style, position, aspect,
                            ai_provider_name, gemini_api_key, openai_api_key, openrouter_api_key,
                            openrouter_strategy,
                            pre_pad, post_pad, min_duration, auto_size,
                            words_per_caption, vertical_offset, horizontal_align):
                status_msg, preview_clip, out_files = run_pipeline(video_file, clips, min_words, max_words, model, openai_model, openrouter_model, fuzzy,
                                font_name, font_size, primary_hex, outline_w, outline_hex, subtitle_style, position, aspect,
                                ai_provider_name, gemini_api_key, openai_api_key, openrouter_api_key,
                                openrouter_strategy,
                                pre_pad, post_pad, min_duration, auto_size,
                                words_per_caption, vertical_offset, horizontal_align)
                requested_clips = max(1, int(clips or 1))
                final_count = len(out_files or [])
                note_msg = _llm_clip_feedback(requested_clips, final_count)
                return status_msg, preview_clip, out_files, gr.update(value=max(1, final_count)), gr.update(value=note_msg)
            
            preview_btn = gr.Button("Preview")
            generate_btn = gr.Button("Generate", visible=False)
            
            preview_btn.click(
                preview_clicked,
                inputs=[video_in, clips, min_words, max_words, model, openai_model, openrouter_model, fuzzy,
                        pre_pad, post_pad, min_duration, auto_size,
                    ai_provider_name, gemini_api_key, openai_api_key, openrouter_api_key, openrouter_strategy,
                        font_name, font_size, primary_hex, outline_w, outline_hex, subtitle_style,
                        position, aspect, words_per_caption, vertical_offset, horizontal_align],
                outputs=[status, preview, generate_btn, clips, llm_recommended_note]
            )
            
            generate_btn.click(
                generate_clicked,
                inputs=[video_in, clips, min_words, max_words, model, openai_model, openrouter_model, fuzzy,
                    font_name, font_size, primary_hex, outline_w, outline_hex, subtitle_style, position, aspect,
                    ai_provider_name, gemini_api_key, openai_api_key, openrouter_api_key, openrouter_strategy,
                    pre_pad, post_pad, min_duration, auto_size,
                    words_per_caption, vertical_offset, horizontal_align],
                outputs=[status, preview, files, clips, llm_recommended_note]
            )

            # Preview button: compute segments and scores without running ffmpeg
            def preview_segments(video_file, clips, min_words, max_words, model, openai_model, openrouter_model, fuzzy,
                                 pre_pad, post_pad, min_duration, auto_size,
                                 ai_provider_name="gemini", gemini_api_key="", openai_api_key="", openrouter_api_key="",
                                 openrouter_strategy="Quality", include_meta=False):
                if not video_file:
                    return {"error": "No video provided"}
                # Reuse pipeline logic but avoid heavy processing: call into call_gemini and alignment
                try:
                    src = Path(video_file)
                    tjson = ensure_transcripts(REPO, src)
                    transcript, words = read_timings(tjson)
                except Exception as e:
                    return {"error": f"Transcription error: {e}"}

                # Resolve API key and model for the selected provider
                api_key = None
                selected_model = model
                if ai_provider_name == "gemini":
                    api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
                    selected_model = model
                elif ai_provider_name == "openai":
                    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
                    selected_model = openai_model
                elif ai_provider_name == "openrouter":
                    api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
                    if openrouter_model == "auto (best quality)":
                        selected_model = "auto"
                    elif openrouter_model == "auto (best free)":
                        selected_model = "auto_free"
                    elif openrouter_model == "auto (strategy)":
                        strategy = (openrouter_strategy or "Quality").strip().lower()
                        if strategy == "low cost":
                            selected_model = "auto_free"
                        elif strategy == "balanced":
                            selected_model = "auto_balanced"
                        else:
                            selected_model = "auto"
                    else:
                        selected_model = openrouter_model

                total_duration = get_video_duration(src)
                suggested_params = _suggest_params(total_duration, len(words))
                requested_clips = max(1, int(clips))
                selection_min_words = max(int(min_words), int(suggested_params["min_words"]))
                selection_max_words = max(int(max_words), int(suggested_params["max_words"]), selection_min_words + 12)
                llm_clip_count = requested_clips

                highs = call_llm(transcript, clips=llm_clip_count, min_words=selection_min_words, max_words=selection_max_words,
                                    model=selected_model, api_key=api_key,
                                    ai_provider_name=ai_provider_name, openai_model=selected_model)
                llm_meta = dict(highs.get("_llm_meta") or {}) if isinstance(highs, dict) else {}
                tokens = tokens_from_words(words)
                out = []
                for idx, h in enumerate((highs.get('highlights', highs) if isinstance(highs, dict) else highs), start=1):
                    hit = align_excerpt_exact_or_fuzzy((h.get("excerpt") if isinstance(h, dict) else str(h)), tokens, use_fuzzy=bool(fuzzy))
                    if not hit:
                        continue
                    # interpret hit as seq or pair
                    s = e = None
                    try:
                        seq = list(hit) if not isinstance(hit, (str, bytes)) else None
                    except Exception:
                        seq = None
                    if seq and len(seq) >= 2:
                        s, e = seq[-2], seq[-1]
                    else:
                        try:
                            s, e = hit
                        except Exception:
                            continue

                    # apply auto_size/pad/min rules (same as pipeline)
                    pad_pre = float(pre_pad)
                    pad_post = float(post_pad)
                    min_dur = float(min_duration)
                    try:
                        if bool(auto_size):
                            if total_duration <= 60:
                                pad_pre = max(0.2, pad_pre * 0.5)
                                pad_post = max(0.5, pad_post * 0.75)
                                min_dur = max(6.0, min_dur * 0.8)
                            elif total_duration <= 300:
                                pad_pre = max(0.5, pad_pre)
                                pad_post = max(1.0, pad_post)
                                min_dur = max(8.0, min_dur)
                            else:
                                pad_pre = max(1.0, pad_pre * 1.5)
                                pad_post = max(2.0, pad_post * 1.5)
                                min_dur = max(10.0, min_dur * 1.25)
                    except Exception:
                        pass

                    start = max(0.0, float(s) - pad_pre)
                    end = float(e) + pad_post
                    if end - start < min_dur:
                        mid = (float(s) + float(e)) / 2.0
                        half = min_dur / 2.0
                        start = max(0.0, mid - half)
                        end = mid + half

                    if total_duration > 0:
                        start = min(start, max(0.0, total_duration - 0.2))
                        end = min(end, total_duration)
                    if end - start < 0.2:
                        continue

                    excerpt = h.get('excerpt') if isinstance(h, dict) else str(h)
                    title = (h.get('title') if isinstance(h, dict) else str(h))[:80]
                    score = _score_excerpt(excerpt or title, s, e, total_duration, selection_min_words, selection_max_words)
                    out.append({"idx": idx, "title": title, "excerpt": excerpt, "start": round(start,3), "end": round(end,3), "score": score})

                out = _refine_clip_candidates(out, llm_clip_count, total_duration)
                for new_idx, seg in enumerate(out, start=1):
                    seg["idx"] = new_idx
                if include_meta:
                    return {
                        "segments": out,
                        "_llm_meta": llm_meta,
                        "_requested_clips": requested_clips,
                        "_recommended_clips": len(out),
                        "_alt_phrases": (_effect_phrase_candidates(transcript, limit=3) if not out else []),
                    }
                return out

            preview_segments_btn = gr.Button("Preview segments")
            preview_segments_btn.click(preview_segments, inputs=[video_in, clips, min_words, max_words, model, openai_model, openrouter_model, fuzzy,
                                                                 pre_pad, post_pad, min_duration, auto_size,
                                                                 ai_provider_name, gemini_api_key, openai_api_key, openrouter_api_key, openrouter_strategy], outputs=[preview_table])

if __name__ == "__main__":
    # Allow Gradio to serve files created in these output directories
    allowed = [
        str(REPO / "segmented_videos" / "input"),
        str(REPO / "segmented_videos" / "input" / "srt"),
        str(REPO / "processed_content")
    ]
    demo.launch(server_name="127.0.0.1", server_port=7860, allowed_paths=allowed)