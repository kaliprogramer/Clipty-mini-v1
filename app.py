"""
Clipty-Mini: Production-Grade AI Video Clipping Studio.
Dynamic model selection (Whisper & Ollama), customizable clip duration/count,
aspect-ratio transformation (Original, 9:16 Blurred, 9:16 Crop),
batch export, and ZIP bundling.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import uuid

import gradio as gr

from core.config import (
    ClipSelectionConfig,
    ExportConfig,
    LayoutMode,
    TranscriptionConfig,
)
from core.clip_manager import select_best_moments
from core.ollama_service import check_ollama_status, get_available_models
from core.transcriber import WhisperModelHandler
from core.video_processor import VideoHandler, create_clips_zip
from helper.logger import logger


# ==========================================
# CONSTANTS & PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "storage" / "video_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
PREFERRED_OLLAMA_MODEL = "qwen2.5:7b"


# ==========================================
# SAFE HELPERS
# ==========================================

def _safe_float(value, fallback: float = 0.0) -> float:
    try:
        return fallback if value is None else float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_int(value, fallback: int = 0) -> int:
    try:
        return fallback if value is None else int(value)
    except (TypeError, ValueError):
        return fallback


def _pick_default_model(models: List[str]) -> Optional[str]:
    if not models:
        return None
    return PREFERRED_OLLAMA_MODEL if PREFERRED_OLLAMA_MODEL in models else models[0]


def _load_initial_models() -> List[str]:
    try:
        return get_available_models() or []
    except Exception as exc:
        logger.warning(f"Could not preload Ollama models: {exc}")
        return []


def get_video_path(video_input) -> Path:
    if not video_input:
        raise gr.Error("Please upload a video file first.")
    if isinstance(video_input, dict):
        video_input = video_input.get("path") or video_input.get("name")
    if not video_input:
        raise gr.Error("Could not determine uploaded video path.")
    path = Path(str(video_input)).expanduser().resolve()
    if not path.is_file():
        raise gr.Error("Uploaded video file was not found on disk.")
    return path


# ==========================================
# 1. OLLAMA STATUS & MODEL REFRESH
# ==========================================

def refresh_ollama_models(host: str):
    host = (host or DEFAULT_OLLAMA_HOST).strip() or DEFAULT_OLLAMA_HOST
    try:
        is_online, message, models = check_ollama_status(host)
    except Exception as exc:
        logger.error(f"Ollama status check failed: {exc}", exc_info=True)
        return (
            gr.update(choices=[], value=None),
            f"🔴 **Ollama unreachable** — `{exc}`. Start `ollama serve` and retry.",
        )
    models = models or []
    default_model = _pick_default_model(models)
    if is_online and models:
        return (
            gr.update(choices=models, value=default_model),
            f"🟢 **Connected** — {len(models)} model(s) available on `{host}`.",
        )
    return (
        gr.update(choices=models, value=default_model),
        message or "⚠️ Ollama is offline. Start it and click **Refresh**.",
    )


# ==========================================
# 2. PRESET HANDLERS
# ==========================================

def apply_preset(preset_name: str) -> Tuple[float, float, int, int]:
    presets = {
        "shorts":   (15.0, 60.0,  5, 7),
        "podcast":  (30.0, 90.0,  6, 7),
        "snippet":  (10.0, 30.0,  8, 6),
        "extended": (60.0, 180.0, 4, 7),
    }
    return presets.get(preset_name, (15.0, 60.0, 5, 7))


# ==========================================
# 3. MAIN VIDEO ANALYSIS PIPELINE
# ==========================================

def process_video_pipeline(
    video_input, whisper_model: str, device: str,
    ollama_model: str, ollama_host: str,
    min_clip_dur: float, max_clip_dur: float,
    max_clips: int, min_score: int,
    progress=gr.Progress(),
):
    if not video_input:
        return "", [], [], [], "⚠️ Please upload a video first.", ""
    if not ollama_model:
        return ("", [], [], [],
                "⚠️ No Ollama model selected — click **Refresh** in the Model panel.", "")

    try:
        progress(0.05, desc="Validating video file...")
        video_path = get_video_path(video_input)

        progress(0.15, desc="Probing technical video metadata...")
        video = VideoHandler(video_path=str(video_path))
        info = video.info()
        if not info:
            raise gr.Error("Could not read video streams. Ensure the file is a valid video.")

        duration_s = _safe_float(info.get("duration"), 0.0)
        metadata_md = (
            "### 📋 Technical Specifications\n\n"
            "| Property | Value |\n"
            "| :--- | :--- |\n"
            f"| **Resolution** | `{info.get('resolution', '—')}` ({info.get('aspect_ratio', '—')}) |\n"
            f"| **Duration** | `{info.get('duration_formatted', '—')}` ({duration_s:.1f}s) |\n"
            f"| **Frame Rate** | `{info.get('fps', '—')} fps` |\n"
            f"| **Video Codec** | `{info.get('codec', '—')}` |\n"
            f"| **File Size** | `{info.get('file_size_mb', '—')} MB` |\n"
            f"| **Bitrate** | `{info.get('bitrate_kbps', '—')} kbps` |"
        )

        progress(0.30, desc=f"Transcribing audio with Whisper [{whisper_model}]...")
        t_config = TranscriptionConfig(
            model_size=whisper_model, device=device,
            compute_type="auto", vad_filter=True,
        )
        transcriber = WhisperModelHandler(path=str(video_path), config=t_config)
        segments = transcriber.transcribe() or []

        if not segments:
            return (metadata_md, [], [], [],
                    "⚠️ No audible speech detected.",
                    "AI moment selection requires spoken dialogue.")

        transcript_table = [
            [
                f"{_safe_float(s.get('start')):.2f}s",
                f"{_safe_float(s.get('end')):.2f}s",
                (s.get("text") or "").strip(),
            ]
            for s in segments
        ]

        progress(0.70, desc=f"Evaluating viral hooks with AI [{ollama_model}]...")
        clip_config = ClipSelectionConfig(
            ollama_model=ollama_model,
            ollama_host=ollama_host or DEFAULT_OLLAMA_HOST,
            min_clip_duration=float(min_clip_dur),
            max_clip_duration=float(max_clip_dur),
            max_clips=int(max_clips),
            min_score=int(min_score),
        )
        result = select_best_moments(segments, clip_config)

        clips_data: List[dict] = []
        clips_table: List[list] = []
        for clip in getattr(result, "clips", []) or []:
            title = clip.title or "Untitled"
            reason = clip.reason or ""
            score = _safe_int(clip.score)
            start = _safe_float(clip.start)
            end = _safe_float(clip.end)
            dur = _safe_float(clip.duration)
            clips_data.append({
                "start": start, "end": end, "duration": dur,
                "title": title, "reason": reason, "score": score,
            })
            clips_table.append([
                len(clips_data), f"{start:.2f}s", f"{end:.2f}s",
                f"{dur:.1f}s", title, reason, f"{score}/10",
            ])

        progress(1.0, desc="Completed!")

        if clips_data:
            ai_status = (
                f"### 🎉 Found {len(clips_data)} high-retention moment(s)\n"
                "Click any row below to select it, then use the export panel."
            )
        else:
            ai_status = (
                "### ⚠️ No moments met the virality criteria\n"
                "Try lowering the **minimum score**, widening the **duration range**, "
                "or using a stronger reasoning model."
            )

        return (metadata_md, transcript_table, clips_table, clips_data,
                "✅ Analysis complete.", ai_status)

    except gr.Error:
        raise
    except Exception as exc:
        logger.error(f"Processing error: {exc}", exc_info=True)
        return ("", [], [], [], f"❌ Processing failed: `{exc}`", "")


# ==========================================
# 4. TABLE ROW SELECTION
# ==========================================

def handle_clip_selection(evt: gr.SelectData, clips: Optional[List[dict]] = None):
    try:
        index = evt.index
        if hasattr(index, "__len__") and not isinstance(index, str):
            row = index[0] if len(index) else 0
        else:
            row = index
        row = _safe_int(row, 0)
        if clips and (row < 0 or row >= len(clips)):
            return gr.update()
        return row + 1
    except Exception:
        return gr.update()


# ==========================================
# 5. EXPORT SINGLE CLIP
# ==========================================

def export_single_clip(
    video_input, clips: Optional[List[dict]], clip_number,
    layout_mode: str, crf: int, normalize_audio: bool,
):
    video_path = get_video_path(video_input)
    if not clips:
        raise gr.Error("No analyzed clips available. Run video analysis first.")
    try:
        clip_idx = int(clip_number) - 1
    except (TypeError, ValueError):
        raise gr.Error("Please enter a valid clip number.")
    if clip_idx < 0 or clip_idx >= len(clips):
        raise gr.Error(f"Clip number must be between 1 and {len(clips)}.")

    target_clip = clips[clip_idx]
    start = _safe_float(target_clip.get("start"))
    end = _safe_float(target_clip.get("end"))
    duration = max(0.1, end - start)

    try:
        layout_enum = LayoutMode(layout_mode)
    except ValueError:
        raise gr.Error(f"Unknown layout mode: `{layout_mode}`")

    export_config = ExportConfig(
        layout=layout_enum, crf=_safe_int(crf, 23),
        normalize_audio=bool(normalize_audio),
    )

    safe_layout = str(layout_mode).replace("/", "_")
    output_filename = f"clip_{clip_idx + 1:02d}_{safe_layout}_{uuid.uuid4().hex[:6]}.mp4"
    output_path = OUTPUT_DIR / output_filename

    handler = VideoHandler(str(video_path))
    try:
        handler.export_clip(output_path, start, duration, export_config)
    except Exception as exc:
        logger.error(f"Export failed: {exc}", exc_info=True)
        raise gr.Error(f"Export failed: {exc}")

    status_markdown = (
        f"### ✅ Clip #{clip_idx + 1} Exported\n\n"
        "| Property | Value |\n| :--- | :--- |\n"
        f"| **Title** | {target_clip.get('title', '—')} |\n"
        f"| **Range** | `{start:.2f}s` → `{end:.2f}s` |\n"
        f"| **Duration** | `{duration:.2f}s` |\n"
        f"| **Layout** | `{layout_mode}` |\n"
        f"| **CRF** | `{_safe_int(crf, 23)}` |\n"
        f"| **File** | `{output_path.name}` |"
    )
    return str(output_path), status_markdown


# ==========================================
# 6. BATCH EXPORT ALL CLIPS & ZIP
# ==========================================

def export_all_clips_and_zip(
    video_input, clips: Optional[List[dict]], layout_mode: str,
    crf: int, normalize_audio: bool, progress=gr.Progress(),
):
    video_path = get_video_path(video_input)
    if not clips:
        raise gr.Error("No clips to export. Please analyze a video first.")
    try:
        layout_enum = LayoutMode(layout_mode)
    except ValueError:
        raise gr.Error(f"Unknown layout mode: `{layout_mode}`")

    export_config = ExportConfig(
        layout=layout_enum, crf=_safe_int(crf, 23),
        normalize_audio=bool(normalize_audio),
    )

    batch_id = uuid.uuid4().hex[:6]
    batch_dir = OUTPUT_DIR / f"batch_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    handler = VideoHandler(str(video_path))
    exported_paths: List[Path] = []
    total = len(clips)
    for idx, c in enumerate(clips):
        title_snippet = (c.get("title") or f"clip {idx + 1}")[:25]
        progress((idx + 1) / (total + 1),
                 desc=f"Rendering clip {idx + 1}/{total} — {title_snippet}...")
        start = _safe_float(c.get("start"))
        duration = max(0.1, _safe_float(c.get("end")) - start)
        clip_file = batch_dir / f"clip_{idx + 1:02d}_{layout_mode}.mp4"
        try:
            handler.export_clip(clip_file, start, duration, export_config)
            exported_paths.append(clip_file)
        except Exception as exc:
            logger.warning(f"Skipping clip {idx + 1}: {exc}")

    if not exported_paths:
        raise gr.Error("All clip exports failed. Check logs for details.")

    progress(0.95, desc="Bundling clips into ZIP archive...")
    zip_path = OUTPUT_DIR / f"clipty_clips_{batch_id}.zip"
    create_clips_zip(exported_paths, zip_path)
    progress(1.0, desc="Done!")

    size_line = ""
    if zip_path.exists():
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        size_line = f"- **Size:** {size_mb:.2f} MB\n"

    status_msg = (
        f"### 🎉 Batch Export Complete\n\n"
        f"- **Exported:** {len(exported_paths)} clip(s)\n"
        f"- **Archive:** `{zip_path.name}`\n{size_line}"
    )
    return str(exported_paths[0]), str(zip_path), status_msg


# ==========================================
# 7. UI HELPERS
# ==========================================

def section_header(number: str, title: str, subtitle: str = "") -> str:
    sub = f'<span class="section-subtitle">{subtitle}</span>' if subtitle else ""
    return (
        f'<div class="section-header">'
        f'<span class="section-number">{number}</span>'
        f'<span class="section-title">{title}</span>'
        f'{sub}</div>'
    )


# ==========================================
# 8. DARK-MODE-AWARE STYLES
# ==========================================

APP_CSS = """
/* ==========================================================
   DESIGN TOKENS — resolve differently per color scheme
   ========================================================== */
:root {
    /* Brand */
    --clipty-primary: #4f46e5;
    --clipty-primary-dark: #4338ca;
    --clipty-primary-light: #818cf8;

    /* Light-mode surface tokens — fall back to Gradio theme vars */
    --clipty-bg: var(--body-background-fill, #f8fafc);
    --clipty-surface: var(--background-fill-primary, #ffffff);
    --clipty-surface-alt: var(--background-fill-secondary, #f1f5f9);
    --clipty-border: var(--border-color-primary, #e2e8f0);
    --clipty-text: var(--body-text-color, #0f172a);
    --clipty-muted: var(--body-text-color-subdued, #64748b);

    /* Hover tint & shadows tuned per mode */
    --clipty-hover-bg: rgba(79, 70, 229, 0.06);
    --clipty-hover-text: var(--clipty-primary-dark);
    --clipty-shadow: 0 1px 3px rgba(15, 23, 42, 0.06),
                     0 4px 12px rgba(15, 23, 42, 0.04);

    --clipty-radius: 12px;
}

/* Gradio applies `.dark` to <html>/<body>/<gradio-app> in dark mode.
   Redefine every token that has an explicit color so dark mode is
   fully covered even if theme vars are missing. */
.dark,
html.dark,
body.dark,
gradio-app.dark,
gradio-app > .dark,
gradio-app .dark {
    --clipty-bg: var(--body-background-fill, #0f172a);
    --clipty-surface: var(--background-fill-primary, #1e293b);
    --clipty-surface-alt: var(--background-fill-secondary, #273449);
    --clipty-border: var(--border-color-primary, #334155);
    --clipty-text: var(--body-text-color, #f1f5f9);
    --clipty-muted: var(--body-text-color-subdued, #94a3b8);

    --clipty-primary-light: #a5b4fc;
    --clipty-hover-bg: rgba(129, 140, 248, 0.16);
    --clipty-hover-text: #c7d2fe;
    --clipty-shadow: 0 1px 3px rgba(0, 0, 0, 0.35),
                     0 4px 14px rgba(0, 0, 0, 0.25);
}

/* ==========================================================
   BASE
   ========================================================== */
.gradio-container {
    max-width: 1520px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                 'Segoe UI', Roboto, sans-serif !important;
    background: var(--clipty-bg);
}

/* Smooth transition when toggling dark mode */
.gradio-container,
.panel-card,
.gr-accordion,
.preset-btn button,
.primary-action-btn button,
.secondary-action-btn button,
.clipty-footer,
.section-header {
    transition: background-color 0.2s ease,
                border-color 0.2s ease,
                color 0.2s ease;
}

/* ==========================================================
   HEADER (always on a dark gradient — same in both modes)
   ========================================================== */
#clipty-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 45%, #4f46e5 100%);
    padding: 32px 36px;
    border-radius: 18px;
    color: #fff;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(49, 46, 129, 0.35);
}
#clipty-header::after {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 85% 20%, rgba(129, 140, 248, 0.35), transparent 45%),
        radial-gradient(circle at 15% 90%, rgba(168, 85, 247, 0.25), transparent 45%);
    pointer-events: none;
}
#clipty-header h1 {
    color: #fff !important;
    margin: 0 0 8px 0;
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    position: relative; z-index: 1;
}
#clipty-header p {
    color: #c7d2fe !important;
    font-size: 1rem; margin: 0;
    position: relative; z-index: 1;
    font-weight: 500;
}
.badge-row {
    display: flex; gap: 8px;
    margin-top: 16px; flex-wrap: wrap;
    position: relative; z-index: 1;
}
.badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 12px;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 999px;
    font-size: 0.78rem; font-weight: 600;
    color: #e0e7ff;
    backdrop-filter: blur(8px);
}

/* ==========================================================
   PANEL CARDS — surface uses token (white in light / slate-800 in dark)
   ========================================================== */
.panel-card {
    border: 1px solid var(--clipty-border);
    border-radius: var(--clipty-radius);
    padding: 18px 20px;
    background: var(--clipty-surface);
    box-shadow: var(--clipty-shadow);
    margin-bottom: 16px;
    color: var(--clipty-text);
}

/* ==========================================================
   SECTION HEADERS
   ========================================================== */
.section-header {
    display: flex; align-items: center; gap: 10px;
    margin: 4px 0 14px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--clipty-border);
}
.section-number {
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px;
    background: linear-gradient(135deg,
                var(--clipty-primary),
                var(--clipty-primary-light));
    color: #fff; border-radius: 8px;
    font-size: 0.78rem; font-weight: 700; flex-shrink: 0;
}
.section-title {
    font-size: 1rem; font-weight: 700;
    color: var(--clipty-text);
}
.section-subtitle {
    font-size: 0.82rem; font-weight: 500;
    color: var(--clipty-muted);
}

/* ==========================================================
   BUTTONS
   ========================================================== */
.primary-action-btn button {
    min-height: 50px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg,
                var(--clipty-primary),
                var(--clipty-primary-dark)) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
}
.primary-action-btn button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5) !important;
}
.secondary-action-btn button {
    min-height: 50px !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
}

/* Preset buttons — now fully theme-tokenized */
.preset-btn button {
    font-size: 0.82rem !important;
    padding: 8px 10px !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    border: 1px solid var(--clipty-border) !important;
    background: var(--clipty-surface) !important;
    color: var(--clipty-text) !important;
}
.preset-btn button:hover {
    border-color: var(--clipty-primary) !important;
    background: var(--clipty-hover-bg) !important;
    color: var(--clipty-hover-text) !important;
}

/* ==========================================================
   DATAFRAME
   ========================================================== */
#ai-table-box .table-wrap {
    border-radius: 12px;
    border: 1px solid var(--clipty-border);
}
#ai-table-box table { font-size: 0.88rem; }

/* ==========================================================
   FOOTER
   ========================================================== */
.clipty-footer {
    text-align: center; padding: 18px 0;
    color: var(--clipty-muted);
    font-size: 0.82rem;
    border-top: 1px solid var(--clipty-border);
    margin-top: 20px;
}

/* ==========================================================
   ACCORDION
   ========================================================== */
.gr-accordion {
    border: 1px solid var(--clipty-border) !important;
    border-radius: var(--clipty-radius) !important;
    margin-bottom: 12px !important;
    overflow: hidden;
    background: var(--clipty-surface) !important;
}
"""


# ==========================================
# 9. GRADIO INTERFACE
# ==========================================

with gr.Blocks(
    title="Clipty Mini — AI Video Clipping Studio",
    css=APP_CSS,
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ),
) as demo:

    gr.HTML("""
        <div id="clipty-header">
            <h1>🎬 Clipty Mini Studio</h1>
            <p>AI-powered viral moment detection • Multi-aspect reframing • One-click batch export</p>
            <div class="badge-row">
                <span class="badge">⚡ Faster-Whisper</span>
                <span class="badge">🧠 Ollama LLM</span>
                <span class="badge">📱 9:16 Auto-Reframe</span>
                <span class="badge">📦 Batch ZIP Export</span>
            </div>
        </div>
    """)

    with gr.Row(equal_height=False):

        # ------- LEFT CONTROL DECK -------
        with gr.Column(scale=1, min_width=400):

            with gr.Group(elem_classes=["panel-card"]):
                gr.HTML(section_header("1", "Upload Video", "MP4, MOV, MKV, WebM"))
                video_input = gr.Video(
                    label="", sources=["upload"],
                    elem_id="video-uploader", show_label=False,
                )
                process_btn = gr.Button(
                    "🚀 Analyze & Find Viral Highlights",
                    variant="primary", elem_classes=["primary-action-btn"],
                )
                process_status = gr.Markdown(
                    "**Ready** — upload a video to get started."
                )

            with gr.Accordion("⚙️  Model & Hardware Selection", open=True):
                gr.HTML(section_header("2", "Whisper & LLM Config"))

                gr.Markdown("**Speech-to-Text**")
                whisper_model_dropdown = gr.Dropdown(
                    label="Whisper Model",
                    choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
                    value="tiny",
                    info="tiny/base = fastest · small/medium = more accurate",
                )
                device_dropdown = gr.Dropdown(
                    label="Hardware Acceleration",
                    choices=["auto", "cpu", "cuda"], value="auto",
                    info="auto picks GPU if available",
                )

                gr.Markdown("**AI Reasoning (Ollama)**")
                _initial_models = _load_initial_models()
                _initial_default = _pick_default_model(_initial_models)

                ollama_model_dropdown = gr.Dropdown(
                    label="Ollama Model",
                    choices=_initial_models,
                    value=_initial_default,
                    allow_custom_value=True,
                    info="Select an installed model or type a custom name",
                )

                with gr.Row():
                    ollama_host_input = gr.Textbox(
                        label="Ollama Host URL",
                        value=DEFAULT_OLLAMA_HOST, scale=4,
                        placeholder=DEFAULT_OLLAMA_HOST,
                    )
                    refresh_models_btn = gr.Button("🔄 Refresh", scale=1)

                ollama_status_box = gr.Markdown(
                    "🟢 Ready — click **Refresh** to verify connection."
                )

            with gr.Accordion("✂️  Clip Length & Virality Controls", open=True):
                gr.HTML(section_header("3", "Presets & Thresholds"))

                gr.Markdown("**Quick Presets**")
                with gr.Row():
                    preset_shorts = gr.Button("📱 Shorts · 15–60s", elem_classes=["preset-btn"])
                    preset_podcast = gr.Button("🎙️ Podcast · 30–90s", elem_classes=["preset-btn"])
                with gr.Row():
                    preset_snippet = gr.Button("⚡ Snippet · 10–30s", elem_classes=["preset-btn"])
                    preset_extended = gr.Button("⏱️ Extended · 60–180s", elem_classes=["preset-btn"])

                min_dur_slider = gr.Slider(
                    label="Minimum Clip Duration (s)",
                    minimum=5.0, maximum=120.0, value=15.0, step=1.0,
                )
                max_dur_slider = gr.Slider(
                    label="Maximum Clip Duration (s)",
                    minimum=15.0, maximum=300.0, value=60.0, step=5.0,
                )
                max_clips_slider = gr.Slider(
                    label="Maximum Number of Clips",
                    minimum=1, maximum=15, value=5, step=1,
                )
                min_score_slider = gr.Slider(
                    label="Minimum Virality Score (1–10)",
                    minimum=5, maximum=9, value=7, step=1,
                    info="Higher = fewer, stronger moments",
                )

            with gr.Accordion("🎨  Export Layout & Video Quality", open=True):
                gr.HTML(section_header("4", "Framing & Encoding"))

                layout_radio = gr.Radio(
                    label="Framing Aspect Ratio",
                    choices=[
                        ("📺 Original (source ratio)", "original"),
                        ("📱 9:16 Blurred background", "vertical_blur"),
                        ("✂️ 9:16 Center crop", "vertical_crop"),
                    ],
                    value="original",
                    info="Turn landscape into vertical shorts automatically",
                )

                with gr.Row():
                    crf_slider = gr.Slider(
                        label="Quality (CRF)",
                        minimum=18, maximum=28, value=23, step=1,
                        info="Lower = better quality, larger file",
                    )
                    normalize_audio_chk = gr.Checkbox(
                        label="Normalize Audio (EBU R128)",
                        value=False, info="Balance dialogue loudness",
                    )

        # ------- RIGHT RESULTS -------
        with gr.Column(scale=2, min_width=640):
            with gr.Tabs():

                with gr.Tab("✨ AI Highlight Moments"):
                    ai_status_md = gr.Markdown(
                        "### 👋 Welcome\n"
                        "Upload a video on the left and click **Analyze** to discover viral highlights."
                    )

                    moments_dataframe = gr.Dataframe(
                        label="AI-Detected Highlights (click a row to select)",
                        headers=["#", "Start", "End", "Duration", "Title", "Virality Analysis", "Score"],
                        datatype=["number", "str", "str", "str", "str", "str", "str"],
                        interactive=False, type="array",
                        wrap=True, max_height=380,
                        elem_id="ai-table-box",
                    )

                    with gr.Group(elem_classes=["panel-card"]):
                        gr.HTML(section_header("5", "Export Clips", "Single or batch"))
                        with gr.Row():
                            selected_clip_num = gr.Number(
                                label="Selected Clip #",
                                value=1, precision=0, minimum=1, scale=1,
                            )
                            export_single_btn = gr.Button(
                                "✂️ Export Selected", variant="primary", scale=2,
                                elem_classes=["primary-action-btn"],
                            )
                            export_all_btn = gr.Button(
                                "📦 Export All → ZIP", variant="secondary", scale=2,
                                elem_classes=["secondary-action-btn"],
                            )

                    with gr.Row():
                        with gr.Column(scale=1):
                            preview_video = gr.Video(
                                label="Rendered Clip Preview", interactive=False,
                            )
                        with gr.Column(scale=1):
                            zip_download_file = gr.File(
                                label="Download All Clips (ZIP)", interactive=False,
                            )

                    export_status_md = gr.Markdown()

                with gr.Tab("📝 Full Transcript"):
                    transcript_dataframe = gr.Dataframe(
                        label="Segment-level Transcription",
                        headers=["Start", "End", "Dialogue"],
                        interactive=False, type="array",
                        wrap=True, max_height=580,
                    )

                with gr.Tab("📋 Technical Metadata"):
                    metadata_output = gr.Markdown(
                        "*Video technical specifications will appear here after analysis.*"
                    )

    gr.HTML(
        '<div class="clipty-footer">Clipty Mini · Built with Gradio · '
        'Whisper + Ollama · All processing runs locally.</div>'
    )

    clips_state = gr.State(value=[])

    # ------- EVENTS -------
    refresh_models_btn.click(
        fn=refresh_ollama_models,
        inputs=[ollama_host_input],
        outputs=[ollama_model_dropdown, ollama_status_box],
    )

    for btn, name in [
        (preset_shorts, "shorts"),
        (preset_podcast, "podcast"),
        (preset_snippet, "snippet"),
        (preset_extended, "extended"),
    ]:
        btn.click(
            fn=lambda _n=name: apply_preset(_n),
            outputs=[min_dur_slider, max_dur_slider, max_clips_slider, min_score_slider],
        )

    process_btn.click(
        fn=process_video_pipeline,
        inputs=[
            video_input, whisper_model_dropdown, device_dropdown,
            ollama_model_dropdown, ollama_host_input,
            min_dur_slider, max_dur_slider, max_clips_slider, min_score_slider,
        ],
        outputs=[
            metadata_output, transcript_dataframe, moments_dataframe,
            clips_state, process_status, ai_status_md,
        ],
        concurrency_limit=1,
        concurrency_id="clipty-pipeline",
    )

    moments_dataframe.select(
        fn=handle_clip_selection,
        inputs=[clips_state],
        outputs=[selected_clip_num],
        show_progress="hidden",
    )

    export_single_btn.click(
        fn=export_single_clip,
        inputs=[video_input, clips_state, selected_clip_num,
                layout_radio, crf_slider, normalize_audio_chk],
        outputs=[preview_video, export_status_md],
    )

    export_all_btn.click(
        fn=export_all_clips_and_zip,
        inputs=[video_input, clips_state, layout_radio, crf_slider, normalize_audio_chk],
        outputs=[preview_video, zip_download_file, export_status_md],
    )


# ==========================================
# LAUNCH
# ==========================================

if __name__ == "__main__":
    demo.launch(
        allowed_paths=[str(OUTPUT_DIR)],
        show_error=True,
    )