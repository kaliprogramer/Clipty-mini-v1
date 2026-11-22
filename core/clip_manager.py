"""
Production AI Moment Selection Engine.
Generates candidate clip windows using speech-boundary heuristics,
evaluates standalone virality using Ollama LLMs with structured schemas,
performs quality verification, and eliminates overlapping segments.
"""

import json
import math
import re
import time
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from core.config import ClipSelectionConfig
from core.ollama_service import get_ollama_client
from helper.logger import logger


# ==================================================
# 1. SCHEMAS
# ==================================================

class AIClipChoice(BaseModel):
    """LLM decision for a single candidate highlight."""
    candidate_id: int = Field(ge=1, description="ID of the candidate window")
    title: str = Field(min_length=3, max_length=120, description="Punchy, engaging video title")
    reason: str = Field(min_length=5, max_length=200, description="Why this works as a standalone viral short")
    score: int = Field(ge=1, le=10, description="Virality and completeness score from 1 to 10")
    opening_hook: bool = Field(description="Does it start with a hook that grabs attention?")
    context_clear: bool = Field(description="Can a stranger understand it without watching the full video?")
    middle_coherent: bool = Field(description="Is the middle logical and engaging without random tangents?")
    ending_payoff: bool = Field(description="Does it complete the thought or punchline cleanly?")


class AISelectionResponse(BaseModel):
    """Structured LLM response containing selected clips."""
    clips: List[AIClipChoice] = Field(default_factory=list)


class Clip(BaseModel):
    """Validated clip metadata ready for UI and FFmpeg export."""
    start: float
    end: float
    duration: float
    title: str
    reason: str
    score: int


class ClipSelection(BaseModel):
    """Collection of final deduplicated clips."""
    clips: List[Clip] = Field(default_factory=list)


# ==================================================
# 2. TRANSCRIPT PREPARATION & CLEANING
# ==================================================

def prepare_transcript(transcript: List[dict]) -> List[dict]:
    """Clean, sort, and validate raw transcript segments."""
    segments = []
    for segment in transcript:
        try:
            start = float(segment.get("start", -1))
            end = float(segment.get("end", -1))
            text = str(segment.get("text", "")).strip()
        except (TypeError, ValueError):
            continue

        if start < 0 or end <= start or not text:
            continue

        segments.append({
            "start": start,
            "end": end,
            "text": text
        })

    segments.sort(key=lambda item: item["start"])
    if not segments:
        raise ValueError("No valid transcript segments found for analysis.")
    return segments


# ==================================================
# 3. HEURISTIC CANDIDATE SCORING
# ==================================================

HOOK_WORDS = {
    "secret", "mistake", "truth", "actually", "impossible", "surprising",
    "dangerous", "never", "wrong", "discovered", "why", "how", "because",
    "hack", "trick", "insane", "million", "billion", "crazy", "warning",
    "rule", "stop", "first", "nobody", "everyone", "biggest", "problem",
    "solution", "result", "worst", "best", "shocking"
}

FILLER_WORDS = {
    "um", "uh", "hmm", "basically", "like", "you know", "sort of", "kind of"
}


def calculate_heuristic_score(text: str, duration: float) -> float:
    """Calculate rule-based virality and engagement score for candidate text."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words or duration <= 0:
        return 0.0

    word_count = len(words)
    words_per_sec = word_count / duration

    # 1. Hook vocabulary boost
    hook_count = sum(1 for w in words if w in HOOK_WORDS)
    score = min(hook_count * 0.4, 2.5)

    # 2. Curiosity & Questions boost
    score += min(text.count("?") * 0.6, 2.0)

    # 3. Conversational pacing (ideal: 2.0 to 3.5 words/sec)
    if 1.8 <= words_per_sec <= 3.8:
        score += 1.5
    elif words_per_sec < 1.0:
        score -= 1.0

    # 4. Sentence boundary check (ends with punctuation)
    if text.rstrip().endswith((".", "!", "?")):
        score += 1.0

    # 5. Penalize excessive fillers
    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    score -= min(filler_count * 0.3, 1.5)

    return max(0.1, round(score, 2))


# ==================================================
# 4. ADAPTIVE CANDIDATE WINDOW GENERATOR
# ==================================================

def build_candidate_windows(
    segments: List[dict],
    min_duration: float,
    max_duration: float
) -> List[dict]:
    """
    Generate overlapping candidate windows within [min_duration, max_duration].
    Snaps endpoints to natural sentence or speech boundaries when possible.
    """
    candidates = []
    target_duration = (min_duration + max_duration) / 2.0
    window_step = max(8.0, target_duration / 2.5)

    start_idx = 0
    candidate_id = 1

    while start_idx < len(segments):
        start_time = segments[start_idx]["start"]
        end_idx = start_idx

        # Expand forward until reaching target duration or boundary
        while end_idx + 1 < len(segments):
            current_duration = segments[end_idx]["end"] - start_time
            if current_duration >= target_duration:
                # If current segment ends with sentence boundary, snap here
                if segments[end_idx]["text"].rstrip().endswith((".", "!", "?")):
                    break
                # Or if next segment would exceed max_duration, stop here
                next_duration = segments[end_idx + 1]["end"] - start_time
                if next_duration > max_duration:
                    break
            end_idx += 1

        selected = segments[start_idx:end_idx + 1]
        if not selected:
            break

        start = selected[0]["start"]
        end = selected[-1]["end"]
        duration = end - start

        if min_duration <= duration <= max_duration:
            text = " ".join(item["text"] for item in selected).strip()

            opening_segments = [item for item in selected if item["start"] < start + min(8.0, duration * 0.25)]
            opening = " ".join(item["text"] for item in opening_segments).strip()

            ending_segments = [item for item in selected if item["end"] > end - min(10.0, duration * 0.3)]
            ending = " ".join(item["text"] for item in ending_segments).strip()

            candidates.append({
                "id": candidate_id,
                "start": start,
                "end": end,
                "duration": duration,
                "text": text,
                "opening": opening,
                "ending": ending,
                "heuristic_score": calculate_heuristic_score(text, duration)
            })
            candidate_id += 1

        # Advance start_index by window_step
        next_time = start_time + window_step
        while start_idx < len(segments) and segments[start_idx]["start"] < next_time:
            start_idx += 1

    return candidates


# ==================================================
# 5. BALANCED CANDIDATE PRE-SELECTION
# ==================================================

def preselect_candidates(candidates: List[dict], max_candidates: int) -> List[dict]:
    """
    Ensure candidates are balanced across the entire video timeline rather than clustering in one spot.
    Uses time buckets and selects the highest heuristic candidate per bucket.
    """
    if not candidates:
        return []

    if len(candidates) <= max_candidates:
        return sorted(candidates, key=lambda c: c["start"])

    video_start = min(c["start"] for c in candidates)
    video_end = max(c["end"] for c in candidates)
    video_duration = max(1.0, video_end - video_start)
    bucket_size = video_duration / max_candidates

    buckets: Dict[int, dict] = {}
    for candidate in candidates:
        bucket_id = min(
            max_candidates - 1,
            int((candidate["start"] - video_start) / bucket_size)
        )
        existing = buckets.get(bucket_id)
        if existing is None or candidate["heuristic_score"] > existing["heuristic_score"]:
            buckets[bucket_id] = candidate

    selected = list(buckets.values())
    selected.sort(key=lambda item: item["start"])
    return selected[:max_candidates]


# ==================================================
# 6. SYSTEM PROMPTS & REQUEST BUILDER
# ==================================================

SYSTEM_PROMPT = """You are an expert social media video editor specializing in YouTube Shorts, TikToks, and Instagram Reels.
Your task is to identify STANDALONE, COMPLETE, and HIGH-RETENTION highlights from video transcripts.

A viewer will encounter this clip with ZERO prior context.
Evaluate each candidate on FOUR MANDATORY CRITERIA:

1. OPENING HOOK (opening_hook):
   Does the first 3-5 seconds grab attention? (e.g. bold claim, intriguing question, shocking fact, funny moment).
   Reject clips that start with mid-thought phrases ("And so...", "Like I said...", "Furthermore...").

2. STANDALONE CONTEXT (context_clear):
   Can a new viewer understand what is being discussed without knowing who the speaker is or what came before?
   Reject clips with unexplained internal references ("as we saw earlier", "my friend Bob who I mentioned").

3. COHERENT MIDDLE (middle_coherent):
   Does the clip present a logical progression (Setup -> Conflict/Explanation -> Resolution)?
   Reject rambling, stuttering, or tangential speech.

4. SATISFYING PAYOFF (ending_payoff):
   Does the clip conclude on a complete thought, punchline, revelation, or clear takeaway?
   Reject clips that cut off mid-sentence ("Because the reason is...", "And then he...").

SCORING RULES:
- 10: Exceptional viral hook and flawless standalone storytelling.
- 8-9: Strong highlight with clear value and clean ending.
- 7: Decent moment, acceptable if options are limited.
- 1-6: Incomplete or boring. Reject.

All four boolean checks must be true for a clip to qualify.
Output strictly valid JSON matching the schema. Never invent facts.
"""


def build_user_prompt(
    candidates: List[dict],
    max_clips: int,
    min_score: int,
    min_dur: float,
    max_dur: float
) -> str:
    """Construct formatted LLM prompt containing candidate information."""
    candidate_sections = []
    for c in candidates:
        candidate_sections.append(
            f"--- CANDIDATE ID: {c['id']} ---\n"
            f"TIME: {c['start']:.2f}s - {c['end']:.2f}s ({c['duration']:.1f}s)\n"
            f"OPENING: \"{c['opening']}\"\n"
            f"TRANSCRIPT: \"{c['text']}\"\n"
            f"ENDING: \"{c['ending']}\"\n"
        )

    return f"""Find up to {max_clips} standalone viral clips from the candidates below.

CONSTRAINTS:
- Only select from candidate IDs provided.
- Clip durations must be between {min_dur:.0f}s and {max_dur:.0f}s.
- Score must be >= {min_score}/10.
- All 4 boolean criteria (opening_hook, context_clear, middle_coherent, ending_payoff) MUST be TRUE.
- If fewer candidates meet the standard, return only the qualifying ones.

CANDIDATES:
{"".join(candidate_sections)}

Return valid JSON with the format:
{{
  "clips": [
    {{
      "candidate_id": 1,
      "title": "Short Punchy Title",
      "reason": "Detailed viral reasoning",
      "score": 9,
      "opening_hook": true,
      "context_clear": true,
      "middle_coherent": true,
      "ending_payoff": true
    }}
  ]
}}
"""


# ==================================================
# 7. LLM REQUEST & ROBUST JSON EXTRACTION
# ==================================================

def extract_json_payload(raw_text: str) -> dict:
    """Robustly extract JSON object from LLM response text, stripping markdown fences if present."""
    clean_text = raw_text.strip()

    # Strip markdown ```json ... ``` blocks
    json_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
    if json_block_match:
        clean_text = json_block_match.group(1).strip()

    # Search for first '{' and last '}'
    first_brace = clean_text.find("{")
    last_brace = clean_text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        clean_text = clean_text[first_brace:last_brace + 1]

    return json.loads(clean_text)


def request_ai_selection(
    candidates: List[dict],
    config: ClipSelectionConfig
) -> AISelectionResponse:
    """Send candidate prompts to Ollama with retry and fallback parsing."""
    client = get_ollama_client(config.ollama_host)
    max_retries = 2
    last_error = None

    prompt = build_user_prompt(
        candidates,
        max_clips=config.max_clips,
        min_score=config.min_score,
        min_dur=config.min_clip_duration,
        max_dur=config.max_clip_duration
    )

    for attempt in range(1, max_retries + 1):
        logger.info(f"Ollama request attempt {attempt}/{max_retries} using model '{config.ollama_model}'...")
        try:
            response = client.chat(
                model=config.ollama_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                format=AISelectionResponse.model_json_schema(),
                options={
                    "temperature": 0.1,
                    "num_ctx": 4096,
                    "num_predict": 2048,
                    "num_thread": 2,
                },
                keep_alive="5m"
            )

            raw_content = response.message.content
            if not raw_content:
                raise ValueError("Ollama returned empty response content.")

            # Parse and validate schema
            try:
                parsed_json = extract_json_payload(raw_content)
                result = AISelectionResponse.model_validate(parsed_json)
                logger.info(f"Ollama successfully returned {len(result.clips)} candidate choices.")
                return result
            except Exception as parse_err:
                logger.warning(f"JSON validation failed on attempt {attempt}: {parse_err}")
                if attempt == max_retries:
                    raise parse_err

        except Exception as exc:
            last_error = exc
            logger.warning(f"Ollama attempt {attempt} failed: {exc}")

    raise RuntimeError(f"Ollama AI selection failed: {last_error}")


# ==================================================
# 8. FALLBACK HEURISTIC SELECTION (GRACEFUL DEGRADATION)
# ==================================================

def fallback_heuristic_selection(
    candidates: List[dict],
    config: ClipSelectionConfig
) -> ClipSelection:
    """
    Fallback highlight extractor used if Ollama is unreachable.
    Uses heuristic ranking so the user still receives valid, actionable clips.
    """
    logger.warning("Using fallback heuristic selection (Ollama was offline or failed).")
    ranked = sorted(candidates, key=lambda c: c["heuristic_score"], reverse=True)
    fallback_clips: List[Clip] = []

    for c in ranked:
        # Heuristic score mapped to 7-9
        mapped_score = min(9, max(7, int(c["heuristic_score"] * 2 + 5)))
        fallback_clips.append(Clip(
            start=c["start"],
            end=c["end"],
            duration=round(c["duration"], 2),
            title=f"Highlight: {c['opening'][:40]}...",
            reason="High speech density and key conversational keywords detected.",
            score=mapped_score
        ))
        if len(fallback_clips) >= config.max_clips:
            break

    return ClipSelection(clips=fallback_clips)


# ==================================================
# 9. VALIDATION & OVERLAP DEDUPLICATION
# ==================================================

def calculate_overlap_ratio(clip_a: Clip, clip_b: Clip) -> float:
    """Calculate the overlap ratio relative to the shorter clip."""
    overlap_start = max(clip_a.start, clip_b.start)
    overlap_end = min(clip_a.end, clip_b.end)
    overlap = max(0.0, overlap_end - overlap_start)
    shorter_duration = min(clip_a.duration, clip_b.duration)
    return overlap / shorter_duration if shorter_duration > 0 else 0.0


def deduplicate_clips(clips: List[Clip], max_overlap: float = 0.40) -> List[Clip]:
    """Remove overlapping clips, prioritizing higher-scoring clips."""
    ranked = sorted(clips, key=lambda c: c.score, reverse=True)
    selected: List[Clip] = []

    for candidate in ranked:
        overlaps = any(
            calculate_overlap_ratio(candidate, existing) >= max_overlap
            for existing in selected
        )
        if not overlaps:
            selected.append(candidate)

    # Sort final clips chronologically for pleasant viewing
    return sorted(selected, key=lambda c: c.start)


# ==================================================
# 10. MAIN ENTRYPOINT
# ==================================================

def select_best_moments(
    transcript: List[dict],
    config: Optional[ClipSelectionConfig] = None
) -> ClipSelection:
    """
    Main pipeline to identify the best standalone clips from video transcripts.
    Accepts dynamic configuration for model, durations, and clip count.
    """
    config = config or ClipSelectionConfig()
    total_start = time.perf_counter()

    # 1. Clean transcript
    segments = prepare_transcript(transcript)
    total_video_duration = segments[-1]["end"] - segments[0]["start"]

    if total_video_duration < config.min_clip_duration:
        raise ValueError(
            f"Video duration ({total_video_duration:.1f}s) is shorter than the "
            f"configured minimum clip length ({config.min_clip_duration:.1f}s)."
        )

    # 2. Build candidate windows
    all_candidates = build_candidate_windows(
        segments,
        min_duration=config.min_clip_duration,
        max_duration=config.max_clip_duration
    )

    if not all_candidates:
        logger.warning("No candidate windows met duration constraints.")
        return ClipSelection(clips=[])

    # 3. Pre-select balanced candidates across the timeline
    candidates = preselect_candidates(all_candidates, max_candidates=config.max_candidates)
    candidate_map = {c["id"]: c for c in candidates}

    # 4. Request LLM evaluation with fallback
    valid_clips: List[Clip] = []
    try:
        ai_result = request_ai_selection(candidates, config)

        for choice in ai_result.clips:
            cand = candidate_map.get(choice.candidate_id)
            if not cand:
                continue

            # Verify quality checks
            if not (choice.opening_hook and choice.context_clear and choice.middle_coherent and choice.ending_payoff):
                continue

            if choice.score < config.min_score:
                continue

            duration = cand["end"] - cand["start"]
            if not (config.min_clip_duration <= duration <= config.max_clip_duration):
                continue

            valid_clips.append(Clip(
                start=cand["start"],
                end=cand["end"],
                duration=round(duration, 2),
                title=choice.title.strip(),
                reason=choice.reason.strip(),
                score=choice.score
            ))

    except Exception as exc:
        logger.warning(f"AI selection encountered error ({exc}). Initiating fallback heuristic selection...")
        return fallback_heuristic_selection(candidates, config)

    # If LLM passed but returned 0 valid clips, use fallback
    if not valid_clips:
        logger.info("AI returned no passing clips; using heuristic highlights.")
        return fallback_heuristic_selection(candidates, config)

    # 5. Deduplicate and trim to max_clips
    final_clips = deduplicate_clips(valid_clips)
    final_clips = final_clips[:config.max_clips]

    elapsed = time.perf_counter() - total_start
    logger.info(f"Selected {len(final_clips)} top moments in {elapsed:.2f}s.")
    return ClipSelection(clips=final_clips)
