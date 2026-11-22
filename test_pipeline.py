"""
End-to-End Test Suite for Clipty-Mini Production Engine.
Validates configuration constraints, video metadata extraction,
candidate window generation, heuristic ranking, and zip bundling.
"""

from pathlib import Path
import pytest
from core.config import (
    ClipSelectionConfig,
    ExportConfig,
    LayoutMode,
    TranscriptionConfig
)
from core.video_processor import VideoHandler, create_clips_zip
from core.clip_manager import (
    calculate_heuristic_score,
    build_candidate_windows,
    preselect_candidates,
    deduplicate_clips,
    select_best_moments,
    Clip
)
from core.ollama_service import check_ollama_status


def test_config_validation():
    print("Testing config validation...")
    cfg = ClipSelectionConfig(
        min_clip_duration=15.0,
        max_clip_duration=60.0,
        max_clips=5
    )
    assert cfg.min_clip_duration == 15.0
    assert cfg.max_clip_duration == 60.0

    # Test invalid duration validation (min >= max)
    try:
        ClipSelectionConfig(min_clip_duration=60.0, max_clip_duration=30.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        print(" -> Caught expected duration validation error.")


def test_candidate_window_generation():
    print("Testing candidate window generation...")
    # Mock transcript segments
    mock_transcript = [
        {"start": 0.0, "end": 5.0, "text": "Here is the biggest secret to building software."},
        {"start": 5.2, "end": 12.0, "text": "Most engineers think you just write code."},
        {"start": 12.5, "end": 20.0, "text": "Actually, the real truth is understanding customer problems."},
        {"start": 20.5, "end": 28.0, "text": "If you don't solve their issue, nobody will use your product."},
        {"start": 28.5, "end": 35.0, "text": "And that is why simplicity always wins."}
    ]

    candidates = build_candidate_windows(
        mock_transcript,
        min_duration=10.0,
        max_duration=30.0
    )
    print(f" -> Generated {len(candidates)} candidate windows.")
    assert len(candidates) > 0
    for c in candidates:
        assert 10.0 <= c["duration"] <= 30.0

    # Test selection with fallback
    cfg = ClipSelectionConfig(
        min_clip_duration=10.0,
        max_clip_duration=30.0,
        max_clips=2
    )
    result = select_best_moments(mock_transcript, cfg)
    print(f" -> Selected {len(result.clips)} clips (fallback/AI).")
    assert len(result.clips) > 0


def test_video_metadata_and_zip():
    print("Testing video metadata and zip archiving...")
    sample_video = Path("storage/video_output/ai_clip_01_8838c9a1.mp4")
    if sample_video.exists():
        vh = VideoHandler(str(sample_video))
        info = vh.info()
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["aspect_ratio"] == "16:9"
        print(f" -> Probed metadata: {info['resolution']}, {info['duration']:.1f}s")

        # Test ZIP archiving
        zip_path = Path("storage/video_output/test_bundle.zip")
        create_clips_zip([sample_video], zip_path)
        assert zip_path.is_file() and zip_path.stat().st_size > 0
        print(f" -> Created ZIP bundle: {zip_path.stat().st_size} bytes")
        zip_path.unlink(missing_ok=True)


def test_deduplication():
    print("Testing overlap deduplication...")
    clip1 = Clip(start=0.0, end=20.0, duration=20.0, title="A", reason="R", score=9)
    clip2 = Clip(start=5.0, end=22.0, duration=17.0, title="B", reason="R", score=8)  # heavily overlaps clip1
    clip3 = Clip(start=40.0, end=60.0, duration=20.0, title="C", reason="R", score=7) # independent

    deduped = deduplicate_clips([clip1, clip2, clip3], max_overlap=0.40)
    print(f" -> Deduplicated from 3 clips to {len(deduped)} clips.")
    assert len(deduped) == 2
    titles = [c.title for c in deduped]
    assert "A" in titles
    assert "C" in titles
    assert "B" not in titles


if __name__ == "__main__":
    test_config_validation()
    test_candidate_window_generation()
    test_video_metadata_and_zip()
    test_deduplication()
    print("\n✅ ALL TESTS PASSED!")
