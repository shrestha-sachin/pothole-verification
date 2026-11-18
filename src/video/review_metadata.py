"""Present database rows and renderer sidecars with the same review fields."""


def normalize_review_metadata(record: dict) -> dict:
    result = dict(record)
    for target, source in (
        ("effective_fps", "effective_inference_fps"),
        ("processing_duration", "processing_duration_sec"),
        ("total_detections", "total_detections_drawn"),
    ):
        result.setdefault(target, result.get(source, 0))
    result.setdefault("unique_tracks", len(result.get("unique_track_ids") or []))
    return result
