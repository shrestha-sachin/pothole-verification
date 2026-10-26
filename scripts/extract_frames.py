"""Extract approximately one frame per second from a road video.

Usage (from project root, with venv active):
    python scripts/extract_frames.py --video data/raw/local_video/your_video.mp4

The output directory is created automatically under data/processed/video_frames/
named after the video stem, e.g.:
    data/processed/video_frames/your_video/

Frame filenames encode the video stem and timestamp in milliseconds:
    gb_video_001_t000012000.jpg  →  12 seconds in

A CSV manifest is written to:
    data/processed/video_frames/your_video/frames.csv

Options:
    --video PATH          Path to source video (required)
    --video-id ID         Short ID for this recording; defaults to video filename stem
    --fps FLOAT           Target frame rate in frames/sec (default: 1.0)
    --output DIR          Override output directory (default: auto under video_frames/)
    --no-overwrite        Exit with an error if output directory already has frames
    --dry-run             Report video metadata without extracting

Dependencies: OpenCV (cv2) is already installed in the project venv.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check — fail clearly before touching any file
# ---------------------------------------------------------------------------
try:
    import cv2
except ImportError:
    sys.exit(
        "ERROR: OpenCV is not installed in the active Python environment.\n"
        "Fix: .venv/bin/pip install opencv-python-headless"
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_FRAMES_ROOT = PROJECT_ROOT / "data" / "processed" / "video_frames"

CSV_FIELDS = [
    "video_id",
    "frame_index",
    "timestamp_ms",
    "timestamp_sec",
    "output_filename",
    "filepath",       # project-relative
    "width",
    "height",
]


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def extract(
    video_path: Path,
    video_id: str,
    output_dir: Path,
    target_fps: float,
    no_overwrite: bool,
    dry_run: bool,
) -> None:
    # ---- Open video -----------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(
            f"ERROR: Cannot open video file: {video_path}\n"
            "  • Check the path and extension (mp4, mov, avi, mkv all work with OpenCV)\n"
            "  • If the file has spaces in its name, quote the path on the command line\n"
            "  • Ensure OpenCV was built with the required codec support"
        )

    source_fps: float = cap.get(cv2.CAP_PROP_FPS)
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_int: int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))

    if source_fps <= 0:
        cap.release()
        sys.exit(
            f"ERROR: Video reports {source_fps} FPS — likely a corrupt or unsupported file.\n"
            "  Try re-encoding with: ffmpeg -i input.mp4 -c:v libx264 output.mp4"
        )

    duration_sec = total_frames / source_fps if total_frames > 0 else 0.0
    expected_extractions = max(1, int(duration_sec * target_fps)) if duration_sec > 0 else "unknown"

    print(f"Video:          {video_path}")
    print(f"Video ID:       {video_id}")
    print(f"Codec:          {fourcc_str}")
    print(f"Source FPS:     {source_fps:.3f}")
    print(f"Total frames:   {total_frames}")
    print(f"Duration:       {duration_sec:.1f} s  ({duration_sec / 60:.1f} min)")
    print(f"Target FPS:     {target_fps:.2f} → ~{expected_extractions} frames expected")
    print()

    if dry_run:
        cap.release()
        print("[dry-run] No frames extracted.")
        return

    # ---- Prepare output directory ----------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = list(output_dir.glob("*.jpg"))
    if existing_frames and no_overwrite:
        cap.release()
        sys.exit(
            f"ERROR: Output directory already contains {len(existing_frames)} .jpg files:\n"
            f"  {output_dir}\n"
            "  Remove --no-overwrite to allow adding/updating, or delete the directory first."
        )
    if existing_frames:
        print(f"  Output directory has {len(existing_frames)} existing frames; skipping those timestamps.")

    manifest_path = output_dir / "frames.csv"
    existing_timestamps: set[int] = set()
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    existing_timestamps.add(int(row["timestamp_ms"]))
                except (ValueError, KeyError):
                    pass

    # ---- Extract --------------------------------------------------------
    frame_index = 0
    saved_count = 0
    skipped_existing = 0
    next_sample_time = 0.0
    interval = 1.0 / target_fps

    new_rows: list[dict] = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        timestamp_sec = frame_index / source_fps
        timestamp_ms = int(round(timestamp_sec * 1000))

        if timestamp_sec + 0.5 / source_fps >= next_sample_time:
            frame_filename = f"{video_id}_t{timestamp_ms:012d}.jpg"
            frame_path = output_dir / frame_filename

            if timestamp_ms in existing_timestamps and frame_path.exists():
                skipped_existing += 1
            else:
                h, w = frame.shape[:2]
                temp_path = frame_path.with_name(f".tmp_{frame_path.name}")
                if not cv2.imwrite(str(temp_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    cap.release()
                    sys.exit(f"ERROR: Failed to write frame to {temp_path}")
                temp_path.replace(frame_path)
                saved_count += 1

                new_rows.append({
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "output_filename": frame_filename,
                    "filepath": rel(frame_path),
                    "width": w,
                    "height": h,
                })

            next_sample_time += interval

        frame_index += 1

    cap.release()

    # ---- Save / merge manifest (deduplicated) ---------------------------
    combined: list[dict] = []
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as fh:
            combined = list(csv.DictReader(fh))
    combined.extend(new_rows)

    seen_files: set[str] = set()
    all_rows: list[dict] = []
    for r in combined:
        fn = r.get("output_filename", "")
        if fn and fn not in seen_files:
            seen_files.add(fn)
            all_rows.append(r)

    all_rows.sort(key=lambda r: int(r.get("timestamp_ms", 0)))

    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"✓ Frames extracted:   {saved_count}")
    print(f"  Skipped (existing): {skipped_existing}")
    print(f"  Total in manifest:  {len(all_rows)}")
    print(f"  Output directory:   {output_dir}")
    print(f"  Manifest:           {rel(manifest_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to source video file (mp4, mov, avi, mkv). Spaces in path are fine.",
    )
    parser.add_argument(
        "--video-id",
        default=None,
        help="Short stable ID for this recording (default: video filename stem). "
             "Used as prefix for frame filenames and the output subdirectory.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Target frames per second to extract (default: 1.0 ≈ one frame/sec).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="DIR",
        help="Override output directory (default: data/processed/video_frames/<video-id>/).",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if the output directory already contains extracted frames.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print video metadata without extracting any frames.",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be a positive number")

    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(
            f"ERROR: Video file not found: {video_path}\n"
            "  • Use the --video flag with the exact path.\n"
            "  • If the path contains spaces, wrap it in quotes:\n"
            '    python scripts/extract_frames.py --video "path/with spaces/file.mp4"'
        )

    video_id = args.video_id or video_path.stem.replace(" ", "_")
    output_dir = Path(args.output) if args.output else VIDEO_FRAMES_ROOT / video_id

    extract(
        video_path=video_path,
        video_id=video_id,
        output_dir=output_dir,
        target_fps=args.fps,
        no_overwrite=args.no_overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
