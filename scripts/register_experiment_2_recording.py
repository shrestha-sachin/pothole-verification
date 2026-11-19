"""Extract reviewable frames from a new Green Bay video without changing it."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment_2_recordings import register_recording


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--recording-id', required=True)
    parser.add_argument('--split', required=True, choices=('train', 'val', 'test'))
    parser.add_argument('--area', required=True, help='Road or collection area')
    parser.add_argument('--camera', required=True, choices=('smartphone', 'dashcam'))
    parser.add_argument('--interval-seconds', type=float, default=2.0)
    parser.add_argument('--captured-at', default='', help='ISO 8601 date/time with offset, if known')
    parser.add_argument('--scene-group-id', default='', help='Known common scene group for repeat passes')
    parser.add_argument('--prior-tuning', action='store_true', help='Record previous use for parameter tuning')
    args = parser.parse_args()
    try:
        directory = register_recording(ROOT, args.video, args.recording_id, split=args.split,
            interval=args.interval_seconds, camera=args.camera, area=args.area, captured_at=args.captured_at,
            scene_group_id=args.scene_group_id, prior_tuning=args.prior_tuning)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f'Recording inventory: {directory / "inventory.csv"}')
    print(f'Open the existing annotation app with -- --collection {args.recording_id}')


if __name__ == '__main__':
    main()
