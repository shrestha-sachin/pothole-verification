"""Versioned Green Bay recording intake through the existing frame extractor."""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import re

from src.us_dataset import file_hash, read_csv, relative, resolve_source, write_csv

RECORDINGS = 'data/processed/experiment_2_recordings'


def collection_path(root, recording_id):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', recording_id):
        raise ValueError('Use a recording ID of up to 64 letters, digits, hyphens or underscores')
    root = Path(root).resolve()
    directory = Path(RECORDINGS)/recording_id
    return resolve_source(root, directory)


def register_recording(root, video, recording_id, *, split, interval, camera,
                       area, captured_at='', scene_group_id='', prior_tuning=False):
    root = Path(root).resolve()
    directory = collection_path(root, recording_id)
    video = resolve_source(root, video)
    if not video.is_relative_to(root/'data/raw/local_video') or not video.is_file():
        raise ValueError('Choose an existing original video under data/raw/local_video/')
    if split not in {'train', 'val', 'test'}:
        raise ValueError('Choose train, val, or test for the entire recording')
    if prior_tuning and split == 'test':
        raise ValueError('A tuning recording belongs in development. Record its prior use with --prior-tuning.')
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError('The frame interval must be a positive number of seconds')
    if captured_at:
        capture_date = datetime.fromisoformat(captured_at)
        if capture_date.tzinfo is None:
            raise ValueError('Add the time-zone offset to the capture date/time')
    if not area.strip():
        raise ValueError('Record the road or collection area')
    legacy_ids = {row['recording_id'].casefold() for row in read_csv(root/'data/metadata/image_inventory.csv')
                  if row.get('recording_id')}
    if {recording_id.casefold(), video.stem.replace(' ', '_').casefold()} & legacy_ids:
        raise ValueError('This recording is already in the original image inventory. Keep its existing split.')
    video_hash = file_hash(video)
    if legacy_ids:
        for original in (root/'data/raw/local_video').rglob('*'):
            original_id = original.stem.replace(' ', '_').casefold()
            if (original_id in legacy_ids and original.is_file() and original.stat().st_size == video.stat().st_size
                    and file_hash(original) == video_hash):
                raise ValueError(f'This video is a copy of {original.name}, which already has a split in the image inventory.')
    for path in sorted(directory.parent.glob('*/recording.json')):
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['video_sha256'] == video_hash:
            raise ValueError(f'This video is already registered as {saved["recording_id"]}')
        if scene_group_id and saved.get('scene_group_id') == scene_group_id and saved['split'] != split:
            raise ValueError('A known scene group cannot span recording splits')

    # Reserve a fresh directory before invoking the legacy writer. The legacy
    # extractor can append to a directory; this entry point always starts empty.
    directory.mkdir(parents=True, exist_ok=False)
    from scripts import extract_frames
    extract_frames.extract(video_path=video, video_id=recording_id, output_dir=directory,
                           target_fps=1/interval, no_overwrite=True, dry_run=False)
    frames = read_csv(directory/'frames.csv')
    if not frames or file_hash(video) != video_hash:
        raise ValueError('No complete frame extraction was produced, or the original video changed')
    inventory = []
    for row in frames:
        path = resolve_source(root, row['filepath'])
        identity = f"e2_{recording_id}_{int(row['frame_index']):07d}"
        inventory.append(dict(image_id=identity, filepath=row['filepath'], original_filename=path.name,
            source=camera, source_image_id=identity, recording_id=recording_id, video_id=recording_id,
            scene_group_id=scene_group_id, timestamp_ms=int(row['timestamp_ms']), frame_index=int(row['frame_index']),
            width=int(row['width']), height=int(row['height']), sha256=file_hash(path),
            image_variant='original extracted frame', license='Local collector', split=split,
            review_status='unreviewed', pothole_count=0, uncertain=False, notes='',
            shadows_present=False, patches_present=False, manholes_present=False))
    write_csv(directory/'inventory.csv', inventory)
    manifest = dict(recording_id=recording_id, video=relative(root, video), video_sha256=video_hash,
                    split=split, prior_tuning=prior_tuning, captured_at=captured_at, area=area,
                    camera=camera, scene_group_id=scene_group_id, interval_seconds=interval,
                    frames_sha256=file_hash(directory/'frames.csv'))
    # Presence of the recording manifest marks the collection ready to review.
    with (directory/'recording.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    return directory


def load_collection(root, directory, *, verify_video=False):
    root = Path(root).resolve()
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'recording.json').read_text(encoding='utf-8'))
    recording = manifest['recording_id']
    if directory != collection_path(root, recording) or manifest['split'] not in {'train', 'val', 'test'}:
        raise ValueError('Invalid collection identity or split')
    if manifest['split'] == 'test' and manifest['prior_tuning']:
        raise ValueError('Recording marked as previously tuned cannot be a protected final recording')
    frame_table = directory/'frames.csv'
    if file_hash(frame_table) != manifest['frames_sha256']:
        raise ValueError(f'Frame extraction manifest changed: {recording}')
    if verify_video:
        video = resolve_source(root, manifest['video'])
        if not video.is_relative_to(root/'data/raw/local_video') or file_hash(video) != manifest['video_sha256']:
            raise ValueError(f'Original recording changed: {recording}')
    frames = {row['frame_index']: row for row in read_csv(frame_table)}
    rows = read_csv(directory/'inventory.csv')
    seen = set()
    for row in rows:
        frame = frames.get(row['frame_index'])
        if row['image_id'] in seen or not frame or str(row['image_id']) != f"e2_{recording}_{int(row['frame_index']):07d}":
            raise ValueError(f'Duplicate or mismatched frame identity: {recording}')
        seen.add(row['image_id'])
        if (row['recording_id'] != recording or row['split'] != manifest['split']
                or row['scene_group_id'] != manifest.get('scene_group_id', '')
                or frame['video_id'] != recording
                or any(frame[key] != row[key] for key in ('filepath', 'timestamp_ms', 'width', 'height'))):
            raise ValueError(f'Frame, recording or split metadata changed: {row["image_id"]}')
        if not resolve_source(root, row['filepath']).is_relative_to(directory):
            raise ValueError('Collection frame escapes the extracted folder')
    if len(rows) != len(frames) or not rows:
        raise ValueError(f'Frame inventory is incomplete: {recording}')
    return manifest, rows


def collections(root, directory):
    if Path(directory).resolve() != (Path(root)/RECORDINGS).resolve():
        raise ValueError('Experiment 2 collections must use the standard recording directory')
    seen_video = set()
    for path in sorted(Path(directory).glob('*/recording.json')):
        manifest, rows = load_collection(root, path.parent, verify_video=True)
        if manifest['video_sha256'] in seen_video:
            raise ValueError('The same video appears in multiple recording collections')
        seen_video.add(manifest['video_sha256'])
        yield manifest, rows
