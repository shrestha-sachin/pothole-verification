"""Append-only, content-bound human dataset review events."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from src.us_dataset import CHARACTERISTICS, file_hash, relative, resolve_source


def pair_key(left, right):
    import hashlib
    return hashlib.sha256('\n'.join(sorted((left, right))).encode()).hexdigest()


def _bind_hash(event, field, path):
    actual = file_hash(path)
    if event.get(field) and event[field] != actual:
        raise ValueError('The source changed after the audit or display. Refresh the audit before reviewing it.')
    event[field] = actual


def save_event(root, directory, event):
    """Save a new decision, never edit an earlier review or a source annotation."""
    root, directory = Path(root).resolve(), Path(directory).resolve()
    if not directory.is_relative_to(root/'results/dataset_reviews'):
        raise ValueError('Review events must be below results/dataset_reviews/')
    event = dict(event)
    if not event.get('reviewer', '').strip() or not event.get('notes', '').strip():
        raise ValueError('A reviewer name and evidence notes are required')
    if event.get('kind') not in {'negative', 'source', 'duplicate'}:
        raise ValueError('Unknown review kind')
    if not event.get('inspection_confirmed'):
        raise ValueError('Confirm that the originals and annotation were inspected')
    event['original_image'] = relative(root, resolve_source(root, event['original_image']))
    _bind_hash(event, 'file_sha256', resolve_source(root, event['original_image']))
    if event['kind'] == 'duplicate':
        if event.get('decision') not in {'different_scenes', 'distinct_frames_same_training_group', 'same_or_related_scene', 'unsure'}:
            raise ValueError('Unknown duplicate decision')
        event['right_image'] = relative(root, resolve_source(root, event['right_image']))
        _bind_hash(event, 'right_sha256', resolve_source(root, event['right_image']))
        event['pair_id'] = pair_key(event['original_image'], event['right_image'])
    else:
        if not event.get('source') or not event.get('original_id'):
            raise ValueError('Image source and original ID are required')
        event['annotation'] = relative(root, resolve_source(root, event['annotation']))
        _bind_hash(event, 'annotation_sha256', resolve_source(root, event['annotation']))
        if event['kind'] == 'negative':
            if event.get('decision') not in {'pothole_free', 'pothole_present', 'unsure'}:
                raise ValueError('Unknown negative-review decision')
            if event['decision'] == 'pothole_free' and not event.get('target_annotation_complete'):
                raise ValueError('A pothole-free decision requires a complete-image cavity check')
        else:
            if event.get('decision') not in {'approved', 'not_approved', 'unsure'}:
                raise ValueError('Unknown source-review decision')
            if event['decision'] == 'approved' and (not event.get('annotation_definition_checked') or not all(str(event.get(k, '')).strip() for k in CHARACTERISTICS)):
                raise ValueError('Source approval requires annotation-definition and suitability checks')
        schema=resolve_source(root,event['annotation']).parent.parent.parent/'data.yaml'
        if event['source']=='roboflow':
            _bind_hash(event, 'source_metadata_sha256', schema)
    event.update(reviewed_at=datetime.now(timezone.utc).isoformat(), event_id=uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (event['reviewed_at'].replace(':', '-')+'_'+event['event_id']+'.json')
    with path.open('x', encoding='utf-8') as f:
        json.dump(event, f, indent=2, sort_keys=True)
    return path


def load_events(root, directory):
    """Latest decision per item wins, including revocations and unsure decisions."""
    root = Path(root).resolve()
    directory = Path(directory).resolve()
    if not directory.is_relative_to(root/'results/dataset_reviews'):
        raise ValueError('Review directory must be below results/dataset_reviews/')
    result = {}
    for path in sorted(directory.glob('*.json')):
        event = json.loads(path.read_text(encoding='utf-8'))
        if not event.get('reviewer') or not event.get('reviewed_at') or not event.get('inspection_confirmed'):
            raise ValueError(f'Incomplete review event: {path.name}')
        kind = event.get('kind')
        if kind == 'duplicate':
            key = (kind, pair_key(event['original_image'], event['right_image']))
        elif kind in {'negative', 'source'}:
            key = (kind, event['source'], event['original_id'])
        else:
            raise ValueError('Unsupported review kind')
        if key not in result or (event['reviewed_at'], event.get('event_id', '')) > (result[key]['reviewed_at'], result[key].get('event_id', '')):
            result[key] = event
    return result


def matches_inputs(root, row, event):
    if not event:
        return False
    try:
        return (event.get('original_image') == row['original_image'] and event.get('annotation') == row['annotation']
                and (not row.get('source_metadata_sha256') or event.get('source_metadata_sha256')==row['source_metadata_sha256'])
                and event.get('file_sha256') == file_hash(resolve_source(root, row['original_image']))
                and event.get('annotation_sha256') == file_hash(resolve_source(root, row['annotation'])))
    except OSError:
        return False
