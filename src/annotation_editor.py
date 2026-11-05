"""Validated editor payloads and atomic inventory updates; independent of Streamlit."""
from __future__ import annotations
import csv
import math
import os
import tempfile
import uuid
from pathlib import Path

CATEGORIES = {'pothole', 'shadow', 'repaired_patch', 'manhole', 'ordinary_road'}


def editor_boxes(annotation):
    result = []
    for category, source in [('pothole', annotation.get('pothole_boxes', [])),
                             ('negative', annotation.get('negative_regions', []))]:
        for box in source:
            result.append({**box, 'id': box.get('_editor_id') or uuid.uuid4().hex,
                           'category': category if category == 'pothole' else box.get('subtype', 'ordinary_road')})
    return result


def apply_editor_boxes(annotation, boxes, width, height):
    """Validate the whole payload before updating; retain extra per-box metadata."""
    potholes, negatives, ids = [], [], set()
    for raw in boxes:
        category = raw.get('category')
        if category not in CATEGORIES:
            raise ValueError('Unknown annotation category')
        identity = str(raw.get('id', '')).strip()
        if not identity or identity in ids:
            raise ValueError('Every box must have a unique editor ID')
        ids.add(identity)
        coords = {}
        for key in ('x', 'y', 'w', 'h'):
            value = raw.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Invalid {key} coordinate')
            coords[key] = int(round(value))
        if (min(coords['x'], coords['y']) < 0 or min(coords['w'], coords['h']) < 1
                or coords['x']+coords['w'] > width or coords['y']+coords['h'] > height):
            raise ValueError('Box lies outside the original image')
        box = {k:v for k,v in raw.items() if k not in ('id', 'category', 'subtype')}
        box.update(coords, _editor_id=identity)
        if category == 'pothole':
            potholes.append(box)
        else:
            box['subtype'] = category
            negatives.append(box)
    return {**annotation, 'pothole_boxes': potholes, 'negative_regions': negatives}


def update_inventory(path: Path, image_id: str, updates: dict):
    """Publish a complete CSV atomically; preserve existing columns and other rows."""
    with path.open(newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        if row.get('image_id') == image_id:
            row.update({k:v for k,v in updates.items() if k in fields})
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8',
                dir=path.parent, suffix='.part', delete=False) as file:
            name = file.name
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()
