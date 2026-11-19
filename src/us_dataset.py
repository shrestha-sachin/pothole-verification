"""Conservative, CPU-only dataset audit and versioned staging primitives.

No ML framework, download, training, source mutation, or evaluation is performed.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath

from PIL import Image, ImageDraw
import yaml

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}
COUNTRIES = ('United_States', 'Norway', 'Czech', 'Japan', 'India', 'China')
CANDIDATE_COUNTRIES = {'Norway', 'Czech'}
CHARACTERISTICS = ('surface_material', 'surface_texture', 'road_markings', 'camera_height_angle',
                   'urban_rural', 'weather_illumination', 'shoulders', 'pothole_appearance',
                   'image_quality', 'annotation_accuracy')
LIMITATIONS = [
    'Development/comparison validation only: historical val assignments are reused; independence from older training is not established.',
    'RDD legacy group IDs are exact-duplicate/sample groups, not verified recording or route IDs. No route-independent validation claim.',
    'Historical test images are excluded, including exact and detected perceptual matches. No new final test is created.',
    '64-bit difference-hash distance <= 4 is a review heuristic, not comprehensive near-duplicate detection.',
    'Separately recorded, manually annotated Green Bay final videos must remain outside training and threshold selection.',
]


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    fields = fields or sorted({k for r in rows for k in r}) or ['status']
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def resolve_source(root, value):
    """Rebase old repo-absolute Windows/POSIX paths; never guess outside the repo."""
    root = Path(root).resolve()
    text = str(value).replace('\\', '/')
    if not text or text in {'.', '..'}:
        raise ValueError('Missing source path')
    foreign_absolute = PureWindowsPath(text).is_absolute() or text.startswith('/')
    if foreign_absolute:
        # Use an existing in-repository absolute path, or its stable repo suffix.
        candidate = Path(text)
        if candidate.is_absolute() and candidate.is_relative_to(root):
            text = candidate.relative_to(root).as_posix()
        else:
            match = re.search(r'(?:^|/)(data|exports|models)/', text)
            if not match:
                raise ValueError('Unportable external source path; provide a repo-relative path')
            text = text[match.start():].lstrip('/')
    result = (root / text).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Source path escapes repository')
    return result


def relative(root, path):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def normalize_country(value):
    key = re.sub(r'[^a-z]', '', value.lower())
    aliases = {'us': 'United_States', 'usa': 'United_States', 'unitedstates': 'United_States',
               'unitedstatesofamerica': 'United_States', 'czechrepublic': 'Czech', 'czechia': 'Czech'}
    aliases.update({re.sub(r'[^a-z]', '', c.lower()): c for c in COUNTRIES})
    if key not in aliases:
        raise ValueError(f'Unsupported RDD country: {value}')
    return aliases[key]


def validate_box(values, width=1, height=1):
    x, y, w, h = map(float, values)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError('nonfinite_or_zero_area_box')
    # Half a millionth tolerates serialization rounding, not annotation repair.
    eps = 0.0000005
    if x - w/2 < -eps or y - h/2 < -eps or x + w/2 > width + eps or y + h/2 > height + eps:
        raise ValueError('box_outside_image')
    return tuple(v for v in (x, y, w, h))


def parse_yolo(text, names, target='pothole'):
    """Validate ALL classes/boxes before mapping only an explicit target to zero."""
    boxes, classes = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or not fields[0].isdigit():
            raise ValueError('malformed_yolo_label')
        cls = int(fields[0])
        if cls not in names:
            raise ValueError('unknown_yolo_class')
        coords = validate_box(fields[1:])
        classes.append(names[cls])
        if names[cls].casefold() == target.casefold():
            boxes.append(coords)
    return boxes, sorted(set(classes))


def parse_voc(path, actual_size):
    node = ET.parse(path).getroot()
    size = (int(node.findtext('size/width')), int(node.findtext('size/height')))
    if size != actual_size or min(size) <= 0:
        raise ValueError('annotation_image_dimension_mismatch')
    boxes, classes = [], []
    width, height = size
    for obj in node.findall('object'):
        cls = obj.findtext('name', '').strip()
        if cls not in {'D00', 'D10', 'D20', 'D40'}:
            raise ValueError('unknown_rdd_class')
        b = obj.find('bndbox')
        if b is None:
            raise ValueError('missing_bounding_box')
        left, top, right, bottom = [float(b.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
        box = validate_box(((left+right)/2, (top+bottom)/2, right-left, bottom-top), width, height)
        classes.append(cls)
        if cls == 'D40':
            boxes.append(tuple(v / scale for v, scale in zip(box, (width, height, width, height))))
    return boxes, sorted(set(classes))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def image_fingerprints(path):
    with Image.open(path) as im:
        rgb = im.convert('RGB')
        pixel = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        small = rgb.convert('L').resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(small.get_flattened_data() if hasattr(small, 'get_flattened_data') else small.getdata())
        bits = 0
        for y in range(8):
            for x in range(8):
                bits = (bits << 1) | int(pixels[y*9+x] > pixels[y*9+x+1])
        return pixel, f'{bits:016x}'


def truth(value):
    return str(value).strip().lower() in {'true', '1', 'yes'}


def reviewed(row):
    return truth(row.get('approved')) and bool(row.get('reviewer', '').strip()) and bool(row.get('reviewed_at', '').strip())


def load_reviews(path):
    result = {}
    for row in read_csv(path) if path else []:
        key = (row.get('source', ''), row.get('original_id', ''))
        if not all(key) and not truth(row.get('approved')):
            continue  # Availability-only rows (e.g. absent Czech source) are not approvals.
        if not all(key) or key in result:
            raise ValueError('Review rows require unique source/original_id keys')
        result[key] = row
    return result


def roboflow_metadata(root, directory):
    directory = Path(directory)
    report = {'status': 'candidate_not_approved', 'path': relative(root, directory),
              'independent_original_image_count': 'unknown', 'sequence_metadata': 'not supplied by YOLO export',
              'origin_overlap': 'unknown until exact/perceptual scan; no negative claim of overlap',
              'license_status': 'export-declared; upstream ownership and attribution chain not independently verified'}
    if not (directory / 'data.yaml').is_file():
        return dict(report, available=False)
    data = yaml.safe_load((directory / 'data.yaml').read_text(encoding='utf-8'))
    names = data.get('names')
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    if data.get('nc') != len(names) or len([n for n in names.values() if str(n).casefold() == 'pothole']) != 1:
        raise ValueError('Roboflow class schema does not identify one explicit pothole class')
    report.update(available=True, metadata=data.get('roboflow', {}), classes=names, splits={},
                  preprocessing_and_augmentation=(directory/'README.roboflow.txt').read_text(encoding='utf-8') if (directory/'README.roboflow.txt').is_file() else 'unknown')
    for split in ('train', 'valid', 'test'):
        images = sorted(p for p in (directory/split/'images').glob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
        report['splits'][split] = {'images': len(images), 'label_files': len(list((directory/split/'labels').glob('*.txt')))}
    return report


class Audit:
    def __init__(self, root, recipe, countries, include_local=True, include_mapillary=True,
                 include_roboflow=False, reviews=None, seed=42, negative_ratio=0.25):
        self.root = Path(root).resolve()
        self.recipe = recipe
        self.countries = sorted({normalize_country(c) for c in countries})
        self.include_local, self.include_mapillary, self.include_roboflow = include_local, include_mapillary, include_roboflow
        self.reviews = load_reviews(reviews)
        self.seed, self.ratio = seed, negative_ratio
        if not math.isfinite(negative_ratio) or not 0 <= negative_ratio <= 2:
            raise ValueError('Training negatives per positive must be between 0 and 2')
        self.records, self.issues, self.duplicates, self.suitability = [], [], [], []
        self.protected = set(recipe.get('protected_recording_ids', []))
        self.historical = read_csv(resolve_source(self.root, recipe['historical_manifest']))
        if not self.historical:
            raise ValueError('Historical manifest is required to protect previous holdouts')
        self.history_counts = dict(Counter(r['split'] for r in self.historical))
        self.old_rdd = {r['original_id']: r for r in self.historical if r['source'].startswith('rdd_')}
        self.rdd_split = {r['image_id']: r for r in read_csv(resolve_source(self.root, recipe['rdd_split_metadata']))}

    def record(self, source, original_id, image, annotation='', **extra):
        row = dict(source=source, original_id=original_id, original_image=relative(self.root, image),
                   annotation=relative(self.root, annotation) if annotation else '', country='', split='', group='',
                   group_provenance='', recording_id='', sequence_id='', video_id='', timestamp_ms='',
                   variant='unspecified', license='', boxes=[], classes=[], reason='', **extra)
        self.records.append(row)
        return row

    def issue(self, row, reason):
        row['reason'] = reason
        self.issues.append({k: row.get(k, '') for k in ('source', 'original_id', 'original_image', 'annotation')} | {'finding': reason})

    def rdd(self):
        base = resolve_source(self.root, self.recipe['rdd_root'])
        # Folder presence determines spelling, never fuzzy substring matching.
        folders = {}
        for p in base.iterdir() if base.exists() else []:
            if p.is_dir():
                try:
                    folders[normalize_country(p.name)] = p
                except ValueError:
                    pass
        for country in COUNTRIES:
            folder = folders.get(country)
            if not folder:
                self.suitability.append(dict(source='rdd_'+country, original_id='', availability='not_on_disk', approved=False,
                                             reason='candidate_not_approved' if country in CANDIDATE_COUNTRIES else 'not_available'))
                continue
            train = folder / folder.name / 'train'
            if not train.is_dir():
                train = folder / 'train'
            images = sorted(p for p in (train/'images').glob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
            for image in images:
                annotation = train/'annotations/xmls'/f'{image.stem}.xml'
                row = self.record('rdd_'+country, image.stem, image, annotation)
                row.update(country=country, license='CC BY-SA 4.0; cite Arya et al. RDD2022; preserve attribution/share-alike', variant='RDD-distributed (may be privacy-blurred)')
                row['source_policy'] = 'enabled' if country in self.countries else 'candidate_not_approved' if country in CANDIDATE_COUNTRIES else 'excluded_by_policy'
                review = self.reviews.get((row['source'], row['original_id']), {})
                old = self.old_rdd.get(image.stem) or self.rdd_split.get(image.stem, {})
                row['split'] = old.get('split', '')
                row['group'] = 'rdd:' + country + ':' + old.get('group_id', old.get('group', image.stem))
                row['group_provenance'] = 'historical_sample_or_exact_duplicate_group; route_unknown'
                if reviewed(review) and review.get('recording_id'):
                    row['recording_id'] = review['recording_id']
                    row['group'] = 'identity:' + review['recording_id']
                    row['group_provenance'] = 'reviewed_recording_id' if reviewed(review) else 'unverified'
                if row['split'] == 'test':
                    row['reason'] = 'historical_test_protected'
                elif country not in self.countries:
                    row['reason'] = 'candidate_not_approved' if country in CANDIDATE_COUNTRIES else 'excluded_by_policy'
                elif country != 'United_States' and not (reviewed(review) and all(review.get(k, '').strip() for k in CHARACTERISTICS)):
                    row['reason'] = 'image_suitability_review_required'
                try:
                    with Image.open(image) as im:
                        row['width'], row['height'] = im.size
                    row['boxes'], row['classes'] = parse_voc(annotation, (row['width'], row['height']))
                    row['annotation_sha256'] = file_hash(annotation)
                    if not row['boxes'] and not row['reason'] and not (reviewed(review) and truth(review.get('target_annotation_complete'))):
                        row['reason'] = 'negative_completeness_not_reviewed'
                except (OSError, ValueError, TypeError, ET.ParseError) as exc:
                    self.issue(row, f'annotation_invalid:{exc}')
                if not row['reason'] and row['split'] not in {'train', 'val'}:
                    # No random image splitting to disguise missing route provenance.
                    if reviewed(review) and review.get('recording_id') and review.get('split') in {'train', 'val'}:
                        row['split'] = review['split']
                    else:
                        row['reason'] = 'split_or_recording_review_required'
                if country in CANDIDATE_COUNTRIES:
                    self.suitability.append(self.review_template(row, review))
            image_stems = {p.stem for p in images}
            for annotation in sorted((train/'annotations/xmls').glob('*.xml')):
                if annotation.stem not in image_stems:
                    self.issues.append(dict(source='rdd_'+country, original_id=annotation.stem,
                        annotation=relative(self.root, annotation), finding='annotation_without_image'))
            # Official unlabelled test data is never used as negative training data.
            for image in sorted(folder.rglob('*')):
                if image.suffix.lower() in IMAGE_SUFFIXES and 'test' in image.parts:
                    row = self.record('rdd_'+country, image.stem, image)
                    row.update(country=country, reason='official_unlabelled_test_excluded')

    def review_template(self, row, existing=None):
        template = {k: row.get(k, '') for k in ('source', 'original_id', 'original_image', 'country', 'classes', 'reason')}
        template.update(approved=False, reviewer='', reviewed_at='', target_annotation_complete=False,
                        recording_id='', split='', availability='on_disk')
        template.update({key: '' for key in CHARACTERISTICS})
        return template | (existing or {})

    def local(self):
        annotations = defaultdict(list)
        for path in resolve_source(self.root, self.recipe['annotations_root']).rglob('*.json'):
            annotations[path.stem].append(path)
        for meta in read_csv(resolve_source(self.root, self.recipe['inventory'])):
            source = 'mapillary' if meta.get('source') == 'mapillary' else 'local'
            try:
                image = resolve_source(self.root, meta.get('filepath', ''))
            except ValueError:
                self.issues.append(dict(source=source, original_id=meta.get('image_id'), finding='unresolvable_image_path'))
                continue
            matches = annotations.get(meta['image_id'], [])
            row = self.record(source, meta['image_id'], image, matches[0] if len(matches) == 1 else '')
            row.update(recording_id=meta.get('recording_id', ''), sequence_id=meta.get('sequence_id', ''),
                       video_id=meta.get('video_id', '') or meta.get('recording_id', ''), timestamp_ms=meta.get('timestamp_ms', ''),
                       license='Mapillary source terms; attribution verification required' if source == 'mapillary' else 'Local collection; permission/consent records retained by collector')
            if (source == 'mapillary' and not self.include_mapillary) or (source == 'local' and not self.include_local):
                row['reason'] = 'excluded_by_policy'
            try:
                if len(matches) != 1:
                    raise ValueError('missing_or_ambiguous_annotation_file')
                ann = json.loads(matches[0].read_text(encoding='utf-8'))
                row['annotation_sha256'] = file_hash(matches[0])
                if meta.get('review_status') != 'reviewed' or ann.get('review_status') != 'reviewed' or truth(meta.get('uncertain')) or truth(ann.get('uncertain')):
                    raise ValueError('unreviewed_or_uncertain_annotation')
                if 'pothole_boxes' not in ann or not isinstance(ann['pothole_boxes'], list):
                    raise ValueError('missing_explicit_pothole_review')
                if resolve_source(self.root, ann.get('source_filepath', '')) != image:
                    raise ValueError('annotation_source_image_disagreement')
                with Image.open(image) as im:
                    width, height = im.size
                if (ann.get('image_width'), ann.get('image_height')) != (width, height):
                    raise ValueError('annotation_image_dimension_mismatch')
                row.update(width=width, height=height, split=meta.get('split', ''),
                           variant=meta.get('image_variant') or 'inventory-selected; raw/redacted status not verified')
                for box in ann['pothole_boxes']:
                    x, y, w, h = [float(box[k]) for k in ('x', 'y', 'w', 'h')]
                    coords = validate_box((x+w/2, y+h/2, w, h), width, height)
                    row['boxes'].append(tuple(v/s for v, s in zip(coords, (width, height, width, height))))
                for box in ann.get('negative_regions', []):
                    x, y, w, h = [float(box[k]) for k in ('x', 'y', 'w', 'h')]
                    validate_box((x+w/2, y+h/2, w, h), width, height)
                    if box.get('subtype') not in {'shadow', 'repaired_patch', 'manhole', 'ordinary_road', 'other'}:
                        raise ValueError('ambiguous_negative_region_subtype')
                group_ids = [v for v in (row['recording_id'], row['sequence_id']) if v]
                if not group_ids:
                    raise ValueError('missing_recording_or_sequence')
                row['group'] = 'identity:' + group_ids[0]
                row['group_aliases'] = ['identity:' + v for v in group_ids]
                row['group_provenance'] = 'inventory_recording_or_sequence'
                if any(v in self.protected for v in group_ids) or row['split'] == 'test':
                    row['reason'] = 'protected_final_recording'
                elif row['split'] not in {'train', 'val'}:
                    raise ValueError('missing_valid_group_split')
                # Do not infer time from filenames or frame numbers when metadata lacks it.
                row['timestamp_ms'] = ann.get('timestamp_ms', row['timestamp_ms'])
                if row['timestamp_ms'] != '' and (isinstance(row['timestamp_ms'], bool) or not math.isfinite(float(row['timestamp_ms'])) or float(row['timestamp_ms']) < 0):
                    raise ValueError('invalid_timestamp_metadata')
                if ann.get('video_id') and row['recording_id'] and ann['video_id'] != row['recording_id']:
                    raise ValueError('annotation_recording_disagreement')
                row['video_id'] = ann.get('video_id') or row['video_id']
                if row['timestamp_ms'] == '' or not meta.get('image_variant'):
                    row['metadata_review_required'] = 'timestamp_missing_and/or_raw_redacted_status_unverified'
                    self.issues.append(dict(source=source, original_id=row['original_id'],
                        original_image=row['original_image'], annotation=row['annotation'], finding=row['metadata_review_required']))
                row['annotation_review'] = 'explicit_full_image_review; ground_truth_verified is a separate event-evaluation flag'
            except (OSError, ValueError, TypeError, KeyError) as exc:
                self.issue(row, str(exc))

    def roboflow(self):
        directory = resolve_source(self.root, self.recipe['roboflow_root'])
        self.rf_report = roboflow_metadata(self.root, directory)
        if not self.rf_report.get('available'):
            return
        names = self.rf_report['classes']
        dimensions, class_counts = Counter(), Counter()
        for split in ('train', 'valid', 'test'):
            for image in sorted((directory/split/'images').glob('*')):
                if image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                label = directory/split/'labels'/f'{image.stem}.txt'
                item_id = split + '/' + image.stem
                row = self.record('roboflow', item_id, image, label)
                row.update(reason='candidate_not_approved', original_partition=split,
                           license=str(self.rf_report.get('metadata', {}).get('license', 'unknown')))
                review = self.reviews.get(('roboflow', item_id), {})
                try:
                    with Image.open(image) as im:
                        row['width'], row['height'] = im.size
                    dimensions[f"{row['width']}x{row['height']}"] += 1
                    row['boxes'], row['classes'] = parse_yolo(label.read_text(encoding='utf-8'), names)
                    row['annotation_sha256'] = file_hash(label)
                    class_counts.update(row['classes'])
                    if self.include_roboflow and reviewed(review) and all(review.get(k, '').strip() for k in CHARACTERISTICS):
                        if not review.get('recording_id') or review.get('split') not in {'train', 'val'}:
                            raise ValueError('roboflow_group_and_split_review_required')
                        if not row['boxes'] and not truth(review.get('target_annotation_complete')):
                            raise ValueError('negative_completeness_not_reviewed')
                        row.update(reason='', split=review['split'], group='identity:'+review['recording_id'],
                                   recording_id=review['recording_id'], group_provenance='manual_source_review')
                    if split == 'test':
                        row['reason'] = 'roboflow_original_test_not_imported'
                except (OSError, ValueError, TypeError) as exc:
                    self.issue(row, str(exc))
                self.suitability.append(self.review_template(row, review))
        self.rf_report.update(dimensions=dict(dimensions), images_by_class=dict(class_counts))
        # Audit orphan labels as well as missing labels.
        for label in sorted(directory.glob('*/labels/*.txt')):
            if not any((label.parent.parent/'images'/f'{label.stem}{ext}').is_file() for ext in IMAGE_SUFFIXES):
                self.issues.append(dict(source='roboflow', original_id=label.stem, finding='label_without_image', annotation=relative(self.root, label)))

    def historical_holdouts(self):
        for old in self.historical:
            if old['split'] != 'test':
                continue
            image = resolve_source(self.root, old['image'])
            if not image.is_file():
                raise ValueError(f'Cannot verify historical protected image: {image}')
            row = self.record('historical_test', old['original_id'], image)
            row.update(reason='historical_test_protected', group=old['group'], split='test')

    def deduplicate(self):
        """Quarantine all members of conflicting/near-duplicate components, not just pairs."""
        pool = [r for r in self.records if not r['reason'] or r['source'] in {'historical_test', 'roboflow'} or r['reason'] == 'protected_final_recording']
        exact, perceptual = {}, []
        for row in pool:
            try:
                path = resolve_source(self.root, row['original_image'])
                row['file_sha256'] = file_hash(path)
                row['pixel_sha256'], row['dhash64'] = image_fingerprints(path)
            except (OSError, ValueError) as exc:
                if row['source'] == 'historical_test':
                    raise ValueError('Cannot fingerprint historical holdout') from exc
                self.issue(row, 'image_decode_failed')
                continue
            prior = exact.get(row['pixel_sha256'])
            if prior:
                kind = 'exact_decoded_pixels'
                self.duplicates.append(dict(kind=kind, left=prior['original_image'], right=row['original_image'], distance=0))
                # Never keep either member with conflicting labels; protected holdouts win.
                if prior['source'] == 'historical_test' or row['source'] == 'historical_test' or 'protected_final_recording' in {prior['reason'], row['reason']}:
                    for member in (prior, row):
                        if member['source'] != 'historical_test' and member['reason'] != 'protected_final_recording':
                            member['reason'] = 'exact_historical_test_overlap'
                elif prior['reason'] == 'conflicting_exact_duplicate_labels' or prior['boxes'] != row['boxes']:
                    prior['reason'] = row['reason'] = 'conflicting_exact_duplicate_labels'
                elif not prior['reason'] and not row['reason']:
                    row['reason'] = 'exact_duplicate'
                elif prior['reason'] in {'near_duplicate_requires_review', 'exact_historical_test_overlap'} and not row['reason']:
                    row['reason'] = prior['reason']
                continue
            exact[row['pixel_sha256']] = row
            bits = int(row['dhash64'], 16)
            for old_bits, previous in perceptual:
                distance = (bits ^ old_bits).bit_count()
                if distance <= 4:
                    self.duplicates.append(dict(kind='perceptual_candidate', left=previous['original_image'], right=row['original_image'], distance=distance))
                    # Conservative: quarantine matches until a later explicit review workflow.
                    if not row['reason']:
                        row['reason'] = 'near_duplicate_requires_review'
                    if not previous['reason']:
                        previous['reason'] = 'near_duplicate_requires_review'
            perceptual.append((bits, row))

    def group_check(self):
        # Known identifiers are global aliases, so one recording cannot cross sources.
        by_alias = defaultdict(list)
        for row in self.records:
            if not row['reason']:
                for alias in row.get('group_aliases', [row['group']]):
                    by_alias[alias].append(row)
                if row.get('recording_id') in self.protected or row.get('sequence_id') in self.protected:
                    row['reason'] = 'protected_final_recording'
        conflicts = []
        for alias, rows in by_alias.items():
            if len({r['split'] for r in rows}) > 1:
                conflicts.append(alias)
                for row in rows:
                    self.issue(row, 'group_split_conflict')
        self.group_conflicts = conflicts

    def select(self):
        rng = random.Random(self.seed)
        for source in sorted({r['source'] for r in self.records}):
            train = [r for r in self.records if r['source'] == source and r['split'] == 'train' and not r['reason']]
            negatives = sorted([r for r in train if not r['boxes']], key=lambda r: r['original_id'])
            rng.shuffle(negatives)
            budget = math.floor(sum(bool(r['boxes']) for r in train) * self.ratio)
            for row in negatives[budget:]:
                row['reason'] = 'training_negative_ratio_limit'
        return [r for r in self.records if not r['reason']]

    def run(self):
        self.rdd()
        self.local()
        self.roboflow()
        self.historical_holdouts()
        # Fingerprint protected records first; duplicates cannot displace holdouts.
        self.records.sort(key=lambda r: (r['source'] != 'historical_test' and r['reason'] != 'protected_final_recording',
            {'test': 0, 'val': 1, 'train': 2}.get(r['split'], 3), r['source'], r['original_id']))
        self.deduplicate()
        rf_paths = {r['original_image'] for r in self.records if r['source'] == 'roboflow'}
        rf_pairs = [r for r in self.duplicates if r['left'] in rf_paths or r['right'] in rf_paths]
        self.rf_report['duplicate_scan'] = dict(candidate_pairs=len(rf_pairs),
            cross_source_pairs=sum((r['left'] in rf_paths) != (r['right'] in rf_paths) for r in rf_pairs),
            limitation='Compared with eligible sources and all historical test images, not every excluded RDD image. Absence of a match does not establish independence.')
        self.group_check()
        selected = self.select()
        summary = dict(seed=self.seed, training_negative_ratio=self.ratio, historical_counts=self.history_counts,
            historical_sources=dict(Counter(r['source'] for r in self.historical)),
            selected_countries=self.countries, kaggle='excluded_by_policy', roboflow='candidate_not_approved' if not self.include_roboflow else 'per_image_review_required',
            evaluation='development_comparison_only; no final test', limitations=LIMITATIONS,
            exclusions=dict(Counter(r['reason'] for r in self.records if r['reason'])),
            annotation_issues=len(self.issues), duplicate_pairs=len(self.duplicates), group_conflicts=self.group_conflicts,
            configuration=self.recipe, effective_options=dict(include_local=self.include_local, include_mapillary=self.include_mapillary, include_roboflow=self.include_roboflow),
            pipeline_sha256=file_hash(Path(__file__)), python_version=sys.version.split()[0],
            packages={name: importlib.metadata.version(name) for name in ('Pillow', 'PyYAML')},
            input_manifest_sha256={key: file_hash(resolve_source(self.root, self.recipe[key]))
                for key in ('historical_manifest', 'rdd_split_metadata', 'inventory') if resolve_source(self.root, self.recipe[key]).is_file()},
            reviews_digest=hashlib.sha256(json.dumps(self.reviews_as_list(), sort_keys=True).encode()).hexdigest(),
            historical_test_pixel_hashes=sorted({r['pixel_sha256'] for r in self.records if r['source'] == 'historical_test' or r['reason'] == 'protected_final_recording' if 'pixel_sha256' in r}),
            splits={}, selected_countries_distribution=dict(Counter(r['country'] or 'local_or_mapillary' for r in selected)))
        for split in ('train', 'val'):
            rows = [r for r in selected if r['split'] == split]
            summary['splits'][split] = dict(images=len(rows), positive_images=sum(bool(r['boxes']) for r in rows),
                reviewed_negative_images=sum(not r['boxes'] for r in rows), pothole_boxes=sum(len(r['boxes']) for r in rows),
                by_source=dict(Counter(r['source'] for r in rows)))
        self.selected, self.summary = selected, summary
        return summary

    def reviews_as_list(self):
        return [self.reviews[k] for k in sorted(self.reviews)]


def contact_sheet(root, rows, path, seed=42, limit=24):
    if not rows:
        canvas = Image.new('RGB', (720, 100), 'white')
        ImageDraw.Draw(canvas).text((15, 35), 'No available images for this category.', fill='black')
        canvas.save(path)
        return
    # Round robin sources plus deterministic IDs avoids sampling only one scene.
    buckets = defaultdict(list)
    for row in rows:
        buckets[row['source'] + ':' + row.get('group', '')].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['original_id']}".encode()).hexdigest())
    sample = []
    while buckets and len(sample) < limit:
        for key in sorted(list(buckets), key=lambda k: hashlib.sha256(f'{seed}:{k}'.encode()).hexdigest()):
            sample.append(buckets[key].pop())
            if not buckets[key]:
                del buckets[key]
            if len(sample) == limit:
                break
    canvas = Image.new('RGB', (4*300, math.ceil(len(sample)/4)*240), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(sample):
        x, y = i % 4 * 300, i // 4 * 240
        try:
            with Image.open(resolve_source(root, row['original_image'])) as im:
                tile = im.convert('RGB')
            overlay = ImageDraw.Draw(tile)
            for cx, cy, w, h in row.get('boxes', []):
                overlay.rectangle(((cx-w/2)*tile.width, (cy-h/2)*tile.height, (cx+w/2)*tile.width, (cy+h/2)*tile.height), outline='red', width=max(2, tile.width//250))
            tile.thumbnail((294, 190))
            canvas.paste(tile, (x, y))
        except (OSError, ValueError):
            draw.text((x+5, y+30), 'Image unavailable', fill='red')
        draw.text((x+4, y+192), row['source'][:43], fill='black')
        draw.text((x+4, y+207), row['original_id'][:43], fill='black')
        draw.text((x+4, y+222), row.get('reason', '')[:43], fill='black')
    canvas.save(path)


def write_reports(audit, destination, reserved=False):
    destination = Path(destination)
    if not reserved:
        destination.mkdir(parents=True, exist_ok=False)
    (destination/'dataset_summary.json').write_text(json.dumps(audit.summary, indent=2, sort_keys=True), encoding='utf-8')
    write_csv(destination/'excluded.csv', [r for r in audit.records if r['reason']])
    write_csv(destination/'annotation_audit.csv', audit.issues,
              ['source', 'original_id', 'original_image', 'annotation', 'finding'])
    write_csv(destination/'duplicate_report.csv', audit.duplicates, ['kind', 'left', 'right', 'distance'])
    write_csv(destination/'source_suitability.csv', audit.suitability)
    statistics = []
    for source in sorted({r['source'] for r in audit.records} | {'kaggle_arcioni'} | {'rdd_'+c for c in COUNTRIES}):
        rows = [r for r in audit.records if r['source'] == source]
        chosen = [r for r in rows if not r['reason']]
        statistics.append(dict(source=source, discovered=len(rows), selected=len(chosen), positive=sum(bool(r['boxes']) for r in chosen),
                               reviewed_negative=sum(not r['boxes'] for r in chosen), pothole_boxes=sum(len(r['boxes']) for r in chosen),
                               historical_v1_contribution=audit.summary['historical_sources'].get({'local': 'local_video', 'mapillary': 'local_mapillary'}.get(source, source), 0),
                               status='excluded_by_policy' if source == 'kaggle_arcioni' else 'not_on_disk' if not rows else 'candidate_not_approved' if source in {'rdd_Norway', 'rdd_Czech', 'roboflow'} else 'see_per_image_audit',
                               license='; '.join(sorted({r['license'] for r in rows if r['license']}))))
    write_csv(destination/'source_statistics.csv', statistics)
    (destination/'roboflow_review.json').write_text(json.dumps(audit.rf_report, indent=2, sort_keys=True), encoding='utf-8')
    sheets = destination/'contact_sheets'
    sheets.mkdir()
    for country in sorted(CANDIDATE_COUNTRIES):
        contact_sheet(audit.root, [r for r in audit.records if r['country'] == country and r.get('width')], sheets/f'{country}_review.jpg', audit.seed)
    for name, rows in [('positive', [r for r in audit.selected if r['boxes']]),
                       ('reviewed_negative', [r for r in audit.selected if not r['boxes']]),
                       ('annotation_errors', [r for r in audit.records if r['reason'].startswith('annotation_invalid')]),
                       ('roboflow_candidate', [r for r in audit.records if r['source'] == 'roboflow'])]:
        contact_sheet(audit.root, rows, sheets/f'{name}.jpg', audit.seed)
    write_csv(destination/'selection_plan.csv', audit.selected)
    (destination/'README.md').write_text('# US-focused dataset audit\n\n' + '\n'.join('- '+x for x in LIMITATIONS) +
        '\n\n`source_suitability.csv` is a review template, not approval. Preserve source/original_id keys. '\
        'Candidate sources require an explicit CLI selection AND approved per-image review rows with reviewer, date, characteristics and recording/split metadata. '\
        'Empty non-target labels are not proof of a pothole-free image.\n', encoding='utf-8')


def version_paths(output, config):
    output, config = Path(output), Path(config)
    if not output.exists() and not config.exists():
        return output, config
    match = re.search(r'_v(\d+)$', output.name)
    base, number = (output.name[:match.start()], int(match[1])+1) if match else (output.name, 2)
    while True:
        new = output.with_name(f'{base}_v{number}')
        cfg = config.with_name(f'detector_{new.name.removeprefix("pothole_")}.yaml')
        if not new.exists() and not cfg.exists():
            return new, cfg
        number += 1


def stage(audit, output, config, *, report_writer=write_reports, finalize=None):
    output, config = Path(output), Path(config)
    if output.exists() or config.exists():
        raise FileExistsError('Output or configuration exists; choose a fresh version')
    if any(audit.summary['splits'][s]['positive_images'] == 0 for s in ('train', 'val')):
        raise ValueError('Audit has an empty or positive-free split; review exclusions before building')
    # Reserve configuration first; never overwrite an experiment, even on a race.
    config.parent.mkdir(parents=True, exist_ok=True)
    with config.open('x', encoding='utf-8') as f:
        f.write('# Reserved by dataset preparation; inspect build_status.json before training.\n')
    # Reserve the output before entering the failure handler. If another process
    # won this name, never change that process's dataset or completion status.
    output.mkdir(parents=True, exist_ok=False)
    try:
        report_writer(audit, output, reserved=True)
        snapshots = output/'preparation_source'
        snapshots.mkdir()
        shutil.copyfile(Path(__file__), snapshots/'us_dataset.py')
        manifest = []
        for row in audit.selected:
            split = row['split']
            image = resolve_source(audit.root, row['original_image'])
            if file_hash(resolve_source(audit.root, row['annotation'])) != row['annotation_sha256']:
                raise ValueError('Source annotation changed after audit; rerun the audit')
            # Stable relative identity avoids machine-path-dependent filenames.
            name = row['source'] + '_' + hashlib.sha256(row['original_image'].encode()).hexdigest()[:20]
            target = output/'images'/split/(name+image.suffix.lower())
            label = output/'labels'/split/(name+'.txt')
            target.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image, target)
            label.write_text(''.join('0 '+' '.join(f'{v:.8f}' for v in box)+'\n' for box in row['boxes']), encoding='utf-8')
            # Manifests store dataset-relative outputs and repo-relative sources.
            manifest.append({k: v for k, v in row.items() if k not in {'boxes', 'group_aliases'}} |
                            dict(image=target.relative_to(output).as_posix(), label=label.relative_to(output).as_posix(), potholes=len(row['boxes'])))
        write_csv(output/'manifest.csv', manifest)
        # Ultralytics resolves ./ list entries relative to the list's directory.
        for split in ('train', 'val'):
            (output/f'{split}.txt').write_text(''.join('./'+r['image']+'\n' for r in manifest if r['split'] == split), encoding='utf-8')
        # Ultralytics directory roots are machine-specific; the launcher resolves this
        # explicit project-relative root before invoking its data loader.
        data = dict(path=relative(audit.root, output), train='train.txt', val='val.txt', names={0: 'pothole'}, nc=1,
                    project_relative=True, evaluation_role='development_comparison_only')
        config.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
        verify(output, require_complete=False)
        if finalize is not None:
            finalize(output)
        (output/'build_status.json').write_text(json.dumps({'status': 'complete'}), encoding='utf-8')
    except Exception:
        # Never recursively delete partially published evidence; mark it incomplete.
        if output.exists():
            (output/'build_status.json').write_text(json.dumps({'status': 'incomplete'}), encoding='utf-8')
        raise


def verify(output, require_complete=True):
    output = Path(output).resolve()
    if require_complete and json.loads((output/'build_status.json').read_text()).get('status') != 'complete':
        raise ValueError('Dataset publication is incomplete')
    summary = json.loads((output/'dataset_summary.json').read_text())
    rows = read_csv(output/'manifest.csv')
    if not rows:
        raise ValueError('Missing or empty manifest')
    hashes, groups, actual = {}, {}, defaultdict(Counter)
    for row in rows:
        if row['split'] not in {'train', 'val'} or 'kaggle' in row['source'].lower():
            raise ValueError('Forbidden split or source')
        if any(row.get(k) in summary['configuration'].get('protected_recording_ids', []) for k in ('recording_id', 'sequence_id', 'video_id')):
            raise ValueError('Protected final recording in training dataset')
        image = (output/row['image']).resolve()
        label = (output/row['label']).resolve()
        if not image.is_relative_to(output) or not label.is_relative_to(output):
            raise ValueError('Output manifest escapes dataset')
        boxes, _ = parse_yolo(label.read_text(), {0: 'pothole'})
        if len(boxes) != int(row['potholes']):
            raise ValueError('Box count mismatch')
        if file_hash(image) != row['file_sha256']:
            raise ValueError('Image content changed')
        pixel, _ = image_fingerprints(image)
        if pixel != row['pixel_sha256'] or pixel in hashes:
            raise ValueError('Duplicate or modified decoded image')
        if pixel in summary['historical_test_pixel_hashes']:
            raise ValueError('Historical test contamination')
        hashes[pixel] = row['split']
        for group in [row['group']] + ['identity:'+row[k] for k in ('recording_id', 'sequence_id') if row.get(k)]:
            if not group or groups.setdefault(group, row['split']) != row['split']:
                raise ValueError('Group split leakage')
        actual[row['split']].update(images=1, positive_images=bool(boxes), reviewed_negative_images=not boxes, pothole_boxes=len(boxes))
    expected = summary['splits']
    for split in ('train', 'val'):
        if any(actual[split][k] != expected[split][k] for k in ('images', 'positive_images', 'reviewed_negative_images', 'pothole_boxes')):
            raise ValueError('Summary count mismatch')
        lines = (output/f'{split}.txt').read_text().splitlines()
        if sorted(lines) != sorted('./'+r['image'] for r in rows if r['split'] == split):
            raise ValueError('Image list mismatch')
    return {k: dict(v) for k, v in actual.items()}
