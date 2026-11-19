"""Experiment 2 source policy, review gates, and reproducible preparation."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import importlib.metadata
from itertools import zip_longest
import json
import math
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET

from PIL import Image
import yaml

from src.dataset_reviews import load_events, matches_inputs, pair_key
from src.us_dataset import (
    Audit, CHARACTERISTICS, IMAGE_SUFFIXES, contact_sheet, file_hash,
    image_fingerprints, read_csv, relative, resolve_source, stage, truth,
    validate_box, verify, write_csv,
)

COUNTRIES = ('Japan', 'Norway')
SOURCES = ('rdd_Japan', 'rdd_Norway', 'local', 'mapillary', 'roboflow')
LIMITATIONS = [
    'Development validation retains historical assignments for approved sources. RDD sample groups do not establish route independence.',
    'Known recording, sequence, scene group and physical-defect identifiers are checked. Unknown identifiers and missed visual matches remain limitations.',
    'Published RDD D40 labels are used when structural checks pass. Independent visual verification of every publisher box has not been performed.',
    'The near-duplicate scan uses 64-bit difference hashes at distance <= 4. Visual review, capture metadata and separate local collections remain necessary.',
    'No final test split is published. Protected historical test fingerprints and registered final recordings are excluded.',
]
REVIEW_FIELDS = ('source', 'original_id', 'original_image', 'annotation', 'split', 'recording_id',
                 'classes', 'boxes', 'reason', 'annotation_sha256', 'file_sha256', 'source_metadata_sha256')
RDD_NEGATIVE_PILOT_SIZE = 144
RDD_CRACK_PROFILES = ('Alligator and other cracks', 'Alligator cracks', 'Longitudinal or transverse cracks',
                      'Other annotated damage', 'No annotated damage')


def yolo_text(boxes):
    return ''.join('0 ' + ' '.join(f'{value:.8f}' for value in box) + '\n' for box in boxes)


def parse_rdd(path, image_name, size, country):
    """Audit all boxes, preserve source coordinates, normalize only D40."""
    xml = ET.parse(path).getroot()
    dimensions = (int(xml.findtext('size/width')), int(xml.findtext('size/height')))
    if dimensions != size or min(size) <= 0:
        raise ValueError('annotation_image_dimension_mismatch')
    filename = (xml.findtext('filename') or '').replace('\\', '/').rsplit('/', 1)[-1]
    if filename and filename != image_name:
        raise ValueError('annotation_filename_mismatch')
    allowed = {'D00', 'D10', 'D20', 'D40'}
    if country == 'Japan':
        allowed.update({'D43', 'D44', 'D50'})
    width, height = size
    targets, classes, originals, seen = [], set(), [], set()
    difficult = False
    for item in xml.findall('object'):
        cls = item.findtext('name', '').strip()
        if cls not in allowed:
            raise ValueError(f'unsupported_rdd_class:{cls or "missing"}')
        box = item.find('bndbox')
        if box is None:
            raise ValueError('missing_bounding_box')
        source_box = [box.findtext(name) for name in ('xmin', 'ymin', 'xmax', 'ymax')]
        left, top, right, bottom = map(float, source_box)
        key = (cls, left, top, right, bottom)
        if key in seen:
            raise ValueError('duplicate_annotation_box')
        seen.add(key)
        normalized = validate_box(((left+right)/2, (top+bottom)/2, right-left, bottom-top), width, height)
        classes.add(cls)
        if cls == 'D40':
            targets.append(tuple(value/scale for value, scale in zip(normalized, (width,height,width,height))))
            originals.append(dict(class_name=cls, xmin=source_box[0], ymin=source_box[1],
                                  xmax=source_box[2], ymax=source_box[3], truncated=item.findtext('truncated')))
            difficult |= truth(item.findtext('difficult', '0'))
    # Identical geometry assigned to different classes, including D40, is ambiguous.
    geometry = Counter(key[1:] for key in seen)
    if any(geometry[key[1:]] > 1 for key in seen if key[0] == 'D40'):
        raise ValueError('conflicting_annotation_classes')
    return targets, sorted(classes), originals, difficult


def _group_ids(row):
    aliases = set(row.get('group_aliases', []))
    physical = row.get('physical_defect_keys') or []
    aliases.update(json.loads(physical) if isinstance(physical,str) else physical)
    if row.get('group'):
        aliases.add(row['group'])
    aliases.update('identity:'+row[key] for key in ('recording_id', 'sequence_id', 'scene_group_id') if row.get(key))
    return aliases


def sample_training(rows, ratio, seed):
    """Keep positives and round-robin negatives across sources and scene/confuser groups."""
    positives = [row for row in rows if row['boxes']]
    negatives = [row for row in rows if not row['boxes']]
    if ratio is None:
        return positives + negatives
    budget = min(len(negatives), math.floor(len(positives)*ratio))
    by_source = defaultdict(lambda: defaultdict(list))
    for row in negatives:
        by_source[row['source']][(row['group'], tuple(row.get('negative_subtypes', [])))].append(row)
    rng = random.Random(seed)
    cycles = {}
    for source, groups in sorted(by_source.items()):
        # Confuser-bearing groups enter the rotation before unspecified backgrounds.
        keys = sorted(groups, key=lambda key: (not bool(key[1]), key))
        for key in keys:
            groups[key].sort(key=lambda row: row['original_image'])
            rng.shuffle(groups[key])
        cycles[source] = deque(keys)
    chosen = []
    while len(chosen) < budget:
        for source in sorted(list(cycles)):
            if len(chosen) == budget:
                break
            cycle = cycles[source]
            key = cycle.popleft()
            chosen.append(by_source[source][key].pop())
            if by_source[source][key]:
                cycle.append(key)
            if not cycle:
                del cycles[source]
    return positives + chosen


class Experiment2Audit(Audit):
    def __init__(self, root, recipe, *, review_dir=None, negative_ratio=1.0, seed=42,
                 include_roboflow=True, include_rdd_negatives=True):
        if negative_ratio is not None and (not math.isfinite(negative_ratio) or not 0 <= negative_ratio <= 2):
            raise ValueError('Negative ratio must be between 0 and 2, or all')
        super().__init__(root, recipe, COUNTRIES, seed=seed, include_roboflow=include_roboflow)
        if not any(row['split'] == 'test' for row in self.historical):
            raise ValueError('Historical test references are required before preparing Experiment 2')
        self.ratio = negative_ratio
        self.include_rdd_negatives = include_rdd_negatives
        self.events = load_events(self.root, review_dir or self.root/'results/dataset_reviews/experiment_2')
        flags = read_csv(resolve_source(self.root, recipe['visual_review_flags'])) if recipe.get('visual_review_flags') else []
        self.visual_flags = {(row['source'], row['original_id']): row['finding'] for row in flags}
        self.review_warnings = []
        self.source_exclusions = {'RDD2022 United States': 'excluded_by_policy', 'RDD2022 India': 'excluded_by_policy',
            'RDD2022 China': 'excluded_by_policy', 'RDD2022 Czech': 'excluded_by_policy', 'Kaggle': 'excluded_by_policy'}
        self.metadata_rows = {}

    def event(self, row, kind):
        event = self.events.get((kind, row['source'], row['original_id']))
        if event and not matches_inputs(self.root, row, event):
            self.review_warnings.append(dict(source=row['source'], original_id=row['original_id'], finding='stale_review_inputs'))
            return None
        return event

    def negative_review(self, row):
        event = self.event(row, 'negative')
        if not event or event.get('decision') != 'pothole_free' or event.get('target_annotation_complete') is not True:
            return False
        row['negative_subtypes'] = sorted({value for value in event.get('negative_subtypes', [])
            if value in {'crack', 'shadow', 'repaired_patch', 'manhole', 'ordinary_road', 'other'}})
        row['negative_review_event'] = event['event_id']
        row['negative_evidence'] = 'whole_image_review_event'
        return True

    def rdd(self):
        base = resolve_source(self.root, self.recipe['rdd_root'])
        for country in COUNTRIES:
            directory = base/country
            if not directory.is_dir():
                self.issues.append(dict(source='rdd_'+country, finding='source_directory_missing'))
                continue
            nested = directory/country
            layout = nested if (nested/'train').is_dir() else directory
            images = sorted(p for p in (layout/'train/images').glob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
            xml_dir = layout/'train/annotations/xmls'
            ids = Counter(image.stem for image in images)
            for image in images:
                path = xml_dir/(image.stem+'.xml')
                row = self.record('rdd_'+country, image.stem, image, path)
                row.update(country=country, license='CC BY-SA 4.0 (RDD2022; credit Arya et al.)',
                           variant='published RDD image', annotation_state='invalid',
                           capture_group_known=False, original_partition='publisher_train')
                old = self.old_rdd.get(image.stem, {})
                split_meta = self.rdd_split.get(image.stem, {})
                known = old.get('split') or split_meta.get('split')
                group = split_meta.get('group_id') or old.get('group') or image.stem
                row.update(split=known or 'train', group=f'rdd:{country}:{group}',
                           group_provenance='historical_sample_group' if known else 'publisher_train_only_sample_fallback')
                try:
                    if ids[image.stem] > 1:
                        raise ValueError('multiple_images_for_one_annotation')
                    if old.get('split') and split_meta.get('split') and old['split'] != split_meta['split']:
                        raise ValueError('historical_split_disagreement')
                    with Image.open(image) as im:
                        row['width'], row['height'] = im.size
                    boxes, classes, originals, difficult = parse_rdd(path, image.name, (row['width'], row['height']), country)
                    row.update(boxes=boxes, classes=classes, original_target_boxes=originals,
                        annotation_sha256=file_hash(path), annotation_state='parsed')
                    review = self.event(row, 'source')
                    finding = self.visual_flags.get((row['source'], row['original_id']))
                    if review and review.get('recording_id') and review.get('decision') == 'approved':
                        row.update(recording_id=review['recording_id'], capture_group_known=True)
                    if known == 'test':
                        row['reason'] = 'historical_test_protected'
                    elif review and review.get('decision') in {'not_approved', 'unsure'}:
                        row['reason'] = 'annotation_marked_ambiguous'
                    elif finding and not (review and review.get('decision') == 'approved'
                                          and review.get('annotation_definition_checked') is True):
                        row['visual_review_note'] = finding
                        self.issue(row, 'visual_cavity_review_required')
                    elif difficult and not (review and review.get('decision') == 'approved'):
                        row['reason'] = 'difficult_cavity_annotation_requires_review'
                    elif not boxes and (not self.include_rdd_negatives or not self.negative_review(row)):
                        row['reason'] = 'negative_completeness_not_reviewed'
                    elif not boxes and not known:
                        decision = self.event(row, 'negative')
                        if decision.get('split') != 'train' or not decision.get('recording_id'):
                            row['reason'] = 'negative_recording_assignment_required'
                        else:
                            row.update(recording_id=decision['recording_id'], capture_group_known=True)
                except (OSError, ValueError, TypeError, ET.ParseError) as exc:
                    self.issue(row, f'annotation_invalid:{exc}' if path.is_file() else 'missing_source_annotation')
                    if known == 'test':
                        row['reason'] = 'historical_test_protected'
            for annotation in sorted(xml_dir.glob('*.xml')):
                if annotation.stem not in ids:
                    self.issues.append(dict(source='rdd_'+country, original_id=annotation.stem,
                        annotation=relative(self.root, annotation), finding='annotation_without_image'))
            for image in sorted((layout/'test').rglob('*')):
                if image.suffix.lower() in IMAGE_SUFFIXES and image.is_file():
                    row = self.record('rdd_'+country, image.stem, image)
                    row.update(country=country, reason='official_unlabelled_test_excluded', original_partition='publisher_test')

    def roboflow(self):
        try:
            super().roboflow()
        except (ValueError, OSError, TypeError, AttributeError, yaml.YAMLError) as exc:
            self.rf_report = {'available': False, 'status': 'export_schema_requires_review'}
            self.issues.append(dict(source='roboflow', finding=f'export_schema_invalid:{exc}'))
            return
        if not self.rf_report.get('available'):
            self.issues.append(dict(source='roboflow', finding='local_export_missing'))
            return
        metadata = self.rf_report['metadata']
        if (metadata.get('project'), metadata.get('workspace'), metadata.get('version')) != (
            'pothole-detection-3uuq2', 'the-american-university-of-the-middle-east', 2
        ) or self.rf_report['classes'].get(5, '').casefold() != 'pothole' or len(self.rf_report['classes']) != 6:
            self.rf_report['status'] = 'export_schema_requires_review'
            self.issues.append(dict(source='roboflow', finding='export_does_not_match_approved_version_and_class_schema'))
            return
        schema_hash = file_hash(resolve_source(self.root, self.recipe['roboflow_root'])/'data.yaml')
        self.rf_report.update(status='source_approved_for_consideration', target_source_class=5,
            license_attribution='Local export declares CC BY 4.0. Retain contributor/source URL, export version and changes; verify upstream attribution before redistribution.')
        for row in self.records:
            if row['source'] != 'roboflow':
                continue
            row['source_metadata_sha256'] = schema_hash
            row['annotation_state'] = 'invalid'
            if not row.get('annotation_sha256'):
                continue  # The shared parser already recorded the missing/malformed label.
            text = resolve_source(self.root, row['annotation']).read_text(encoding='utf-8')
            lines = [(int(fields[0]), *(float(value) for value in fields[1:]))
                     for line in text.splitlines() if (fields := line.split())]
            if len(set(lines)) != len(lines):
                self.issue(row, 'duplicate_annotation_box')
                continue
            target_geometry = {line[1:] for line in lines if line[0] == 5}
            if any(line[0] != 5 and line[1:] in target_geometry for line in lines):
                self.issue(row, 'conflicting_annotation_classes')
                continue
            row['annotation_state'] = 'parsed'
            row['original_target_boxes'] = [line for line in lines if line[0] == 5]
            if row['original_partition'] == 'test':
                row['reason'] = 'roboflow_original_test_excluded'
                continue
            if not self.include_roboflow:
                row['reason'] = 'roboflow_excluded_by_option'
                continue
            event = self.event(row, 'source')
            if not (event and event.get('decision') == 'approved' and event.get('annotation_definition_checked') is True
                    and all(str(event.get(field, '')).strip() for field in CHARACTERISTICS)):
                row['reason'] = 'roboflow_image_review_required'
                continue
            if event.get('split') != 'train' or not event.get('recording_id'):
                row['reason'] = 'roboflow_capture_group_required'
                continue
            if not row['boxes'] and not self.negative_review(row):
                row['reason'] = 'negative_completeness_not_reviewed'
                continue
            row.update(split='train', recording_id=event['recording_id'], group='identity:'+event['recording_id'],
                       group_provenance='reviewed_capture_group', reason='', source_approval_event=event['event_id'])

    def local(self):
        inventory = resolve_source(self.root, self.recipe['inventory'])
        annotations = defaultdict(list)
        annotation_root = resolve_source(self.root, self.recipe['annotations_root'])
        for path in sorted(annotation_root.rglob('*.json')):
            if path.relative_to(annotation_root).parts[0] != 'experiment_2':
                annotations[path.stem].append(path)
        for meta in read_csv(inventory):
            if meta.get('source') not in {'local', 'video', 'mapillary', 'smartphone', 'dashcam'}:
                self.issues.append(dict(source=meta.get('source', ''), original_id=meta.get('image_id', ''), finding='inventory_source_excluded'))
                continue
            source = 'mapillary' if meta['source'] == 'mapillary' else 'local'
            matches = annotations[meta['image_id']]
            self._local_image(meta, source, matches[0] if len(matches) == 1 else None)

        from src.experiment_2_recordings import collections
        for recording, rows in collections(self.root, resolve_source(self.root, self.recipe['recordings_root'])):
            self.metadata_rows[recording['recording_id']] = recording
            if recording['split'] == 'test':
                self.protected.add(recording['recording_id'])
            for meta in rows:
                path = self.root/'data/annotations/experiment_2'/recording['recording_id']/(meta['image_id']+'.json')
                self._local_image(meta, 'local', path, recording)

    def _local_image(self, meta, source, annotation, recording=None):
        identity = meta['image_id']
        try:
            path = resolve_source(self.root, meta['filepath'])
        except ValueError as exc:
            self.issues.append(dict(source=source, original_id=identity, finding=f'invalid_source_path:{exc}'))
            return
        row = self.record(source, identity, path, annotation)
        row.update(recording_id=meta.get('recording_id', ''), sequence_id=meta.get('sequence_id', ''),
                   scene_group_id=meta.get('scene_group_id', ''), split=meta.get('split', ''),
                   source_image_id=meta.get('source_image_id', ''), variant=meta.get('image_variant') or 'existing inventory image',
                   license=meta.get('license') or ('Mapillary source terms and attribution' if source == 'mapillary' else 'Local collector'),
                   annotation_state='unreviewed', review_status=meta.get('review_status', 'unreviewed'))
        group = row['recording_id'] or row['sequence_id'] or row['scene_group_id']
        if group:
            row.update(group='identity:'+group, group_provenance='recording_manifest' if recording else 'reviewed_inventory_assignment')
        if recording:
            row.update(original_video=recording['video'], source_video_sha256=recording['video_sha256'],
                       frame_index=meta['frame_index'], prior_tuning=recording['prior_tuning'])
        protected = row['split'] == 'test' or any(row[key] and row[key] in self.protected
            for key in ('recording_id', 'sequence_id', 'scene_group_id'))
        try:
            if not annotation or not annotation.is_file():
                raise ValueError('annotation_missing_or_ambiguous')
            ann = json.loads(annotation.read_text(encoding='utf-8'))
            if ann.get('review_status') != 'reviewed' or row['review_status'] != 'reviewed' or truth(ann.get('uncertain')) or truth(meta.get('uncertain')):
                row['review_status'] = 'uncertain' if (truth(ann.get('uncertain')) or truth(meta.get('uncertain'))
                     or 'uncertain' in {ann.get('review_status'), row['review_status']}) else 'unreviewed'
                raise ValueError('unreviewed_or_uncertain_annotation')
            if ann.get('image_id') != identity or resolve_source(self.root, ann.get('source_filepath', '')) != path:
                raise ValueError('annotation_source_image_disagreement')
            if meta.get('sha256') and file_hash(path) != meta['sha256']:
                raise ValueError('inventory_image_hash_changed')
            with Image.open(path) as im:
                width, height = im.size
            if (ann.get('image_width'), ann.get('image_height')) != (width, height):
                raise ValueError('annotation_image_dimension_mismatch')
            if not isinstance(ann.get('pothole_boxes'), list):
                raise ValueError('missing_explicit_pothole_review')
            if meta.get('pothole_count') not in ('', None) and int(meta['pothole_count']) != len(ann['pothole_boxes']):
                raise ValueError('inventory_annotation_box_count_disagreement')
            if not group or row['split'] not in {'train', 'val', 'test'}:
                raise ValueError('recording_or_sequence_assignment_required')
            video = ann.get('video_id', '') or row['recording_id']
            if row['recording_id'] and video != row['recording_id']:
                raise ValueError('annotation_recording_disagreement')
            timestamp = ann.get('timestamp_ms', '')
            if timestamp != '' and (isinstance(timestamp, bool) or not math.isfinite(float(timestamp)) or float(timestamp) < 0):
                raise ValueError('invalid_timestamp_metadata')
            if recording and (ann.get('video_id') != row['recording_id'] or timestamp == '' or float(timestamp) != int(meta['timestamp_ms'])):
                raise ValueError('recording_frame_timestamp_mismatch')
            normalized, physical, confusers = [], [], set()
            for source_box in ann['pothole_boxes']:
                normalized.append(_native_box(source_box, width, height))
                if source_box.get('physical_defect_id'):
                    physical.append('physical:'+str(source_box['physical_defect_id']))
                if source_box.get('gt_event_id') and video:
                    physical.append(f"event:{video}:{source_box['gt_event_id']}")
            if len(set(normalized)) != len(normalized):
                raise ValueError('duplicate_annotation_box')
            negative_boxes = set()
            for neg in ann.get('negative_regions', []):
                normalized_negative = _native_box(neg, width, height)
                if neg.get('subtype') not in {'shadow', 'repaired_patch', 'manhole', 'ordinary_road', 'other', 'crack'}:
                    raise ValueError('ambiguous_negative_region_subtype')
                key = (neg['subtype'], normalized_negative)
                if key in negative_boxes or normalized_negative in normalized:
                    raise ValueError('duplicate_or_conflicting_non_target_box')
                negative_boxes.add(key)
                confusers.add(neg['subtype'])
            for flag, subtype in [('shadows_present', 'shadow'), ('patches_present', 'repaired_patch'), ('manholes_present', 'manhole')]:
                if truth(ann.get(flag)):
                    confusers.add(subtype)
            if recording and not normalized and ann.get('target_annotation_complete') is not True:
                raise ValueError('whole_image_negative_review_required')
            row.update(width=width, height=height, boxes=normalized, physical_defect_keys=sorted(set(physical)),
                       photo_id=row['source_image_id'], annotation_state='parsed', original_target_boxes=ann['pothole_boxes'],
                       video_id=video, timestamp_ms=timestamp, negative_subtypes=sorted(confusers),
                       annotation_sha256=file_hash(annotation), negative_evidence='native_whole_image_review' if not normalized else '')
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.issue(row, str(exc))
        if protected:
            row['reason'] = 'protected_final_recording'

    def group_check(self):
        """Check connected recording, sequence, scene and defect aliases before sampling."""
        groups = defaultdict(list)
        for row in self.records:
            if not row['reason'] or row['reason'] in {'protected_final_recording', 'historical_test_protected'}:
                for alias in _group_ids(row):
                    groups[alias].append(row)
        remaining = set(groups)
        self.group_conflicts = []
        for initial in sorted(groups):
            if initial not in remaining:
                continue
            pending = [initial]
            component, aliases = {}, set()
            while pending:
                alias = pending.pop()
                if alias not in remaining:
                    continue
                remaining.remove(alias)
                aliases.add(alias)
                for row in groups[alias]:
                    component[id(row)] = row
                    pending.extend(_group_ids(row) & remaining)
            splits = {row['split'] for row in component.values() if row['split'] in {'train', 'val', 'test'}}
            if any(row['reason'] == 'protected_final_recording' for row in component.values()):
                splits.add('test')
            if len(splits) > 1:
                self.group_conflicts.append(dict(groups=sorted(aliases), splits=sorted(splits), images=len(component)))
                for row in component.values():
                    if not row['reason']:
                        self.issue(row, 'recording_sequence_or_defect_split_conflict')

    def historical_references(self):
        super().historical_holdouts()
        for old in self.historical:
            if old['split'] != 'val':
                continue
            image = resolve_source(self.root, old['image'])
            if not image.is_file():
                raise ValueError(f'Missing historical validation reference: {image}')
            row = self.record('historical_val_reference', old['original_id'], image)
            row.update(split='val', reason='historical_validation_fingerprint_reference')

    def _pair(self, first, second, kind, distance, resolution):
        self.duplicates.append(dict(kind=kind, left=first['original_image'], right=second['original_image'],
            left_source=first['source'], right_source=second['source'], left_split=first['split'], right_split=second['split'],
            left_group=first['group'], right_group=second['group'], left_file_sha256=first.get('file_sha256', ''),
            right_file_sha256=second.get('file_sha256', ''), distance=distance, resolution=resolution,
            pair_id=pair_key(first['original_image'], second['original_image'])))

    def _released_pair(self, left, right):
        event = self.events.get(('duplicate', pair_key(left['original_image'], right['original_image'])))
        if not event:
            return False
        content = {row['original_image']: row['file_sha256'] for row in (left, right)}
        if (set(content) != {event.get('original_image'), event.get('right_image')}
                or content.get(event.get('original_image')) != event.get('file_sha256')
                or content.get(event.get('right_image')) != event.get('right_sha256')):
            return False
        return event.get('decision') == 'different_scenes' or (
            event.get('decision') == 'distinct_frames_same_training_group'
            and left['split'] == right['split'] == 'train' and bool(_group_ids(left) & _group_ids(right)))

    def deduplicate(self):
        """Fingerprint all eligible images and holdouts. Keep review-only RF isolated."""
        protected = {'historical_test_protected', 'protected_final_recording'}
        references = {'historical_test', 'historical_val_reference'}
        pool = [row for row in self.records if not row['reason'] or row['source'] in references
                or row['reason'] == 'protected_final_recording' or
                (row['source'] == 'roboflow' and row.get('annotation_state') == 'parsed')]
        order = {'test': 0, 'val': 1, 'train': 2}
        pool.sort(key=lambda row: (order.get(row['split'], 3), row['source'], row['original_image']))
        clusters = defaultdict(list)
        for row in pool:
            try:
                path = resolve_source(self.root, row['original_image'])
                row['file_sha256'] = file_hash(path)
                row['pixel_sha256'], row['dhash64'] = image_fingerprints(path)
                clusters[row['pixel_sha256']].append(row)
            except (OSError, ValueError) as exc:
                if row['source'] in references or row['reason'] in protected:
                    raise ValueError(f'Could not fingerprint a protected reference: {row["original_image"]}') from exc
                self.issue(row, 'image_decode_failed')

        test_hashes, previous_val_hashes, candidates, anchors, review_only = set(), set(), [], [], []
        for digest, cluster in clusters.items():
            active = [row for row in cluster if not row['reason']]
            held_out = next((row for row in cluster if row['reason'] in protected), None)
            old_val = next((row for row in cluster if row['source'] == 'historical_val_reference'), None)
            if held_out:
                test_hashes.add(digest)
                anchors.append((held_out, 'protected_test'))
            if old_val:
                previous_val_hashes.add(digest)
                anchors.append((old_val, 'historical_validation'))
            for row in active:
                reference = held_out or (old_val if row['split'] == 'train' else None)
                if reference:
                    row['reason'] = 'exact_protected_test_overlap' if held_out else 'exact_historical_validation_overlap'
                    self._pair(reference, row, 'exact_decoded_pixels', 0, row['reason'])
            active = [row for row in active if not row['reason']]
            if active:
                first, *rest = active
                labels = {tuple(sorted(yolo_text(row['boxes']).splitlines())) for row in active}
                if len(labels) > 1:
                    first['reason'] = 'conflicting_exact_duplicate_labels'
                else:
                    candidates.append(first)
                for row in rest:
                    row['reason'] = first['reason'] or 'exact_duplicate'
                    self._pair(first, row, 'exact_decoded_pixels', 0, row['reason'])
            if any(row['source'] == 'roboflow' and row['reason'] for row in cluster):
                review_only.extend(row for row in cluster if row['source'] == 'roboflow' and row['reason']
                                   and row['annotation_state'] == 'parsed')

        # Compare all training/validation candidates before negative sampling. A
        # removed row still contributes pairs so a chain of matches cannot escape.
        fingerprints = [(int(row['dhash64'], 16), row) for row in candidates]
        for index, (bits, left) in enumerate(fingerprints):
            for other_bits, right in fingerprints[index+1:]:
                distance = (bits ^ other_bits).bit_count()
                if distance > 4:
                    continue
                released = self._released_pair(left, right)
                self._pair(left, right, 'perceptual_candidate', distance, 'reviewed_distinct' if released else 'quarantined')
                if not released:
                    left['reason'] = right['reason'] = 'near_duplicate_requires_review'
        for bits, row in fingerprints:
            for reference, kind in anchors:
                if (kind == 'historical_validation' and row['split'] != 'train') or reference['pixel_sha256'] == row['pixel_sha256']:
                    continue
                distance = (bits ^ int(reference['dhash64'], 16)).bit_count()
                if distance <= 4:
                    row['reason'] = 'suspected_'+kind+'_overlap'
                    self._pair(reference, row, 'perceptual_candidate', distance, row['reason'])
        review_references = [row for _, row in fingerprints] + [row for row, _ in anchors]
        audit_pairs = set()
        for left in review_only:
            for right in review_references + review_only:
                if left is right:
                    continue
                key = pair_key(left['original_image'], right['original_image'])
                if key in audit_pairs:
                    continue
                distance = (int(left['dhash64'], 16) ^ int(right['dhash64'], 16)).bit_count()
                if distance <= 4:
                    kind = 'exact_decoded_pixels' if left['pixel_sha256'] == right['pixel_sha256'] else 'perceptual_candidate'
                    self._pair(left, right, kind, distance, 'roboflow_review_only')
                    audit_pairs.add(key)
        self.test_hashes = sorted(test_hashes)
        self.previous_val_hashes = sorted(previous_val_hashes)

    def run(self):
        self.rdd()
        self.local()
        self.roboflow()
        self.historical_references()
        self.group_check()
        self.deduplicate()
        self.eligible = [row for row in self.records if not row['reason']]
        training = [row for row in self.eligible if row['split'] == 'train']
        validation = [row for row in self.eligible if row['split'] == 'val']
        chosen = sample_training(training, self.ratio, self.seed)
        kept = {id(row) for row in chosen}
        for row in training:
            if id(row) not in kept:
                row['reason'] = 'training_negative_ratio_limit'
        self.selected = sorted(chosen + validation, key=lambda row: (row['split'], row['source'], row['original_image']))
        self.ratio_comparisons = [dict(requested_ratio=ratio if ratio is not None else 'all',
            **split_counts(sample_training(training, ratio, self.seed)), val_images=len(validation))
            for ratio in (0.25, 0.5, 1.0, None)]
        inventory = {key: file_hash(resolve_source(self.root, self.recipe[key]))
                     for key in ('historical_manifest', 'inventory', 'rdd_split_metadata', 'visual_review_flags')
                     if self.recipe.get(key) and resolve_source(self.root, self.recipe[key]).is_file()}
        self.summary = dict(experiment='pothole_experiment_2', seed=self.seed,
            requested_negative_ratio=self.ratio if self.ratio is not None else 'all',
            splits={'train': split_counts(chosen), 'val': split_counts(validation)},
            configuration={**self.recipe, 'protected_recording_ids': sorted(self.protected)},
            approved_sources=list(SOURCES), excluded_sources=self.source_exclusions,
            excludes=dict(Counter(row['reason'] for row in self.records if row['reason'] and row['source'] in SOURCES)),
            historical_test_pixel_hashes=self.test_hashes, historical_validation_pixel_hashes=self.previous_val_hashes,
            historical_v1_split_counts=self.history_counts,
            validation_fingerprint=validation_digest(validation),
            review_event_count=len(self.events), input_manifest_sha256=inventory,
            pipeline_sha256=file_hash(Path(__file__)),
            dependencies={name:importlib.metadata.version(name) for name in ('Pillow','PyYAML')},
            options=dict(include_roboflow=self.include_roboflow, include_rdd_negatives=self.include_rdd_negatives),
            reviewed_capture_metadata=self.metadata_rows,
            limitations=LIMITATIONS)
        return self.summary


def _native_box(box, width, height):
    x, y, w, h = [float(box[key]) for key in ('x', 'y', 'w', 'h')]
    normalized = validate_box((x+w/2, y+h/2, w, h), width, height)
    return tuple(value/scale for value, scale in zip(normalized, (width,height,width,height)))


def split_counts(rows):
    positives = sum(bool(row['boxes']) for row in rows)
    negatives = len(rows) - positives
    return dict(images=len(rows), positive_images=positives, reviewed_negative_images=negatives,
                pothole_boxes=sum(len(row['boxes']) for row in rows),
                realized_negative_per_positive=negatives/positives if positives else None,
                by_source=dict(Counter(row['source'] for row in rows)))


def validation_digest(rows):
    """Validation membership, pixels and label bytes must agree across ratio builds."""
    identity = sorted((row['source'], row['original_id'], row['pixel_sha256'], row['file_sha256'],
                      row.get('label_sha256') or hashlib.sha256(yolo_text(row['boxes']).encode()).hexdigest()) for row in rows)
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def split_report(audit):
    by_split = {split: [row for row in audit.selected if row['split'] == split] for split in ('train', 'val')}
    group_sets = {split: set().union(*(_group_ids(row) for row in rows)) for split, rows in by_split.items()}
    current = {split: {row['pixel_sha256'] for row in rows} for split, rows in by_split.items()}
    group_overlap = sorted(group_sets['train'] & group_sets['val'])
    pixel_overlap = sorted(current['train'] & current['val'])
    protected_overlap = sorted((current['train'] | current['val']) & set(audit.test_hashes))
    old_validation_overlap = sorted(current['train'] & set(audit.previous_val_hashes))
    return dict(passes=not (group_overlap or pixel_overlap or protected_overlap or old_validation_overlap),
        selected_group_overlap=group_overlap, selected_pixel_overlap=pixel_overlap,
        historical_test_or_registered_final_overlap=protected_overlap,
        historical_validation_in_training=old_validation_overlap, group_conflicts_quarantined=audit.group_conflicts,
        near_duplicate_pairs=sum(pair['kind'] == 'perceptual_candidate' for pair in audit.duplicates),
        exact_pairs=sum(pair['kind'] == 'exact_decoded_pixels' for pair in audit.duplicates),
        unresolved_mapillary_images=sum(r['source'] == 'mapillary' and r['reason'] == 'near_duplicate_requires_review' for r in audit.records),
        protected_pixel_hashes=len(audit.test_hashes), validation_fingerprint=audit.summary['validation_fingerprint'],
        validation_role='development_comparison_only', final_test_published=False, limitations=LIMITATIONS)


def source_statistics(audit):
    summaries = []
    for source in SOURCES:
        rows = [row for row in audit.records if row['source'] == source]
        train, val = [[row for row in audit.selected if row['source'] == source and row['split'] == split]
                      for split in ('train', 'val')]
        summaries.append(dict(source=source, discovered=len(rows),
            parsed_positive_candidates=sum(bool(row['boxes']) and row.get('annotation_state') == 'parsed' for row in rows),
            parsed_empty_target_candidates=sum(not row['boxes'] and row.get('annotation_state') == 'parsed' for row in rows),
            training_positives=sum(bool(row['boxes']) for row in train), training_negatives=sum(not row['boxes'] for row in train),
            validation_positives=sum(bool(row['boxes']) for row in val), validation_negatives=sum(not row['boxes'] for row in val),
            pothole_boxes=sum(len(row['boxes']) for row in train + val), excluded=len(rows)-len(train)-len(val),
            unreviewed_or_uncertain=sum(row.get('review_status') in {'unreviewed','uncertain'} or row['reason'] in
                {'negative_completeness_not_reviewed','roboflow_image_review_required','annotation_marked_ambiguous',
                 'difficult_cavity_annotation_requires_review','visual_cavity_review_required','unreviewed_or_uncertain_annotation'} for row in rows),
            exclusion_reasons=dict(Counter(row['reason'] for row in rows if row['reason'])),
            licenses=sorted({row['license'] for row in rows if row['license']})))
    return summaries


def rdd_crack_profile(classes):
    """Describe publisher crack labels for prioritization only."""
    cracks = {'D00', 'D10', 'D20'}.intersection(classes)
    if 'D20' in cracks:
        return RDD_CRACK_PROFILES[0] if len(cracks) > 1 else RDD_CRACK_PROFILES[1]
    if cracks:
        return RDD_CRACK_PROFILES[2]
    return RDD_CRACK_PROFILES[3] if classes else RDD_CRACK_PROFILES[4]


def _interleave_review_rows(queues):
    for group in zip_longest(*queues):
        yield from (row for row in group if row is not None)


def rdd_negative_pilot(candidates, seed):
    """Rotate across countries first, then published crack profiles within each country."""
    buckets = defaultdict(lambda: defaultdict(list))
    for row in candidates:
        if row['split'] == 'train' and row['group_provenance'] == 'historical_sample_group':
            buckets[row['source']][row['published_crack_profile']].append(row)
    country_queues = []
    for _, profiles in sorted(buckets.items()):
        for rows in profiles.values():
            rows.sort(key=lambda row: (
                hashlib.sha256(f"{seed}:{row['original_id']}".encode()).digest(), row['original_image']))
        country_queues.append(_interleave_review_rows([profiles[p] for p in RDD_CRACK_PROFILES if p in profiles]))
    sample, seen_groups = [], set()
    for row in _interleave_review_rows(country_queues):
        group = (row['source'], row['group'])
        if group in seen_groups:
            continue
        seen_groups.add(group)
        sample.append(row)
        if len(sample) == RDD_NEGATIVE_PILOT_SIZE:
            break
    return sample


def _negative_ratio_review_targets(positives, verified_negatives):
    ratios = (0.25, 0.5, 1.0, 'all')
    targets = []
    for current, next_ratio in zip(ratios, ratios[1:]):
        cap = math.floor(positives * current)
        distinct = next_ratio == 'all' or cap < math.floor(positives * next_ratio)
        targets.append(dict(first_policy=current, second_policy=next_ratio, first_policy_cap=cap,
            minimum_verified_training_negatives=cap + 1 if distinct else None,
            additional_verified_training_negatives=max(0, cap + 1 - verified_negatives) if distinct else None))
    return targets


def review_queues(audit, report):
    folder = report/'review_queue'
    folder.mkdir()
    candidates = [dict(row, published_crack_profile=rdd_crack_profile(row['classes']))
                  for row in audit.records if row['source'] in {'rdd_Japan','rdd_Norway'} and
                  row['reason'] in {'negative_completeness_not_reviewed','negative_recording_assignment_required'}]
    fields = (*REVIEW_FIELDS, 'group_provenance', 'published_crack_profile')
    write_csv(folder/'rdd_negative_candidates.csv', candidates, fields)
    sample = [dict(row, file_sha256=file_hash(resolve_source(audit.root, row['original_image'])))
              for row in rdd_negative_pilot(candidates, audit.seed)]
    write_csv(folder/'rdd_negative_batch_001.csv', sample, fields)
    for page, start in enumerate(range(0, len(sample), 48), 1):
        contact_sheet(audit.root, sample[start:start+48], folder/f'rdd_negative_batch_{page:03}.jpg',
                      seed=audit.seed, limit=48)
    training = audit.ratio_comparisons[-1]
    positives, verified = training['positive_images'], training['reviewed_negative_images']
    plan = dict(training_positives=positives, available_verified_training_negatives=verified,
        batch_candidates=len(sample), batch_by_source=dict(Counter(row['source'] for row in sample)),
        batch_by_crack_profile=dict(Counter(row['published_crack_profile'] for row in sample)),
        comparison_targets=_negative_ratio_review_targets(positives, verified))
    (folder/'rdd_negative_review_plan.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
    write_csv(folder/'supplemental_sources.csv', [r for r in audit.records if r['source'] == 'roboflow'
        and r.get('annotation_state') == 'parsed' and r['reason']], REVIEW_FIELDS)
    positives = [r for r in audit.records if r['source'] in {'rdd_Japan','rdd_Norway'} and r['boxes'] and r['split'] in {'train','val'}]
    positives.sort(key=lambda row: (not bool(row.get('visual_review_note')), row['source'], row['original_id']))
    write_csv(folder/'rdd_positive_checks.csv', positives, (*REVIEW_FIELDS, 'visual_review_note'))
    write_csv(folder/'duplicate_pairs.csv', [pair for pair in audit.duplicates if pair['kind'] == 'perceptual_candidate'
        and 'test' not in {pair['left_split'], pair['right_split']}
        and pair['resolution'] in {'quarantined','roboflow_review_only'}])


def write_report(audit, destination, reserved=False):
    destination = Path(destination)
    if not reserved:
        destination.mkdir(parents=True, exist_ok=False)
    statistics = source_statistics(audit)
    audit.summary['source_statistics'] = statistics
    audit.summary['unreviewed_or_uncertain_images'] = sum(row['unreviewed_or_uncertain'] for row in statistics)
    for name, contents in [
        ('dataset_summary.json', audit.summary), ('split_integrity_report.json', split_report(audit)),
        ('roboflow_review.json', {**audit.rf_report, 'selected': sum(row['source'] == 'roboflow' for row in audit.selected),
            'similarity_pairs': sum('roboflow' in {pair['left_source'],pair['right_source']} for pair in audit.duplicates)}),
        ('review_events.json', list(audit.events.values())),
    ]:
        (destination/name).write_text(json.dumps(contents, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    write_csv(destination/'source_statistics.csv', statistics)
    write_csv(destination/'excluded.csv', [row for row in audit.records if row['reason']])
    write_csv(destination/'annotation_audit.csv', audit.issues+audit.review_warnings,
        ['source','original_id','original_image','annotation','finding'])
    write_csv(destination/'duplicate_report.csv', audit.duplicates,
        ['kind','left','right','left_source','right_source','left_split','right_split','left_group','right_group',
         'left_file_sha256','right_file_sha256','distance','resolution','pair_id'])
    write_csv(destination/'mapillary_similarity_review.csv', [pair for pair in audit.duplicates
        if 'mapillary' in {pair['left_source'],pair['right_source']}])
    write_csv(destination/'selection_plan.csv', audit.selected)
    write_csv(destination/'negative_ratio_comparison.csv', audit.ratio_comparisons)
    counts = defaultdict(list)
    for row in audit.records:
        if row['source'] in {'local','mapillary','roboflow'}:
            counts[(row['source'],row['group'] or 'unknown',row['split'])].append(row)
    write_csv(destination/'recording_statistics.csv', [dict(source=source, group=group, split=split, available=len(rows),
        exclusions=dict(Counter(row['reason'] for row in rows if row['reason'])),
        **split_counts([row for row in rows if not row['reason']])) for (source, group, split), rows in sorted(counts.items())])
    review_queues(audit, destination)
    samples = destination/'contact_sheets'
    samples.mkdir()
    for source in SOURCES:
        rows = [r for r in audit.records if r['source'] == source and r['boxes'] and r['split'] != 'test'
                and r.get('original_partition') != 'test' and r['reason'] != 'protected_final_recording']
        contact_sheet(audit.root, rows, samples/f'{source}_positives.jpg', audit.seed)
    (destination/'README.md').write_text(
        '# Pothole detection experiment 2\n\n'
        'See `docs/experiment_2_workflow.md` from the repository root for commands and review instructions.\n\n'
        'This report uses original D40 labels for Japan and Norway and reviewed Green Bay and Mapillary annotations. '
        'Roboflow version 2 uses class 5 after image and capture-group review. Output has one pothole class.\n\n'
        'The development validation set is fixed for a given review snapshot and independent of negative sampling. '
        'Use `source_statistics.csv`, `excluded.csv`, `review_queue/` and `split_integrity_report.json` for the actual selections.\n',
        encoding='utf-8')


def _verify_experiment_reports(output):
    summary = json.loads((output/'dataset_summary.json').read_text(encoding='utf-8'))
    report = json.loads((output/'split_integrity_report.json').read_text(encoding='utf-8'))
    rows = read_csv(output/'manifest.csv')
    integrity = {row['label']: row['sha256'] for row in read_csv(output/'label_integrity.csv')}
    if not report['passes'] or report['final_test_published'] or (output/'test.txt').exists():
        raise ValueError('Experiment 2 split integrity report failed')
    if set(integrity) != {row['label'] for row in rows}:
        raise ValueError('Label integrity index does not match the manifest')
    groups = {}
    for row in rows:
        if row['source'] not in SOURCES:
            raise ValueError('Source is outside the Experiment 2 allowlist')
        if row['source'] in {'rdd_Japan','rdd_Norway'} and row['country'] != row['source'].removeprefix('rdd_'):
            raise ValueError('RDD source/country mismatch')
        if row['split'] == 'train' and row['pixel_sha256'] in summary['historical_validation_pixel_hashes']:
            raise ValueError('Historical validation image entered training')
        if row['source'] == 'roboflow' and (row['split'] != 'train' or not row.get('source_approval_event') or not row['recording_id']):
            raise ValueError('Roboflow image lacks a reviewed training capture group')
        if not int(row['potholes']) and not row.get('negative_evidence'):
            raise ValueError('Empty labels require explicit negative evidence')
        for alias in _group_ids(row):
            if groups.setdefault(alias,row['split']) != row['split']:
                raise ValueError(f'Split overlap in {alias}')
        label = output/row['label']
        image = output/row['image']
        if label.parent != output/'labels'/row['split'] or image.parent != output/'images'/row['split'] or label.stem != image.stem:
            raise ValueError('YOLO image/label paths do not match')
        if file_hash(label) != integrity[row['label']]:
            raise ValueError('Prepared label bytes changed')
        row['label_sha256'] = hashlib.sha256(label.read_text(encoding='utf-8').encode()).hexdigest()
    if validation_digest([row for row in rows if row['split'] == 'val']) != summary['validation_fingerprint']:
        raise ValueError('Development validation membership or labels changed')


def verify_experiment(output):
    output = Path(output).resolve()
    counts = verify(output)
    _verify_experiment_reports(output)
    return counts


def stage_experiment(audit, output, config):
    def finalize(destination):
        labels = [dict(label=row['label'], sha256=file_hash(destination/row['label']))
                  for row in read_csv(destination/'manifest.csv')]
        write_csv(destination/'label_integrity.csv', labels)
        for module in ('experiment_2.py','dataset_reviews.py','experiment_2_recordings.py'):
            shutil.copyfile(Path(__file__).with_name(module), destination/'preparation_source'/module)
        _verify_experiment_reports(destination.resolve())

    stage(audit, output, config, report_writer=write_report, finalize=finalize)
