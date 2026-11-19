"""V4: fixed development validation, reviewed expansion and accountable sampling.

The earlier builder and all earlier dataset versions remain unchanged.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

from PIL import Image, ImageChops, ImageDraw, ImageStat

from src.dataset_reviews import load_events, matches_inputs, pair_key
from src.us_dataset import (Audit, CHARACTERISTICS, contact_sheet, file_hash, image_fingerprints,
    read_csv, relative, resolve_source, stage, truth, validate_box, verify, write_csv, write_reports)


def negative_sample(rows, ratio, seed):
    """Global budget; round robin sources, then scene/confuser strata. No replacement."""
    positives = [r for r in rows if r['boxes']]
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not row['boxes']:
            key = (row['group'], '+'.join(row.get('classes') or row.get('confusers') or ['reviewed_background']))
            buckets[row['source']][key].append(row)
    rng = random.Random(seed)
    for source in sorted(buckets):
        for key in sorted(buckets[source]):
            buckets[source][key].sort(key=lambda r: r['original_id'])
            rng.shuffle(buckets[source][key])
    budget = sum(len(bucket)for groups in buckets.values()for bucket in groups.values()) if ratio is None else math.floor(len(positives)*ratio)
    selected = []
    cycles = {source: deque(sorted(groups)) for source,groups in buckets.items()}
    while buckets and len(selected) < budget:
        for source in sorted(list(buckets)):
            if len(selected) >= budget:
                break
            groups = buckets[source]
            key = cycles[source].popleft()
            selected.append(groups[key].pop())
            if groups[key]:cycles[source].append(key)
            else:del groups[key]
            if not groups:
                del buckets[source]
    return positives + selected


class V4Audit(Audit):
    def __init__(self, root, recipe, countries, *, review_dir=None, intake=None, assignments=None, **kwargs):
        if kwargs.get('reviews'):
            raise ValueError('V4 requires content-bound review events, not unbound legacy approval CSVs')
        requested_ratio=kwargs.pop('negative_ratio',None)
        super().__init__(root, recipe, countries, negative_ratio=0 if requested_ratio is None else requested_ratio, **kwargs)
        self.ratio=requested_ratio
        if any(c not in {'United_States', 'Norway', 'Czech'} for c in self.countries):
            raise ValueError('V4 permits US and explicitly reviewed Norway/Czech sources only')
        self.events = load_events(self.root, review_dir or self.root/'results/dataset_reviews/v4')
        self.review_warnings = []
        reference = resolve_source(self.root, recipe['validation_reference'])
        verify(reference)
        self.reference = reference
        self.frozen = {(r['source'], r['original_id']): r for r in read_csv(reference/'manifest.csv') if r['split'] == 'val'}
        if not self.frozen:
            raise ValueError('A nonempty frozen development validation manifest is required')
        self.v3_rows = read_csv(reference/'manifest.csv')
        self.v3_exclusions = read_csv(reference/'excluded.csv')
        self.intake = Path(intake) if intake else None
        self.assignments = {}
        for row in read_csv(assignments) if assignments else []:
            identity = row.get('recording_id', '')
            if not identity or identity in self.assignments:
                raise ValueError('Recording assignments require unique explicit recording IDs')
            if not truth(row.get('approved')) or not row.get('reviewer') or not row.get('reviewed_at'):
                raise ValueError('Recording assignment is not explicitly approved')
            if (row.get('split'), row.get('role')) not in {('train', 'training'), ('val', 'development'), ('test', 'final')}:
                raise ValueError('Invalid recording role/split combination')
            self.assignments[identity] = row
            if row['role'] == 'final':
                self.protected.add(identity)

    def apply_image_review(self, row):
        if row['reason'].startswith('annotation_invalid') or not row.get('annotation_sha256'):
            return  # Approval never repairs an annotation or manufactures a label.
        source_event = self.events.get(('source', row['source'], row['original_id']))
        negative_event = self.events.get(('negative', row['source'], row['original_id']))
        event = source_event if row['source'] != 'rdd_United_States' else negative_event
        if not event:
            return
        if not matches_inputs(self.root, row, event):
            self.review_warnings.append(dict(source=row['source'], original_id=row['original_id'], finding='stale_review_inputs'))
            return
        if row['source'] != 'rdd_United_States':
            if not (event.get('decision') == 'approved' and event.get('annotation_definition_checked')
                    and all(str(event.get(k, '')).strip() for k in CHARACTERISTICS)):
                return
        elif event.get('decision') != 'pothole_free' or row['boxes']:
            return
        if not row['boxes']:
            complete_event = negative_event if row['source'] != 'rdd_United_States' else event
            if not (matches_inputs(self.root, row, complete_event) and complete_event.get('decision') == 'pothole_free'
                    and complete_event.get('target_annotation_complete')):
                return
        # Never release historical held-out, invalid, or globally disabled records.
        if row['reason'] in {'historical_test_protected', 'official_unlabelled_test_excluded', 'roboflow_original_test_not_imported', 'excluded_by_policy'}:
            return
        if row['split'] not in {'train', 'val'} or row['source'] == 'roboflow':
            if not event.get('recording_id') or event.get('split') != 'train':
                self.review_warnings.append(dict(source=row['source'], original_id=row['original_id'], finding='review_requires_explicit_training_recording'))
                return
            row.update(split='train', recording_id=event['recording_id'], group='identity:'+event['recording_id'],
                       group_provenance='explicit_content_bound_source_review')
        row['reason'] = ''
        row['approval_event_id'] = event.get('event_id', '')
        row['approval_reviewer'] = event['reviewer']

    def rdd(self):
        super().rdd()
        for row in self.records:
            if row['country'] in self.countries:
                self.apply_image_review(row)

    def roboflow(self):
        super().roboflow()
        for row in self.records:
            if row['source']=='roboflow':
                row['source_metadata_sha256']=file_hash(resolve_source(self.root,self.recipe['roboflow_root'])/'data.yaml')
        if self.include_roboflow:
            for row in self.records:
                if row['source'] == 'roboflow':
                    self.apply_image_review(row)

    def local(self):
        super().local()
        if not self.intake:
            return
        known = {(r['source'], r['original_id']) for r in self.records}
        for meta in read_csv(self.intake):
            source, identity = meta.get('source'), meta.get('image_id')
            if source not in {'local', 'mapillary'} or not identity or (source, identity) in known:
                raise ValueError('Intake source/ID is invalid or duplicates an existing image')
            known.add((source, identity))
            image, annotation = resolve_source(self.root, meta['filepath']), resolve_source(self.root, meta['annotation'])
            row = self.record(source, identity, image, annotation)
            row['intake_manifest'] = relative(self.root, self.intake)
            try:
                ann = json.loads(annotation.read_text(encoding='utf-8'))
                recording = meta.get('recording_id', '')
                assignment = self.assignments.get(recording)
                if not assignment:
                    raise ValueError('intake_recording_assignment_required')
                row.update(recording_id=recording, video_id=recording, sequence_id=meta.get('sequence_id', ''),
                    group='identity:'+recording, split=assignment['split'], group_provenance='explicit_recording_assignment',
                    variant=meta.get('image_variant', ''), license=meta.get('license', ''), annotation_sha256=file_hash(annotation))
                row['group_aliases'] = ['identity:'+v for v in (recording, row['sequence_id']) if v]
                if ann.get('review_status') != 'reviewed' or truth(ann.get('uncertain')) or 'pothole_boxes' not in ann:
                    raise ValueError('intake_annotation_review_required')
                if ann.get('video_id') != recording or resolve_source(self.root, ann.get('source_filepath', '')) != image:
                    raise ValueError('intake_annotation_recording_or_image_mismatch')
                if not row['variant'] or not row['license'] or meta.get('timestamp_ms', '') == '':
                    raise ValueError('intake_timestamp_variant_license_required')
                timestamp = float(meta['timestamp_ms'])
                if not math.isfinite(timestamp) or timestamp < 0 or ann.get('timestamp_ms') is None or float(ann['timestamp_ms']) != timestamp:
                    raise ValueError('intake_timestamp_mismatch')
                row['timestamp_ms'] = timestamp
                with Image.open(image) as im:
                    width, height = im.size
                if (ann.get('image_width'), ann.get('image_height')) != (width, height):
                    raise ValueError('intake_annotation_dimension_mismatch')
                row.update(width=width, height=height)
                for box in ann['pothole_boxes']:
                    x, y, w, h = [float(box[k]) for k in ('x', 'y', 'w', 'h')]
                    coords = validate_box((x+w/2, y+h/2, w, h), width, height)
                    row['boxes'].append(tuple(v/s for v,s in zip(coords, (width,height,width,height))))
                if not row['boxes'] and not truth(ann.get('target_annotation_complete')):
                    raise ValueError('intake_negative_completeness_required')
                if assignment['role'] == 'final':
                    row['reason'] = 'protected_final_recording'
                elif source == 'local' and not self.include_local or source == 'mapillary' and not self.include_mapillary:
                    row['reason'] = 'excluded_by_policy'
            except (KeyError, ValueError, TypeError, OSError) as exc:
                self.issue(row, str(exc))

    def deduplicate(self):
        # Include every verified training negative in the scan BEFORE any sampling.
        super().deduplicate()
        by_path = {r['original_image']: r for r in self.records}
        unresolved, accepted = defaultdict(list), []
        for pair in self.duplicates:
            left, right = by_path[pair['left']], by_path[pair['right']]
            event = self.events.get(('duplicate', pair_key(pair['left'], pair['right'])))
            protected = any(r['source'] == 'historical_test' or r['reason'] == 'protected_final_recording' for r in (left, right))
            if protected and any((r['source'],r['original_id']) in self.frozen for r in (left,right)):
                raise ValueError('Protected final data overlaps the frozen development validation; do not tune on this recording')
            hashes = {r['original_image']: r.get('file_sha256') for r in (left, right)}
            related_training = bool(event and event.get('decision')=='distinct_frames_same_training_group'
                and left['split']==right['split']=='train' and left['group'] and left['group']==right['group'])
            cleared = (pair['kind'] == 'perceptual_candidate' and not protected and event
                and (event.get('decision') == 'different_scenes' or related_training) and event.get('inspection_confirmed')
                and hashes.get(event.get('original_image')) == event.get('file_sha256')
                and hashes.get(event.get('right_image')) == event.get('right_sha256'))
            pair['review_status'] = ('explicitly_distinct_training_frames' if related_training else 'explicitly_distinct_scenes') if cleared else 'requires_review'
            if cleared:
                pair['review_event_id'] = event['event_id']
                accepted.append(pair)
            elif pair['kind'] == 'perceptual_candidate':
                unresolved[pair['left']].append(pair)
                unresolved[pair['right']].append(pair)
        for row in self.records:
            if row['reason'] == 'near_duplicate_requires_review' and row['original_image'] not in unresolved:
                row['reason'] = ''
                row['duplicate_review'] = 'all_candidate_pairs_explicitly_cleared'
        # The reference is anchored. Any newly introduced neighbor is withheld;
        # adding training data must never silently shrink development validation.
        for key, reference in self.frozen.items():
            matches = [r for r in self.records if (r['source'],r['original_id']) == key]
            if len(matches) != 1:
                raise ValueError('Frozen validation image missing/ambiguous in source inventory')
            row = matches[0]
            if row['reason'] not in {'', 'near_duplicate_requires_review'}:
                raise ValueError('Frozen validation has a new safety problem: '+row['reason'])
            if row.get('pixel_sha256') != reference['pixel_sha256'] or row.get('file_sha256') != reference['file_sha256']:
                raise ValueError('Frozen validation source content changed')
            if file_hash(resolve_source(self.root, row['annotation'])) != reference.get('annotation_sha256'):
                raise ValueError('Frozen validation annotation changed')
            expected_label = (self.reference/reference['label']).read_text(encoding='utf-8')
            generated = ''.join('0 '+' '.join(f'{v:.8f}' for v in box)+'\n' for box in row['boxes'])
            if generated != expected_label:
                raise ValueError('Frozen validation annotation interpretation changed')
            row['reason'] = ''
            row['validation_reference'] = relative(self.root, self.reference)

    def select(self):
        for row in self.records:
            if not row['reason'] and row['split'] == 'val' and (row['source'], row['original_id']) not in self.frozen:
                row['reason'] = 'outside_frozen_development_validation'
        self.eligible_before_sampling = [dict(r) for r in self.records if not r['reason']]
        training = [r for r in self.eligible_before_sampling if r['split']=='train']
        kept = negative_sample(training, self.ratio, self.seed)
        keep_keys = {(r['source'],r['original_id']) for r in kept}
        for row in self.records:
            if not row['reason'] and row['split']=='train' and (row['source'],row['original_id']) not in keep_keys:
                row['reason'] = 'training_negative_ratio_limit'
        self.ratio_comparison = []
        for ratio in (.25, .5, 1.0, None):
            selected = negative_sample(training, ratio, self.seed)
            self.ratio_comparison.append(dict(negative_ratio=ratio if ratio is not None else 'all_reviewed', training_images=len(selected),
                positive_images=sum(bool(r['boxes']) for r in selected), negative_images=sum(not r['boxes'] for r in selected),
                boxes=sum(len(r['boxes']) for r in selected), frozen_validation_images=len(self.frozen)))
        return [r for r in self.records if not r['reason']]

    def run(self):
        super().run()
        actual_val = {(r['source'],r['original_id']) for r in self.selected if r['split']=='val'}
        if actual_val != set(self.frozen):
            raise ValueError('Recording/split safety conflicts prevent preserving frozen validation; resolve them before building')
        self.summary.update(workflow='v4_global_negatives_fixed_validation', pipeline_v4_sha256=file_hash(Path(__file__)),
            frozen_validation_manifest=relative(self.root,self.reference/'manifest.csv'),
            frozen_validation_manifest_sha256=file_hash(self.reference/'manifest.csv'),
            negative_budget_scope='all_reviewed_training_negatives; no_undersampling' if self.ratio is None else 'global_training; balanced_sources_and_recordings',
            approval_events=sorted(e.get('event_id','') for e in self.events.values()),
            approval_event_digests={e.get('event_id',''):hashlib.sha256(json.dumps(e,sort_keys=True).encode()).hexdigest() for e in self.events.values()},
            additional_training_positives_recovered=len({(r['source'],r['original_id']) for r in self.selected if r['split']=='train' and r['boxes']}
                - {(r['source'],r['original_id']) for r in self.v3_rows if r['split']=='train' and int(r['potholes'])}),
            ratio_comparisons=self.ratio_comparison)
        return self.summary


def reports_v4(audit, destination):
    destination = Path(destination)
    stats, recordings, reasons = [], [], []
    for source in sorted({r['source'] for r in audit.records}):
        rows = [r for r in audit.records if r['source']==source]
        stats.append(dict(source=source, discovered=len(rows),
            positive_images_with_parsed_boxes=sum(bool(r['boxes']) for r in rows),
            negative_candidates_with_parsed_annotations=sum(not r['boxes'] and bool(r.get('annotation_sha256')) and bool(r.get('width')) for r in rows),
            selected_train=sum(not r['reason'] and r['split']=='train' for r in rows),
            selected_val=sum(not r['reason'] and r['split']=='val' for r in rows),
            selected_positive=sum(not r['reason'] and bool(r['boxes']) for r in rows),
            selected_negative=sum(not r['reason'] and not r['boxes'] for r in rows),
            selected_boxes=sum(len(r['boxes']) for r in rows if not r['reason'])))
        for reason, count in sorted(Counter(r['reason'] or 'selected' for r in rows).items()):
            part = [r for r in rows if (r['reason'] or 'selected')==reason]
            reasons.append(dict(source=source, reason=reason, images=count, positives=sum(bool(r['boxes']) for r in part),
                                negatives_or_unlabelled=sum(not r['boxes'] for r in part)))
    by_recording = defaultdict(list)
    for row in audit.records:
        by_recording[(row['source'],row['group'] or 'metadata_missing',row['split'])].append(row)
    for key,rows in sorted(by_recording.items()):
        selected = [r for r in rows if not r['reason']]
        recordings.append(dict(source=key[0],group=key[1],split=key[2],available_images=len(rows),selected_images=len(selected),
            positives=sum(bool(r['boxes']) for r in selected),negatives=sum(not r['boxes'] for r in selected),
            boxes=sum(len(r['boxes']) for r in selected),exclusions=dict(Counter(r['reason'] for r in rows if r['reason'])),
            group_provenance='; '.join(sorted({r['group_provenance'] for r in rows}))))
    write_csv(destination/'source_inventory_v4.csv', stats)
    write_csv(destination/'source_exclusion_breakdown.csv', reasons)
    write_csv(destination/'recording_statistics.csv', recordings)
    write_csv(destination/'negative_ratio_comparison.csv', audit.ratio_comparison)
    write_csv(destination/'review_warnings.csv', audit.review_warnings)
    write_csv(destination/'duplicate_decisions_v4.csv', audit.duplicates)
    with (destination/'review_events.json').open('x',encoding='utf-8') as f:
        json.dump(list(audit.events.values()),f,indent=2,sort_keys=True)
    queue = destination/'review_queue'
    queue.mkdir()
    # US crack/background candidates remain unapproved, even if no D40 was found.
    negatives = [r for r in audit.records if r['source']=='rdd_United_States' and r['reason']=='negative_completeness_not_reviewed']
    write_csv(queue/'us_negative_candidates.csv', negatives)
    candidates = [r for r in audit.records if r['source'] in {'rdd_Norway','rdd_Czech','roboflow'} and r.get('width')
                  and r['reason'] not in {'historical_test_protected','roboflow_original_test_not_imported'}]
    write_csv(queue/'supplemental_sources.csv', candidates)
    write_csv(queue/'duplicate_pairs.csv', audit.duplicates)
    # Stable stratified batch prioritizes varied crack categories, not imagined hardness.
    buckets = defaultdict(list)
    for row in negatives:
        buckets['+'.join(row['classes']) or 'no_damage_boxes'].append(row)
    sample = []
    for bucket in buckets.values():
        bucket.sort(key=lambda r: (r['split']!='train',hashlib.sha256(f"{audit.seed}:{r['original_id']}".encode()).hexdigest()),reverse=True)
    while buckets and len(sample)<60:
        for key in sorted(list(buckets)):
            sample.append(buckets[key].pop())
            if not buckets[key]:del buckets[key]
            if len(sample)==60:break
    write_csv(queue/'us_negative_batch_001.csv', sample)
    contact_sheet(audit.root,sample,queue/'us_negative_batch_001.jpg',audit.seed,limit=60)
    # Visual evidence and numerical differences are triage, never automatic approval.
    pair_dir = queue/'pair_images'; pair_dir.mkdir()
    analyses = []
    prior_paths = {r['original_image'] for r in audit.v3_exclusions if r['reason']=='near_duplicate_requires_review'}
    for pair in audit.duplicates:
        if pair['left'] not in prior_paths and pair['right'] not in prior_paths:
            continue
        canvas = Image.new('RGB',(1000,440),'white'); draw=ImageDraw.Draw(canvas)
        thumbnails=[]
        for index,key in enumerate(('left','right')):
            with Image.open(resolve_source(audit.root,pair[key])) as im:
                original=im.convert('RGB'); thumbnails.append(original.resize((256,256)))
                original.thumbnail((490,385));canvas.paste(original,(index*500,0))
            draw.text((index*500+3,390),Path(pair[key]).name,fill='black')
        pair_id = pair_key(pair['left'],pair['right'])
        rmse = sum(ImageStat.Stat(ImageChops.difference(*thumbnails)).rms)/3
        draw.text((3,414),f"dHash distance {pair['distance']} | resized RGB RMSE {rmse:.2f} | human review required",fill='black')
        canvas.save(pair_dir/f'{pair_id}.jpg')
        analyses.append(dict(pair_id=pair_id,**pair,resized_rgb_rmse=round(rmse,4),
            assessment='not_exact_duplicate; same_scene_vs_false_hash_match_requires_human_review'))
    write_csv(destination/'mapillary_quarantine_investigation.csv', analyses)
    (queue/'README.md').write_text('Review originals and labels in the dataset-review app. No queue row is approved. '\
        'Record Pothole-free only after a full-image cavity check. Pothole present needs a separate reviewed annotation; Unsure remains excluded. '\
        'Every similarity pair touching an image must be explicitly cleared as Different scenes before that image can return. '\
        'Do not use small contact-sheet tiles as the sole basis for approval.\n',encoding='utf-8')


def stage_v4(audit, output, config):
    stage(audit, output, config)
    output = Path(output)
    # Shared staging completed safely; mark V4 incomplete until its extra checks pass.
    (output/'build_status.json').write_text('{"status":"incomplete"}',encoding='utf-8')
    try:
        reports_v4(audit,output)
        manifest=read_csv(output/'manifest.csv')
        write_csv(output/'label_integrity.csv',[dict(source=r['source'],original_id=r['original_id'],label=r['label'],
            label_sha256=file_hash(output/r['label']))for r in manifest])
        for path in (Path(__file__),Path(__file__).with_name('dataset_reviews.py')):
            shutil.copyfile(path,output/'preparation_source'/path.name)
        verify_v4(output,require_complete=False,project_root=audit.root)
        (output/'build_status.json').write_text('{"status":"complete"}',encoding='utf-8')
    except Exception:
        raise


def verify_v4(output, require_complete=True, project_root=None):
    counts = verify(output,require_complete=require_complete)
    output = Path(output)
    summary = json.loads((output/'dataset_summary.json').read_text(encoding='utf-8'))
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    # The reference is read only, and its manifest identity is pinned at build time.
    reference = resolve_source(root,summary['frozen_validation_manifest'])
    if not reference.is_file() and reference == root/'data/training/pothole_us_v3/manifest.csv':
        reference = root/'results/dataset_archive/retired_us_builds/data/training/pothole_us_v3/manifest.csv'
    if file_hash(reference) != summary['frozen_validation_manifest_sha256']:
        raise ValueError('Frozen validation manifest changed')
    expected = {(r['source'],r['original_id']):r for r in read_csv(reference) if r['split']=='val'}
    actual = {(r['source'],r['original_id']):r for r in read_csv(output/'manifest.csv') if r['split']=='val'}
    integrity=read_csv(output/'label_integrity.csv')
    all_rows=read_csv(output/'manifest.csv')
    if {(r['source'],r['original_id']) for r in integrity}!={(r['source'],r['original_id']) for r in all_rows}:
        raise ValueError('Label integrity index is incomplete')
    for row in integrity:
        path=(output/row['label']).resolve()
        if not path.is_relative_to(output.resolve()) or file_hash(path)!=row['label_sha256']:
            raise ValueError('Prepared label content changed')
    if set(expected)!=set(actual):raise ValueError('Frozen validation membership changed')
    for key,row in actual.items():
        if row['pixel_sha256']!=expected[key]['pixel_sha256']:
            raise ValueError('Frozen validation image changed')
        if (output/row['label']).read_bytes() != (reference.parent/expected[key]['label']).read_bytes():
            raise ValueError('Frozen validation labels changed')
    return counts
