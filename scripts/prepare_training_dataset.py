"""Stage verified image/label pairs and balance training only; preserve V3 holdouts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_kaggle_roads import pixel_hash, pothole_labels


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def validate_labels(text):
    if any(line.split()[0] != '0' for line in text.splitlines() if line.strip()):
        raise ValueError('Final detector labels must contain only class 0')
    return pothole_labels(text)


def sample_negatives(records, ratio, seed):
    """Preserve positives, cap negatives per source, round-robin hard-negative strata."""
    rng = random.Random(seed)
    selected = [r for r in records if r['boxes']]
    for source in sorted({r['source'] for r in records}):
        positives = sum(bool(r['boxes']) for r in records if r['source'] == source)
        buckets = defaultdict(list)
        for row in records:
            if row['source'] == source and not row['boxes']:
                buckets[row['stratum']].append(row)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        budget = math.floor(positives * ratio)
        while budget and any(buckets.values()):
            for key in sorted(buckets):
                if budget and buckets[key]:
                    selected.append(buckets[key].pop())
                    budget -= 1
    return selected


def collect():
    rows = []
    # Authoritative labels are in split-specific processed folders, NOT raw images/.
    for r in read_csv(ROOT / 'data/processed/rdd2022_v3/split_metadata.csv'):
        rows.append(dict(image=ROOT / r['filepath'], label=ROOT / 'data/processed/rdd2022_v3/labels' / r['split'] / (r['image_id'] + '.txt'),
                         source='rdd_' + r['country'], split=r['split'], group='rdd:' + r['group_id'],
                         stratum=r['tier'] or 'positive', original_id=r['image_id']))
    inventory = {r['image_id']: r for r in read_csv(ROOT / 'data/metadata/image_inventory.csv')}
    for r in read_csv(ROOT / 'exports/yolo_labels/export_manifest.csv'):
        meta = inventory[r['image_id']]
        if meta['review_status'] != 'reviewed' or meta.get('uncertain', '').lower() == 'true':
            continue
        if r['split'] not in {'train', 'val', 'test'} or meta['split'] != r['split']:
            raise ValueError(f'Local split metadata disagreement: {r["image_id"]}')
        group = meta['recording_id'] or meta['sequence_id']
        if not group:
            raise ValueError(f'Missing recording/sequence ID: {r["image_id"]}')
        tags = [tag for tag in ('shadows_present', 'patches_present', 'manholes_present') if meta.get(tag, '').lower() == 'true']
        rows.append(dict(image=ROOT / r['image_path'], label=ROOT / r['label_file'], source='local_' + meta['source'],
                         split=r['split'], group='local:' + group, stratum='+'.join(tags) or 'other', original_id=r['image_id']))
    for r in read_csv(ROOT / 'data/processed/kaggle_arcioni_v4/manifest.csv'):
        image = Path(r['image'])
        # Entire Kaggle source stays in training: no adjacent-frame split leakage.
        raw_label = Path(r['source_image']).parent.parent / 'labels-YOLO' / (Path(r['source_image']).stem + '.txt')
        classes = sorted({line.split()[0] for line in raw_label.read_text().splitlines() if line.strip()})
        rows.append(dict(image=image, label=image.parent.parent / 'labels' / (image.stem + '.txt'),
                         source='kaggle_arcioni', split='train', group='kaggle:arcioni_all',
                         stratum='+'.join(classes) or 'background', original_id=Path(r['source_image']).stem))
    return rows


def prepare(output, config_path, ratio=0.25, seed=42):
    if not 0 <= ratio <= 1:
        raise ValueError('Negative-to-positive ratio must be between 0 and 1')
    if output.exists() or config_path.exists():
        raise FileExistsError('Outputs already exist; choose new paths')
    records = collect()
    groups = defaultdict(set)
    for row in records:
        groups[row['group']].add(row['split'])
    conflicts = {g: sorted(s) for g, s in groups.items() if len(s) > 1}
    if conflicts:
        raise ValueError(f'Recording/group split overlap: {conflicts}')
    # Holdouts take precedence over duplicate training images. Never move a
    # held-out image into training to resolve overlap.
    records.sort(key=lambda r: ({'test': 0, 'val': 1, 'train': 2}[r['split']], str(r['image'])))
    seen, unique, exclusions = {}, [], []
    for index, row in enumerate(records):
        if index % 500 == 0:
            print(f'Checking images and labels: {index}/{len(records)}', flush=True)
        row['boxes'] = validate_labels(row['label'].read_text(encoding='utf-8'))
        row['hash'] = pixel_hash(row['image'])
        if row['hash'] in seen:
            previous = seen[row['hash']]
            # Different annotations on identical pixels require manual review.
            if sorted(row['boxes']) != sorted(previous['boxes']):
                raise ValueError(f'Conflicting duplicate labels: {row["image"]}, {previous["image"]}')
            exclusions.append(dict(image=str(row['image']), reason='exact_duplicate', kept_split=previous['split']))
            continue
        seen[row['hash']] = row
        unique.append(row)
    training = sample_negatives([r for r in unique if r['split'] == 'train'], ratio, seed)
    selected = training + [r for r in unique if r['split'] != 'train']
    kept = {r['hash'] for r in selected}
    exclusions.extend(dict(image=str(r['image']), reason='training_negative_subsample', kept_split='') for r in unique if r['hash'] not in kept)
    summary = dict(seed=seed, train_negative_to_positive_target=ratio, splits={}, excluded=dict(Counter(r['reason'] for r in exclusions)),
                   limitations=['RDD splits have sample-level fallback; route independence is not verified.',
                                'Kaggle is training-only; no Kaggle test performance claim.',
                                'Decoded-pixel duplicates removed; recompressed/adjacent-frame near-duplicates not comprehensively detected.',
                                'Local held-out test recording is still needed for end-to-end event evaluation.',
                                'Existing RDD evaluation splits were already subsampled; metrics do not estimate natural road prevalence.',
                                'Holdouts apply to this new training recipe; older checkpoints may have seen these images.'])
    for split in ('train', 'val', 'test'):
        part = [r for r in selected if r['split'] == split]
        pos = sum(bool(r['boxes']) for r in part)
        if not part or not pos:
            raise ValueError(f'Empty or positive-free {split} split')
        summary['splits'][split] = dict(images=len(part), positive_images=pos, negative_images=len(part)-pos,
                                       pothole_boxes=sum(len(r['boxes']) for r in part),
                                       by_source={s: dict(Counter('positive' if r['boxes'] else 'negative' for r in part if r['source'] == s)) for s in sorted({r['source'] for r in part})})
    output.mkdir(parents=True)
    manifests = defaultdict(list)
    audit = []
    for row in selected:
        name = row['source'] + '_' + hashlib.sha256(str(row['image']).encode()).hexdigest()[:20]
        image_dir, label_dir = output / 'images' / row['split'], output / 'labels' / row['split']
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        target = image_dir / (name + row['image'].suffix.lower())
        shutil.copy2(row['image'], target)
        (label_dir / (name + '.txt')).write_text('\n'.join(row['boxes']) + ('\n' if row['boxes'] else ''), encoding='utf-8')
        manifests[row['split']].append(target.resolve().as_posix())
        audit.append(dict(source=row['source'], original_id=row['original_id'], original_image=str(row['image']), image=target.resolve().as_posix(),
                          label=(label_dir / (name+'.txt')).resolve().as_posix(), split=row['split'], group=row['group'], stratum=row['stratum'],
                          pixel_sha256=row['hash'], potholes=len(row['boxes'])))
    for split, paths in manifests.items():
        (output / (split + '.txt')).write_text('\n'.join(sorted(paths)) + '\n', encoding='utf-8')
    for name, values in [('manifest.csv', audit), ('excluded.csv', exclusions)]:
        if values:
            with (output / name).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    (output / 'dataset_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    config = dict(path=ROOT.as_posix(), names={0: 'pothole'}, **{s: (output / (s+'.txt')).resolve().as_posix() for s in manifests})
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/training/pothole_v1')
    parser.add_argument('--config-output', type=Path, default=ROOT / 'configs/detector_ready.yaml')
    parser.add_argument('--negative-ratio', type=float, default=0.25, help='Training negatives per positive image; 0.25 gives 20%% background')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verify-only', action='store_true', help='Load all staged splits through Ultralytics and verify actual label counts')
    args = parser.parse_args()
    if args.verify_only:
        from ultralytics.data.dataset import YOLODataset
        config = yaml.safe_load(args.config_output.read_text())
        summary = json.loads((args.output / 'dataset_summary.json').read_text())
        for split, expected in summary['splits'].items():
            dataset = YOLODataset(img_path=config[split], data={'names': {0: 'pothole'}, 'nc': 1}, imgsz=640, augment=False, cache=False)
            actual = (len(dataset.labels), sum(len(r['cls']) > 0 for r in dataset.labels), sum(len(r['cls']) for r in dataset.labels))
            target = (expected['images'], expected['positive_images'], expected['pothole_boxes'])
            if actual != target:
                raise ValueError(f'{split}: actual {actual} != expected {target}')
            print(f'{split}: verified images, positive images, boxes = {actual}')
        return
    print(json.dumps(prepare(args.output, args.config_output, args.negative_ratio, args.seed), indent=2))


if __name__ == '__main__':
    main()
