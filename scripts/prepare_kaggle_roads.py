"""Add Arcioni road images to training only, preserving existing evaluation splits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
HANDLE = 'lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes'


def pothole_labels(text: str) -> list[str]:
    """Validate every annotation; retain only class-zero axis-aligned boxes."""
    result = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] not in {'0', '1', '2'}:
            raise ValueError(f'Invalid YOLO box: {line}')
        x, y, w, h = map(float, fields[1:])
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)):
            raise ValueError(f'Invalid coordinates: {line}')
        if min(w, h) <= 0:
            if fields[0] != '0':
                continue  # Source has a zero-height crack; non-target labels are discarded.
            raise ValueError(f'Empty box: {line}')
        left, top = max(0.0, x-w/2), max(0.0, y-h/2)
        right, bottom = min(1.0, x+w/2), min(1.0, y+h/2)
        if fields[0] == '0':
            result.append('0 ' + ' '.join(f'{v:.6f}' for v in
                          ((left+right)/2, (top+bottom)/2, right-left, bottom-top)))
    return result


def pixel_hash(path: Path) -> str:
    # Detect identical decoded images even when metadata/file bytes differ.
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def sources(config: dict, split: str, root: Path) -> list[Path]:
    value = config.get(split, [])
    return [(root / p).resolve() for p in ([value] if isinstance(value, str) else value)]


def prepare(source: Path, output: Path, base: Path, config_output: Path) -> dict:
    if output.exists() or config_output.exists():
        raise FileExistsError('Output dataset/config already exists; choose new paths.')
    config = yaml.safe_load(base.read_text(encoding='utf-8'))
    dataset_root = Path(config.get('path', '.'))
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    existing_sources = {s: sources(config, s, dataset_root) for s in ('train', 'val', 'test')}
    missing_test = [str(p) for p in existing_sources['test'] if not p.is_file()]
    # Do not advertise a usable benchmark when the base test manifests are absent.
    if missing_test:
        existing_sources['test'] = []
    seen = set()
    for manifests in existing_sources.values():
        for manifest in manifests:
            if not manifest.is_file():
                raise ValueError(f'Expected existing image manifest: {manifest}')
            for entry in manifest.read_text(encoding='utf-8').splitlines():
                if not entry.strip():
                    continue
                # Match Ultralytics: ./ entries are relative to the manifest;
                # other relative entries are relative to the working directory.
                image = manifest.parent / entry[2:] if entry.startswith('./') else ROOT / entry
                seen.add(pixel_hash(image))
    images = sorted((source / 'images').glob('*'))
    images = [p for p in images if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    if not images:
        raise ValueError(f'No images found under {source}')
    records = []
    duplicates = 0
    for image in images:
        label = source / 'labels-YOLO' / f'{image.stem}.txt'
        boxes = pothole_labels(label.read_text(encoding='utf-8'))  # Missing labels are errors.
        digest = pixel_hash(image)
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        records.append((image, boxes, digest))
    if not records:
        raise ValueError('No new images remain after duplicate filtering')
    # Validate everything before publishing a new dataset; never modify source labels.
    (output / 'images').mkdir(parents=True)
    (output / 'labels').mkdir()
    manifest = []
    audit = []
    for index, (image, boxes, digest) in enumerate(records):
        name = f'arcioni_{index:05d}'
        target = output / 'images' / (name + image.suffix.lower())
        shutil.copy2(image, target)
        (output / 'labels' / (name + '.txt')).write_text('\n'.join(boxes) + ('\n' if boxes else ''), encoding='utf-8')
        manifest.append(target.resolve().as_posix())
        audit.append([image.as_posix(), target.resolve().as_posix(), digest, len(boxes), 'train'])
    train_manifest = output / 'train.txt'
    train_manifest.write_text('\n'.join(manifest) + '\n', encoding='utf-8')
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_image', 'image', 'pixel_sha256', 'potholes', 'split'])
        writer.writerows(audit)
    summary = dict(source=HANDLE, source_directory=str(source.resolve()), source_images=len(images),
                   added_images=len(records), positive_images=sum(bool(b) for _, b, _ in records),
                   negative_images=sum(not b for _, b, _ in records),
                   pothole_instances=sum(len(b) for _, b, _ in records), duplicates_excluded=duplicates,
                   missing_base_test_manifests=missing_test,
                   policy='Entire new source is training-only; original val/test unchanged. Exact decoded-pixel deduplication only; near-duplicates not ruled out.',
                   license='CC BY 4.0 per publication', citation='https://doi.org/10.1038/s41598-026-46679-4')
    (output / 'dataset_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    combined = {'path': ROOT.as_posix(), 'names': {0: 'pothole'}}
    for split, paths in existing_sources.items():
        if paths:
            combined[split] = [p.as_posix() for p in paths]
    combined['train'].append(train_manifest.resolve().as_posix())
    config_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.write_text(yaml.safe_dump(combined, sort_keys=False), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Downloaded data folder containing images/ and labels-YOLO/')
    parser.add_argument('--base-config', type=Path, default=ROOT / 'configs/detector.yaml')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed/kaggle_arcioni_v4')
    parser.add_argument('--config-output', type=Path, default=ROOT / 'configs/detector_kaggle.yaml')
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output, args.base_config, args.config_output), indent=2))


if __name__ == '__main__':
    main()
