"""Inventory generated data and retired US builds without changing them."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.us_dataset import file_hash, read_csv, resolve_source, write_csv


ARCHIVE = Path('results/dataset_archive/retired_us_builds')
RETIRED = (
    'data/training/pothole_us_v3', 'data/training/pothole_us_v4',
    'configs/detector_us_v3.yaml', 'configs/detector_us_v4.yaml',
    'results/dataset_audits/us_v3_final', 'results/dataset_audits/v4_preparation',
)
GUIDES = ('us_dataset_v4.md', 'us_dataset_preparation.md', 'us_dataset_audit.md')
PROCESSED = {
    'rdd2022_v3': ('RDD2022', 'prepare_training_dataset.py, detector_v3.yaml, split integrity tests, Experiment 2 split provenance', 'V1 requires these exact split manifests and labels'),
    'rdd2022': ('RDD2022', 'detector.yaml, detector_combined.yaml, detector_kaggle.yaml, RDD conversion and subsampling scripts', 'Legacy experiment configurations still reference this conversion'),
    'kaggle_arcioni_v4': ('Kaggle Arcioni', 'prepare_training_dataset.py, detector_kaggle.yaml, legacy run metadata', 'V1 reproduction reads these transformed images and labels'),
    'verifier_crops': ('Green Bay / Mapillary', 'crop_inventory.csv, train_verifier.py, generate_crops.py', 'Verifier training and research provenance depend on the crop inventory'),
    'video_frames': ('Green Bay video', 'image_inventory.csv, annotation JSON, annotate.py, export_yolo.py, generate_crops.py', 'Reviewed image pixels and manual box coordinates depend on these frames'),
    'experiment_2_recordings': ('Green Bay video', 'recording manifests, annotation workspace, Experiment 2 preparation', 'Recorded frames and manual reviews are preparation inputs'),
}


def processed_inventory(root):
    raw = root/'data/raw'
    rdd_xml = {p.stem: p.relative_to(root).as_posix() for p in sorted((raw/'RDD2022').rglob('*.xml'))}
    kaggle = {}
    for row in read_csv(root/'data/processed/kaggle_arcioni_v4/manifest.csv'):
        if row.get('image') and row.get('source_image'):
            source_path = resolve_source(root, row['source_image'])
            kaggle[Path(row['image']).stem] = source_path
    inventory = {row['image_id']: row.get('filepath', '') for row in read_csv(root/'data/metadata/image_inventory.csv')}
    crops = {row['crop_id']: row['source_image_id'] for row in read_csv(root/'data/metadata/crop_inventory.csv')}
    annotations = defaultdict(list)
    for path in sorted((root/'data/annotations').rglob('*.json')):
        try:
            doc = json.loads(path.read_text(encoding='utf-8'))
            image = resolve_source(root, doc['source_filepath']).relative_to(root).as_posix()
            annotations[image].append(path.relative_to(root).as_posix())
        except (KeyError, ValueError, OSError):
            continue
    videos = {p.stem.replace(' ', '_'): p.relative_to(root).as_posix()
              for p in (raw/'local_video').iterdir() if p.is_file()}
    rows = []
    for path in sorted((root/'data/processed').rglob('*')):
        if not path.is_file():
            continue
        identity = path.relative_to(root).as_posix()
        directory = path.relative_to(root/'data/processed').parts[0]
        source, references, reason = PROCESSED.get(directory, ('unknown', 'unverified', 'Provenance and dependencies require review'))
        original = ''
        if directory.startswith('rdd2022'):
            original = rdd_xml.get(path.stem, '')
        elif directory == 'kaggle_arcioni_v4':
            source_path = kaggle.get(path.stem)
            if source_path:
                if path.suffix == '.txt':
                    source_path = source_path.parent.parent/'labels-YOLO'/(source_path.stem+'.txt')
                original = source_path.relative_to(root).as_posix()
        elif directory == 'video_frames':
            original = videos.get(path.parent.name, '')
        elif directory == 'verifier_crops':
            original = inventory.get(crops.get(path.stem), '')
        dependent = list(annotations.get(identity, ()))
        if original and directory == 'verifier_crops':
            dependent.extend(annotations.get(original, ()))
        action = 'keep' if directory in PROCESSED else 'review_keep_in_place'
        if path.suffix == '.bak':
            action, reason = 'review_keep_in_place', 'Historical backup may contain a different split or unique provenance'
        rows.append(dict(path=str(path), repo_path=identity, source=source, bytes=path.stat().st_size,
            sha256=file_hash(path), authoritative_original=original,
            original_exists=bool(original and (root/original).is_file()), annotation_dependencies=dependent,
            references=references, action=action, reason=reason))
    by_size_hash = defaultdict(list)
    for row in rows:
        by_size_hash[(row['bytes'], row['sha256'])].append(row)
    for matches in by_size_hash.values():
        for row in matches:
            row['identical_processed_files'] = len(matches)-1
    return rows


def retired_inventory(root):
    plan, rows = [], []
    machine_configs = root/'results/machine_configs'
    generated_configs = []
    for version in (3, 4):
        for path in sorted(machine_configs.glob(f'detector_us_v{version}_*.yaml')):
            import yaml
            data = yaml.safe_load(path.read_text(encoding='utf-8'))
            if isinstance(data, dict) and data.get('path') and resolve_source(root, data['path']) == root/f'data/training/pothole_us_v{version}':
                generated_configs.append(path.relative_to(root).as_posix())
    targets = [(name, (ARCHIVE/name).as_posix(), 'generated') for name in (*RETIRED, *generated_configs)]
    targets.extend((f'docs/{name}', f'docs/archive/{name}', 'authored historical guide') for name in GUIDES)
    for source, destination, provenance in targets:
        path = root/source
        if not path.exists():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Archive source is outside the workspace: {source}')
        entries = sorted(path.rglob('*')) if path.is_dir() else [path]
        files = [entry for entry in entries if entry.is_file()]
        plan.append(dict(source=source, destination=destination, files=len(files),
                         bytes=sum(p.stat().st_size for p in files), provenance=provenance))
        for file in files:
            if file.is_symlink() or not file.resolve().is_relative_to(root):
                raise ValueError(f'Archive contains linked or external content: {file}')
            # Preserve every file in a reversible archive, including historical notes.
            rows.append(dict(path=str(file), repo_path=file.relative_to(root).as_posix(), target=source,
                archive_path=(Path(destination)/file.relative_to(path)).as_posix() if path.is_dir() else destination,
                bytes=file.stat().st_size, sha256=file_hash(file), provenance=provenance,
                action='archive_reversibly', v1_dependency=False,
                reason='Retired US V3/V4 result or historical guide. V1 uses RDD V3 source conversion, which is retained.'))
    return plan, rows


def audit(root, report):
    root = Path(root).resolve()
    report = Path(report).resolve()
    if not report.is_relative_to(root/'results'):
        raise ValueError('Write the audit to a new directory under results/')
    report.mkdir(parents=True, exist_ok=False)
    processed = processed_inventory(root)
    plan, retired = retired_inventory(root)
    write_csv(report/'processed_inventory.csv', processed)
    write_csv(report/'retired_inventory.csv', retired)
    write_csv(report/'archive_plan.csv', plan, ['source', 'destination', 'files', 'bytes', 'provenance'])
    summary = dict(processed_files=len(processed), processed_bytes=sum(r['bytes'] for r in processed),
        processed_actions=dict(Counter(r['action'] for r in processed)), archive_roots=len(plan),
        archive_files=len(retired), archive_bytes=sum(r['bytes'] for r in retired),
        v2='retained; separate archival decision',
        shared_review_code='retained', permanent_deletions=0)
    (report/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(ROOT, args.report), indent=2))
