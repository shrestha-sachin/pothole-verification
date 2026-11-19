"""Synthetic sources only; no training, downloads, or real dataset mutations."""
import csv
import json
import random
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
import yaml

from src.us_dataset import (Audit, file_hash, image_fingerprints, normalize_country, parse_voc,
                            parse_yolo, resolve_source, roboflow_metadata, stage, verify, version_paths, write_csv)
from src.portable_dataset import materialize_config
from scripts.import_roboflow_candidate import validate_export


def picture(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    im = Image.new('RGB', (32, 24))
    im.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(32*24)])
    im.save(path)
    return path


def voc(path, cls='D40', box=(5, 4, 15, 12)):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('<annotation><size><width>32</width><height>24</height></size><object><name>' + cls + '</name><bndbox>' +
                    ''.join(f'<{k}>{v}</{k}>' for k, v in zip(('xmin','ymin','xmax','ymax'), box)) + '</bndbox></object></annotation>')


@pytest.fixture
def source(tmp_path):
    root = tmp_path
    history = []
    for i, (name, split, cls) in enumerate([('US_A','train','D40'),('US_B','val','D40'),('US_T','test','D40'),('US_N','train','D20')]):
        image = picture(root/f'data/raw/RDD2022/United_States/train/images/{name}.png', i)
        voc(image.parent.parent/f'annotations/xmls/{name}.xml', cls)
        history.append(dict(source='rdd_United_States', original_id=name, image=image.relative_to(root).as_posix(),
                            split=split, group='sample_'+name))
    write_csv(root/'history.csv', history)
    (root/'data/annotations').mkdir(parents=True)
    recipe = dict(historical_manifest='history.csv', rdd_split_metadata='missing.csv', rdd_root='data/raw/RDD2022',
                  inventory='inventory.csv', annotations_root='data/annotations', roboflow_root='data/raw/rf',
                  protected_recording_ids=[])
    return root, recipe


def run(source, **kwargs):
    root, recipe = source
    audit = Audit(root, recipe, kwargs.pop('countries', ['United_States']), **kwargs)
    audit.run()
    return audit


def test_default_source_exclusion_and_no_unreviewed_crack_negatives(source):
    audit = run(source)
    assert {r['original_id'] for r in audit.selected} == {'US_A','US_B'}
    assert audit.summary['kaggle'] == 'excluded_by_policy'
    assert next(r for r in audit.records if r['original_id']=='US_N')['reason'] == 'negative_completeness_not_reviewed'
    assert not any(r['split']=='test' for r in audit.selected)


@pytest.mark.parametrize('alias,canonical', [('USA','United_States'),('United States','United_States'),('Czech Republic','Czech'),('czechia','Czech')])
def test_country_aliases(alias, canonical):
    assert normalize_country(alias) == canonical


def test_bad_country_does_not_fuzzy_match():
    with pytest.raises(ValueError):
        normalize_country('../United_States_extra')


def test_candidate_country_flag_does_not_approve_images(source):
    root, _ = source
    image = picture(root/'data/raw/RDD2022/Norway/train/images/NO_A.png', 20)
    voc(image.parent.parent/'annotations/xmls/NO_A.xml')
    audit = run(source, countries=['US','Norway','Czech'])
    assert not any(r['country']=='Norway' for r in audit.selected)
    assert any(r.get('availability')=='not_on_disk' and r['source']=='rdd_Czech' for r in audit.suitability)


def test_strict_rdd_class_and_boxes(tmp_path):
    p = tmp_path/'a.xml'
    voc(p, 'D20')
    assert parse_voc(p, (32,24))[0] == []
    voc(p, 'D40', (-1,2,15,20))
    with pytest.raises(ValueError, match='outside'):
        parse_voc(p, (32,24))
    voc(p)
    with pytest.raises(ValueError, match='dimension'):
        parse_voc(p, (100,100))


@pytest.mark.parametrize('text', ['5 .5 .5 0 .2', '5 nan .5 .2 .2', '5 .1 .5 .4 .2', '9 .5 .5 .2 .2','5 .5 .5 .2'])
def test_roboflow_invalid_labels_never_silently_become_negatives(text):
    with pytest.raises(ValueError):
        parse_yolo(text, {0:'crack',5:'pothole'})


def test_roboflow_class_five_maps_only_explicit_pothole():
    boxes, classes = parse_yolo('0 .5 .5 .2 .2\n5 .5 .5 .3 .3', {0:'crack',5:'pothole'})
    assert len(boxes) == 1 and boxes[0][2] == .3
    assert classes == ['crack','pothole']


def test_missing_rdd_annotation_excluded(source):
    root, _ = source
    (root/'data/raw/RDD2022/United_States/train/annotations/xmls/US_A.xml').unlink()
    audit = run(source)
    assert any(r['original_id']=='US_A' for r in audit.issues)
    assert not any(r['original_id']=='US_A' for r in audit.selected)


def test_staging_reproducible_portable_and_non_destructive(source):
    root, _ = source
    original = {p: file_hash(p) for p in root.rglob('*') if p.is_file()}
    a, b = run(source), run(source)
    first, second = root/'data/training/pothole_us_v2', root/'data/training/pothole_us_v3'
    cfg, cfg2 = root/'configs/detector_us_v2.yaml', root/'configs/detector_us_v3.yaml'
    stage(a, first, cfg)
    stage(b, second, cfg2)
    assert (first/'manifest.csv').read_bytes() == (second/'manifest.csv').read_bytes()
    assert verify(first)['train']['images'] == 1
    assert yaml.safe_load(cfg.read_text())['path'] == 'data/training/pothole_us_v2'
    runtime = materialize_config(cfg, root)
    assert Path(yaml.safe_load(runtime.read_text())['path']) == first
    assert all(file_hash(p) == digest for p, digest in original.items())
    with pytest.raises(FileExistsError):
        stage(a, first, cfg)
    next_output, _ = version_paths(first, cfg)
    assert next_output.name == 'pothole_us_v4'


def test_cross_platform_repo_paths_and_escape(tmp_path):
    assert resolve_source(tmp_path, 'C:\\old\\repo\\data\\raw\\a.jpg') == tmp_path/'data/raw/a.jpg'
    assert resolve_source(tmp_path, '/Users/person/project/data/raw/a.jpg') == tmp_path/'data/raw/a.jpg'
    with pytest.raises(ValueError):
        resolve_source(tmp_path, '../../secret')
    with pytest.raises(ValueError):
        resolve_source(tmp_path, 'D:/external/a.jpg')


def test_exact_duplicate_annotation_conflicts_quarantine_all_members(source):
    audit = run(source)
    a = next(r for r in audit.records if r['original_id']=='US_A')
    a['reason'] = ''
    b = dict(a, original_id='copy1', boxes=[(.5,.5,.1,.1)])
    c = dict(a, original_id='copy2')
    audit.records = [a,b,c]
    audit.deduplicate()
    assert all(r['reason']=='conflicting_exact_duplicate_labels' for r in audit.records)


def test_known_recording_cannot_cross_sources_or_splits(source):
    audit = run(source)
    a, b = [r for r in audit.records if not r['reason']]
    a['group'] = b['group'] = 'identity:recording'
    audit.group_check()
    assert a['reason'] == b['reason'] == 'group_split_conflict'


def test_perceptual_matches_require_review(source):
    audit = run(source)
    row = next(r for r in audit.records if r['original_id']=='US_A')
    a = dict(row, reason='')
    b = dict(row, reason='', original_id='near')
    # Force distinct exact hashes and close perceptual fingerprints to test policy.
    audit.records = [a,b]
    with patch('src.us_dataset.image_fingerprints', side_effect=[('a','0000000000000000'),('b','0000000000000001')]):
        audit.deduplicate()
    assert a['reason'] == b['reason'] == 'near_duplicate_requires_review'


def test_roboflow_export_schema_and_metadata(source):
    root, recipe = source
    directory = root/'data/raw/rf'
    image = picture(directory/'train/images/a.png', 77)
    label = directory/'train/labels/a.txt'
    label.parent.mkdir()
    label.write_text('5 .5 .5 .2 .2')
    (directory/'data.yaml').write_text(yaml.safe_dump(dict(nc=6,names=['Road-defects','Rutting','crack repair','lateral crack','longitudinal crack','pothole'],roboflow={'version':2,'license':'CC BY 4.0'})))
    report = roboflow_metadata(root, directory)
    assert report['splits']['train']['images'] == 1
    assert report['independent_original_image_count'] == 'unknown'
    audit = run(source, include_roboflow=True)
    assert not any(r['source']=='roboflow' for r in audit.selected)
    assert next(r for r in audit.records if r['source']=='roboflow')['reason'] == 'candidate_not_approved'


def test_incomplete_build_cannot_be_used_for_training(source):
    root, _ = source
    directory = root/'data/training/pothole_us_v2'
    directory.mkdir(parents=True)
    (directory/'build_status.json').write_text('{"status":"incomplete"}')
    config = root/'cfg.yaml'
    config.write_text(yaml.safe_dump(dict(project_relative=True,path='data/training/pothole_us_v2',evaluation_role='development_comparison_only')))
    with pytest.raises(ValueError, match='incomplete'):
        materialize_config(config, root)


def add_local(source, *, uncertain=False, boxes=None, sequence='sequence1', recording='', split='train', source_name='mapillary'):
    root, _ = source
    image = picture(root/'data/raw/mapillary/local.png', 199)
    annotation = dict(image_id='local', source_filepath=image.relative_to(root).as_posix(),
        image_width=32, image_height=24, review_status='reviewed', uncertain=uncertain,
        pothole_boxes=boxes if boxes is not None else [], timestamp_ms=0)
    (root/'data/annotations/local.json').write_text(json.dumps(annotation))
    write_csv(root/'inventory.csv', [dict(image_id='local', filepath=image.relative_to(root).as_posix(), source=source_name,
        review_status='reviewed', uncertain=False, sequence_id=sequence, recording_id=recording, split=split)])


def test_uncertain_local_never_included_as_negative(source):
    add_local(source, uncertain=True)
    audit = run(source)
    assert not any(r['source']=='mapillary' for r in audit.selected)
    assert any(r['finding']=='unreviewed_or_uncertain_annotation' for r in audit.issues)


def test_missing_local_recording_metadata_requires_review(source):
    add_local(source, sequence='')
    audit = run(source)
    assert any(r['finding']=='missing_recording_or_sequence' for r in audit.issues)


def test_final_local_recordings_remain_outside_preparation(source):
    add_local(source, recording='final_green_bay', boxes=[dict(x=2,y=3,w=5,h=6)])
    source[1]['protected_recording_ids'] = ['final_green_bay']
    audit = run(source)
    row = next(r for r in audit.records if r['source']=='mapillary')
    assert row['reason'] == 'protected_final_recording'
    assert row['pixel_sha256'] in audit.summary['historical_test_pixel_hashes']


def test_local_mapillary_selection_is_configurable(source):
    add_local(source, boxes=[dict(x=2,y=3,w=5,h=6)])
    assert any(r['source']=='mapillary' for r in run(source).selected)
    assert not any(r['source']=='mapillary' for r in run(source, include_mapillary=False).selected)


def test_importer_rejects_missing_labels_before_copy(source):
    root, _ = source
    directory = root/'data/raw/rf'
    picture(directory/'train/images/a.png', 99)
    (directory/'data.yaml').write_text(yaml.safe_dump(dict(nc=1,names=['pothole'])))
    with pytest.raises(FileNotFoundError):
        validate_export(directory)


def test_near_duplicate_protected_recording_cannot_leak(source):
    audit = run(source)
    row = next(r for r in audit.records if r['original_id']=='US_A')
    protected, proposed = dict(row, reason='protected_final_recording'), dict(row, reason='',original_id='new')
    audit.records = [protected, proposed]
    with patch('src.us_dataset.image_fingerprints', side_effect=[('a','0000000000000000'),('b','0000000000000001')]):
        audit.deduplicate()
    assert protected['reason']=='protected_final_recording'
    assert proposed['reason']=='near_duplicate_requires_review'


def test_explicit_negative_completeness_review_is_required_and_respected(source):
    root, _ = source
    reviews = root/'reviews.csv'
    write_csv(reviews, [dict(source='rdd_United_States',original_id='US_N',approved=True,
        reviewer='synthetic reviewer', reviewed_at='2026-11-18', target_annotation_complete=True)])
    audit = run(source, reviews=reviews, negative_ratio=1)
    assert any(r['original_id']=='US_N' and not r['boxes'] for r in audit.selected)


def test_verify_detects_a_split_removed_from_manifest(source):
    root, _ = source
    output, config = root/'out', root/'cfg.yaml'
    stage(run(source), output, config)
    rows = list(csv.DictReader((output/'manifest.csv').open()))
    with (output/'manifest.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows(r for r in rows if r['split']=='train')
    with pytest.raises(ValueError, match='Summary count mismatch'):
        verify(output)


def test_output_reservation_race_never_changes_competing_dataset(source):
    root, _ = source
    output, config = root/'out', root/'cfg.yaml'
    audit = run(source)
    original_mkdir = Path.mkdir
    def competing_create(path, *args, **kwargs):
        if path == output:
            original_mkdir(path)
            (path/'build_status.json').write_text('{"status":"complete","owner":"other"}')
            raise FileExistsError('Another process reserved this output')
        return original_mkdir(path, *args, **kwargs)
    with patch.object(Path, 'mkdir', competing_create), pytest.raises(FileExistsError):
        stage(audit, output, config)
    assert json.loads((output/'build_status.json').read_text())['owner']=='other'


def test_source_annotation_changes_stop_publication(source):
    root, _ = source
    audit = run(source)
    row = audit.selected[0]
    (root/row['annotation']).write_text('changed after audit')
    with pytest.raises(ValueError, match='annotation changed'):
        stage(audit, root/'out', root/'cfg.yaml')
    assert json.loads((root/'out/build_status.json').read_text())['status']=='incomplete'
