"""Experiment 2 tests use synthetic data and never load detector weights."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from scripts.audit_dataset_cleanup import audit as cleanup_audit
from src.dataset_reviews import save_event
from src.experiment_2 import (Experiment2Audit, parse_rdd, rdd_crack_profile, rdd_negative_pilot,
                              sample_training, stage_experiment, verify_experiment, write_report)
from src.experiment_2_recordings import load_collection, register_recording
from src.portable_dataset import materialize_config
from src.us_dataset import file_hash, read_csv, write_csv
from tests.test_us_dataset import picture, voc


def add_rdd(root, country, identity, seed, cls='D40'):
    train = root/f'data/raw/RDD2022/{country}/train'
    image = picture(train/'images'/f'{identity}.png', seed)
    label = train/'annotations/xmls'/f'{identity}.xml'
    voc(label, cls)
    return image, label


def add_object(path, cls, coordinates=(16, 2, 29, 4)):
    fields = ('xmin','ymin','xmax','ymax')
    box = ''.join(f'<{field}>{value}</{field}>' for field,value in zip(fields,coordinates))
    annotation = f'<object><name>{cls}</name><bndbox>{box}</bndbox></object>'
    path.write_text(path.read_text().replace('</annotation>', annotation+'</annotation>'))


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path
    history = []
    examples = [('Japan','JP_train','train','D40'), ('Japan','JP_val','val','D40'),
                ('Japan','JP_test','test','D40'), ('Norway','NO_train','train','D40'),
                ('Japan','JP_crack','train','D20')]
    for index,(country,identity,split,cls) in enumerate(examples):
        image, label = add_rdd(root,country,identity,index,cls)
        if identity == 'JP_train':
            for code, coordinates in [('D43',(1,1,2,2)),('D44',(2,2,3,3)),('D50',(3,3,4,4))]:
                add_object(label,code,coordinates)
        history.append(dict(source='rdd_'+country, original_id=identity, image=image.relative_to(root).as_posix(),
                            split=split, group=identity))
    add_rdd(root,'United_States','excluded_us',10)
    add_rdd(root,'India','excluded_india',11)
    write_csv(root/'history.csv', history)
    return root, dict(historical_manifest='history.csv',rdd_split_metadata='missing.csv',rdd_root='data/raw/RDD2022',
        inventory='data/metadata/image_inventory.csv',annotations_root='data/annotations',roboflow_root='data/raw/rf',
        recordings_root='data/processed/experiment_2_recordings',protected_recording_ids=[])


def run(fixture, **kwargs):
    audit = Experiment2Audit(*fixture, **kwargs)
    audit.run()
    return audit


def test_japan_extra_classes_are_validated_but_only_d40_is_emitted(fixture):
    audit = run(fixture)
    row = next(row for row in audit.selected if row['original_id'] == 'JP_train')
    assert row['classes'] == ['D40','D43','D44','D50']
    assert len(row['boxes']) == 1
    assert row['original_target_boxes'][0]['xmin'] == '5'
    assert {row['source'] for row in audit.selected} == {'rdd_Japan','rdd_Norway'}
    assert not any(row['original_id'] in {'JP_test','JP_crack','excluded_us','excluded_india'} for row in audit.selected)


@pytest.mark.parametrize('code,coords,error', [
    ('D40',(5,4,15,12),'duplicate_annotation_box'),
    ('D44',(5,4,15,12),'conflicting_annotation_classes'),
    ('D50',(1,1,1,3),'zero_area_box'),
    ('D43',(-1,1,3,3),'box_outside_image'),
    ('unknown',(1,1,2,2),'unsupported_rdd_class'),
])
def test_no_silent_clipping_or_removal_of_bad_boxes(tmp_path,code,coords,error):
    path = tmp_path/'image.xml'
    voc(path)
    add_object(path,code,coords)
    with pytest.raises(ValueError,match=error):
        parse_rdd(path,'image.png',(32,24),'Japan')


def image_event(root, image, label, **extra):
    return dict(original_image=image.relative_to(root).as_posix(),annotation=label.relative_to(root).as_posix(),
                reviewer='Fixture reviewer',notes='Synthetic whole-image review',inspection_confirmed=True,**extra)


def test_negative_approval_requires_explicit_full_image_review_and_current_hash(fixture):
    root,_ = fixture
    train = root/'data/raw/RDD2022/Japan/train'
    image,label = train/'images/JP_crack.png',train/'annotations/xmls/JP_crack.xml'
    event = image_event(root,image,label,kind='negative',source='rdd_Japan',original_id='JP_crack',
        decision='pothole_free',target_annotation_complete=True,negative_subtypes=['crack'])
    assert not any(row['original_id']=='JP_crack' for row in run(fixture).selected)
    save_event(root,root/'results/dataset_reviews/experiment_2',event)
    row = next(row for row in run(fixture).selected if row['original_id']=='JP_crack')
    assert not row['boxes'] and row['negative_subtypes']==['crack']
    voc(label,'D20',box=(6,4,15,12))
    assert not any(row['original_id']=='JP_crack' for row in run(fixture).selected)


def test_roboflow_class_five_and_explicit_capture_approval(fixture):
    from src.us_dataset import CHARACTERISTICS
    root,_ = fixture
    rf = root/'data/raw/rf'
    image = picture(rf/'valid/images/frame.png',40)
    label = rf/'valid/labels/frame.txt'
    label.parent.mkdir()
    label.write_text('0 .5 .5 .1 .1\n5 .6 .6 .2 .2\n')
    (rf/'data.yaml').write_text(yaml.safe_dump(dict(nc=6,names=['damage','rutting','repair','lateral','longitudinal','pothole'],
        roboflow=dict(workspace='the-american-university-of-the-middle-east',project='pothole-detection-3uuq2',version=2,license='CC BY 4.0'))))
    audit = run(fixture)
    candidate = next(row for row in audit.records if row['source']=='roboflow')
    assert candidate['reason']=='roboflow_image_review_required' and len(candidate['boxes'])==1
    directory = root/'results/dataset_reviews/experiment_2'
    event = image_event(root,image,label,kind='source',source='roboflow',original_id='valid/frame',decision='approved',
        annotation_definition_checked=True,split='train',**{key:'reviewed synthetic evidence' for key in CHARACTERISTICS})
    save_event(root,directory,event)
    assert not any(row['source']=='roboflow' for row in run(fixture).selected)
    save_event(root,directory,{**event,'recording_id':'confirmed_capture'})
    row = next(row for row in run(fixture).selected if row['source']=='roboflow')
    assert row['split']=='train' and row['original_target_boxes'][0][0]==5
    assert not any(row['source']=='roboflow' for row in run(fixture,include_roboflow=False).selected)


def test_exact_test_contamination_across_sources_and_historical_val_reuse_are_blocked(fixture):
    root,_ = fixture
    train = root/'data/raw/RDD2022/Japan/train/images'
    test_copy,_ = add_rdd(root,'Norway','same_as_test',100)
    val_copy,_ = add_rdd(root,'Norway','same_as_val',101)
    test_copy.write_bytes((train/'JP_test.png').read_bytes())
    val_copy.write_bytes((train/'JP_val.png').read_bytes())
    rows = {row['original_id']:row for row in run(fixture).records if row['source'] not in {'historical_test','historical_val_reference'}}
    assert rows['same_as_test']['reason']=='exact_protected_test_overlap'
    assert rows['same_as_val']['reason']=='exact_historical_validation_overlap'


def test_perceptual_pairs_are_reported_and_require_explicit_review(fixture):
    root,_ = fixture
    audit = run(fixture)
    rows = [dict(row) for row in audit.selected if row['split']=='train']
    audit.records = rows
    custom = [('one','0000000000000000'),('two','0000000000000001')]
    with patch('src.experiment_2.image_fingerprints',side_effect=custom):
        audit.deduplicate()
    assert all(row['reason']=='near_duplicate_requires_review' for row in rows)
    assert any(row['resolution']=='quarantined' for row in audit.duplicates)
    for row in rows:
        row['reason']=''
    from src.dataset_reviews import load_events
    path = root/'results/dataset_reviews/experiment_2'
    save_event(root,path,dict(kind='duplicate',original_image=rows[0]['original_image'],right_image=rows[1]['original_image'],
        decision='different_scenes',inspection_confirmed=True,reviewer='Fixture reviewer',notes='Distinct fixture scenes'))
    audit.events = load_events(root,path)
    with patch('src.experiment_2.image_fingerprints',side_effect=custom):
        audit.deduplicate()
    assert all(not row['reason'] for row in rows)


def test_group_aliases_connect_recordings_and_physical_defects(fixture):
    audit = Experiment2Audit(*fixture)
    audit.records = [dict(source='local',original_id='a',group='r1',recording_id='one',split='train',reason=''),
                     dict(source='local',original_id='b',group='r2',recording_id='one',
                          physical_defect_keys=['physical:same_defect'],split='train',reason=''),
                     dict(source='mapillary',original_id='c',group='mly',physical_defect_keys=['physical:same_defect'],split='val',reason='')]
    audit.group_check()
    assert len(audit.group_conflicts)==1
    assert all(row['reason']=='recording_sequence_or_defect_split_conflict' for row in audit.records)


def test_balance_never_drops_positives_or_fabricates_negatives():
    positives = [dict(source='positive-source',group='same',original_image=f'positive/{i}',boxes=[(0.5,)*4]) for i in range(8)]
    negatives = [dict(source='background-source',group=str(i%3),original_image=f'negative/{i}',boxes=[]) for i in range(10)]
    for ratio, size in [(0.25,10),(0.5,12),(1.,16),(2.,18),(None,18)]:
        selection = sample_training(positives+negatives,ratio,42)
        assert len(selection)==size and selection[:8]==positives
        assert len({row['original_image'] for row in selection})==size
        assert selection == sample_training(positives+negatives,ratio,42)


def test_build_and_verify_keep_validation_fixed_and_do_not_touch_inputs(fixture):
    root,_ = fixture
    original = {path:file_hash(path) for path in root.rglob('*') if path.is_file()}
    a,b = run(fixture,negative_ratio=.25),run(fixture,negative_ratio=1)
    assert a.summary['validation_fingerprint']==b.summary['validation_fingerprint']
    output,config = root/'data/training/pothole_experiment_2',root/'configs/detector_experiment_2.yaml'
    stage_experiment(b,output,config)
    assert verify_experiment(output)['val']['images']==1
    data = yaml.safe_load(materialize_config(config,root).read_text())
    assert Path(data['path']) == output and 'test' not in data
    with pytest.raises(FileExistsError):
        stage_experiment(b,output,config)
    assert all(file_hash(path)==digest for path,digest in original.items())
    target = output/read_csv(output/'manifest.csv')[0]['label']
    target.write_text('0 .6 .6 .1 .1\n')
    with pytest.raises(ValueError,match='Prepared label bytes changed'):
        verify_experiment(output)


def test_cleanup_inventory_leaves_linked_processed_inputs_and_raw_data_untouched(fixture):
    root,_ = fixture
    (root/'data/raw/local_video').mkdir(parents=True)
    keep = root/'data/processed/rdd2022_v3/train.txt';keep.parent.mkdir(parents=True);keep.write_text('V1 dependency')
    retired = root/'data/training/pothole_us_v4/manifest.csv';retired.parent.mkdir(parents=True);retired.write_text('retired bytes')
    report = root/'results/cleanup'
    summary = cleanup_audit(root,report)
    assert summary['archive_files']==1 and summary['permanent_deletions']==0
    assert read_csv(report/'processed_inventory.csv')[0]['action']=='keep'
    assert read_csv(report/'retired_inventory.csv')[0]['sha256']==file_hash(retired)
    assert keep.read_text()=='V1 dependency' and retired.read_text()=='retired bytes'


def test_new_recording_opens_existing_annotation_dashboard(fixture,monkeypatch):
    from scripts import extract_frames
    from src import annotations as ann_io
    from streamlit.testing.v1 import AppTest

    root,_ = fixture
    video = root/'data/raw/local_video/greenbay.avi';video.parent.mkdir(parents=True);video.write_bytes(b'synthetic')
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    def extract(**kwargs):
        assert kwargs['no_overwrite'] and kwargs['target_fps']==0.5
        frame = picture(kwargs['output_dir']/'f.png',97)
        write_csv(kwargs['output_dir']/'frames.csv',[dict(video_id='greenbay',frame_index=1,timestamp_ms=2000,
            filepath=frame.relative_to(root).as_posix(),width=32,height=24)])
    monkeypatch.setattr(extract_frames,'extract',extract)
    folder = register_recording(root,video.relative_to(root),'greenbay',split='train',interval=2,
                                camera='dashcam',area='Green Bay')
    metadata,rows = load_collection(root,folder,verify_video=True)
    assert rows[0]['review_status']=='unreviewed' and metadata['video_sha256']==digest
    assert not any(row['original_id']==rows[0]['image_id'] for row in run(fixture).selected)
    monkeypatch.setattr(ann_io,'ANNOTATIONS_ROOT',root/'data/annotations')
    source = Path('dashboard/annotate.py').read_text(encoding='utf-8')
    source = source.replace('PROJECT_ROOT = Path(__file__).resolve().parents[1]',f'PROJECT_ROOT = Path({str(root)!r})')
    source = source.replace('parser.parse_known_args()',"parser.parse_known_args(['--collection','greenbay'])")
    source = source.replace("str(Path(__file__).parent / 'annotation_canvas')",repr(str(Path('dashboard/annotation_canvas').resolve())))
    app = AppTest.from_string(source).run(timeout=30)
    assert not app.exception
    next(w for w in app.selectbox if w.label=='Review decision').set_value('reviewed')
    next(w for w in app.checkbox if w.label.startswith('I checked the entire')).set_value(True)
    next(w for w in app.button if w.label=='Save review').click().run()
    assert not app.exception and not app.error
    annotation = json.loads((root/'data/annotations/experiment_2/greenbay'/f"{rows[0]['image_id']}.json").read_text())
    assert annotation['timestamp_ms']==2000 and annotation['target_annotation_complete']
    assert any(row['original_id']==rows[0]['image_id'] and not row['boxes'] for row in run(fixture).selected)
    assert file_hash(video)==digest and not (root/'data/metadata/image_inventory.csv').exists()


def test_legacy_training_video_cannot_be_registered_as_validation(tmp_path):
    original = tmp_path/'data/raw/local_video/Green Bay Training.mp4'
    original.parent.mkdir(parents=True)
    original.write_bytes(b'legacy training video')
    renamed = original.with_name('new_name.mp4')
    renamed.write_bytes(original.read_bytes())
    inventory = tmp_path/'data/metadata/image_inventory.csv'
    inventory.parent.mkdir(parents=True)
    write_csv(inventory, [dict(recording_id='Green_Bay_Training', split='train')])

    for video in (original, renamed):
        with pytest.raises(ValueError, match='(original|image) inventory'):
            register_recording(tmp_path, video, 'new_validation', split='val', interval=2,
                               camera='dashcam', area='Green Bay')
    assert not (tmp_path/'data/processed').exists()


def test_rdd_negative_pilot_interleaves_countries_and_preserves_split_groups():
    def row(country, number, classes, *, split='train', known=True, group=None):
        image = f'{country}_{number}'
        return dict(source='rdd_'+country, original_id=image, original_image=image+'.jpg', split=split,
                    group=group or image, published_crack_profile=rdd_crack_profile(classes),
                    group_provenance='historical_sample_group' if known else 'publisher_train_only_sample_fallback')

    japan = [row('Japan', index, ['D20']) for index in range(8)]
    norway = [row('Norway', 1, ['D20', 'D00']), row('Norway', 2, ['D10'])]
    candidates = japan + norway + [row('Japan', 10, ['D00'], group=japan[0]['group']),
        row('Japan', 11, ['D20'], split='val'), row('Norway', 12, ['D10'], split='test'),
        row('Norway', 13, ['D20'], known=False)]
    selected = rdd_negative_pilot(candidates, 42)

    assert [item['source'] for item in selected[:4]] == ['rdd_Japan', 'rdd_Norway'] * 2
    assert len(selected) == len({item['group'] for item in selected}) == 10
    assert all(item['split']=='train' and item['group_provenance']=='historical_sample_group' for item in selected)
    assert selected == rdd_negative_pilot(list(reversed(candidates)), 42)


def test_ambiguous_cavity_is_withheld_and_review_media_omits_test_images(fixture,monkeypatch):
    root,recipe = fixture
    flags = root/'flags.csv'
    write_csv(flags,[dict(source='rdd_Japan',original_id='JP_train',finding='Cavity boundary unclear')])
    audit = run((root,{**recipe,'visual_review_flags':'flags.csv'}))
    row = next(row for row in audit.records if row['source']=='rdd_Japan' and row['original_id']=='JP_train')
    assert row['reason']=='visual_cavity_review_required' and row['boxes']
    def check_sheet(_root,rows,path,*args,**kwargs):
        assert all(row['split']!='test' for row in rows), path
    monkeypatch.setattr('src.experiment_2.contact_sheet',check_sheet)
    write_report(audit,root/'report')
    queue = read_csv(root/'report/review_queue/rdd_positive_checks.csv')
    assert queue[0]['original_id']=='JP_train' and queue[0]['visual_review_note']=='Cavity boundary unclear'
    assert all(row['original_id']!='JP_test' for row in queue)
