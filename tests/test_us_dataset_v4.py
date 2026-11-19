"""V4 expansion must not buy image count by weakening review or split safety."""
import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.dataset_reviews import save_event, load_events, pair_key
from src.us_dataset import Audit, CHARACTERISTICS, read_csv, stage, write_csv
from src.us_dataset_v4 import V4Audit, negative_sample, stage_v4, verify_v4
from tests.test_us_dataset import source, picture, voc, add_local


@pytest.fixture
def v4(source):
    root,recipe=source
    audit=Audit(root,recipe,['United_States']);audit.run()
    reference=root/'data/training/pothole_us_v3'
    stage(audit,reference,root/'configs/detector_us_v3.yaml')
    recipe={**recipe,'validation_reference':'data/training/pothole_us_v3'}
    return root,recipe


def build_audit(v4,**kwargs):
    audit=V4Audit(*v4,['United_States'],**kwargs);audit.run();return audit


def event_for(v4,name='US_N',**extra):
    root,_=v4
    return dict(kind='negative',source='rdd_United_States',original_id=name,
        original_image=f'data/raw/RDD2022/United_States/train/images/{name}.png',
        annotation=f'data/raw/RDD2022/United_States/train/annotations/xmls/{name}.xml',
        decision='pothole_free',target_annotation_complete=True,inspection_confirmed=True,
        reviewer='Synthetic reviewer',notes='Synthetic full-image review',**extra)


def test_global_budget_uses_verified_negatives_across_source_boundaries():
    rows=[dict(source='us',group='us',original_id=str(i),boxes=[(.5,.5,.1,.1)])for i in range(8)]
    rows += [dict(source='local',group='local'+str(i%3),original_id='n'+str(i),boxes=[])for i in range(20)]
    for ratio,expected in [(.25,2),(.5,4),(1,8)]:
        selected=negative_sample(rows,ratio,42)
        assert sum(not r['boxes']for r in selected)==expected
        assert len({r['original_id']for r in selected})==len(selected)
        assert selected==negative_sample(rows,ratio,42)
    assert len(negative_sample(rows,20,42))==len(rows) # never manufacture or resample
    assert len(negative_sample(rows,None,42))==len(rows)


def test_frozen_validation_and_protected_test(v4):
    audit=build_audit(v4)
    assert {r['original_id']for r in audit.selected if r['split']=='val'}=={'US_B'}
    assert not any(r['original_id']=='US_T'for r in audit.selected)
    root,_=v4
    stage_v4(audit,root/'data/training/pothole_us_v4',root/'configs/detector_us_v4.yaml')
    assert verify_v4(root/'data/training/pothole_us_v4',project_root=root)['val']['images']==1


def test_us_negative_requires_content_bound_explicit_approval(v4):
    assert not any(r['original_id']=='US_N'for r in build_audit(v4).selected)
    root,_=v4
    save_event(root,root/'results/dataset_reviews/v4',event_for(v4))
    audit=build_audit(v4,negative_ratio=1)
    assert any(r['original_id']=='US_N'and not r['boxes']for r in audit.selected)


def test_stale_approval_and_revocation_cannot_release_an_image(v4):
    root,_=v4;directory=root/'results/dataset_reviews/v4'
    save_event(root,directory,event_for(v4))
    picture(root/'data/raw/RDD2022/United_States/train/images/US_N.png',177)
    assert not any(r['original_id']=='US_N'for r in build_audit(v4,negative_ratio=1).selected)
    assert build_audit(v4).review_warnings
    save_event(root,directory,event_for(v4))
    revoked=event_for(v4);revoked['decision']='unsure'
    save_event(root,directory,revoked)
    assert not any(r['original_id']=='US_N'for r in build_audit(v4,negative_ratio=1).selected)


def test_no_legacy_unbound_review_override(v4,tmp_path):
    with pytest.raises(ValueError,match='content-bound'):
        V4Audit(*v4,['US'],reviews=tmp_path/'unbound.csv')


def test_changed_frozen_annotation_fails_instead_of_shrinking_validation(v4):
    root,_=v4
    voc(root/'data/raw/RDD2022/United_States/train/annotations/xmls/US_B.xml',box=(6,4,15,12))
    with pytest.raises(ValueError,match='Frozen validation annotation changed'):
        build_audit(v4)


def test_review_events_are_append_only_and_never_modify_originals(v4):
    root,_=v4;directory=root/'results/dataset_reviews/v4';event=event_for(v4)
    before=(root/event['annotation']).read_bytes()
    first=save_event(root,directory,event);second=save_event(root,directory,event)
    assert first!=second and first.is_file() and second.is_file()
    assert (root/event['annotation']).read_bytes()==before
    assert len(load_events(root,directory))==1
    event['target_annotation_complete']=False
    with pytest.raises(ValueError,match='complete-image'):
        save_event(root,directory,event)


def test_source_flag_never_approves_norway(v4):
    root,_=v4
    image=picture(root/'data/raw/RDD2022/Norway/train/images/NO_A.png',233)
    voc(image.parent.parent/'annotations/xmls/NO_A.xml')
    audit=V4Audit(*v4,['US','Norway']);audit.run()
    assert not any(r['source']=='rdd_Norway'for r in audit.selected)


def test_new_recording_requires_explicit_assignment_and_metadata(v4):
    root,_=v4
    image=picture(root/'data/raw/new_capture/frame.png',388)
    annotation=root/'data/new_annotations/frame.json';annotation.parent.mkdir()
    ann=dict(source_filepath=image.relative_to(root).as_posix(),video_id='new_drive',timestamp_ms=1200,
             image_width=32,image_height=24,review_status='reviewed',uncertain=False,pothole_boxes=[dict(x=1,y=2,w=3,h=4)])
    annotation.write_text(json.dumps(ann))
    intake=root/'intake.csv'
    write_csv(intake,[dict(source='local',image_id='newframe',filepath=image.relative_to(root).as_posix(),
         annotation=annotation.relative_to(root).as_posix(),recording_id='new_drive',timestamp_ms=1200,
         image_variant='raw',license='collector permission')])
    audit=build_audit(v4,intake=intake)
    assert any(r['finding']=='intake_recording_assignment_required'for r in audit.issues)
    assignments=root/'assignments.csv'
    write_csv(assignments,[dict(recording_id='new_drive',split='train',role='training',approved=True,reviewer='Synthetic',reviewed_at='2026-11-19')])
    audit=build_audit(v4,intake=intake,assignments=assignments)
    assert any(r['original_id']=='newframe'for r in audit.selected)
    ann['timestamp_ms']=1400;annotation.write_text(json.dumps(ann))
    assert not any(r['original_id']=='newframe'for r in build_audit(v4,intake=intake,assignments=assignments).selected)


def test_duplicate_hash_alone_never_clears_pairs(v4):
    audit=build_audit(v4)
    left=next(r for r in audit.records if r['original_id']=='US_A')
    right=dict(left,original_id='second',reason='',group='identity:same')
    left.update(reason='',group='identity:same')
    audit.records=[left,right];audit.frozen={}
    with patch('src.us_dataset.image_fingerprints',side_effect=[('x','0000000000000000'),('y','0000000000000001')]):
        audit.deduplicate()
    assert all(r['reason']=='near_duplicate_requires_review'for r in audit.records)


def test_duplicate_release_requires_matching_pair_event(v4):
    root,_=v4;audit=build_audit(v4)
    left=next(r for r in audit.records if r['original_id']=='US_A')
    image=picture(root/'data/raw/copy.png',456)
    right=dict(left,original_id='second',original_image=image.relative_to(root).as_posix(),reason='')
    left['reason']='';audit.records=[left,right];audit.frozen={}
    save_event(root,root/'results/dataset_reviews/v4',dict(kind='duplicate',original_image=left['original_image'],right_image=right['original_image'],
        decision='different_scenes',inspection_confirmed=True,reviewer='Synthetic',notes='Distinct physical streets in synthetic fixture'))
    audit.events=load_events(root,root/'results/dataset_reviews/v4')
    with patch('src.us_dataset.image_fingerprints',side_effect=[('x','0000000000000000'),('y','0000000000000001')]):
        audit.deduplicate()
    assert not left['reason'] and not right['reason']


def test_ratio_validation_and_source_exclusions(v4):
    for value in [-1,float('nan'),3]:
        with pytest.raises(ValueError):V4Audit(*v4,['US'],negative_ratio=value)
    with pytest.raises(ValueError,match='permits US'):
        V4Audit(*v4,['US','Japan'])


def test_norway_needs_both_source_flag_and_explicit_image_approval(v4):
    root,_=v4
    image=picture(root/'data/raw/RDD2022/Norway/train/images/NO_A.png',299)
    annotation=image.parent.parent/'annotations/xmls/NO_A.xml';voc(annotation)
    event=dict(kind='source',source='rdd_Norway',original_id='NO_A',original_image=image.relative_to(root).as_posix(),
        annotation=annotation.relative_to(root).as_posix(),decision='approved',annotation_definition_checked=True,
        inspection_confirmed=True,reviewer='Synthetic reviewer',notes='Synthetic annotation and suitability review',
        recording_id='verified_norway_capture',split='train',**{k:'synthetic reviewed evidence'for k in CHARACTERISTICS})
    save_event(root,root/'results/dataset_reviews/v4',event)
    assert not any(r['source']=='rdd_Norway'for r in build_audit(v4).selected)
    audit=V4Audit(*v4,['US','Norway']);audit.run()
    assert any(r['source']=='rdd_Norway'for r in audit.selected)


def test_training_label_tampering_is_detected_even_when_box_count_matches(v4):
    root,_=v4;output=root/'data/training/pothole_us_v4'
    stage_v4(build_audit(v4),output,root/'configs/detector_us_v4.yaml')
    row=next(r for r in read_csv(output/'manifest.csv')if r['split']=='train')
    (output/row['label']).write_text('0 .5 .5 .1 .1\n')
    with pytest.raises(ValueError,match='label content'):
        verify_v4(output,project_root=root)


def test_preexisting_output_and_frozen_versions_are_not_modified(v4):
    from src.us_dataset import file_hash
    root,_=v4;reference=root/'data/training/pothole_us_v3'
    before={p:file_hash(p)for p in reference.rglob('*')if p.is_file()}
    audit=build_audit(v4);output=root/'data/training/pothole_us_v4';config=root/'configs/detector_us_v4.yaml'
    stage_v4(audit,output,config)
    with pytest.raises(FileExistsError):stage_v4(audit,output,config)
    assert all(file_hash(p)==digest for p,digest in before.items())


def test_review_app_saves_human_negative_without_editing_annotations(v4):
    from streamlit.testing.v1 import AppTest
    root,_=v4
    queue=root/'results/dataset_audits/v4_preparation/audit/review_queue';queue.mkdir(parents=True)
    row=event_for(v4);row.update(boxes=[],classes=['D20'],split='train',recording_id='')
    write_csv(queue/'us_negative_batch_001.csv',[row])
    source_text=Path('dashboard/dataset_review.py').read_text(encoding='utf-8')
    source_text=source_text.replace('results/dataset_audits/experiment_2_final','results/dataset_audits/v4_preparation/audit')
    source_text=source_text.replace('results/dataset_reviews/experiment_2','results/dataset_reviews/v4')
    source_text=source_text.replace('ROOT=Path(__file__).resolve().parents[1]',f'ROOT=Path({str(root)!r})')
    script=root/'review_ui.py';script.write_text(source_text,encoding='utf-8')
    before=(root/row['annotation']).read_bytes()
    app=AppTest.from_file(str(script)).run(timeout=30)
    assert not app.exception
    next(w for w in app.text_input if w.label=='Your name or reviewer ID').set_value('Synthetic UI reviewer')
    next(w for w in app.selectbox if w.label=='What does the whole image show?').set_value('pothole_free')
    next(w for w in app.checkbox if 'complete image'in w.label).set_value(True)
    next(w for w in app.checkbox if 'inspected the originals'in w.label).set_value(True)
    app.text_area[0].set_value('Synthetic complete-image cavity check')
    next(w for w in app.button if w.label=='Save review decision').click().run()
    assert not app.exception and not app.error
    assert len(list((root/'results/dataset_reviews/v4').glob('*.json')))==1
    assert (root/row['annotation']).read_bytes()==before


def test_final_recording_that_matches_validation_blocks_the_build(v4):
    root,_=v4
    import shutil
    image=root/'data/raw/final/frame.png';image.parent.mkdir(parents=True)
    shutil.copyfile(root/'data/raw/RDD2022/United_States/train/images/US_B.png',image)
    annotation=root/'final_annotation.json'
    annotation.write_text(json.dumps(dict(source_filepath=image.relative_to(root).as_posix(),video_id='final_drive',timestamp_ms=0,
        image_width=32,image_height=24,review_status='reviewed',pothole_boxes=[dict(x=5,y=4,w=10,h=8)])))
    intake=root/'intake.csv';assignments=root/'assignments.csv'
    write_csv(intake,[dict(source='local',image_id='final_frame',filepath=image.relative_to(root).as_posix(),
        annotation=annotation.relative_to(root).as_posix(),recording_id='final_drive',timestamp_ms=0,image_variant='raw',license='collector permission')])
    write_csv(assignments,[dict(recording_id='final_drive',split='test',role='final',approved=True,reviewer='Synthetic',reviewed_at='2026-11-19')])
    with pytest.raises(ValueError,match='Protected final data overlaps'):
        build_audit(v4,intake=intake,assignments=assignments)


def test_related_frame_review_does_not_release_different_recordings(v4):
    root,_=v4;audit=build_audit(v4)
    left=next(r for r in audit.records if r['original_id']=='US_A')
    image=picture(root/'data/raw/other_scene.png',555)
    right=dict(left,original_id='other',original_image=image.relative_to(root).as_posix(),reason='',group='identity:other')
    left['reason']='';audit.records=[left,right];audit.frozen={}
    save_event(root,root/'results/dataset_reviews/v4',dict(kind='duplicate',original_image=left['original_image'],right_image=right['original_image'],
        decision='distinct_frames_same_training_group',inspection_confirmed=True,reviewer='Synthetic',notes='Distinct fixture frames'))
    audit.events=load_events(root,root/'results/dataset_reviews/v4')
    with patch('src.us_dataset.image_fingerprints',side_effect=[('x','0000000000000000'),('y','0000000000000001')]):
        audit.deduplicate()
    assert left['reason']==right['reason']=='near_duplicate_requires_review'


def test_review_input_hashes_reject_changed_annotation(v4):
    root,_=v4;directory=root/'results/dataset_reviews/v4'
    save_event(root,directory,event_for(v4))
    voc(root/'data/raw/RDD2022/United_States/train/annotations/xmls/US_N.xml','D20',(6,5,15,13))
    audit=build_audit(v4,negative_ratio=1)
    assert not any(r['original_id']=='US_N'for r in audit.selected)
    assert audit.review_warnings[0]['finding']=='stale_review_inputs'


def test_duplicate_review_ui_shows_annotations_and_requires_explicit_decision(v4):
    from streamlit.testing.v1 import AppTest
    root,_=v4;audit=build_audit(v4)
    rows=[r for r in audit.records if r['original_id']in {'US_A','US_B'}]
    report=root/'results/dataset_audits/v4_preparation/audit';queue=report/'review_queue';queue.mkdir(parents=True)
    write_csv(report/'excluded.csv',rows)
    write_csv(queue/'duplicate_pairs.csv',[dict(left=rows[0]['original_image'],right=rows[1]['original_image'],kind='perceptual_candidate',distance=1)])
    source_text=Path('dashboard/dataset_review.py').read_text(encoding='utf-8').replace('ROOT=Path(__file__).resolve().parents[1]',f'ROOT=Path({str(root)!r})')
    source_text=source_text.replace('results/dataset_audits/experiment_2_final','results/dataset_audits/v4_preparation/audit')
    source_text=source_text.replace('results/dataset_reviews/experiment_2','results/dataset_reviews/v4')
    script=root/'review_ui.py';script.write_text(source_text,encoding='utf-8')
    app=AppTest.from_file(str(script)).run(timeout=30)
    next(w for w in app.radio if w.label=='Review task').set_value('Suspected duplicate pairs').run()
    assert not app.exception
    assert len(app.code)==2 # both original annotation files are visible
    assert not (root/'results/dataset_reviews/v4').exists()
    next(w for w in app.text_input if w.label=='Your name or reviewer ID').set_value('Synthetic UI reviewer')
    next(w for w in app.selectbox if w.label=='Scene relationship').set_value('same_or_related_scene')
    app.text_area[0].set_value('Synthetic review: keep quarantined')
    next(w for w in app.checkbox if 'inspected the originals'in w.label).set_value(True)
    next(w for w in app.button if w.label=='Save review decision').click().run()
    assert not app.exception
    events=load_events(root,root/'results/dataset_reviews/v4')
    assert next(iter(events.values()))['decision']=='same_or_related_scene'
