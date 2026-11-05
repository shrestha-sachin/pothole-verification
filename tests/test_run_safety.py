"""Synthetic regression fixtures: no real footage, checkpoints or training required."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from src.database import Database
from src.evaluation.matcher import load_video_ground_truth, parse_ground_truth_csv
from src.tracking.event_grouper import CandidateEvent


def event(eid, video='video'):
    return CandidateEvent(eid, video, 'segment', 1, 1000, 1200, 0, 1,
                          .9, .8, 2, 10, 10, 60, 60)


@pytest.fixture
def db(tmp_path):
    db = Database(tmp_path / 'test.db')
    db.register_video('video', {'filepath': 'synthetic.mp4', 'duration_sec': 5})
    db.save_events([event('original')])
    db.save_review('original', 'confirmed', notes='Keep this decision')
    db.mark_processing_done('video', 25)
    return db


def test_publication_failure_rolls_back_everything(db):
    with patch.object(db, 'save_events', side_effect=RuntimeError('synthetic write failure')):
        with pytest.raises(RuntimeError):
            db.replace_completed_run('video', {'duration_sec': 10}, [event('replacement')], {'run_id': 'new'}, 50)
    assert db.get_event('original')['review']['reviewer_notes'] == 'Keep this decision'
    assert db.get_video('video')['duration_sec'] == 5
    assert db.get_video('video')['processing_status'] == 'done'
    assert json.loads(db.export_run_archive()) == []


def test_success_archives_old_run_and_transfers_once(db):
    result = db.replace_completed_run('video', {'duration_sec': 5},
              [event('new1'), event('new2')], {'run_id': 'new'}, 25)
    assert result == {'transferred': 1, 'unmatched': 0}
    assert sum(e['review'] is not None for e in db.get_events()) == 1
    archive = json.loads(db.export_run_archive())
    assert json.loads(archive[0]['snapshot'])['events'][0]['review']['decision'] == 'confirmed'
    assert json.loads(db.get_video('video')['run_config'])['run_id'] == 'new'


def test_non_rejection_has_no_rejection_category(db):
    db.save_review('original', 'confirmed', 'shadow')
    assert db.get_event('original')['review']['fp_category'] is None


def test_wrong_video_never_loaded(tmp_path):
    p = tmp_path / 'data/annotations/video'; p.mkdir(parents=True)
    (p/'green_bay_wrong.json').write_text(json.dumps({'video_id':'other', 'review_status':'reviewed'}))
    assert load_video_ground_truth('video', tmp_path) == []


def test_local_ids_require_verification(tmp_path):
    p = tmp_path / 'data/annotations/video'; p.mkdir(parents=True)
    data = {'video_id':'video', 'review_status':'reviewed', 'timestamp_ms':1000,
            'pothole_boxes':[{'x':10,'y':10,'w':50,'h':50,'gt_event_id':'physical_1'}]}
    f=p/'frame.json'; f.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='human verified'):
        load_video_ground_truth('video', tmp_path)
    data['ground_truth_verified']=True; f.write_text(json.dumps(data))
    assert load_video_ground_truth('video', tmp_path)[0].gt_event_id == 'physical_1'


@pytest.mark.parametrize('row', ['video,1000,0,0,10,10,', 'other,1000,0,0,10,10,A', 'video,,0,0,10,10,A'])
def test_bad_csv_is_actionable(row):
    with pytest.raises(ValueError, match='CSV row 2'):
        parse_ground_truth_csv('video_id,timestamp_ms,x1,y1,x2,y2,gt_event_id\n'+row, 'video')


def test_same_physical_id_preserved_across_frames():
    a=parse_ground_truth_csv('timestamp_ms,x1,y1,x2,y2,gt_event_id\n1000,0,0,10,10,A\n1200,0,0,10,10,A')
    assert len({g.gt_event_id for g in a}) == 1


def test_dashboard_navigation_and_clear(tmp_path):
    from streamlit.testing.v1 import AppTest
    source=Path('dashboard/app.py').read_text()
    source=source.replace('DB_PATH = PROJECT_ROOT / "results" / "pothole_verification.db"',
                          f'DB_PATH = Path({str(tmp_path / "ui.db")!r})')
    app=AppTest.from_string(source).run(timeout=30)
    assert not app.exception
    app.button[0].click().run()
    assert app.session_state['nav_page'] == 'Process Video'
    database=app.session_state['db']
    database.register_video('video', {'duration_sec': 5})
    database.save_events([event('one'),event('two')])
    app.radio(key='nav_page').set_value('Review Events').run()
    assert not app.exception
    app.button(key='btn_next_one').click().run()
    assert app.session_state['selected_event_id'] == 'two'
    app.selectbox(key='widget_dec_two').set_value('confirmed').run()
    app.button(key='btn_save_two').click().run()
    assert database.get_event('two')['review']['decision'] == 'confirmed'
    app.button(key='btn_clear_two').click().run()
    assert app.selectbox(key='widget_dec_two').value == 'unreviewed'
    for page in ('Overview','Evaluation','Export'):
        app.radio(key='nav_page').set_value(page).run()
        assert not app.exception

@pytest.mark.parametrize('stage', ['detector', 'processing', 'verifier', 'persistence'])
def test_pipeline_failures_keep_previous_run(db, tmp_path, monkeypatch, stage):
    """Execute the actual pipeline functions with explicitly synthetic model services."""
    import ast, sys, time, uuid, shutil, hashlib, logging
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from src.evaluation.policies import PolicyConfig, apply_policies
    module = ast.parse(Path('dashboard/app.py').read_text())
    functions = [n for n in module.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('_execute_video_pipeline', '_run_video_pipeline')]
    detector = MagicMock()
    if stage == 'detector':
        detector.side_effect = RuntimeError('synthetic detector failure')
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=detector))
    grouper = MagicMock()
    grouper.process_video_stream.return_value = []
    if stage == 'processing':
        grouper.process_video_stream.side_effect = RuntimeError('synthetic processing failure')
    verifier_loader = MagicMock()
    verifier_loader.return_value.is_configured.return_value = False
    if stage == 'verifier':
        verifier_loader.side_effect = RuntimeError('synthetic verifier failure')
    if stage == 'persistence':
        monkeypatch.setattr(db, 'save_events', MagicMock(side_effect=RuntimeError('synthetic persistence failure')))
    ui = MagicMock()
    context = dict(Path=Path, Optional=__import__('typing').Optional, Database=Database,
        EVIDENCE_ROOT=tmp_path/'evidence', uuid=uuid, shutil=shutil, hashlib=hashlib,
        json=json, time=time, st=ui, logger=logging.getLogger(__name__),
        get_video_metadata=lambda _: {'duration_sec':5}, auto_device=lambda:'cpu',
        EventGrouper=MagicMock(return_value=grouper), iter_frames=lambda *a,**k: iter([]),
        load_verifier=verifier_loader, PolicyConfig=PolicyConfig, apply_policies=apply_policies)
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'dashboard/app.py', 'exec'), context)
    checkpoint=tmp_path/'dummy.pt'; checkpoint.write_bytes(b'synthetic')
    context['_execute_video_pipeline'](db=db, video_path=tmp_path/'dummy.mp4',video_id='video',
        model_path=checkpoint,verifier_path=None,sample_fps=5,conf_threshold=.35,
        iou_threshold=.45,imgsz=640,road_segment_id='segment',policy_b_min_obs=2,
        policy_b_window=5,verifier_threshold=.5,preserve_reviews=True,generate_clips=False)
    assert db.get_event('original')['review']['decision'] == 'confirmed'
    assert db.get_video('video')['processing_status'] == 'done'
    assert ui.error.called
