"""Synthetic annotation data: no real dataset is modified by these checks."""
import csv
import json
from pathlib import Path

import pytest
from PIL import Image
from src.annotation_editor import apply_editor_boxes, editor_boxes, update_inventory
from src import annotations as ann_io


def test_edit_preserves_physical_ids_and_other_metadata():
    doc={'pothole_boxes':[{'x':10,'y':20,'w':30,'h':40,'gt_event_id':'physical-7','note':'keep'}],
         'negative_regions':[], 'video_id':'source','timestamp_ms':1000}
    boxes=editor_boxes(doc)
    boxes[0]['x']=15
    result=apply_editor_boxes(doc,boxes,100,100)
    assert result['pothole_boxes'][0]['gt_event_id']=='physical-7'
    assert result['pothole_boxes'][0]['note']=='keep'
    assert result['video_id']=='source'
    assert result['timestamp_ms']==1000


def test_reclassification_and_deletion():
    doc={'pothole_boxes':[], 'negative_regions':[]}
    result=apply_editor_boxes(doc,[dict(id='a',category='shadow',x=1,y=2,w=3,h=4)],100,100)
    assert result['negative_regions'][0]['subtype']=='shadow'
    assert not result['pothole_boxes']
    assert apply_editor_boxes(result,[],100,100)['negative_regions']==[]


@pytest.mark.parametrize('edit',[{'x':-1},{'w':0},{'h':float('nan')},{'category':'unknown'},{'x':99}])
def test_invalid_payload_rejected_without_mutation(edit):
    doc={'pothole_boxes':[], 'negative_regions':[]}
    box=dict(id='a',category='pothole',x=1,y=2,w=3,h=4);box.update(edit)
    with pytest.raises(ValueError):apply_editor_boxes(doc,[box],100,100)
    assert doc['pothole_boxes']==[]


def test_atomic_inventory_preserves_other_rows(tmp_path):
    p=tmp_path/'inventory.csv'
    p.write_text('image_id,review_status,notes\na,unreviewed,keep\nb,reviewed,other\n')
    update_inventory(p,'a',{'review_status':'reviewed','extra':'ignored'})
    rows=list(csv.DictReader(p.open()))
    assert rows[0]==dict(image_id='a',review_status='reviewed',notes='keep')
    assert rows[1]['notes']=='other'
    assert not list(tmp_path.glob('*.part'))


def test_annotation_workspace_review_and_navigation(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(ann_io,'ANNOTATIONS_ROOT',tmp_path/'annotations')
    image=tmp_path/'one.jpg';Image.new('RGB',(640,480),'gray').save(image)
    inv=tmp_path/'inventory.csv'
    inv.write_text('image_id,filepath,original_filename,source,review_status,pothole_count\na,'+str(image)+',one.jpg,pilot,unreviewed,0\nb,'+str(image)+',two.jpg,pilot,unreviewed,0\n')
    source=Path('dashboard/annotate.py').read_text()
    source=source.replace("INVENTORY_PATH = PROJECT_ROOT / 'data/metadata/image_inventory.csv'",f'INVENTORY_PATH = Path({str(inv)!r})')
    component=str(Path('dashboard/annotation_canvas').resolve())
    source=source.replace("str(Path(__file__).parent / 'annotation_canvas')",repr(component))
    app=AppTest.from_string(source).run(timeout=30)
    assert not app.exception
    assert len(app.button)==4  # One navigation pair and one review action pair
    next(w for w in app.selectbox if w.label=='Review decision').set_value('reviewed')
    next(w for w in app.text_input if w.label=='Notes').set_value('Synthetic review note')
    app.button[3].click().run()
    assert not app.exception
    assert app.session_state['annotation_image']=='b'
    saved=json.loads(ann_io.annotation_path('a').read_text())
    assert saved['review_status']=='reviewed'
    assert saved['notes']=='Synthetic review note'
    app.button[0].click().run()
    assert app.session_state['annotation_image']=='a'
    assert not app.exception
