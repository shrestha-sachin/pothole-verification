"""RoadLens annotation workspace. Run: python -m streamlit run dashboard/annotate.py"""
from __future__ import annotations
import base64
import csv
import io
import sys
import os
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src import annotations as ann_io
from src.annotation_editor import editor_boxes, apply_editor_boxes, update_inventory

INVENTORY_PATH = PROJECT_ROOT / 'data/metadata/image_inventory.csv'
FRAME_ROOT = PROJECT_ROOT / 'data/processed/video_frames'
canvas_component = components.declare_component('roadlens_annotation_editor', path=str(Path(__file__).parent / 'annotation_canvas'))

st.set_page_config(page_title='RoadLens · Annotation', page_icon=':material/crop_free:', layout='wide')
st.markdown('''<style>
.block-container{max-width:1600px;padding-top:2.5rem;padding-bottom:2rem}
.annotation-heading{font-size:.75rem;font-weight:600;letter-spacing:.09em;opacity:.65;margin-bottom:.4rem}
/* Leave Streamlit header and button hit areas intact. Canvas styles are iframe-scoped. */
</style>''', unsafe_allow_html=True)


def inventory():
    if not INVENTORY_PATH.exists():
        return []
    with INVENTORY_PATH.open(newline='', encoding='utf-8') as file:
        return [row for row in csv.DictReader(file) if row.get('image_id')]


@st.cache_data(show_spinner=False)
def preview(path, modified):
    with Image.open(path) as source:
        image = source.convert('RGB')
        width, height = image.size
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        out = io.BytesIO(); image.save(out, format='JPEG', quality=92)
    return 'data:image/jpeg;base64,'+base64.b64encode(out.getvalue()).decode(), width, height


@st.cache_data(ttl=30, show_spinner=False)
def frame_index():
    result = {}
    for file in FRAME_ROOT.rglob('frames.csv'):
        with file.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                result[row.get('filepath', '')] = row
    return result


def persist(annotation):
    path = ann_io.save(annotation)
    updates = {key: str(annotation.get(key, '')).lower() if isinstance(annotation.get(key), bool)
               else annotation.get(key, '') for key in
               ('review_status','shadows_present','patches_present','manholes_present','uncertain','notes')}
    updates['pothole_count'] = str(len(annotation['pothole_boxes']))
    update_inventory(INVENTORY_PATH, annotation['image_id'], updates)
    return path


def flush_pending_edit(image_id):
    """Drain a completed component edit before a full-page navigation rerun."""
    payload=(st.session_state.get('editor_'+image_id)
             or st.session_state.get('annotation_pending_payload_'+image_id))
    revkey='annotation_rev_'+image_id
    key='annotation_doc_'+image_id
    if not payload or key not in st.session_state or payload.get('image_id')!=image_id:
        return
    if payload.get('revision',0)<=st.session_state.get(revkey,0):
        return
    st.session_state['annotation_pending_payload_'+image_id]=payload
    doc=st.session_state[key]
    updated=apply_editor_boxes(doc,payload['boxes'],doc['image_width'],doc['image_height'])
    updated['ground_truth_verified']=False
    persist(updated)
    st.session_state[key]=updated
    st.session_state['annotation_boxes_'+image_id]=payload['boxes']
    st.session_state[revkey]=payload['revision']
    st.session_state.pop('annotation_error_'+image_id,None)
    st.session_state.pop('annotation_pending_payload_'+image_id,None)

# Full reruns include the latest completed edit, even if navigation immediately
# follows pointer release. Never discard it by switching the selected widget first.
for doc_key in list(st.session_state):
    if doc_key.startswith('annotation_doc_'):
        try:
            flush_pending_edit(doc_key.removeprefix('annotation_doc_'))
        except (ValueError,OSError,KeyError,TypeError) as exc:
            st.error(f'Pending annotation could not be saved: {exc}. Resolve the error before navigating.')
            st.stop()


rows = inventory()
if not rows:
    st.title('Annotation workspace')
    st.info('No image inventory found. Run python scripts/build_inventory.py, then reopen this page.', icon=':material/photo_library:')
    st.stop()

with st.sidebar:
    st.markdown('### RoadLens')
    st.caption('ANNOTATION WORKSPACE')
    st.divider()
    source = st.selectbox('Source', ['All sources']+sorted({r.get('source','unknown') for r in rows}))
    status = st.selectbox('Review status', ['All statuses','unreviewed','reviewed','uncertain'])
    query = st.text_input('Find image or sequence', placeholder='Filename, video or image ID').strip().lower()
    filtered = [r for r in rows if (source=='All sources' or r.get('source')==source)
                and (status=='All statuses' or r.get('review_status','unreviewed')==status)
                and (not query or query in ' '.join(str(r.get(k,'')) for k in
                     ('image_id','original_filename','sequence_id','recording_id','video_id')).lower())]
    st.caption(f'{len(filtered)} matching images')
    if not filtered:
        st.info('No images match. Adjust the filters.'); st.stop()
    ids = [r['image_id'] for r in filtered]
    pending = st.session_state.pop('annotation_pending_image', None)
    if pending in ids:
        st.session_state['annotation_image'] = pending
    if st.session_state.get('annotation_image') not in ids:
        st.session_state['annotation_image'] = ids[0]
    chosen = st.selectbox('Image', ids, key='annotation_image',
        format_func=lambda value: next((r.get('original_filename') or value) for r in filtered if r['image_id']==value))
    reviewed = sum(r.get('review_status')=='reviewed' for r in rows)
    st.progress(reviewed/len(rows), text=f'{reviewed} of {len(rows)} reviewed')
    st.caption('Box edits save automatically. Mark the image reviewed only after checking all visible road areas.')
    with st.expander('Workflow guide', icon=':material/help:'):
        st.markdown('1. **Draw** a tight box around each pothole.\n2. Change category for shadows, patches or covers.\n3. Use **Select** to move or resize a box.\n4. Complete the review below the image.')
        st.caption('Navigation preserves saved boxes. Review-form changes require Save review. No batch-generated labels are added.')

row = next(r for r in filtered if r['image_id']==chosen)
idx = ids.index(chosen)
key = 'annotation_doc_'+chosen
if key not in st.session_state:
    doc = ann_io.load(chosen)
    doc['source_filepath'] = row['filepath']
    st.session_state[key] = doc
    st.session_state['annotation_boxes_'+chosen] = editor_boxes(doc)
    st.session_state['annotation_rev_'+chosen] = 0
annotation = st.session_state[key]

st.markdown('<div class="annotation-heading">DATASET / IMAGE ANNOTATION</div>', unsafe_allow_html=True)
heading, nav = st.columns([5,2], vertical_alignment='center')
with heading:
    st.subheader(row.get('original_filename') or chosen, anchor=False)
    st.caption(f"Image {idx+1} of {len(ids)} · {annotation.get('review_status','unreviewed').title()} · {row.get('source','Local footage')}")
with nav:
    left,right=st.columns(2)
    if left.button('Previous', icon=':material/chevron_left:', disabled=idx==0, width='stretch'):
        st.session_state['annotation_pending_image']=ids[idx-1]; st.rerun()
    if right.button('Next', icon=':material/chevron_right:', disabled=idx==len(ids)-1, width='stretch'):
        st.session_state['annotation_pending_image']=ids[idx+1]; st.rerun()

path = PROJECT_ROOT / row['filepath']
try:
    frame_modified = path.stat().st_mtime_ns
    image_data,width,height = preview(str(path),frame_modified)
except (OSError,ValueError) as exc:
    st.error(f'Cannot open source image: {exc}', icon=':material/broken_image:'); st.stop()
annotation.update(image_width=width,image_height=height)


@st.fragment
def editor():
    revkey='annotation_rev_'+chosen
    boxeskey='annotation_boxes_'+chosen
    error_key='annotation_error_'+chosen
    payload=canvas_component(image_id=chosen,image=image_data,width=width,height=height,
        boxes=st.session_state[boxeskey],revision=st.session_state[revkey],
        save_error=st.session_state.get(error_key,''),key='editor_'+chosen,default=None)
    if (payload and payload.get('image_id')==chosen
            and payload.get('revision',0)>st.session_state[revkey]
            and payload.get('revision')!=st.session_state.get('annotation_failed_rev_'+chosen)):
        try:
            updated=apply_editor_boxes(st.session_state[key],payload['boxes'],width,height)
            # A geometric edit invalidates previously confirmed evaluation IDs/coverage.
            updated['ground_truth_verified']=False
            persist(updated)
            st.session_state[key]=updated
            st.session_state[boxeskey]=payload['boxes']
            st.session_state[revkey]=payload['revision']
            st.session_state.pop(error_key,None)
            st.session_state.pop('annotation_pending_payload_'+chosen,None)
            st.rerun(scope='fragment')
        except (ValueError,OSError,KeyError,TypeError) as exc:
            st.session_state['annotation_pending_payload_'+chosen]=payload
            st.session_state[error_key]=str(exc)
            st.session_state['annotation_failed_rev_'+chosen]=payload.get('revision')
            st.rerun(scope='fragment')
    if st.session_state.get(error_key):
        st.error('Box edits are not saved. Resolve the storage error, then use Retry save in the editor.')


def save_redacted_frame(path, boxes, width, height, expected_mtime):
    """Atomically replace the extracted frame using its original image format."""
    if path.stat().st_mtime_ns != expected_mtime:
        raise ValueError("The frame changed. Reopen it before saving.")
    with Image.open(path) as source:
        image_format = source.format
        frame = source.convert('RGB')
    frame.info.clear()
    if frame.size != (width, height):
        raise ValueError('The source dimensions changed. Reopen the frame before saving.')
    masks = apply_editor_boxes({}, boxes, width, height)['pothole_boxes']
    draw = ImageDraw.Draw(frame)
    for box in masks:
        x, y, w, h = (box[field] for field in ('x', 'y', 'w', 'h'))
        draw.rectangle((x, y, x + w - 1, y + h - 1), fill='black')
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            options = {'quality': 95, 'subsampling': 0} if image_format == 'JPEG' else {}
            frame.save(temporary, format=image_format, **options)
        if path.stat().st_mtime_ns != expected_mtime:
            raise ValueError('The frame changed during saving. Reopen it before saving again.')
        os.chmod(temporary_path, path.stat().st_mode & 0o777)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)



@st.fragment
def plate_redaction():
    st.caption(
        'Draw a box over each readable plate. Saving permanently applies the masks '
        'to this image file at its original size.'
    )
    mask_key = 'plate_masks_' + chosen
    draft = st.session_state.get(mask_key, {'revision': 0, 'boxes': []})
    frame_id = f'plate:{chosen}:{frame_modified}'
    if draft.get('image_id') != frame_id:
        draft = {'revision': 0, 'boxes': []}
    payload = canvas_component(
        image_id=frame_id, image=image_data, width=width, height=height,
        boxes=draft['boxes'], revision=draft['revision'], privacy=True,
        key='plate_editor_' + chosen, default=None,
    )
    if payload and payload.get('image_id') == frame_id and payload['revision'] > draft['revision']:
        try:
            apply_editor_boxes({}, payload['boxes'], width, height)
        except ValueError as exc:
            st.error(str(exc))
            return
        draft = payload
        st.session_state[mask_key] = draft
        # The next component render acknowledges the edit without replacing the canvas.
        try:
            st.rerun(scope='fragment')
        except st.errors.StreamlitAPIException:
            st.rerun()

    if st.session_state.pop('plate_saved_' + chosen, False):
        st.success('Redactions saved to the existing image file.')
    if st.button('Save redactions to this frame', disabled=not draft['boxes'],
                 icon=':material/shield:', key='plate_save_' + chosen):
        try:
            save_redacted_frame(path, draft['boxes'], width, height, frame_modified)
        except (OSError, ValueError) as exc:
            st.error(f'Could not save the frame. {exc}')
        else:
            st.session_state.pop(mask_key, None)
            st.session_state['plate_saved_' + chosen] = True
            st.rerun()


if st.toggle('Redact license plates', key='plate_redaction_mode'):
    plate_redaction()
else:
    editor()

st.divider()
st.markdown('#### Complete image review')
st.caption('Drawing saves boxes; it does not mark the image reviewed. Saving a reviewed image with no pothole boxes explicitly records zero visible potholes.')
with st.form('review_'+chosen, border=False):
    a,b=st.columns([1,2])
    with a:
        disposition=st.selectbox('Review decision', ['reviewed','uncertain','unreviewed'],
            index=1 if annotation.get('review_status')=='uncertain' else 0,
            format_func=lambda v:{'reviewed':'Reviewed — labels checked','uncertain':'Uncertain — exclude from training','unreviewed':'Not yet reviewed'}[v])
    with b:
        notes=st.text_input('Notes',value=annotation.get('notes',''),placeholder='Visibility, pavement condition, or labeling uncertainty')
    f1,f2,f3=st.columns(3)
    shadows=f1.checkbox('Shadows present',value=annotation.get('shadows_present',False))
    patches=f2.checkbox('Repaired patches present',value=annotation.get('patches_present',False))
    manholes=f3.checkbox('Manhole covers present',value=annotation.get('manholes_present',False))
    with st.expander('Evaluation metadata',icon=':material/verified:'):
        st.caption('Optional: use the same physical pothole ID across frames of one defect. Set IDs by selecting boxes in the editor. Verify full-video coverage separately on the Evaluation page.')
        meta=frame_index().get(row['filepath'],{})
        video_id=st.text_input('Exact source video ID',value=annotation.get('video_id') or meta.get('video_id',''))
        timestamp=str(annotation.get('timestamp_ms',''))
        if not timestamp and meta.get('timestamp_sec'):
            try: timestamp=str(round(float(meta['timestamp_sec'])*1000))
            except ValueError: pass
        timestamp_input=st.text_input('Source timestamp (milliseconds)',value=timestamp)
        verified=st.checkbox('I verified this frame’s physical pothole IDs and source alignment',value=annotation.get('ground_truth_verified',False))
    save,advance=st.columns(2)
    saved=save.form_submit_button('Save review',icon=':material/check:',width='stretch')
    next_saved=advance.form_submit_button('Save review & next',type='primary',icon=':material/arrow_forward:',width='stretch')
    if saved or next_saved:
        # Read the latest document: the canvas fragment may have updated it independently.
        updated={**st.session_state[key], 'review_status':disposition,'uncertain':disposition=='uncertain',
                 'shadows_present':shadows,'patches_present':patches,'manholes_present':manholes,'notes':notes,
                 'ground_truth_verified':verified}
        try:
            if timestamp_input.strip():
                ts=int(timestamp_input)
                if ts<0: raise ValueError('Timestamp must be nonnegative')
                updated['timestamp_ms']=ts
            updated['video_id']=video_id.strip()
            if verified and (not video_id.strip() or not timestamp_input.strip()
                    or any(not str(b.get('gt_event_id','')).strip() for b in updated['pothole_boxes'])
                    or disposition!='reviewed'):
                raise ValueError('Verified ground truth requires source video, timestamp, a physical ID on every pothole, and a reviewed decision.')
            persist(updated); st.session_state[key]=updated
            st.session_state['annotation_notice']='Review saved.'
            if next_saved and idx<len(ids)-1:
                st.session_state['annotation_pending_image']=ids[idx+1]
            st.rerun()
        except (ValueError,OSError) as exc:
            st.error(f'Could not save review: {exc}',icon=':material/error:')
if st.session_state.pop('annotation_notice',None):
    st.toast('Review saved.',icon=':material/check_circle:')
with st.expander('Annotation file',icon=':material/data_object:'):
    st.caption(str(ann_io.annotation_path(chosen).relative_to(PROJECT_ROOT)))
    import json
    st.download_button('Download annotation JSON',json.dumps(st.session_state[key],indent=2),chosen+'.json','application/json',icon=':material/download:')
