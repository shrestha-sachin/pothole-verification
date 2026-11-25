"""RoadLens annotation workspace. Run: python -m streamlit run dashboard/annotate.py"""
from __future__ import annotations
import base64
import argparse
import csv
import io
import json
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
from src.preprocessing.roi import DEFAULT_SOURCE_ROIS, apply_roi_crop
from src.preprocessing.quality import assess_image_quality

INVENTORY_PATH = PROJECT_ROOT / 'data/metadata/image_inventory.csv'
ANNOTATED_SOURCES = {
    'local_dashcam': PROJECT_ROOT / 'data/annotated/local/dashcam',
    'local_smartphone': PROJECT_ROOT / 'data/annotated/local/smartphone',
    'mapillary': PROJECT_ROOT / 'data/annotated/mapillary',
    'rdd2022_usa': PROJECT_ROOT / 'data/annotated/rdd2022_usa',
}
canvas_component = components.declare_component('roadlens_annotation_editor', path=str(Path(__file__).parent / 'annotation_canvas'))

st.set_page_config(page_title='RoadLens · Annotation', page_icon=':material/crop_free:', layout='wide')
parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
parser.add_argument('--source', default='')
source_arg = parser.parse_known_args()[0].source
annotation_source = None
st.markdown('''<style>
.block-container{max-width:1600px;padding-top:2.5rem;padding-bottom:2rem}
.annotation-heading{font-size:.75rem;font-weight:600;letter-spacing:.09em;opacity:.65;margin-bottom:.4rem}
/* Leave Streamlit header and button hit areas intact. Canvas styles are iframe-scoped. */
</style>''', unsafe_allow_html=True)


def inventory():
    if INVENTORY_PATH.exists():
        with INVENTORY_PATH.open(newline='', encoding='utf-8') as file:
            return [row for row in csv.DictReader(file) if row.get('image_id')]
    combined = []
    for src_name, src_dir in ANNOTATED_SOURCES.items():
        meta_csv = src_dir / 'metadata' / 'annotations.csv'
        if not meta_csv.exists():
            continue
        with meta_csv.open(newline='', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                img_id = row.get('image_id', '').strip()
                if not img_id:
                    continue
                row.setdefault('source', src_name)
                row['source'] = src_name
                row.setdefault('filepath', (src_dir / 'images' / f'{img_id}.jpg').relative_to(PROJECT_ROOT).as_posix())
                row.setdefault('original_filename', f'{img_id}.jpg')
                row['_meta_csv'] = str(meta_csv)
                row['_src_dir'] = str(src_dir)
                combined.append(row)
    return combined


@st.cache_data(show_spinner=False)
def preview(path, modified):
    with Image.open(path) as source:
        image = source.convert('RGB')
        width, height = image.size
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        out = io.BytesIO(); image.save(out, format='JPEG', quality=92)
    return 'data:image/jpeg;base64,'+base64.b64encode(out.getvalue()).decode(), width, height


@st.cache_data(ttl=30, show_spinner=False)
def manifest_index():
    master = {}
    excluded = {}
    assisted = {}
    m_path = PROJECT_ROOT / 'data/manifests/master_manifest.csv'
    e_path = PROJECT_ROOT / 'data/manifests/excluded_images.csv'
    a_path = PROJECT_ROOT / 'data/manifests/assisted_proposals.json'
    if m_path.exists():
        with m_path.open(newline='', encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                master[r.get('source_image_id', '')] = r
    if e_path.exists():
        with e_path.open(newline='', encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                excluded[r.get('image_id', '')] = r
    if a_path.exists():
        try:
            assisted = json.loads(a_path.read_text(encoding='utf-8'))
        except Exception:
            assisted = {}
    return master, excluded, assisted


@st.cache_data(ttl=15, show_spinner=False)
def json_region_index():
    """Index per-image negative_region subtype counts and train splits without loading image pixels."""
    idx = {}
    for src_name, src_dir in ANNOTATED_SOURCES.items():
        meta_dir = src_dir / 'metadata'
        if not meta_dir.exists():
            continue
        for jp in meta_dir.glob('*.json'):
            try:
                doc = json.loads(jp.read_text(encoding='utf-8'))
            except Exception:
                continue
            neg_counts = {}
            for nr in doc.get('negative_regions', []):
                sub = str(nr.get('subtype', 'ordinary_road')).strip().lower()
                neg_counts[sub] = neg_counts.get(sub, 0) + 1
            idx[jp.stem] = {
                'shadow_boxes': neg_counts.get('shadow', 0),
                'patch_boxes': neg_counts.get('repaired_patch', 0),
                'manhole_boxes': neg_counts.get('manhole', 0),
                'shadows_present': bool(doc.get('shadows_present')),
                'negative_subcategory': str(doc.get('negative_subcategory', '')).strip().lower(),
                'has_gt_event_id': any(bool(str(b.get('gt_event_id', '')).strip()) for b in doc.get('pothole_boxes', [])),
            }
    return idx


def persist(annotation):
    neg_subs = {str(nr.get('subtype', '')).strip().lower() for nr in annotation.get('negative_regions', [])}
    if 'shadow' in neg_subs:
        annotation['shadows_present'] = True
    if 'repaired_patch' in neg_subs:
        annotation['patches_present'] = True
    if 'manhole' in neg_subs:
        annotation['manholes_present'] = True
    if INVENTORY_PATH.exists():
        path = ann_io.save(annotation, source=annotation_source)
        updates = {key: str(annotation.get(key, '')).lower() if isinstance(annotation.get(key), bool)
                   else annotation.get(key, '') for key in
                   ('review_status','shadows_present','patches_present','manholes_present','uncertain','notes')}
        updates['pothole_count'] = str(len(annotation['pothole_boxes']))
        update_inventory(INVENTORY_PATH, annotation['image_id'], updates)
        return path
    src_name = annotation.get('_src_name') or 'local_dashcam'
    src_dir = ANNOTATED_SOURCES.get(src_name, PROJECT_ROOT / 'data/annotated/local/dashcam')
    img_id = annotation['image_id']
    meta_json = src_dir / 'metadata' / f'{img_id}.json'
    meta_json.parent.mkdir(parents=True, exist_ok=True)
    meta_json.write_text(json.dumps(annotation, indent=2), encoding='utf-8')
    iw = max(1, int(annotation.get('image_width') or 1))
    ih = max(1, int(annotation.get('image_height') or 1))
    yolo_lines = []
    for b in annotation.get('pothole_boxes', []):
        cx = (float(b['x']) + float(b['w']) / 2.0) / iw
        cy = (float(b['y']) + float(b['h']) / 2.0) / ih
        nw = float(b['w']) / iw
        nh = float(b['h']) / ih
        yolo_lines.append(f'0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')
    if annotation.get('review_status') == 'reviewed':
        lbl_path = src_dir / 'labels' / f'{img_id}.txt'
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        lbl_path.write_text('\n'.join(yolo_lines) + ('\n' if yolo_lines else ''), encoding='utf-8')
    meta_csv = src_dir / 'metadata' / 'annotations.csv'
    if meta_csv.exists():
        updates = {key: str(annotation.get(key, '')).lower() if isinstance(annotation.get(key), bool)
                   else annotation.get(key, '') for key in
                   ('review_status','shadows_present','patches_present','manholes_present','uncertain','notes','negative_subcategory')}
        updates['pothole_count'] = str(len(annotation.get('pothole_boxes', [])))
        update_inventory(meta_csv, img_id, updates)
    json_region_index.clear()
    return meta_json


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
    master_idx, excl_idx, assist_idx = manifest_index()
    region_idx = json_region_index()
    source_options = ['All sources'] + sorted({r.get('source', 'unknown') for r in rows})
    default_src_idx = source_options.index(source_arg) if source_arg in source_options else 0
    source = st.selectbox('Source', source_options, index=default_src_idx)
    status = st.selectbox('Review status', ['All statuses','unreviewed','reviewed','uncertain'])
    elig_filter = st.selectbox('Dataset eligibility', ['All images', 'Selected in active dataset', 'Excluded / Flagged'])
    triage_options = [
        'All frames',
        'Shadow-annotated frames',
        'Train frames needing shadow regions',
    ]
    if assist_idx:
        triage_options += [
            'Unreviewed frames',
            'Potential potholes',
            'High-confidence candidates',
            'Uncertain detections',
            'Potential hard negatives',
            'Images with quality issues',
            'Near-duplicate frames',
        ]
    assist_filter = st.selectbox('Model-assisted triage', triage_options)
    diversify_scenes = st.checkbox('Prioritize diverse road scenes', value=True) if assist_idx else False
    query = st.text_input('Find image or sequence', placeholder='Filename, video or image ID').strip().lower()

    def _is_train_candidate(r):
        sp = str(r.get('split') or '').strip().lower()
        if sp == 'test':
            return False
        m_rec = master_idx.get(r['image_id'], {})
        if m_rec:
            return m_rec.get('split') == 'train'
        if sp:
            return sp == 'train'
        return str(r.get('parent_video', '')) in {'local_dashcam_video_0007', 'local_dashcam_video_0008', 'local_dashcam_video_0009'}

    def _matches_assist(r):
        if assist_filter == 'All frames':
            return True
        reg = region_idx.get(r['image_id'], {})
        sub_csv = str(r.get('negative_subcategory', '')).strip().lower()
        has_shadow_flag = (
            str(r.get('shadows_present', '')).strip().lower() in {'true', '1', 'yes'}
            or reg.get('shadows_present', False)
            or reg.get('shadow_boxes', 0) > 0
            or reg.get('negative_subcategory') == 'shadow'
            or sub_csv == 'shadow'
        )
        if assist_filter == 'Shadow-annotated frames':
            return has_shadow_flag
        if assist_filter == 'Train frames needing shadow regions':
            return _is_train_candidate(r) and reg.get('shadow_boxes', 0) == 0 and r.get('review_status') != 'uncertain'
        if assist_filter == 'Unreviewed frames':
            return r.get('review_status', 'unreviewed') == 'unreviewed'
        ap = assist_idx.get(r['image_id'], {})
        m_rec = master_idx.get(r['image_id'], {})
        e_rec = excl_idx.get(r['image_id'], {})
        bucket = ap.get('assist_bucket', '')
        q_flags = ap.get('quality_flags', []) or ([m_rec['quality_flags']] if m_rec.get('quality_flags') else [])
        is_dup = bool(ap.get('near_duplicate')) or ('duplicate' in e_rec.get('reason', ''))
        if assist_filter == 'Potential potholes':
            return bool(ap.get('boxes'))
        if assist_filter == 'High-confidence candidates':
            return bucket == 'high_conf_candidate'
        if assist_filter == 'Uncertain detections':
            return bucket == 'uncertain_detection'
        if assist_filter == 'Potential hard negatives':
            return bucket == 'potential_hard_negative'
        if assist_filter == 'Images with quality issues':
            return bool(q_flags)
        if assist_filter == 'Near-duplicate frames':
            return is_dup
        return True

    filtered = [r for r in rows if (source=='All sources' or r.get('source')==source)
                and (status=='All statuses' or r.get('review_status','unreviewed')==status)
                and (elig_filter=='All images'
                     or (elig_filter=='Selected in active dataset' and r['image_id'] in master_idx)
                     or (elig_filter=='Excluded / Flagged' and (r['image_id'] in excl_idx or r.get('review_status')=='uncertain')))
                and _matches_assist(r)
                and (not query or query in ' '.join(str(r.get(k,'')) for k in
                     ('image_id','original_filename','sequence_id','parent_video','group_id')).lower())]
    if diversify_scenes and assist_idx:
        # Interleave across videos, placing non-duplicate candidate frames before repeated/clean frames
        from collections import defaultdict as _dd
        by_vid = _dd(list)
        for r in filtered:
            ap = assist_idx.get(r['image_id'], {})
            pri = (1 if ap.get('near_duplicate') else 0, 0 if ap.get('boxes') else 1, -float(ap.get('max_conf', 0.0)))
            by_vid[r.get('parent_video') or r.get('source', '')].append((pri, r))
        for k in by_vid:
            by_vid[k].sort(key=lambda x: x[0])
        diversified = []
        vkeys = sorted(by_vid)
        while any(by_vid.values()):
            for k in vkeys:
                if by_vid[k]:
                    diversified.append(by_vid[k].pop(0)[1])
        filtered = diversified
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
    m_rec = master_idx.get(chosen)
    e_rec = excl_idx.get(chosen)
    if m_rec or e_rec:
        with st.expander('Quality, Split & ROI Status', expanded=True):
            if m_rec:
                st.caption(f"**Split:** `{m_rec.get('split')}` · **Category:** `{m_rec.get('category')}`")
                if m_rec.get('negative_subcategory'):
                    st.caption(f"**Hard-Neg Type:** `{m_rec.get('negative_subcategory')}`")
                st.caption(f"**Sharpness:** `{m_rec.get('laplacian_var')}` · **Brightness:** `{m_rec.get('brightness_mean')}`")
                if m_rec.get('quality_flags'):
                    st.warning(f"Quality flags: {m_rec.get('quality_flags')}")
                st.caption(f"**ROI Crop:** `{m_rec.get('roi_crop_px')}` -> `{m_rec.get('proc_width')}x{m_rec.get('proc_height')}`")
            if e_rec:
                reason = e_rec.get('reason', '')
                st.error(f"**Excluded ({e_rec.get('split')}):** {reason}")
                if 'duplicate' in reason:
                    st.caption("**Duplicate flag:** Flagged by SHA-256 / dHash check")
    st.caption('Box edits save automatically. Mark the image reviewed only after checking all visible road areas.')
    with st.expander('Workflow guide', icon=':material/help:'):
        st.markdown('1. **Draw** a tight box around each pothole.\n2. Change category for shadows, patches or covers.\n3. Use **Select** to move or resize a box.\n4. Complete the review below the image.')
        st.caption('Navigation preserves saved boxes. Review-form changes require Save review. No batch-generated labels are added.')

row = next(r for r in filtered if r['image_id']==chosen)
idx = ids.index(chosen)
key = 'annotation_doc_'+chosen
if key not in st.session_state:
    if INVENTORY_PATH.exists():
        doc = ann_io.load(chosen, source=annotation_source)
    else:
        src_dir = Path(row['_src_dir'])
        meta_json = src_dir / 'metadata' / f'{chosen}.json'
        if meta_json.exists():
            doc = json.loads(meta_json.read_text(encoding='utf-8'))
        else:
            doc = ann_io._empty(chosen)
            doc['review_status'] = row.get('review_status', 'unreviewed')
            lbl_txt = src_dir / 'labels' / f'{chosen}.txt'
            img_p = PROJECT_ROOT / row['filepath']
            if lbl_txt.exists() and img_p.exists():
                with Image.open(img_p) as im:
                    iw, ih = im.size
                for ln in lbl_txt.read_text(encoding='utf-8').splitlines():
                    if not ln.strip():
                        continue
                    _, cx, cy, bw, bh = map(float, ln.split())
                    doc['pothole_boxes'].append({
                        'x': int(round((cx - bw / 2.0) * iw)),
                        'y': int(round((cy - bh / 2.0) * ih)),
                        'w': max(1, int(round(bw * iw))),
                        'h': max(1, int(round(bh * ih))),
                    })
        doc['_src_name'] = row.get('source', 'local_dashcam')
        if row.get('parent_video'):
            doc.setdefault('video_id', row['parent_video'])
        if row.get('timestamp_ms'):
            doc.setdefault('timestamp_ms', int(row['timestamp_ms']))
        ap = assist_idx.get(chosen, {})
        if (not meta_json.exists() and row.get('review_status', 'unreviewed') == 'unreviewed'
                and ap.get('boxes') and not doc['pothole_boxes']):
            doc['pothole_boxes'] = [
                {'x': b['x'], 'y': b['y'], 'w': b['w'], 'h': b['h'], '_editor_id': b['id']}
                for b in ap['boxes']
            ]
            st.session_state['annotation_assist_loaded_' + chosen] = ap
    doc['source_filepath'] = row['filepath']
    st.session_state[key] = doc
    st.session_state['annotation_boxes_'+chosen] = editor_boxes(doc)
    st.session_state['annotation_rev_'+chosen] = 0
annotation = st.session_state[key]
if st.session_state.get('annotation_assist_loaded_' + chosen) and annotation.get('review_status', 'unreviewed') == 'unreviewed':
    ap_info = st.session_state['annotation_assist_loaded_' + chosen]
    st.info(
        f"YOLO26x V5 model-assisted proposals loaded ({len(ap_info.get('boxes', []))} box(es), max conf={ap_info.get('max_conf', 0):.2f}, "
        f"triage={ap_info.get('assist_bucket')}). Accept/adjust valid boxes or delete false positives before clicking Save review.",
        icon=':material/auto_fix_high:',
    )

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


t1, t2 = st.columns(2)
redact_mode = t1.toggle('Redact license plates', key='plate_redaction_mode')
roi_preview_mode = t2.toggle('Preview Road ROI & Transformed Boxes', key='roi_preview_mode')

if redact_mode:
    plate_redaction()
elif roi_preview_mode:
    src_name = row.get('source', 'rdd2022_usa')
    roi = DEFAULT_SOURCE_ROIS.get(src_name, DEFAULT_SOURCE_ROIS['rdd2022_usa'])
    norm_boxes = [
        (0, (float(b['x']) + float(b['w']) / 2.0) / width, (float(b['y']) + float(b['h']) / 2.0) / height,
         float(b['w']) / width, float(b['h']) / height)
        for b in annotation.get('pothole_boxes', [])
    ]
    with Image.open(path) as src_im:
        res_roi = apply_roi_crop(src_im.convert('RGB'), norm_boxes, roi)
    prev_im = res_roi.image.copy()
    cw, ch = prev_im.size
    draw = ImageDraw.Draw(prev_im)
    for _, cx, cy, bw, bh in res_roi.boxes:
        x1, y1, x2, y2 = (cx - bw / 2.0) * cw, (cy - bh / 2.0) * ch, (cx + bw / 2.0) * cw, (cy + bh / 2.0) * ch
        draw.rectangle((x1, y1, x2, y2), outline='lime', width=max(2, cw // 300))
    st.caption(f"ROI preview ({src_name}): {width}x{height} -> {cw}x{ch} | Boxes retained: {len(res_roi.boxes)}/{res_roi.original_box_count}, truncated: {res_roi.truncated_boxes}, dropped: {res_roi.dropped_boxes}")
    st.image(prev_im, width='stretch')
else:
    n_potholes = len(annotation.get('pothole_boxes', []))
    nr_subs = {}
    for nr in annotation.get('negative_regions', []):
        s = str(nr.get('subtype', 'ordinary_road')).strip().lower()
        nr_subs[s] = nr_subs.get(s, 0) + 1
    st.caption(
        f"**Boxes on frame:** Pothole=`{n_potholes}` · Shadow=`{nr_subs.get('shadow', 0)}` · "
        f"Repaired patch=`{nr_subs.get('repaired_patch', 0)}` · Manhole=`{nr_subs.get('manhole', 0)}` · "
        f"Ordinary road=`{nr_subs.get('ordinary_road', 0)}`  "
        f"_(Tip: Choose **Shadow** in the canvas toolbar Box category to box cast shadows for verifier training.)_"
    )
    editor()
    if annotation.get('pothole_boxes'):
        sizes = ', '.join(f"{b['w']}x{b['h']}px" for b in annotation['pothole_boxes'])
        st.caption(f"Annotated box sizes (native pixels): {sizes}")

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
    f1,f2,f3,f4=st.columns(4)
    shadows=f1.checkbox('Shadows present',value=annotation.get('shadows_present',False))
    patches=f2.checkbox('Repaired patches present',value=annotation.get('patches_present',False))
    manholes=f3.checkbox('Manhole covers present',value=annotation.get('manholes_present',False))
    neg_sub_opts = ['', 'shadow', 'repaired_patch', 'manhole_cover', 'road_marking', 'crack', 'ordinary_background', 'other']
    cur_sub = annotation.get('negative_subcategory') or row.get('negative_subcategory', '')
    neg_sub = f4.selectbox('Hard-negative subcategory', neg_sub_opts, index=neg_sub_opts.index(cur_sub) if cur_sub in neg_sub_opts else 0)
    with st.expander('Evaluation metadata',icon=':material/verified:'):
        st.caption('Optional: use the same physical pothole ID across frames of one defect. Set IDs by selecting boxes in the editor. Verify full-video coverage separately on the Evaluation page.')
        meta = {}
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
                 'negative_subcategory':neg_sub,'ground_truth_verified':verified}
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
    st.caption(str(ann_io.annotation_path(chosen, source=annotation_source)))
    import json
    st.download_button('Download annotation JSON',json.dumps(st.session_state[key],indent=2),chosen+'.json','application/json',icon=':material/download:')
