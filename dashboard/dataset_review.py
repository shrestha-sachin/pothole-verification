"""Human dataset triage. Writes new review events only, never source labels."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import streamlit as st
from PIL import Image, ImageDraw
from src.dataset_reviews import save_event, pair_key
from src.us_dataset import CHARACTERISTICS, read_csv, resolve_source

st.set_page_config(page_title='RoadLens | Dataset review',layout='wide')


def show_original(row):
    image=resolve_source(ROOT,row['original_image'])
    with Image.open(image) as im:
        overlay=im.convert('RGB')
    draw=ImageDraw.Draw(overlay)
    for cx,cy,w,h in json.loads(row.get('boxes') or '[]'):
        draw.rectangle(((cx-w/2)*overlay.width,(cy-h/2)*overlay.height,(cx+w/2)*overlay.width,(cy+h/2)*overlay.height),
                       outline='red',width=max(2,overlay.width//350))
    st.image(overlay,caption='Original-resolution source with existing pothole boxes overlaid',width='stretch')
    st.caption(f"{row['original_image']} | source class names: {row.get('classes','')} | assigned split: {row.get('split') or 'not assigned'}")
    if row.get('published_crack_profile'):
        st.caption(f"Publisher labels indicate {row['published_crack_profile'].lower()}. Check the full road surface for cavities.")
    if row.get('group_provenance') == 'publisher_train_only_sample_fallback':
        st.info('This RDD image needs verified capture-group evidence before it can enter training.')
    if row.get('visual_review_note'):
        st.info(row['visual_review_note'])
    if row.get('annotation'):
        with st.expander('Read the original annotation'):
            path=resolve_source(ROOT,row['annotation'])
            st.code(path.read_text(encoding='utf-8'),language={'.json':'json','.xml':'xml'}.get(path.suffix,'text'))


@st.cache_data(show_spinner=False, max_entries=24)
def cached_csv(path, revision):
    return read_csv(path)


def table(path):
    return cached_csv(str(path), path.stat().st_mtime_ns) if path.is_file() else []


def main():
    notice=st.session_state.pop('dataset_review_notice',None)
    pending=st.session_state.pop('pending_dataset_item',None)
    if pending is not None:st.session_state['dataset_item']=pending
    st.title('Dataset review')
    if notice:st.success(notice)
    st.caption('Inspect full images and existing annotations. Decisions are recorded separately; original images, labels and datasets are kept unchanged.')
    default_report = 'results/dataset_audits/experiment_2_pretraining'
    if not (ROOT/default_report/'review_queue').is_dir():
        default_report = 'results/dataset_audits/experiment_2_final'
    with st.sidebar:
        report_text=st.text_input('Audit directory',value=default_report)
        review_text=st.text_input('Review event directory',value='results/dataset_reviews/experiment_2')
        reviewer=st.text_input('Your name or reviewer ID')
        mode=st.radio('Review task',['RDD negatives: first batch','RDD negatives: all','Supplemental source suitability',
                                    'RDD positive label check','Suspected duplicate pairs'])
    try:
        report=resolve_source(ROOT,report_text)
        directory=resolve_source(ROOT,review_text)
        files={'RDD negatives: first batch':'rdd_negative_batch_001.csv','RDD negatives: all':'rdd_negative_candidates.csv',
               'Supplemental source suitability':'supplemental_sources.csv','RDD positive label check':'rdd_positive_checks.csv',
               'Suspected duplicate pairs':'duplicate_pairs.csv'}
        queue=table(report/'review_queue'/files[mode])
        if not queue and mode.startswith('RDD negatives'):
            # Prior audits remain reviewable even when opened from the archive.
            queue=table(report/'review_queue'/files[mode].replace('rdd_', 'us_'))
        if queue and mode.startswith('RDD negatives'):
            countries = sorted({row['source'] for row in queue})
            if len(countries) > 1:
                country = st.selectbox('RDD country', ['All', *countries], format_func=lambda value: value.removeprefix('rdd_'))
                if country != 'All':
                    queue = [row for row in queue if row['source'] == country]
        if mode=='Supplemental source suitability' and st.checkbox('Show pothole-positive candidates only',value=True):
            queue=[r for r in queue if json.loads(r.get('boxes') or '[]')]
        if not queue:
            st.info('This queue is empty. Run prepare_experiment_2.py --dry-run to generate a new audit.');return
        by_id={(pair_key(r['left'],r['right']) if 'left'in r else r['source']+':'+r['original_id']):r for r in queue}
        st.caption(f'{len(by_id):,} items in this queue.')
        if st.session_state.get('dataset_item') not in by_id:
            st.session_state['dataset_item']=next(iter(by_id))
        item=st.selectbox('Choose an item',list(by_id),key='dataset_item')
        row=by_id[item]
        if mode=='Suspected duplicate pairs':
            # Scene decisions need the original labels too: related views can
            # have inconsistent cavity annotations or different occlusions.
            evidence={r['original_image']:r for r in table(report/'excluded.csv')+table(report/'selection_plan.csv')}
            for col,key in zip(st.columns(2),('left','right')):
                with col:
                    original=evidence.get(row[key])
                    if original and original.get('annotation'):
                        show_original(original)
                    else:
                        path=resolve_source(ROOT,row[key]);st.image(str(path),caption=row[key],width='stretch')
            st.info(f"Difference-hash distance: {row['distance']}. Equal or close hashes do not establish a duplicate. Inspect road geometry, buildings, markings and viewpoints.")
        else:
            if row['source'] == 'roboflow':
                triage = table(ROOT/'docs/reviews/roboflow_v2_label_triage.csv')
                note = next((item for item in triage if item['original_id'] == row['original_id'] and
                             all(item[field] == row.get(field) for field in ('file_sha256', 'annotation_sha256'))), None)
                if note:
                    st.info(f"Preliminary observation for image {note['review_number']}. {note['finding']}")
            show_original(row)
        with st.form('decision_'+item):
            if mode=='Suspected duplicate pairs':
                decision=st.selectbox('Scene relationship',[None,'different_scenes','distinct_frames_same_training_group','same_or_related_scene','unsure'],
                    format_func=lambda x:{None:'Choose a decision','different_scenes':'Different scenes; hash match is misleading',
                    'distinct_frames_same_training_group':'Distinct frames worth keeping within the same training recording',
                    'same_or_related_scene':'Same or related scene; keep quarantined','unsure':'Unsure; keep quarantined'}[x])
                event=dict(kind='duplicate',original_image=row['left'],right_image=row['right'],decision=decision,
                           file_sha256=row.get('left_file_sha256'),right_sha256=row.get('right_file_sha256'))
            elif mode.startswith('RDD negatives'):
                decision=st.selectbox('What does the whole image show?',[None,'pothole_free','pothole_present','unsure'],
                    format_func=lambda x:{None:'Choose a decision','pothole_free':'No visible pothole or open pavement cavity',
                    'pothole_present':'Pothole present; annotation needs correction','unsure':'Unsure; keep excluded'}[x])
                complete=st.checkbox('I checked the complete image for visible cavities, including unboxed areas')
                subtypes=st.multiselect('Observed non-pothole conditions', ['crack','shadow','repaired_patch','manhole','ordinary_road','other'])
                event=dict(kind='negative',source=row['source'],original_id=row['original_id'],original_image=row['original_image'],
                           annotation=row['annotation'],decision=decision,target_annotation_complete=complete,negative_subtypes=subtypes)
            else:
                decision=st.selectbox('Source suitability decision',[None,'approved','not_approved','unsure'])
                st.caption('Approve this image only; this does not approve the country or the dataset. Road cracks, intact repairs, shadows and manholes are not potholes.')
                fields={key:st.text_input(key.replace('_',' ').capitalize()) for key in CHARACTERISTICS}
                definition=st.checkbox('I checked each label against the visible open-cavity pothole definition')
                complete=st.checkbox('If no pothole boxes exist: I checked the whole image and confirmed it is pothole-free')
                event=dict(kind='source',source=row['source'],original_id=row['original_id'],original_image=row['original_image'],
                           annotation=row['annotation'],decision=decision,annotation_definition_checked=definition,
                           target_annotation_complete=complete,**fields)
            if mode!='Suspected duplicate pairs':
                st.caption('For an image with no known split/recording, supply a verified recording/group ID and choose Training. Do not guess capture IDs from filenames. Unknown grouping keeps it excluded.')
                recording=st.text_input('Verified recording/group ID (when required)',value=row.get('recording_id',''))
                split=st.selectbox('Reviewed assignment',['','train','val','test'],index=['','train','val','test'].index(row.get('split','')))
                event.update(recording_id=recording.strip(),split=split)
                event.update({key:row[key] for key in ('file_sha256','annotation_sha256','source_metadata_sha256') if row.get(key)})
            notes=st.text_area('Evidence notes',placeholder='Describe the cavity check, scene differences, annotation concerns, and capture metadata evidence.')
            inspected=st.checkbox('I inspected the originals and annotation evidence, rather than relying on a contact sheet or a model prediction')
            submitted=st.form_submit_button('Save review decision',type='primary')
        if submitted:
            if decision is None:st.error('Choose a decision before saving.');return
            event.update(reviewer=reviewer.strip(),notes=notes.strip(),inspection_confirmed=inspected)
            if event['kind']=='source' and decision=='approved' and row['source']=='roboflow' and (
                    not event['recording_id'] or event['split']!='train'):
                raise ValueError('Roboflow approval needs a verified capture-group ID and a training assignment')
            if event['kind']=='source' and decision=='approved' and not json.loads(row.get('boxes') or '[]') and not complete:
                raise ValueError('Empty-label source approval also needs an explicit whole-image negative check')
            path=save_event(ROOT,directory,event)
            if event['kind']=='source' and decision=='approved' and not json.loads(row.get('boxes') or '[]'):
                save_event(ROOT,directory,{**event,'kind':'negative','decision':'pothole_free'})
            st.session_state['dataset_review_notice']=f"Decision recorded in {path.name}. Run the preparation audit again to check eligibility."
            ids=list(by_id);index=ids.index(item)
            if index+1<len(ids):
                st.session_state['pending_dataset_item']=ids[index+1]
            st.rerun()
    except (ValueError,OSError,KeyError,TypeError) as exc:
        st.error(f'Review could not be completed: {exc}')


if __name__=='__main__':main()
