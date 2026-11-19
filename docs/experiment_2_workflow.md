# Pothole detection Experiment 2

Use `data/training/pothole_experiment_2_v2/` with `configs/detector_experiment_2.yaml`.
The build uses Japan and Norway RDD2022, reviewed Green Bay and Mapillary images, and an image-level review gate for the local Roboflow version 2 export. US, India, China and Czech RDD2022 and the Kaggle datasets are outside this experiment.

## Collect and store inputs

| Input | Location |
| --- | --- |
| RDD2022 Japan and Norway images and original XML | `data/raw/RDD2022/Japan/` and `data/raw/RDD2022/Norway/` |
| Green Bay videos | `data/raw/local_video/` |
| Green Bay still images and Mapillary sequences | `data/raw/green_bay_images/` and `data/raw/mapillary/` with source IDs in `data/metadata/image_inventory.csv` |
| Roboflow Pothole Detection v2 YOLO export | `data/raw/RoboFlow Dataset/` with `data.yaml`, the source labels and the export README files |
| Human annotation files | `data/annotations/` |
| Extracted frames and other derived working files | `data/processed/` |
| Prepared dataset, its manifest and audit | `data/training/pothole_experiment_2_v2/` |

The raw folder is the source of record. Processed images can still be essential when manual boxes point to their exact pixels. Keep the raw data, frame manifests, review decisions, and annotation files together when transferring a dataset to another workstation.

RDD2022 identifies D40 as the pothole target and lists CC BY-SA 4.0 for the images. Retain source identifiers, attribute Arya et al., and keep the license with redistributed material. The Japan XML includes D43, D44 and D50. The builder checks the geometry of those boxes and keeps their class names in provenance. Only D40 converts to YOLO class `0`. The published D40 labels receive structural checks and a visual-review queue. An independent person has yet to inspect every published box. [RDD2022 release and attribution](https://github.com/sekilab/RoadDamageDetector/blob/master/README.md#dataset)

The Roboflow export contains **55 training and 18 validation images, zero test images, and 73 matching labels**. All export images are 640×640. The six class names include `pothole` at index 5. The export README records auto-orientation, stretch resizing, and no augmentation. Its `data.yaml` declares CC BY 4.0. Retain the publisher, project, version, source URL and license, and identify changes such as remapped labels when redistributing. Verify upstream contributor attribution before redistribution. The source capture groups and independent original-image count are unavailable in the export. [Roboflow project v2](https://universe.roboflow.com/the-american-university-of-the-middle-east/pothole-detection-3uuq2/dataset/2) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Audit, review and prepare

Run these from the repository root in Windows PowerShell. The audit reads images, XML, YOLO labels and manual annotations and writes reports in `results/`. It leaves a queue for images that need a decision. Use a fresh directory each time you pass `--review-report`.

```powershell
.venv\Scripts\python.exe scripts/prepare_experiment_2.py --dry-run --review-report results/dataset_audits/experiment_2_audit_next
Import-Csv results/dataset_audits/experiment_2_audit_next/annotation_audit.csv | Format-Table source,original_id,finding
Import-Csv results/dataset_audits/experiment_2_audit_next/source_statistics.csv | Format-Table source,training_positives,training_negatives,validation_positives,validation_negatives,excluded
```

Use the existing annotation editor for Green Bay and Mapillary boxes. Use the shared review tool for RDD background and ambiguous cavity decisions, Roboflow approvals, and suspected duplicate pairs. The app opens `results/dataset_audits/experiment_2_pretraining` when available. Select a new audit in the sidebar after refreshing the queues. Each decision records the reviewer, the inspected source content, and the evidence. Keep `results/dataset_reviews/experiment_2/` with the data archive.

```powershell
.venv\Scripts\python.exe -m streamlit run dashboard/annotate.py
.venv\Scripts\python.exe -m streamlit run dashboard/dataset_review.py
```

- Use `scripts/build_inventory.py` to refresh the original still-image inventory when adding Green Bay or Mapillary images. New video collections use the separate registration command below.
- See the entire road surface before confirming a pothole-free image. A crack-only XML or an empty Roboflow target label still needs a complete-image review. In the pretraining audit, `RDD negatives: first batch` opens a 144-image queue with 93 Japan and 51 Norway candidates that already have historical training assignments. The country filter also works in the full queue. Record a subtype only when you see it. Keep uncertain images in the queue.
- For Roboflow, check the visible cavity and every supplied box. Record an evidenced capture group and a training assignment. The export partitions lack verified sequence independence. Eligible images enter training after this review. Empty-target images also need a whole-image negative decision. `--exclude-roboflow` enables a comparison without that source.
- Compare suspected pairs at original resolution and inspect their labels and sequence metadata. A reviewer may clear a misleading match or identify useful distinct frames within a common training group. Other unresolved pairs stay quarantined. A content change invalidates the applicable decision. Protected test matches and exact duplicate exclusions cannot be released through the similarity queue.
- The report flags invalid dimensions, coordinates, duplicate boxes, identical geometry with conflicting classes and declared difficult D40 boxes. Keep ambiguous cavities in review. Use the RDD positive-label queue to record an image-level concern without editing its XML.

The default cap is one verified negative per training positive. All eligible positives are kept. The builder samples negatives across sources, then across available scene and confuser groups using seed 42. It skips sampling when the verified pool fits the cap. A typical confuser is a shadow, patch, crack or manhole. Adjacent similar frames stay in one split and flagged pairs remain held for review.

```powershell
# Reproduce the current selection in a fresh destination.
.venv\Scripts\python.exe scripts/prepare_experiment_2.py --build --negative-ratio 1.0 --seed 42 --output data/training/pothole_experiment_2_reproduction --config-output configs/detector_experiment_2_reproduction.yaml

# Build another ratio with a new destination. Use 0.25, 0.50, 1.00 or all.
.venv\Scripts\python.exe scripts/prepare_experiment_2.py --build --negative-ratio 0.50 --seed 42 --output data/training/pothole_experiment_2_ratio050 --config-output configs/detector_experiment_2_ratio050.yaml

.venv\Scripts\python.exe scripts/prepare_experiment_2.py --verify-only --output data/training/pothole_experiment_2_v2
.venv\Scripts\python.exe scripts/train_detector.py --data configs/detector_experiment_2.yaml --dry-run
```

The first build defaults to `data/training/pothole_experiment_2/`. The builder uses a new version suffix if the requested dataset or configuration already exists. `--verify-only` without an output reads the active YAML for the current dataset location. For macOS or a Modal job, bring the checkout and compiled dataset together and use the same launcher with the platform's Python. The permanent YAML uses repository-relative paths. The dry run resolves a machine-local configuration and validates the paths.

## Split decisions and actual build

The build retains V1's known split assignments for approved source images, including its protected test exclusions. It uses the annotated publisher training partition for new RDD images without a historical assignment, keeps those images in training and flags the missing route metadata. Exact pixels and suspected visual similarity are checked across candidate sources and against historical holdouts. Existing Green Bay recording IDs, Mapillary sequences, Roboflow capture groups, declared route groups and known physical-defect IDs stay in one split. Validation membership and label fingerprints remain stable across ratio changes with the same data and review snapshot.

| Source | Training positives | Training negatives | Validation positives | Validation negatives | Pothole boxes, both splits |
| --- | ---: | ---: | ---: | ---: | ---: |
| RDD2022 Japan | 1,089 | 0 | 135 | 0 | 2,001 |
| RDD2022 Norway | 200 | 0 | 26 | 0 | 395 |
| Green Bay video | 38 | 116 | 0 | 0 | 90 |
| Mapillary | 7 | 148 | 2 | 27 | 12 |
| Roboflow | 0 | 0 | 0 | 0 | 0 |
| **Total** | **1,334** | **264** | **163** | **27** | **2,498** |

There are **1,598 training images**, **190 development-validation images**, **1,497 positive images** and **291 verified negatives**. Training contains 2,232 boxes. Validation contains 266 boxes. The training ratio is **1 positive to 0.1979 negatives**, or about **5.05 positives per negative**. The 0.25, 0.50, 1.0 and all-negative policies currently select the same 264 reviewed training negatives because the available pool is below even the 0.25 cap. A separate 0.50 audit reproduced the selection counts and validation fingerprint.

The source audit found 1,112 historical Japan training positives and 204 Norway training positives. Of those, 1,089 and 200 passed the current selection. The valid Japan mixed-class XML adds eligible positives with non-target boxes that the former US workflow's parser could not use. Japan supplied 1,390 images with a D40 class in the publisher's annotated partition. One of those had conflicting class geometry. Two further Japan images without D40 had invalid geometry or duplicate boxes. Four Japan images with ambiguous cavities are withheld pending a decision. Their review notes are recorded in `configs/experiment_2_label_flags.csv`, with no edits to the original boxes. The first build was archived before the corrected version was activated. The source exclusions are in `source_statistics.csv` and `excluded.csv`.

The build excluded **21,979 images** from the five source inventories. This includes 16,978 unconfirmed RDD backgrounds, 4,667 publisher test images without labels, 206 historical test rows in these approved raw sources, 73 unresolved Roboflow images, 42 images quarantined by candidate-to-candidate similarity, six images with a possible historical holdout match, four ambiguous Japan positives, and three malformed Japan annotation pairs. There are **17,055 images awaiting background, source, or cavity review** and zero unreviewed Green Bay or Mapillary inventory rows in this build. Similarity quarantines and malformed annotations are counted separately from that review number. The 481 historical reference copies audited for contamination are separate reference rows in `excluded.csv` and fall outside the source-exclusion total.

The scan reports **83 perceptual pairs and zero exact duplicate pairs** within its scanned pool. Four candidate images are held for possible overlap with historical protected test images and two for possible overlap with historical validation. `split_integrity_report.json` reports zero selected group overlap, zero selected pixel overlap, and zero historical protected test pixel matches against 221 reference images. Difference hashes can miss similar scenes, and RDD route identifiers are unavailable. Treat all reported validation as development evidence. The dataset contains train and val only.

The 66 Mapillary pairs involve 22 quarantined images. The source manifests place every pair in the same sequence and split, 1 to 8 seconds apart. Twenty-one pairs have equal difference hashes and different image pixels. The reviewed examples `677110924437253` and `937712467294716` and `965851698016133` and `2402157519990938` show changes in traffic while the road scene stays effectively the same. Those frames have distinct pixels and limited additional scene coverage. All 22 images remain in the review queue.

## Pretraining review

The local Roboflow export has 52 positive candidates with 104 class-5 boxes. Preliminary inspection covered all 52 images in contact sheets and eight at their original exported resolution. Roboflow `train/18_jpeg` and `valid/22_jpeg` share the same building, carport and road. `train/22_jpeg` and `valid/25_jpeg` show the same pavement damage with three and five boxes, respectively. The export provides no verified capture groups, and those examples show why its original partitions are insufficient evidence of independence. The review form displays the observations when image and label hashes match. No Roboflow approval was recorded. [Image-level observations](reviews/roboflow_v2_label_triage.csv)

The RDD starter queue includes 102 images with publisher alligator-crack labels. Its 144 entries were selected across countries and crack profiles from known training assignments. The first 48 were screened for obvious review difficulties. Original-resolution checks showed wet glare in `Norway_002452`, deep shade in `Norway_006457`, and dense cracking or patch seams in `Japan_003813` and `Japan_011694`. Use the complete-image review form for each decision. Publisher labels and this screening do not establish pothole-free status. The queue and ratio targets are under `review_queue/` in the pretraining audit.

| Training negative policies compared | Minimum verified pool with 1,334 positives | Additional verified negatives needed |
| --- | ---: | ---: |
| 0.25 and 0.50 | 334 | 70 |
| 0.50 and 1.00 | 668 | 404 |
| 1.00 and all | 1,335 | 1,071 |

The targets count eligible negatives after review and duplicate checks. Recheck `rdd_negative_review_plan.json` when approvals change the number of training positives. Use the same development-validation set for each ratio comparison.

## Add Green Bay footage in the existing editor

Place a copy of a new original under `data/raw/local_video/`. Choose the entire recording's role before extraction. Use a stable ID that matches the video stem with spaces replaced by underscores when you plan to use the recording in the research dashboard. Give related repeat passes the same `--scene-group-id` when you know their relationship. Use `--captured-at` with an offset if known. Record prior tuning with `--prior-tuning` and give that recording a development role.

`Green Bay Road 5 min.mp4` already supplies 154 reviewed training frames. Registration checks the original inventory and rejects reusing those source bytes under a new name. `Second Video Green Bay.mp4` has two completed detector runs and zero manual reviews in the workspace. Its route independence is unverified. For a new development-validation recording, capture a road segment that does not overlap the training footage. Record the route or segment, capture information, and any prior use. Reserve the entire original with `--split val` before annotating it. The pretraining audit records the current local inputs in `green_bay_recording_readiness.csv`.

```powershell
.venv\Scripts\python.exe scripts/register_experiment_2_recording.py --video "data/raw/local_video/GB_North_01.mp4" --recording-id GB_North_01 --split train --camera dashcam --area "Green Bay north side" --interval-seconds 2
.venv\Scripts\python.exe -m streamlit run dashboard/annotate.py -- --collection GB_North_01
```

For a separate validation capture, substitute its actual filename and road details in this example.

```powershell
.venv\Scripts\python.exe scripts/register_experiment_2_recording.py --video "data/raw/local_video/GB_Validation_01.mp4" --recording-id GB_Validation_01 --split val --camera dashcam --area "Verified separate Green Bay road segment" --scene-group-id GB_Validation_Route_01 --interval-seconds 2
.venv\Scripts\python.exe -m streamlit run dashboard/annotate.py -- --collection GB_Validation_01
```

The tool reserves a new collection under `data/processed/experiment_2_recordings/`. Review each frame in the editor and draw tight boxes around visible cavities. After looking at the entire road area, check the complete review box and save. A saved reviewed frame with zero boxes is an explicit negative. The collection saves annotation files under `data/annotations/experiment_2/<recording-id>/`, alongside the source image path, locked video ID and locked timestamp. Editing boxes requires completing that frame's review again. The preparation checks hashes, image dimensions and recording metadata before selecting the frames. A collection assigned `--split test` is reserved and fingerprinted for exclusion. Use a new dataset output for a changed collection or approved review snapshot.

## Cleanup and the first experiment

The cleanup audit files record per-file sizes, SHA256 hashes, source and annotation dependencies, and the disposition proposed at the time of the audit. The local archives were later deleted, including the retired US V3/V4 builds, the first Experiment 2 draft, and the legacy `data/processes/` conversion. Recreate a read-only inventory of the current folders with a fresh report path.

```powershell
.venv\Scripts\python.exe scripts/audit_dataset_cleanup.py --report results/dataset_audits/experiment_2_cleanup_next
Import-Csv results/dataset_audits/experiment_2_cleanup/archive_plan.csv | Format-Table source,destination,files,bytes
Import-Csv results/dataset_audits/experiment_2_cleanup/processed_inventory.csv | Where-Object action -ne keep | Format-Table repo_path,action,reason
```

The audit covered 14,607 processed files. The earlier RDD conversions under `data/processed/`, V1 split labels, converted Kaggle inputs, reviewed frames and verifier crops have remaining dependencies. V2 remains under `data/training/`. The first experiment used 2,815 training, 260 validation and 221 protected test images. Keep `configs/detector_ready.yaml`, its source preparation inputs, the V1 results, and `models/yolo26x_ready_960.pt` together. Its reported precision 0.6541, recall 0.4739, mAP50 0.5556 and mAP50-95 0.2374 are historical results. [Historical V1 guide](training_ready.md)

## Check readiness and train after approval

```powershell
.venv\Scripts\python.exe -X utf8 -m pytest tests/test_experiment_2.py tests/test_us_dataset.py tests/test_us_dataset_v4.py tests/test_training_dataset.py tests/test_rdd2022_conversion.py tests/test_dataset_safety.py tests/test_image_label_resolution.py tests/test_v3_pipeline_hygiene.py tests/test_annotation_editor.py tests/test_ground_truth_loader.py tests/test_run_safety.py -q -p no:cacheprovider
```

All 130 tests passed after the pretraining review changes. The final output passed `--verify-only`. The launcher dry run resolved 1,598 train and 190 validation images, with zero test entries. Both annotation dashboards opened against the current data in a Streamlit smoke test.

The dataset is structurally ready for a controlled development run. Japan contributes about 82% of training positives, and the Green Bay positive video frames come from one recording. Validation has only two Mapillary positives and no Green Bay recording. The training negatives are from Green Bay/Mapillary, so negative conditions from Japan and Norway are underrepresented. Before committing to a full run aimed at US performance, review approved-source hard negatives and collect a separate, reviewed Green Bay development recording with positive examples. A full 50-epoch run on the existing build is suitable only as a controlled, source-limited comparison.

The later command uses the previous 960px, batch 2, 50-epoch and CUDA device 0 settings, the existing launcher optimizer, and a new checkpoint. Execute it after approving the dataset.

```powershell
.venv\Scripts\python.exe scripts/train_detector.py --data configs/detector_experiment_2.yaml --model yolo26x.pt --imgsz 960 --batch 2 --epochs 50 --device 0 --experiment-name pothole_experiment_2_960 --output models/yolo26x_experiment_2_960.pt
```

For the capstone comparison, keep the detector and input videos fixed when comparing the detector alone, temporal filtering with BoT-SORT, and filtering plus MobileNetV3-Small verification. Tune all settings on development recordings. Measure physical-event recall, report precision, duplicate reports and false reports per minute on independent Green Bay recordings with complete timing and physical-defect annotations. Keep score thresholds frozen before using a protected recording. The preparation task produced no training or evaluation scores.
