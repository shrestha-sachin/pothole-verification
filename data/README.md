# Data folders

The active build is [Experiment 2](../docs/experiment_2_workflow.md). It uses Japan and Norway RDD2022, reviewed Green Bay and Mapillary images and explicitly reviewed Roboflow v2 images when eligible.

| Folder | Purpose |
| --- | --- |
| `raw/` | Original downloads, still images and video recordings |
| `annotations/` | Manual boxes and review decisions |
| `metadata/` | Image inventory, source identifiers and provenance |
| `processed/experiment_2_recordings/` | Extracted frames and inventories for new Green Bay recordings |
| `processed/video_frames/` | Existing Green Bay frames referenced by manual annotations |
| `processed/rdd2022_v3/` | V1 split metadata and labels, also the Experiment 2 RDD split reference |
| `processed/kaggle_arcioni_v4/` | Converted images and labels required to reproduce V1 |
| `processed/verifier_crops/` | Separate classifier data indexed by the crop inventory |
| `training/pothole_experiment_2_v2/` | Prepared YOLO images, labels, and audit reports |
| `training/pothole_v1/` | Completed first experiment and its protected test split |
| `training/pothole_us_v2/` | Retained initial US comparison |

Keep the source image used by each annotation. The first experiment and its inputs are documented in the V1 guide. [V1 guide](../docs/training_ready.md)
