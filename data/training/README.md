# Prepared detector datasets

Use **`pothole_experiment_2_v2/`** and `configs/detector_experiment_2.yaml` for the active Experiment 2 workflow. Its train and development-validation images have matching class-zero pothole labels. The build ships with its source manifest, exclusions, review queues and split integrity report. [Experiment 2 workflow](../../docs/experiment_2_workflow.md)

| Historical V1 split | Images | Pothole-positive | Negative |
| --- | ---: | ---: | ---: |
| Train | 2,815 | 2,253 | 562 |
| Validation | 260 | 179 | 81 |
| Protected test | 221 | 177 | 44 |

The completed first experiment uses `pothole_v1/`, `configs/detector_ready.yaml`, and `models/yolo26x_ready_960.pt`. `pothole_us_v2/` is retained. [V1 guide](../../docs/training_ready.md)
