# Data folders

Use these three folders:

- **raw/<batch>/**: unchanged original photos.
- **annotated/<batch>/**: clean working images and the matching bounding-box coordinates.
- **output/<batch>/**: annotation gallery, generated train/validation/test datasets, trained models and analysis.

For the current batch, use the name `next_batch`. Its final annotations are already saved.
The older 23-photo collection is named `real_photos`. Previous experiments and recovery material are in `output/archive/`.

Inside an annotated batch, `images/` contains clean photos, and `labels/` contains YOLO `.txt` coordinates. In this context, “labels” means annotations for all object classes. Coloured boxes are rendered only in the output gallery.

Settings are in `../workflow.yaml`. Start with `../README.md` and `../docs/simple_workflow.md`.
Image data, annotations, checkpoints and results stay local and are excluded from Git.
