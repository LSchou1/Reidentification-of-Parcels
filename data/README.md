# Three data folders

- `raw/<batch>/`: unchanged original photographs.
- `annotated/<batch>/`: pictures with reference boxes directly in the folder and ONE `annotations.json`.
- `output/<batch>/`: annotation gallery, named experiments and the fixed split record.

Open `output/next_batch/index.html` for current results. The current batch is already annotated.
Raw photos stay clean; JSON stores original-pixel `[x1, y1, x2, y2]` boxes, physical group IDs and annotation review status.
YOLO files are generated automatically under output when splitting. Do not draw boxes into training images.
Older reports and retired projects are stored in `tmp/archive/` at project root.
Settings: `../workflow.yaml`. Guide: `../docs/simple_workflow.md`.
Photos, coordinates, weights, reports and archives stay local and are excluded from Git.

Disposable generated training data, previews and scratch files go in `../tmp/`. Deleting tmp does not delete source data or reports.
