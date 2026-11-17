# Current 100-photo annotations

Final user corrections are saved in `data/annotated/next_batch/annotations.json`.
The folder contains only 100 pictures with reference boxes and this one coordinate JSON.
The saved boxes are unchanged: 485 parcels, 123 barcodes, 31 shipping labels,
and 113 excluded regions. `review_id` preserves the A01–B50 review references.

Box lists use original-image `[x1, y1, x2, y2]` pixels. `barcode`, `label` and
`parcel` are training classes; `excluded` regions are review information only.
Each photo's physical `group` must be confirmed before splitting. Group IDs are
not inferred from appearance. These reused parcel views may be one connected
group, requiring new independent parcels for validation/test.

Gallery: `tmp/next_batch/annotation_review/index.html`.
Regenerate it with `.venv/bin/python -m src.export_annotation_gallery`.
The editor exports the same simple `annotations.json` format; browser drafts
require export/import before changing the permanent coordinate file.

Older coordinate formats, the original final export and review notes were
preserved in `tmp/archive/20261117/annotation_metadata/next_batch/` during cleanup.
They are recovery material, not dependencies of the active dataset.
See `docs/simple_workflow.md` for the inference → supervised workflow.
