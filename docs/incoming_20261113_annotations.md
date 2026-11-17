# Final incoming photograph annotations — 17 November 2026

The final user export `annotations.manual (1).json` has now been imported for all
100 images, preserving every supplied annotation record exactly. Current final
counts: **485 parcel, 123 barcode, and 31 shipping-label boxes**, plus 113 excluded
regions. Twenty-four records differ from the preceding assistant review.
The exact export and the previous active annotations are backed up in
`data/annotated/next_batch/final_import_20261117/`. The final metadata is in
`review_provenance.json` under `final_annotation_set`.

The gallery and YOLO/COCO files have been regenerated from this final set.
Earlier review counts and uncertainty notes below describe the history of the
assistant review; they do not supersede the final annotations. Final means the
user has finished editing this set, not that model performance has been tested.
No dataset split or training was started.

## Previous assistant review

The 100 incoming photographs have separate clean copies in
`data/annotated/next_batch/images/`. Original images are unchanged.
Open `data/output/next_batch/annotation_review/index.html` for the current gallery
and editor. Full-size marked images are in its `annotated/` directory; these
marked images must not be used as the model's input.

The user's exported corrections were imported from `annotations.manual.json`.
The file contained changes to **A17–A27**, and those eleven annotation records
(including their notes) were preserved exactly. The remaining **89 images** were
visually rechecked. Boxes changed in 71 of those images; some empty or simple
frames needed no box changes. Changes include missing parcel instances, small
parcels beneath shipping labels, worn labels incorrectly excluded as dimension
stickers, and incorrectly placed or clipped barcode boxes.

At the end of the preceding assistant review, counts were **471 parcel**, **111 barcode**, and **28 shipping-label**
boxes. These are occurrences across frames, not numbers of independent objects.
There are also 113 excluded candidate regions, which are displayed in orange
but are not exported as training targets. The original pass contained 305,
72, and 6 target boxes respectively, so the earlier conclusion that labels were
very scarce was partly caused by missed annotations.

## Annotation conventions

- `0: barcode`: visible 1D barcode patterns, including direct packaging print,
  folds, worn print, and visible fragments. Boxes contain the visible pattern.
- `1: label`: a physical shipping/address label. Printed packaging graphics and
  dimension/test stickers remain excluded. A shipping label and the barcode
  on it have separate, nested annotations.
- `2: parcel`: one box per distinguishable physical parcel, around its visible
  extents, clipped to the image edges. A small parcel on a larger parcel is a
  separate object. Hidden parcel extents are not reconstructed.

Worn labels are still labels. QR and stacked 2D symbols are outside the current
1D barcode definition. B08 and B09 contain small stacked/code-like printed
regions excluded from the 1D class; B33 contains an ambiguous, partly hidden
printed fragment. These are kept as excluded candidates, with notes in the
gallery, rather than silently represented as confirmed 1D barcodes.

Overlapping dark mailers retain uncertainty where their borders are hidden.
The relevant gallery cards say so. These are assistant visual annotations,
not independently human-verified ground truth. The user's preserved frames may
also contain areas they have not yet corrected; preserving them does not imply
that all their objects were independently verified in this review.

## Files and review history

- `annotations.manual.json`: current original-pixel boxes, keyed by image ID.
- `labels/all/*.txt`: YOLO boxes for all 100 images.
- `annotations.coco.json`: COCO export of the same target boxes.
- `review_provenance.json`: protected user edits, reviewed images, before/after
  counts, images with changed boxes, and remaining class-review points.
- `review_backup_20261117/before_recheck.manual.json`: previous assistant boxes.
- `review_backup_20261117/user_export.json`: exact copy of the user's export.
- `export_annotations.py`: regenerates annotations and the interactive gallery.
- `gallery_editor.js`: editor behavior, embedded into the generated gallery.

Browser drafts are scoped to the annotation revision. Refreshing this revised
gallery loads the updated reviewed boxes instead of an older saved draft.
Old drafts remain stored under their old keys; they are not deleted. New edits
save under the current revision. Export JSON when finished so corrections can
be applied to the source annotations and regenerated exports.

Regenerate from the repository directory with:

```sh
.venv/bin/python -m src.export_annotation_gallery
```

No training, dataset split, threshold selection, or model evaluation was performed
as part of this annotation review.
