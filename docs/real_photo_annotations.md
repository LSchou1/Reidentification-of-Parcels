# Earlier 23-photo development collection

Originals are in `data/raw/real_photos/`. Pictures with reference boxes and one
`annotations.json` are in `data/annotated/real_photos/`.

This earlier collection was annotated for barcode and shipping-label detection.
It has no parcel annotation field: missing parcel boxes must not be treated as
reviewed negative examples. It remains development material, not an independent
three-class train/validation/test benchmark. All photos retain the group
`development_collection_23`.

Current coordinate records use the same filename-keyed pixel format as the new
batch. Earlier policies, detailed review records, exports and reports are kept
in `tmp/archive/20261117/`. The active project does not depend on those backups.
See `docs/simple_workflow.md` for the current data layout and commands.
