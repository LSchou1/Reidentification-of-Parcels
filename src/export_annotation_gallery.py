"""Build a review gallery from clean photos and annotations.json; outputs stay separate."""
from __future__ import annotations

import argparse
import json
import os
import html
import hashlib
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from workflow import batch_configs, read_config
from src.barcode_dataset import EXTENSIONS
from src.workflow_reports import write_output_index

CODE_ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"b": (0, "barcode", "#25b9ff"), "l": (1, "shipping label", "#39d879"), "p": (2, "parcel", "#ff5757")}


def draw_review(image_path, annotation, thumb_path, marked_path):
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for key, (_, label, color) in CLASSES.items():
        for box in annotation.get(key, []):
            draw.rectangle(tuple(box), outline=color, width=7)
            draw.text((box[0] + 3, box[1] + 3), label, fill=color, stroke_width=2, stroke_fill="black")
    for box in annotation.get("e", []):
        draw.rectangle(tuple(box), outline="#ffb82e", width=7)
        draw.text((box[0] + 3, box[1] + 3), "excluded sticker", fill="#ffb82e", stroke_width=2, stroke_fill="black")
    image.save(marked_path, quality=92)
    image.thumbnail((900, 760))
    image.save(thumb_path, quality=88)


def export_gallery(settings):
    """Render one batch; main calls this once for each selected batch."""
    ROOT = Path(settings["annotations"])
    IMAGE_DIR = Path(settings["input"])
    REVIEW_DIR = Path(settings["temporary"]) / "annotation_review"
    MARKED_DIR, THUMB_DIR = ROOT, Path(settings["temporary"]) / "thumbnails"
    WORK_DIR = Path(settings["temporary"])
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    records = json.loads((ROOT / "annotations.json").read_text())
    annotations = {name: {"b": record.get("barcode", []), "l": record.get("label", []),
                         "p": record.get("parcel", []), "e": record.get("excluded", []),
                         "note": record.get("note", "")} for name, record in records.items()}
    provenance_path = WORK_DIR / "review_provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    user_corrected = set(provenance.get("user_corrected_images", []))
    final_set = provenance.get("final_annotation_set", {}).get("status") == "user_final"
    files = sorted(IMAGE_DIR / name for name in records)
    if any(not p.resolve().is_relative_to(IMAGE_DIR.resolve()) or not p.is_file() for p in files):
        raise ValueError("Each photo needs exactly one annotations.json record.")
    MARKED_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [], "categories": [{"id": i, "name": n} for i, n, _ in CLASSES.values()]}
    html_rows = []
    counts = {n: 0 for _, n, _ in CLASSES.values()}
    for image_id, path in enumerate(files, 1):
        name = path.relative_to(IMAGE_DIR).as_posix()
        key = records[name].get("review_id", name)
        ann = annotations[name]
        with Image.open(path) as im:
            w, h = ImageOps.exif_transpose(im).size
        for short, (class_id, label, color) in CLASSES.items():
            for box in ann.get(short, []):
                x1, y1, x2, y2 = map(int, box)
                if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                    raise ValueError(f"Invalid {key} {label} box: {box} for {w}x{h}")
                counts[label] += 1
                coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": image_id, "category_id": class_id, "bbox": [x1, y1, x2-x1, y2-y1], "area": (x2-x1)*(y2-y1), "iscrowd": 0})
        coco["images"].append({"id": image_id, "file_name": name, "width": w, "height": h})
        thumb = str(Path(name).with_suffix(".jpg")).replace("/", "__")
        marked_link = html.escape(os.path.relpath(MARKED_DIR / thumb, REVIEW_DIR), quote=True)
        draw_review(path, ann, THUMB_DIR / thumb, MARKED_DIR / thumb)
        note = ("Your corrected boxes were imported and preserved. Earlier assistant notes for this image have been superseded."
                if key in user_corrected else ann.get("note", ""))
        if final_set:
            note = "Final annotations supplied by you and saved unchanged. Earlier assistant review notes are historical."
        excluded = len(ann.get("e", []))
        html_rows.append(f'<article id="{key}"><h2>{key} <small>{path.name}</small></h2><a href="{marked_link}"><img src="{os.path.relpath(THUMB_DIR / thumb, REVIEW_DIR)}" onerror="this.onerror=null;this.src=&quot;{marked_link}&quot;" loading="lazy"></a><p>Parcels: {len(ann.get("p", []))} · Barcodes: {len(ann.get("b", []))} · Shipping labels: {len(ann.get("l", []))} · Excluded stickers: {excluded}</p><p>{html.escape(note)}</p><p><a href="{os.path.relpath(path, REVIEW_DIR)}">Clean image</a> · <a href="{marked_link}">Full-size annotated copy</a> · <button class="edit-image" data-key="{key}">Add or edit boxes</button></p></article>')
    editor_data = {}
    for path in files:
        name = path.relative_to(IMAGE_DIR).as_posix()
        key = records[name].get("review_id", name)
        with Image.open(path) as im:
            w, h = ImageOps.exif_transpose(im).size
        review_note = ("Your corrected boxes were imported and preserved. Earlier assistant notes for this image have been superseded."
                       if key in user_corrected else annotations[name].get("note", ""))
        if final_set:
            review_note = "Final annotations supplied by you and saved unchanged. Earlier assistant review notes are historical."
        editor_data[key] = {"src": f"{os.path.relpath(path, REVIEW_DIR)}", "width": w, "height": h, "filename": name, "record": records[name], "annotation": annotations[name], "review_note": review_note}
    embedded_data = json.dumps(editor_data, ensure_ascii=True).replace("</", "<\\/")
    (WORK_DIR / "annotations.coco.json").write_text(json.dumps(coco, indent=2) + "\n")
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Incoming photo annotation review</title><style>
body{font:16px system-ui;background:#10151b;color:#e8edf2;margin:24px}header{position:sticky;top:0;z-index:2;background:#10151bf2;padding:12px;border-bottom:1px solid #334}article{max-width:960px;margin:28px auto;padding:18px;background:#1b232c;border-radius:12px}article img{width:100%;height:auto}small{display:block;font-size:12px;overflow-wrap:anywhere;color:#9bb}a{color:#75c8ff}button,select{font:inherit;padding:8px 12px;border-radius:7px;border:1px solid #667;background:#263442;color:#fff;cursor:pointer}button:disabled{opacity:.5;cursor:default}.top-actions{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.hint{color:#c4d3df;font-size:14px}.editor{position:fixed;inset:0;z-index:5;background:#0c1117f5;overflow:auto;padding:18px}.editor[hidden]{display:none}.editor-head{position:sticky;top:-18px;background:#10151bf5;padding:12px 0;z-index:2}.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.canvas-frame{position:relative;display:inline-block;line-height:0;margin-top:14px;max-width:100%}.canvas-frame img{display:block;max-width:min(100%,1200px);max-height:72vh;width:auto;height:auto;object-fit:contain}.canvas-frame canvas{position:absolute;inset:0;cursor:crosshair;touch-action:none}.box-list{max-width:1200px;line-height:1.8}.box-list button{padding:3px 8px;margin-left:8px}.box-row{cursor:pointer;padding:2px 6px}.box-row.selected{background:#314354;border-radius:5px}.status{color:#9fe3b0;min-height:1.5em}.class-b{color:#25b9ff}.class-l{color:#39d879}.class-p{color:#ff5757}.class-e{color:#ffb82e}
/* The stage keeps a fixed size; the box list cannot shift it while dragging. */
.editor{padding:0;overflow:hidden;display:flex;flex-direction:column;background:#0c1117}
.editor-head{position:static;padding:12px 18px;flex:none;border-bottom:1px solid #334}
.editor-head h2{font-size:17px;margin:0 0 10px}.editor-head .hint{margin:10px 0 0}
.editor-workspace{display:grid;grid-template-columns:minmax(0,1fr) 270px;flex:1;min-height:0}
.image-stage{min-width:0;min-height:0;display:flex;align-items:center;justify-content:center;padding:16px;overflow:hidden}
.canvas-frame{display:block;margin:0;max-width:none;flex:none;user-select:none;-webkit-user-select:none}
.canvas-frame img{width:100%;height:100%;max-width:none;max-height:none;pointer-events:none}
.canvas-frame canvas{display:block;width:100%;height:100%;touch-action:none;outline:none}
.editor-sidebar{overflow:auto;padding:14px;border-left:1px solid #334;background:#171f28}
.box-row{display:flex;align-items:center;gap:8px;padding:8px;margin:4px 0;line-height:1.5;border:1px solid transparent;font-size:13px}
.box-row.selected{border-color:#75c8ff}.box-row button{margin-left:auto}
.box-row .coordinates{display:block;font-size:11px;color:#abc;font-variant-numeric:tabular-nums}
button[aria-pressed="true"]{background:#23618d;border-color:#8ed1ff}
@media(max-width:850px){.editor-workspace{grid-template-columns:minmax(0,1fr) 190px}.controls{gap:5px}.controls button,.controls select{padding:6px;font-size:13px}.editor-head .hint{font-size:12px}.editor-sidebar{padding:8px}}
</style></head><body>
<header><h1>Incoming parcel photos — annotation review</h1><p>Blue: barcode · Green: shipping label · Red: parcel · Orange: excluded sticker. Click “Add or edit boxes” on a photo to draw boxes around missing objects or remove existing ones.</p><p class="hint">Use Select / move to correct boxes, or Draw box to add a missing object. Drafts save in this browser. Use Export annotations when you finish, then share the downloaded JSON so the dataset files can be updated.</p><div class="top-actions"><button id="export">Export annotations JSON</button><label>Import an edited JSON <input id="import" type="file" accept=".json,application/json"></label><span id="save-status" class="status">Draft storage is local to this browser.</span></div></header>
''' + "\n".join(html_rows) + '''
<section id="editor" class="editor" hidden>
  <div class="editor-head">
    <h2 id="editor-title"></h2>
    <div class="controls">
      <button id="previous">← Previous</button><button id="next">Next →</button>
      <button id="select-mode" aria-pressed="true">Select / move (V)</button>
      <button id="draw-mode" aria-pressed="false">Draw box (B)</button>
      <label>New box <select id="class"><option value="b">Barcode</option><option value="l">Shipping label</option><option value="p">Parcel</option></select></label>
      <button id="undo" disabled>Undo</button><button id="delete-selected" disabled>Delete selected</button>
      <button id="export-editor">Export JSON</button><button id="close">Close</button>
    </div>
    <p class="hint" id="interaction-hint">Click a box to select it. Drag to move; drag a handle to resize. Use Draw box to add an object inside an existing parcel.</p>
  </div>
  <div class="editor-workspace">
    <div id="image-stage" class="image-stage"><div id="canvas-frame" class="canvas-frame"><img id="canvas-image" alt="Image being annotated" draggable="false"><canvas id="box-canvas" tabindex="0" aria-label="Annotation editing canvas"></canvas></div></div>
    <aside class="editor-sidebar"><p id="editor-note" class="hint"></p><h3>Boxes</h3><div id="box-list" class="box-list"></div></aside>
  </div>
</section>
<script>const originalData=__ANNOTATIONS__; const annotationRevision=__ANNOTATION_REVISION__;</script>
<script>__EDITOR_SCRIPT__</script></body></html>'''
    page = page.replace("__ANNOTATIONS__", embedded_data)
    revision = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()[:16]
    page = page.replace("__ANNOTATION_REVISION__", json.dumps(revision))
    page = page.replace("__EDITOR_SCRIPT__", (CODE_ROOT / "src/annotation_editor.js").read_text())
    (REVIEW_DIR / "index.html").write_text(page)
    write_output_index(settings)
    print(f"Exported {len(files)} images; boxes: {counts}; COCO objects: {len(coco['annotations'])}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CODE_ROOT / "workflow.yaml")
    for settings in batch_configs(read_config(parser.parse_args(argv).config)):
        if "batch" in settings:
            print(f"Batch: {settings['batch']}")
        export_gallery(settings)


if __name__ == "__main__":
    main()
