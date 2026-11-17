> Advanced commands. For the simple JSON-based workflow, use [simple_workflow.md](simple_workflow.md). These low-level examples generate their own training format; they are not the annotated batch folder.

# Stregkoder: inferens, egne data og supervised finjustering

Kør kommandoerne fra `bachelor-detection` med `.venv/bin/python`.
Notebooken `barcode_detection.ipynb` har også celler til de nye funktioner.
Der er endnu ikke trænet på kommende egne driftsdata. Opsætningen kan kontrolleres
teknisk med et lille prøvedatasæt; sådanne resultater dokumenterer ikke kvaliteten
på transportbåndet.

## 1. Inferens: find stregkoder og eventuelt hele fragtlabels

```sh
.venv/bin/python -m src.detect_barcodes --with-labels --confidence 0.001 --label-confidence 0.001 --output data/output/baseline
```

Stregkoder findes med den eksisterende specialtrænede YOLOv8s-model. Whole-label
funktionen betyder den fysiske fragtetiket, ikke tekstgenkendelse/OCR. Den bruger
som udgangspunkt den eksisterende YOLO-World med beskrivelser af en shipping label.
Det er en baseline, hvis egnethed skal måles på egne annotationer.

Efter finjustering kan en fælles model med klasserne `barcode` og `label` bruges:

```sh
.venv/bin/python -m src.detect_barcodes --weights data/output/training/own_v1/detector/weights/best.pt --with-labels --output data/output/own_v1
```

En separat trænet labelmodel kan angives med `--label-weights /sti/label_model.pt`.
Klassen skal hedde `label`. Lilla bokse viser labels; grøn/orange viser aflæste/
ikke-aflæste stregkoder. `barcodes.json` har separate lister `detections` og `labels`.
En stregkode knyttes til en label, hvis dens centrum ligger i præcis én labelboks.
Flere mulige labels gemmes som tvetydige; det er geometrisk tilknytning, ikke en
verificeret pakkeidentitet. Søgning i hele billedet bevarer uafhængigheden af labeldetektoren.

## 2. Egne data skal annoteres

```sh
.venv/bin/python -m src.train_barcodes init --dataset tmp/manual_dataset --with-labels
```

Udelad `--with-labels`, hvis datasættet kun skal bruges til stregkoder.
Skabelonen opretter denne struktur, men ingen billeder eller falske referencebokse:

```text
own/
  data.yaml
  groups.csv
  images/train/    labels/train/
  images/val/      labels/val/
  images/test/     labels/test/
```

Brug eksempelvis CVAT til at tegne bokse og eksportere i Ultralytics YOLO-format.
Klassernes rækkefølge i eksporten skal passe til `names` i `data.yaml`:
`0: barcode`, `1: label`, når begge bruges. Koden accepterer mapper eller YOLO
billedlister i YAML. Billedlister fortolkes relativt til listens placering.
Træningen får derefter absolutte billedstier.

Hvis eksporten i stedet er **COCO JSON**, kan den konverteres direkte. Opret
skabelonen først, og importér hver allerede adskilt datadel til det tilsvarende
split. Importen opdeler ikke beslægtede frames automatisk:

```sh
.venv/bin/python -m src.train_barcodes import-coco --annotations /sti/val_annotations.json --images /sti/rene_val_billeder --data tmp/manual_dataset/data.yaml --split val
```

Hvis kategorierne hedder eksempelvis `Stregkode` og `Fragtlabel`, tilføjes
`--barcode-category Stregkode --label-category Fragtlabel`. Kategori-ID'er mappes
til klassernes rækkefølge i jeres YAML. Rene billeder kopieres uden ændringer;
dimensioner og bokse kontrolleres, og eksisterende filer overskrives ikke.
Eksporten skal være fuldt gennemgået: COCO-billeder uden relevante bokse bliver
negative billeder med en tom annotationsfil. Crowd-bokse afvises til manuel gennemgang.

Hvis leverancen kun indeholder billeder med indtegnede bokse, skal I bede om
den oprindelige koordinat-eksport og de rene billeder. Andre formater skal først
identificeres og konverteres; de kan ikke behandles som YOLO ved at omdøbe filer.

En annotationslinje er `klasse x_center y_center bredde højde`, normaliseret til 0–1.
Alle fem værdier skal være til stede; koordinaterne skal holde boksen inden for billedet.
En fil med ekstra confidencekolonne er modeloutput og accepteres ikke som referenceformat.

- Markér selve stregkodemønstret og hele fragtetiketten som to separate bokse.
- Markér alle synlige instanser af de konfigurerede klasser, også ulæselige stregkoder.
- Brug en fælles politik for delvist skjulte koder; annotér ikke et usynligt objekt ved at gætte.
- Gennemgå negative billeder og gem en tom `.txt`. Manglende `.txt` betyder uannoteret.
- Modelbokse kan være forslag, men skal kontrolleres, rettes og suppleres manuelt.
- Bevar rene billeder. Tegnede bokse må ikke være brændt ind i modellens input.

Opdel efter optagelse og fysisk pakke. Naboframes og samme pakke må ikke ligge i
både træning og validering/test. `groups.csv` kan registrere dette:

```csv
image,group
images/train/frame_001.jpg,recording_A
images/train/frame_002.jpg,recording_A
images/val/frame_101.jpg,recording_B
images/test/frame_201.jpg,recording_C
```

Hvis grupper angives, skal alle billeder have en gruppe, og en gruppe må kun
forekomme i ét split. Vælg grupper, så både fælles pakker og beslægtede frames
holdes samlet. Uden grupper kontrolleres kun identiske dekodede billedpixels;
det kan ikke opdage alle næsten ens frames. Et header-only groups.csv er tomt.

```sh
.venv/bin/python -m src.train_barcodes check --data tmp/manual_dataset/data.yaml
```

## 3. Supervised finjustering

```sh
.venv/bin/python -m src.train_barcodes train --data tmp/manual_dataset/data.yaml --dry-run
.venv/bin/python -m src.train_barcodes train --data tmp/manual_dataset/data.yaml --output data/output/training/own_v1 --epochs 100 --batch 4 --device cpu
```

`--dry-run` kontrollerer data og udskriver opsætningen uden at indlæse modellen,
oprette et træningsresultat eller opdatere vægte. En rigtig kørsel starter fra
de eksisterende stregkodevægte; Ultralytics tilpasser klassifikationens output
til datasættets klasser. En ny klasse, `label`, skal læres fra de nye annotationer.
På understøttet Apple-hardware kan `--device mps` afprøves; CPU er standard.

Gemte filer omfatter `experiment.json`, absolutte billedlister, Ultralytics'
træningskurver og `detector/weights/best.pt`. Manifestet indeholder indstillinger,
versionsnumre og hashes af startvægte, billeder og annotationsfiler. En eksisterende
resultatmappe overskrives ikke. Testsplit leveres ikke til træning/modelvalg.

De indledende værdier er et udgangspunkt, ikke optimerede hyperparametre.
Træningsopsætningen bruger moderat rotation/lysstyrkevariation og ingen spejling.
Vælg billedstørrelse og batch efter data og hardware. Undersøg, om koder bliver
for små ved nedskalering. Aflæsning bruger fortsat udsnit fra originalopløsningen.

## 4. Mål output med referenceannotationer

Kør samme evaluering på baseline og finjusteret model, med samme split,
billedstørrelse, IoU-definition og regler for valg af confidencegrænse:

```sh
.venv/bin/python -m src.evaluate_barcodes --data tmp/manual_dataset/data.yaml --weights models/barcode_yolov8s.pt --output data/output/evaluation/baseline
.venv/bin/python -m src.evaluate_barcodes --data tmp/manual_dataset/data.yaml --weights data/output/training/own_v1/detector/weights/best.pt --output data/output/evaluation/own_v1
```

Denne indgang måler boksdetektion uden afkodning. En baseline uden labelklasse
vil ikke forudsige fragtlabels; det skal indgå i fortolkningen. For at evaluere
den samlede baseline med YOLO-World-labels skal inferens først køres på samme
billedsplit, og det gemte output sendes ind:

```sh
.venv/bin/python -m src.detect_barcodes --input tmp/manual_dataset/images/val --with-labels --confidence 0.001 --label-confidence 0.001 --output data/output/baseline_val
.venv/bin/python -m src.evaluate_barcodes --data tmp/manual_dataset/data.yaml --predictions data/output/baseline_val/barcodes.json --output data/output/evaluation/baseline_combined
```

Der skal være output for hvert referencebillede, også når modellen fandt nul
objekter. Filhash og dimensioner skal passe. Referencekoordinater og modelbokse
skal referere til samme rene original. Evaluationens `--confidence 0.25` og
`--iou 0.5` er standarder, som kan ændres. Modelfund gemt under en højere
inferensgrænse kan ikke genskabes ved evalueringen.

Rapporten `index.html` indeholder:

- **TP, FP, FN og precision/recall/F1** ved den valgte confidence og IoU.
- **IoU pr. korrekt matchet fund** og gennemsnit/median/minimum/maksimum samlet
  og pr. klasse. IoU = overlapareal / samlet areal (intersection / union).
  Rapporten viser også et histogram over de korrekte match og den valgte IoU-grænse.
  Uden korrekte match er IoU udefineret (`null` i JSON og — i tabellen).
  Disse tal gælder kun fund med korrekt klasse og tilstrækkeligt overlap ved
  den valgte confidencegrænse. Oversete objekter indgår i recall, ikke som
  opdigtede IoU-værdier på 0. Læs derfor altid IoU sammen med recall og antallet af match.
- **Confusion matrix** med faktisk klasse på rækkerne og forudsagt på kolonnerne.
  Background-kolonnen er oversete objekter; background-rækken er falske fund.
  En forkert klasse på den rigtige position ses uden for diagonalen.
  Der tælles ingen TN og beregnes ingen almindelig accuracy.
- **Confidencekurver** for precision, recall og F1 samt scorefordeling for
  korrekte/falske fund. En højere scoregrænse kan også fjerne korrekte fund.
- **AP50 og AP50–95/mAP** med 101 recallpunkter og én-til-én matching i faldende
  confidenceorden. AP bruger alle gemte fund, ikke kun visningsgrænsen.
  Dette er en eksplicit, enkel detektionsevaluering; den implementerer ikke COCO's
  area-/crowd-/maxDets-delanalyser. Tal kan afvige fra Ultralytics' matcher.
  Klasser uden referenceobjekter har udefineret AP og udelades fra mAP.
- **Billedgennemgang** med referencebokse, korrekte fund, falske fund og oversete objekter.

Confidence er en score, ikke automatisk en kalibreret sandsynlighed. Confusion
matrixen beregnes ved en valgt scoregrænse; de to størrelser skal ikke multipliceres.
Precision = TP/(TP+FP), recall = TP/(TP+FN). I rapportens simple brøker vises 0,
hvis nævneren er tom; kontrollér altid antallene sammen med brøkerne.
Den samlede oversigt viser precision, recall og gennemsnitlig IoU ved siden af
hinanden. Billedkortene viser confidence og IoU for hvert korrekt matchet fund.

Brug valideringssplit til at vælge confidence og øvrige indstillinger. Brug først
`--split test` til den afsluttende sammenligning, når valgene er fastlagt.
Kontrollér også vanskelige forhold som små koder, refleksioner, sløring og vinkler.

En vellykket ZXing-afkodning dokumenterer ikke i sig selv korrekt indhold.
Det kræver verificerede kodeværdier som separat reference. Denne evaluering
måler lokalisering og klasse; den måler ikke OCR, payloadnøjagtighed, tracking
eller korrekt pakkeidentitet. Antal aflæsninger forbliver beskrivende statistik.

## Dokumentation

- [Ultralytics: træning](https://docs.ultralytics.com/modes/train/)
- [Ultralytics: evaluering](https://docs.ultralytics.com/modes/val/)
- [Ultralytics: YOLO-World](https://docs.ultralytics.com/models/yolo-world/)
- [CVAT: Ultralytics YOLO-eksport](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/)
