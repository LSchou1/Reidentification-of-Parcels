# Annotationer af de 23 eksisterende fotos

De rene originaler er bevaret. Der er lavet selvstændige, byte-identiske kopier i
`data/annotated/real_photos/images/all/`. Billeder med indtegnede bokse
ligger separat i `data/output/archive/annotation_review/annotated/`; de må ikke bruges
som modellens input.

Åbn `data/output/archive/annotation_review/index.html` for at gennemgå alle 23 billeder,
koordinater, udsnit og noter. Blå viser stregkoder, grøn viser fragtlabels, og
orange viser andre mærkater, der er registreret men ikke medtaget som målobjekter.

## Indhold

- 23 rene billedkopier i original opløsning, 4284 × 5712 pixels.
- 23 YOLO-annotationsfiler med klasserne `0: barcode` og `1: label`.
- 27 stregkodebokse og 10 fragtlabelbokse.
- 8 gennemgåede billeder uden målobjekter, repræsenteret med tomme `.txt`-filer.
- `annotations.json`: pixelkoordinater, regler, attributter og gennemgangsnoter.
- `annotations.coco.json`: samme målobjekter som COCO-bokse.
- `review_summary.csv` og `summary.json`: oversigt og kontrolresultater.
- `export_annotations.py`: genskaber YOLO/COCO-eksport og de markerede kopier fra
  koordinaterne i `annotations.json` uden at ændre de rene billeder.

Alle billeder og hvert målobjekts udsnit er visuelt gennemgået af assistenten.
Boksene er specificeret ud fra de rene billeder og nærbilleder med koordinatgitter;
de er ikke kopieret fra detektorens output. Dette er assistentens visuelle
annotationer og er endnu ikke uafhængigt kontrolleret af et menneske.

## Regler og vanskelige tilfælde

`barcode` omfatter synlige 1D-stregmønstre, også direkte tryk på emballagen.
En synlig kode annoteres også, når den er slidt, krøllet eller ikke kan aflæses.
Delvist skjulte koder får kun en boks om den synlige del. Tekst, logoer og
2D-symboler er ikke medtaget som 1D-stregkoder.

`label` omfatter den fysiske forsendelses-/adresselabel, told-/postdokumentet eller
en selvstændig sporingsmærkat. Emballagens trykte grafik, test-/dimensionsmærkater,
blanke papirrester og små hjælpe-/håndteringsmærkater er udeladt under denne
definition. Otte sådanne områder er bevaret som `excluded_candidates` i JSON
og vises med orange bokse, så definitionen senere kan ændres uden at lede igen.

Otte målobjekter har `needs_boundary_review: true`. Det drejer sig om meget små
kantfragmenter, falmede streger og foldede/overlappende labelkanter. For de tre
sammensatte scener annoteres hovedområdet med postlabel; gamle papirrester, der
ikke sikkert kan adskilles i selvstændige labels, er beskrevet i noterne.
Gennemgå disse markerede tilfælde før en endelig, publiceret evaluering.

## Udviklingssæt uden kunstig trænings-/testopdeling

De samme pakker optræder i flere fotos, og de sidste fotos kombinerer dem i én
scene. Hele samlingen holdes derfor som én udviklingsgruppe. Der er ingen
automatisk træning/testopdeling og ingen træning er startet.

`data.yaml` angiver kun `val: images/all`, så den eksisterende evaluator kan
læse annotationspuljen. `groups.csv` placerer alle fotos i samme udviklingsgruppe.
Navnet `val` er her evaluatorens tekniske indgang; det gør ikke samlingen til et
uafhængigt testsæt. Træningskommandoen kræver fortsat et særskilt træningssplit
og vil derfor afvise denne valideringspulje som et færdigt træningsdatasæt.

Når nye billeder er klar, skal træning/validering/test opdeles med hensyn til
fælles pakker og optagelser. Den allerede undersøgte samling er egnet som pilot-
og udviklingsmateriale, ikke som dokumentation for generalisering til ny drift.

Regenerér de afledte filer efter en koordinatrettelse:

```sh
.venv/bin/python data/annotated/real_photos/export_annotations.py
```
