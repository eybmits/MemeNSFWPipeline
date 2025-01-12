# Image Analysis Pipeline

Python-Pipeline zur Analyse von Bildern in TAR-Archiven (OCR, Meme-Erkennung, NSFW-Erkennung).

## Installation

```bash
pip install pandas numpy pillow torch torchvision
```

## Ausführung

```bash
python main.py --input_dir EINGABE_VERZEICHNIS --model_path MODELL_PFAD --output_dir AUSGABE_VERZEICHNIS
```

### Beispiel
```bash
python main.py --input_dir ./input_images --model_path ./model/model.pth --output_dir ./results
```

## Ausgabe

Für jedes TAR-Archiv wird eine CSV-Datei mit folgenden Informationen erstellt:
- Bildname
- Extrahierter Text
- Meme-Wahrscheinlichkeit
- NSFW-Wahrscheinlichkeit

Die Ergebnisse finden Sie in: `output_dir/archiv_name_results.csv`