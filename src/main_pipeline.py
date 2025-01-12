#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import logging
from pathlib import Path
import pandas as pd

# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """
    Führt einen Shell-Befehl aus und überprüft den Rückgabestatus.
    """
    logger.info(f"Starte: {description}")
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.info(f"{description} abgeschlossen.")
        logger.debug(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Fehler beim Ausführen von {description}: {e.stderr}")
        raise RuntimeError(f"{description} fehlgeschlagen.") from e

def get_tar_contents(tar_path):
    """
    Liest alle Dateinamen aus einem TAR-Archiv.
    """
    try:
        result = subprocess.run(
            ['tar', '-tf', tar_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Filtere nur Bilddateien
        files = [os.path.basename(f.strip()) for f in result.stdout.split('\n') if 
                f.strip().lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        return files
    except subprocess.CalledProcessError as e:
        logger.error(f"Fehler beim Lesen des TAR-Archivs: {e.stderr}")
        raise RuntimeError("TAR-Archiv konnte nicht gelesen werden.") from e

def find_csv_file(directory, pattern):
    """
    Findet eine CSV-Datei in einem Verzeichnis basierend auf einem Muster.
    """
    for file in os.listdir(directory):
        if file.endswith('.csv') and pattern in file:
            return os.path.join(directory, file)
    raise FileNotFoundError(f"Keine passende CSV-Datei mit Pattern '{pattern}' gefunden in: {directory}")

def merge_results(ocr_csv, prediction_csv, nsfw_csv, output_path, all_images):
    """
    Führt die OCR-, Prediction- und NSFW-Ergebnisse in einer CSV-Datei zusammen.
    Berücksichtigt dabei alle Bilder aus dem TAR, auch nicht verarbeitbare.
    """
    # OCR-Ergebnisse laden
    ocr_df = pd.read_csv(ocr_csv)
    ocr_df = ocr_df.rename(columns={'extracted_text': 'ocr_text'})
    
    # "no text" durch leeren String ersetzen
    ocr_df['ocr_text'] = ocr_df['ocr_text'].replace('no text', '')

    # Prediction-Ergebnisse laden
    pred_df = pd.read_csv(prediction_csv)
    
    # NSFW-Ergebnisse laden
    nsfw_df = pd.read_csv(nsfw_csv)
    
    # Pfade normalisieren für den Merge
    ocr_df['image_path'] = ocr_df['image_path'].apply(lambda x: os.path.basename(x))
    pred_df['image_path'] = pred_df['image_path'].apply(lambda x: os.path.basename(x))
    nsfw_df['image_path'] = nsfw_df['image_path'].apply(lambda x: os.path.basename(x))
    
    # DataFrame für alle Bilder erstellen
    all_images_df = pd.DataFrame({'image_path': all_images})
    
    # DataFrames zusammenführen
    merged_df = pd.merge(
        all_images_df,
        ocr_df,
        on='image_path',
        how='left'
    )
    
    merged_df = pd.merge(
        merged_df,
        pred_df,
        on='image_path',
        how='left'
    )
    
    merged_df = pd.merge(
        merged_df,
        nsfw_df,
        on='image_path',
        how='left'
    )
    
    # Leere Werte mit entsprechenden Defaults füllen
    merged_df['ocr_text'] = merged_df['ocr_text'].fillna('')
    merged_df['meme_probability'] = merged_df['meme_probability'].fillna(0.0)
    merged_df['is_meme'] = merged_df['is_meme'].fillna(False)
    merged_df['nsfw_probability'] = merged_df['nsfw_probability'].fillna(0.0)
    merged_df['is_nsfw'] = merged_df['is_nsfw'].fillna(False)
    
    # Sortieren nach Dateipfad
    merged_df = merged_df.sort_values('image_path')
    
    # Spalten neu ordnen
    final_columns = [
        'image_path',
        'ocr_text',
        'meme_probability',
        'is_meme',
        'nsfw_probability',
        'is_nsfw'
    ]
    merged_df = merged_df[final_columns]
    
    # Ergebnisse speichern
    merged_df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Kombinierte Ergebnisse gespeichert in: {output_path}")
    
    # Statistiken ausgeben
    logger.info("\nGesamtstatistiken:")
    logger.info(f"Gesamtanzahl Bilder im TAR: {len(all_images)}")
    logger.info(f"Erfolgreich verarbeitete Bilder: {len(ocr_df)}")
    logger.info(f"Bilder mit OCR-Text: {merged_df['ocr_text'].notna().sum()}")
    logger.info(f"Als Meme klassifiziert: {len(merged_df[merged_df['is_meme'] == True])}")
    logger.info(f"Als NSFW klassifiziert: {len(merged_df[merged_df['is_nsfw'] == True])}")

def process_single_tar(images_tar, model_path, output_dir):
    """
    Verarbeitet eine einzelne TAR-Datei.
    """
    # Definieren von Unterverzeichnissen
    tar_basename = os.path.splitext(os.path.basename(images_tar))[0]
    tar_output_dir = os.path.join(output_dir, tar_basename)
    
    ocr_output_dir = os.path.join(tar_output_dir, 'ocr_results')
    prediction_output_dir = os.path.join(tar_output_dir, 'prediction_results')
    nsfw_output_dir = os.path.join(tar_output_dir, 'nsfw_results')
    
    os.makedirs(ocr_output_dir, exist_ok=True)
    os.makedirs(prediction_output_dir, exist_ok=True)
    os.makedirs(nsfw_output_dir, exist_ok=True)

    try:
        # Liste aller Bilder im TAR erhalten
        all_images = get_tar_contents(images_tar)
        logger.info(f"Gefundene Bilder im TAR: {len(all_images)}")

        # Schritt 1: OCR-Verarbeitung
        ocr_command = [
            sys.executable,
            'ocr_processor.py',
            '--images_tar', images_tar,
            '--output_dir', ocr_output_dir
        ]
        run_command(ocr_command, f"OCR-Verarbeitung der Bilder für {tar_basename}")

        # OCR-CSV finden
        ocr_csv = find_csv_file(ocr_output_dir, 'ocr')

        # Schritt 2: Meme Prediction
        prediction_command = [
            sys.executable,
            'meme_prediction.py',
            '--csv_file', ocr_csv,
            '--images_tar', images_tar,
            '--model_path', model_path,
            '--save_folder', prediction_output_dir
        ]
        run_command(prediction_command, f"Meme Prediction für {tar_basename}")

        # Prediction-CSV finden
        prediction_csv = find_csv_file(prediction_output_dir, 'predictions')

        # Schritt 3: NSFW Detection
        nsfw_command = [
            sys.executable,
            'nsfw_detection.py',
            '--csv_file', ocr_csv,
            '--images_tar', images_tar,
            '--save_folder', nsfw_output_dir
        ]
        run_command(nsfw_command, f"NSFW Detection für {tar_basename}")

        # NSFW-CSV finden
        nsfw_csv = find_csv_file(nsfw_output_dir, 'nsfw')

        # Schritt 4: Alle Ergebnisse zusammenführen
        final_output_path = os.path.join(output_dir, f'{tar_basename}_results.csv')
        merge_results(ocr_csv, prediction_csv, nsfw_csv, final_output_path, all_images)

        logger.info(f"Pipeline für {tar_basename} erfolgreich abgeschlossen.")
        return final_output_path

    except Exception as e:
        logger.error(f"Pipeline für {tar_basename} fehlgeschlagen: {str(e)}")
        return None

def main(input_dir, model_path, output_dir):
    """
    Hauptfunktion für die Pipeline.
    """
    start_time = os.times()

    # Alle TAR-Dateien im Eingabeverzeichnis finden
    tar_files = [f for f in os.listdir(input_dir) if f.endswith('.tar')]
    
    if not tar_files:
        logger.error(f"Keine TAR-Dateien im Verzeichnis {input_dir} gefunden.")
        sys.exit(1)

    logger.info(f"Gefundene TAR-Dateien: {len(tar_files)}")
    
    # Liste für erfolgreich verarbeitete Dateien
    successful_results = []
    failed_files = []

    # Verarbeite jede TAR-Datei
    for tar_file in tar_files:
        tar_path = os.path.join(input_dir, tar_file)
        logger.info(f"\nVerarbeite TAR-Datei: {tar_file}")
        
        try:
            result_path = process_single_tar(tar_path, model_path, output_dir)
            if result_path:
                successful_results.append(result_path)
            else:
                failed_files.append(tar_file)
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung von {tar_file}: {str(e)}")
            failed_files.append(tar_file)

    # Abschlussbericht
    logger.info("\n=== Verarbeitungsbericht ===")
    logger.info(f"Erfolgreich verarbeitete TAR-Dateien: {len(successful_results)}/{len(tar_files)}")
    if failed_files:
        logger.info("Fehlgeschlagene Dateien:")
        for failed_file in failed_files:
            logger.info(f"- {failed_file}")

    duration = os.times()[4] - start_time[4]
    logger.info(f"\nGesamtzeit der Pipeline: {duration:.2f} Sekunden")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline zur OCR-Verarbeitung und Meme Prediction.")
    parser.add_argument('--input_dir', type=str, required=True, 
                      help='Verzeichnis mit den TAR-Dateien.')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Pfad zum trainierten Modell (.pth).')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Basisverzeichnis zum Speichern der Ergebnisse.')

    args = parser.parse_args()

    # Eingabeverzeichnis überprüfen
    if not os.path.isdir(args.input_dir):
        logger.error(f"Das angegebene Eingabeverzeichnis existiert nicht: {args.input_dir}")
        sys.exit(1)

    if not os.path.isfile(args.model_path):
        logger.error(f"Die angegebene Modell-Datei existiert nicht: {args.model_path}")
        sys.exit(1)

    # Ausgabeverzeichnis erstellen
    os.makedirs(args.output_dir, exist_ok=True)

    # Hauptfunktion aufrufen
    main(args.input_dir, args.model_path, args.output_dir)