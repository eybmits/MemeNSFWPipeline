#!/usr/bin/env python3
import argparse
import logging
import os
import tarfile
import pandas as pd
from transformers import pipeline
from PIL import Image
import io
import torch

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_image_from_tar(tar_path, image_path):
    """
    Lädt ein Bild aus einer TAR-Datei und stellt sicher, dass die Berechtigungen der extrahierten Datei 
    angepasst werden.
    """
    with tarfile.open(tar_path, 'r') as tar:
        try:
            # Alle Mitglieder des TAR-Files auflisten
            all_members = tar.getmembers()
            
            # Basename des gesuchten Bildes
            target_basename = os.path.basename(image_path)
            
            # Suche nach dem Dateinamen im TAR
            matching_member = None
            for member in all_members:
                if os.path.basename(member.name) == target_basename:
                    matching_member = member
                    break
            
            if matching_member is None:
                raise ValueError(f"Bild nicht gefunden: {target_basename}")
                
            # Bild extrahieren und laden
            img_file = tar.extractfile(matching_member)
            if img_file is None:
                raise ValueError(f"Bild konnte nicht extrahiert werden: {target_basename}")
                
            extracted_path = os.path.join(os.path.dirname(image_path), matching_member.name)
            
            # Berechtigungen anpassen, falls extrahiert
            if os.path.isfile(extracted_path):
                os.chmod(extracted_path, 0o666)
                logger.info(f"Berechtigungen aktualisiert: {extracted_path}")
            
            img_data = img_file.read()
            return Image.open(io.BytesIO(img_data))
            
        except Exception as e:
            logger.error(f"Fehler beim Laden von {image_path}: {str(e)}")
            return None


def process_images(csv_file, images_tar, save_folder):
    """
    Verarbeitet Bilder mit dem NSFW-Detektionsmodell.
    """
    # NSFW-Detektionsmodell laden
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Verwende Gerät: {device}")
    
    classifier = pipeline(
        "image-classification",
        model="Falconsai/nsfw_image_detection",
        device=device
    )

    # CSV-Datei laden
    df = pd.read_csv(csv_file)
    results = []
    nsfw_threshold = 0.5  # Schwellenwert für NSFW-Klassifikation

    total_images = len(df)
    logger.info(f"Starte NSFW-Überprüfung für {total_images} Bilder")

    for idx, row in df.iterrows():
        image_path = row['image_path']
        
        if idx % 10 == 0:
            logger.info(f"Verarbeite Bild {idx+1}/{total_images}")

        try:
            # Bild aus TAR laden
            image = load_image_from_tar(images_tar, image_path)
            if image is None:
                logger.warning(f"Überspringe Bild {image_path}: Konnte nicht geladen werden")
                results.append({
                    'image_path': os.path.basename(image_path),
                    'nsfw_probability': None,
                    'is_nsfw': None
                })
                continue

            # NSFW-Erkennung durchführen
            prediction = classifier(image)
            
            # Ergebnisse extrahieren
            nsfw_score = next((p['score'] for p in prediction if p['label'] == 'nsfw'), 0.0)
            is_nsfw = nsfw_score > nsfw_threshold
            
            results.append({
                'image_path': os.path.basename(image_path),
                'nsfw_probability': nsfw_score,
                'is_nsfw': not is_nsfw
            })
            
            logger.debug(f"NSFW Score für {image_path}: {nsfw_score} (NSFW: {is_nsfw})")

        except Exception as e:
            logger.error(f"Fehler bei Bild {image_path}: {str(e)}")
            results.append({
                'image_path': os.path.basename(image_path),
                'nsfw_probability': None,
                'is_nsfw': None
            })

    # Ergebnisse in DataFrame konvertieren
    results_df = pd.DataFrame(results)
    
    # Statistiken ausgeben
    nsfw_count = len(results_df[results_df['is_nsfw'] == True])
    logger.info(f"\nNSFW-Statistiken:")
    logger.info(f"Gesamtanzahl Bilder: {len(results_df)}")
    logger.info(f"Als NSFW klassifiziert (P > {nsfw_threshold}): {nsfw_count}")

    # Ergebnisse speichern
    output_path = os.path.join(save_folder, 'nsfw_results.csv')
    results_df.to_csv(output_path, index=False)
    logger.info(f"Ergebnisse gespeichert in: {output_path}")

    return output_path

def main():
    parser = argparse.ArgumentParser(description="NSFW-Erkennung für Bilder.")
    parser.add_argument('--csv_file', type=str, required=True,
                      help='Pfad zur CSV-Datei mit Bildpfaden')
    parser.add_argument('--images_tar', type=str, required=True,
                      help='Pfad zur TAR-Datei mit den Bildern')
    parser.add_argument('--save_folder', type=str, required=True,
                      help='Verzeichnis zum Speichern der Ergebnisse')

    args = parser.parse_args()

    # Überprüfen der Eingaben
    if not os.path.isfile(args.csv_file):
        logger.error(f"CSV-Datei nicht gefunden: {args.csv_file}")
        return 1
    
    if not os.path.isfile(args.images_tar):
        logger.error(f"TAR-Datei nicht gefunden: {args.images_tar}")
        return 1

    # Ausgabeverzeichnis erstellen
    os.makedirs(args.save_folder, exist_ok=True)

    try:
        process_images(args.csv_file, args.images_tar, args.save_folder)
        return 0
    except Exception as e:
        logger.error(f"Fehler bei der Verarbeitung: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())