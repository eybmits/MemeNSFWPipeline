#!/usr/bin/env python3
import tarfile
import os
import pytesseract
from PIL import Image
import multiprocessing as mp
from pathlib import Path
import tempfile
import logging
from typing import List, Tuple
import time
import pandas as pd
from tqdm import tqdm
import argparse

# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_temp_dir() -> str:
    """Erstellt ein temporäres Verzeichnis für die extrahierten Dateien."""
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Temporäres Verzeichnis erstellt: {temp_dir}")
    return temp_dir

def extract_tar(tar_path: str, extract_path: str) -> List[str]:
    """Extrahiert das TAR-Archiv und passt Berechtigungen der extrahierten Dateien an."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    image_files = []
    
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if Path(member.name).suffix.lower() in image_extensions:
                tar.extract(member, extract_path)
                extracted_path = os.path.join(extract_path, member.name)
                
                # Berechtigungen anpassen, um Lesbarkeit/Schreibbarkeit sicherzustellen
                if os.path.isfile(extracted_path):
                    os.chmod(extracted_path, 0o666)
                    logger.info(f"Berechtigungen aktualisiert: {extracted_path}")
                
                image_files.append(extracted_path)
    
    logger.info(f"{len(image_files)} Bilddateien gefunden und extrahiert.")
    return image_files


def process_image(args: Tuple[str, str]) -> Tuple[str, str, str]:
    """
    Verarbeitet ein einzelnes Bild mit OCR und gibt den Bildpfad,
    den Status und den extrahierten OCR-Text zurück.

    Args:
        args: Ein Tupel bestehend aus (image_path, output_dir)

    Returns:
        Tuple[str, str, str]: (original_path, status, extracted_text)
    """
    image_path, output_dir = args
    language = 'deu+eng'  # Festgelegte OCR-Sprachen
    try:
        # Bild laden und OCR durchführen
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=language)
        
        # Originalen Bildpfad aus dem TAR-Archiv extrahieren
        original_path = str(Path(image_path).relative_to(Path(output_dir).parent))
        
        # Text bereinigen: nur Zeilenumbrüche entfernen, Leerzeichen behalten
        cleaned_text = ' '.join(text.splitlines()).strip()
        
        if cleaned_text:
            return original_path, "success", cleaned_text
        else:
            return original_path, "success", "no text"  # Änderung hier: "no text" statt None
                
    except Exception as e:
        logger.error(f"Fehler bei {image_path}: {str(e)}")
        return image_path, f"error: {str(e)}", "no text"  # Auch hier "no text" bei Fehlern

def create_csv_output(results: List[Tuple[str, str, str]], tar_name: str, output_dir: str):
    """Erstellt eine CSV-Datei mit den OCR-Ergebnissen."""
    # CSV-Dateiname basierend auf dem TAR-Namen erstellen
    csv_filename = os.path.join(output_dir, f"{Path(tar_name).stem}_ocr.csv")
    
    # DataFrame erstellen
    df = pd.DataFrame(results, columns=['image_path', 'status', 'extracted_text'])
    
    # Fehlerhafte Verarbeitungen im Log ausgeben
    errors = df[df['status'].str.startswith('error:', na=False)]
    if not errors.empty:
        logger.error("Fehler bei der Verarbeitung folgender Bilder:")
        for _, row in errors.iterrows():
            logger.error(f"- {row['image_path']}: {row['status']}")
    
    # Status-Spalte entfernen und CSV speichern
    df = df[['image_path', 'extracted_text']]
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    logger.info(f"CSV-Datei erstellt: {csv_filename}")
    
    # Statistik ausgeben
    total = len(df)
    with_text = len(df[df['extracted_text'] != "no text"])
    without_text = len(df[df['extracted_text'] == "no text"])
    
    logger.info("Statistik:")
    logger.info(f"- Gesamt verarbeitete Bilder: {total}")
    logger.info(f"- Bilder mit erkanntem Text: {with_text}")
    logger.info(f"- Bilder ohne erkannten Text: {without_text}")

def main(images_tar: str, output_dir: str):
    """Hauptfunktion für die OCR-Verarbeitung."""
    start_time = time.time()
    
    # Temporäres Verzeichnis erstellen
    temp_dir = setup_temp_dir()
    
    try:
        # TAR-Datei extrahieren
        image_files = extract_tar(images_tar, temp_dir)
        
        if not image_files:
            logger.error("Keine Bilddateien im TAR-Archiv gefunden.")
            return
        
        # Ausgabeverzeichnis erstellen
        os.makedirs(output_dir, exist_ok=True)
        
        # Multiprocessing-Pool erstellen
        num_cores = mp.cpu_count()
        logger.info(f"Verwende {num_cores} CPU-Kerne für die Verarbeitung.")
        
        # Argumente für die Verarbeitung vorbereiten
        process_args = [(img, temp_dir) for img in image_files]
        
        # Parallele Verarbeitung mit Fortschrittsanzeige
        with mp.Pool(num_cores) as pool:
            results = list(tqdm(
                pool.imap(process_image, process_args),
                total=len(process_args),
                desc="Verarbeite Bilder",
                unit="Bild"
            ))
        
        # CSV-Datei erstellen
        create_csv_output(results, images_tar, output_dir)
        
        # Ergebnisse ausgeben
        successful = sum(1 for _, status, _ in results if status == "success")
        logger.info(f"Verarbeitung abgeschlossen: {successful}/{len(image_files)} erfolgreich.")
        
    finally:
        # Aufräumen des temporären Verzeichnisses
        try:
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(temp_dir)
            logger.info(f"Temporäres Verzeichnis entfernt: {temp_dir}")
        except Exception as e:
            logger.warning(f"Fehler beim Entfernen des temporären Verzeichnisses: {str(e)}")
    
    duration = time.time() - start_time
    logger.info(f"Gesamtzeit: {duration:.2f} Sekunden")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR-Verarbeitung von Bildern in einem TAR-Archiv.")
    parser.add_argument('--images_tar', type=str, required=True, help='Pfad zur TAR-Datei, die die Bilder enthält.')
    parser.add_argument('--output_dir', type=str, required=True, help='Verzeichnis zum Speichern der OCR-Ergebnisse.')
    
    args = parser.parse_args()
    
    # Überprüfen, ob die angegebene TAR-Datei existiert
    if not os.path.isfile(args.images_tar):
        logger.error(f"Die angegebene TAR-Datei existiert nicht: {args.images_tar}")
        exit(1)
    
    # Hauptfunktion aufrufen
    main(args.images_tar, args.output_dir)