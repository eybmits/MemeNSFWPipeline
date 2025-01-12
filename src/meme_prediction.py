import os
import re
import tarfile
import tempfile
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from transformers import BertTokenizer, BertModel

import nltk
from nltk.corpus import stopwords

from tqdm import tqdm
from torch.cuda.amp import autocast

# Debug: show current working environment and path
print("Current working directory (os.getcwd()):", os.getcwd())

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
img_size = (224, 224)
dropout_rate = 0.3

# Initialize NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
special_tokens_dict = {'additional_special_tokens': ['[NO_TEXT]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

print("Loading BERT model...")
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.resize_token_embeddings(len(tokenizer))
bert_model.eval()  # Set BERT to evaluation mode
bert_model = bert_model.to(device)  # Move BERT to appropriate device

# Define the text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        # Konvertiere Nicht-String-Werte zu Strings, falls notwendig
        text = str(text)
        
    if text == "no text":
        return "[NO_TEXT]"
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text


# Define the custom model classes
class BertTextProcessor(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(BertTextProcessor, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.dropout(pooled_output)

class ImageCNN(nn.Module):
    def __init__(self, output_dim, dropout_rate=dropout_rate):
        super(ImageCNN, self).__init__()
        self.vgg = models.vgg16(weights='DEFAULT')
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG parameters
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, output_dim)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, image):
        features = self.vgg(image)
        return self.dropout(features)

class HistogramCNN(nn.Module):
    def __init__(self, output_dim, dropout_rate=0.5):
        super(HistogramCNN, self).__init__()
        # Define the Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Define the Fully Connected Layers
        self.fc1 = nn.Linear(64, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, histogram):
        # Reshape the input for convolutional layers
        x = histogram.unsqueeze(1)  # Adding channel dimension
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.global_max_pool(x)
        x = x.squeeze(-1)  # Remove last dimension after pooling
        
        # Pass through fully connected layers
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        features = self.fc3(x)
        return features

class MCNN(nn.Module):
    def __init__(self, text_output_dim, img_output_dim, hist_output_dim, dropout_rate=dropout_rate):
        super(MCNN, self).__init__()
        self.text_processor = BertTextProcessor(dropout_rate)
        self.image_cnn = ImageCNN(img_output_dim, dropout_rate)
        self.histogram_cnn = HistogramCNN(hist_output_dim, dropout_rate)
        self.fc1 = nn.Linear(text_output_dim + img_output_dim + hist_output_dim, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask, images, histograms):
        text_features = self.text_processor(input_ids, attention_mask)
        image_features = self.image_cnn(images)
        histogram_features = self.histogram_cnn(histograms)
        combined_features = torch.cat((text_features, image_features, histogram_features), dim=1)
        x = self.relu(self.fc1(combined_features))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x).squeeze()

# Define the dataset
class MemePredictionDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, max_samples=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels['filename'] = self.img_labels.iloc[:, 0].apply(lambda x: os.path.basename(x))
        self.img_dir = img_dir
        self.transform = transform
        self.max_samples = max_samples if max_samples is not None else len(self.img_labels)
        self.image_paths = self._collect_image_paths()

        print(f"Number of image paths found: {len(self.image_paths)}")
        print(f"Number of rows in labels.csv: {len(self.img_labels)}")

    def _collect_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.img_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError:
            print(f"Cannot identify image file {img_path}. Skipping.")
            # Skip this image by returning the next one
            return self.__getitem__((idx + 1) % len(self))

        # Find the corresponding label
        img_name = os.path.basename(img_path)
        label_row = self.img_labels[self.img_labels['filename'] == img_name]
        text = label_row.iloc[0, 1] if not label_row.empty else "[NO_TEXT]"

        if self.transform:
            image = self.transform(image)

        # Preprocess text
        preprocessed_text = preprocess_text(text)
        
        # Encode the preprocessed text
        encoded_inputs = tokenizer(
            preprocessed_text, 
            return_tensors='pt', 
            max_length=128, 
            padding='max_length', 
            truncation=True
        )
        input_ids = encoded_inputs['input_ids'].squeeze(0)
        attention_mask = encoded_inputs['attention_mask'].squeeze(0)

        # Create histogram
        histogram = torch.tensor(extract_histogram(np.array(image)), dtype=torch.float32)

        return input_ids, attention_mask, image, histogram, img_path

# Histogram function
def extract_histogram(image):
    img_array = np.array(image)
    height, width, _ = img_array.shape
    total_pixels = height * width

    white_pixels = np.sum(np.all(img_array >= 230, axis=2))
    off_white_pixels = np.sum(np.all((img_array >= 200) & (img_array < 230), axis=2))
    black_pixels = np.sum(np.all(img_array <= 20, axis=2))
    off_black_pixels = np.sum(np.all((img_array > 20) & (img_array <= 50), axis=2))

    red_pixels = np.sum(img_array[:, :, 0])
    green_pixels = np.sum(img_array[:, :, 1])
    blue_pixels = np.sum(img_array[:, :, 2])

    img_array_reshaped = img_array.reshape(-1, 3)
    dominant_colors = Counter(tuple(color) for color in img_array_reshaped).most_common(10)
    dominant_color_pixels = [count for _, count in dominant_colors]
    dominant_color_pixels += [0] * (30 - len(dominant_color_pixels))

    histogram = [
        *dominant_color_pixels,
        white_pixels, off_white_pixels, black_pixels, off_black_pixels,
        red_pixels, green_pixels, blue_pixels
    ]
    histogram += [0] * (70 - len(histogram))

    return np.array(histogram) / total_pixels  # Normalize

def extract_tar(tar_path, extract_path):
    if not tarfile.is_tarfile(tar_path):
        raise ValueError(f"Die Datei {tar_path} ist kein gültiges TAR-Archiv.")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_path)
    print(f"Extrahiert {tar_path} nach {extract_path}")

def find_csv_file(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.casv'):
                return os.path.join(root, file)
    raise FileNotFoundError("Keine CSV-Datei im TAR-Archiv gefunden.")

def main():
    parser = argparse.ArgumentParser(description="Meme Prediction Script")
    parser.add_argument('--csv_file', type=str, required=True, help='Pfad zur CSV-Datei mit den Anmerkungen.')
    parser.add_argument('--images_tar', type=str, required=True, help='Pfad zur TAR-Datei, die die Bilder enthält.')
    parser.add_argument('--model_path', type=str, required=True, help='Pfad zum trainierten Modell (.pth).')
    parser.add_argument('--save_folder', type=str, required=True, help='Verzeichnis zum Speichern der Ergebnisse.')

    args = parser.parse_args()

    # Erstellen des Ausgabeordners
    os.makedirs(args.save_folder, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Extrahieren der Bilder-TAR
        images_extract_path = os.path.join(tmp_dir, 'images')
        os.makedirs(images_extract_path, exist_ok=True)
        extract_tar(args.images_tar, images_extract_path)

        # Laden des Modells mit CPU-Unterstützung
        print("Loading MCNN model...")
        model = MCNN(768, 128, 128, dropout_rate=dropout_rate).to(device)
        
        # Laden des Modells mit geeigneter Gerätezuteilung
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(args.model_path))
        else:
            model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        
        model.eval()

        # Define the image transformation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"Loading Dataset from csv={args.csv_file}, img_dir={images_extract_path} ...")
        prediction_dataset = MemePredictionDataset(
            annotations_file=args.csv_file,
            img_dir=images_extract_path,
            transform=transform
        )

        prediction_loader = DataLoader(
            prediction_dataset, 
            batch_size=16,
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 0  # Reduce workers for CPU
        )

        results = []
        meme_threshold = 0.90  # Schwellenwert für Meme-Klassifikation

        # Inference loop with torch.inference_mode and Mixed Precision
        model.eval()
        total_images = len(prediction_dataset)

        with torch.inference_mode():
            processed_images = 0
            progress_bar = tqdm(total=total_images, desc="Processing images")

            for input_ids, attention_mask, images, histograms, paths in prediction_loader:
                batch_size = len(paths)

                # Move to appropriate device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                images = images.to(device)
                histograms = histograms.to(device)

                # Forward pass
                if torch.cuda.is_available():
                    with autocast():
                        outputs = model(input_ids, attention_mask, images, histograms)
                else:
                    outputs = model(input_ids, attention_mask, images, histograms)

                # Calculate probabilities
                probabilities = torch.sigmoid(outputs)

                # Process each image in the batch
                for i, prob in enumerate(probabilities):
                    path_str = paths[i]
                    pred_value = prob.item()  # Probability as float
                    is_meme = pred_value >= meme_threshold
                    label_str = 'Meme' if is_meme else 'Not a Meme'

                    # Append prediction, probability, and boolean meme status
                    results.append({
                        'image_path': path_str,
                        'prediction': label_str,
                        'meme_probability': pred_value,
                        'is_meme': is_meme
                    })

                # Update progress bar
                processed_images += batch_size
                progress_bar.update(batch_size)

            progress_bar.close()

        # Save results to CSV
        df = pd.DataFrame(results)
        output_csv_path = os.path.join(args.save_folder, "output_predictions.csv")

        print(f"Attempting to write CSV to: {output_csv_path}")
        df.to_csv(output_csv_path, index=False)

        # Verify CSV was written
        if os.path.isfile(output_csv_path):
            print(f"CSV was successfully written to: {output_csv_path}")
            
            # Print statistics
            print("\nStatistiken:")
            print(f"Gesamtanzahl Bilder: {len(df)}")
            print(f"Als Meme klassifiziert: {df['is_meme'].sum()}")
            print(f"Nicht als Meme klassifiziert: {len(df) - df['is_meme'].sum()}")
        else:
            print("CSV was NOT found. Please check paths and permissions!")

        print(f"Predictions saved to {output_csv_path}")

if __name__ == "__main__":
    main()