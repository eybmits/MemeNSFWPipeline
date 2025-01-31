# Image Analysis Pipeline

Python pipeline for analyzing images in TAR archives:
- OCR text extraction
- Meme detection
- NSFW content detection

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Important: The pipeline must be run from the src directory!

```bash
# First, change to the src directory
cd path/to/project/src

# Then run the pipeline
python main_pipeline.py --input_dir INPUT_DIRECTORY --model_path MODEL_PATH --output_dir OUTPUT_DIRECTORY
```

Example:
```bash
cd /path/to/project/src && \
python main_pipeline.py \
    --input_dir /path/to/your/tar/files \
    --model_path /path/to/model.pth \
    --output_dir /path/to/results
```

### Parameters

- `--input_dir`: Directory containing TAR archives to process
- `--model_path`: Path to the model file (model.pth)
- `--output_dir`: Directory for output files

## Model

The model file (model.pth) needs to be downloaded separately due to file size limitations.
[Insert model download link here]