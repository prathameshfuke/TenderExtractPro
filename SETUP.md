# SETUP.md -- Installation and Model Download

## Prerequisites

- Python 3.10 or later
- pip (Python package manager)
- Tesseract OCR (for scanned document support)
- Poppler (required by pdf2image for scanned PDF rendering)

## Step 1: Create Virtual Environment

```bash
cd d:\TenderExtractPro
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

If `llama-cpp-python` fails to build from source, install the pre-built wheel:

```bash
pip install llama-cpp-python --prefer-binary
```

## Step 3: Install Tesseract OCR

**Windows:**
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

After installation, add Tesseract to PATH or set the environment variable:

```bash
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

## Step 4: Install Poppler (for scanned PDFs)

**Windows:**
Download from: https://github.com/oschwartz10612/poppler-windows/releases

Extract and add the `bin/` folder to your system PATH.

**Linux:**
```bash
sudo apt-get install poppler-utils
```

## Step 5: Download the LLM Model

The pipeline uses Phi-3-mini-4k-instruct quantized to Q4 GGUF (approximately 2.4 GB).

Download from HuggingFace:

```
https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
```

Place the file in the `models/` directory:

```bash
mkdir models
# Download the file into models/
# e.g., using curl:
curl -L -o models/Phi-3-mini-4k-instruct-q4.gguf \
  "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
```

Or set the path via environment variable:

```bash
set LLM_MODEL_PATH=D:\path\to\Phi-3-mini-4k-instruct-q4.gguf
```

## Step 6: Verify Installation

Run the individual module smoke tests:

```bash
python -m tender_extraction.ingestion
python -m tender_extraction.table_extraction
python -m tender_extraction.chunking
python -m tender_extraction.retrieval
python -m tender_extraction.schemas
python -m tender_extraction.validation
```

Stages 1-4 and validation run without the LLM. The extraction smoke test requires the model:

```bash
python -m tender_extraction.extraction
```

## Step 7: Run the Full Pipeline

```bash
python -m tender_extraction.main dataset/globaltender1576.pdf -o out.json --verbose
```

Inspect the output:

```bash
python -m json.tool out.json | head -80
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL_PATH` | `models/Phi-3-mini-4k-instruct-q4.gguf` | Path to the GGUF model file |
| `TESSERACT_CMD` | `tesseract` | Path to Tesseract OCR binary |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
