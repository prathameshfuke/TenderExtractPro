# SETUP.md -- Installation and Model Download

## Prerequisites

- Python 3.12 or later
- `uv` or `pip`
- Tesseract OCR (for scanned document support)
- Poppler (required by pdf2image for scanned PDF rendering)

## Step 1: Create Virtual Environment

```bash
cd TenderExtractPro
uv venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
```

## Step 2: Install Dependencies

```bash
uv pip install -r requirements.txt
```

### CUDA-enabled installs

The text LLM only uses the GPU when `llama-cpp-python` is installed with a CUDA-enabled build. A default CPU wheel will still run, but it will ignore `n_gpu_layers`.

```bash
uv pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

If your CUDA version is not 12.1, swap `cu121` for the matching wheel index documented by `llama-cpp-python`.

Embeddings and the optional Qwen2-VL fallback use PyTorch. Install a CUDA-enabled PyTorch build if you want those models on GPU:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
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

## Step 5: Download the Text LLM

The default pipeline uses **Mistral-7B-Instruct v0.2 Q4_K_M** via `llama-cpp-python`.

Download from HuggingFace:

```
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

Place the file in the `models/` directory:

```bash
mkdir models
# Download the file into models/
# e.g., using curl:
curl -L -o models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
```

Or set the path via environment variable:

```bash
set LLM_MODEL_PATH=D:\path\to\mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

## Step 6: Optional Multimodal Table Fallback

For OCR-heavy or visually complex tables, enable the optional Qwen2-VL path.

Recommended baseline for 8 GB VRAM-class devices:

```bash
set MULTIMODAL_ENABLED=1
set MULTIMODAL_MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct
set MULTIMODAL_LOCAL_ONLY=1
```

Download the model into the Hugging Face cache first, or point `MULTIMODAL_MODEL_PATH` at a local model directory.

The code loads the vision model only during table extraction and releases it before the Mistral text extraction stage.

## Step 7: Verify Installation

Run the individual module smoke tests:

```bash
uv run python -m tender_extraction.ingestion
uv run python -m tender_extraction.table_extraction
uv run python -m tender_extraction.chunking
uv run python -m tender_extraction.retrieval
uv run python -m tender_extraction.schemas
uv run python -m tender_extraction.validation
```

Stages 1-4 and validation run without the LLM. The extraction smoke test requires the model:

```bash
uv run python -m tender_extraction.extraction
```

## Step 8: Run the Full Pipeline

```bash
uv run python -m tender_extraction.main dataset/globaltender1576.pdf -o out.json --verbose
```

Inspect the output:

```bash
uv run python -m json.tool out.json | head -80
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL_PATH` | `models/mistral-7b-instruct-v0.2.Q4_K_M.gguf` | Path to the GGUF text model file |
| `MULTIMODAL_ENABLED` | `0` | Enable Qwen2-VL table fallback for OCR/structure-heavy pages |
| `MULTIMODAL_MODEL_NAME` | `Qwen/Qwen2-VL-2B-Instruct` | Hugging Face model ID for the vision extractor |
| `MULTIMODAL_MODEL_PATH` | unset | Optional local path to the Qwen2-VL model directory |
| `MULTIMODAL_LOCAL_ONLY` | `1` | Avoid surprise model downloads during runtime |
| `TESSERACT_CMD` | `tesseract` | Path to Tesseract OCR binary |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
