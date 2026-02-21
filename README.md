# TenderExtractPro 🎯

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

AI-powered extraction of **technical specifications** and **scope of work** from tender documents using advanced RAG (Retrieval-Augmented Generation) techniques.

---

## ✨ Features

- 🔍 **Hybrid Retrieval** — BM25 + semantic embeddings (FAISS) for 18–25% precision improvement
- 📊 **Table Extraction** — Dedicated pipeline preserving table structure (critical: 70%+ of specs are in tables)
- 🎯 **Anti-Hallucination** — Mandatory citations, grounding verification, constrained Pydantic schemas
- 🔗 **Source Citations** — Every extraction includes chunk_id, page number, and exact source text
- 📄 **Multi-Format** — PDF (text/scanned), DOCX, JPG, PNG
- 🚀 **CPU-Only** — Runs on CPU using quantized Mistral-7B (no GPU needed)
- 📝 **Structured JSON** — Validated output with confidence scores (HIGH / MEDIUM / LOW)

---

## 🏗 Architecture

```
Document → Ingestion → Table Extraction → Chunking → Retrieval → LLM Extraction → Validation → JSON Output
             │              │                 │            │              │               │
         PDF/DOCX/IMG   pdfplumber       Hierarchical   BM25+FAISS   Mistral-7B     Grounding
         Tesseract OCR  structure        section-aware  hybrid       llama.cpp      fuzzy-match
                        preservation     metadata       fusion       Pydantic       confidence
```

---

## 📦 Prerequisites

### 1. Python 3.9+

### 2. Tesseract OCR

| OS | Install |
|---|---|
| **Windows** | Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and install to `C:\Program Files\Tesseract-OCR\` |
| **Ubuntu/Debian** | `sudo apt install tesseract-ocr` |
| **macOS** | `brew install tesseract` |

### 3. Poppler (for PDF → image conversion)

| OS | Install |
|---|---|
| **Windows** | Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases) and add `bin/` to PATH |
| **Ubuntu/Debian** | `sudo apt install poppler-utils` |
| **macOS** | `brew install poppler` |

### 4. LLM Model (Mistral-7B-Instruct GGUF)

Download the quantized model (~4 GB):

```bash
# Create a models directory
mkdir -p ~/models

# Download the Q4 quantized model from HuggingFace
# Option 1: Using wget
wget -O ~/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Option 2: Using huggingface-cli
pip install huggingface-hub
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir ~/models
```

Set the environment variable (or update `config.py`):
```bash
export LLM_MODEL_PATH="$HOME/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/TenderExtractPro.git
cd TenderExtractPro

# Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run on a Document

```bash
# CLI usage
python -m tender_extraction.main path/to/tender.pdf -o output.json

# With verbose logging
python -m tender_extraction.main path/to/tender.pdf -o output.json --verbose
```

### 3. Use as a Library

```python
from tender_extraction.main import TenderExtractionPipeline

pipeline = TenderExtractionPipeline()
result = pipeline.run("path/to/tender.pdf", output_path="result.json")

# Access results
for spec in result["technical_specifications"]:
    print(f"{spec['item_name']}: {spec['specification_text']}")
    print(f"  Source: page {spec['source']['page']}, confidence: {spec['confidence']}")
```

### 4. Run Tests

```bash
# Run the test suite (no LLM or Tesseract needed for tests)
python tests/test_pipeline.py

# Or with pytest
python -m pytest tests/test_pipeline.py -v
```

---

## 📁 Project Structure

```
TenderExtractPro/
├── tender_extraction/
│   ├── __init__.py           # Package init
│   ├── config.py             # Central configuration (all tunable parameters)
│   ├── schemas.py            # Pydantic v2 models for validated output
│   ├── ingestion.py          # Multi-format document loading + OCR
│   ├── table_extraction.py   # Dedicated table pipeline (pdfplumber)
│   ├── chunking.py           # Hierarchical chunking with metadata
│   ├── retrieval.py          # Hybrid BM25 + FAISS retrieval
│   ├── extraction.py         # LLM integration + anti-hallucination
│   ├── validation.py         # Grounding verification + confidence
│   └── main.py               # Pipeline orchestration + CLI
├── tests/
│   └── test_pipeline.py      # 14 tests covering all modules
├── sample_output/
│   └── example_output.json   # Example extraction result
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

Edit `tender_extraction/config.py` or set environment variables:

| Setting | Default | Description |
|---|---|---|
| `TESSERACT_CMD` | Auto-detected | Path to tesseract binary |
| `LLM_MODEL_PATH` | `~/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf` | Path to GGUF model |
| BM25 weight | 0.4 | Keyword retrieval weight |
| Embedding weight | 0.6 | Semantic retrieval weight |
| High confidence | ≥ 0.90 | Grounding threshold for HIGH |
| Medium confidence | ≥ 0.60 | Grounding threshold for MEDIUM |
| Chunk size | 200–500 tokens | Min/max tokens per chunk |

---

## 📊 Output Format

See [`sample_output/example_output.json`](sample_output/example_output.json) for a complete example.

```json
{
  "technical_specifications": [
    {
      "item_name": "Steel Reinforcement Bars",
      "specification_text": "Grade 60 steel bars conforming to ASTM A615",
      "unit": "kg",
      "numeric_value": "500",
      "tolerance": "± 5%",
      "standard_reference": "ASTM A615",
      "material": "Steel Grade 60",
      "source": {"chunk_id": "...", "page": 15, "exact_text": "..."},
      "confidence": "HIGH"
    }
  ],
  "scope_of_work": {
    "tasks": [...],
    "exclusions": [...]
  }
}
```

---

## 🔒 Anti-Hallucination Safeguards

1. **Constrained generation** — All output validated against Pydantic schemas
2. **Mandatory citations** — Every extraction requires `chunk_id` + `page` + `exact_text`
3. **Grounding verification** — Fuzzy-match extracted text against source chunks
4. **NOT_FOUND enforcement** — Missing fields return `"NOT_FOUND"`, never guessed values
5. **Confidence scoring** — HIGH (≥90% match), MEDIUM (≥60%), LOW (<60%)
6. **Rejection threshold** — Extractions below 40% grounding are silently dropped

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.
