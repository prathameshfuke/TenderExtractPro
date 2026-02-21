# TenderExtractPro

A production-grade RAG pipeline for extracting technical specifications and scope of work from tender documents. Processes real PDF, DOCX, and image files through a 6-stage pipeline: ingestion, table extraction, chunking, hybrid retrieval, LLM extraction, and grounding validation.

## Architecture

```
Input Document (PDF/DOCX/Image)
        |
        v
[1. Ingestion] -- pdfplumber + Tesseract OCR fallback
        |
        v
[2. Table Extraction] -- pdfplumber dual-strategy (bordered + borderless)
        |
        v
[3. Chunking] -- Section-aware hierarchical chunking
        |
        v
[4. Hybrid Retrieval] -- BM25 (rank_bm25) + FAISS semantic search
        |
        v
[5. LLM Extraction] -- Mistral-7B-Instruct via llama-cpp-python
        |
        v
[6. Validation] -- rapidfuzz grounding verification
        |
        v
Structured JSON Output
```

## Quick Start

```bash
# 1. Setup environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Download LLM model (see SETUP.md for details)
mkdir models
# Download mistral-7b-instruct-v0.2.Q4_K_M.gguf into models/

# 3. Run pipeline
python -m tender_extraction.main dataset/globaltender1576.pdf -o output.json --verbose
```

See [SETUP.md](SETUP.md) for detailed installation instructions including Tesseract, Poppler, and model download.

## Project Structure

```
TenderExtractPro/
  tender_extraction/
    config.py           -- Centralized configuration
    schemas.py          -- Pydantic v2 models for structured output
    ingestion.py        -- PDF/DOCX/image loading with OCR fallback
    table_extraction.py -- Dedicated table parsing pipeline
    chunking.py         -- Section-aware hierarchical chunking
    retrieval.py        -- Hybrid BM25 + FAISS retrieval
    extraction.py       -- LLM-powered specification extraction
    validation.py       -- Grounding verification using rapidfuzz
    main.py             -- Pipeline orchestration and CLI
  tests/
    test_pipeline.py    -- Integration and unit tests
  dataset/              -- Real tender PDF files for testing
  models/               -- LLM model files (not in version control)
  sample_output/        -- Example extraction output
```

## Output Format

The pipeline produces a JSON file with the following structure:

```json
{
  "technical_specifications": [
    {
      "item_name": "Steel Reinforcement Bars",
      "specification_text": "Grade 60 conforming to ASTM A615",
      "unit": "kg",
      "numeric_value": "500",
      "tolerance": "NOT_FOUND",
      "standard_reference": "ASTM A615",
      "material": "Steel Grade 60",
      "source": {
        "chunk_id": "table_001_row_3_a1b2c3d4",
        "page": 15,
        "exact_text": "Steel reinforcement bars Grade 60 per ASTM A615"
      },
      "confidence": "HIGH"
    }
  ],
  "scope_of_work": {
    "tasks": [
      {
        "task_description": "Site preparation and leveling",
        "deliverables": ["Cleared site", "Completion report"],
        "timeline": "2 weeks",
        "dependencies": ["Site access approval"],
        "source": {
          "chunk_id": "chunk_8_e5f6a7b8",
          "page": 8,
          "exact_text": "The contractor shall carry out site preparation..."
        }
      }
    ],
    "exclusions": [
      {
        "item": "Furniture and interior decoration",
        "source": {"chunk_id": "chunk_12_c3d4e5f6", "page": 12}
      }
    ]
  }
}
```

Every extracted value includes a source citation pointing back to the exact chunk and page in the original document. Fields not found in the document are set to `"NOT_FOUND"` -- the system never invents values.

## Anti-Hallucination Safeguards

1. **Prompt Engineering**: The LLM prompt explicitly instructs "use NOT_FOUND for missing fields, NEVER invent values" and requires source citations for every extraction.

2. **Grounding Verification**: After LLM extraction, every spec is fuzzy-matched against source chunks using `rapidfuzz.fuzz.partial_ratio`. Specs with grounding score below 0.40 are rejected.

3. **Confidence Scoring**: Each spec receives a confidence level based on grounding quality:
   - HIGH (score >= 0.90): Near-exact match to source text
   - MEDIUM (score >= 0.60): Paraphrased or minor OCR differences
   - LOW (score < 0.60): Uncertain, flagged for manual review

4. **Pydantic Validation**: Output is validated against strict Pydantic v2 models. Empty specification text is rejected by a field validator.

## Configuration

All tunable parameters are centralized in `tender_extraction/config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `ocr.dpi` | 300 | OCR rendering resolution |
| `ocr.scanned_char_threshold` | 50 | Chars below which a page is treated as scanned |
| `chunking.max_chunk_tokens` | 400 | Maximum tokens per text chunk |
| `retrieval.bm25_weight` | 0.4 | BM25 weight in score fusion |
| `retrieval.embedding_weight` | 0.6 | Embedding weight in score fusion |
| `llm.temperature` | 0.1 | LLM generation temperature |
| `validation.min_grounding_ratio` | 0.40 | Minimum grounding score to accept a spec |

## Running Tests

```bash
# Run all tests
python tests/test_pipeline.py

# Run individual module smoke tests (no LLM required for most)
python -m tender_extraction.ingestion
python -m tender_extraction.table_extraction
python -m tender_extraction.chunking
python -m tender_extraction.retrieval
python -m tender_extraction.schemas
python -m tender_extraction.validation
```

## Dependencies

- pdfplumber -- PDF text and table extraction
- python-docx -- DOCX document loading
- pytesseract + Pillow + pdf2image -- OCR pipeline
- rank-bm25 -- BM25 keyword retrieval
- sentence-transformers -- Semantic embeddings (all-MiniLM-L6-v2)
- faiss-cpu -- Vector similarity search
- llama-cpp-python -- LLM inference (Mistral-7B GGUF)
- pydantic -- Output schema validation
- rapidfuzz -- Fuzzy string matching for grounding
- tiktoken -- Token counting

## License

MIT
