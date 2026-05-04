<h1 align="center">
  <span style="color: #2b6cb0">Tender</span><span style="color: #2d3748">ExtractPro</span>
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/React-18-61dafb?style=for-the-badge&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-Mistral%207B-orange?style=for-the-badge" />
</p>

A production-grade RAG pipeline for extracting technical specifications and scope of work from tender documents. Processes real PDF, DOCX, and image files through a 6-stage pipeline: ingestion, table extraction, chunking, hybrid retrieval, LLM extraction, and grounding validation. 

<span style="color: #e53e3e; font-weight: bold;">NEW:</span> Includes an LLM-powered <strong>Scoring and Ranking</strong> mechanism to evaluate the extracted tender against a customized Company Profile to determine match score and cost feasibility.

## Interface Overview

<div align="center">
  <p><strong>Upload & Processing</strong></p>
  <img src="assets/upload_processing.png" width="80%" alt="Upload and Processing" />
  <br><br>
  
  <p><strong>Extracted Specifications</strong></p>
  <img src="assets/extracted_specs.png" width="80%" alt="Extracted Specifications" />
  <br><br>

  <p><strong>Detailed Component View</strong></p>
  <img src="assets/spec_drawer.png" width="80%" alt="Specification Details Drawer" />
  <br><br>

  <p><strong>Source Grounding & Evidence Match</strong></p>
  <img src="assets/evidence_match.png" width="45%" alt="Evidence Match" />
  <img src="assets/parameter_map.png" width="45%" alt="Parameter Map" />
  <br><br>

  <p><strong>Match Score & Ranking Analysis</strong></p>
  <img src="assets/tender_match_analysis.png" width="80%" alt="Match Score Analysis" />
  <br><br>

  <p><strong>Interactive Document Q&A with AI Assistant</strong></p>
  <img src="assets/ask_document_chat.png" width="80%" alt="Ask Document Q&A" />
  <img src="assets/document_qa.png" width="80%" alt="Document Q&A Interface" />
</div>

## Architecture

```mermaid
graph TD
    A[Input Document<br>PDF/DOCX/Image] --> B[1. Ingestion<br>pdfplumber + Tesseract OCR fallback]
    B --> C[2. Table Extraction<br>pdfplumber dual-strategy]
    C --> D[3. Chunking<br>Section-aware hierarchical chunking]
    D --> E[4. Hybrid Retrieval<br>BM25 + FAISS semantic search]
    E --> F[5. LLM Extraction<br>Mistral-7B / Phi-3 via llama-cpp-python]
    F --> G[6. Validation<br>rapidfuzz grounding verification]
    G --> H[7. Scoring & Ranking<br>Match evaluation against Company Profile]
    H --> I[Structured JSON Output & Match Score]
    
    style A fill:#2d3748,color:#fff
    style I fill:#2b6cb0,color:#fff
    style B fill:#4a5568,color:#fff
    style C fill:#4a5568,color:#fff
    style D fill:#4a5568,color:#fff
    style E fill:#4a5568,color:#fff
    style F fill:#4a5568,color:#fff
    style G fill:#4a5568,color:#fff
    style H fill:#4a5568,color:#fff
```

## Quick Start

### 1. Setup Backend
```bash
# Setup environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run the API server
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Setup Frontend
```bash
cd frontend
npm install
npm run dev
```

See [SETUP.md](SETUP.md) for detailed installation instructions including Tesseract, Poppler, and model download.

## Features

### Extracted Tender Elements
The pipeline produces a JSON file containing:
- **Technical Specifications**: Components, specs, and constraints.
- **Scope of Work**: Summaries, deliverables, exclusions, and locations.
- Every extracted value includes a source citation pointing back to the exact chunk and page in the original document.

### Match Scoring & Ranking
You can define a **Company Profile** (via the UI or `company_profile.json`) specifying capabilities, budget constraints, and operational exclusions. The pipeline evaluates the tender's requirements against this profile to output:
- **Match Score (0-100)**: Quantitative alignment score.
- **Cost Feasibility (High/Medium/Low)**: Budget match based on the company's financial capabilities.
- **Strategic Reasoning**: Paragraph outlining the reasoning and any potential red flags.

### Anti-Hallucination Safeguards
1. **Prompt Engineering**: The LLM prompt explicitly instructs "use NOT_FOUND for missing fields, NEVER invent values" and requires source citations.
2. **Grounding Verification**: After LLM extraction, every spec is fuzzy-matched against source chunks using `rapidfuzz`. Specs with grounding score below 0.40 are rejected.
3. **Pydantic Validation**: Output is validated against strict Pydantic v2 models.

## Advanced RAG Enhancements (Inspired by SimpleRAG)

1. **Semantic Chunking**
   - **Logic**: Replaced fixed-size splitting with `SemanticChunker` (from `langchain-experimental`). It identifies natural breakpoints based on embedding distances, ensuring that paragraphs are only split when the meaning changes.
   - **Optimization**: Implemented lazy-loading for the chunking embedding model to prevent high latency and memory churn during document processing.

2. **Parent-Child Retrieval**
   - **Logic**: Implemented a sophisticated indexing strategy. Small "child" spans (~200 words) are indexed in Qdrant for high-precision semantic matching, while the full "parent" chunks are returned to the LLM.
   - **Benefit**: This provides the LLM with much richer context than standard chunking while maintaining pinpoint accuracy during retrieval.

3. **Enhanced Extraction & Grounding**
   - **Prompts**: Refined system and user prompts to use professional "Tender Analyst" personas with strict verbatim extraction rules.
   - **JSON Repair**: Upgraded the JSON repair logic to handle common LLM failure modes like unescaped newlines and markdown fences more robustly.

4. **Environment & Stability Fixes**
   - **Virtual Environment**: Ensured all dependencies are correctly isolated in the project's local `venv`.
   - **LLM Fix**: Resolved the `PrefetchVirtualMemory` error on Windows by disabling `mmap` during model loading.
   - **Dependencies**: Updated `requirements.txt` to include `langchain` and related libraries.

## Project Structure

```
TenderExtractPro/
  api/
    main.py             -- FastAPI server with extraction and scoring endpoints
  frontend/             -- React + Vite User Interface
  tender_extraction/
    config.py           -- Centralized configuration
    schemas.py          -- Pydantic v2 models for structured output
    scoring.py          -- LLM matching logic for company profile
    extraction.py       -- LLM-powered specification extraction
    main.py             -- Pipeline orchestration and CLI
  company_profile.json  -- Active company profile configurations
  dataset/              -- Real tender PDF files for testing
  models/               -- LLM model files (not in version control)
```

## Configuration

All tunable parameters are centralized in `tender_extraction/config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `chunking.max_chunk_tokens` | 400 | Maximum tokens per text chunk |
| `retrieval.bm25_weight` | 0.4 | BM25 weight in score fusion |
| `retrieval.embedding_weight` | 0.6 | Embedding weight in score fusion |
| `llm.temperature` | 0.1 | LLM generation temperature |
| `validation.min_grounding_ratio` | 0.40 | Minimum grounding score to accept a spec |

## Contributors

A big thank you to our contributors! 🙌

- **[Gaurav Varu](https://github.com/gauravvaru)** - Fix: Preserve filtered components in scoring logic ([#1](https://github.com/prathameshfuke/TenderExtractPro/pull/1))

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for more details.

## License

MIT
