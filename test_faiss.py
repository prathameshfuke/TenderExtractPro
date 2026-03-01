import faiss
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_faiss():
    try:
        logger.info("Loading SentenceTransformer model...")
        model = SentenceTransformer("all-mpnet-base-v2")
        logger.info("Model loaded. Generating embeddings...")
        
        texts = ["Document 1", "Document 2", "Another test doc"]
        embeddings = model.encode(texts, convert_to_numpy=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        logger.info("Initializing FAISS index...")
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        
        logger.info("Adding embeddings to FAISS index...")
        index.add(embeddings)
        
        logger.info("FAISS index size: %d", index.ntotal)
        
        query = "Test document"
        logger.info("Encoding query...")
        query_emb = model.encode([query], convert_to_numpy=True)
        query_emb = np.array(query_emb, dtype=np.float32)
        
        logger.info("Searching FAISS index...")
        D, I = index.search(query_emb, 2)
        
        logger.info("Search results:")
        for idx in I[0]:
            logger.info(" - %s", texts[idx])
            
        logger.info("FAISS test successful!")
    except Exception as e:
        logger.error("FAISS test failed: %s", e)
        raise

if __name__ == "__main__":
    test_faiss()
