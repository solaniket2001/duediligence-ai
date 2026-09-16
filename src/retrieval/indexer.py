import json
import logging
import gc
from pathlib import Path
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("vector_indexer")

def build_index():
    # Adjust this to match where your JSON files are saved (e.g., data/processed or data/json_chunks)
    json_dir = Path("data/processed") 
    chroma_dir = Path("data/chroma")
    
    if not json_dir.exists():
        logger.error(f"No processed data found in {json_dir}.")
        return

    json_files = list(json_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files. Preparing documents...")

    if len(json_files) == 0:
        logger.warning("No files to index.")
        return

    # Initialize Embeddings
    logger.info("Initializing FastEmbed Embeddings (BAAI/bge-small-en-v1.5)...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # Initialize ChromaDB connection
    vectorstore = Chroma(
        embedding_function=embeddings, 
        persist_directory=str(chroma_dir),
        collection_metadata={"hnsw:sync_threshold": 10} # Forces HNSW segment to flush to disk safely!
    )
    
    # BATCHING LOGIC: Process 25 files at a time to prevent SIGTERM: 15 (Out of Memory)
    BATCH_SIZE = 25
    total_batches = (len(json_files) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(json_files), BATCH_SIZE):
        batch_files = json_files[i:i+BATCH_SIZE]
        documents = []
        
        for file_path in batch_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    doc = Document(
                        page_content=data['content'],
                        metadata={
                            "document_id": data.get('document_id', ''),
                            "ticker": data.get('ticker', ''),
                            "year": str(data.get('year', '')),
                            "chunk_type": data.get('chunk_type', '')
                        }
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to read {file_path.name}: {e}")
                
        current_batch = (i // BATCH_SIZE) + 1
        logger.info(f"Embedding batch {current_batch} of {total_batches} ({len(documents)} documents)...")
        
        # Add to database
        vectorstore.add_documents(documents)
        
        # Force Python to release the RAM used by this batch
        del documents
        gc.collect()

    logger.info(f"Success! ChromaDB saved to disk at: {chroma_dir}")

if __name__ == "__main__":
    build_index()