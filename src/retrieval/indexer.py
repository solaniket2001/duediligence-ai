import os
import json
import logging
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma

# Configure Enterprise Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("vector_indexer")

def index_processed_data(processed_dir: Path, persist_dir: Path):
    if not processed_dir.exists():
        logger.error(f"Processed directory not found: {processed_dir}")
        return
        
    json_files = list(processed_dir.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found to index.")
        return

    logger.info(f"Found {len(json_files)} JSON files. Preparing documents...")
    
    documents = []
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            content = data.pop("content")
            metadata = data 
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
            
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")

    # THE PIVOT: Local, free embeddings instead of OpenAI
    
    logger.info("Initializing FastEmbed Embeddings (BAAI/bge-small-en-v1.5)...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    logger.info(f"Embedding {len(documents)} financial tables and compiling database locally...")
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_dir)
    )
    
    logger.info(f"Success! ChromaDB saved to disk at: {persist_dir}")

if __name__ == "__main__":
    PROCESSED_DATA_DIR = Path("data/processed")
    CHROMA_DB_DIR = Path("data/chroma")
    
    index_processed_data(PROCESSED_DATA_DIR, CHROMA_DB_DIR)