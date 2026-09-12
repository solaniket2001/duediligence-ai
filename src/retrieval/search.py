import logging
from pathlib import Path
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("vector_search")

def search_financial_data(query: str, persist_dir: Path, k: int = 1):
    """
    Connects to the local ChromaDB, embeds the user query, 
    and returns the most mathematically similar financial table.
    """
    if not persist_dir.exists():
        logger.error("ChromaDB not found. Please run indexer.py first.")
        return

    logger.info("Initializing Local Embeddings Engine...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    logger.info(f"Connecting to Vector Database at {persist_dir}...")
    vector_store = Chroma(
        persist_directory=str(persist_dir), 
        embedding_function=embeddings
    )
    
    logger.info(f"Searching for: '{query}'")
    
    # Perform the similarity search
    # k=1 means we only want the single most relevant table
    results = vector_store.similarity_search(query, k=k)
    
    if not results:
        logger.warning("No results found.")
        return
        
    print("\n" + "="*60)
    print("🎯 VECTOR SEARCH RESULT MATCH 🎯")
    print("="*60)
    
    for idx, doc in enumerate(results):
        print(f"\nMatch {idx+1}:")
        print(f"Document ID: {doc.metadata.get('document_id')}")
        print(f"Filing Type: {doc.metadata.get('filing_type')} | Year: {doc.metadata.get('year', 'N/A')}")
        print("-" * 60)
        print(doc.page_content)
        print("-" * 60)

if __name__ == "__main__":
    CHROMA_DB_DIR = Path("data/chroma")
    
    # We ask the database a plain English question
    user_query = "What were the net sales and revenue for Apple's products like the iPhone, Mac, and iPad?"
    
    search_financial_data(user_query, CHROMA_DB_DIR)