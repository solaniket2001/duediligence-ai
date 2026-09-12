import logging
from pathlib import Path
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from flashrank import Ranker, RerankRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("enterprise_retriever")

class SECRetriever:
    def __init__(self, persist_dir: Path):
        if not persist_dir.exists():
            raise FileNotFoundError(f"ChromaDB not found at {persist_dir}")
        
        logger.info("Initializing Dense Embeddings Engine (Stage 1)...")
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        self.vector_store = Chroma(
            persist_directory=str(persist_dir), 
            embedding_function=self.embeddings
        )
        
        logger.info("Initializing FlashRank Cross-Encoder (Stage 2)...")
        self.ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp")

    def search(self, query: str, top_k: int = 5, rerank_top_k: int = 1, metadata_filter: dict = None, score_cliff: float = 0.15):
        logger.info(f"Query: '{query}'")
        
        # STAGE 1: Dense Retrieval (Fetch top 5 candidates)
        initial_results = self.vector_store.similarity_search_with_relevance_scores(
            query, 
            k=top_k,
            filter=metadata_filter
        )
        
        if not initial_results:
            logger.warning("No semantic matches found.")
            return []
            
        logger.info(f"Stage 1 returned {len(initial_results)} candidates. Passing to Cross-Encoder...")
        
        # Format candidates for FlashRank
        passages = []
        for idx, (doc, _) in enumerate(initial_results):
            passages.append({
                "id": str(idx),
                "text": doc.page_content,
                "meta": doc.metadata
            })
            
        # STAGE 2: Cross-Encoder Reranking
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self.ranker.rerank(rerank_request)
        
        if not reranked_results:
            return []
            
        # THE FIX: Adaptive Score Cliff Detection
        top_score = reranked_results[0]['score']
        logger.info(f"Cross-Encoder Top Score was: {top_score:.4f}")
        
        final_results = []
        for res in reranked_results:
            # We keep the chunk ONLY if it is mathematically close to the top result
            if (top_score - res['score']) <= score_cliff:
                final_results.append(res)
                
        final_results = final_results[:rerank_top_k]
        
        if not final_results:
            logger.warning("Matches found, but rejected by score cliff detection.")
            
        return final_results

if __name__ == "__main__":
    CHROMA_DB_DIR = Path("data/chroma")
    user_query = "What were the net sales and revenue for Apple's products like the iPhone, Mac, and iPad?"
    
    try:
        retriever = SECRetriever(CHROMA_DB_DIR)
        
        # We explicitly filter for the year 2025 and use a 0.15 score cliff
        results = retriever.search(
            query=user_query,
            top_k=5,
            rerank_top_k=1,
            metadata_filter={"year": "2025"},
            score_cliff=0.15
        )
        
        print("\n" + "="*60)
        print("🎯 ENTERPRISE RERANKED RESULT 🎯")
        print("="*60)
        
        for idx, res in enumerate(results):
            meta = res['meta']
            print(f"\nMatch {idx+1} | Confidence Score: {res['score']:.4f}")
            print(f"Document ID: {meta.get('document_id')}")
            print(f"Filing Type: {meta.get('filing_type')} | Year: {meta.get('year')}")
            print("-" * 60)
            print(res['text'])
            print("-" * 60)
            
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")