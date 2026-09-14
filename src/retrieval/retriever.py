import logging
from pathlib import Path
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
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
        
        # TOWER 1: DENSE VECTOR ENGINE (Semantic Meaning)
        
        logger.info("Initializing Dense Embeddings Engine...")
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        self.vector_store = Chroma(
            persist_directory=str(persist_dir), 
            embedding_function=self.embeddings
        )
        
        # TOWER 2: SPARSE BM25 ENGINE (Exact Keyword Matching)
        
        logger.info("Initializing BM25 Sparse Engine...")
        
        # Pull the raw documents directly from Chroma to ensure perfectly synced data
        
        db_data = self.vector_store.get(include=['documents', 'metadatas'])
        self.all_documents = [
            Document(page_content=doc, metadata=meta) 
            for doc, meta in zip(db_data['documents'], db_data['metadatas'])
        ]
        
        if self.all_documents:
            self.bm25 = BM25Retriever.from_documents(self.all_documents)
            self.bm25.k = 5
        else:
            self.bm25 = None
            
        # THE JUDGE: CROSS-ENCODER RERANKER
        
        logger.info("Initializing FlashRank Cross-Encoder...")
        self.ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp")

    def search(self, query: str, top_k: int = 5, rerank_top_k: int = 1, metadata_filter: dict = None, score_cliff: float = 0.15):
        logger.info(f"Query: '{query}'")
        
        # Format Chroma Filter for multi-key queries
        chroma_filter = None
        if metadata_filter:
            if len(metadata_filter) > 1:
                chroma_filter = {"$and": [{k: v} for k, v in metadata_filter.items()]}
            else:
                chroma_filter = metadata_filter

        # TOWER 1: Execute Dense Search
        dense_docs = self.vector_store.similarity_search(query, k=top_k, filter=chroma_filter)
        
        # TOWER 2: Execute Sparse Search
        sparse_docs = self.bm25.invoke(query) if self.bm25 else []
        
        # Manually enforce metadata filters on the BM25 results
        if metadata_filter:
            filtered_sparse = []
            for doc in sparse_docs:
                match = True
                for k, v in metadata_filter.items():
                    if doc.metadata.get(k) != v:
                        match = False
                        break
                if match:
                    filtered_sparse.append(doc)
            sparse_docs = filtered_sparse
            
        # MERGER: Combine and Deduplicate hits based on document_id
        unique_docs = {}
        for doc in dense_docs + sparse_docs:
            doc_id = doc.metadata.get("document_id")
            if doc_id not in unique_docs:
                unique_docs[doc_id] = doc
                
        combined_results = list(unique_docs.values())
        logger.info(f"Hybrid Search merged {len(dense_docs)} dense + {len(sparse_docs)} sparse hits into {len(combined_results)} unique candidates.")
        
        if not combined_results:
            logger.warning("No matches found in Hybrid Search.")
            return []
            
        # THE JUDGE: Cross-Encoder Reranking
        passages = []
        for idx, doc in enumerate(combined_results):
            passages.append({
                "id": str(idx),
                "text": doc.page_content,
                "meta": doc.metadata
            })
            
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self.ranker.rerank(rerank_request)
        
        if not reranked_results:
            return []
            
        # Adaptive Score Cliffing
        top_score = reranked_results[0]['score']
        logger.info(f"Cross-Encoder Top Score was: {top_score:.4f}")
        
        final_results = []
        for res in reranked_results:
            if (top_score - res['score']) <= score_cliff:
                final_results.append(res)
                
        return final_results[:rerank_top_k]