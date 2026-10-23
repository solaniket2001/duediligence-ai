import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from flashrank import Ranker, RerankRequest
from src.retrieval.chroma_lock import chroma_lock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("enterprise_retriever")

_SHARED_EMBEDDINGS: Optional[FastEmbedEmbeddings] = None
_SHARED_RANKER: Optional[Ranker] = None


def get_shared_embeddings() -> FastEmbedEmbeddings:
    global _SHARED_EMBEDDINGS
    if _SHARED_EMBEDDINGS is None:
        logger.info("Initializing Dense Embeddings Engine (Singleton)...")
        _SHARED_EMBEDDINGS = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _SHARED_EMBEDDINGS


def get_shared_ranker() -> Ranker:
    global _SHARED_RANKER
    if _SHARED_RANKER is None:
        logger.info("Initializing FlashRank Cross-Encoder (Singleton)...")
        _SHARED_RANKER = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp")
    return _SHARED_RANKER


class SECRetriever:
    def __init__(self, persist_dir: Path):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir = persist_dir

        self.embeddings = get_shared_embeddings()
        self.ranker = get_shared_ranker()

        self.vector_store = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=self.embeddings,
        )

        self.all_documents: List[Document] = []
        self.bm25: Optional[BM25Retriever] = None
        self.refresh_sparse_index()

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.llm = (
            ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
                google_api_key=api_key,
                max_output_tokens=300,
                max_retries=1,
            )
            if api_key
            else None
        )

    def refresh_sparse_index(self) -> None:
        """Reloads ChromaDB client & HNSW segments from disk after JIT subprocess writes."""
        with chroma_lock(self.persist_dir):
            try:
                from chromadb.api.client import SharedSystemClient
                SharedSystemClient.clear_system_cache()
            except Exception:
                pass
            self.vector_store = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings,
            )
            db_data = self.vector_store.get(include=["documents", "metadatas"])
        docs = db_data.get("documents") or []
        metas = db_data.get("metadatas") or []
        self.all_documents = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(docs, metas)
            if doc
        ]
        if self.all_documents:
            self.bm25 = BM25Retriever.from_documents(self.all_documents)
            self.bm25.k = 5
        else:
            self.bm25 = None

    def generate_hypothetical_passage(self, query: str) -> str:
        if not self.llm:
            return query

        system_prompt = (
            "You are an SEC EDGAR filing drafter. Write a concise, 2-3 sentence hypothetical excerpt "
            "from an official SEC Form 10-K or 20-F disclosure that directly answers the user's research query. "
            "Use formal US GAAP terminology, accounting phrasing, and regulatory boilerplate. "
            "Do NOT include conversational filler, headings, or disclaimers."
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Query: {query}"),
            ])
            raw_content = response.content
            if isinstance(raw_content, list):
                synthetic_doc = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in raw_content
                )
            else:
                synthetic_doc = str(raw_content)

            return synthetic_doc.strip() or query
        except Exception as e:
            logger.warning("HyDE generation failed (%s). Falling back to raw query.", e)
            return query

    def search(
        self,
        query: str,
        top_k: int = 5,
        rerank_top_k: int = 1,
        metadata_filter: Optional[dict] = None,
        score_cliff: float = 0.15,
        enable_hyde: bool = True,
    ) -> List[Dict[str, Any]]:
        logger.info("Retrieval Query: '%s' | Filter: %s | HyDE: %s", query, metadata_filter, enable_hyde)

        chroma_filter = None
        if metadata_filter:
            if len(metadata_filter) > 1:
                chroma_filter = {"$and": [{k: v} for k, v in metadata_filter.items()]}
            else:
                chroma_filter = metadata_filter

        with chroma_lock(self.persist_dir):
            try:
                dense_docs = self.vector_store.similarity_search(query, k=top_k, filter=chroma_filter)
            except Exception as chroma_exc:
                logger.warning("Chroma HNSW cache stale (%s); reloading client from disk...", chroma_exc)
                try:
                    from chromadb.api.client import SharedSystemClient
                    SharedSystemClient.clear_system_cache()
                except Exception:
                    pass
                self.vector_store = Chroma(
                    persist_directory=str(self.persist_dir),
                    embedding_function=self.embeddings,
                )
                dense_docs = self.vector_store.similarity_search(query, k=top_k, filter=chroma_filter)

            hyde_docs: List[Document] = []
            if enable_hyde:
                hypothetical_passage = self.generate_hypothetical_passage(query)
                if hypothetical_passage != query:
                    hyde_docs = self.vector_store.similarity_search(
                        hypothetical_passage, k=top_k, filter=chroma_filter
                    )

        sparse_docs = self.bm25.invoke(query) if self.bm25 else []
        if metadata_filter:
            filtered_sparse = []
            for doc in sparse_docs:
                if all(doc.metadata.get(k) == v for k, v in metadata_filter.items()):
                    filtered_sparse.append(doc)
            sparse_docs = filtered_sparse

        unique_docs: Dict[str, Document] = {}
        for doc in dense_docs + hyde_docs + sparse_docs:
            doc_id = doc.metadata.get("document_id") or doc.page_content[:50]
            if doc_id not in unique_docs:
                unique_docs[doc_id] = doc

        combined_results = list(unique_docs.values())
        if not combined_results:
            logger.warning("No matches found across any retrieval tower.")
            return []

        passages = [
            {"id": str(idx), "text": doc.page_content, "meta": doc.metadata}
            for idx, doc in enumerate(combined_results)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self.ranker.rerank(rerank_request)
        if not reranked_results:
            return []

        top_score = reranked_results[0]["score"]
        final_results = [
            res
            for res in reranked_results
            if (top_score - res["score"]) <= score_cliff
        ]

        return final_results[:rerank_top_k]