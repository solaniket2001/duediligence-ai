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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
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
            embedding_function=self.embeddings,
        )

        # TOWER 2: SPARSE BM25 ENGINE (Exact Keyword Matching)
        logger.info("Initializing BM25 Sparse Engine...")
        db_data = self.vector_store.get(include=["documents", "metadatas"])
        self.all_documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(db_data["documents"], db_data["metadatas"])
        ]

        if self.all_documents:
            self.bm25 = BM25Retriever.from_documents(self.all_documents)
            self.bm25.k = 5
        else:
            self.bm25 = None

        # THE JUDGE: CROSS-ENCODER RERANKER
        logger.info("Initializing FlashRank Cross-Encoder...")
        self.ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp")

        # HYDE GENERATOR: Fast LLM handle
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.llm = (
            ChatGoogleGenerativeAI(
                model="gemini-3.5-flash-lite",
                google_api_key=api_key,
                max_output_tokens=300,
            )
            if api_key
            else None
        )

    def generate_hypothetical_passage(self, query: str) -> str:
        """HyDE: Generates a synthetic SEC filing excerpt matching GAAP/SEC taxonomy."""
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

            synthetic_text = synthetic_doc.strip()
            logger.info("HyDE generated synthetic passage: %s", synthetic_text[:120] + "...")
            return synthetic_text
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
        logger.info("Retrieval Query: '%s' | HyDE: %s", query, enable_hyde)

        # Format Chroma Filter
        chroma_filter = None
        if metadata_filter:
            if len(metadata_filter) > 1:
                chroma_filter = {"$and": [{k: v} for k, v in metadata_filter.items()]}
            else:
                chroma_filter = metadata_filter

        # TOWER 1: Dense Search (Original Query)
        dense_docs = self.vector_store.similarity_search(query, k=top_k, filter=chroma_filter)

        # TOWER 1b: HyDE Dense Search (Document-to-Document matching)
        hyde_docs: List[Document] = []
        if enable_hyde:
            hypothetical_passage = self.generate_hypothetical_passage(query)
            if hypothetical_passage != query:
                hyde_docs = self.vector_store.similarity_search(
                    hypothetical_passage, k=top_k, filter=chroma_filter
                )

        # TOWER 2: Sparse Keyword Search (BM25)
        sparse_docs = self.bm25.invoke(query) if self.bm25 else []
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

        # MERGER: Deduplicate hits across Dense, HyDE, and Sparse pools
        unique_docs: Dict[str, Document] = {}
        for doc in dense_docs + hyde_docs + sparse_docs:
            doc_id = doc.metadata.get("document_id") or doc.page_content[:50]
            if doc_id not in unique_docs:
                unique_docs[doc_id] = doc

        combined_results = list(unique_docs.values())
        logger.info(
            "Multi-Tower Merged %d Standard Dense + %d HyDE + %d Sparse into %d unique candidates.",
            len(dense_docs),
            len(hyde_docs),
            len(sparse_docs),
            len(combined_results),
        )

        if not combined_results:
            logger.warning("No matches found across any retrieval tower.")
            return []

        # THE JUDGE: Cross-Encoder Reranking
        passages = [
            {"id": str(idx), "text": doc.page_content, "meta": doc.metadata}
            for idx, doc in enumerate(combined_results)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self.ranker.rerank(rerank_request)

        if not reranked_results:
            return []

        top_score = reranked_results[0]["score"]
        logger.info("Cross-Encoder Top Score was: %.4f", top_score)

        # Adaptive Score Cliffing
        final_results = [
            res
            for res in reranked_results
            if (top_score - res["score"]) <= score_cliff
        ]

        return final_results[:rerank_top_k]