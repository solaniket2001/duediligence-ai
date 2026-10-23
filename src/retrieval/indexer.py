import argparse
import gc
import json
import logging
from pathlib import Path

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.retrieval.chroma_lock import chroma_lock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vector_indexer")


def build_index(ticker: str = None):
    processed_dir = Path("data/processed")
    chroma_dir = Path("data/chroma")
    chroma_dir.mkdir(parents=True, exist_ok=True)

    if not processed_dir.exists():
        logger.error("No processed directory found at %s.", processed_dir)
        return

    if ticker:
        ticker_upper = ticker.strip().upper()
        ticker_subfolder = processed_dir / ticker_upper
        if ticker_subfolder.exists():
            json_files = sorted(ticker_subfolder.glob("*.json"))
            logger.info("Reading %d JSON files from folder: %s", len(json_files), ticker_subfolder)
        else:
            json_files = sorted(processed_dir.glob(f"*{ticker_upper}*.json"))
            logger.info("Reading %d flat JSON files matching ticker '%s'", len(json_files), ticker_upper)
    else:
        json_files = sorted(processed_dir.rglob("*.json"))
        logger.info("Full mode: Found %d total JSON files across all folders.", len(json_files))

    if not json_files:
        logger.warning("No JSON files found to index.")
        return

    logger.info("Initializing FastEmbed Embeddings (BAAI/bge-small-en-v1.5)...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    with chroma_lock(chroma_dir):
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=str(chroma_dir),
        )



        BATCH_SIZE = 25
        total_batches = (len(json_files) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(0, len(json_files), BATCH_SIZE):
            batch_files = json_files[i : i + BATCH_SIZE]
            documents = []

            for file_path in batch_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        doc = Document(
                            page_content=data["content"],
                            metadata={
                                "document_id": data.get("document_id", ""),
                                "ticker": str(data.get("ticker", "")).upper(),
                                "year": str(data.get("year", "")),
                                "chunk_type": data.get("chunk_type", ""),
                            },
                        )
                        documents.append(doc)
                except Exception as e:
                    logger.error("Failed to read %s: %s", file_path.name, e)

            current_batch = (i // BATCH_SIZE) + 1
            logger.info(
                "Embedding batch %d of %d (%d documents)...",
                current_batch,
                total_batches,
                len(documents),
            )

            if documents:
                vectorstore.add_documents(documents)
            del documents
            gc.collect()

    logger.info("Success! ChromaDB saved to disk at: %s", chroma_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index SEC filing JSONs into ChromaDB.")
    parser.add_argument("--ticker", type=str, default=None, help="Specific ticker to index.")
    args = parser.parse_args()

    build_index(ticker=args.ticker)