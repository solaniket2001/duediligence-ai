import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Ensure Python can find our custom retriever module
sys.path.append(os.getcwd())
from src.retrieval.retriever import SECRetriever

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("financial_agent")

# Securely load environment variables
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please add it.")

# LangChain defaults to looking for GOOGLE_API_KEY, so we map it here automatically
os.environ["GOOGLE_API_KEY"] = gemini_key

def run_financial_analysis(query: str, year: str = "2025"):
    """Orchestrates the RAG pipeline: Retrieval + LLM Generation"""
    
    # STAGE 1: RETRIEVAL (Phase 2)
    db_dir = Path("data/chroma")
    retriever = SECRetriever(db_dir)
    
    logger.info("Fetching relevant financial data...")
    retrieved_docs = retriever.search(
        query=query, 
        top_k=5, 
        rerank_top_k=1, 
        metadata_filter={"year": year},
        score_cliff=0.15
    )
    
    if not retrieved_docs:
        logger.warning("No data retrieved. Cannot perform analysis.")
        return "I could not find relevant financial data in the database to answer this question."
        
    # Combine the retrieved tables into a single context string
    context = "\n\n".join([doc['text'] for doc in retrieved_docs])
    
    # STAGE 2: GENERATION (Phase 3)
    logger.info("Initializing Google Gemini Engine...")
    
    # Initialize the 3.1 model with max_retries for native 503/429 handling
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.0, 
        max_output_tokens=1024,
        max_retries=1
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an elite financial analyst. "
         "Answer the user's question using ONLY the provided context. "
         "Do not hallucinate or use outside knowledge. "
         "Present the data clearly, format numbers professionally, and highlight key year-over-year changes if present.\n\n"
         "CONTEXT:\n{context}"),
        ("human", "{query}")
    ])
    
    # Build the LangChain Pipeline (LCEL) with jittered exponential backoff
    llm_with_retry = llm.with_retry(
        stop_after_attempt=1,
        wait_exponential_jitter=True
    )
    
    chain = prompt | llm_with_retry | StrOutputParser()
    
    logger.info("Generating financial analysis...")
    response = chain.invoke({"context": context, "query": query})
    
    print("\n" + "="*60)
    print("📊 AI FINANCIAL ANALYSIS 📊")
    print("="*60)
    print(response)
    print("="*60)

if __name__ == "__main__":
    test_query = "What were the net sales and revenue for Apple's products like the iPhone, Mac, and iPad?"
    run_financial_analysis(test_query, year="2025")