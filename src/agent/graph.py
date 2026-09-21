import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, TypedDict

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from duckduckgo_search import DDGS

from src.retrieval.retriever import SECRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("graph_engine")

SCORE_CONFIDENCE_THRESHOLD = 0.80
MAX_RETRIEVAL_RETRIES = 2


# --- State Definition ---
class DueDiligenceState(TypedDict, total=False):
    ticker: str
    company_name: str
    peer_ticker: Optional[str]
    peer_name: Optional[str]
    financial_query: Optional[str]
    financial_score: float
    financial_retry_count: int
    financial_context: List[str]
    risk_context: List[str]
    peer_context: List[str]
    quantitative_analysis: str
    news_context: str
    final_memo: str


# --- Dynamic Retriever Factory ---
def get_retriever() -> SECRetriever:
    """Instantiates a fresh SECRetriever handle dynamically."""
    return SECRetriever(Path("data/chroma"))


def fetch_context_chunks_with_score(
    query: str,
    ticker: str,
    top_k: int = 5,
) -> Tuple[List[str], float]:
    """Retrieve context strictly filtered by ticker and return chunks alongside top cross-encoder score."""
    retriever = get_retriever()
    requested_ticker = ticker.strip().upper()

    hits = retriever.search(
        query=query,
        top_k=max(top_k * 5, 25),
        rerank_top_k=top_k,
        metadata_filter={"ticker": requested_ticker},
    )

    if not hits:
        logger.warning("No retrieval results matched ticker=%s.", requested_ticker)
        return [], 0.0

    chunks: List[str] = []
    top_score = 0.0

    for hit in hits:
        if isinstance(hit, dict):
            metadata = hit.get("meta", {}) or hit.get("metadata", {})
            content = hit.get("text", hit.get("page_content", str(hit)))
            score = hit.get("score") or metadata.get("score") or metadata.get("relevance_score") or 0.0
        else:
            metadata = getattr(hit, "metadata", {})
            content = getattr(hit, "page_content", str(hit))
            score = getattr(hit, "score", None) or metadata.get("score") or metadata.get("relevance_score") or 0.0

        try:
            score_val = float(score)
            if score_val > top_score:
                top_score = score_val
        except (ValueError, TypeError):
            pass

        document_id = metadata.get("document_id", "DOC_UNKNOWN")
        chunks.append(f"[{document_id}]\n{content}")

    return chunks, top_score


def fetch_context_chunks(
    query: str,
    ticker: str,
    top_k: int = 5,
) -> List[str]:
    """Backward-compatible helper returning only chunk strings."""
    chunks, _ = fetch_context_chunks_with_score(query=query, ticker=ticker, top_k=top_k)
    return chunks


def _resolve_peer(state: DueDiligenceState) -> Tuple[Optional[str], Optional[str]]:
    """Support both peer_ticker/peer_name and common UI aliases."""
    peer_ticker = (
        state.get("peer_ticker")
        or state.get("peer")
        or state.get("peer_symbol")
    )
    peer_name = state.get("peer_name") or state.get("peer_company_name")

    if peer_ticker:
        peer_ticker = str(peer_ticker).strip().upper()

    return peer_ticker, peer_name


def get_llm() -> ChatGoogleGenerativeAI:
    """Configures the synthesis LLM."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        max_output_tokens=4096,
    )


# --- Graph Nodes ---
def extract_financials(state: DueDiligenceState) -> dict:
    current_query = state.get("financial_query")
    if not current_query:
        # Explicitly search for financial statement line items
        current_query = (
            f"Consolidated Statements of Operations Total revenues Automotive sales "
            f"Total cost of revenues Gross profit Net income {state['ticker']}"
        )

    logger.info(
        "Agent 1: Extracting Financial Data for %s (%s) [Query: '%s']...",
        state["company_name"],
        state["ticker"],
        current_query,
    )

    chunks, top_score = fetch_context_chunks_with_score(
        current_query,
        ticker=state["ticker"],
        top_k=5,
    )

    logger.info("Agent 1: Top Cross-Encoder score achieved: %.4f", top_score)

    return {
        "financial_context": chunks,
        "financial_score": top_score,
        "financial_query": current_query,
    }


def rewrite_financial_query(state: DueDiligenceState) -> dict:
    llm = get_llm()
    current_query = state.get("financial_query", "")
    score = state.get("financial_score", 0.0)
    retries = state.get("financial_retry_count", 0)

    logger.warning(
        "Reflection Node Triggered: Retrieval score (%.4f) below threshold (%.2f). Refection attempt %d/%d.",
        score,
        SCORE_CONFIDENCE_THRESHOLD,
        retries + 1,
        MAX_RETRIEVAL_RETRIES,
    )

    prompt = f"""You are an SEC EDGAR retrieval optimization specialist.
The following query generated insufficient relevance (FlashRank Score: {score:.4f}) against 10-K/20-F filings for {state['company_name']} ({state['ticker']}):
Query: "{current_query}"

Rewrite this search query to use formal US GAAP / SEC reporting taxonomy (e.g., segment revenues, disaggregated revenue disclosures, statements of operations line items, sales by reportable segment).
Return ONLY the raw rewritten search query text without quotation marks or explanations."""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw_content = response.content
    if isinstance(raw_content, list):
        rewritten_text = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in raw_content
        )
    else:
        rewritten_text = str(raw_content)

    refined_query = rewritten_text.strip().replace('"', "")
    logger.info("Reflection complete. Generated refined query: '%s'", refined_query)

    return {
        "financial_query": refined_query,
        "financial_retry_count": retries + 1,
    }


def should_refine_financials(state: DueDiligenceState) -> str:
    score = state.get("financial_score", 0.0)
    retries = state.get("financial_retry_count", 0)
    chunks = state.get("financial_context", [])

    if (score < SCORE_CONFIDENCE_THRESHOLD or not chunks) and retries < MAX_RETRIEVAL_RETRIES:
        return "rewrite_financial_query"
    return "extract_risks"


def extract_risks(state: DueDiligenceState) -> dict:
    logger.info("Agent 2: Extracting Risk Factors for %s...", state["company_name"])
    query = (
        "Item 1A Risk Factors business risks regulatory legal competition "
        f"market exposure {state['company_name']} {state['ticker']}"
    )
    chunks = fetch_context_chunks(query, ticker=state["ticker"], top_k=5)
    return {"risk_context": chunks}


def extract_peer(state: DueDiligenceState) -> dict:
    peer_id, peer_name = _resolve_peer(state)
    if not peer_id or peer_id in {"NONE", "N/A"}:
        logger.info("Agent 3: No peer specified. Skipping peer extraction.")
        return {"peer_context": []}

    target_label = f"{peer_name} ({peer_id})" if peer_name else peer_id
    logger.info("Agent 3: Extracting Peer Analysis for %s...", target_label)
    query = f"Consolidated Statements of Operations Total Net Sales Revenue {peer_name or peer_id} {peer_id}"
    chunks = fetch_context_chunks(query, ticker=peer_id, top_k=5)
    return {"peer_context": chunks}


def calculate_metrics(state: DueDiligenceState) -> dict:
    """Agent 3.5 (Math): Calculates derived quantitative ratios using Chain-of-Thought."""
    logger.info("Math Agent: Computing derived quantitative ratios and YoY growth...")
    llm = get_llm()

    financial_block = "\n\n".join(state.get("financial_context", []))
    peer_block = "\n\n".join(state.get("peer_context", []))

    system_prompt = (
        "You are a Quantitative Financial Analyst. Analyze the provided raw SEC financial tables for the target and peer. "
        "Use step-by-step reasoning to calculate derived metrics: "
        "Year-over-Year (YoY) revenue growth, gross margins, and operating margins if the data is available. "
        "CRITICAL FORMATTING RULES:\n"
        "- DO NOT use LaTeX syntax, backslashes, or \\frac{} expressions.\n"
        "- Write calculations in clean plain text (e.g., '(94,827 - 97,690) / 97,690 = -2.93%').\n"
        "- Format monetary amounts with 'USD' or '$' followed directly by numbers without backslashes (e.g., '$94,827 million').\n"
        "- If a metric cannot be calculated due to missing line items, explicitly state 'Insufficient data'."
    )

    user_prompt = f"""
    TARGET COMPANY: {state['company_name']}
    TARGET FINANCIALS:
    {financial_block or "No data available."}

    PEER FINANCIALS:
    {peer_block or "No data available."}

    Perform the quantitative analysis:
    """

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    raw_content = response.content
    analysis_text = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in raw_content) if isinstance(raw_content, list) else str(raw_content)
    
    return {"quantitative_analysis": analysis_text}


import feedparser

def fetch_live_news(state: DueDiligenceState) -> dict:
    """Agent 4 (News): Fetches live market news via RSS and synthesizes sentiment."""
    logger.info("Agent 4 (News): Fetching live market intelligence for %s...", state["company_name"])
    
    ticker = state["ticker"].upper().strip()
    feed_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    
    try:
        feed = feedparser.parse(feed_url)
        entries = feed.entries[:5]
        
        if not entries:
            logger.warning("Agent 4: No live RSS headlines found for %s.", ticker)
            return {"news_context": "No recent financial news detected for this ticker."}
            
        news_snippets = []
        for item in entries:
            title = getattr(item, "title", "Market Update")
            summary = getattr(item, "summary", "")
            news_snippets.append(f"Headline: {title}\nSummary: {summary}")
            
        raw_news = "\n---\n".join(news_snippets)
        
        llm = get_llm()
        prompt = (
            f"Analyze these recent financial news headlines for {state['company_name']} ({state['ticker']}):\n\n"
            f"{raw_news}\n\n"
            "Synthesize a concise 1-paragraph institutional 'Current Market Outlook' detailing "
            "prevailing sentiment (Bullish, Bearish, or Neutral) and near-term market catalysts."
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_content = response.content
        sentiment_text = "".join(
            chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            for chunk in raw_content
        ) if isinstance(raw_content, list) else str(raw_content)
        
        return {"news_context": sentiment_text.strip()}
        
    except Exception as e:
        logger.error("Agent 4 (News) failed: %s", e)
        return {"news_context": "Live market news extraction temporarily unavailable."}

def synthesize_memo(state: DueDiligenceState) -> dict:
    logger.info("Agent 5: Synthesizing Investment Memo for %s...", state["company_name"])
    llm = get_llm()

    financial_block = "\n\n".join(state.get("financial_context", []))
    risk_block = "\n\n".join(state.get("risk_context", []))
    peer_block = "\n\n".join(state.get("peer_context", []))
    quantitative_block = state.get("quantitative_analysis", "")
    news_block = state.get("news_context", "")

    peer_target = state.get("peer_name") or state.get("peer_ticker") or "None"

    system_prompt = (
        "You are a Lead Portfolio Manager writing an institutional investment memorandum. "
        "Adhere strictly to these rules:\n"
        "1. Every factual statement or financial metric must include an exact bracketed document citation (e.g., [DOC_ID]).\n"
        "2. ZERO HALLUCINATION: Ground claims exclusively in the provided context, Quantitative Analysis, and Live Market Outlook.\n"
        "3. ZERO CROSS-CONTAMINATION: Do not mix peer metrics with the target company's metrics.\n"
        "4. FORMATTING: DO NOT use LaTeX math equations or backslashes. Write all numbers and percentages as plain text (e.g., '$94,827 million', '18.03%').\n"
        "5. Structure your response into exactly five sections: "
        "1. EXECUTIVE SUMMARY, 2. FINANCIAL PERFORMANCE, 3. PEER COMPARISON, 4. KEY RISKS, 5. CURRENT MARKET OUTLOOK, followed by CONCLUSION / RECOMMENDATION.\n"
        "6. If peer context is empty, explicitly state that peer filing data is unavailable without speculating."
    )
    
    user_prompt = f"""
TARGET COMPANY: {state['company_name']} ({state['ticker']})
PEER TARGET: {peer_target}

=== FINANCIAL CONTEXT ===
{financial_block or "No specific financial tables retrieved."}

=== QUANTITATIVE ANALYSIS ===
{quantitative_block or "No derived metrics calculated."}

=== LIVE MARKET OUTLOOK ===
{news_block or "No live market intelligence retrieved."}

=== RISK CONTEXT ===
{risk_block or "No specific risk factors retrieved."}

=== PEER CONTEXT ===
{peer_block or "No peer documents retrieved."}

Please synthesize the comprehensive Investment Memorandum now:
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    raw_content = response.content
    memo_text = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in raw_content) if isinstance(raw_content, list) else str(raw_content)

    return {"final_memo": memo_text}


# --- Graph Construction ---
def build_due_diligence_graph():
    builder = StateGraph(DueDiligenceState)

    # Nodes
    builder.add_node("extract_financials", extract_financials)
    builder.add_node("rewrite_financial_query", rewrite_financial_query)
    builder.add_node("extract_risks", extract_risks)
    builder.add_node("extract_peer", extract_peer)
    builder.add_node("calculate_metrics", calculate_metrics)
    builder.add_node("fetch_live_news", fetch_live_news)
    builder.add_node("synthesize_memo", synthesize_memo)

    # Routing
    builder.add_edge(START, "extract_financials")
    builder.add_conditional_edges(
        "extract_financials",
        should_refine_financials,
        {
            "rewrite_financial_query": "rewrite_financial_query",
            "extract_risks": "extract_risks",
        },
    )
    builder.add_edge("rewrite_financial_query", "extract_financials")
    builder.add_edge("extract_risks", "extract_peer")
    
    # Sequential intelligence layer
    builder.add_edge("extract_peer", "calculate_metrics")
    builder.add_edge("calculate_metrics", "fetch_live_news")
    builder.add_edge("fetch_live_news", "synthesize_memo")
    builder.add_edge("synthesize_memo", END)

    return builder.compile()


def normalize_graph_input(payload: dict) -> DueDiligenceState:
    peer_ticker = payload.get("peer_ticker") or payload.get("peer") or payload.get("peer_symbol")
    peer_name = payload.get("peer_name") or payload.get("peer_company_name")

    return {
        "ticker": str(payload["ticker"]).strip().upper(),
        "company_name": str(payload["company_name"]).strip(),
        "peer_ticker": str(peer_ticker).strip().upper() if peer_ticker else None,
        "peer_name": str(peer_name).strip() if peer_name else None,
        "financial_query": None,
        "financial_score": 0.0,
        "financial_retry_count": 0,
        "financial_context": [],
        "risk_context": [],
        "peer_context": [],
        "quantitative_analysis": "",
        "news_context": "",
        "final_memo": "",
    }


if __name__ == "__main__":
    test_result = build_due_diligence_graph().invoke(
        normalize_graph_input({
            "ticker": "TSLA",
            "company_name": "Tesla, Inc.",
            "peer_ticker": "MSFT",
            "peer_name": "MICROSOFT CORP",
        })
    )
    print("Test run completed successfully.")