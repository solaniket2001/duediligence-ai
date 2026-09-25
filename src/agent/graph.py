import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from dotenv import load_dotenv
load_dotenv()

import yfinance as yf
import feedparser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

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
    risk_score: int
    peer_context: List[str]
    quantitative_analysis: str
    market_data: dict
    news_context: str
    final_memo: str
    chat_history: List[str]


# --- Dynamic Retriever Factory ---
def get_retriever() -> SECRetriever:
    return SECRetriever(Path("data/chroma"))

def fetch_context_chunks_with_score(query: str, ticker: str, top_k: int = 5) -> Tuple[List[str], float]:
    retriever = get_retriever()
    requested_ticker = ticker.strip().upper()

    hits = retriever.search(
        query=query,
        top_k=max(top_k * 5, 25),
        rerank_top_k=top_k,
        metadata_filter={"ticker": requested_ticker},
    )

    if not hits:
        return [], 0.0

    chunks = []
    top_score = 0.0
    for hit in hits:
        metadata = hit.get("meta", {}) if isinstance(hit, dict) else getattr(hit, "metadata", {})
        content = hit.get("text", hit.get("page_content", str(hit))) if isinstance(hit, dict) else getattr(hit, "page_content", str(hit))
        score = hit.get("score") or metadata.get("score") or metadata.get("relevance_score") or 0.0

        try:
            score_val = float(score)
            if score_val > top_score:
                top_score = score_val
        except (ValueError, TypeError):
            pass

        doc_id = metadata.get("document_id", "DOC_UNKNOWN")
        chunks.append(f"[{doc_id}]\n{content}")

    return chunks, top_score

def fetch_context_chunks(query: str, ticker: str, top_k: int = 5) -> List[str]:
    chunks, _ = fetch_context_chunks_with_score(query=query, ticker=ticker, top_k=top_k)
    return chunks

def _resolve_peer(state: DueDiligenceState) -> Tuple[Optional[str], Optional[str]]:
    peer_ticker = state.get("peer_ticker") or state.get("peer") or state.get("peer_symbol")
    peer_name = state.get("peer_name") or state.get("peer_company_name")
    if peer_ticker:
        peer_ticker = str(peer_ticker).strip().upper()
    return peer_ticker, peer_name

def get_llm() -> ChatGoogleGenerativeAI:
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
        current_query = (
            f"Consolidated Statements of Operations Total revenues Automotive sales "
            f"Total cost of revenues Gross profit Net income {state['ticker']}"
        )

    logger.info("Agent 1: Extracting Financial Data for %s (%s)...", state["company_name"], state["ticker"])
    chunks, top_score = fetch_context_chunks_with_score(current_query, ticker=state["ticker"], top_k=5)
    logger.info("Agent 1: Top Cross-Encoder score achieved: %.4f", top_score)
    
    return {"financial_context": chunks, "financial_score": top_score, "financial_query": current_query}

def rewrite_financial_query(state: DueDiligenceState) -> dict:
    llm = get_llm()
    current_query = state.get("financial_query", "")
    score = state.get("financial_score", 0.0)
    retries = state.get("financial_retry_count", 0)

    logger.warning("Reflection Node Triggered: Retrieval score (%.4f) below threshold.", score)
    prompt = f"Rewrite this search query to use formal US GAAP / SEC reporting taxonomy for {state['company_name']} ({state['ticker']}):\n'{current_query}'\nReturn ONLY the raw rewritten search query text."
    
    response = llm.invoke([HumanMessage(content=prompt)])
    raw_content = response.content
    rewritten_text = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in raw_content) if isinstance(raw_content, list) else str(raw_content)
    refined_query = rewritten_text.strip().replace('"', "")
    
    return {"financial_query": refined_query, "financial_retry_count": retries + 1}

def should_refine_financials(state: DueDiligenceState) -> str:
    score = state.get("financial_score", 0.0)
    retries = state.get("financial_retry_count", 0)
    chunks = state.get("financial_context", [])

    if (score < SCORE_CONFIDENCE_THRESHOLD or not chunks) and retries < MAX_RETRIEVAL_RETRIES:
        return "rewrite_financial_query"
    return "extract_risks"

def extract_risks(state: DueDiligenceState) -> dict:
    logger.info("Agent 2: Extracting Risk Factors & Scoring for %s...", state["company_name"])
    query = f"Item 1A Risk Factors business risks regulatory legal competition market exposure {state['company_name']} {state['ticker']}"
    chunks = fetch_context_chunks(query, ticker=state["ticker"], top_k=5)
    
    # Calculate Quantitative Risk Score
    llm = get_llm()
    risk_text = "\n\n".join(chunks)
    prompt = (
        f"Analyze these risk factors for {state['company_name']} ({state['ticker']}).\n\n"
        f"Risk Context:\n{risk_text}\n\n"
        "Evaluate the severity of these risks and assign a quantitative Risk Score from 1 to 100 (where 100 is maximum risk).\n"
        "You must include a line at the very end of your response exactly formatted as 'SCORE: X' where X is the integer score."
    )
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_content = response.content
        analysis_text = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in raw_content) if isinstance(raw_content, list) else str(raw_content)
        
        score_match = re.search(r'SCORE:\s*(\d+)', analysis_text)
        risk_score = int(score_match.group(1)) if score_match else 50
    except Exception as e:
        logger.error("Risk scoring failed: %s", e)
        risk_score = 50

    return {"risk_context": chunks, "risk_score": risk_score}

def extract_text_from_response(content: Any) -> str:
    """Extracts raw text whether response.content is a str, list of strings, or list of content block dicts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        extracted = []
        for item in content:
            if isinstance(item, str):
                extracted.append(item)
            elif isinstance(item, dict):
                extracted.append(item.get("text", ""))
            elif hasattr(item, "text"):
                extracted.append(getattr(item, "text", ""))
            else:
                extracted.append(str(item))
        return "".join(extracted)
    return str(content)

def discover_peer(state: DueDiligenceState) -> dict:
    peer_ticker = state.get("peer_ticker")
    peer_name = state.get("peer_name")
    
    # 1. If a valid ticker is already supplied, do nothing
    if peer_ticker and isinstance(peer_ticker, str) and peer_ticker.strip().upper() not in {"NONE", "N/A", ""}:
        return {}
        
    llm = get_llm()
    
    # 2. If user typed a company name (e.g., 'microsoft'), resolve it to its ticker
    if peer_name and isinstance(peer_name, str) and peer_name.strip().upper() not in {"NONE", "N/A", ""}:
        logger.info("Resolving ticker for peer name: %s...", peer_name)
        prompt = (
            f"Identify the primary US stock ticker symbol for the company named '{peer_name}'. "
            "Return ONLY the uppercase ticker symbol (e.g., MSFT) and nothing else."
        )
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw_text = extract_text_from_response(response.content)
            
            # Remove isolated "I" or "A" so they don't trigger the 1-5 letter regex
            clean_text = re.sub(r'\b(I|A)\b', '', raw_text)
            
            # Search for consecutive uppercase letters in the ORIGINAL text
            match = re.search(r'\b[A-Z]{1,5}\b', clean_text)
            resolved_ticker = match.group(0) if match else raw_text.strip().upper()
            
            logger.info("Resolved %s to ticker: %s", peer_name, resolved_ticker)
            return {"peer_ticker": resolved_ticker, "peer_name": peer_name}
        except Exception as e:
            logger.error("Failed to resolve peer name to ticker: %s", e)

    # 3. If both are blank, discover the competitor autonomously
    logger.info("Autonomous Agent: Discovering strategic peer for %s...", state["company_name"])
    prompt = (
        f"Identify the single closest publicly traded US competitor for {state['company_name']} ({state['ticker']}). "
        "Return ONLY their stock ticker symbol (e.g., AAPL)."
    )
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_text = extract_text_from_response(response.content)
        
        # Remove isolated "I" or "A" so they don't trigger the 1-5 letter regex
        clean_text = re.sub(r'\b(I|A)\b', '', raw_text)
        
        # Search for consecutive uppercase letters in the ORIGINAL text
        match = re.search(r'\b[A-Z]{1,5}\b', clean_text)
        discovered_ticker = match.group(0) if match else raw_text.strip().upper()
        
        logger.info("Autonomous Agent selected peer: %s", discovered_ticker)
        return {"peer_ticker": discovered_ticker, "peer_name": discovered_ticker}
    except Exception as e:
        logger.error("Peer discovery failed: %s", e)
        return {"peer_ticker": None, "peer_name": None}

def extract_peer(state: DueDiligenceState) -> dict:
    peer_id, peer_name = _resolve_peer(state)
    if not peer_id or peer_id in {"NONE", "N/A"}:
        logger.info("Agent 3: No peer specified. Skipping peer extraction.")
        return {"peer_context": []}

    logger.info("Agent 3: Extracting Peer Analysis for %s...", peer_id)
    query = f"Consolidated Statements of Operations Total Net Sales Revenue {peer_name or peer_id} {peer_id}"
    chunks = fetch_context_chunks(query, ticker=peer_id, top_k=5)
    return {"peer_context": chunks}

def calculate_metrics(state: DueDiligenceState) -> dict:
    logger.info("Math Agent: Computing derived quantitative ratios...")
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
        "- Format monetary amounts with 'USD' or '$' followed directly by numbers without backslashes.\n"
        "- If a metric cannot be calculated, explicitly state 'Insufficient data'."
    )
    user_prompt = f"TARGET COMPANY: {state['company_name']}\nTARGET FINANCIALS:\n{financial_block or 'No data.'}\n\nPEER FINANCIALS:\n{peer_block or 'No data.'}\n\nPerform quantitative analysis:"

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    raw_content = response.content
    analysis_text = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in raw_content) if isinstance(raw_content, list) else str(raw_content)
    
    return {"quantitative_analysis": analysis_text}

def fetch_live_fundamentals(state: DueDiligenceState) -> dict:
    logger.info("Live Fundamentals Agent: Pulling market data via yfinance...")
    
    def get_stock_data(ticker_symbol: str):
        if not ticker_symbol or ticker_symbol in {"NONE", "N/A"}:
            return None
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            hist = stock.history(period="6mo")
            hist_prices = hist['Close'].tolist()
            hist_dates = [d.strftime('%Y-%m-%d') for d in hist.index]
            
            return {
                "price": info.get("currentPrice", info.get("regularMarketPrice")),
                "market_cap": info.get("marketCap"),
                "forward_pe": info.get("forwardPE"),
                "history_dates": hist_dates,
                "history_prices": hist_prices
            }
        except Exception as e:
            logger.error("Failed to fetch yfinance data for %s: %s", ticker_symbol, e)
            return None

    market_data = {
        "target": get_stock_data(state["ticker"]),
        "peer": get_stock_data(state.get("peer_ticker"))
    }
    
    return {"market_data": market_data}

def fetch_live_news(state: DueDiligenceState) -> dict:
    logger.info("Agent 4 (News): Fetching live market intelligence via RSS...")
    ticker = state["ticker"].upper().strip()
    feed_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    
    try:
        feed = feedparser.parse(feed_url)
        entries = feed.entries[:5]
        if not entries:
            return {"news_context": "No recent financial news detected for this ticker."}
            
        news_snippets = [f"Headline: {getattr(item, 'title', 'Update')}\nSummary: {getattr(item, 'summary', '')}" for item in entries]
        raw_news = "\n---\n".join(news_snippets)
        
        llm = get_llm()
        prompt = f"Analyze these recent headlines for {state['company_name']} ({state['ticker']}):\n\n{raw_news}\n\nSynthesize a 1-paragraph institutional 'Current Market Outlook' detailing prevailing sentiment (Bullish/Bearish/Neutral) and catalysts."
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_content = response.content
        sentiment_text = "".join(chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in raw_content) if isinstance(raw_content, list) else str(raw_content)
        
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
    
    system_prompt = (
        "You are a Lead Portfolio Manager writing an institutional investment memorandum. Adhere strictly to these rules:\n"
        "1. Every factual statement or financial metric must include an exact bracketed document citation (e.g., [DOC_ID]).\n"
        "2. ZERO HALLUCINATION: Ground claims exclusively in the provided context, Quantitative Analysis, and Live Market Outlook.\n"
        "3. ZERO CROSS-CONTAMINATION: Do not mix peer metrics with the target company's metrics.\n"
        "4. FORMATTING: DO NOT use LaTeX math equations or backslashes. Write numbers as plain text.\n"
        "5. Structure into: 1. EXECUTIVE SUMMARY, 2. FINANCIAL PERFORMANCE, 3. PEER COMPARISON, 4. KEY RISKS, 5. CURRENT MARKET OUTLOOK, followed by CONCLUSION / RECOMMENDATION."
    )
    user_prompt = f"TARGET COMPANY: {state['company_name']} ({state['ticker']})\n=== FINANCIAL CONTEXT ===\n{financial_block}\n=== QUANTITATIVE ANALYSIS ===\n{state.get('quantitative_analysis')}\n=== LIVE MARKET OUTLOOK ===\n{state.get('news_context')}\n=== RISK CONTEXT ===\n{risk_block}\n=== PEER CONTEXT ===\n{peer_block}\n\nPlease synthesize the Investment Memorandum:"

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    raw_content = response.content
    memo_text = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in raw_content) if isinstance(raw_content, list) else str(raw_content)
    
    return {"final_memo": memo_text}


# --- Graph Construction ---
def build_due_diligence_graph():
    builder = StateGraph(DueDiligenceState)

    builder.add_node("extract_financials", extract_financials)
    builder.add_node("rewrite_financial_query", rewrite_financial_query)
    builder.add_node("extract_risks", extract_risks)
    builder.add_node("discover_peer", discover_peer)
    builder.add_node("extract_peer", extract_peer)
    builder.add_node("calculate_metrics", calculate_metrics)
    builder.add_node("fetch_live_fundamentals", fetch_live_fundamentals)
    builder.add_node("fetch_live_news", fetch_live_news)
    builder.add_node("synthesize_memo", synthesize_memo)

    builder.add_edge(START, "extract_financials")
    builder.add_conditional_edges("extract_financials", should_refine_financials, {"rewrite_financial_query": "rewrite_financial_query", "extract_risks": "extract_risks"})
    builder.add_edge("rewrite_financial_query", "extract_financials")
    builder.add_edge("extract_risks", "discover_peer")
    builder.add_edge("discover_peer", "extract_peer")
    builder.add_edge("extract_peer", "calculate_metrics")
    builder.add_edge("calculate_metrics", "fetch_live_fundamentals")
    builder.add_edge("fetch_live_fundamentals", "fetch_live_news")
    builder.add_edge("fetch_live_news", "synthesize_memo")
    builder.add_edge("synthesize_memo", END)

    return builder.compile()

def normalize_graph_input(payload: dict) -> DueDiligenceState:
    peer_ticker = payload.get("peer_ticker") or payload.get("peer")
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
        "risk_score": 50,
        "peer_context": [],
        "quantitative_analysis": "",
        "market_data": {},
        "news_context": "",
        "final_memo": "",
        "chat_history": []
    }