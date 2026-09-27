import logging
import os
import re
import difflib
import subprocess
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

def get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        max_output_tokens=4096,
    )

def extract_text_from_response(content: Any) -> str:
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

# --- Entity Resolution Engine ---
def get_available_local_tickers() -> List[str]:
    filings_dir = Path("data/raw/sec-edgar-filings")
    if filings_dir.exists():
        return [d.name.upper() for d in filings_dir.iterdir() if d.is_dir()]
    return []

def resolve_company_identifier(
    identifier: str, default_ticker: Optional[str] = None
) -> Tuple[str, str]:
    clean = (identifier or "").strip()
    if not clean:
        return default_ticker or "UNKNOWN", default_ticker or "UNKNOWN"

    upper_clean = clean.upper()

    COMMON_TICKERS = {
        "APPLE": ("AAPL", "Apple Inc."),
        "AAPL": ("AAPL", "Apple Inc."),
        "NVIDIA": ("NVDA", "NVIDIA Corporation"),
        "NVDA": ("NVDA", "NVIDIA Corporation"),
        "MICROSOFT": ("MSFT", "Microsoft Corporation"),
        "MSFT": ("MSFT", "Microsoft Corporation"),
        "TESLA": ("TSLA", "Tesla Inc."),
        "TSLA": ("TSLA", "Tesla Inc."),
        "GOOGLE": ("GOOGL", "Alphabet Inc."),
        "ALPHABET": ("GOOGL", "Alphabet Inc."),
        "META": ("META", "Meta Platforms Inc."),
        "AMAZON": ("AMZN", "Amazon.com Inc."),
        "AMD": ("AMD", "Advanced Micro Devices, Inc."),
        "INTEL": ("INTC", "Intel Corporation"),
        "INTC": ("INTC", "Intel Corporation"),
    }

    if upper_clean in COMMON_TICKERS:
        return COMMON_TICKERS[upper_clean]

    local_tickers = get_available_local_tickers()
    if local_tickers:
        if upper_clean in local_tickers:
            return upper_clean, upper_clean

        matches = difflib.get_close_matches(upper_clean, local_tickers, n=1, cutoff=0.6)
        if matches:
            matched_ticker = matches[0]
            logger.info("Fuzzy Matcher corrected typo '%s' -> '%s'", clean, matched_ticker)
            return matched_ticker, matched_ticker

    try:
        llm = get_llm()
        prompt = (
            f"You are a financial entity resolution engine. The user entered: '{clean}'.\n"
            "Detect and correct any typos or spelling mistakes.\n"
            "Identify the primary US stock exchange ticker and official corporate name.\n"
            "Respond ONLY in this exact format: TICKER|Company Name\n"
            "Example: NVDA|NVIDIA Corporation"
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = extract_text_from_response(resp.content).strip()

        if "|" in text:
            t, n = text.split("|", 1)
            resolved_ticker = re.sub(r"[^A-Z]", "", t.upper().strip())
            resolved_name = n.strip()
            logger.info("LLM Entity Resolver resolved '%s' -> %s (%s)", clean, resolved_ticker, resolved_name)
            return resolved_ticker or upper_clean, resolved_name
    except Exception as e:
        logger.error("Failed to resolve identifier '%s': %s", clean, e)

    return upper_clean, clean

# --- Autonomous JIT Ingestion Engine ---
def ensure_ticker_ingested(ticker: str):
    """Autonomously downloads and embeds SEC filings if the ticker is missing from the database."""
    if not ticker or ticker in {"NONE", "N/A", "UNKNOWN"}:
        return
        
    ticker = ticker.upper()
    ticker_dir = Path(f"data/raw/sec-edgar-filings/{ticker}")
    
    if not ticker_dir.exists():
        logger.warning(f"Autonomous JIT Ingestion Triggered: '{ticker}' missing from local vector database.")
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = "/app"
            
            # Now targeting src/retrieval/ingest.py
            cmd_positional = ["python", "src/retrieval/ingest.py", ticker]
            logger.info(f"Attempting ingestion: {' '.join(cmd_positional)}")
            result = subprocess.run(cmd_positional, capture_output=True, text=True, env=env)
            
            if result.returncode != 0:
                logger.warning(f"Positional arg failed. Retrying with --ticker flag. Error: {result.stderr}")
                cmd_flag = ["python", "src/retrieval/ingest.py", "--ticker", ticker]
                result = subprocess.run(cmd_flag, capture_output=True, text=True, env=env, check=True)
                
            logger.info(f"JIT Ingestion completed successfully for {ticker}.")
        except subprocess.CalledProcessError as e:
            logger.error(f"JIT Ingestion FAILED for {ticker}:\n{e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error triggering ingestion for {ticker}: {e}")
    else:
        logger.info(f"Ticker '{ticker}' verified in local vector database.")

# --- Graph Nodes ---
def extract_financials(state: DueDiligenceState) -> dict:
    raw_target = state.get("ticker") or state.get("company_name") or "AAPL"
    resolved_ticker, resolved_name = resolve_company_identifier(raw_target)

    # Trigger JIT Ingestion
    ensure_ticker_ingested(resolved_ticker)

    current_query = state.get("financial_query")
    if not current_query:
        current_query = (
            f"Consolidated Statements of Operations Total revenues Net sales "
            f"Total cost of revenues Gross profit Net income {resolved_ticker}"
        )

    logger.info("Agent 1: Extracting Financial Data for %s (%s)...", resolved_name, resolved_ticker)
    chunks, top_score = fetch_context_chunks_with_score(current_query, ticker=resolved_ticker, top_k=5)
    
    return {
        "ticker": resolved_ticker,
        "company_name": resolved_name,
        "financial_context": chunks, 
        "financial_score": top_score, 
        "financial_query": current_query
    }

def rewrite_financial_query(state: DueDiligenceState) -> dict:
    llm = get_llm()
    current_query = state.get("financial_query", "")
    score = state.get("financial_score", 0.0)
    retries = state.get("financial_retry_count", 0)

    logger.warning("Reflection Node Triggered: Retrieval score (%.4f) below threshold.", score)
    prompt = f"Rewrite this search query to use formal US GAAP / SEC reporting taxonomy for {state['company_name']} ({state['ticker']}):\n'{current_query}'\nReturn ONLY the raw rewritten search query text."
    
    response = llm.invoke([HumanMessage(content=prompt)])
    refined_query = extract_text_from_response(response.content).strip().replace('"', "")
    
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
        analysis_text = extract_text_from_response(response.content)
        
        score_match = re.search(r'SCORE:\s*(\d+)', analysis_text)
        risk_score = int(score_match.group(1)) if score_match else 50
    except Exception:
        risk_score = 50

    return {"risk_context": chunks, "risk_score": risk_score}

def discover_peer(state: DueDiligenceState) -> dict:
    peer_raw = state.get("peer_ticker") or state.get("peer_name")
    
    if peer_raw and isinstance(peer_raw, str) and peer_raw.strip().upper() not in {"NONE", "N/A", ""}:
        resolved_ticker, resolved_name = resolve_company_identifier(peer_raw)
        logger.info("Resolved Peer '%s' -> %s (%s)", peer_raw, resolved_ticker, resolved_name)
        return {"peer_ticker": resolved_ticker, "peer_name": resolved_name}

    logger.info("Autonomous Agent: Discovering strategic peer for %s...", state.get("company_name", state.get("ticker")))
    prompt = (
        f"Identify the single closest publicly traded US competitor for {state.get('company_name')} ({state.get('ticker')}). "
        "Return ONLY their stock ticker symbol (e.g., AAPL)."
    )
    
    try:
        llm = get_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_text = extract_text_from_response(response.content)
        clean_text = re.sub(r'\b(I|A)\b', '', raw_text)
        match = re.search(r'\b[A-Z]{1,5}\b', clean_text)
        discovered_ticker = match.group(0) if match else raw_text.strip().upper()
        return {"peer_ticker": discovered_ticker, "peer_name": discovered_ticker}
    except Exception as e:
        logger.error("Peer discovery failed: %s", e)
        return {"peer_ticker": None, "peer_name": None}

def extract_peer(state: DueDiligenceState) -> dict:
    peer_id = state.get("peer_ticker")
    peer_name = state.get("peer_name")
    
    if not peer_id or peer_id in {"NONE", "N/A"}:
        return {"peer_context": []}

    ensure_ticker_ingested(peer_id)

    logger.info("Agent 3: Extracting Peer Analysis for %s...", peer_id)
    query = f"Consolidated Statements of Operations Total Net Sales Revenue {peer_name or peer_id} {peer_id}"
    chunks = fetch_context_chunks(query, ticker=peer_id, top_k=5)
    return {"peer_context": chunks}

def calculate_metrics(state: DueDiligenceState) -> dict:
    logger.info("Math Agent: Computing derived quantitative ratios (Strict Isolation)...")
    llm = get_llm()
    financial_block = "\n\n".join(state.get("financial_context", []))
    peer_block = "\n\n".join(state.get("peer_context", []))

    system_prompt = (
        "You are a Quantitative Financial Analyst. Analyze the provided raw SEC financial tables. "
        "Use step-by-step reasoning to calculate derived metrics: "
        "Year-over-Year (YoY) revenue growth, gross margins, and operating margins if the data is available. "
        "CRITICAL FORMATTING RULES:\n"
        "- DO NOT use LaTeX syntax.\n"
        "- Write calculations in plain text (e.g., '(94,827 - 97,690) / 97,690 = -2.93%').\n"
        "- Format monetary amounts with 'USD' or '$' directly before numbers.\n"
        "- If a metric cannot be calculated, explicitly state 'Insufficient data'."
    )
    
    target_prompt = f"TARGET COMPANY: {state['company_name']} ({state['ticker']})\nFINANCIALS:\n{financial_block or 'No data.'}\n\nPerform quantitative analysis:"
    target_response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=target_prompt)])
    target_analysis = extract_text_from_response(target_response.content)

    peer_ticker = state.get("peer_ticker") or "Peer"
    if peer_block.strip():
        peer_prompt = f"PEER COMPANY: {state.get('peer_name', peer_ticker)} ({peer_ticker})\nFINANCIALS:\n{peer_block}\n\nPerform quantitative analysis:"
        peer_response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=peer_prompt)])
        peer_analysis = extract_text_from_response(peer_response.content)
    else:
        peer_analysis = "Insufficient data."

    combined_analysis = (
        f"{state['ticker']} Quantitative Analysis:\n{target_analysis}\n\n"
        f"{peer_ticker} Quantitative Analysis:\n{peer_analysis}"
    )
    
    return {"quantitative_analysis": combined_analysis}

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
        except Exception:
            return None

    return {"market_data": {
        "target": get_stock_data(state["ticker"]),
        "peer": get_stock_data(state.get("peer_ticker"))
    }}

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
        
        return {"news_context": extract_text_from_response(response.content).strip()}
    except Exception:
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
    
    return {"final_memo": extract_text_from_response(response.content)}

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
        "ticker": str(payload["ticker"]).strip(),
        "company_name": str(payload["company_name"]).strip(),
        "peer_ticker": str(peer_ticker).strip() if peer_ticker else None,
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