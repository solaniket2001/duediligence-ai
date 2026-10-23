import logging
import os
import re
import asyncio
import sys
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
from src.retrieval.chroma_lock import chroma_lock

try:
    from src.utils.ticker_resolver import resolve_ticker
except Exception:
    resolve_ticker = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("graph_engine")

SCORE_CONFIDENCE_THRESHOLD = 0.35
MAX_RETRIEVAL_RETRIES = 2

_RETRIEVER_INSTANCE: Optional[SECRetriever] = None

_LIVE_LOG_QUEUE: Optional[asyncio.Queue] = None

def set_live_log_queue(q: Optional[asyncio.Queue]) -> None:
    global _LIVE_LOG_QUEUE
    _LIVE_LOG_QUEUE = q

async def emit_log(msg: str) -> dict:
    if _LIVE_LOG_QUEUE is not None:
        await _LIVE_LOG_QUEUE.put(msg)
    return {"live_log": msg}



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
    live_log: str


def get_retriever(refresh: bool = False) -> SECRetriever:
    global _RETRIEVER_INSTANCE
    chroma_path = Path("data/chroma")
    chroma_path.mkdir(parents=True, exist_ok=True)
    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = SECRetriever(chroma_path)
    elif refresh:
        _RETRIEVER_INSTANCE.refresh_sparse_index()
    return _RETRIEVER_INSTANCE


def ticker_is_indexed(ticker: str) -> bool:
    """Return True only if ChromaDB has documents AND a readable HNSW segment on disk."""
    import shutil
    chroma_path = Path("data/chroma")
    try:
        retriever = get_retriever()
        target = ticker.strip().upper()
        with chroma_lock(chroma_path):
            if retriever.vector_store._collection.count() == 0:
                return False
            hits = retriever.vector_store.similarity_search(
                query=target, k=1, filter={"ticker": target}
            )
            return len(hits) > 0
    except Exception as exc:
        logger.warning("HNSW index check failed for %s (%s). Resetting data/chroma...", ticker, exc)
        global _RETRIEVER_INSTANCE
        _RETRIEVER_INSTANCE = None
        if chroma_path.exists():
            for item in chroma_path.iterdir():
                if item.name != ".chroma.lock":
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
        return False


def fetch_context_chunks_with_score(query: str, ticker: str, top_k: int = 5, prefer_tables: bool = False) -> Tuple[List[str], float]:
    retriever = get_retriever()
    requested_ticker = ticker.strip().upper()
    hits = retriever.search(
        query=query,
        top_k=max(top_k * 6, 30),
        rerank_top_k=top_k,
        metadata_filter={"ticker": requested_ticker},
    )
    if prefer_tables:
        table_hits = retriever.search(
            query="Consolidated Statements of Operations Income Total revenue net sales gross margin operating income net income",
            top_k=40,
            rerank_top_k=top_k,
            metadata_filter={"ticker": requested_ticker, "chunk_type": "table"},
            enable_hyde=False,
            score_cliff=1.0,
        )
        # Filter out short Table-of-Contents / index tables that only list page numbers
        valid_tables = [
            h for h in table_hits
            if len(str(h.get("text", ""))) > 250 and "page" not in str(h.get("text", "")).lower()[:120]
        ]
        seen_ids = set()
        merged_hits = []
        for h in (valid_tables[:3] + hits):
            doc_id = (h.get("meta") or {}).get("document_id")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                merged_hits.append(h)
        hits = merged_hits[: max(top_k, 6)]
    if not hits:
        return [], 0.0
    chunks = []
    top_score = 0.0
    for hit in hits:
        metadata = hit.get("meta", {}) if isinstance(hit, dict) else getattr(hit, "metadata", {})
        content = (
            hit.get("text", hit.get("page_content", str(hit)))
            if isinstance(hit, dict)
            else getattr(hit, "page_content", str(hit))
        )
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
    model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        max_output_tokens=4096,
        max_retries=1,
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


def resolve_company_identifier(identifier: str, default_ticker: Optional[str] = None) -> Tuple[str, str]:
    clean = (identifier or "").strip()
    if not clean:
        return default_ticker or "UNKNOWN", default_ticker or "UNKNOWN"
    upper_clean = clean.upper()

    COMMON_TICKERS = {
        "APPLE": ("AAPL", "Apple Inc."), "AAPL": ("AAPL", "Apple Inc."),
        "NVIDIA": ("NVDA", "NVIDIA Corporation"), "NVDA": ("NVDA", "NVIDIA Corporation"),
        "MICROSOFT": ("MSFT", "Microsoft Corporation"), "MSFT": ("MSFT", "Microsoft Corporation"),
        "TESLA": ("TSLA", "Tesla, Inc."), "TSLA": ("TSLA", "Tesla, Inc."),
        "GOOGLE": ("GOOGL", "Alphabet Inc."), "ALPHABET": ("GOOGL", "Alphabet Inc."),
        "GOOGL": ("GOOGL", "Alphabet Inc."), "GOOG": ("GOOGL", "Alphabet Inc."),
        "META": ("META", "Meta Platforms, Inc."), "AMAZON": ("AMZN", "Amazon.com, Inc."),
        "AMZN": ("AMZN", "Amazon.com, Inc."), "AMD": ("AMD", "Advanced Micro Devices, Inc."),
        "INTEL": ("INTC", "Intel Corporation"), "INTC": ("INTC", "Intel Corporation"),
        "NETFLIX": ("NFLX", "Netflix, Inc."), "NFLX": ("NFLX", "Netflix, Inc."),
    }
    if upper_clean in COMMON_TICKERS:
        return COMMON_TICKERS[upper_clean]

    # 1. Official SEC company_tickers.json resolution
    if resolve_ticker is not None:
        try:
            sec_ticker, sec_title = resolve_ticker(clean)
            if sec_ticker:
                return sec_ticker.upper().strip(), (sec_title or sec_ticker).strip()
        except Exception as exc:
            logger.warning("SEC ticker_resolver fallback for '%s': %s", clean, exc)

    # 2. If valid 1-5 letter ticker format, check yFinance or LLM for company name
    try:
        llm = get_llm()
        prompt = (
            f"Identify the official US stock ticker symbol and official company name for '{clean}'. "
            "Respond ONLY in the exact format: TICKER|Company Name"
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = extract_text_from_response(resp.content).strip()
        if "|" in text:
            t, n = text.split("|", 1)
            cleaned_t = re.sub(r"[^A-Z]", "", t.upper().strip())
            if cleaned_t:
                return cleaned_t, n.strip()
    except Exception:
        pass

    return re.sub(r"[^A-Z]", "", upper_clean) or upper_clean, clean


# --- ASYNC AUTONOMOUS JIT ENGINE ---
async def async_ensure_ticker_ingested(ticker: str):
    """Checks if ticker is downloaded, parsed, and indexed; if not, runs live JIT ingestion."""
    if not ticker or ticker.upper() in {"NONE", "N/A", "UNKNOWN"}:
        return

    ticker = ticker.strip().upper()
    project_root = Path(__file__).resolve().parents[2]
    ticker_dir = project_root / f"data/raw/sec-edgar-filings/{ticker}"
    processed_dir = project_root / f"data/processed/{ticker}"
    legacy_processed = list((project_root / "data/processed").glob(f"*{ticker}*.json")) if (project_root / "data/processed").exists() else []

    has_processed = (processed_dir.exists() and any(processed_dir.glob("*.json"))) or bool(legacy_processed)
    is_indexed = await asyncio.to_thread(ticker_is_indexed, ticker)

    if ticker_dir.exists() and has_processed and is_indexed:
        yield f"[System] {ticker} verified in local ChromaDB vector index."
        return

    yield f"[JIT Trigger] {ticker} missing from local index. Booting autonomous SEC EDGAR ingestion pipeline..."

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(project_root), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["PYTHONUNBUFFERED"] = "1"

    process = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-m", "src.retrieval.ingest", "--ticker", ticker,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(project_root),
        env=env,
    )

    assert process.stdout is not None
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        text_line = line.decode("utf-8", errors="replace").strip()
        if text_line:
            yield f"[JIT: {ticker}] {text_line}"

    await process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"Automatic SEC ingestion failed for {ticker} (exit code {process.returncode})."
        )

    # Refresh the in-memory Chroma & BM25 index after subprocess finishes writing
    await asyncio.to_thread(get_retriever, True)
    yield f"[Success] {ticker} SEC 10-K is now fully ingested and indexed in ChromaDB."


# --- Graph Nodes ---
async def extract_financials(state: DueDiligenceState):
    raw_target = state.get("ticker") or state.get("company_name") or "AAPL"
    resolved_ticker, resolved_name = await asyncio.to_thread(resolve_company_identifier, raw_target)
    yield await emit_log(f"[System] Target resolved: {resolved_name} ({resolved_ticker})")

    async for log_line in async_ensure_ticker_ingested(resolved_ticker):
        yield await emit_log(log_line)

    current_query = (
        state.get("financial_query")
        or f"What were total revenue, gross profit, operating income, and net income in the Consolidated Statements of Operations for {resolved_name} ({resolved_ticker})?"
    )

    yield await emit_log(f"[Agent 1] Extracting Financial Statements for {resolved_name} ({resolved_ticker})...")
    chunks, top_score = await asyncio.to_thread(
        fetch_context_chunks_with_score, current_query, resolved_ticker, 6, True
    )

    yield {
        "ticker": resolved_ticker,
        "company_name": resolved_name,
        "financial_context": chunks,
        "financial_score": top_score,
        "financial_query": current_query,
    }


async def rewrite_financial_query(state: DueDiligenceState):
    llm = get_llm()
    current_query = state.get("financial_query", "")
    score = state.get("financial_score", 0.0)
    retries = state.get("financial_retry_count", 0)

    yield await emit_log(f"[Reflection Node] Retrieval confidence ({score:.4f}) below threshold. Rewriting query...")
    prompt = (
        f"Rewrite this search query to use formal US GAAP / SEC reporting taxonomy for "
        f"{state['company_name']} ({state['ticker']}):\n'{current_query}'\n"
        "Return ONLY the raw rewritten search query text."
    )

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    refined_query = extract_text_from_response(response.content).strip().replace('"', "")

    yield {"financial_query": refined_query, "financial_retry_count": retries + 1}


def should_refine_financials(state: DueDiligenceState) -> str:
    score = state.get("financial_score", 0.0)
    retries = state.get("financial_retry_count", 0)
    chunks = state.get("financial_context", [])
    if (score < SCORE_CONFIDENCE_THRESHOLD or not chunks) and retries < MAX_RETRIEVAL_RETRIES:
        return "rewrite_financial_query"
    return "extract_risks"


async def extract_risks(state: DueDiligenceState):
    yield await emit_log(f"[Agent 2] Extracting Item 1A Risk Factors for {state['company_name']}...")
    query = f"Item 1A Risk Factors business risks regulatory legal competition supply chain market exposure {state['company_name']} {state['ticker']}"
    chunks = await asyncio.to_thread(fetch_context_chunks, query, state["ticker"], 5)

    llm = get_llm()
    risk_text = "\n\n".join(chunks)
    prompt = (
        f"Analyze these risk factors for {state['company_name']} ({state['ticker']}).\n\n"
        f"Risk Context:\n{risk_text}\n\n"
        "Evaluate the severity of these risks and assign a quantitative Risk Score from 1 to 100 (where 100 is maximum risk).\n"
        "You must include a line at the very end of your response exactly formatted as 'SCORE: X' where X is the integer score."
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        analysis_text = extract_text_from_response(response.content)
        score_match = re.search(r"SCORE:\s*(\d+)", analysis_text)
        risk_score = int(score_match.group(1)) if score_match else 50
    except Exception:
        risk_score = 50

    yield {"risk_context": chunks, "risk_score": risk_score}


async def discover_peer(state: DueDiligenceState):
    peer_raw = state.get("peer_ticker") or state.get("peer_name")

    if peer_raw and isinstance(peer_raw, str) and peer_raw.strip().upper() not in {"NONE", "N/A", ""}:
        resolved_ticker, resolved_name = await asyncio.to_thread(resolve_company_identifier, peer_raw)
        yield await emit_log(f"[System] Benchmark peer resolved: {resolved_name} ({resolved_ticker}).")
        yield {"peer_ticker": resolved_ticker, "peer_name": resolved_name}
        return

    yield await emit_log(f"[Agent Engine] Discovering primary US competitor for {state.get('company_name')}...")
    prompt = (
        f"Identify the single closest publicly traded US competitor (with SEC 10-K filings) for "
        f"{state.get('company_name')} ({state.get('ticker')}). "
        f"Do NOT return {state.get('ticker')}. Return ONLY their stock ticker symbol (e.g., MSFT)."
    )

    try:
        llm = get_llm()
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw_text = extract_text_from_response(response.content).strip().upper()
        match = re.search(r"\b[A-Z]{1,5}\b", raw_text)
        discovered_ticker = match.group(0) if match else raw_text
        resolved_ticker, resolved_name = await asyncio.to_thread(resolve_company_identifier, discovered_ticker)
        yield await emit_log(f"[Agent Engine] Peer discovered: {resolved_name} ({resolved_ticker}).")
        yield {"peer_ticker": resolved_ticker, "peer_name": resolved_name}
    except Exception:
        yield {"peer_ticker": None, "peer_name": None}


async def extract_peer(state: DueDiligenceState):
    peer_id = state.get("peer_ticker")
    peer_name = state.get("peer_name")

    if not peer_id or peer_id in {"NONE", "N/A"}:
        yield {"peer_context": []}
        return

    async for log_line in async_ensure_ticker_ingested(peer_id):
        yield await emit_log(log_line)

    yield await emit_log(f"[Agent 3] Extracting Peer Financials for {peer_name or peer_id} ({peer_id})...")
    query = f"Consolidated Statements of Operations Total Net Sales Revenue Operating Income Net Income {peer_name or peer_id} {peer_id}"
    chunks, _ = await asyncio.to_thread(fetch_context_chunks_with_score, query, peer_id, 6, True)
    yield {"peer_context": chunks}


async def calculate_metrics(state: DueDiligenceState):
    yield await emit_log("[Math Agent] Computing quantitative ratios under strict isolation...")
    llm = get_llm()
    financial_block = "\n\n".join(state.get("financial_context", []))
    peer_block = "\n\n".join(state.get("peer_context", []))

    system_prompt = (
        "You are a Quantitative Financial Analyst. Analyze the provided raw SEC financial tables. "
        "Use step-by-step reasoning to calculate derived metrics: Year-over-Year (YoY) revenue growth, "
        "gross margins, and operating margins if the data is available. CRITICAL FORMATTING RULES:\n"
        "- DO NOT use LaTeX syntax.\n"
        "- Write calculations in plain text (e.g., '(94,827 - 97,690) / 97,690 = -2.93%').\n"
        "- Format monetary amounts with 'USD' or '$' directly before numbers.\n"
        "- If a metric cannot be calculated, explicitly state 'Insufficient data'.\n"
        "- Format any summary comparison as clean Markdown bullet points (do NOT use pipe '|' tables)."
    )

    target_prompt = f"TARGET COMPANY: {state['company_name']} ({state['ticker']})\nFINANCIALS:\n{financial_block or 'No data.'}\n\nPerform quantitative analysis:"
    target_response = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=target_prompt)])
    target_analysis = extract_text_from_response(target_response.content)

    peer_ticker = state.get("peer_ticker") or "Peer"
    if peer_block.strip():
        peer_prompt = f"PEER COMPANY: {state.get('peer_name', peer_ticker)} ({peer_ticker})\nFINANCIALS:\n{peer_block}\n\nPerform quantitative analysis:"
        peer_response = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=peer_prompt)])
        peer_analysis = extract_text_from_response(peer_response.content)
    else:
        peer_analysis = "Insufficient data."

    combined_analysis = f"{state['ticker']} Quantitative Analysis:\n{target_analysis}\n\n{peer_ticker} Quantitative Analysis:\n{peer_analysis}"
    yield {"quantitative_analysis": combined_analysis}


async def fetch_live_fundamentals(state: DueDiligenceState):
    yield await emit_log("[Data Agent] Pulling real-time market pricing via yFinance...")

    def get_stock_data(ticker_symbol: Optional[str]):
        if not ticker_symbol or ticker_symbol in {"NONE", "N/A"}:
            return None
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info or {}
            hist = stock.history(period="6mo")
            return {
                "price": info.get("currentPrice", info.get("regularMarketPrice")),
                "market_cap": info.get("marketCap"),
                "forward_pe": info.get("forwardPE"),
                "history_dates": [d.strftime("%Y-%m-%d") for d in hist.index],
                "history_prices": hist["Close"].tolist(),
            }
        except Exception:
            return None

    target_data, peer_data = await asyncio.gather(
        asyncio.to_thread(get_stock_data, state["ticker"]),
        asyncio.to_thread(get_stock_data, state.get("peer_ticker")),
    )

    yield {"market_data": {"target": target_data, "peer": peer_data}}


async def fetch_live_news(state: DueDiligenceState):
    yield await emit_log("[News Agent] Fetching live market intelligence via RSS...")
    ticker = state["ticker"].upper().strip()
    feed_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"

    try:
        feed = await asyncio.to_thread(feedparser.parse, feed_url)
        entries = feed.entries[:5]
        if not entries:
            yield {"news_context": "No recent financial news detected for this ticker."}
            return

        news_snippets = [
            f"Headline: {getattr(item, 'title', 'Update')}\nSummary: {getattr(item, 'summary', '')}"
            for item in entries
        ]
        raw_news = "\n---\n".join(news_snippets)

        llm = get_llm()
        prompt = (
            f"Analyze these recent headlines for {state['company_name']} ({state['ticker']}):\n\n"
            f"{raw_news}\n\n"
            "Synthesize a 1-paragraph institutional 'Current Market Outlook' detailing prevailing sentiment (Bullish/Bearish/Neutral) and catalysts."
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        yield {"news_context": extract_text_from_response(response.content).strip()}
    except Exception:
        yield {"news_context": "Live market news extraction temporarily unavailable."}


async def synthesize_memo(state: DueDiligenceState):
    yield await emit_log(f"[Lead PM Agent] Synthesizing final Investment Memo for {state['company_name']}...")
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
    user_prompt = (
        f"TARGET COMPANY: {state['company_name']} ({state['ticker']})\n"
        f"=== FINANCIAL CONTEXT ===\n{financial_block}\n"
        f"=== QUANTITATIVE ANALYSIS ===\n{state.get('quantitative_analysis')}\n"
        f"=== LIVE MARKET OUTLOOK ===\n{state.get('news_context')}\n"
        f"=== RISK CONTEXT ===\n{risk_block}\n"
        f"=== PEER CONTEXT ===\n{peer_block}\n\n"
        "Please synthesize the Investment Memorandum:"
    )

    response = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    yield {"final_memo": extract_text_from_response(response.content)}


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
    builder.add_conditional_edges(
        "extract_financials",
        should_refine_financials,
        {"rewrite_financial_query": "rewrite_financial_query", "extract_risks": "extract_risks"},
    )
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
        "chat_history": [],
    }
