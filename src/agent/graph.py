import logging
import os
from pathlib import Path
from typing import Any, List, Optional, TypedDict

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


# --- State Definition ---
class DueDiligenceState(TypedDict, total=False):
    ticker: str
    company_name: str
    peer_ticker: Optional[str]
    peer_name: Optional[str]
    financial_context: List[str]
    risk_context: List[str]
    peer_context: List[str]
    final_memo: str


# --- Dynamic Retriever Factory ---
def get_retriever() -> SECRetriever:
    """Instantiates a fresh SECRetriever handle dynamically."""
    return SECRetriever(Path("data/chroma"))

def fetch_context_chunks(
    query: str,
    ticker: str,
    top_k: int = 5,
) -> List[str]:
    """Retrieve context strictly filtered by ticker via the SECRetriever."""
    retriever = get_retriever()
    requested_ticker = ticker.strip().upper()

    # Pass the exact filter and top_k parameters the search method expects
    hits = retriever.search(
        query=query, 
        top_k=max(top_k * 5, 25),      # Cast a wide net for the dense/sparse hybrid
        rerank_top_k=top_k,            # Return top 5 chunks after FlashRank judging
        metadata_filter={"ticker": requested_ticker}
    )

    if not hits:
        logger.warning("No retrieval results matched ticker=%s.", requested_ticker)
        return []

    chunks: List[str] = []
    for hit in hits:
        # FlashRank returns a dictionary format: {'text': '...', 'meta': {...}}
        if isinstance(hit, dict):
            metadata = hit.get("meta", {})
            content = hit.get("text", str(hit))
        else:
            # Fallback just in case it returns a LangChain Document
            metadata = getattr(hit, "metadata", {})
            content = getattr(hit, "page_content", str(hit))
            
        document_id = metadata.get("document_id", "DOC_UNKNOWN")
        chunks.append(f"[{document_id}]\n{content}")

    return chunks


def _resolve_peer(state: DueDiligenceState) -> tuple[Optional[str], Optional[str]]:
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
    """Agent 1: Extracts financial and operational statements for the target company."""
    logger.info(
        "Agent 1: Extracting Financial Data for %s (%s)...",
        state["company_name"],
        state["ticker"],
    )

    query = (
        "Consolidated Statements of Operations Net Sales Revenue "
        f"Product breakdown by segment {state['company_name']} {state['ticker']}"
    )

    chunks = fetch_context_chunks(
        query,
        ticker=state["ticker"],
        top_k=5,
    )
    return {"financial_context": chunks}


def extract_risks(state: DueDiligenceState) -> dict:
    """Agent 2: Extracts Item 1A Risk Factors for the target company."""
    logger.info("Agent 2: Extracting Risk Factors for %s...", state["company_name"])

    query = (
        "Item 1A Risk Factors business risks regulatory legal competition "
        f"market exposure {state['company_name']} {state['ticker']}"
    )

    chunks = fetch_context_chunks(
        query,
        ticker=state["ticker"],
        top_k=5,
    )
    return {"risk_context": chunks}


def extract_peer(state: DueDiligenceState) -> dict:
    """Agent 3: Extracts comparable metrics for the peer entity if specified."""
    peer_id, peer_name = _resolve_peer(state)

    if not peer_id or peer_id in {"NONE", "N/A"}:
        logger.info("Agent 3: No peer specified. Skipping peer extraction.")
        return {"peer_context": []}

    target_label = f"{peer_name} ({peer_id})" if peer_name else peer_id
    logger.info("Agent 3: Extracting Peer Analysis for %s...", target_label)

    query_name = peer_name or peer_id
    query = (
        "Consolidated Statements of Operations Total Net Sales Revenue "
        f"{query_name} {peer_id}"
    )

    chunks = fetch_context_chunks(
        query,
        ticker=peer_id,
        top_k=5,
    )
    return {"peer_context": chunks}


def synthesize_memo(state: DueDiligenceState) -> dict:
    """Agent 4: Synthesizes the final investment memo with strict provenance citations."""
    logger.info("Agent 4: Synthesizing Investment Memo for %s...", state["company_name"])
    llm = get_llm()

    financial_block = "\n\n".join(state.get("financial_context", []))
    risk_block = "\n\n".join(state.get("risk_context", []))
    peer_block = "\n\n".join(state.get("peer_context", []))
    
    peer_target = state.get("peer_name") or state.get("peer_ticker") or "None"

    system_prompt = (
        "You are a Lead Portfolio Manager writing an institutional investment memorandum. "
        "Adhere strictly to these rules:\n"
        "1. Every factual statement or financial metric must include an exact bracketed document citation (e.g., [DOC_ID]).\n"
        "2. ZERO HALLUCINATION: Ground claims exclusively in the provided context.\n"
        "3. ZERO CROSS-CONTAMINATION: Do not mix peer metrics with the target company's metrics.\n"
        "4. Structure your response into exactly four sections: "
        "1. EXECUTIVE SUMMARY, 2. FINANCIAL PERFORMANCE, 3. PEER COMPARISON, 4. KEY RISKS, followed by CONCLUSION / RECOMMENDATION.\n"
        "5. If peer context is empty, explicitly state that peer filing data is unavailable without speculating."
    )

    user_prompt = f"""
TARGET COMPANY: {state['company_name']} ({state['ticker']})
PEER TARGET: {peer_target}

=== FINANCIAL CONTEXT ===
{financial_block or "No specific financial tables retrieved."}

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
    if isinstance(raw_content, list):
        text_parts = [
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in raw_content
        ]
        memo_text = "".join(text_parts)
    else:
        memo_text = str(raw_content)

    return {"final_memo": memo_text}


# --- Graph Construction ---
def build_due_diligence_graph():
    """Builds and compiles the multi-agent LangGraph workflow."""
    builder = StateGraph(DueDiligenceState)

    builder.add_node("extract_financials", extract_financials)
    builder.add_node("extract_risks", extract_risks)
    builder.add_node("extract_peer", extract_peer)
    builder.add_node("synthesize_memo", synthesize_memo)

    builder.add_edge(START, "extract_financials")
    builder.add_edge("extract_financials", "extract_risks")
    builder.add_edge("extract_risks", "extract_peer")
    builder.add_edge("extract_peer", "synthesize_memo")
    builder.add_edge("synthesize_memo", END)

    return builder.compile()


def normalize_graph_input(payload: dict) -> DueDiligenceState:
    """Normalize UI/API input before starting the graph."""
    peer_ticker = (
        payload.get("peer_ticker")
        or payload.get("peer")
        or payload.get("peer_symbol")
    )

    peer_name = (
        payload.get("peer_name")
        or payload.get("peer_company_name")
    )

    return {
        "ticker": str(payload["ticker"]).strip().upper(),
        "company_name": str(payload["company_name"]).strip(),
        "peer_ticker": (
            str(peer_ticker).strip().upper()
            if peer_ticker
            else None
        ),
        "peer_name": str(peer_name).strip() if peer_name else None,
        "financial_context": [],
        "risk_context": [],
        "peer_context": [],
        "final_memo": "",
    }


if __name__ == "__main__":
    test_result = build_due_diligence_graph().invoke(
        normalize_graph_input({
            "ticker": "NVDA",
            "company_name": "NVIDIA CORP",
            "peer_ticker": "AMD",
            "peer_name": "ADVANCED MICRO DEVICES INC",
        })
    )
    print("Test run completed successfully.")