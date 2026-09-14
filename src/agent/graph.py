import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.path.append(os.getcwd())
from src.retrieval.retriever import SECRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enterprise_graph")

load_dotenv()

logger.info("Waking up AI Retrieval and Generation Engines...")
retriever = SECRetriever(Path("data/chroma"))
#  Google api
llm = ChatGoogleGenerativeAI( model="gemini-3.5-flash-lite", temperature=0.0, max_output_tokens=2500, max_retries=3 )

# ---------------------------------------------------------
# DYNAMIC STATE SCHEMA
# ---------------------------------------------------------
class MemoState(TypedDict):
    company_name: str
    ticker: str
    year: str
    peers: List[str]
    financial_data: str
    risk_data: str
    peer_data: str
    final_memo: str

def format_docs_with_citations(docs, max_chars_per_doc=4000):
    if not docs:
        return "No data found."
    formatted_chunks = []
    for doc in docs:
        doc_id = doc.get("meta", {}).get("document_id", "Unknown")
        content = doc.get("text", "")[:max_chars_per_doc]
        formatted_chunks.append(f"--- SOURCE ID: {doc_id} ---\n{content}")
    return "\n\n".join(formatted_chunks)

# ---------------------------------------------------------
# AGENTS WITH FINANCIAL-ALIGNED QUERIES
# ---------------------------------------------------------
def financial_analyst_agent(state: MemoState):
    logger.info(f"Agent 1: Extracting Financial Data for {state['company_name']} ({state['ticker']})...")
    # Accounting-aligned query: targets standard SEC Income Statement headers
    query = (
        f"Consolidated Statements of Operations Net Sales Revenue "
        f"Product breakdown by segment {state['company_name']} {state['ticker']}"
    )
    docs = retriever.search(
        query=query, 
        top_k=5, 
        rerank_top_k=3, 
        metadata_filter={"chunk_type": "table", "ticker": state["ticker"]}, 
        score_cliff=0.15
    )
    return {"financial_data": format_docs_with_citations(docs)}

def risk_analyst_agent(state: MemoState):
    logger.info(f"Agent 2: Extracting Risk Factors for {state['company_name']}...")
    # Legal-aligned query: targets Item 1A disclosures
    query = (
        f"Item 1A Risk Factors business risks regulatory legal competition "
        f"market exposure {state['company_name']} {state['ticker']}"
    )
    docs = retriever.search(
        query=query, 
        top_k=5, 
        rerank_top_k=3, 
        metadata_filter={"chunk_type": "text", "ticker": state["ticker"]}, 
        score_cliff=0.15
    )
    return {"risk_data": format_docs_with_citations(docs)}

def peer_analyst_agent(state: MemoState):
    logger.info(f"Agent 3: Extracting Peer Analysis for {', '.join(state['peers'])}...")
    if not state["peers"]:
        return {"peer_data": "No peer comparison requested."}
        
    peer_contexts = []
    for peer in state["peers"]:
        query = f"Consolidated Statements of Operations Total Net Sales Revenue {peer}"
        docs = retriever.search(
            query=query, 
            top_k=5, 
            rerank_top_k=3, 
            metadata_filter={"chunk_type": "table", "ticker": peer},
            score_cliff=0.15
        )
        peer_contexts.append(f"PEER [{peer}]:\n" + format_docs_with_citations(docs))
        
    return {"peer_data": "\n\n".join(peer_contexts)}

def portfolio_manager_agent(state: MemoState):
    logger.info(f"Agent 4: Synthesizing Investment Memo for {state['company_name']}...")
    
    # Fully dynamic system prompt with no hardcoded ticker names
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an elite Lead Portfolio Manager compiling a formal Investment Memo.\n\n"
         "STRICT RULES:\n"
         "1. PROVENANCE: You MUST cite the exact 'SOURCE ID' in brackets for every financial figure or risk factor you cite (e.g., [{ticker}_10-K_2025_table_14]).\n"
         "2. ZERO CROSS-CONTAMINATION: When reporting peer data, ONLY use data that originates from that peer's specific SOURCE ID. Never attribute {company_name}'s ({ticker}) metrics to a peer.\n"
         "3. TRANSPARENCY: If peer data is not present in the provided context, explicitly state that filing data for the peer is unavailable rather than fabricating or substituting metrics.\n\n"
         "FINANCIAL CONTEXT:\n{financial_data}\n\n"
         "RISK FACTOR CONTEXT:\n{risk_data}\n\n"
         "PEER COMPETITION CONTEXT:\n{peer_data}\n\n"
         "Produce a structured memo with Executive Summary, Financial Performance, Peer Comparison, and Key Risks."),
        ("human", "Compile the comprehensive investment memo for {company_name} ({ticker}) for fiscal year {year}.")
    ])
    
    chain = prompt | llm | StrOutputParser()
    memo = chain.invoke({
        "company_name": state["company_name"],
        "ticker": state["ticker"],
        "year": state["year"],
        "financial_data": state["financial_data"],
        "risk_data": state["risk_data"],
        "peer_data": state["peer_data"]
    })
    return {"final_memo": memo}

# ---------------------------------------------------------
# GRAPH PIPELINE ASSEMBLY
# ---------------------------------------------------------
def build_due_diligence_graph():
    workflow = StateGraph(MemoState)
    
    workflow.add_node("financial_analyst", financial_analyst_agent)
    workflow.add_node("risk_analyst", risk_analyst_agent)
    workflow.add_node("peer_analyst", peer_analyst_agent)
    workflow.add_node("portfolio_manager", portfolio_manager_agent)
    
    workflow.add_edge(START, "financial_analyst")
    workflow.add_edge("financial_analyst", "risk_analyst")
    workflow.add_edge("risk_analyst", "peer_analyst")
    workflow.add_edge("peer_analyst", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)
    
    return workflow.compile()

if __name__ == "__main__":
    app = build_due_diligence_graph()
    
    # Fully dynamic parameter dictionary: easily driven by UI inputs
    test_inputs = {
        "company_name": "Apple",
        "ticker": "AAPL",
        "year": "2025",
        "peers": ["MSFT"],
        "financial_data": "",
        "risk_data": "",
        "peer_data": "",
        "final_memo": ""
    }
    
    result = app.invoke(test_inputs)
    print("\n" + "="*80)
    print("📝 GENERATED INVESTMENT MEMO")
    print("="*80)
    print(result["final_memo"])