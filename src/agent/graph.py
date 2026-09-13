import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Ensure Python can find our custom retriever module
sys.path.append(os.getcwd())
from src.retrieval.retriever import SECRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enterprise_graph")

load_dotenv()

logger.info("Waking up AI Retrieval and Generation Engines...")
retriever = SECRetriever(Path("data/chroma"))
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0, max_tokens=2500)

# ---------------------------------------------------------
# 1. UPGRADED STATE (Added Peer Tracking)
# ---------------------------------------------------------
class MemoState(TypedDict):
    company_name: str
    year: str
    peers: List[str]
    financial_data: str
    risk_data: str
    peer_data: str
    final_memo: str

# Helper to format context with explicit Citations
# Helper to format context with explicit Citations
def format_docs_with_citations(docs):
    if not docs:
        return "No data found."
    
    # Extract data from the dictionary structure returned by FlashRank
    formatted_chunks = []
    for doc in docs:
        # Access the dictionary keys: 'meta' and 'text'
        doc_id = doc.get("meta", {}).get("document_id", "Unknown")
        content = doc.get("text", "")
        formatted_chunks.append(f"--- SOURCE ID: {doc_id} ---\n{content}")
        
    return "\n\n".join(formatted_chunks)

# ---------------------------------------------------------
# 2. THE AGENTS
# ---------------------------------------------------------
def financial_analyst_agent(state: MemoState):
    logger.info("Agent 1: Extracting Financial Data...")
    docs = retriever.search(
        query=f"What are the net sales, product breakdown, and revenue figures for {state['company_name']}?", 
        top_k=5, rerank_top_k=2, 
        metadata_filter={"year": state["year"], "chunk_type": "table"}, 
        score_cliff=0.15
    )
    return {"financial_data": format_docs_with_citations(docs)}

def risk_analyst_agent(state: MemoState):
    logger.info("Agent 2: Extracting Risk Factors...")
    docs = retriever.search(
        query=f"What are the primary business risks, regulatory challenges, and competition risks for {state['company_name']}?", 
        top_k=5, rerank_top_k=3, 
        metadata_filter={"year": state["year"], "chunk_type": "text"}, 
        score_cliff=0.15
    )
    return {"risk_data": format_docs_with_citations(docs)}

def peer_analyst_agent(state: MemoState):
    logger.info(f"Agent 3: Extracting Peer Analysis for {', '.join(state['peers'])}...")
    if not state["peers"]:
        return {"peer_data": "No peer comparison requested."}
        
    peer_contexts = []
    for peer in state["peers"]:
        docs = retriever.search(
            query=f"What is the total revenue and net sales for {peer}?", 
            top_k=3, rerank_top_k=1, 
            metadata_filter={"year": state["year"], "chunk_type": "table"}
        )
        peer_contexts.append(f"PEER: {peer}\n" + format_docs_with_citations(docs))
        
    return {"peer_data": "\n\n".join(peer_contexts)}

def portfolio_manager_agent(state: MemoState):
    logger.info("Agent 4: Synthesizing Enterprise Investment Memo...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an elite Lead Portfolio Manager. Compile a formal Investment Memo using ONLY the subordinate agents' findings.\n\n"
         "CRITICAL INSTRUCTION: You MUST cite the 'SOURCE ID' in brackets for every metric or risk factor you mention (e.g., [AAPL_10-K_2025_table_4]).\n\n"
         "FINANCIAL DATA:\n{financial_data}\n\n"
         "RISK FACTORS:\n{risk_data}\n\n"
         "PEER COMPETITION DATA:\n{peer_data}\n\n"
         "Format the memo with professional markdown, including an Executive Summary, Financial Performance, Peer Comparison, and Key Risks."),
        ("human", "Write the investment memo for {company_name} ({year}).")
    ])
    
    chain = prompt | llm | StrOutputParser()
    memo = chain.invoke({
        "financial_data": state["financial_data"],
        "risk_data": state["risk_data"],
        "peer_data": state["peer_data"],
        "company_name": state["company_name"],
        "year": state["year"]
    })
    return {"final_memo": memo}

# ---------------------------------------------------------
# 3. BUILD THE ASSEMBLY LINE
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
    
    # We initialize the state with Microsoft as a peer
    inputs = {
        "company_name": "Apple",
        "year": "2025",
        "peers": ["Microsoft"], 
        "financial_data": "",
        "risk_data": "",
        "peer_data": "",
        "final_memo": ""
    }
    
    result = app.invoke(inputs)
    
    print("\n" + "="*80)
    print("📝 ENTERPRISE INVESTMENT MEMO (WITH CITATIONS) 📝")
    print("="*80)
    print(result["final_memo"])
    print("="*80)