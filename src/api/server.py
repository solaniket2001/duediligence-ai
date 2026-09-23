import os
import sys
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv()

# Ensure the src module is in the path
sys.path.append(os.getcwd())
from src.agent.graph import build_due_diligence_graph, normalize_graph_input

# Fallback in case ticker_resolver is deprecated in your new workflow
try:
    from src.utils.ticker_resolver import resolve_ticker
except ImportError:
    pass

app = FastAPI(
    title="Due Diligence AI Engine",
    description="Autonomous multi-agent SEC financial analysis",
    version="1.0.0"
)

# 1. Define Request/Response Models aligning with Streamlit UI
class AnalysisRequest(BaseModel):
    ticker: str
    company_name: str
    peer_ticker: Optional[str] = None
    peer_name: Optional[str] = None

class AnalysisResponse(BaseModel):
    ticker: str
    company_name: str
    peer_ticker: Optional[str]
    peer_name: Optional[str]
    financial_score: float
    financial_retries: int
    quantitative_analysis: str
    news_context: str
    final_memo: str
    risk_score: int
    market_data: Dict[str, Any]

# 2. Build the Graph once at startup
app_graph = build_due_diligence_graph()

@app.get("/")
async def health_check():
    return {
        "status": "online", 
        "engine": "Due Diligence API", 
        "docs": "Visit /docs for the Swagger UI"
    }

@app.post("/api/analyze", response_model=AnalysisResponse)
async def run_analysis(payload: AnalysisRequest):
    """
    Main endpoint to trigger the multi-agent due diligence pipeline.
    """
    try:
        # Normalize input for LangGraph
        graph_input = normalize_graph_input(payload.dict())

        # Run the graph asynchronously to avoid blocking the FastAPI event loop
        result = await app_graph.ainvoke(graph_input)

        return AnalysisResponse(
            ticker=result.get("ticker", payload.ticker),
            company_name=result.get("company_name", payload.company_name),
            peer_ticker=result.get("peer_ticker"),
            peer_name=result.get("peer_name"),
            financial_score=result.get("financial_score", 0.0),
            financial_retries=result.get("financial_retry_count", 0),
            quantitative_analysis=result.get("quantitative_analysis", ""),
            news_context=result.get("news_context", ""),
            final_memo=result.get("final_memo", ""),
            risk_score=result.get("risk_score", 50),
            market_data=result.get("market_data", {})
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))