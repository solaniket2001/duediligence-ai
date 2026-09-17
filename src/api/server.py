import os
import sys
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# Ensure the src module is in the path
sys.path.append(os.getcwd())
from src.agent.graph import build_due_diligence_graph, normalize_graph_input
from src.utils.ticker_resolver import resolve_ticker

app = FastAPI(
    title="Due Diligence AI Engine",
    description="Autonomous multi-agent SEC financial analysis",
    version="1.0.0"
)

# 1. Define Request/Response Models
class DueDiligenceRequest(BaseModel):
    target: str
    peer: Optional[str] = None
    target_year: str = "2025"

class DueDiligenceResponse(BaseModel):
    target_ticker: str
    target_name: str
    peer_ticker: Optional[str]
    peer_name: Optional[str]
    memo_markdown: str

# 2. Build the Graph once at startup
app_graph = build_due_diligence_graph()

@app.get("/")
async def health_check():
    return {
        "status": "online", 
        "engine": "Due Diligence API", 
        "docs": "Visit /docs for the Swagger UI"
    }

@app.post("/api/v1/analyze", response_model=DueDiligenceResponse)
async def analyze_company(request: DueDiligenceRequest):
    """
    Main endpoint to trigger the multi-agent due diligence pipeline.
    """
    try:
        # Resolve tickers
        target_ticker, target_name = resolve_ticker(request.target)
        peer_ticker, peer_name = resolve_ticker(request.peer) if request.peer else (None, None)
        
        if not target_ticker:
            raise HTTPException(status_code=400, detail=f"Invalid target ticker: {request.target}")

        # Normalize input for LangGraph
        graph_input = normalize_graph_input({
            "ticker": target_ticker,
            "company_name": target_name,
            "peer_ticker": peer_ticker,
            "peer_name": peer_name,
        })

        # Run the graph asynchronously to avoid blocking the FastAPI event loop
        # .ainvoke() is the async version of .invoke()
        result = await app_graph.ainvoke(graph_input)

        return DueDiligenceResponse(
            target_ticker=target_ticker,
            target_name=target_name,
            peer_ticker=peer_ticker,
            peer_name=peer_name,
            memo_markdown=result["final_memo"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))