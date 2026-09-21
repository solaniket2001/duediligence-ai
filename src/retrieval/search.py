import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.agent.graph import build_due_diligence_graph, normalize_graph_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("api_server")

app = FastAPI(title="Argus Due Diligence API", version="2.0.0")

# Initialize the LangGraph engine globally so it boots up once when the server starts
logger.info("Initializing LangGraph Multi-Agent Engine...")
graph = build_due_diligence_graph()

class AnalysisRequest(BaseModel):
    ticker: str
    company_name: str
    peer_ticker: Optional[str] = None
    peer_name: Optional[str] = None

@app.post("/api/analyze")
def run_analysis(payload: AnalysisRequest):
    logger.info("Received execution request: %s vs %s", payload.ticker, payload.peer_ticker)
    try:
        initial_state = normalize_graph_input(payload.dict())
        result = graph.invoke(initial_state)
        
        return {
            "ticker": result.get("ticker"),
            "company_name": result.get("company_name"),
            "peer_ticker": result.get("peer_ticker"),
            "peer_name": result.get("peer_name"),
            "financial_score": result.get("financial_score", 0.0),
            "financial_retries": result.get("financial_retry_count", 0),
            "quantitative_analysis": result.get("quantitative_analysis", ""),
            "news_context": result.get("news_context", ""),
            "final_memo": result.get("final_memo", ""),
        }
    except Exception as e:
        logger.error("Pipeline failure: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Start the server on port 8000
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)