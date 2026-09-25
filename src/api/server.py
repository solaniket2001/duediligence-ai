import os
import sys
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# Ensure the src module is in the path
sys.path.append(os.getcwd())
from src.agent.graph import build_due_diligence_graph, normalize_graph_input
from src.db.database import engine, Base, get_db
from src.db.models import DueDiligenceReport

# Fallback in case ticker_resolver is deprecated in your new workflow
try:
    from src.utils.ticker_resolver import resolve_ticker
except ImportError:
    pass

# 1. Define the Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Auto-create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield  # Application runs and handles requests here
    
    # Shutdown: Cleanly dispose of the connection pool
    await engine.dispose()

# 2. Initialize FastAPI with the lifespan handler
app = FastAPI(
    title="Due Diligence AI Engine",
    description="Autonomous multi-agent SEC financial analysis",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 3. Define Request/Response Models aligning with Streamlit UI
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

# 4. Build the Graph once at startup
app_graph = build_due_diligence_graph()

@app.get("/")
async def health_check():
    return {
        "status": "online", 
        "engine": "Due Diligence API", 
        "docs": "Visit /docs for the Swagger UI"
    }

@app.post("/api/analyze", response_model=AnalysisResponse)
async def run_analysis(payload: AnalysisRequest, db: AsyncSession = Depends(get_db)):
    try:
        graph_input = normalize_graph_input(payload.dict())
        result = await app_graph.ainvoke(graph_input)
        
        # Extract values
        target_ticker = result.get("ticker", payload.ticker)
        company_name = result.get("company_name", payload.company_name)
        peer_ticker = result.get("peer_ticker")
        risk_score = result.get("risk_score", 50)
        final_memo = result.get("final_memo", "")
        
        # Save to PostgreSQL
        new_report = DueDiligenceReport(
            target_ticker=target_ticker,
            target_name=company_name,
            peer_ticker=peer_ticker,
            financial_score=result.get("financial_score", 0.0),
            risk_score=risk_score,
            quantitative_analysis=result.get("quantitative_analysis", ""),
            news_context=result.get("news_context", ""),
            final_memo=final_memo,
            market_data=result.get("market_data", {})
        )
        db.add(new_report)
        await db.commit()
        await db.refresh(new_report)

        # Return to Streamlit UI
        return AnalysisResponse(
            ticker=target_ticker,
            company_name=company_name,
            peer_ticker=peer_ticker,
            peer_name=result.get("peer_name"),
            financial_score=result.get("financial_score", 0.0),
            financial_retries=result.get("financial_retry_count", 0),
            quantitative_analysis=result.get("quantitative_analysis", ""),
            news_context=result.get("news_context", ""),
            final_memo=final_memo,
            risk_score=risk_score,
            market_data=result.get("market_data", {})
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))