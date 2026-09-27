import os
import sys
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.getcwd())
from src.agent.graph import build_due_diligence_graph, normalize_graph_input
from src.db.database import engine, Base, get_db
from src.db.models import DueDiligenceReport

try:
    from src.utils.ticker_resolver import resolve_ticker
except ImportError:
    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield 
    await engine.dispose()

app = FastAPI(
    title="Due Diligence AI Engine",
    description="Autonomous multi-agent SEC financial analysis",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows seamless connection through Codespace port forwarding
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    ticker: str
    company_name: str
    peer_ticker: Optional[str] = None
    peer_name: Optional[str] = None

app_graph = build_due_diligence_graph()

@app.get("/")
async def health_check():
    return {"status": "online", "engine": "Due Diligence API"}

@app.post("/api/analyze")
async def run_analysis(payload: AnalysisRequest, db: AsyncSession = Depends(get_db)):
    
    async def event_generator():
        try:
            graph_input = normalize_graph_input(payload.dict())
            
            # 1. Notify frontend that process has started
            yield f"data: {json.dumps({'type': 'log', 'message': f'System Boot: Initializing Due Diligence Engine for {payload.ticker}...'})}\n\n"
            
            final_state = None
            
            # 2. Stream LangGraph node updates dynamically as they finish
            async for output in app_graph.astream(graph_input, stream_mode="updates"):
                for node_name, node_state in output.items():
                    # If the node yielded a live_log, stream it immediately to the UI
                    if isinstance(node_state, dict) and "live_log" in node_state:
                        msg = node_state["live_log"]
                        yield f"data: {json.dumps({'type': 'log', 'message': msg})}\n\n"
                    
                    # Update our running state
                    if final_state is None:
                        final_state = {}
                    final_state.update(node_state)
                
            # 3. Extract final values
            target_ticker = final_state.get("ticker", payload.ticker)
            company_name = final_state.get("company_name", payload.company_name)
            peer_ticker = final_state.get("peer_ticker")
            risk_score = final_state.get("risk_score", 50)
            final_memo = final_state.get("final_memo", "")
            
            # 4. Save to Database
            yield f"data: {json.dumps({'type': 'log', 'message': 'Saving final report to PostgreSQL...'})}\n\n"
            
            new_report = DueDiligenceReport(
                target_ticker=target_ticker,
                target_name=company_name,
                peer_ticker=peer_ticker,
                financial_score=final_state.get("financial_score", 0.0),
                risk_score=risk_score,
                quantitative_analysis=final_state.get("quantitative_analysis", ""),
                news_context=final_state.get("news_context", ""),
                final_memo=final_memo,
                market_data=final_state.get("market_data", {})
            )
            db.add(new_report)
            await db.commit()
            
            # 5. Send the final compiled payload to trigger the UI render
            final_payload = {
                "ticker": target_ticker,
                "company_name": company_name,
                "peer_ticker": peer_ticker,
                "peer_name": final_state.get("peer_name"),
                "financial_score": final_state.get("financial_score", 0.0),
                "financial_retries": final_state.get("financial_retry_count", 0),
                "quantitative_analysis": final_state.get("quantitative_analysis", ""),
                "news_context": final_state.get("news_context", ""),
                "final_memo": final_memo,
                "risk_score": risk_score,
                "market_data": final_state.get("market_data", {})
            }
            
            yield f"data: {json.dumps({'type': 'complete', 'data': final_payload})}\n\n"
            
        except Exception as e:
            # If the system crashes (like the memory error), send it directly to the UI!
            yield f"data: {json.dumps({'type': 'error', 'message': f'System Error: {str(e)}'})}\n\n"

    # Wrap the generator in a StreamingResponse with the SSE MIME type
    return StreamingResponse(event_generator(), media_type="text/event-stream")