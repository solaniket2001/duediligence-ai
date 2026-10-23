import os
import sys
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List, Dict

from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.getcwd())
from src.agent.graph import (
    set_live_log_queue,
    build_due_diligence_graph,
    normalize_graph_input,
    fetch_context_chunks,
    get_llm,
    extract_text_from_response,
)
from src.db.database import engine, Base, get_db
from src.db.models import DueDiligenceReport


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(
    title="Due Diligence AI Engine",
    description="Autonomous multi-agent SEC financial analysis & Live Interrogation API",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    ticker: str
    company_name: str
    peer_ticker: Optional[str] = None
    peer_name: Optional[str] = None


class ChatMessageItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    ticker: str
    company_name: Optional[str] = None
    peer_ticker: Optional[str] = None
    memo: str = ""
    quantitative_analysis: str = ""
    news_context: str = ""
    chat_history: List[ChatMessageItem] = []


app_graph = build_due_diligence_graph()


@app.get("/")
async def health_check():
    return {"status": "online", "engine": "Due Diligence API"}


@app.post("/api/analyze")
async def run_analysis(payload: AnalysisRequest, db: AsyncSession = Depends(get_db)):
    async def analysis_generator():
        try:
            payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
            graph_input = normalize_graph_input(payload_dict)

            yield f"data: {json.dumps({'type': 'log', 'message': f'System Boot: Initializing Due Diligence Engine for {payload.ticker}...'})}\n\n"

            final_state = dict(graph_input)

            log_queue: asyncio.Queue = asyncio.Queue()
            set_live_log_queue(log_queue)
            emitted_logs = set()

            async def run_graph_stream():
                try:
                    async for output in app_graph.astream(graph_input, stream_mode="updates"):
                        for _, node_state in output.items():
                            if isinstance(node_state, dict):
                                final_state.update(node_state)
                finally:
                    await log_queue.put(None)

            graph_task = asyncio.create_task(run_graph_stream())
            while True:
                msg = await log_queue.get()
                if msg is None:
                    break
                if msg not in emitted_logs:
                    emitted_logs.add(msg)
                    yield f"data: {json.dumps({'type': 'log', 'message': msg})}\n\n"
            await graph_task
            set_live_log_queue(None)

            target_ticker = final_state.get("ticker", payload.ticker)
            company_name = final_state.get("company_name", payload.company_name)
            peer_ticker = final_state.get("peer_ticker")
            risk_score = final_state.get("risk_score", 50)
            final_memo = final_state.get("final_memo", "")

            yield f"data: {json.dumps({'type': 'log', 'message': 'Saving final report to PostgreSQL...'})}\n\n"

            try:
                new_report = DueDiligenceReport(
                    target_ticker=target_ticker,
                    target_name=company_name,
                    peer_ticker=peer_ticker,
                    financial_score=final_state.get("financial_score", 0.0),
                    risk_score=risk_score,
                    quantitative_analysis=final_state.get("quantitative_analysis", ""),
                    news_context=final_state.get("news_context", ""),
                    final_memo=final_memo,
                    market_data=final_state.get("market_data", {}),
                )
                db.add(new_report)
                await db.commit()
            except Exception as db_err:
                yield f"data: {json.dumps({'type': 'log', 'message': f'[Warning] DB save skipped: {str(db_err)}'})}\n\n"

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
                "market_data": final_state.get("market_data", {}),
            }

            yield f"data: {json.dumps({'type': 'complete', 'data': final_payload})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'System Error: {str(e)}'})}\n\n"

    return StreamingResponse(
        analysis_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat")
async def interrogate_memo(payload: ChatRequest):
    """Live Chat Analysis / Interrogation Mode backed by Memo + live SEC ChromaDB RAG."""
    try:
        # Retrieve additional raw SEC 10-K chunks for the specific question
        sec_chunks = await asyncio.to_thread(
            fetch_context_chunks, payload.question, payload.ticker, 4
        )
        peer_chunks: List[str] = []
        if payload.peer_ticker and payload.peer_ticker not in {"NONE", "N/A"}:
            peer_chunks = await asyncio.to_thread(
                fetch_context_chunks, payload.question, payload.peer_ticker, 2
            )

        rag_block = "\n\n".join(sec_chunks + peer_chunks)

        system_instruction = (
            f"You are Argus (KORE Engine), an elite institutional financial analyst AI. "
            f"You are in Live Interrogation Mode for {payload.company_name or payload.ticker} ({payload.ticker})"
            f"{f' benchmarked against {payload.peer_ticker}' if payload.peer_ticker else ''}.\n\n"
            "Answer the user's question with institutional rigor using the Investment Memo, Quantitative Analysis, "
            "and freshly retrieved SEC 10-K filing chunks below. Cite specific document IDs (e.g., [DOC_ID]) when referencing SEC filing data. "
            "Do not use LaTeX math syntax; write formulas and numbers in clean plain text or Markdown tables. "
            "If the information is genuinely absent from both the memo and SEC filings, state that clearly.\n\n"
            f"=== INVESTMENT MEMORANDUM ===\n{payload.memo}\n\n"
            f"=== QUANTITATIVE ANALYSIS ===\n{payload.quantitative_analysis}\n\n"
            f"=== LIVE NEWS CONTEXT ===\n{payload.news_context}\n\n"
            f"=== DIRECT SEC 10-K RETRIEVAL CHUNKS ===\n{rag_block or 'No additional SEC chunks retrieved.'}"
        )

        messages = [SystemMessage(content=system_instruction)]
        for msg in payload.chat_history[-10:]:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))

        messages.append(HumanMessage(content=payload.question))

        llm = get_llm()
        response = await llm.ainvoke(messages)
        answer = extract_text_from_response(response.content).strip()

        return {
            "answer": answer,
            "sources_count": len(sec_chunks) + len(peer_chunks),
        }
    except Exception as exc:
        return {
            "answer": f"Interrogation Error: {str(exc)}",
            "sources_count": 0,
        }