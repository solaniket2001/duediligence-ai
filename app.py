import os
import re
import json
import requests
import markdown
import plotly.graph_objects as go
from fpdf import FPDF, XPos, YPos
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from plotly.subplots import make_subplots

load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Argus | Autonomous Due Diligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Markdown Sanitizer ---
def clean_markdown_for_streamlit(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'\\+(\$?\d)', r'\1', text)
    cleaned = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1 / \2)', cleaned)
    cleaned = re.sub(r'(?<!\\)\$(\d)', r'\\$\1', cleaned)
    return cleaned

# --- Plotly Renderers ---
def plot_stock_history(target_ticker, peer_ticker, market_data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    target_hist = market_data.get("target", {})
    if target_hist and target_hist.get("history_dates"):
        fig.add_trace(
            go.Scatter(
                x=target_hist["history_dates"], 
                y=target_hist["history_prices"], 
                mode='lines',
                name=f"{target_ticker} (Left)",
                line=dict(color='#e11d48', width=2)
            ),
            secondary_y=False
        )
        
    peer_hist = market_data.get("peer", {})
    if peer_hist and peer_hist.get("history_dates"):
        fig.add_trace(
            go.Scatter(
                x=peer_hist["history_dates"], 
                y=peer_hist["history_prices"], 
                mode='lines',
                name=f"{peer_ticker} (Right)",
                line=dict(color='#3b82f6', width=2)
            ),
            secondary_y=True
        )
        
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False),
        yaxis=dict(title=f"{target_ticker} ($)", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        yaxis2=dict(title=f"{peer_ticker} ($)", showgrid=False),
        height=280
    )
    return fig

def plot_risk_gauge(score):
    color = "#34d399" if score < 40 else "#fbbf24" if score < 70 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'color': color, 'size': 36}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.2)"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': "rgba(52, 211, 153, 0.1)"},
                {'range': [40, 70], 'color': "rgba(251, 191, 36, 0.1)"},
                {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.1)"}
            ]
        }
    ))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=10, b=20),
        height=240
    )
    return fig

# --- Interrogation Mode Backend ---
def get_memo_answer(question: str, memo: str) -> str:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite", 
            google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        messages = [
            SystemMessage(content=f"You are Argus, an elite financial AI. Answer the user's question based strictly on this investment memo. If the answer is not in the memo, say so.\n\nMEMO CONTEXT:\n{memo}")
        ]
        for msg in st.session_state.get("chat_history", []):
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
                
        messages.append(HumanMessage(content=question))
        response = llm.invoke(messages)
        
        if isinstance(response.content, list):
            return "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in response.content)
        return str(response.content)
    except Exception as e:
        return f"Interrogation failed: {e}"

# --- CSS Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background: radial-gradient(circle at top right, #111827, #0b0f17 40%, #030712 100%); color: #f3f4f6; }
    .bento-card {
        background: rgba(17, 24, 39, 0.55); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5); margin-bottom: 20px; transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .bento-card:hover { border-color: rgba(225, 29, 72, 0.4); }
    .metric-pill {
        display: inline-flex; align-items: center; gap: 6px; background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08); padding: 4px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 500; color: #94a3b8;
    }
    .status-badge {
        display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 0.05em; padding: 4px 10px; border-radius: 6px; font-weight: 600;
    }
    .badge-live { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-score { background: rgba(225, 29, 72, 0.15); color: #fb7185; border: 1px solid rgba(225, 29, 72, 0.3); }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #e11d48, #be123c); color: #ffffff; border: none; border-radius: 8px; font-weight: 600;
        padding: 10px 24px; transition: all 0.2s ease; box-shadow: 0 4px 15px rgba(225, 29, 72, 0.3);
    }
    div.stButton > button:first-child:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(225, 29, 72, 0.45); }
</style>
""", unsafe_allow_html=True)

# --- PDF Generation Pipeline ---
class PDFMaker(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_pdf(memo_text: str, target: str, peer: str) -> bytes:
    pdf = PDFMaker()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Investment Memo: {target} vs {peer or 'Independent'}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
    pdf.ln(4)
    clean_text = memo_text.replace("’", "'").replace("‘", "'").replace('“', '"').replace('”', '"').replace("—", "-").replace("–", "-").replace("\\$", "$").replace("\\\\$", "$")
    clean_text = re.sub(r"([A-Za-z0-9\]\.])\s+(\*\s+Item)", r"\1\n\n\2", clean_text)
    html_content = markdown.markdown(clean_text).replace("\\$", "$")
    pdf.write_html(html_content)
    return bytes(pdf.output())


# --- Sidebar Navigation & Input ---
with st.sidebar:
    st.markdown("### ⚡ Engine Config")
    st.markdown("<p style='font-size:0.85rem; color:#94a3b8;'>Multi-agent SEC EDGAR hybrid evaluation pipeline with cyclic self-reflection.</p>", unsafe_allow_html=True)
    st.divider()

    target_ticker = st.text_input("Target Ticker", value="TSLA").upper().strip()
    target_name = st.text_input("Target Name", value="Tesla, Inc.").strip()

    st.markdown("---")
    peer_ticker_input = st.text_input("Peer Ticker (Leave blank for Auto-Discovery)", value="").upper().strip()
    peer_name_input = st.text_input("Peer Name", value="").strip()

    st.divider()
    use_api = st.checkbox("Connect via FastAPI (port 8000)", value=True)
    execute_btn = st.button("Generate Due Diligence", width='stretch')

# --- Dashboard Header ---
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 24px;">
    <div>
        <h1 style="margin:0; font-size:2rem; font-weight:700; letter-spacing:-0.03em;">ARGUS // FINANCIAL INTELLIGENCE</h1>
        <p style="margin:0; font-size:0.9rem; color:#94a3b8;">Autonomous SEC 10-K RAG • Quantitative Math Engine • Live Sentiment</p>
    </div>
    <div style="display:flex; gap:10px;">
        <span class="status-badge badge-live">LIVE API ACTIVE</span>
        <span class="status-badge badge-score">HYDE ENABLED</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Execution Controller ---
if execute_btn:
    st.session_state["chat_history"] = []
    payload = {
        "ticker": target_ticker,
        "company_name": target_name,
        "peer_ticker": peer_ticker_input if peer_ticker_input else None,
        "peer_name": peer_name_input if peer_name_input else None,
    }

    with st.status("Executing Multi-Agent Workflow...", expanded=True) as status:
        st.write("🔍 Running HyDE & Multi-Tower Search...")
        st.write("📈 Plotting Market Fundamentals & Peer Discovery...")
        st.write("📊 Crunching Quantitative Ratios & YoY Margins...")
        st.write("🌐 Scraping Live Financial News & Sentiment...")

        try:
            if use_api:
                res = requests.post("http://localhost:8000/api/analyze", json=payload, timeout=3000)
                data = res.json()
            else:
                from src.agent.graph import build_due_diligence_graph, normalize_graph_input
                graph = build_due_diligence_graph()
                data = graph.invoke(normalize_graph_input(payload))

            st.session_state["analysis_data"] = data
            status.update(label="Analysis Pipeline Complete", state="complete", expanded=False)
        except Exception as err:
            status.update(label=f"Execution Failed: {err}", state="error")
            st.stop()

# --- Main Bento Grid Display ---
if "analysis_data" in st.session_state:
    data = st.session_state["analysis_data"]
    
    # 1. Top Metric Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="bento-card" style="padding:16px;">
            <div style="font-size:0.75rem; color:#94a3b8;">TARGET ENTITY</div>
            <div style="font-size:1.2rem; font-weight:700;">{data.get('company_name', target_name)}</div>
            <span class="metric-pill" style="margin-top:6px;">{data.get('ticker', target_ticker)}</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="bento-card" style="padding:16px;">
            <div style="font-size:0.75rem; color:#94a3b8;">BENCHMARK PEER</div>
            <div style="font-size:1.2rem; font-weight:700;">{data.get('peer_name') or 'AUTONOMOUS'}</div>
            <span class="metric-pill" style="margin-top:6px;">{data.get('peer_ticker') or 'DISCOVERED'}</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        score_val = data.get("financial_score", 0.0)
        st.markdown(f"""
        <div class="bento-card" style="padding:16px;">
            <div style="font-size:0.75rem; color:#94a3b8;">RETRIEVAL CONFIDENCE</div>
            <div style="font-size:1.2rem; font-weight:700; color:{'#34d399' if score_val >= 0.80 else '#fb7185'};">
                {score_val:.4f}
            </div>
            <span class="metric-pill" style="margin-top:6px;">Threshold: 0.80</span>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="bento-card" style="padding:16px;">
            <div style="font-size:0.75rem; color:#94a3b8;">REFLECTION LOOPS</div>
            <div style="font-size:1.2rem; font-weight:700;">{data.get('financial_retries', 0)}</div>
            <span class="metric-pill" style="margin-top:6px;">Self-Correct Cycles</span>
        </div>
        """, unsafe_allow_html=True)

    # 2. Charts Row (Plotly Integration)
    st.markdown("<h3 style='font-size:1.1rem; font-weight:600; margin-top:10px; margin-bottom:12px;'>📊 Live Market & Risk Analytics</h3>", unsafe_allow_html=True)
    col_chart, col_gauge = st.columns([3, 1])
    
    with col_chart:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.8rem; color:#94a3b8; margin-bottom:10px; font-weight:600;'>6-MONTH PRICE ACTION</div>", unsafe_allow_html=True)
        market_data = data.get("market_data", {})
        if market_data:
            st.plotly_chart(plot_stock_history(data.get("ticker", ""), data.get("peer_ticker", ""), market_data), width='stretch')
        else:
            st.info("Market history data temporarily unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_gauge:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.8rem; color:#94a3b8; text-align:center; font-weight:600;'>QUANTITATIVE RISK SCORE</div>", unsafe_allow_html=True)
        risk_score = data.get("risk_score", 50)
        st.plotly_chart(plot_risk_gauge(risk_score), width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Intelligence Cards Row
    grid_left, grid_right = st.columns([1, 1])
    with grid_left:
        st.markdown("""
        <div class="bento-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <h3 style="margin:0; font-size:1.1rem; font-weight:600;">📈 Quantitative Reasoning</h3>
                <span class="status-badge badge-live">MATH AGENT</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(clean_markdown_for_streamlit(data.get("quantitative_analysis", "") or "No quantitative ratios derived."))
        st.markdown("</div>", unsafe_allow_html=True)

    with grid_right:
        st.markdown("""
        <div class="bento-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <h3 style="margin:0; font-size:1.1rem; font-weight:600;">🌐 Live Market Outlook</h3>
                <span class="status-badge badge-score">RSS SENTIMENT</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(clean_markdown_for_streamlit(data.get("news_context", "") or "No real-time market news retrieved."))
        st.markdown("</div>", unsafe_allow_html=True)

    # 4. Full Memo Row & PDF Export
    st.markdown("""
    <div class="bento-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
            <h3 style="margin:0; font-size:1.25rem; font-weight:600;">📑 Executive Investment Memorandum</h3>
        </div>
    """, unsafe_allow_html=True)
    memo_text = data.get("final_memo", "")
    st.markdown(clean_markdown_for_streamlit(memo_text))
    st.divider()

    pdf_bytes = generate_pdf(memo_text=memo_text, target=data.get("ticker", target_ticker), peer=data.get("peer_ticker", ""))
    st.download_button(
        label="📥 Download Institutional PDF Memo",
        data=pdf_bytes,
        file_name=f"{data.get('ticker')}_Memo.pdf",
        mime="application/pdf"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 5. Interrogation Mode
    st.divider()
    st.markdown("### 💬 Interrogation Mode")
    st.markdown("<p style='color:#94a3b8; margin-bottom:20px;'>Talk directly to the Argus engine to cross-examine this investment memo.</p>", unsafe_allow_html=True)
    
    for msg in st.session_state.get("chat_history", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Ask a follow-up question about the financial data or risks..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing memo..."):
                answer = get_memo_answer(prompt, memo_text)
                st.markdown(answer)
                st.session_state["chat_history"].append({"role": "user", "content": prompt})
                st.session_state["chat_history"].append({"role": "assistant", "content": answer})