"use client";

import { useState } from "react";
import {
  Activity,
  Database,
  Terminal,
  Loader2,
  ArrowRight,
  TrendingUp,
  FileText,
  Scale,
  Newspaper,
  BarChart3,
  AlertTriangle,
  Target,
  Briefcase,
  CheckCircle2
} from "lucide-react";
import { motion } from "framer-motion";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ComposedChart,
  Legend
} from "recharts";
import ReactMarkdown from "react-markdown";

interface AnalysisResult {
  ticker: string;
  company_name: string;
  peer_ticker?: string;
  peer_name?: string;
  financial_score: number;
  financial_retries: number;
  quantitative_analysis: string;
  news_context: string;
  final_memo: string;
  risk_score: number;
  market_data: Record<string, any>;
}

// --- Parsing Utilities ---
const parseMemoSections = (text: string) => {
  if (!text) return null;
  const sections = text.split('### ');
  const result: Record<string, string> = {};

  sections.forEach(sec => {
    const lowerSec = sec.toLowerCase();
    const content = sec.replace(/^.*?\n/, '').trim(); 
    if (lowerSec.includes('executive summary')) result.summary = content;
    else if (lowerSec.includes('financial performance')) result.financial = content;
    else if (lowerSec.includes('peer comparison')) result.peer = content;
    else if (lowerSec.includes('key risks')) result.risks = content;
    else if (lowerSec.includes('market outlook')) result.outlook = content;
    else if (lowerSec.includes('conclusion')) result.conclusion = content;
  });

  return Object.keys(result).length > 0 ? result : { raw: text };
};

const parseQuantSections = (text: string) => {
  if (!text) return { target: "", peer: "", raw: text };
  
  const targetMatch = text.match(/(?:###.*|\**Target Company.*?\**\s*\n)([\s\S]*?)(?=\**Peer Company|###|$)/i);
  const peerMatch = text.match(/(?:###.*|\**Peer Company.*?\**\s*\n)([\s\S]*?)(?=\**Target Company|###|$)/i);

  const target = targetMatch ? targetMatch[1].replace(/^- /gm, '').trim() : "";
  const peer = peerMatch ? peerMatch[1].replace(/^- /gm, '').trim() : "";

  return { target, peer, raw: (!target && !peer) ? text : null };
};

// --- Custom Markdown Renderer ---
const MarkdownBlock = ({ content }: { content: string }) => (
  <ReactMarkdown
    components={{
      p: ({ node, ...props }) => <p className="mb-4 leading-relaxed text-zinc-300 last:mb-0" {...props} />,
      strong: ({ node, ...props }) => <strong className="font-semibold text-white" {...props} />,
      ul: ({ node, ...props }) => <ul className="list-disc pl-5 mb-4 space-y-2 marker:text-red-500 text-zinc-300" {...props} />,
      li: ({ node, ...props }) => <li className="pl-1" {...props} />,
      h4: ({ node, ...props }) => <h4 className="font-medium text-zinc-200 mt-4 mb-2" {...props} />,
    }}
  >
    {content}
  </ReactMarkdown>
);

export default function Home() {
  const [ticker, setTicker] = useState("");
  const [peerTicker, setPeerTicker] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [data, setData] = useState<AnalysisResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!ticker) return;

    setIsAnalyzing(true);
    setErrorMsg(null);
    setData(null);

    try {
      const payload: Record<string, any> = {
        ticker: ticker.toUpperCase(),
        company_name: ticker.toUpperCase(),
      };

      if (peerTicker) {
        payload.peer_ticker = peerTicker.toUpperCase();
        payload.peer_name = peerTicker.toUpperCase();
      }

      const response = await fetch("http://127.0.0.1:8001/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const result: AnalysisResult = await response.json();
      setData(result);
    } catch (error) {
      console.error("Connection failed:", error);
      setErrorMsg("Error: Failed to reach the FastAPI backend on port 8001.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const chartData = (() => {
    if (!data?.market_data?.target?.history_dates) return [];
    const dates = data.market_data.target.history_dates;
    const targetPrices = data.market_data.target.history_prices || [];
    const peerPrices = data.market_data.peer?.history_prices || [];

    return dates.map((date: string, index: number) => ({
      date: date.split("-").slice(1).join("/"),
      targetPrice: targetPrices[index] ? Number(targetPrices[index].toFixed(2)) : null,
      peerPrice: peerPrices[index] ? Number(peerPrices[index].toFixed(2)) : null,
    }));
  })();
  const isTimeSeries = chartData.length > 0;

  const parsedMemo = data ? parseMemoSections(data.final_memo) : null;
  const parsedQuant = data ? parseQuantSections(data.quantitative_analysis) : null;

  return (
    <main className="min-h-screen p-6 md:p-12 lg:p-20 flex flex-col items-center">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="w-full max-w-6xl mb-8 flex flex-col md:flex-row md:items-end justify-between gap-4"
      >
        <div>
          <h1 className="text-4xl md:text-5xl font-bold tracking-tighter text-white">
            KORE <span className="text-red-600">ENGINE</span>
          </h1>
          <p className="text-zinc-400 mt-2 flex items-center gap-3 text-sm">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span>
            </span>
            Autonomous Multi-Agent SEC Intelligence Platform
          </p>
        </div>
      </motion.div>

      {/* Control / Target Acquisition Panel */}
      <div className="w-full max-w-6xl mb-8">
        <div className="glass-panel p-6 rounded-2xl border border-zinc-800">
          <form onSubmit={handleAnalyze} className="flex flex-col md:flex-row gap-4">
            <input
              id="target-ticker"
              name="target-ticker"
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              placeholder="Target Ticker (e.g., AAPL)"
              className="h-12 w-full md:w-1/2 bg-zinc-950/70 rounded-xl border border-zinc-800 px-5 text-white placeholder:text-zinc-600 focus:outline-none focus:border-red-500/50 uppercase tracking-wider text-sm"
              disabled={isAnalyzing}
              required
            />
            <div className="relative w-full md:w-1/2">
              <input
                id="peer-ticker"
                name="peer-ticker"
                type="text"
                value={peerTicker}
                onChange={(e) => setPeerTicker(e.target.value)}
                placeholder="Peer Ticker (e.g., MSFT)"
                className="h-12 w-full bg-zinc-950/70 rounded-xl border border-zinc-800 px-5 text-white placeholder:text-zinc-600 focus:outline-none focus:border-red-500/50 uppercase tracking-wider text-sm pr-14"
                disabled={isAnalyzing}
              />
              <button
                type="submit"
                disabled={isAnalyzing || !ticker}
                className="absolute right-1.5 top-1.5 h-9 px-4 bg-red-600 hover:bg-red-500 text-white rounded-lg flex items-center justify-center gap-2 transition-all disabled:opacity-50 text-xs font-semibold"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <span>Run Memo</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </div>
          </form>
          {errorMsg && <p className="mt-3 text-xs text-red-400 font-mono">{errorMsg}</p>}
        </div>
      </div>

      {/* Main Analysis Display - Bento Grid */}
      {data ? (
        <div className="w-full max-w-6xl grid grid-cols-1 md:grid-cols-3 gap-6">
          
          {/* Top Row: KPI Cards */}
          <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between">
            <div>
              <span className="text-xs uppercase tracking-widest text-zinc-500 font-mono">Target Overview</span>
              <h2 className="text-3xl font-bold text-white mt-1">{data.ticker}</h2>
            </div>
            <div className="grid grid-cols-2 gap-4 mt-6">
              <div className="p-4 bg-zinc-950/60 rounded-xl border border-zinc-800/80">
                <span className="text-xs text-zinc-500 block mb-1">Risk Score</span>
                <span className="text-2xl font-bold text-red-400">{data.risk_score}<span className="text-xs text-zinc-600">/100</span></span>
              </div>
              <div className="p-4 bg-zinc-950/60 rounded-xl border border-zinc-800/80">
                <span className="text-xs text-zinc-500 block mb-1">Financial Score</span>
                <span className="text-2xl font-bold text-zinc-200">{data.financial_score}</span>
              </div>
            </div>
          </div>

          <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-2 text-zinc-500 text-xs uppercase tracking-widest font-mono">
                <Scale className="w-4 h-4 text-zinc-400" />
                <span>Peer Benchmark</span>
              </div>
              <h3 className="text-2xl font-bold text-white mt-2">{data.peer_ticker || "None"}</h3>
            </div>
            <div className="mt-4 p-3 bg-zinc-950/40 rounded-xl border border-zinc-800/40 text-xs text-zinc-400">
              Cross-contamination strict isolation enabled.
            </div>
          </div>

          <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between">
            <div className="flex items-center gap-2 text-zinc-500 text-xs uppercase tracking-widest font-mono">
              <Terminal className="w-4 h-4 text-red-500" />
              <span>Agent Graph Metrics</span>
            </div>
            <div className="space-y-2 text-sm text-zinc-300 mt-4">
              <div className="flex justify-between border-b border-zinc-800/60 pb-2"><span className="text-zinc-500">Retry Count:</span><span className="font-mono text-zinc-200">{data.financial_retries}</span></div>
              <div className="flex justify-between border-b border-zinc-800/60 pb-2"><span className="text-zinc-500">Vector Store:</span><span className="font-mono text-zinc-200">SEC EDGAR</span></div>
              <div className="flex justify-between"><span className="text-zinc-500">Database:</span><span className="font-mono text-red-400">Postgres</span></div>
            </div>
          </div>

          {/* Interactive Chart */}
          <div className="glass-panel p-8 rounded-2xl md:col-span-3 space-y-4">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div className="flex items-center gap-2 text-white font-semibold text-lg">
                <BarChart3 className="w-5 h-5 text-red-500" />
                <span>Market Fundamentals & Price Context</span>
              </div>
              <span className="text-xs font-mono text-zinc-500 uppercase">yFinance Live Stream</span>
            </div>
            {isTimeSeries && (
              <div className="h-[300px] w-full pt-4">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartData}>
                    <defs>
                      <linearGradient id="redGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="#ef4444" stopOpacity={0.0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f1f23" vertical={false} />
                    <XAxis dataKey="date" stroke="#52525b" fontSize={11} tickLine={false} axisLine={false} />
                    <YAxis yAxisId="left" stroke="#52525b" fontSize={11} tickLine={false} axisLine={false} tickFormatter={(v) => `$${v}`} />
                    <YAxis yAxisId="right" orientation="right" stroke="#52525b" fontSize={11} tickLine={false} axisLine={false} tickFormatter={(v) => `$${v}`} />
                    <Tooltip contentStyle={{ backgroundColor: "#09090b", borderColor: "#27272a", borderRadius: "8px", fontSize: "12px" }} />
                    <Legend iconType="circle" wrapperStyle={{ fontSize: "12px", paddingTop: "10px" }} />
                    <Area yAxisId="left" type="monotone" dataKey="targetPrice" name={data.ticker} stroke="#ef4444" strokeWidth={2} fill="url(#redGradient)" />
                    {data.peer_ticker && <Line yAxisId="right" type="monotone" dataKey="peerPrice" name={data.peer_ticker} stroke="#71717a" strokeWidth={2} dot={false} />}
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* Parsed Executive Memo Cards */}
          {parsedMemo?.raw ? (
            <div className="glass-panel p-8 rounded-2xl md:col-span-3">
              <MarkdownBlock content={parsedMemo.raw} />
            </div>
          ) : (
            <>
              {parsedMemo?.summary && (
                <div className="glass-panel p-8 rounded-2xl md:col-span-3 space-y-4">
                  <div className="flex items-center gap-2 text-white font-semibold text-lg border-b border-zinc-800 pb-3">
                    <FileText className="w-5 h-5 text-red-500" /><span>Executive Summary</span>
                  </div>
                  <div className="text-sm"><MarkdownBlock content={parsedMemo.summary} /></div>
                </div>
              )}
              {parsedMemo?.financial && (
                <div className="glass-panel p-8 rounded-2xl md:col-span-1 space-y-4">
                  <div className="flex items-center gap-2 text-white font-semibold text-lg border-b border-zinc-800 pb-3">
                    <TrendingUp className="w-5 h-5 text-red-500" /><span>Financial Performance</span>
                  </div>
                  <div className="text-sm"><MarkdownBlock content={parsedMemo.financial} /></div>
                </div>
              )}
              {parsedMemo?.peer && (
                <div className="glass-panel p-8 rounded-2xl md:col-span-1 space-y-4">
                  <div className="flex items-center gap-2 text-white font-semibold text-lg border-b border-zinc-800 pb-3">
                    <Scale className="w-5 h-5 text-red-500" /><span>Peer Comparison</span>
                  </div>
                  <div className="text-sm"><MarkdownBlock content={parsedMemo.peer} /></div>
                </div>
              )}
              {parsedMemo?.risks && (
                <div className="glass-panel p-8 rounded-2xl md:col-span-1 space-y-4">
                  <div className="flex items-center gap-2 text-white font-semibold text-lg border-b border-zinc-800 pb-3">
                    <AlertTriangle className="w-5 h-5 text-red-500" /><span>Key Risks</span>
                  </div>
                  <div className="text-sm"><MarkdownBlock content={parsedMemo.risks} /></div>
                </div>
              )}
              {parsedMemo?.outlook && (
                <div className="glass-panel p-8 rounded-2xl md:col-span-2 space-y-4">
                  <div className="flex items-center gap-2 text-white font-semibold text-lg border-b border-zinc-800 pb-3">
                    <Briefcase className="w-5 h-5 text-red-500" /><span>Market Outlook</span>
                  </div>
                  <div className="text-sm"><MarkdownBlock content={parsedMemo.outlook} /></div>
                </div>
              )}
              {parsedMemo?.conclusion && (
                <div className="glass-panel p-8 rounded-2xl md:col-span-3 space-y-4 border border-red-900/30 bg-red-950/10">
                  <div className="flex items-center gap-2 text-white font-semibold text-lg border-b border-zinc-800 pb-3">
                    <CheckCircle2 className="w-5 h-5 text-red-500" /><span>Recommendation</span>
                  </div>
                  <div className="text-sm"><MarkdownBlock content={parsedMemo.conclusion} /></div>
                </div>
              )}
            </>
          )}

          {/* Parsed Quantitative Data Cards */}
          {parsedQuant?.raw ? (
            <div className="glass-panel p-8 rounded-2xl md:col-span-3 space-y-4">
              <div className="text-xs font-mono text-zinc-300">
                <MarkdownBlock content={parsedQuant.raw} />
              </div>
            </div>
          ) : (
            <>
              {parsedQuant?.target && (
                <div className="glass-panel p-8 rounded-2xl md:col-span-1 space-y-4">
                  <div className="flex items-center gap-2 text-white font-semibold text-lg border-b border-zinc-800 pb-3">
                    <Target className="w-5 h-5 text-red-500" />
                    <span>{data.ticker} Quantitative Analysis</span>
                  </div>
                  <div className="text-xs font-mono text-zinc-300">
                    <MarkdownBlock content={parsedQuant.target} />
                  </div>
                </div>
              )}
              {parsedQuant?.peer && (
                <div className="glass-panel p-8 rounded-2xl md:col-span-2 space-y-4">
                  <div className="flex items-center gap-2 text-white font-semibold text-lg border-b border-zinc-800 pb-3">
                    <Scale className="w-5 h-5 text-red-500" />
                    <span>{data.peer_ticker} Quantitative Analysis</span>
                  </div>
                  <div className="text-xs font-mono text-zinc-300">
                    <MarkdownBlock content={parsedQuant.peer} />
                  </div>
                </div>
              )}
            </>
          )}

          {/* News Context */}
          {data.news_context && (
            <div className="glass-panel p-8 rounded-2xl md:col-span-3 space-y-4">
              <div className="flex items-center gap-2 text-white font-semibold text-lg border-b border-zinc-800 pb-3">
                <Newspaper className="w-5 h-5 text-red-500" />
                <span>Live Intelligence & Sentiment Context</span>
              </div>
              <div className="text-sm text-zinc-300">
                <MarkdownBlock content={data.news_context} />
              </div>
            </div>
          )}

          {/* News Context */}
          {data.news_context && (
            <div className="glass-panel p-8 rounded-2xl md:col-span-3 space-y-4">
              <div className="flex items-center gap-2 text-white font-semibold text-lg border-b border-zinc-800 pb-3">
                <Newspaper className="w-5 h-5 text-red-500" />
                <span>Live Intelligence & Sentiment Context</span>
              </div>
              <div className="text-sm bg-zinc-950/40 p-4 rounded-xl border border-zinc-800/50">
                <MarkdownBlock content={data.news_context} />
              </div>
            </div>
          )}
        </div>
      ) : null}
    </main>
  );
}