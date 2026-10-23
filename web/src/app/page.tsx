"use client";

import { useState, useRef, useEffect } from "react";
import {
  Terminal,
  Loader2,
  ArrowRight,
  FileText,
  Scale,
  Newspaper,
  BarChart3,
  AlertTriangle,
  Target,
  CheckCircle2,
  Search,
  Sparkles,
  MessageSquare,
  Send,
  Bot,
  User,
  Download,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { motion } from "framer-motion";
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
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

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sourcesCount?: number;
}

const parseQuantSections = (text: string, targetTicker: string, peerTicker?: string) => {
  if (!text) return { target: "", peer: "" };
  const peerKey = peerTicker ? `${peerTicker} Quantitative Analysis:` : "Peer Quantitative Analysis:";
  const targetKey = `${targetTicker} Quantitative Analysis:`;
  if (text.includes(peerKey)) {
    const parts = text.split(peerKey);
    const targetContent = parts[0].replace(targetKey, "").trim();
    const peerContent = parts[1]?.trim() || "";
    return { target: targetContent, peer: peerContent };
  }
  const targetMatch = text.match(
    /(?:1\.\s*Target Company|Target Company|###.*Target)([\s\S]*?)(?=2\.\s*Peer Company|Peer Company|###.*Peer|$)/i
  );
  const peerMatch = text.match(/(?:2\.\s*Peer Company|Peer Company|###.*Peer)([\s\S]*?)$/i);
  return {
    target: targetMatch ? targetMatch[1].trim() : text,
    peer: peerMatch ? peerMatch[1].trim() : "",
  };
};

const MarkdownBlock = ({ content }: { content: string }) => (
  <ReactMarkdown
    components={{
      h1: ({ ...props }) => (
        <h1 className="text-xl font-bold text-white mt-6 mb-3 border-b border-zinc-800 pb-2" {...props} />
      ),
      h2: ({ ...props }) => (
        <h2 className="text-lg font-semibold text-white mt-5 mb-2 flex items-center gap-2" {...props} />
      ),
      h3: ({ ...props }) => <h3 className="text-md font-medium text-red-400 mt-4 mb-2" {...props} />,
      p: ({ ...props }) => <p className="mb-4 leading-relaxed text-zinc-300 last:mb-0 text-sm" {...props} />,
      strong: ({ ...props }) => <strong className="font-semibold text-white" {...props} />,
      ul: ({ ...props }) => (
        <ul className="list-disc pl-5 mb-4 space-y-1 marker:text-red-500 text-zinc-300 text-sm" {...props} />
      ),
      li: ({ ...props }) => <li className="pl-1" {...props} />,
    }}
  >
    {content}
  </ReactMarkdown>
);

export default function Home() {
  const [targetInput, setTargetInput] = useState("");
  const [peerInput, setPeerInput] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [data, setData] = useState<AnalysisResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [liveLogs, setLiveLogs] = useState<string[]>([]);
  const [showLogsAfterComplete, setShowLogsAfterComplete] = useState(false);

  // Live Chat / Interrogation Mode State
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isChatLoading, setIsChatLoading] = useState(false);

  const terminalEndRef = useRef<HTMLDivElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [liveLogs]);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatHistory, isChatLoading]);

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!targetInput.trim()) return;

    setIsAnalyzing(true);
    setErrorMsg(null);
    setData(null);
    setLiveLogs([]);
    setChatHistory([]);
    setShowLogsAfterComplete(false);

    try {
      const payload: Record<string, any> = {
        ticker: targetInput.trim(),
        company_name: targetInput.trim(),
      };
      if (peerInput.trim()) {
        payload.peer_ticker = peerInput.trim();
        payload.peer_name = peerInput.trim();
      }

      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error(`HTTP error ${response.status}`);
      if (!response.body) throw new Error("ReadableStream not supported by browser.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.replace("data: ", "");
            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.type === "log") {
                setLiveLogs((prev) => [...prev, parsed.message]);
              } else if (parsed.type === "complete") {
                setData(parsed.data);
                setIsAnalyzing(false);
              } else if (parsed.type === "error") {
                setErrorMsg(parsed.message);
                setIsAnalyzing(false);
              }
            } catch (err) {
              console.error("Error parsing stream data chunk", err);
            }
          }
        }
      }
    } catch (error) {
      console.error("Connection failed:", error);
      setErrorMsg("Error: Failed to reach the FastAPI backend or stream interrupted.");
      setIsAnalyzing(false);
    }
  };

  const sendChatQuestion = async (questionText: string) => {
    const trimmed = questionText.trim();
    if (!trimmed || !data || isChatLoading) return;

    const updatedHistory: ChatMessage[] = [...chatHistory, { role: "user", content: trimmed }];
    setChatHistory(updatedHistory);
    setChatInput("");
    setIsChatLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: trimmed,
          ticker: data.ticker,
          company_name: data.company_name,
          peer_ticker: data.peer_ticker,
          memo: data.final_memo,
          quantitative_analysis: data.quantitative_analysis,
          news_context: data.news_context,
          chat_history: updatedHistory.map((m) => ({ role: m.role, content: m.content })),
        }),
      });

      if (!response.ok) throw new Error(`Chat HTTP error ${response.status}`);
      const resJson = await response.json();

      setChatHistory((prev) => [
        ...prev,
        {
          role: "assistant",
          content: resJson.answer || "No response generated.",
          sourcesCount: resJson.sources_count,
        },
      ]);
    } catch (err) {
      setChatHistory((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Failed to reach the Interrogation Engine. Check that FastAPI is running.",
        },
      ]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const handleChatSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendChatQuestion(chatInput);
  };

  const handleDownloadMarkdown = () => {
    if (!data) return;
    const blob = new Blob([data.final_memo], { type: "text/markdown;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `${data.ticker}_Investment_Memo.md`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
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
  const quant = data
    ? parseQuantSections(data.quantitative_analysis, data.ticker, data.peer_ticker)
    : { target: "", peer: "" };

  const suggestedPrompts = data
    ? [
        `Break down ${data.ticker}'s primary revenue drivers and YoY margin shifts.`,
        `What are the top 3 existential or regulatory risks in ${data.ticker}'s 10-K?`,
        `How does ${data.ticker} compare against ${data.peer_ticker || "its peer"} on operating efficiency?`,
      ]
    : [];

  return (
    <main className="min-h-screen p-6 md:p-12 lg:p-20 flex flex-col items-center">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="w-full max-w-6xl mb-10"
      >
        <h1 className="text-4xl md:text-5xl font-bold tracking-tighter text-white">
          KORE <span className="text-red-600">ENGINE</span>
        </h1>
        <p className="text-zinc-400 mt-2 flex items-center gap-3 text-sm">
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span>
          </span>
          Autonomous Multi-Agent SEC Intelligence Platform • JIT Ingestion & Live Interrogation
        </p>
      </motion.div>

      {/* Dual Intelligent Command Bar */}
      <div className="w-full max-w-6xl mb-10">
        <div className="bg-zinc-950/70 backdrop-blur-xl p-2.5 rounded-2xl border border-zinc-800 shadow-2xl">
          <form onSubmit={handleAnalyze} className="flex flex-col md:flex-row gap-3">
            <div className="relative flex-1">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Search className="h-4 w-4 text-zinc-500" />
              </div>
              <input
                type="text"
                value={targetInput}
                onChange={(e) => setTargetInput(e.target.value)}
                placeholder="Target Company or Ticker (e.g. Apple, MSTR, PLTR)"
                className="h-14 w-full bg-zinc-900/60 rounded-xl border border-transparent pl-11 pr-4 text-white placeholder:text-zinc-500 focus:outline-none focus:border-red-500/50 focus:bg-zinc-900 tracking-wide text-sm transition-all"
                disabled={isAnalyzing}
                required
              />
            </div>

            <div className="relative flex-1">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Sparkles className="h-4 w-4 text-zinc-500" />
              </div>
              <input
                type="text"
                value={peerInput}
                onChange={(e) => setPeerInput(e.target.value)}
                placeholder="Peer Company/Ticker (or leave blank to auto-discover)"
                className="h-14 w-full bg-zinc-900/60 rounded-xl border border-transparent pl-11 pr-36 text-white placeholder:text-zinc-500 focus:outline-none focus:border-red-500/50 focus:bg-zinc-900 tracking-wide text-sm transition-all"
                disabled={isAnalyzing}
              />

              <button
                type="submit"
                disabled={isAnalyzing || !targetInput.trim()}
                className="absolute right-2 top-2 bottom-2 px-5 bg-red-600 hover:bg-red-500 text-white rounded-lg flex items-center justify-center gap-2 transition-all disabled:opacity-50 text-sm font-semibold shadow-lg shadow-red-900/20"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Analyzing</span>
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
        </div>
        {errorMsg && (
          <p className="mt-4 text-xs text-red-400 font-mono flex items-center gap-2">
            <AlertTriangle className="w-4 h-4" /> {errorMsg}
          </p>
        )}
      </div>

      {/* LIVE AGENT TERMINAL (Shows while analyzing or when toggled open) */}
      {(isAnalyzing || (data && showLogsAfterComplete)) && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-6xl glass-panel p-6 rounded-2xl border border-zinc-800/80 shadow-2xl flex flex-col h-96 font-mono text-sm overflow-hidden mb-8"
        >
          <div className="flex items-center justify-between mb-4 border-b border-zinc-800/80 pb-3 text-zinc-500">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-red-500" />
              <span className="uppercase tracking-widest text-xs">
                LangGraph & JIT Autonomous Execution Log
              </span>
            </div>
            {data && (
              <button
                onClick={() => setShowLogsAfterComplete(false)}
                className="text-xs text-zinc-400 hover:text-white"
              >
                Close Logs
              </button>
            )}
          </div>

          <div className="flex-1 overflow-y-auto space-y-2.5 pb-4 pr-2 custom-scrollbar">
            {liveLogs.map((log, i) => (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                key={i}
                className={`text-zinc-300 ${
                  log.includes("[Error]") || log.includes("failed") ? "text-red-400" : ""
                } ${log.includes("[Success]") ? "text-emerald-400" : ""} ${
                  log.includes("[JIT") ? "text-amber-300" : ""
                }`}
              >
                <span className="text-zinc-600 mr-3 hidden md:inline-block">[{i + 1}]</span>
                {log}
              </motion.div>
            ))}

            {isAnalyzing && (
              <div className="flex items-center gap-3 text-zinc-500 mt-4">
                <Loader2 className="w-4 h-4 animate-spin text-red-500" />
                <span className="animate-pulse">Executing autonomous workflow...</span>
              </div>
            )}

            <div ref={terminalEndRef} />
          </div>
        </motion.div>
      )}

      {/* Main Analysis Display - Bento Grid */}
      {data && !isAnalyzing && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-6xl grid grid-cols-1 md:grid-cols-3 gap-6"
        >
          {/* Target KPI Card */}
          <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between">
            <div>
              <span className="text-xs uppercase tracking-widest text-zinc-500 font-mono">
                Target Overview
              </span>
              <h2 className="text-3xl font-bold text-white mt-1">{data.ticker}</h2>
              <p className="text-xs text-zinc-400 mt-1">{data.company_name}</p>
            </div>
            <div className="grid grid-cols-2 gap-4 mt-6">
              <div className="p-4 bg-zinc-950/60 rounded-xl border border-zinc-800/80">
                <span className="text-xs text-zinc-500 block mb-1">Risk Score</span>
                <span className="text-2xl font-bold text-red-400">
                  {data.risk_score}
                  <span className="text-xs text-zinc-600">/100</span>
                </span>
              </div>
              <div className="p-4 bg-zinc-950/60 rounded-xl border border-zinc-800/80">
                <span className="text-xs text-zinc-500 block mb-1">Retrieval Score</span>
                <span className="text-2xl font-bold text-zinc-200">
                  {Number(data.financial_score).toFixed(4)}
                </span>
              </div>
            </div>
          </div>

          {/* Peer KPI Card */}
          <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-2 text-zinc-500 text-xs uppercase tracking-widest font-mono">
                <Scale className="w-4 h-4 text-zinc-400" />
                <span>Peer Benchmark</span>
              </div>
              <h3 className="text-2xl font-bold text-white mt-2">{data.peer_ticker || "None"}</h3>
              {data.peer_name && <p className="text-xs text-zinc-400 mt-1">{data.peer_name}</p>}
            </div>
            <div className="mt-4 p-3 bg-zinc-950/40 rounded-xl border border-zinc-800/40 text-xs text-zinc-400 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-500" />
              Strict data isolation enabled.
            </div>
          </div>

          {/* System Telemetry */}
          <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between">
            <div className="flex items-center justify-between text-zinc-500 text-xs uppercase tracking-widest font-mono">
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-red-500" />
                <span>System Telemetry</span>
              </div>
              <button
                onClick={() => setShowLogsAfterComplete((prev) => !prev)}
                className="text-xs text-red-400 hover:text-red-300 flex items-center gap-1"
              >
                {showLogsAfterComplete ? "Hide Logs" : `View Logs (${liveLogs.length})`}
                {showLogsAfterComplete ? (
                  <ChevronUp className="w-3.5 h-3.5" />
                ) : (
                  <ChevronDown className="w-3.5 h-3.5" />
                )}
              </button>
            </div>
            <div className="space-y-3 text-sm text-zinc-300 mt-4">
              <div className="flex justify-between border-b border-zinc-800/60 pb-2">
                <span className="text-zinc-500">Retrieval Retries:</span>
                <span className="font-mono text-zinc-200">{data.financial_retries}</span>
              </div>
              <div className="flex justify-between border-b border-zinc-800/60 pb-2">
                <span className="text-zinc-500">Vector Engine:</span>
                <span className="font-mono text-zinc-200">ChromaDB + BM25 + HyDE</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">JIT SEC Engine:</span>
                <span className="font-mono text-emerald-400">Active</span>
              </div>
            </div>
          </div>

          {/* Chart */}
          <div className="glass-panel p-8 rounded-2xl md:col-span-3 space-y-4">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div className="flex items-center gap-2 text-white font-semibold text-lg">
                <BarChart3 className="w-5 h-5 text-red-500" />
                <span>Market Fundamentals & 6-Month Price Action</span>
              </div>
              <span className="text-xs font-mono text-zinc-500 uppercase">yFinance Live Stream</span>
            </div>
            {isTimeSeries ? (
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
                    <YAxis
                      yAxisId="left"
                      stroke="#52525b"
                      fontSize={11}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(v) => `$${v}`}
                    />
                    <YAxis
                      yAxisId="right"
                      orientation="right"
                      stroke="#52525b"
                      fontSize={11}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(v) => `$${v}`}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#09090b",
                        borderColor: "#27272a",
                        borderRadius: "8px",
                        fontSize: "12px",
                        color: "#fff",
                      }}
                      itemStyle={{ color: "#fff" }}
                    />
                    <Legend iconType="circle" wrapperStyle={{ fontSize: "12px", paddingTop: "10px" }} />
                    <Area
                      yAxisId="left"
                      type="monotone"
                      dataKey="targetPrice"
                      name={data.ticker}
                      stroke="#ef4444"
                      strokeWidth={2}
                      fill="url(#redGradient)"
                    />
                    {data.peer_ticker && (
                      <Line
                        yAxisId="right"
                        type="monotone"
                        dataKey="peerPrice"
                        name={data.peer_ticker}
                        stroke="#71717a"
                        strokeWidth={2}
                        dot={false}
                      />
                    )}
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p className="text-xs text-zinc-500 font-mono">
                Market price history unavailable for {data.ticker}.
              </p>
            )}
          </div>

          {/* Institutional Investment Memorandum */}
          <div className="glass-panel p-8 rounded-2xl md:col-span-3 space-y-4 shadow-xl">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div className="flex items-center gap-2 text-white font-semibold text-lg">
                <FileText className="w-5 h-5 text-red-500" />
                <span>Institutional Investment Memorandum</span>
              </div>
              <button
                onClick={handleDownloadMarkdown}
                className="px-3.5 py-1.5 bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 rounded-lg text-xs text-zinc-200 flex items-center gap-2 transition-all"
              >
                <Download className="w-3.5 h-3.5 text-red-400" />
                <span>Export Memo (.MD)</span>
              </button>
            </div>
            <div className="text-zinc-300">
              <MarkdownBlock content={data.final_memo} />
            </div>
          </div>

          {/* Quantitative Analysis Cards */}
          <div className="glass-panel p-8 rounded-2xl md:col-span-1 space-y-4">
            <div className="flex items-center gap-2 text-white font-semibold text-lg border-b border-zinc-800 pb-3">
              <Target className="w-5 h-5 text-red-500" />
              <span>{data.ticker} Quantitative</span>
            </div>
            <div className="text-xs font-mono text-zinc-300">
              <MarkdownBlock content={quant.target || "Insufficient data."} />
            </div>
          </div>

          <div className="glass-panel p-8 rounded-2xl md:col-span-2 space-y-4">
            <div className="flex items-center gap-2 text-white font-semibold text-lg border-b border-zinc-800 pb-3">
              <Scale className="w-5 h-5 text-red-500" />
              <span>{data.peer_ticker || "Peer"} Quantitative</span>
            </div>
            <div className="text-xs font-mono text-zinc-300">
              <MarkdownBlock content={quant.peer || "Insufficient data."} />
            </div>
          </div>

          {/* Live Market Intelligence */}
          {data.news_context && (
            <div className="glass-panel p-8 rounded-2xl md:col-span-3 space-y-4">
              <div className="flex items-center gap-2 text-white font-semibold text-lg border-b border-zinc-800 pb-3">
                <Newspaper className="w-5 h-5 text-red-500" />
                <span>Live Intelligence & Sentiment Context</span>
              </div>
              <div className="text-sm bg-zinc-950/40 p-5 rounded-xl border border-zinc-800/50 shadow-inner">
                <MarkdownBlock content={data.news_context} />
              </div>
            </div>
          )}

          {/* LIVE INTERROGATION MODE (CHAT ANALYSIS) */}
          <div className="glass-panel p-8 rounded-2xl md:col-span-3 space-y-6 border border-red-950/40">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-zinc-800 pb-4">
              <div>
                <div className="flex items-center gap-2 text-white font-semibold text-lg">
                  <MessageSquare className="w-5 h-5 text-red-500" />
                  <span>Live Interrogation Mode — Analyst Q&A</span>
                </div>
                <p className="text-xs text-zinc-400 mt-1">
                  Cross-examine this investment memorandum and query {data.ticker}
                  {data.peer_ticker ? ` & ${data.peer_ticker}` : ""} SEC 10-K filings directly via hybrid RAG.
                </p>
              </div>
              <span className="text-xs font-mono px-3 py-1 rounded-full bg-red-500/10 border border-red-500/30 text-red-400 self-start md:self-auto">
                RAG + MEMO GROUNDED
              </span>
            </div>

            {/* Suggested Quick Prompts */}
            {chatHistory.length === 0 && (
              <div className="flex flex-wrap gap-2">
                {suggestedPrompts.map((prompt, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => sendChatQuestion(prompt)}
                    className="text-xs bg-zinc-900/80 hover:bg-zinc-800 border border-zinc-800 hover:border-red-500/40 text-zinc-300 px-3.5 py-2 rounded-xl transition-all text-left"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            )}

            {/* Chat Message Thread */}
            <div className="max-h-96 overflow-y-auto space-y-4 pr-2 custom-scrollbar">
              {chatHistory.map((msg, index) => (
                <div
                  key={index}
                  className={`flex gap-3 p-4 rounded-xl border ${
                    msg.role === "user"
                      ? "bg-zinc-900/50 border-zinc-800/80 ml-0 md:ml-12"
                      : "bg-zinc-950/80 border-zinc-800/60 mr-0 md:mr-12"
                  }`}
                >
                  <div className="mt-0.5 shrink-0">
                    {msg.role === "user" ? (
                      <div className="w-7 h-7 rounded-lg bg-zinc-800 flex items-center justify-center">
                        <User className="w-4 h-4 text-zinc-300" />
                      </div>
                    ) : (
                      <div className="w-7 h-7 rounded-lg bg-red-600/20 border border-red-500/30 flex items-center justify-center">
                        <Bot className="w-4 h-4 text-red-400" />
                      </div>
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-mono uppercase text-zinc-500">
                        {msg.role === "user" ? "Portfolio Manager" : "KORE Analyst Engine"}
                      </span>
                      {msg.role === "assistant" && typeof msg.sourcesCount === "number" && (
                        <span className="text-[11px] font-mono text-emerald-400">
                          {msg.sourcesCount} SEC chunks verified
                        </span>
                      )}
                    </div>
                    <div className="text-sm text-zinc-200">
                      <MarkdownBlock content={msg.content} />
                    </div>
                  </div>
                </div>
              ))}

              {isChatLoading && (
                <div className="flex items-center gap-3 p-4 rounded-xl bg-zinc-950/60 border border-zinc-800/60 text-zinc-400 text-sm font-mono">
                  <Loader2 className="w-4 h-4 animate-spin text-red-500" />
                  <span>Retrieving SEC 10-K tables & synthesizing answer...</span>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Chat Input Bar */}
            <form onSubmit={handleChatSubmit} className="flex gap-3">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder={`Ask a specific follow-up question about ${data.ticker}'s financials, footnotes, or risks...`}
                className="h-12 flex-1 bg-zinc-900/70 rounded-xl border border-zinc-800 px-4 text-white placeholder:text-zinc-500 focus:outline-none focus:border-red-500/50 text-sm"
                disabled={isChatLoading}
              />
              <button
                type="submit"
                disabled={isChatLoading || !chatInput.trim()}
                className="h-12 px-5 bg-red-600 hover:bg-red-500 disabled:opacity-50 text-white rounded-xl flex items-center gap-2 text-sm font-semibold transition-all"
              >
                <span>Ask</span>
                <Send className="w-4 h-4" />
              </button>
            </form>
          </div>
        </motion.div>
      )}
    </main>
  );
}