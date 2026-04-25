import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  Activity,
  Cpu,
  Download,
  FileDown,
  FileText,
  LineChart,
  Loader2,
  Newspaper,
  RefreshCw,
  Search,
  Server,
  Sparkles,
  Terminal,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/contexts/AuthContext";
import { aiUrl, backendUrl } from "@/lib/config";

type PdfCompileOk = {
  ok: true;
  blob: Blob;
  latexFixed?: string;
  repairRounds?: number;
};

async function compileReportPdfToBlob(
  baseUrl: string,
  latex: string,
  ticker: string,
  opts?: { repairWithGemini?: boolean; maxRepairRounds?: number },
): Promise<PdfCompileOk | { ok: false; message: string }> {
  try {
    const res = await fetch(`${baseUrl}/report/pdf`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        latex,
        ticker,
        repair_with_gemini: opts?.repairWithGemini !== false,
        max_repair_rounds: opts?.maxRepairRounds ?? 3,
        response_format: "json",
      }),
    });
    const ct = res.headers.get("content-type") || "";
    if (!ct.includes("application/json")) {
      const msg = (await res.text()).slice(0, 1200) || `Unexpected response (${res.status})`;
      return { ok: false, message: msg };
    }
    const j = (await res.json()) as {
      pdf_base64?: string | null;
      latex_compiled?: string;
      repair_rounds?: number;
      error?: string | null;
    };
    if (!res.ok || j.error) {
      const msg =
        typeof j.error === "string" && j.error.trim()
          ? j.error
          : `PDF compile failed (${res.status})`;
      return { ok: false, message: msg };
    }
    const b64 = j.pdf_base64;
    if (!b64 || typeof b64 !== "string") {
      return { ok: false, message: "PDF response missing pdf_base64" };
    }
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) {
      arr[i] = bin.charCodeAt(i);
    }
    const blob = new Blob([arr], { type: "application/pdf" });
    return {
      ok: true,
      blob,
      latexFixed:
        typeof j.latex_compiled === "string" && j.latex_compiled.trim()
          ? j.latex_compiled
          : undefined,
      repairRounds:
        typeof j.repair_rounds === "number" && Number.isFinite(j.repair_rounds)
          ? j.repair_rounds
          : undefined,
    };
  } catch (e) {
    return {
      ok: false,
      message: e instanceof Error ? e.message : "PDF request failed",
    };
  }
}

type ServiceState = "idle" | "ok" | "error";

type ReportStep = {
  id: string;
  label: string;
  status: string;
  detail: string;
};

type ReportPayload = {
  ticker: string;
  company_name: string;
  latex: string;
  npv: number | null;
  discount_rate: number;
  discount_rate_basis?: string | null;
  discount_rate_source?: string | null;
  fcf_projection: number[];
  warnings: string[];
  steps: ReportStep[];
  report_mode?: string;
  data_sources?: string[];
  gemini_model?: string | null;
  reasoning?: string | null;
  price_target_usd?: number | null;
  price_target_horizon_months?: number | null;
  price_target_basis?: string | null;
  headlines?: { title: string; content: string; url: string }[];
  junior_latex?: string | null;
  post_peer_junior_latex?: string | null;
  junior_research_memo?: string | null;
  post_peer_junior_research_memo?: string | null;
  investment_recommendation?: string | null;
  workbook_json_excerpt?: string | null;
  critic_memo?: string | null;
  lead_pm_synthesis?: string | null;
};

function formatReportMode(mode: string | undefined): string {
  if (!mode) return "—";
  const labels: Record<string, string> = {
    gemini: "Narrative model",
    gemini_memo: "Research memo (Lead PM PDF)",
    gemini_portfolio: "Multi-agent + Lead PM PDF",
    deterministic: "Templated draft",
    deterministic_yfinance: "Templated draft",
  };
  return labels[mode] ?? mode;
}

function formatDiscountRowLabel(source: string | null | undefined): string {
  if (source === "junior_model") return "Discount (Junior)";
  if (source === "seed_no_key") return "Discount (seed)";
  if (source === "seed_fallback") return "Discount (seed fallback)";
  return "Discount (inferred)";
}

function formatDataSourceLabel(id: string): string {
  const labels: Record<string, string> = {
    sec_edgar: "SEC EDGAR",
    sec_facts: "Filing facts",
    yfinance_market: "Market data",
    searxng: "Headline scan",
    go_npv: "Valuation bridge",
    gemini: "Narrative model",
  };
  return labels[id] ?? id;
}

const AGENT_IDS = [
  "junior_researcher",
  "critic",
  "lead_portfolio_manager",
] as const;

type AgentId = (typeof AGENT_IDS)[number];

type AgentStripState = "idle" | "running" | "done" | "error";

const AGENT_LABELS_UI: Record<AgentId, string> = {
  junior_researcher: "Junior Researcher",
  critic: "The Critic",
  lead_portfolio_manager: "The Lead Portfolio Manager",
};

const INITIAL_AGENT_BOARD: Record<AgentId, AgentStripState> = {
  junior_researcher: "idle",
  critic: "idle",
  lead_portfolio_manager: "idle",
};

function streamAgentLabel(agent: string | undefined): string {
  if (!agent) return "Pipeline";
  return AGENT_LABELS_UI[agent as AgentId] ?? agent;
}

function agentStripBadgeClass(s: AgentStripState): string {
  if (s === "running")
    return "border-primary/60 bg-primary/15 text-primary animate-pulse-slow";
  if (s === "done") return "border-emerald-500/40 bg-emerald-500/10 text-emerald-200";
  if (s === "error") return "border-destructive/50 bg-destructive/15 text-destructive";
  return "border-border bg-muted/30 text-muted-foreground";
}

type StreamEnvelope =
  | {
      type: "phase";
      id: string;
      status: string;
      label?: string;
      detail?: string;
      agent?: string;
    }
  | { type: "agent_status"; agent: string; status: string; detail?: string }
  | { type: "gemini_request"; call_site?: string; agent?: string }
  | { type: "reasoning"; text: string; truncated?: boolean; agent?: string }
  | { type: "reasoning_delta"; text: string; agent?: string }
  | {
      type: "price_target";
      usd: number;
      horizon_months?: number | null;
      basis?: string;
      agent?: string;
    }
  | { type: "complete"; report: ReportPayload }
  | { type: "error"; detail: unknown };

function useHealth(url: string, path: string) {
  const [state, setState] = useState<ServiceState>("idle");
  const [detail, setDetail] = useState<string>("");

  const ping = useCallback(async () => {
    setState("idle");
    setDetail("");
    try {
      const res = await fetch(`${url}${path}`);
      const text = await res.text();
      if (!res.ok) {
        setState("error");
        setDetail(text.slice(0, 200));
        return;
      }
      setState("ok");
      setDetail(text);
    } catch (e) {
      setState("error");
      setDetail(e instanceof Error ? e.message : "request failed");
    }
  }, [path, url]);

  useEffect(() => {
    void ping();
  }, [ping]);

  return { state, detail, ping };
}

function StatusDot({ state }: { state: ServiceState }) {
  const color =
    state === "ok"
      ? "bg-primary shadow-[0_0_12px_hsl(var(--primary)/0.55)]"
      : state === "error"
        ? "bg-destructive"
        : "bg-muted-foreground animate-pulse-slow";
  return (
    <span className="relative flex h-2.5 w-2.5">
      <span
        className={`inline-flex h-2.5 w-2.5 rounded-full ${color}`}
        aria-hidden
      />
    </span>
  );
}

function healthTitle(label: string, url: string, state: ServiceState, detail: string) {
  const status =
    state === "ok" ? "online" : state === "error" ? "offline" : "checking";
  const tail = detail.trim() ? ` — ${detail.trim().slice(0, 180)}` : "";
  return `${label} (${url}): ${status}${tail}`;
}

const TICKER_PATTERN = /^[A-Za-z0-9.\-]{1,12}$/;

async function persistReportForUser(
  report: ReportPayload,
  token: string,
): Promise<{ ok: true } | { ok: false; message: string }> {
  try {
    const res = await fetch(`${aiUrl}/api/saved-reports`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ report }),
    });
    const text = await res.text();
    if (!res.ok) {
      let msg = text.slice(0, 400) || `Save failed (${res.status})`;
      try {
        const j = JSON.parse(text) as { error?: string };
        if (typeof j.error === "string" && j.error.trim()) msg = j.error;
      } catch {
        /* use msg */
      }
      return { ok: false, message: msg };
    }
    return { ok: true };
  } catch (e) {
    return {
      ok: false,
      message: e instanceof Error ? e.message : "Save request failed",
    };
  }
}

export function ResearchConsole() {
  const { savedId } = useParams<{ savedId?: string }>();
  const { accessToken } = useAuth();
  const tokenRef = useRef<string | null>(null);
  tokenRef.current = accessToken;

  const go = useHealth(backendUrl, "/health");
  const ai = useHealth(aiUrl, "/health");

  const [ticker, setTicker] = useState("MSFT");
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState<string>("");
  const [report, setReport] = useState<ReportPayload | null>(null);
  const [streamEvents, setStreamEvents] = useState<string[]>([]);
  /** Interleaved phases + streamed reasoning, tagged by agent where the server provides it. */
  const [agentConversation, setAgentConversation] = useState<string>("");
  const [geminiRequestsThisRun, setGeminiRequestsThisRun] = useState(0);
  const [agentBoard, setAgentBoard] =
    useState<Record<AgentId, AgentStripState>>(INITIAL_AGENT_BOARD);
  const reportStreamRef = useRef<EventSource | null>(null);
  const transcriptAgentRef = useRef<string | null>(null);

  const [pdfBusy, setPdfBusy] = useState(false);
  const [pdfMessage, setPdfMessage] = useState<string>("");

  const [outputTab, setOutputTab] = useState<"pdf" | "latex">("pdf");
  const [pdfPreviewUrl, setPdfPreviewUrl] = useState<string | null>(null);
  const [pdfPreviewBusy, setPdfPreviewBusy] = useState(false);
  const [pdfPreviewError, setPdfPreviewError] = useState<string>("");
  const [librarySaveMessage, setLibrarySaveMessage] = useState<string>("");
  const [savedLoadError, setSavedLoadError] = useState<string>("");
  const [savedLoading, setSavedLoading] = useState(false);

  useEffect(() => {
    if (!savedId?.trim()) {
      setSavedLoadError("");
      setSavedLoading(false);
      return;
    }
    const idNum = Number.parseInt(savedId, 10);
    if (!Number.isFinite(idNum) || idNum < 1) {
      setSavedLoadError("Invalid saved report link.");
      return;
    }
    const tok = accessToken;
    if (!tok) {
      setSavedLoadError("Sign in to open saved reports.");
      return;
    }
    let cancelled = false;
    setSavedLoading(true);
    setSavedLoadError("");
    void (async () => {
      const res = await fetch(`${aiUrl}/api/saved-reports/${idNum}`, {
        headers: { Authorization: `Bearer ${tok}` },
      });
      const raw = await res.text();
      if (cancelled) return;
      setSavedLoading(false);
      if (!res.ok) {
        let msg = raw.slice(0, 400) || `Could not load report (${res.status})`;
        try {
          const j = JSON.parse(raw) as { error?: string };
          if (typeof j.error === "string" && j.error.trim()) msg = j.error;
        } catch {
          /* keep msg */
        }
        setSavedLoadError(msg);
        setReport(null);
        return;
      }
      try {
        const body = JSON.parse(raw) as { report?: ReportPayload };
        if (!body.report || typeof body.report !== "object") {
          setSavedLoadError("Saved report payload is missing.");
          setReport(null);
          return;
        }
        setReport(body.report);
        const sym = body.report.ticker?.trim();
        if (sym) setTicker(sym.toUpperCase());
        setStreamEvents([]);
        setAgentConversation("");
        setReportError("");
      } catch {
        setSavedLoadError("Could not parse saved report.");
        setReport(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [savedId, accessToken]);

  useEffect(() => {
    return () => {
      reportStreamRef.current?.close();
      reportStreamRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!report?.latex?.trim()) {
      setPdfPreviewUrl((u) => {
        if (u) URL.revokeObjectURL(u);
        return null;
      });
      setPdfPreviewError("");
      setPdfPreviewBusy(false);
      return;
    }
    let cancelled = false;
    setPdfPreviewBusy(true);
    setPdfPreviewError("");
    void (async () => {
      const r = await compileReportPdfToBlob(
        aiUrl,
        report.latex,
        report.ticker,
      );
      if (cancelled) return;
      setPdfPreviewBusy(false);
      if (!r.ok) {
        setPdfPreviewError(r.message);
        setPdfPreviewUrl((u) => {
          if (u) URL.revokeObjectURL(u);
          return null;
        });
        setOutputTab("latex");
        return;
      }
      const fixedTex = r.latexFixed?.trim();
      if (fixedTex) {
        setReport((prev) => {
          if (!prev || prev.latex === fixedTex) return prev;
          return { ...prev, latex: fixedTex };
        });
      }
      if (r.repairRounds != null && r.repairRounds > 0) {
        setPdfMessage(
          `Server applied ${r.repairRounds} LaTeX repair pass(es); the LaTeX tab now matches the compiled PDF.`,
        );
      }
      const url = URL.createObjectURL(r.blob);
      setPdfPreviewUrl((old) => {
        if (old) URL.revokeObjectURL(old);
        return url;
      });
      setOutputTab("pdf");
    })();
    return () => {
      cancelled = true;
    };
  }, [report?.latex, report?.ticker, aiUrl]);

  const generateReport = () => {
    setLibrarySaveMessage("");
    setReportError("");
    setReport(null);
    setStreamEvents([]);
    setAgentConversation("");
    transcriptAgentRef.current = null;
    setGeminiRequestsThisRun(0);
    setAgentBoard({ ...INITIAL_AGENT_BOARD });
    setPdfPreviewUrl((u) => {
      if (u) URL.revokeObjectURL(u);
      return null;
    });
    setPdfPreviewError("");
    setPdfPreviewBusy(false);
    setOutputTab("pdf");
    const sym = ticker.trim().toUpperCase();
    if (!TICKER_PATTERN.test(sym)) {
      setReportError("Enter a valid ticker (1–12 letters, digits, . or -).");
      return;
    }
    reportStreamRef.current?.close();
    reportStreamRef.current = null;

    setReportLoading(true);
    let finished = false;
    const q = new URLSearchParams({ ticker: sym });
    const es = new EventSource(`${aiUrl}/report/stream?${q.toString()}`);
    reportStreamRef.current = es;

    es.onmessage = (ev) => {
      let data: StreamEnvelope;
      try {
        data = JSON.parse(ev.data) as StreamEnvelope;
      } catch {
        setReportError("Invalid stream payload from AI service");
        finished = true;
        es.close();
        setReportLoading(false);
        return;
      }
      if (data.type === "agent_status") {
        const raw = data.agent;
        if (!(AGENT_IDS as readonly string[]).includes(raw)) return;
        const aid = raw as AgentId;
        const next = data.status as AgentStripState;
        const allowed: AgentStripState[] = ["idle", "running", "done", "error"];
        setAgentBoard((prev) => ({
          ...prev,
          [aid]: allowed.includes(next) ? next : prev[aid],
        }));
        const detail =
          typeof data.detail === "string" && data.detail.trim()
            ? `\n${data.detail.trim()}`
            : "";
        const line = `[${AGENT_LABELS_UI[aid]}] status · ${next}${detail}`;
        setAgentConversation((prev) => (prev ? `${prev}\n\n${line}` : line));
        return;
      }
      if (data.type === "gemini_request") {
        setGeminiRequestsThisRun((n) => n + 1);
        return;
      }
      if (data.type === "phase") {
        const parts = [
          data.id,
          data.status,
          data.label,
          data.detail,
        ].filter(Boolean);
        const who =
          data.agent && typeof data.agent === "string"
            ? `[${AGENT_LABELS_UI[data.agent as AgentId] ?? data.agent}] `
            : "";
        setStreamEvents((prev) => [...prev, `${who}${parts.join(" · ")}`]);
        const convWho = streamAgentLabel(
          typeof data.agent === "string" ? data.agent : undefined,
        );
        const convLine = `[${convWho}] ${[data.id, data.status, data.label, data.detail].filter(Boolean).join(" · ")}`;
        setAgentConversation((prev) => (prev ? `${prev}\n\n${convLine}` : convLine));
        return;
      }
      if (data.type === "reasoning") {
        const lab = streamAgentLabel(
          typeof data.agent === "string" ? data.agent : undefined,
        );
        transcriptAgentRef.current =
          typeof data.agent === "string" ? data.agent : null;
        const tail = data.truncated ? "\n\n[truncated in stream]" : "";
        const block = `── ${lab} · reasoning\n${data.text}${tail}`;
        setAgentConversation((prev) => (prev ? `${prev}\n\n${block}` : block));
        return;
      }
      if (data.type === "reasoning_delta") {
        if (data.text) {
          const agent = typeof data.agent === "string" ? data.agent : null;
          setAgentConversation((prev) => {
            let prefix = "";
            if (agent && agent !== transcriptAgentRef.current) {
              transcriptAgentRef.current = agent;
              const lab = streamAgentLabel(agent);
              prefix = prev ? `\n\n── ${lab} · stream\n` : `── ${lab} · stream\n`;
            }
            return `${prev}${prefix}${data.text}`;
          });
        }
        return;
      }
      if (data.type === "price_target") {
        const hm =
          data.horizon_months != null ? `${data.horizon_months}m horizon` : "horizon n/a";
        const line = `price_target · $${data.usd.toFixed(2)} · ${hm}${data.basis ? ` · ${data.basis.slice(0, 120)}` : ""}`;
        setStreamEvents((prev) => [...prev, line]);
        const lab = streamAgentLabel(
          typeof data.agent === "string" ? data.agent : undefined,
        );
        const conv = `[${lab}] ${line}`;
        setAgentConversation((prev) => (prev ? `${prev}\n\n${conv}` : conv));
        return;
      }
      if (data.type === "error") {
        finished = true;
        es.close();
        setAgentBoard((prev) => {
          const next = { ...prev };
          (AGENT_IDS as readonly AgentId[]).forEach((id) => {
            if (next[id] === "running") next[id] = "error";
          });
          return next;
        });
        const detail =
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail);
        setReportError(detail || "stream error");
        setReportLoading(false);
        return;
      }
      if (data.type === "complete") {
        finished = true;
        es.close();
        setReport(data.report);
        setReportLoading(false);
        setLibrarySaveMessage("");
        const t = tokenRef.current;
        if (t) {
          void (async () => {
            const out = await persistReportForUser(data.report, t);
            if (out.ok) {
              setLibrarySaveMessage("Saved to your library.");
            } else {
              setLibrarySaveMessage(`Could not auto-save: ${out.message}`);
            }
          })();
        }
      }
    };

    es.onerror = () => {
      if (!finished) {
        setAgentBoard((prev) => {
          const next = { ...prev };
          (AGENT_IDS as readonly AgentId[]).forEach((id) => {
            if (next[id] === "running") next[id] = "error";
          });
          return next;
        });
        setReportError((prev) =>
          prev || "Report stream disconnected (check AI service / network).",
        );
      }
      es.close();
      setReportLoading(false);
    };
  };

  const downloadTex = () => {
    if (!report?.latex) return;
    const blob = new Blob([report.latex], {
      type: "text/plain;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${report.ticker}_equity_note.tex`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadPdf = async () => {
    if (!report?.latex) return;
    setPdfMessage("");
    setPdfBusy(true);
    try {
      const r = await compileReportPdfToBlob(
        aiUrl,
        report.latex,
        report.ticker,
      );
      if (!r.ok) {
        setPdfMessage(r.message);
        return;
      }
      const fixedTexDl = r.latexFixed?.trim();
      if (fixedTexDl) {
        setReport((prev) => {
          if (!prev || prev.latex === fixedTexDl) return prev;
          return { ...prev, latex: fixedTexDl };
        });
      }
      if (r.repairRounds != null && r.repairRounds > 0) {
        setPdfMessage(
          `Downloaded PDF after ${r.repairRounds} LaTeX repair pass(es); source tab updated.`,
        );
      }
      const url = URL.createObjectURL(r.blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${report.ticker}_equity_note.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setPdfMessage(e instanceof Error ? e.message : "PDF download failed");
    } finally {
      setPdfBusy(false);
    }
  };

  const refreshPdfPreview = useCallback(async () => {
    if (!report?.latex) return;
    setPdfPreviewBusy(true);
    setPdfPreviewError("");
    const r = await compileReportPdfToBlob(
      aiUrl,
      report.latex,
      report.ticker,
    );
    setPdfPreviewBusy(false);
    if (!r.ok) {
      setPdfPreviewError(r.message);
      setPdfPreviewUrl((u) => {
        if (u) URL.revokeObjectURL(u);
        return null;
      });
      setOutputTab("latex");
      return;
    }
    const fixedTexRf = r.latexFixed?.trim();
    if (fixedTexRf) {
      setReport((prev) => {
        if (!prev || prev.latex === fixedTexRf) return prev;
        return { ...prev, latex: fixedTexRf };
      });
    }
    if (r.repairRounds != null && r.repairRounds > 0) {
      setPdfMessage(
        `Server applied ${r.repairRounds} LaTeX repair pass(es); the LaTeX tab now matches the compiled PDF.`,
      );
    }
    const url = URL.createObjectURL(r.blob);
    setPdfPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return url;
    });
    setOutputTab("pdf");
  }, [aiUrl, report?.latex, report?.ticker]);

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-[radial-gradient(ellipse_at_top,_hsl(152_76%_42%/0.12),_transparent_55%),radial-gradient(ellipse_at_bottom,_hsl(215_28%_14%/0.9),_hsl(224_71%_4%))]">
      <header className="border-b border-border/80 bg-card/40 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 rounded-md border border-border bg-secondary/60 p-2">
              <Terminal className="h-6 w-6 text-primary" aria-hidden />
            </div>
            <div>
              <p className="font-mono text-xs uppercase tracking-[0.2em] text-muted-foreground">
                Live pipeline
              </p>
              <h1 className="text-2xl font-semibold tracking-tight text-foreground">
                {savedId ? "Saved research" : "Equity research workspace"}
              </h1>
              <p className="mt-1 max-w-xl text-sm text-muted-foreground">
                {savedId
                  ? "This note was loaded from your library. Generate a new run from Research anytime."
                  : "Enter a ticker to assemble filing facts, market history, a headline scan, and an illustrative valuation bridge into a LaTeX research note with automatic PDF preview."}
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2">
            <div
              className="flex items-center gap-2 rounded-md border border-border/70 bg-background/50 px-2 py-1.5"
              role="status"
              aria-label="Backend service health"
            >
              <span
                className="flex items-center gap-1.5"
                title={healthTitle("Valuation engine", backendUrl, go.state, go.detail)}
              >
                <Server className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden />
                <StatusDot state={go.state} />
              </span>
              <span className="h-3 w-px shrink-0 bg-border" aria-hidden />
              <span
                className="flex items-center gap-1.5"
                title={healthTitle("Research API", aiUrl, ai.state, ai.detail)}
              >
                <Sparkles className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden />
                <StatusDot state={ai.state} />
                <span
                  className="min-w-[1.25rem] text-center font-mono text-[10px] tabular-nums text-muted-foreground"
                  title="Gemini API calls counted this report run (SSE stream only; PDF repair calls are separate)"
                  aria-label={`Gemini requests this run: ${geminiRequestsThisRun}`}
                >
                  {geminiRequestsThisRun}
                </span>
              </span>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="shrink-0"
              onClick={() => {
                void go.ping();
                void ai.ping();
              }}
            >
              <RefreshCw className="h-4 w-4" />
              Refresh probes
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-8">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-12">
          <Card className="border-primary/25 bg-card/70 shadow-lg shadow-primary/5 md:col-span-2 xl:col-span-12">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 normal-case tracking-normal text-foreground">
                <LineChart className="h-5 w-5 text-primary" />
                Equity research report
              </CardTitle>
              <CardDescription>
                Live build via{" "}
                <span className="font-mono text-foreground">GET /report/stream</span>{" "}
                (SSE): Junior Researcher ↔ The Critic (peer rounds) → Lead Portfolio Manager; one-shot JSON
                at{" "}
                <span className="font-mono text-foreground">POST /report/generate</span>.
                Not investment advice.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {savedLoading ? (
                <p className="flex items-center gap-2 font-mono text-xs text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin shrink-0" />
                  Loading saved report…
                </p>
              ) : null}
              {savedLoadError ? (
                <p className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                  {savedLoadError}
                  {!accessToken && savedId ? (
                    <>
                      {" "}
                      <Link to="/login" className="font-medium underline underline-offset-2">
                        Sign in
                      </Link>
                    </>
                  ) : null}
                  {accessToken && savedId ? (
                    <>
                      {" "}
                      <Link to="/library" className="font-medium underline underline-offset-2">
                        Back to library
                      </Link>
                    </>
                  ) : null}
                </p>
              ) : null}
              {librarySaveMessage ? (
                <p
                  className={`rounded-md border px-3 py-2 font-mono text-xs ${
                    librarySaveMessage.startsWith("Could not")
                      ? "border-amber-500/40 bg-amber-500/10 text-amber-100"
                      : "border-primary/35 bg-primary/10 text-primary/95"
                  }`}
                >
                  {librarySaveMessage}
                  {librarySaveMessage === "Saved to your library." ? (
                    <>
                      {" "}
                      <Link
                        to="/library"
                        className="font-medium text-foreground underline underline-offset-2"
                      >
                        View library
                      </Link>
                    </>
                  ) : null}
                </p>
              ) : null}
              <div className="grid gap-4 sm:grid-cols-[1fr_auto] sm:items-end">
                <div className="space-y-2">
                  <Label htmlFor="ticker">Ticker</Label>
                  <Input
                    id="ticker"
                    value={ticker}
                    onChange={(e) => setTicker(e.target.value)}
                    placeholder="e.g. MSFT"
                    autoCapitalize="characters"
                    spellCheck={false}
                  />
                </div>
                <Button
                  className="h-9 w-full sm:w-auto"
                  onClick={() => generateReport()}
                  disabled={reportLoading}
                >
                  <Search className="h-4 w-4" />
                  {reportLoading ? "Building…" : "Generate report"}
                </Button>
              </div>

              {reportError ? (
                <p className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 font-mono text-xs text-destructive">
                  {reportError}
                </p>
              ) : null}

              {reportLoading || streamEvents.length > 0 || agentConversation ? (
                <div className="space-y-3">
                  <div className="rounded-md border border-border bg-background/40 p-3">
                    <p className="font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
                      Agents (live)
                    </p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {AGENT_IDS.map((id) => (
                        <div
                          key={id}
                          className={`flex min-w-[10rem] flex-1 flex-col gap-0.5 rounded-md border px-2 py-1.5 sm:max-w-[14rem] sm:flex-none ${agentStripBadgeClass(agentBoard[id])}`}
                        >
                          <span className="text-[11px] font-medium leading-snug text-foreground/95">
                            {AGENT_LABELS_UI[id]}
                          </span>
                          <span className="font-mono text-[9px] uppercase tracking-wide opacity-90">
                            {agentBoard[id]}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                  <div className="rounded-md border border-border bg-background/40 p-3">
                    <p className="font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
                      Pipeline (live)
                    </p>
                    <pre className="mt-2 max-h-40 overflow-auto font-mono text-[10px] leading-relaxed text-muted-foreground">
                      {streamEvents.length
                        ? streamEvents.join("\n")
                        : reportLoading
                          ? "Connecting…"
                          : "—"}
                    </pre>
                  </div>
                  <div className="rounded-md border border-border bg-background/40 p-3">
                    <p className="font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
                      Agent conversation (live)
                    </p>
                    <pre className="mt-2 max-h-52 overflow-auto whitespace-pre-wrap font-mono text-[10px] leading-relaxed text-foreground/90">
                      {agentConversation ||
                        (reportLoading
                          ? "Waiting for phases and model output…"
                          : "—")}
                    </pre>
                  </div>
                  </div>
                </div>
              ) : null}

              {report ? (
                <div className="space-y-6">
                  <section className="min-w-0 space-y-3 rounded-md border border-border bg-background/50 p-4">
                    <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                      <p className="font-mono text-xs uppercase tracking-wide text-muted-foreground">
                        Report output
                      </p>
                      <div className="flex flex-wrap items-center gap-2">
                        <div className="flex rounded-md border border-border bg-background/60 p-0.5">
                          <Button
                            type="button"
                            variant={outputTab === "pdf" ? "secondary" : "ghost"}
                            size="sm"
                            className="h-7 px-2.5 text-xs"
                            onClick={() => setOutputTab("pdf")}
                          >
                            PDF preview
                          </Button>
                          <Button
                            type="button"
                            variant={outputTab === "latex" ? "secondary" : "ghost"}
                            size="sm"
                            className="h-7 px-2.5 text-xs"
                            onClick={() => setOutputTab("latex")}
                          >
                            LaTeX source
                          </Button>
                        </div>
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          className="h-7 gap-1 px-2 text-xs"
                          onClick={() => void refreshPdfPreview()}
                          disabled={!report.latex || pdfPreviewBusy}
                          title="Recompile PDF from current LaTeX"
                        >
                          <RefreshCw
                            className={`h-3.5 w-3.5 ${pdfPreviewBusy ? "animate-spin" : ""}`}
                          />
                          Refresh PDF
                        </Button>
                      </div>
                    </div>
                    {pdfPreviewError ? (
                      <p className="mb-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-2 font-mono text-[10px] text-amber-100/95 whitespace-pre-wrap">
                        {pdfPreviewError}
                      </p>
                    ) : null}
                    <div className="relative min-h-[20rem] max-h-[32rem] overflow-hidden rounded-md border border-border bg-black/40">
                      {outputTab === "pdf" ? (
                        <>
                          {pdfPreviewBusy ? (
                            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-background/80 backdrop-blur-sm">
                              <Loader2 className="h-8 w-8 animate-spin text-primary" />
                              <p className="font-mono text-xs text-muted-foreground">
                                Compiling PDF with server pdflatex…
                              </p>
                            </div>
                          ) : null}
                          {pdfPreviewUrl ? (
                            <iframe
                              key={pdfPreviewUrl}
                              title={`${report.ticker} equity note PDF preview`}
                              src={pdfPreviewUrl}
                              className="h-[32rem] w-full border-0 bg-white"
                            />
                          ) : !pdfPreviewBusy ? (
                            <div className="flex h-[32rem] flex-col items-center justify-center gap-2 px-6 text-center">
                              <FileText className="h-10 w-10 text-muted-foreground" />
                              <p className="font-mono text-xs text-muted-foreground">
                                PDF preview unavailable. Check AI{" "}
                                <span className="text-foreground">/health</span>{" "}
                                (<span className="text-foreground">pdflatex_available</span>
                                ) or open the LaTeX tab.
                              </p>
                            </div>
                          ) : null}
                        </>
                      ) : (
                        <pre className="max-h-[32rem] overflow-auto p-4 font-mono text-[11px] leading-relaxed text-muted-foreground">
                          {report.latex}
                        </pre>
                      )}
                    </div>
                    {pdfMessage ? (
                      <p className="rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-2 font-mono text-[10px] text-amber-100/95 whitespace-pre-wrap">
                        {pdfMessage}
                      </p>
                    ) : null}
                    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                      <Button
                        variant="outline"
                        size="sm"
                        className="min-w-0 w-full justify-center gap-2"
                        onClick={downloadTex}
                      >
                        <Download className="h-4 w-4 shrink-0" />
                        <span className="truncate">Download .tex</span>
                      </Button>
                      <Button
                        variant="default"
                        size="sm"
                        className="min-w-0 w-full justify-center gap-2"
                        onClick={() => void downloadPdf()}
                        disabled={pdfBusy}
                      >
                        <FileDown className="h-4 w-4 shrink-0" />
                        <span className="truncate">
                          {pdfBusy ? "Building PDF…" : "Download .pdf"}
                        </span>
                      </Button>
                    </div>
                    <p className="font-mono text-[10px] text-muted-foreground">
                      PDF uses server{" "}
                      <span className="text-foreground">pdflatex</span> plus optional{" "}
                      <span className="text-foreground">Model</span> repair passes on
                      compile errors (see{" "}
                      <span className="text-foreground">GET /health</span> →{" "}
                      <span className="text-foreground">latex_repair_available</span>
                      ).
                    </p>
                  </section>

                  <div className="grid gap-6 lg:grid-cols-12 lg:items-start">
                    <div className="min-w-0 space-y-2 rounded-md border border-border bg-background/50 p-4 lg:col-span-6 xl:col-span-7">
                      <p className="font-mono text-xs uppercase tracking-wide text-muted-foreground">
                        Summary
                      </p>
                      <p className="text-sm font-semibold text-foreground">
                        {report.company_name}{" "}
                        <span className="font-mono text-primary">({report.ticker})</span>
                      </p>
                      {report.investment_recommendation?.trim() ||
                      (report.price_target_usd != null &&
                        Number.isFinite(report.price_target_usd)) ? (
                        <div className="mt-3 rounded-md border border-primary/35 bg-primary/10 p-3">
                          <p className="font-mono text-[10px] uppercase tracking-wide text-primary/90">
                            Desk callout
                          </p>
                          {report.investment_recommendation?.trim() ? (
                            <p className="mt-1 text-base font-bold tracking-tight text-foreground">
                              {report.investment_recommendation.trim()}
                            </p>
                          ) : null}
                          {report.price_target_usd != null &&
                          Number.isFinite(report.price_target_usd) ? (
                            <p className="mt-1 font-mono text-sm text-foreground">
                              12-m price target:{" "}
                              <span className="font-semibold">
                                $
                                {report.price_target_usd.toLocaleString(undefined, {
                                  minimumFractionDigits: 2,
                                  maximumFractionDigits: 2,
                                })}
                              </span>
                              {report.price_target_horizon_months != null ? (
                                <span className="text-muted-foreground">
                                  {" "}
                                  · {report.price_target_horizon_months} mo horizon
                                </span>
                              ) : null}
                            </p>
                          ) : report.investment_recommendation?.trim() ? (
                            <p className="mt-1 text-xs text-muted-foreground">
                              Price target USD not in payload—open the PDF page-1 box or check
                              warnings.
                            </p>
                          ) : null}
                        </div>
                      ) : null}
                      <dl className="mt-3 space-y-2 font-mono text-[11px] text-muted-foreground">
                        <div className="flex justify-between gap-2">
                          <dt>Mode</dt>
                          <dd className="text-foreground">
                            {formatReportMode(report.report_mode)}
                          </dd>
                        </div>
                        {report.gemini_model ? (
                          <div className="flex justify-between gap-2">
                            <dt>Model</dt>
                            <dd className="text-foreground">{report.gemini_model}</dd>
                          </div>
                        ) : null}
                        <div className="flex justify-between gap-2">
                          <dt>Illustrative NPV</dt>
                          <dd className="text-foreground">
                            {report.npv != null ? report.npv.toLocaleString() : "n/a"}
                          </dd>
                        </div>
                        <div className="flex justify-between gap-2">
                          <dt>{formatDiscountRowLabel(report.discount_rate_source)}</dt>
                          <dd className="text-foreground">
                            {(report.discount_rate * 100).toFixed(1)}%
                          </dd>
                        </div>
                        {report.discount_rate_basis ? (
                          <p className="mt-1 text-[10px] leading-snug text-muted-foreground">
                            {report.discount_rate_basis}
                          </p>
                        ) : null}
                        {report.price_target_usd != null &&
                        Number.isFinite(report.price_target_usd) ? (
                          <>
                            <div className="flex justify-between gap-2">
                              <dt>Price target</dt>
                              <dd className="text-foreground">
                                ${report.price_target_usd.toLocaleString(undefined, {
                                  minimumFractionDigits: 2,
                                  maximumFractionDigits: 2,
                                })}
                              </dd>
                            </div>
                            {report.price_target_horizon_months != null ? (
                              <div className="flex justify-between gap-2">
                                <dt>PT horizon</dt>
                                <dd className="text-foreground">
                                  {report.price_target_horizon_months} mo
                                </dd>
                              </div>
                            ) : null}
                          </>
                        ) : (
                          <div className="flex justify-between gap-2">
                            <dt>Price target</dt>
                            <dd className="text-muted-foreground">—</dd>
                          </div>
                        )}
                      </dl>
                      {report.price_target_basis ? (
                        <p className="mt-2 text-[11px] leading-snug text-muted-foreground">
                          <span className="font-medium text-foreground/90">PT basis: </span>
                          {report.price_target_basis}
                        </p>
                      ) : null}
                      {report.data_sources?.length ? (
                        <div className="mt-3 flex flex-wrap gap-1.5">
                          {report.data_sources.map((s) => (
                            <span
                              key={s}
                              className="rounded border border-border bg-secondary/50 px-2 py-0.5 font-mono text-[10px] text-muted-foreground"
                            >
                              {formatDataSourceLabel(s)}
                            </span>
                          ))}
                        </div>
                      ) : null}
                      {report.warnings.length ? (
                        <ul className="mt-3 list-disc space-y-1 pl-4 text-xs text-amber-200/90">
                          {report.warnings.map((w) => (
                            <li key={w}>{w}</li>
                          ))}
                        </ul>
                      ) : null}
                      {report.reasoning ? (
                        <div className="mt-3 rounded-md border border-primary/20 bg-primary/5 p-3">
                          <p className="font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
                            Junior Researcher — reasoning
                          </p>
                          <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-foreground/90">
                            {report.reasoning}
                          </pre>
                        </div>
                      ) : null}
                      {report.critic_memo?.trim() ? (
                        <div className="mt-3 rounded-md border border-border bg-secondary/20 p-3">
                          <p className="font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
                            The Critic — memo
                          </p>
                          <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-foreground/90">
                            {report.critic_memo}
                          </pre>
                        </div>
                      ) : null}
                      {(() => {
                        const lpmStep = report.steps?.find((s) => s.id === "lead_pm");
                        const fromSynth = report.lead_pm_synthesis?.trim();
                        const fromStep = lpmStep?.detail?.trim();
                        const body = fromSynth || fromStep;
                        if (!body) return null;
                        const heading = fromSynth
                          ? "Lead PM — cover synthesis"
                          : "Lead PM — pass detail";
                        return (
                          <div className="mt-3 rounded-md border border-emerald-500/25 bg-emerald-500/5 p-3">
                            <p className="font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
                              {heading}
                            </p>
                            <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-foreground/90">
                              {body}
                            </pre>
                          </div>
                        );
                      })()}
                    </div>
                    <aside className="min-w-0 lg:col-span-6 xl:col-span-5">
                      <div className="max-h-[min(28rem,calc(100vh-6rem))] overflow-y-auto rounded-lg border border-primary/20 bg-gradient-to-b from-card/90 to-card/60 p-3 shadow-md shadow-primary/5 backdrop-blur-sm xl:sticky xl:top-4">
                        <p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
                          <Newspaper className="h-3.5 w-3.5 text-primary" aria-hidden />
                          Headline scan
                        </p>
                        <p className="mt-1 text-[10px] leading-snug text-muted-foreground">
                          Snippets from search; verify at source before relying on them.
                        </p>
                        {report.headlines && report.headlines.length > 0 ? (
                          <ul className="mt-3 space-y-3 border-t border-border/60 pt-3">
                            {report.headlines.map((h, idx) => (
                              <li
                                key={`${h.url || h.title}-${idx}`}
                                className="rounded-md border border-border/50 bg-background/40 p-2.5 text-[11px] leading-snug"
                              >
                                {h.url ? (
                                  <a
                                    href={h.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="font-medium text-primary hover:underline"
                                  >
                                    {h.title || "Untitled"}
                                  </a>
                                ) : (
                                  <span className="font-medium text-foreground">
                                    {h.title || "Untitled"}
                                  </span>
                                )}
                                {h.content ? (
                                  <p className="mt-1.5 text-muted-foreground line-clamp-4">
                                    {h.content}
                                  </p>
                                ) : null}
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="mt-3 rounded-md border border-dashed border-border/80 bg-muted/10 px-2 py-3 text-center text-[11px] text-muted-foreground">
                            No headlines for this run (search empty or unavailable).
                          </p>
                        )}
                      </div>
                    </aside>
                  </div>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">
                  Try <span className="font-mono text-foreground">MSFT</span>,{" "}
                  <span className="font-mono text-foreground">AAPL</span>, or{" "}
                  <span className="font-mono text-foreground">JPM</span>
                </p>
              )}
            </CardContent>
          </Card>

          <Card className="md:col-span-2 xl:col-span-12">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-primary" />
                Pipeline trace
              </CardTitle>
              <CardDescription>
                Last report build steps (filings → market → search → headlines →
                junior discount → FCF → bridge → narrative → editor).
              </CardDescription>
            </CardHeader>
            <CardContent>
              {report?.steps?.length ? (
                <ul className="space-y-2 font-mono text-[11px] text-muted-foreground">
                  {report.steps.map((s, idx) => (
                    <li
                      key={`${idx}-${s.id}`}
                      className="rounded-md border border-border/70 bg-background/40 px-3 py-2"
                    >
                      <span
                        className={
                          s.status === "ok"
                            ? "text-primary"
                            : s.status === "warn"
                              ? "text-amber-300"
                              : s.status === "error"
                                ? "text-destructive"
                                : "text-muted-foreground"
                        }
                      >
                        [{s.status}]
                      </span>{" "}
                      <span className="text-foreground">{s.label}</span>
                      {s.detail ? (
                        <span className="mt-1 block text-[10px] text-muted-foreground">
                          {s.detail}
                        </span>
                      ) : null}
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="rounded-md border border-dashed border-border/80 bg-muted/20 p-4 font-mono text-xs text-muted-foreground">
                  <p className="flex items-center gap-2 text-foreground">
                    <Cpu className="h-4 w-4" />
                    No run yet
                  </p>
                  <p className="mt-2 leading-relaxed">
                    Generate a report to populate this trace.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </main>

      {/* <footer className="border-t border-border/60 py-6 text-center text-xs text-muted-foreground">
        <p className="font-mono">
          Stack:{" "}
          <span className="text-foreground">docker compose up -d</span> +{" "}
          <span className="text-foreground">npm run dev</span> in{" "}
          <span className="text-foreground">frontend/</span>. Set{" "}
          <span className="text-foreground">GEMINI_API_KEY</span>,{" "}
          <span className="text-foreground">SEC_USER_AGENT</span> in your shell or{" "}
          <span className="text-foreground">.env</span> (see repo{" "}
          <span className="text-foreground">.env.example</span>).
        </p>
      </footer> */}
    </div>
  );
}
