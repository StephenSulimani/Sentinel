import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
};

function formatReportMode(mode: string | undefined): string {
  if (!mode) return "—";
  const labels: Record<string, string> = {
    gemini: "Narrative model",
    deterministic: "Templated draft",
    deterministic_yfinance: "Templated draft",
  };
  return labels[mode] ?? mode;
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

type StreamEnvelope =
  | { type: "phase"; id: string; status: string; label?: string; detail?: string }
  | { type: "reasoning"; text: string; truncated?: boolean }
  | { type: "reasoning_delta"; text: string }
  | {
      type: "price_target";
      usd: number;
      horizon_months?: number | null;
      basis?: string;
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

const TICKER_PATTERN = /^[A-Za-z0-9.\-]{1,12}$/;

export default function App() {
  const go = useHealth(backendUrl, "/health");
  const ai = useHealth(aiUrl, "/health");

  const [ticker, setTicker] = useState("MSFT");
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState<string>("");
  const [report, setReport] = useState<ReportPayload | null>(null);
  const [streamEvents, setStreamEvents] = useState<string[]>([]);
  const [reasoningLive, setReasoningLive] = useState<string>("");
  const reportStreamRef = useRef<EventSource | null>(null);

  const [discount, setDiscount] = useState("0.10");
  const [flowsText, setFlowsText] = useState("100, 110, 121, 133.1, 146.41");
  const [npvResult, setNpvResult] = useState<string>("");
  const [npvBusy, setNpvBusy] = useState(false);

  const [tex, setTex] = useState<string>("");
  const [texBusy, setTexBusy] = useState(false);

  const [pdfBusy, setPdfBusy] = useState(false);
  const [pdfMessage, setPdfMessage] = useState<string>("");

  const [outputTab, setOutputTab] = useState<"pdf" | "latex">("pdf");
  const [pdfPreviewUrl, setPdfPreviewUrl] = useState<string | null>(null);
  const [pdfPreviewBusy, setPdfPreviewBusy] = useState(false);
  const [pdfPreviewError, setPdfPreviewError] = useState<string>("");

  const parseFlows = useMemo(() => {
    return flowsText
      .split(/[,\s]+/)
      .map((s) => s.trim())
      .filter(Boolean)
      .map((s) => Number(s));
  }, [flowsText]);

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
    setReportError("");
    setReport(null);
    setStreamEvents([]);
    setReasoningLive("");
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
      if (data.type === "phase") {
        const parts = [
          data.id,
          data.status,
          data.label,
          data.detail,
        ].filter(Boolean);
        setStreamEvents((prev) => [...prev, parts.join(" · ")]);
        return;
      }
      if (data.type === "reasoning") {
        setReasoningLive((prev) => {
          const next = prev ? `${prev}\n\n` : "";
          const tail = data.truncated ? "\n\n[truncated in stream]" : "";
          return `${next}${data.text}${tail}`;
        });
        return;
      }
      if (data.type === "reasoning_delta") {
        if (data.text) {
          setReasoningLive((prev) => prev + data.text);
        }
        return;
      }
      if (data.type === "price_target") {
        const hm =
          data.horizon_months != null ? `${data.horizon_months}m horizon` : "horizon n/a";
        const line = `price_target · $${data.usd.toFixed(2)} · ${hm}${data.basis ? ` · ${data.basis.slice(0, 120)}` : ""}`;
        setStreamEvents((prev) => [...prev, line]);
        return;
      }
      if (data.type === "error") {
        finished = true;
        es.close();
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
      }
    };

    es.onerror = () => {
      if (!finished) {
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

  const runNpv = async () => {
    setNpvBusy(true);
    setNpvResult("");
    const rate = Number(discount);
    if (!Number.isFinite(rate) || rate <= -1) {
      setNpvResult("discount must be a number > -1");
      setNpvBusy(false);
      return;
    }
    const flows = parseFlows;
    if (flows.some((n) => !Number.isFinite(n))) {
      setNpvResult("cash flows must be numbers");
      setNpvBusy(false);
      return;
    }
    try {
      const res = await fetch(`${backendUrl}/npv`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cash_flows: flows,
          discount_rate: rate,
        }),
      });
      const text = await res.text();
      setNpvResult(text);
    } catch (e) {
      setNpvResult(e instanceof Error ? e.message : "request failed");
    } finally {
      setNpvBusy(false);
    }
  };

  const loadSampleTex = async () => {
    setTexBusy(true);
    setTex("");
    try {
      const res = await fetch(`${aiUrl}/report/sample`);
      const text = await res.text();
      setTex(res.ok ? text : `error ${res.status}: ${text.slice(0, 400)}`);
    } catch (e) {
      setTex(e instanceof Error ? e.message : "request failed");
    } finally {
      setTexBusy(false);
    }
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,_hsl(152_76%_42%/0.12),_transparent_55%),radial-gradient(ellipse_at_bottom,_hsl(215_28%_14%/0.9),_hsl(224_71%_4%))]">
      <header className="border-b border-border/80 bg-card/40 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 rounded-md border border-border bg-secondary/60 p-2">
              <Terminal className="h-6 w-6 text-primary" aria-hidden />
            </div>
            <div>
              <p className="font-mono text-xs uppercase tracking-[0.2em] text-muted-foreground">
                Project Sentinel
              </p>
              <h1 className="text-2xl font-semibold tracking-tight text-foreground">
                Equity analyst console
              </h1>
              <p className="mt-1 max-w-xl text-sm text-muted-foreground">
                Enter a ticker to assemble filing facts, market history, a
                headline scan, and an illustrative valuation bridge into a LaTeX
                research note with automatic PDF preview.
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="outline"
              size="sm"
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
                (SSE); one-shot JSON at{" "}
                <span className="font-mono text-foreground">POST /report/generate</span>.
                Not investment advice.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
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

              {reportLoading || streamEvents.length > 0 || reasoningLive ? (
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
                      Analyst reasoning (live)
                    </p>
                    <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap font-mono text-[10px] leading-relaxed text-foreground/90">
                      {reasoningLive ||
                        (reportLoading
                          ? "Waiting for streamed reasoning…"
                          : "—")}
                    </pre>
                  </div>
                </div>
              ) : null}

              {report ? (
                <div className="grid gap-4 lg:grid-cols-12 lg:items-start">
                  <div className="space-y-2 rounded-md border border-border bg-background/50 p-4 lg:col-span-3">
                    <p className="font-mono text-xs uppercase tracking-wide text-muted-foreground">
                      Summary
                    </p>
                    <p className="text-sm font-semibold text-foreground">
                      {report.company_name}{" "}
                      <span className="font-mono text-primary">({report.ticker})</span>
                    </p>
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
                        <dt>Discount (inferred)</dt>
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
                          Final reasoning
                        </p>
                        <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-foreground/90">
                          {report.reasoning}
                        </pre>
                      </div>
                    ) : null}
                    {pdfMessage ? (
                      <p className="mt-3 rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-2 font-mono text-[10px] text-amber-100/95 whitespace-pre-wrap">
                        {pdfMessage}
                      </p>
                    ) : null}
                    <div className="mt-4 flex flex-col gap-2 sm:flex-row">
                      <Button
                        variant="outline"
                        size="sm"
                        className="w-full sm:flex-1"
                        onClick={downloadTex}
                      >
                        <Download className="h-4 w-4" />
                        Download .tex
                      </Button>
                      <Button
                        variant="default"
                        size="sm"
                        className="w-full sm:flex-1"
                        onClick={() => void downloadPdf()}
                        disabled={pdfBusy}
                      >
                        <FileDown className="h-4 w-4" />
                        {pdfBusy ? "Building PDF…" : "Download .pdf"}
                      </Button>
                    </div>
                    <p className="mt-2 font-mono text-[10px] text-muted-foreground">
                      PDF uses server{" "}
                      <span className="text-foreground">pdflatex</span> plus optional{" "}
                      <span className="text-foreground">Model</span> repair passes on
                      compile errors (see{" "}
                      <span className="text-foreground">GET /health</span> →{" "}
                      <span className="text-foreground">latex_repair_available</span>
                      ).
                    </p>
                  </div>
                  <div className="min-w-0 lg:col-span-6">
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
                  </div>
                  <aside className="lg:col-span-3">
                    <div className="rounded-lg border border-primary/20 bg-gradient-to-b from-card/90 to-card/60 p-3 shadow-md shadow-primary/5 backdrop-blur-sm lg:sticky lg:top-4 lg:max-h-[calc(100vh-5rem)] lg:overflow-y-auto">
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
              ) : (
                <p className="text-sm text-muted-foreground">
                  Try <span className="font-mono text-foreground">MSFT</span>,{" "}
                  <span className="font-mono text-foreground">AAPL</span>, or{" "}
                  <span className="font-mono text-foreground">JPM</span> with
                  the research stack running (
                  <span className="font-mono text-foreground">docker compose up</span>
                  ).
                </p>
              )}
            </CardContent>
          </Card>

          <Card className="md:col-span-1 xl:col-span-4">
            <CardHeader className="flex flex-row items-start justify-between gap-3 space-y-0">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Server className="h-4 w-4 text-primary" />
                  Valuation engine
                </CardTitle>
                <CardDescription className="font-mono text-[11px]">
                  {backendUrl}
                </CardDescription>
              </div>
              <StatusDot state={go.state} />
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center justify-between rounded-md border border-border bg-background/60 px-3 py-2 font-mono text-xs">
                <span className="text-muted-foreground">GET /health</span>
                <span
                  className={
                    go.state === "ok"
                      ? "text-primary"
                      : go.state === "error"
                        ? "text-destructive"
                        : "text-muted-foreground"
                  }
                >
                  {go.state === "ok"
                    ? "ONLINE"
                    : go.state === "error"
                      ? "OFFLINE"
                      : "…"}
                </span>
              </div>
              <pre className="max-h-28 overflow-auto rounded-md border border-border bg-black/30 p-3 font-mono text-[11px] leading-relaxed text-muted-foreground">
                {go.detail || "—"}
              </pre>
            </CardContent>
          </Card>

          <Card className="md:col-span-1 xl:col-span-4">
            <CardHeader className="flex flex-row items-start justify-between gap-3 space-y-0">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-primary" />
                  Research API
                </CardTitle>
                <CardDescription className="font-mono text-[11px]">
                  {aiUrl}
                </CardDescription>
              </div>
              <StatusDot state={ai.state} />
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center justify-between rounded-md border border-border bg-background/60 px-3 py-2 font-mono text-xs">
                <span className="text-muted-foreground">GET /health</span>
                <span
                  className={
                    ai.state === "ok"
                      ? "text-primary"
                      : ai.state === "error"
                        ? "text-destructive"
                        : "text-muted-foreground"
                  }
                >
                  {ai.state === "ok"
                    ? "ONLINE"
                    : ai.state === "error"
                      ? "OFFLINE"
                      : "…"}
                </span>
              </div>
              <pre className="max-h-28 overflow-auto rounded-md border border-border bg-black/30 p-3 font-mono text-[11px] leading-relaxed text-muted-foreground">
                {ai.detail || "—"}
              </pre>
            </CardContent>
          </Card>

          <Card className="md:col-span-2 xl:col-span-4">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-primary" />
                Pipeline trace
              </CardTitle>
              <CardDescription>
                Last report build steps (filings → market → discount → FCF →
                bridge → headlines → narrative → editor).
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

          <Card className="md:col-span-2 xl:col-span-7">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 normal-case tracking-normal text-foreground">
                <Cpu className="h-4 w-4 text-primary" />
                Scenario NPV
              </CardTitle>
              <CardDescription>
                POST <span className="font-mono text-foreground">/npv</span> with
                JSON body for ad-hoc scenarios.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="discount">Discount rate</Label>
                  <Input
                    id="discount"
                    value={discount}
                    onChange={(e) => setDiscount(e.target.value)}
                    placeholder="0.10"
                  />
                </div>
                <div className="space-y-2 sm:col-span-2">
                  <Label htmlFor="flows">Cash flows (comma-separated)</Label>
                  <Input
                    id="flows"
                    value={flowsText}
                    onChange={(e) => setFlowsText(e.target.value)}
                    placeholder="100, 110, 121"
                  />
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button onClick={() => void runNpv()} disabled={npvBusy}>
                  {npvBusy ? "Running…" : "Run scenario"}
                </Button>
                <Button variant="secondary" onClick={() => void runNpv()}>
                  Re-run
                </Button>
              </div>
              <pre className="max-h-48 overflow-auto rounded-md border border-border bg-black/40 p-4 font-mono text-xs leading-relaxed text-primary">
                {npvResult || "—"}
              </pre>
            </CardContent>
          </Card>

          <Card className="md:col-span-2 xl:col-span-5">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 normal-case tracking-normal text-foreground">
                <FileText className="h-4 w-4 text-primary" />
                LaTeX template smoke test
              </CardTitle>
              <CardDescription>
                Static sample via{" "}
                <span className="font-mono text-foreground">GET /report/sample</span>.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button
                variant="outline"
                onClick={() => void loadSampleTex()}
                disabled={texBusy}
              >
                {texBusy ? "Loading…" : "Load sample .tex"}
              </Button>
              <pre className="max-h-[28rem] overflow-auto rounded-md border border-border bg-black/40 p-4 font-mono text-[11px] leading-relaxed text-muted-foreground">
                {tex || "—"}
              </pre>
            </CardContent>
          </Card>
        </div>
      </main>

      <footer className="border-t border-border/60 py-6 text-center text-xs text-muted-foreground">
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
      </footer>
    </div>
  );
}
