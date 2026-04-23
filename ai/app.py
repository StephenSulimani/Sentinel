"""Flask service for Project Sentinel AI slice.

SearXNG JSON API: SEARXNG_URL (default http://localhost:8080 on host; http://searxng:8080 in compose).
Go math engine URL: GO_MATH_URL (default http://localhost:8081 for host dev).
Gemini: GEMINI_API_KEY + optional GEMINI_MODEL (default gemini-2.0-flash).
SEC EDGAR: SEC_USER_AGENT must identify your deployment (SEC policy).
Market history: yfinance (no Alpaca).
"""

from __future__ import annotations

import base64
import json
import os
import queue
import re
import threading
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field, ValidationError

from latex_pdf import compile_latex_pdf_with_repair, pdflatex_available
from report_builder import ReportResult, latex_escape, result_to_json_dict
from research_pipeline import generate_full_report

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Local dev: pick up repo-root `.env` when not injected by Docker Compose.
load_dotenv(_REPO_ROOT / ".env", override=False)
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

GO_MATH_URL = os.environ.get("GO_MATH_URL", "http://localhost:8081").rstrip("/")
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080").rstrip("/")

_DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


def _clean_secret(value: str | None) -> str | None:
    if not value:
        return None
    s = str(value).strip().strip('"').strip("'")
    return s or None


def _gemini_api_key() -> str | None:
    return _clean_secret(os.environ.get("GEMINI_API_KEY"))


def _gemini_model() -> str:
    return _clean_secret(os.environ.get("GEMINI_MODEL")) or _DEFAULT_GEMINI_MODEL

SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "ProjectSentinel/1.0 (mailto:dev@localhost)",
)

_template_dir = Path(__file__).resolve().parent / "templates"
_env = Environment(
    loader=FileSystemLoader(str(_template_dir)),
    autoescape=select_autoescape(enabled_extensions=()),
)
_env.filters["latex"] = latex_escape


class NPVPayload(BaseModel):
    cash_flows: list[float] = Field(default_factory=list)
    discount_rate: float


class ReportGeneratePayload(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=12)


class ReportPdfPayload(BaseModel):
    latex: str = Field(..., min_length=20, max_length=1_500_000)
    ticker: str = Field(default="NOTE", min_length=1, max_length=12)
    repair_with_gemini: bool = True
    max_repair_rounds: int = Field(default=3, ge=0, le=5)
    response_format: Literal["pdf", "json"] = "pdf"


def create_app() -> Flask:
    app = Flask(__name__)

    @app.after_request
    def _cors(response: Response) -> Response:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    @app.before_request
    def _cors_preflight() -> Response | None:
        if request.method == "OPTIONS":
            return Response(status=204)
        return None

    @app.get("/health")
    def health() -> tuple[dict[str, object], int]:
        return {
            "status": "ok",
            "gemini_configured": bool(_gemini_api_key()),
            "pdflatex_available": pdflatex_available(),
            "latex_repair_available": bool(_gemini_api_key()) and pdflatex_available(),
        }, 200

    @app.post("/demo-npv")
    def demo_npv() -> tuple[Response, int]:
        body = request.get_json(silent=True)
        if body is None:
            return jsonify({"error": "expected JSON body"}), 400
        try:
            payload = NPVPayload.model_validate(body)
        except ValidationError as exc:
            return jsonify({"error": exc.errors()}), 400

        try:
            r = httpx.post(
                f"{GO_MATH_URL}/npv",
                json={
                    "cash_flows": payload.cash_flows,
                    "discount_rate": payload.discount_rate,
                },
                timeout=10.0,
            )
        except httpx.RequestError as exc:
            return jsonify({"error": f"upstream math service: {exc}"}), 502

        content_type = r.headers.get("content-type", "")
        if "application/json" in content_type:
            return Response(
                r.content,
                status=r.status_code,
                content_type="application/json",
            )
        return Response(r.content, status=r.status_code)

    @app.post("/report/generate")
    def report_generate() -> tuple[Response, int]:
        body = request.get_json(silent=True)
        if body is None:
            return jsonify({"error": "expected JSON body"}), 400
        try:
            payload = ReportGeneratePayload.model_validate(body)
        except ValidationError as exc:
            return jsonify({"error": exc.errors()}), 400

        try:
            with httpx.Client() as client:
                result = generate_full_report(
                    ticker=payload.ticker,
                    go_math_url=GO_MATH_URL,
                    searxng_url=SEARXNG_URL,
                    httpx_client=client,
                    sec_user_agent=SEC_USER_AGENT,
                    gemini_api_key=_gemini_api_key(),
                    gemini_model=_gemini_model(),
                )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 502
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500

        return jsonify(result_to_json_dict(result)), 200

    @app.post("/report/pdf")
    def report_pdf() -> tuple[Response, int]:
        """Compile report LaTeX to PDF (``pdflatex`` in the AI container or local TeX install)."""
        body = request.get_json(silent=True)
        if body is None:
            return jsonify({"error": "expected JSON body"}), 400
        try:
            payload = ReportPdfPayload.model_validate(body)
        except ValidationError as exc:
            return jsonify({"error": exc.errors()}), 400

        if not pdflatex_available():
            return jsonify(
                {
                    "error": "pdflatex not available on this server; rebuild the AI image or install TeX Live.",
                }
            ), 503

        sym = re.sub(r"[^A-Za-z0-9._-]+", "", payload.ticker.strip().upper()) or "NOTE"
        pdf_bytes, err, latex_out, repair_rounds = compile_latex_pdf_with_repair(
            payload.latex,
            jobname=f"{sym}_equity_note",
            gemini_api_key=_gemini_api_key(),
            gemini_model=_gemini_model(),
            max_repair_rounds=payload.max_repair_rounds,
            enable_repair=payload.repair_with_gemini,
        )

        if payload.response_format == "json":
            body: dict[str, object] = {
                "latex_compiled": latex_out,
                "repair_rounds": repair_rounds,
                "error": None if pdf_bytes and not err else (err or "PDF build failed"),
            }
            if pdf_bytes and not err:
                body["pdf_base64"] = base64.b64encode(pdf_bytes).decode("ascii")
                return jsonify(body), 200
            body["pdf_base64"] = None
            return jsonify(body), 422

        if not pdf_bytes or err:
            return jsonify({"error": err or "PDF build failed"}), 422

        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{sym}_equity_note.pdf"',
            },
        )

    @app.get("/report/stream")
    def report_stream() -> Response:
        """Server-Sent Events: phase updates, optional Gemini reasoning, then complete report JSON."""
        ticker = (request.args.get("ticker") or "").strip()
        try:
            payload = ReportGeneratePayload(ticker=ticker)
        except ValidationError as exc:
            err = json.dumps({"type": "error", "detail": exc.errors()})

            def err_only():
                yield f"data: {err}\n\n"

            return Response(
                err_only(),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        out_q: queue.Queue[dict[str, object] | None] = queue.Queue()
        result_holder: list[ReportResult | None] = [None]
        error_box: list[str] = []

        def emit(ev: dict[str, object]) -> None:
            out_q.put(ev)

        def worker() -> None:
            try:
                with httpx.Client() as client:
                    result_holder[0] = generate_full_report(
                        ticker=payload.ticker,
                        go_math_url=GO_MATH_URL,
                        searxng_url=SEARXNG_URL,
                        httpx_client=client,
                        sec_user_agent=SEC_USER_AGENT,
                        gemini_api_key=_gemini_api_key(),
                        gemini_model=_gemini_model(),
                        emit=emit,
                    )
            except ValueError as exc:
                error_box.append(str(exc))
            except Exception as exc:  # noqa: BLE001
                error_box.append(str(exc))
            finally:
                out_q.put(None)

        threading.Thread(target=worker, daemon=True).start()

        def event_iter():
            while True:
                item = out_q.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item, default=str)}\n\n"
            if error_box:
                yield f"data: {json.dumps({'type': 'error', 'detail': error_box[0]})}\n\n"
            elif result_holder[0] is not None:
                rep = result_to_json_dict(result_holder[0])
                yield f"data: {json.dumps({'type': 'complete', 'report': rep}, default=str)}\n\n"

        return Response(
            event_iter(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/report/sample")
    def report_sample() -> Response:
        tmpl = _env.get_template("report.tex.j2")
        tex = tmpl.render(
            ticker="DEMO",
            report_date="2026-04-21",
            company_name="Demo Corp",
            thesis="Cash flows are illustrative only; not investment advice.",
            risks="Model risk, liquidity, macro, and single-name concentration.",
            dcf_summary_rows=[
                ("WACC (illustrative)", "10.0%"),
                ("Terminal growth", "2.5%"),
                ("Implied NPV (5y FCF stub)", "see /npv JSON demo"),
            ],
            snapshot_rows=[("Exchange", "n/a"), ("Currency", "USD")],
            headlines=[],
            price_target_usd=142.0,
            price_target_horizon_months=12,
            price_target_basis="Illustrative demo target only; not from live data.",
        )
        return Response(
            tex,
            mimetype="text/plain; charset=utf-8",
            headers={"Content-Disposition": 'inline; filename="sample_report.tex"'},
        )

    return app


app = create_app()
