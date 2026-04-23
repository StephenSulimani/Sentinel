"""Compile LaTeX source to PDF using ``pdflatex`` (two passes for stable references)."""

from __future__ import annotations

import re
import shutil
import time
import subprocess
import tempfile
from pathlib import Path

from gemini_report import (
    is_transient_gemini_error_message,
    repair_latex_for_pdflatex_log,
)


def pdflatex_available() -> bool:
    return shutil.which("pdflatex") is not None


def compile_latex_pdf(tex: str, *, jobname: str = "equity_note") -> tuple[bytes | None, str]:
    """Return ``(pdf_bytes, error_message)``. ``error_message`` is empty on success."""
    if not pdflatex_available():
        return None, "pdflatex not installed (install TeX Live in the AI image or locally)."

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", jobname).strip("_") or "equity_note"
    if len(safe) > 80:
        safe = safe[:80]

    tex = tex.strip()
    if not tex:
        return None, "empty LaTeX source"

    with tempfile.TemporaryDirectory(prefix="sentinel_tex_") as tmp:
        tmp_path = Path(tmp)
        tex_path = tmp_path / f"{safe}.tex"
        tex_path.write_text(tex, encoding="utf-8")
        log_parts: list[str] = []
        for pass_num in (1, 2):
            proc = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"-jobname={safe}",
                    str(tex_path.name),
                ],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.stdout:
                log_parts.append(proc.stdout[-4000:])
            if proc.stderr:
                log_parts.append(proc.stderr[-2000:])
            if proc.returncode != 0:
                tail = "\n".join(log_parts)[-6000:]
                return None, f"pdflatex pass {pass_num} failed (exit {proc.returncode}):\n{tail}"

        pdf_path = tmp_path / f"{safe}.pdf"
        if not pdf_path.is_file():
            return None, "pdflatex finished but PDF file was not produced."

        return pdf_path.read_bytes(), ""


def compile_latex_pdf_nonhalt(tex: str, *, jobname: str = "equity_note") -> tuple[bytes | None, str]:
    """Run ``pdflatex`` without ``-halt-on-error``; return PDF bytes if a PDF is produced.

    Used as a last resort when strict compilation and model repair did not yield
    a PDF (some notes still produce a viewable PDF with non-fatal errors).
    """
    if not pdflatex_available():
        return None, "pdflatex not installed (install TeX Live in the AI image or locally)."

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", jobname).strip("_") or "equity_note"
    if len(safe) > 80:
        safe = safe[:80]

    tex = tex.strip()
    if not tex:
        return None, "empty LaTeX source"

    with tempfile.TemporaryDirectory(prefix="sentinel_tex_") as tmp:
        tmp_path = Path(tmp)
        tex_path = tmp_path / f"{safe}.tex"
        tex_path.write_text(tex, encoding="utf-8")
        log_parts: list[str] = []
        for pass_num in (1, 2):
            proc = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    f"-jobname={safe}",
                    str(tex_path.name),
                ],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.stdout:
                log_parts.append(proc.stdout[-4000:])
            if proc.stderr:
                log_parts.append(proc.stderr[-2000:])
            if proc.returncode != 0:
                log_parts.append(f"[nonhalt pass {pass_num} exit {proc.returncode}]")

        pdf_path = tmp_path / f"{safe}.pdf"
        if pdf_path.is_file():
            return pdf_path.read_bytes(), ""

        tail = "\n".join(log_parts)[-6000:]
        return None, f"pdflatex (nonhalt) did not produce a PDF:\n{tail}"


def compile_latex_pdf_with_repair(
    tex: str,
    *,
    jobname: str = "equity_note",
    gemini_api_key: str | None = None,
    gemini_model: str | None = None,
    max_repair_rounds: int = 3,
    enable_repair: bool = True,
) -> tuple[bytes | None, str, str, int]:
    """Compile to PDF; on failure optionally ask Gemini to fix LaTeX and retry.

    Returns ``(pdf_bytes, error_message, latex_used, repair_rounds)``.
    ``latex_used`` is the TeX that produced ``pdf_bytes`` when successful, else
    the last source attempted. ``repair_rounds`` counts successful model repairs
    before a successful strict ``pdflatex`` run.
    """
    key = (gemini_api_key or "").strip() or None
    model = (gemini_model or "").strip() or "gemini-2.0-flash"
    rounds = max(0, min(int(max_repair_rounds), 5))
    current = tex.strip()
    last_err = ""
    repairs_done = 0

    for round_idx in range(rounds + 1):
        pdf, err = compile_latex_pdf(current, jobname=jobname)
        if pdf:
            return pdf, "", current, repairs_done
        last_err = err
        if round_idx >= rounds or not key or not enable_repair:
            break
        fixed, rep_err = repair_latex_for_pdflatex_log(
            api_key=key,
            model=model,
            latex_source=current,
            pdflatex_log_tail=err,
        )
        # Rare: provider still returns 503 in the error string after inner retries; wait and retry once more.
        extra = 0
        while rep_err and is_transient_gemini_error_message(rep_err) and extra < 3:
            time.sleep(min(45.0, 8.0 * (extra + 1)))
            fixed, rep_err = repair_latex_for_pdflatex_log(
                api_key=key,
                model=model,
                latex_source=current,
                pdflatex_log_tail=err,
            )
            extra += 1
        if rep_err:
            return None, f"{last_err}\n[latex repair {round_idx + 1}] {rep_err}", current, repairs_done
        if not fixed.strip():
            return None, f"{last_err}\n[latex repair {round_idx + 1}] empty fix from model", current, repairs_done
        if fixed.strip() == current.strip():
            return (
                None,
                f"{last_err}\n[latex repair {round_idx + 1}] model returned unchanged source",
                current,
                repairs_done,
            )
        current = fixed.strip()
        repairs_done += 1

    # Last resort: non-stopmode pdflatex (may produce a PDF when halt-on-error would abort).
    for label, candidate in (("repaired", current), ("original", tex.strip())):
        pdf_loose, loose_err = compile_latex_pdf_nonhalt(candidate, jobname=jobname)
        if pdf_loose:
            return pdf_loose, "", candidate, repairs_done
        last_err = f"{last_err}\n[best-effort {label} nonhalt pdflatex] {loose_err}"

    return None, last_err, current, repairs_done
