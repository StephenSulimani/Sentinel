"""Registration, login, and per-user saved reports."""

from __future__ import annotations

import json
import logging
import re
from flask import Flask, jsonify, request
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError
from werkzeug.security import check_password_hash, generate_password_hash

from auth_tokens import (
    _DEFAULT_DEV_SECRET,
    decode_access_token,
    issue_access_token,
    jwt_secret,
)
from extensions import db
from models import SavedReport, User

_log = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_email(email: str) -> str:
    return email.strip().lower()


def register_auth_routes(app: Flask) -> None:
    @app.post("/api/auth/register")
    def auth_register():
        body = request.get_json(silent=True)
        if body is None:
            return jsonify({"error": "expected JSON body"}), 400
        try:
            payload = AuthCredentials.model_validate(body)
        except ValidationError as exc:
            return jsonify({"error": exc.errors()}), 400
        email = normalize_email(payload.email)
        if not _EMAIL_RE.match(email):
            return jsonify({"error": "invalid email"}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({"error": "email already registered"}), 409
        user = User(
            email=email,
            password_hash=generate_password_hash(payload.password),
        )
        db.session.add(user)
        db.session.commit()
        token = issue_access_token(user_id=user.id)
        return (
            jsonify(
                {
                    "access_token": token,
                    "user": {"id": user.id, "email": user.email},
                }
            ),
            201,
        )

    @app.post("/api/auth/login")
    def auth_login():
        body = request.get_json(silent=True)
        if body is None:
            return jsonify({"error": "expected JSON body"}), 400
        try:
            payload = AuthCredentials.model_validate(body)
        except ValidationError as exc:
            return jsonify({"error": exc.errors()}), 400
        email = normalize_email(payload.email)
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, payload.password):
            return jsonify({"error": "invalid email or password"}), 401
        token = issue_access_token(user_id=user.id)
        return jsonify({"access_token": token, "user": {"id": user.id, "email": user.email}}), 200

    @app.get("/api/auth/me")
    def auth_me():
        uid = _current_user_id()
        if uid is None:
            return jsonify({"error": "unauthorized"}), 401
        user = db.session.get(User, uid)
        if not user:
            return jsonify({"error": "unauthorized"}), 401
        return jsonify({"user": {"id": user.id, "email": user.email}}), 200

    @app.post("/api/saved-reports")
    def saved_reports_create():
        uid = _current_user_id()
        if uid is None:
            return jsonify({"error": "unauthorized"}), 401
        body = request.get_json(silent=True)
        if body is None:
            return jsonify({"error": "expected JSON body"}), 400
        report = body.get("report")
        if not isinstance(report, dict):
            return jsonify({"error": "report must be a JSON object"}), 400
        ticker = str(report.get("ticker") or "").strip().upper()[:32] or "NOTE"
        company = str(report.get("company_name") or ticker).strip()[:512] or ticker
        try:
            dumped = json.dumps(report, default=str)
        except (TypeError, ValueError) as exc:
            return jsonify({"error": f"report not serializable: {exc}"}), 400
        row = SavedReport(user_id=uid, ticker=ticker, company_name=company, report_json=dumped)
        db.session.add(row)
        db.session.commit()
        return (
            jsonify(
                {
                    "id": row.id,
                    "ticker": row.ticker,
                    "company_name": row.company_name,
                    "created_at": row.created_at.isoformat() + "Z",
                }
            ),
            201,
        )

    @app.get("/api/saved-reports")
    def saved_reports_list():
        uid = _current_user_id()
        if uid is None:
            return jsonify({"error": "unauthorized"}), 401
        rows = (
            SavedReport.query.filter_by(user_id=uid)
            .order_by(SavedReport.created_at.desc())
            .limit(200)
            .all()
        )
        return (
            jsonify(
                {
                    "items": [
                        {
                            "id": r.id,
                            "ticker": r.ticker,
                            "company_name": r.company_name,
                            "created_at": r.created_at.isoformat() + "Z",
                        }
                        for r in rows
                    ]
                }
            ),
            200,
        )

    @app.get("/api/saved-reports/<int:report_id>")
    def saved_reports_get(report_id: int):
        uid = _current_user_id()
        if uid is None:
            return jsonify({"error": "unauthorized"}), 401
        row = SavedReport.query.filter_by(id=report_id, user_id=uid).first()
        if not row:
            return jsonify({"error": "not found"}), 404
        try:
            report = json.loads(row.report_json)
        except json.JSONDecodeError:
            return jsonify({"error": "stored report is corrupt"}), 500
        return (
            jsonify(
                {
                    "id": row.id,
                    "ticker": row.ticker,
                    "company_name": row.company_name,
                    "created_at": row.created_at.isoformat() + "Z",
                    "report": report,
                }
            ),
            200,
        )

    @app.delete("/api/saved-reports/<int:report_id>")
    def saved_reports_delete(report_id: int):
        uid = _current_user_id()
        if uid is None:
            return jsonify({"error": "unauthorized"}), 401
        row = SavedReport.query.filter_by(id=report_id, user_id=uid).first()
        if not row:
            return jsonify({"error": "not found"}), 404
        db.session.delete(row)
        db.session.commit()
        return jsonify({"ok": True}), 200


class AuthCredentials(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8, max_length=256)


def _bearer_token() -> str | None:
    h = request.headers.get("Authorization", "")
    if h.lower().startswith("bearer "):
        return h[7:].strip() or None
    return None


def _current_user_id() -> int | None:
    tok = _bearer_token()
    if not tok:
        return None
    return decode_access_token(tok)


def init_auth(app: Flask) -> None:
    """Create auth tables if missing (idempotent; safe across repeated app init)."""
    secret = jwt_secret()
    if secret == _DEFAULT_DEV_SECRET:
        _log.warning(
            "JWT_SECRET_KEY is unset; using insecure dev default. Set JWT_SECRET_KEY in production."
        )
    with app.app_context():
        try:
            names = set(inspect(db.engine).get_table_names())
        except Exception as exc:  # noqa: BLE001
            _log.warning("Could not inspect database before create_all: %s", exc)
            names = set()

        # Fast path: avoid calling create_all when both tables are already there (prevents
        # rare SQLite / multi-init races that surface as "table users already exists").
        if names >= {"users", "saved_reports"}:
            return

        try:
            db.create_all()
        except OperationalError as exc:
            msg = (str(getattr(exc, "orig", None)) + str(exc)).lower()
            if "already exists" in msg:
                _log.info(
                    "Database tables already present (create_all skipped): %s",
                    getattr(exc, "orig", exc),
                )
                return
            raise
