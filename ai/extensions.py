"""Shared Flask extensions (imported by models and app)."""

from __future__ import annotations

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
