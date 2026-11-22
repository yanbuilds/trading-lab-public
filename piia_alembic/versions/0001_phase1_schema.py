"""Phase-1 PIIA schema.

Revision ID: 0001_phase1
Revises:
Create Date: 2026-11-16
PostgreSQL-compatible types are used, but PostgreSQL itself is not tested in this slice.
"""
from __future__ import annotations
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from piia.intelligence.db import install_immutability, IMMUTABLE_TABLES

revision: str = "0001_phase1"
down_revision: Union[str, Sequence[str], None] = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        "instruments",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("symbol", sa.String(64), nullable=False),
        sa.Column("instrument_type", sa.String(32), nullable=False),
        sa.Column("venue", sa.String(64), nullable=True),
        sa.Column("contract_symbol", sa.String(64), nullable=True),
        sa.Column("settlement_currency", sa.String(16), nullable=True),
        sa.Column("contract_size", sa.Float(), nullable=True),
        sa.Column("market_type", sa.String(32), nullable=False),
        sa.Column("asset_class", sa.String(32), nullable=False),
        sa.Column("quote_currency", sa.String(16), nullable=False),
        sa.Column("display_name", sa.String(128), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
    )
    op.create_table(
        "evidence_items",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("provider", sa.String(128), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("quote_locator", sa.Text(), nullable=False, server_default=""),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("timezone", sa.String(64), nullable=False, server_default="UTC"),
        sa.Column("data_quality", sa.String(32), nullable=False),
        sa.Column("methodology", sa.Text(), nullable=False, server_default=""),
        sa.Column("methodology_version", sa.String(64), nullable=False, server_default="unspecified"),
        sa.Column("vintage", sa.String(64), nullable=True),
        sa.Column("label", sa.String(64), nullable=False, server_default="user_provided"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("content_hash", name="uq_evidence_hash"),
    )
    op.create_table(
        "theses",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("instrument_id", sa.String(64), sa.ForeignKey("instruments.id"), nullable=False),
        sa.Column("author", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("label", sa.String(64), nullable=False),
        sa.Column("current_revision_id", sa.String(64), nullable=True),
    )
    op.create_table(
        "thesis_revisions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("thesis_id", sa.String(64), sa.ForeignKey("theses.id"), nullable=False),
        sa.Column("revision_no", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("author", sa.String(128), nullable=False),
        sa.Column("direction", sa.String(16), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("invalidation", sa.Text(), nullable=False),
        sa.Column("subjective_confidence", sa.Float(), nullable=False),
        sa.Column("change_reason", sa.Text(), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("is_example", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.UniqueConstraint("thesis_id", "revision_no", name="uq_thesis_revno"),
    )
    op.create_table(
        "evidence_links",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("revision_id", sa.String(64), sa.ForeignKey("thesis_revisions.id"), nullable=False),
        sa.Column("evidence_id", sa.String(64), sa.ForeignKey("evidence_items.id"), nullable=False),
        sa.Column("relation", sa.String(16), nullable=False),
        sa.Column("note", sa.Text(), nullable=False, server_default=""),
        sa.UniqueConstraint("revision_id", "evidence_id", "relation", name="uq_rev_ev_rel"),
    )
    op.create_table(
        "counterarguments",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("revision_id", sa.String(64), sa.ForeignKey("thesis_revisions.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("generation", sa.String(32), nullable=False),
        sa.Column("model_name", sa.String(128), nullable=True),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("evidence_ids_json", sa.JSON(), nullable=False),
    )
    op.create_table(
        "analyses",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("revision_id", sa.String(64), sa.ForeignKey("thesis_revisions.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("generation", sa.String(32), nullable=False),
        sa.Column("model_name", sa.String(128), nullable=True),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("evidence_ids_json", sa.JSON(), nullable=False),
        sa.Column("order_permissions", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_table(
        "portfolio_snapshots",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("label", sa.String(64), nullable=False),
        sa.Column("cash", sa.Float(), nullable=False),
        sa.Column("liabilities", sa.Float(), nullable=False),
        sa.Column("positions_json", sa.JSON(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
    )
    op.create_table(
        "risk_configs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.UniqueConstraint("config_hash", name="uq_risk_config_hash"),
    )
    op.create_table(
        "plans",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revision_id", sa.String(64), sa.ForeignKey("thesis_revisions.id"), nullable=False),
        sa.Column("portfolio_id", sa.String(64), sa.ForeignKey("portfolio_snapshots.id"), nullable=False),
        sa.Column("risk_config_id", sa.String(64), sa.ForeignKey("risk_configs.id"), nullable=False),
        sa.Column("plan_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("actionable", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("side", sa.String(16), nullable=False, server_default="long"),
        sa.Column("leverage", sa.Float(), nullable=False, server_default="1"),
        sa.Column("margin_mode", sa.String(32), nullable=True),
        sa.Column("stop_loss", sa.Float(), nullable=True),
        sa.Column("take_profit", sa.Float(), nullable=True),
        sa.Column("exit_condition", sa.Text(), nullable=True),
        sa.Column("margin_model_supported", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("liquidation_model_supported", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("request_json", sa.JSON(), nullable=False),
        sa.Column("risk_result_json", sa.JSON(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "decisions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("plan_id", sa.String(64), sa.ForeignKey("plans.id"), nullable=False),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("recommendation", sa.String(32), nullable=False),
        sa.Column("user_choice", sa.String(32), nullable=False),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        sa.Column("approval_binding_json", sa.JSON(), nullable=False),
        sa.Column("valid", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("invalid_reason", sa.Text(), nullable=True),
    )
    op.create_table(
        "outcomes",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revision_id", sa.String(64), sa.ForeignKey("thesis_revisions.id"), nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("start_price", sa.Float(), nullable=True),
        sa.Column("end_price", sa.Float(), nullable=True),
        sa.Column("quote_source", sa.Text(), nullable=True),
        sa.Column("quote_observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result", sa.String(32), nullable=False),
        sa.Column("directional_move", sa.Float(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
    )
    op.create_index("ix_outcomes_revision", "outcomes", ["revision_id"])
    op.create_table(
        "audit_events",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("entity_type", sa.String(64), nullable=False),
        sa.Column("entity_id", sa.String(64), nullable=False),
        sa.Column("request_id", sa.String(64), nullable=True),
        sa.Column("payload_json", sa.JSON(), nullable=False),
    )
    op.create_table(
        "idempotency_keys",
        sa.Column("key", sa.String(128), primary_key=True),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("response_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("key", "action", name="uq_idem_key_action"),
    )
    # Install dialect-specific immutability triggers.
    install_immutability(op.get_bind())

def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect == "sqlite":
        for table in IMMUTABLE_TABLES:
            op.execute(sa.text(f"DROP TRIGGER IF EXISTS {table}_no_update"))
            op.execute(sa.text(f"DROP TRIGGER IF EXISTS {table}_no_delete"))
    elif dialect == "postgresql":
        for table in IMMUTABLE_TABLES:
            op.execute(sa.text(f"DROP TRIGGER IF EXISTS {table}_no_mutation ON {table}"))
        op.execute(sa.text("DROP FUNCTION IF EXISTS piia_deny_mutation()"))
    for table in [
        "idempotency_keys", "audit_events", "outcomes", "decisions", "plans",
        "risk_configs", "portfolio_snapshots", "analyses", "counterarguments", "evidence_links",
        "thesis_revisions", "theses", "evidence_items", "instruments",
    ]:
        op.drop_table(table)
