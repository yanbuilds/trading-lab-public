"""Research runs, kill switch, and local paper ledger.

Revision ID: 0002_paper
Revises: 0001_phase1
PostgreSQL is schema-compatible but not tested.
"""
from __future__ import annotations
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from piia.intelligence.db import install_immutability

revision: str = "0002_paper"
down_revision: Union[str, Sequence[str], None] = "0001_phase1"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        "research_runs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("thesis_id", sa.String(64), sa.ForeignKey("theses.id"), nullable=False),
        sa.Column("revision_id", sa.String(64), sa.ForeignKey("thesis_revisions.id"), nullable=False),
        sa.Column("mode", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("independent_llm_review", sa.String(16), nullable=False),
        sa.Column("prompt_hash", sa.String(64), nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("model_name", sa.String(128), nullable=True),
        sa.Column("model_version", sa.String(64), nullable=True),
        sa.Column("token_cost", sa.Float(), nullable=True),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("result_json", sa.JSON(), nullable=False),
        sa.Column("actor", sa.String(128), nullable=False),
    )
    op.create_table(
        "kill_switch_events",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False, server_default=""),
    )
    op.create_table(
        "paper_accounts",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("label", sa.String(64), nullable=False),
        sa.Column("starting_cash", sa.Float(), nullable=False),
        sa.Column("cash", sa.Float(), nullable=False),
        sa.Column("liabilities", sa.Float(), nullable=False, server_default="0"),
        sa.Column("realized_pnl", sa.Float(), nullable=False, server_default="0"),
        sa.Column("engine", sa.String(32), nullable=False, server_default="local_simplified"),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
    )
    op.create_table(
        "paper_ledger",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("account_id", sa.String(64), sa.ForeignKey("paper_accounts.id"), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("cash_after", sa.Float(), nullable=False),
        sa.Column("note", sa.Text(), nullable=False, server_default=""),
        sa.Column("ref_id", sa.String(64), nullable=True),
        sa.Column("provenance", sa.String(64), nullable=False, server_default="assumed"),
    )
    op.create_table(
        "paper_orders",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("account_id", sa.String(64), sa.ForeignKey("paper_accounts.id"), nullable=False),
        sa.Column("plan_id", sa.String(64), nullable=True),
        sa.Column("decision_id", sa.String(64), nullable=True),
        sa.Column("research_run_id", sa.String(64), nullable=True),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("side", sa.String(16), nullable=False),
        sa.Column("qty", sa.Float(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("leverage", sa.Float(), nullable=False),
        sa.Column("fee", sa.Float(), nullable=False),
        sa.Column("stop_loss", sa.Float(), nullable=True),
        sa.Column("reduce_only", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("sandbox", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.create_table(
        "paper_positions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("account_id", sa.String(64), sa.ForeignKey("paper_accounts.id"), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("instrument_id", sa.String(64), nullable=False),
        sa.Column("side", sa.String(16), nullable=False),
        sa.Column("qty", sa.Float(), nullable=False),
        sa.Column("entry", sa.Float(), nullable=False),
        sa.Column("stop_loss", sa.Float(), nullable=True),
        sa.Column("leverage", sa.Float(), nullable=False),
        sa.Column("margin", sa.Float(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
        sa.Column("entry_order_id", sa.String(64), nullable=False),
        sa.Column("exit_order_id", sa.String(64), nullable=True),
        sa.Column("realized_pnl", sa.Float(), nullable=True),
    )
    install_immutability(op.get_bind())

def downgrade() -> None:
    for table in ["paper_positions", "paper_orders", "paper_ledger", "paper_accounts", "kill_switch_events", "research_runs"]:
        op.drop_table(table)
