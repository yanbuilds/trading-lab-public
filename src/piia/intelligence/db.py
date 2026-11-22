from __future__ import annotations
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from piia.intelligence.models import Base
from piia.intelligence.paths import default_database_url, DATA

IMMUTABLE_TABLES = (
    "evidence_items",
    "thesis_revisions",
    "outcomes",
    "risk_configs",
    "audit_events",
    "portfolio_snapshots",
    "research_runs",
    "kill_switch_events",
    "paper_ledger",
)

def sqlite_trigger_statements(present: set[str] | None = None) -> list[str]:
    stmts: list[str] = []
    for table in IMMUTABLE_TABLES:
        if present is not None and table not in present:
            continue
        stmts.append(
            f"CREATE TRIGGER IF NOT EXISTS {table}_no_update BEFORE UPDATE ON {table} "
            f"BEGIN SELECT RAISE(ABORT, '{table} are immutable'); END"
        )
        stmts.append(
            f"CREATE TRIGGER IF NOT EXISTS {table}_no_delete BEFORE DELETE ON {table} "
            f"BEGIN SELECT RAISE(ABORT, '{table} are immutable'); END"
        )
    return stmts

PG_FUNCTION = (
    "CREATE OR REPLACE FUNCTION piia_deny_mutation() RETURNS trigger AS $$\n"
    "BEGIN\n"
    "  RAISE EXCEPTION '% are immutable', TG_TABLE_NAME;\n"
    "END;\n"
    "$$ LANGUAGE plpgsql"
)

def pg_trigger_statements(present: set[str] | None = None) -> list[str]:
    stmts = [PG_FUNCTION]
    for table in IMMUTABLE_TABLES:
        if present is not None and table not in present:
            continue
        stmts.append(f"DROP TRIGGER IF EXISTS {table}_no_mutation ON {table}")
        stmts.append(
            f"CREATE TRIGGER {table}_no_mutation "
            f"BEFORE UPDATE OR DELETE ON {table} "
            f"FOR EACH ROW EXECUTE FUNCTION piia_deny_mutation()"
        )
    return stmts


def make_engine(url: str | None = None) -> Engine:
    url = url or default_database_url()
    if url.startswith("sqlite:///"):
        DATA.mkdir(parents=True, exist_ok=True)
        connect_args = {"check_same_thread": False}
        engine = create_engine(url, future=True, connect_args=connect_args)
        @event.listens_for(engine, "connect")
        def _fk(dbapi_conn, _rec):  # pragma: no cover
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
    else:
        engine = create_engine(url, future=True)
    return engine


def _existing_tables(bind) -> set[str]:
    insp = __import__("sqlalchemy", fromlist=["inspect"]).inspect(bind)
    return set(insp.get_table_names())

def install_immutability(bind) -> None:
    dialect = bind.dialect.name
    present = _existing_tables(bind)
    if dialect == "sqlite":
        stmts = sqlite_trigger_statements(present)
    elif dialect == "postgresql":
        stmts = pg_trigger_statements(present)
    else:
        raise RuntimeError(f"unsupported dialect {dialect}")
    from sqlalchemy.engine import Engine
    if isinstance(bind, Engine):
        with bind.begin() as conn:
            for stmt in stmts:
                conn.execute(text(stmt))
    else:
        for stmt in stmts:
            bind.execute(text(stmt))


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def create_all(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    install_immutability(engine)
