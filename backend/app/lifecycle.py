"""Application startup and shutdown resource management."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from sqlalchemy.engine import make_url
import structlog

from app.config import get_settings
from app.db.database import SessionLocal, init_db, settings as database_settings
from app.services.auth import get_auth_service
from app.services.bootstrap import ensure_demo_account
from app.services.ingestion_jobs import IngestionJobWorker

logger = structlog.get_logger()


def _database_url_diagnostic(database_url: str) -> dict[str, object]:
    """Return only safe connection metadata for a startup diagnostic.

    Args:
        database_url: Configured SQLAlchemy database URL.

    Returns:
        Connection fields that cannot expose the password itself.
    """
    parsed_url = make_url(database_url)
    password = parsed_url.password
    return {
        "scheme": parsed_url.get_backend_name(),
        "host": parsed_url.host,
        "port": parsed_url.port,
        "user": parsed_url.username,
        "db": parsed_url.database,
        "password_present": password is not None,
        "password_length": len(password) if password is not None else 0,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and shut down resources owned by the application.

    Args:
        app: FastAPI application entering or leaving its lifespan.

    Returns:
        An async context manager that yields while the app is serving.
    """
    logger.info("STARTUP CHECKPOINT: lifespan entered")
    logger.info("Starting OpenNotebookLM", version="0.1.0")

    # Auth dependencies are otherwise built on the first protected request.
    # Resolve them now so a production instance cannot advertise readiness
    # while every authenticated request is guaranteed to fail.
    logger.info("STARTUP CHECKPOINT: before auth service")
    get_auth_service()
    logger.info("STARTUP CHECKPOINT: auth service ready")

    logger.info("STARTUP CHECKPOINT: before init_db")
    logger.info(
        "DATABASE_URL diagnostic",
        **_database_url_diagnostic(database_settings.database_url),
    )
    init_db()
    logger.info("STARTUP CHECKPOINT: init_db complete")
    logger.info("Database initialized")
    logger.info("STARTUP CHECKPOINT: before demo account")

    settings = get_settings()

    # A fresh database would otherwise be a locked door: public registration is
    # closed by default, so without this the only way in is a shell command.
    with SessionLocal() as db:
        demo_account = ensure_demo_account(
            db,
            get_auth_service(),
            enabled=settings.seed_demo_user,
            username=settings.demo_username,
            email=settings.demo_email,
            password=settings.demo_password,
        )
    if demo_account is not None:
        # Warning, not info: this deployment publishes working credentials on
        # its sign-in page, and that belongs somewhere an operator will see it.
        logger.warning(
            "Demo account is enabled and its credentials are published on the "
            "sign-in page. Set SEED_DEMO_USER=false for any deployment others "
            "can reach.",
            username=demo_account.username,
        )

    Path("./data").mkdir(exist_ok=True)
    Path("./models").mkdir(exist_ok=True)
    Path("./uploads").mkdir(exist_ok=True)

    ingestion_worker = IngestionJobWorker(
        session_factory=SessionLocal,
        processor=getattr(app.state, "ingestion_job_processor", None),
        concurrency=settings.ingestion_worker_concurrency,
    )
    await ingestion_worker.start()
    app.state.ingestion_worker = ingestion_worker
    logger.info(
        "Durable ingestion worker started",
        concurrency=settings.ingestion_worker_concurrency,
    )

    logger.info("STARTUP CHECKPOINT: before yield")
    try:
        yield
    finally:
        await ingestion_worker.stop()
        logger.info("Shutting down OpenNotebookLM")
