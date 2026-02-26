# Migration Strategy

## Current State
Tables are created at startup via `create_all_tables()` in `db/session.py`.
This is intentional for early scaffolding — no migration tool yet.

## Planned: Alembic Integration (before any schema change in production)

### Setup steps (when ready)
```bash
cd backend
poetry add alembic
poetry run alembic init app/db/migrations
```

### Key changes needed
1. In `alembic.ini` — point `sqlalchemy.url` to an env var, NOT a hardcoded string.
2. In `app/db/migrations/env.py`:
   - Import `Base` from `app.db.base`
   - Import all models so Alembic detects them
   - Use async engine config (`run_async_migrations`)
3. Generate first migration from current models:
```bash
   poetry run alembic revision --autogenerate -m "initial schema"
   poetry run alembic upgrade head
```
4. Remove `create_all_tables()` from startup once Alembic owns the schema.

## Rules going forward
- Never edit a committed migration; always create a new revision.
- Every schema change = new Alembic revision.
- Run `alembic upgrade head` in CI before any DB-touching tests.