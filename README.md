# Databricks AI Stock Market Research Assistant

An educational, production-minded capstone that helps a user research public companies using grounded market data and financial news. The project will combine a Databricks-hosted frontend, Lakebase application storage, Spark ingestion and enrichment, semantic retrieval, MCP tools, and a Databricks Agent Bricks supervisor.

Phase 1 establishes the relational schema and the configuration/database boundary. Application routes, external API calls, pipelines, embeddings, retrieval, MCP tools, agent integration, and frontend behavior remain intentionally unimplemented.

## Capstone requirements

- A frontend hosted as a Databricks App
- Lakebase PostgreSQL for relational application and research data
- Massive Stocks API integration for company, price, and news facts
- A Spark data pipeline for ingestion and enrichment
- Processing of unstructured financial-news content
- Text chunking, embeddings, and semantic retrieval
- An MCP server hosted as a separate Databricks App
- A Databricks Agent Bricks / Supervisor Agent that uses the MCP tools
- MCP/agent capabilities for both reads and retrieval and real Lakebase writes
- Source metadata retained through ingestion and retrieval so answers can cite their evidence
- Secrets held in Databricks secrets or local untracked environment configuration, never in frontend code or git

## Phase 1 foundation

The idempotent schema in `sql/001_core_schema.sql` defines the ten MVP tables:

- `users`, `watchlists`, and `watchlist_tickers` for user-scoped saved symbols
- `companies` and `price_snapshots` for normalized Massive company and market data
- `news_articles`, `news_article_tickers`, and `news_article_chunks` for citation-ready news content
- `research_notes` and `analysis_reports` for durable agent and user writes

The schema uses PostgreSQL-native types, `TIMESTAMPTZ` operational timestamps, JSONB source payloads, lifecycle-aware foreign keys, and focused lookup indexes. It deliberately does not enable pgvector or add embedding columns; vector capability and model dimensions will be confirmed in the later RAG phase.

`app/config.py` reads either standard PostgreSQL environment fields (`PGHOST`, `PGDATABASE`, `PGUSER`, `PGPORT`, `PGSSLMODE`, and optional `PGPASSWORD`) or an optional `LAKEBASE_URL` for local/legacy compatibility. Only non-sensitive settings have defaults. Configuration is validated only when a database connection is requested, so offline tooling and tests can import the application without credentials.

`app/db.py` provides a small psycopg 3 connection and transaction context boundary. It does not fetch secrets, embed schema DDL, or execute business queries. For a new deployment, the preferred direction is a Databricks App PostgreSQL resource with platform-managed OAuth and rotating credentials. The exact live connection mode will be finalized after confirming whether the capstone environment uses Lakebase Autoscaling or an existing Provisioned resource; no endpoint name or OAuth flow is guessed here.

## Phase 2 market-data and service foundation

Phase 2 adds framework-independent primitives with one directional dependency flow:

```text
Massive Stocks REST API
          |
          v
     MassiveClient
          |
          v
  StockResearchService
          |
          v
    StockRepository
          |
          v
 Lakebase PostgreSQL
```

`app/massive_client.py` uses the current company-overview, daily aggregate, and ticker-news endpoints. It authenticates through a bearer header, reuses one HTTP session, follows official `next_url` pagination, applies explicit timeouts, validates ticker and date input, and converts upstream failures into safe errors. Normalized records remain separate from HTTP response objects while retaining the raw Massive payload for provenance. News sentiment is stored on each article/ticker relationship, including null-sentiment relationships for referenced tickers without an insight.

`app/repositories.py` provides parameterized, idempotent SQL operations for users, the deterministic default watchlist, companies, daily price snapshots, news articles, article/ticker relationships, and read-side price/news retrieval. It opens transactions only when a method is called and contains no HTTP logic.

`app/services.py` composes Massive reads with repository writes for company, price-history, and news refreshes. Later Flask routes and MCP tools will call this same service layer instead of duplicating API or SQL behavior.

All Phase 2 tests use mocked HTTP sessions and database connections. No live Massive request, Lakebase connection, or deployment validation is claimed.

## Proposed architecture

```text
User
  |
  v
Databricks App (frontend + application API)
  |
  +-----------------------> Lakebase PostgreSQL
  |                           - users and watchlists
  |                           - company and price snapshots
  |                           - news documents and chunks
  |                           - embeddings and source metadata
  |                           - notes and analysis reports
  |
  v
Databricks Agent Bricks / Supervisor Agent
  |
  v
MCP Server Databricks App
  |                         \
  v                          v
Domain services          Retrieval services
  |                         |
  +--> Massive Stocks API   +--> Lakebase/pgvector
  +--> Lakebase reads       +--> grounded news chunks
  +--> Lakebase writes

Scheduled Databricks workflow
  |
  v
Spark ingestion and enrichment pipeline
  +--> Massive company, price, and news data
  +--> article extraction and normalization
  +--> chunking and embedding generation
  +--> idempotent persistence to Lakebase
```

The frontend and MCP server are separate deployment units. Shared business rules will live below those boundaries so the UI and agent use the same validation, Massive integration, and Lakebase access behavior. MCP tools will remain thin and return stable, client-safe results.

## Planned implementation phases

### Phase 0 — Repository initialization (complete)

- Inspect the Day 1–3 homework repositories as read-only references.
- Establish project guidance, architecture notes, ignore rules, and empty top-level directories.
- Initialize an independent git repository.

### Phase 1 — Foundation and Lakebase model (complete)

- Define configuration boundaries and local/deployed secret resolution.
- Design idempotent Lakebase schema migrations for users, watchlists, companies, price snapshots, news, research notes, and reports.
- Add database helpers, input validation, and focused unit tests.

### Phase 2 — Massive integration and application services (current)

- Implement a resilient Massive client with explicit timeouts, pagination, rate-limit handling, and normalized responses.
- Add reusable company, historical-price, news, and watchlist services.
- Persist source timestamps and provenance with market facts.

### Phase 3 — Spark news and embedding pipeline

- Ingest financial-news documents for selected companies.
- Extract and normalize article text while preserving URL, publisher, ticker, and publication metadata.
- Chunk text, generate embeddings, and idempotently upsert documents and vectors into Lakebase/pgvector.
- Package the pipeline as a repeatable Databricks workflow.

### Phase 4 — Semantic retrieval

- Implement bounded top-k vector search with metadata filters and relevance scores.
- Return citation-ready source metadata with every retrieved chunk.
- Test deterministic parsing, validation, and retrieval contracts.

### Phase 5 — MCP server and agent integration

- Host a FastMCP server as a Databricks App.
- Expose thin tools for company data, historical performance, semantic news search, watchlist reads/writes, research notes, and analysis reports.
- Add structured, client-safe errors and tests proving tool delegation.
- Configure the Databricks Agent Bricks supervisor to compose primitive tools and stay grounded in tool results.

### Phase 6 — Frontend and end-to-end validation

- Build the Databricks App research interface without exposing credentials.
- Connect user identity, watchlists, research conversations, citations, notes, and saved reports.
- Exercise read, retrieval, and write paths end to end and capture capstone evidence.

### Phase 7 — Hardening and presentation

- Add observability, deployment documentation, failure-path checks, and rubric-focused tests.
- Review privacy, secret handling, SQL safety, ingestion idempotency, and citation quality.
- Prepare demonstration evidence and identify optional stretch goals separately from the MVP.

## Repository layout

```text
app/          Databricks frontend app and application boundary
mcp_server/   FastMCP Databricks App and thin tool boundary
pipelines/    Spark ingestion, extraction, chunking, and embedding workflows
sql/          Idempotent Lakebase schema and migration scripts
tests/        Focused deterministic and boundary tests
evidence/     Screenshots and other capstone proof artifacts
```

## Reference repositories

The sibling Day 1, Day 2, and Day 3 homework repositories are read-only references. This project will reuse proven patterns selectively, but it will not copy support-ticket, weather, paper-trading, or other legacy application logic.
