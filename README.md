# Databricks AI Stock Market Research Assistant

An educational, production-minded capstone that helps a user research public companies using grounded market data and financial news. The project will combine a Databricks-hosted frontend, Lakebase application storage, Spark ingestion and enrichment, semantic retrieval, MCP tools, and a Databricks Agent Bricks supervisor.

Phases 1 and 2 establish the relational schema, configuration/database boundary, and Massive-backed service layer. Phases 3B and 3C add Lakebase Autoscaling OAuth connectivity and validate the live stock-data ingestion path. Application routes, pipelines, embeddings, retrieval, MCP tools, agent integration, and frontend behavior remain intentionally unimplemented.

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

`app/config.py` reads either standard PostgreSQL environment fields (`PGHOST`, `PGDATABASE`, `PGUSER`, `PGPORT`, and `PGSSLMODE`) plus `ENDPOINT_NAME` for Databricks OAuth, or an optional `LAKEBASE_URL` for isolated local/legacy compatibility. OAuth mode does not use `PGPASSWORD`. Only non-sensitive settings have defaults. Configuration is validated only when a database connection is requested, so offline tooling and tests can import the application without credentials.

`app/db.py` provides a small psycopg 3 connection and transaction context boundary. It does not embed schema DDL or execute business queries. Lakebase Autoscaling OAuth credential generation is described below.

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

## Phase 3B Lakebase Autoscaling connectivity

Each new OAuth database connection follows the Databricks-recommended flow:

```text
Local developer                         Future deployed Databricks App
Databricks CLI OAuth                    App service principal
          |                                      |
          +---------------+----------------------+
                          v
                  WorkspaceClient
                          |
                          v
          generate_database_credential()
                          |
                          v
       short-lived Lakebase OAuth password
                          |
                          v
                       psycopg
```

The database credential is generated immediately before a new connection and is held only long enough to pass it to psycopg. No permanent database password is required or stored. `LAKEBASE_URL`, when deliberately configured, takes precedence solely as a legacy/local compatibility path and does not invoke Databricks credential generation.

The confirmed local Lakebase Autoscaling environment can be selected in PowerShell without setting `PGPASSWORD`:

```powershell
$env:PGHOST="ep-proud-waterfall-d8765xn2.database.us-east-2.cloud.databricks.com"
$env:PGDATABASE="databricks_postgres"
$env:PGUSER="sharifip1234@gmail.com"
$env:PGPORT="5432"
$env:PGSSLMODE="require"
$env:ENDPOINT_NAME="projects/stock-research-capstone/branches/production/endpoints/primary"
```

The local Databricks CLI profile must already be authenticated to the target workspace. From the repository root, run the scripts in order:

```powershell
.\.venv\Scripts\python.exe scripts\check_lakebase.py
.\.venv\Scripts\python.exe scripts\apply_schema.py
.\.venv\Scripts\python.exe scripts\verify_schema.py
```

`check_lakebase.py` prints only the connected user and database. `apply_schema.py` executes the canonical idempotent `sql/001_core_schema.sql` in the normal transaction boundary and never seeds data. `verify_schema.py` succeeds only when the public schema contains exactly the ten expected MVP tables. These commands are manual live operations and are not part of the offline unit-test suite.

## Phase 3C live stock-data validation

The existing application path was successfully validated end to end against the real Massive Stocks API and Lakebase Autoscaling:

```text
Massive Stocks API
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
Lakebase Autoscaling
```

The bounded AAPL validation confirmed company metadata, nine historical daily aggregate bars, and five recent financial-news articles. Real multi-ticker article relationships remained normalized, and AAPL-specific sentiment was returned from `news_article_tickers`.

After a rate-limit-aware second refresh of the same range, the counts remained stable at one company row, nine price rows, five distinct associated articles, and five AAPL article associations. The company timestamp advanced, confirming that existing records were updated instead of duplicated.

The live validation scripts are:

- `scripts/refresh_stock_data.py` for the bounded Massive-to-Lakebase refresh
- `scripts/verify_stock_data.py` for database-only counts and relational checks

## Phase 4A Spark article extraction and chunking

Phase 4A begins with persisted news rather than calling Massive from Spark:

```text
news_articles
      |
      v
Spark JDBC read
      |
      v
LEFT ANTI JOIN against processed article IDs
      |
      v
distributed webpage extraction (mapInPandas)
      |
      v
trafilatura main-content extraction
      |
      +---- blocked/unavailable/short page
      |                 |
      |                 v
      |        persisted title/description fallback
      |                 |
      +-----------------+
               |
               v
deterministic overlapping chunks
               |
               v
news_article_chunks
```

Publisher sites can legitimately block or restrict automated retrieval. When a body cannot be accessed or extracted, the already-ingested Massive title and description remain grounded unstructured source text, so the bounded Spark job uses them rather than fabricating content or failing the whole batch. Articles with no usable body or metadata are skipped.

The Databricks notebook source is `pipelines/process_news_content.py`. Spark performs bounded distributed extraction and chunk generation, then the small final chunk set is collected to the driver and persisted per article through an atomic delete-and-replace repository operation. This keeps stale chunks from earlier versions from surviving without introducing a complex distributed database writer.

Embeddings, vector storage, semantic search, and RAG are intentionally deferred to Phase 4B. The Phase 4A notebook has not yet been live-validated in a Databricks Spark runtime.

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

### Phase 2 — Massive integration and application services (complete)

- Implement a resilient Massive client with explicit timeouts, pagination, rate-limit handling, and normalized responses.
- Add reusable company, historical-price, news, and watchlist services.
- Persist source timestamps and provenance with market facts.

### Phase 3B — Lakebase Autoscaling connectivity (complete)

- Generate short-lived Lakebase credentials through the Databricks SDK for each new OAuth connection.
- Provide safe connection, idempotent schema-application, and exact schema-verification scripts.
- Keep all ordinary unit tests offline through injected clients and connections.

### Phase 3C — Live stock-data ingestion validation (complete)

- Validate company, daily-bar, and recent-news ingestion against Massive and Lakebase Autoscaling.
- Confirm normalized multi-ticker news relationships and ticker-specific sentiment.
- Confirm idempotent repeated refreshes without duplicate company, price, article, or association rows.

### Phase 4A — Spark article extraction and chunking (offline implementation)

- Read bounded, unprocessed news from Lakebase through Spark JDBC and an anti join.
- Extract article bodies through partition-scoped HTTP sessions and trafilatura, with metadata fallback.
- Generate deterministic overlapping chunks and atomically replace each article's persisted chunks.

### Phase 4B — Embeddings and semantic retrieval (planned)

- Select and validate the embedding model and storage extension.
- Add vector persistence and bounded citation-ready semantic retrieval.

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
