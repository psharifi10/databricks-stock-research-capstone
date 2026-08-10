# Databricks AI Stock Market Research Assistant

An educational, production-minded capstone that helps a user research public companies using grounded market data and financial news. The project combines a Databricks-hosted web interface, Lakebase application storage, Spark ingestion and enrichment, semantic retrieval, and read/write MCP tools that were validated with an AI client.

The completed path covers Massive ingestion, Lakebase Autoscaling OAuth connectivity, Serverless Spark article processing, MiniLM embeddings, vector retrieval, grounded research-context assembly, a FastMCP application boundary, and a compact research dashboard. The application does not generate investment advice and does not run an LLM internally.

## Capstone requirements

- A frontend hosted as a Databricks App
- Lakebase PostgreSQL for relational application and research data
- Massive Stocks API integration for company, price, and news facts
- A Spark data pipeline for ingestion and enrichment
- Processing of unstructured financial-news content
- Text chunking, embeddings, and semantic retrieval
- A FastMCP server and web frontend hosted together in one Databricks App
- Agent-ready MCP tools validated through Databricks AI Playground
- MCP/agent capabilities for both reads and retrieval and real Lakebase writes
- Source metadata retained through ingestion and retrieval so answers can cite their evidence
- Secrets held in Databricks secrets or local untracked environment configuration, never in frontend code or git

## Phase 1 foundation

The idempotent schema in `sql/001_core_schema.sql` defines the ten MVP tables:

- `users`, `watchlists`, and `watchlist_tickers` for user-scoped saved symbols
- `companies` and `price_snapshots` for normalized Massive company and market data
- `news_articles`, `news_article_tickers`, and `news_article_chunks` for citation-ready news content
- `research_notes` and `analysis_reports` for durable agent and user writes

The core schema uses PostgreSQL-native types, `TIMESTAMPTZ` operational timestamps, JSONB source payloads, lifecycle-aware foreign keys, and focused lookup indexes. Phase 4B adds its vector columns and index through the separate idempotent `sql/002_chunk_embeddings.sql` migration without changing the ten logical entities.

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

`app/services.py` composes Massive reads with repository writes for company, price-history, and news refreshes. The MCP tools and web research route reuse the service layer instead of duplicating API or SQL behavior.

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
$env:PGHOST="<lakebase-host>"
$env:PGDATABASE="<database-name>"
$env:PGUSER="<database-role>"
$env:PGPORT="5432"
$env:PGSSLMODE="require"
$env:ENDPOINT_NAME="<endpoint-resource-name>"
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
Spark PostgreSQL/JDBC read
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
bundled PostgreSQL Spark writer
               |
               v
news_article_chunks
```

Publisher sites can legitimately block or restrict automated retrieval. When a body cannot be accessed or extracted, the already-ingested Massive title and description remain grounded unstructured source text, so the bounded Spark job uses them rather than fabricating content or failing the whole batch. Articles with no usable body or metadata are skipped.

The Databricks notebook source is `pipelines/process_news_content.py`. On Databricks Serverless, Spark performs bounded distributed extraction and chunk generation, then writes the materialized chunk DataFrame through the bundled PostgreSQL Spark data source into `news_article_chunks`. Append mode is safe in this pipeline because the left anti join selects only article IDs with no existing chunks; already-processed articles are excluded before extraction.

The local application continues to use psycopg and repository-level replacement semantics through `app/db.py` and `app/repositories.py`. The Databricks Spark pipeline uses Spark PostgreSQL/JDBC reads and the bundled PostgreSQL writer, avoiding native psycopg and direct-JVM dependencies in the Databricks Python process.

The Phase 4A Spark PostgreSQL read/write path was successfully live-validated on Databricks Serverless. The first run extracted all five eligible article bodies and persisted 20 chunks. A second run produced zero eligible articles, confirming that the left anti join prevents already-processed articles from being written again.

## Phase 4B embeddings and semantic retrieval

Phase 4B keeps each chunk and its embedding in the same Lakebase row:

```text
news_article_chunks where embedding is null
                    |
                    v
bounded Spark PostgreSQL/JDBC read
                    |
                    v
distributed mapInPandas embedding
sentence-transformers/all-MiniLM-L6-v2
                    |
                    v
normalized 384-dimensional vectors
                    |
                    v
OAuth Lakebase Data API RPC update
                    |
                    v
VECTOR(384) + lakebase_ann cosine index
                    |
                    v
parameterized semantic retrieval with <=>
```

`sql/002_chunk_embeddings.sql` enables `lakebase_vector`, adds nullable `embedding VECTOR(384)` and `embedding_model` columns to `news_article_chunks`, and creates the `news_article_chunks_embedding_ann` index with `vector_cosine_ops`. The `lakebase_vector` extension is currently Beta. The query path uses cosine distance (`<=>`) and returns both distance and `1 - distance` similarity.

The Databricks notebook source is `pipelines/embed_news_chunks.py`. It loads one `sentence-transformers/all-MiniLM-L6-v2` model per Spark partition, embeds bounded pandas batches, validates every vector dimension, and never prints vectors. Existing chunk rows are updated through a parameterized Lakebase Data API RPC; no duplicate chunks or table overwrite is used. The Data API must be enabled, its schema cache refreshed after applying the migration, and the notebook identity must have the documented table/function permissions. The non-secret Data API base URL is supplied through a widget.

Application-side semantic retrieval remains independent of Spark. `QueryEmbeddingService` lazily loads the same model, `SemanticNewsSearchService` delegates the validated vector to `StockRepository.search_news_chunks`, and `scripts/search_news.py` prints bounded ranked chunk previews. For example:

```powershell
.\.venv\Scripts\python.exe scripts\search_news.py "Apple CEO succession" --ticker AAPL --top-k 5
```

The Phase 4B migration, embedding path, and cosine retrieval were validated with Lakebase, and semantic news retrieval was subsequently exercised through the deployed MCP server. LLM answer generation remains outside the application; the returned context is RAG-ready for an external AI client.

## Phase 5A grounded research context

Phase 5A assembles deterministic, JSON-serializable evidence without calling an LLM:

```text
research question
       |
       v
query embedding
       |
       v
Lakebase cosine retrieval
       |
       v
article diversification (at most two chunks per article)
       |
       v
structured Lakebase company, price, news, and semantic evidence
       |
       v
bounded research context
```

`ResearchContextService` combines persisted company metadata, a bounded recent price window, ticker-specific recent news, and citation-ready semantic chunks. It preserves retrieval order, source URLs, article and chunk identifiers, publication metadata, sentiment, and similarity while excluding vectors from the returned context. `scripts/build_research_context.py` provides a concise local preview of counts and evidence titles. The service now supports both MCP retrieval and the web research endpoint without embedding LLM generation in the application.

## Phase 5B read-only MCP server

Phase 5B exposes the existing grounded reads through one thin custom MCP boundary:

```text
Databricks Agent (later)
        |
        v
Custom MCP server
        |
        v
thin MCP tools
        |
        v
service layer
        |
        v
Lakebase relational/vector data
```

`mcp_server/stock_research_mcp.py` runs FastMCP over Streamable HTTP at `/mcp` and provides five tools: `get_company`, `get_price_history`, `search_financial_news`, `build_research_context`, and `health`. All tool results use consistent structured success/error envelopes, and service failures are sanitized before crossing the MCP boundary.

The Phase 5B tools are read-only. Write tools follow in a separate phase, and no LLM runs inside the MCP server.

## Phase 6A safe Lakebase write actions

Phase 6A adds four explicit-action mutations through the existing service and repository boundaries:

```text
Agent
  |
  v
MCP write tool
  |
  v
ResearchActionService
  |
  v
StockRepository
  |
  v
Lakebase
```

The write tools are `add_to_watchlist`, `remove_from_watchlist`, `save_research_note`, and `save_analysis_report`. Each accepts an explicit user email for Phase 6A, validates and normalizes its inputs, and returns bounded structured metadata through the standard MCP result envelope.

Mutations require explicit user intent. The MCP server never adds a ticker or saves a note or report as a side effect of retrieval or generated research.

## Phase 8 Databricks App web interface

The existing `mcp-stock-research` Databricks App serves the dashboard, deterministic research, Supervisor synthesis, and MCP protocol from the same stateless FastMCP process:

- `/mcp` preserves all nine read/write MCP tools.
- `/` serves the stock research dashboard.
- `/api/research` validates a ticker and question, then delegates directly to `ResearchContextService`.
- `/api/agent` sends a grounded prompt to the configured Supervisor Agent model-serving endpoint.

The dashboard presents the Supervisor's synthesized answer separately above persisted company metadata, the latest close and bounded recent prices, ticker-linked recent news, and citation-ready semantic evidence. If Supervisor synthesis is unavailable, deterministic grounded research still renders. Browser rendering uses DOM `textContent`; external sources are limited to HTTP(S) links with `target="_blank"` and `rel="noopener noreferrer"`. The frontend receives no vectors, credentials, raw source payloads, or OAuth configuration. The Supervisor dashboard integration is implemented but is not claimed as live-validated until the updated app is deployed and tested.

## Final architecture

```text
Massive Stocks API
        |
        v
service / ingestion layer
        |
        v
Lakebase relational data
        |
        v
Spark unstructured article pipeline
        |
        v
MiniLM embeddings
        |
        v
Lakebase vector retrieval
        |
        v
FastMCP stock research server
        |
        v
Databricks Supervisor Agent
        |
        v
Databricks App dashboard
```

Massive supplies third-party company, price, and news facts. Spark processes unstructured article bodies into deterministic chunks, MiniLM creates embeddings, and Lakebase stores both relational application data and vector-searchable evidence. The FastMCP server exposes nine safe read/write tools over the existing service boundaries. The Supervisor chooses those tools when evidence or an explicit action is needed, while `ResearchContextService` continues to provide separately visible deterministic grounded data.

## AI client and agent status

A Databricks Supervisor Agent named `Stock Research Assistant` was created, and `mcp-stock-research` was registered as its custom MCP tool. Live validation confirmed that the Supervisor selected `build_research_context` for an Apple research question and selected `add_to_watchlist` for an explicit watchlist action. It correctly respected the idempotent no-op result when AAPL was already present. The Databricks App dashboard now integrates with the Supervisor endpoint in source for synthesized responses, while grounded structured research remains separately visible through `ResearchContextService`. This dashboard-to-Supervisor integration still requires deployment and live validation.

## Live MCP validation

The following deployed behaviors were validated without recording personal email addresses:

- `health` returned a successful service response.
- `get_company(AAPL)` returned persisted company metadata.
- `search_financial_news(AAPL, ...)` returned grounded semantic evidence.
- `add_to_watchlist` completed successfully.
- `save_research_note` completed successfully and returned `note_id` 1.
- `remove_from_watchlist` was callable and executed correctly; no claim is made that a row was removed when the result was `removed=false`.
- The Supervisor selected `build_research_context` for a live Apple research question.
- The Supervisor selected `add_to_watchlist` for an explicit action and preserved the already-present no-op result.

## Validation Evidence

- [Successful Databricks App research dashboard query](evidence/dashboard_query.png)

## Rubric alignment

- [x] Spark pipeline — bounded Serverless extraction, chunking, and embedding workflows
- [x] Third-party Massive API — company, daily-price, and news ingestion
- [x] Unstructured data processing — article extraction with grounded metadata fallback
- [x] Lakebase relational tables — ten normalized application/research entities
- [x] Embeddings, vector retrieval, and RAG context — MiniLM, cosine search, and `ResearchContextService`
- [x] AI read/write tool use — Supervisor-selected MCP retrieval and explicit write actions
- [x] Databricks App frontend — deterministic research validated live; Supervisor summary integration implemented with redeployment and live validation pending

## Submission hygiene

The repository ignores local virtual environments, `.env` files, Python and test caches, model caches, IDE metadata, generated datasets, build output, credentials, and archives. `.env.example` remains tracked with placeholders only. Submission artifacts must not include ignored local environments or caches.

## Implementation phases

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

### Phase 4A — Spark article extraction and chunking (complete)

- Read bounded, unprocessed news from Lakebase through Spark JDBC and an anti join.
- Extract article bodies through partition-scoped HTTP sessions and trafilatura, with metadata fallback.
- Generate deterministic overlapping chunks and append only chunks whose article IDs passed the unprocessed-article anti join.

### Phase 4B — Embeddings and semantic retrieval (complete)

- Embed bounded unprocessed chunks with `all-MiniLM-L6-v2` through Spark.
- Persist 384-dimensional vectors through a Serverless-safe Lakebase Data API update.
- Retrieve bounded citation-ready chunks through the Lakebase cosine ANN index.

### Phase 5 — MCP server and Supervisor Agent integration (complete)

- Host a FastMCP server as a Databricks App.
- Expose thin tools for company data, historical performance, semantic news search, watchlist reads/writes, research notes, and analysis reports.
- Add structured, client-safe errors and tests proving tool delegation.
- Register the MCP server as the Supervisor Agent's custom tool.
- Validate Supervisor-selected grounded retrieval and explicit write-tool execution.

### Phase 8 — Frontend implementation (complete; redeployment pending)

- Serve a compact research interface from the existing FastMCP Databricks App.
- Reuse `ResearchContextService` for company, price, news, and semantic evidence.
- Display a Supervisor-generated synthesis separately without replacing deterministic research.
- Preserve safe browser rendering, MCP behavior, and credential isolation.

### Phase 9 — Submission documentation and hygiene (complete)

- Record final architecture, live validation status, and rubric alignment.
- Review privacy, secret handling, ignored local artifacts, and safe failure paths.
- Keep final deployment and submission packaging as explicit operator steps.

## Repository layout

```text
app/          Databricks frontend app and application boundary
mcp_server/   FastMCP tools, web API route, and static dashboard assets
pipelines/    Spark ingestion, extraction, chunking, and embedding workflows
sql/          Idempotent Lakebase schema and migration scripts
tests/        Focused deterministic and boundary tests
evidence/     Screenshots and other capstone proof artifacts
```

## Reference repositories

The sibling Day 1, Day 2, and Day 3 homework repositories are read-only references. This project will reuse proven patterns selectively, but it will not copy support-ticket, weather, paper-trading, or other legacy application logic.
