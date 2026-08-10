# Project

Databricks AI Stock Market Research Assistant capstone.

# Goal

Build a concise but production-minded educational capstone demonstrating Databricks Apps, Lakebase, Spark, unstructured-data RAG, Massive Stocks API integration, MCP tools, and Databricks Agent Bricks.

# Reference repositories

Treat the three sibling homework repositories as read-only reference material:

- Day 1: Lakebase + Databricks App CRUD patterns
- Day 2: Massive API, watchlist, stock news, Spark embeddings, pgvector/RAG patterns
- Day 3: FastMCP, adapter/service separation, Agent Bricks integration, structured error handling and tests

Never modify the reference repositories.

# Engineering principles

- Prefer the simplest implementation that clearly satisfies the capstone rubric.
- Do not overengineer.
- Separate external API logic, database logic, agent/MCP boundaries, and frontend concerns.
- Keep MCP tools thin.
- Put reusable business/API/database logic below the MCP layer.
- Never let an LLM invent current stock prices, company data, or news.
- Market facts must come from Massive or persisted Lakebase data.
- Semantic-news answers must come from retrieved source documents.
- Preserve source metadata so the agent can cite where information came from.
- Parameterize SQL values; do not interpolate untrusted user input into SQL identifiers or clauses.
- Validate stock symbols and user input.
- Return client-safe structured errors from MCP boundaries.
- Never expose credentials to the frontend.
- Do not commit `.env`, API keys, Lakebase URLs, passwords, tokens, or Databricks secrets.
- Prefer Databricks App PostgreSQL resources with platform-managed OAuth and rotating credentials for new Lakebase deployments.
- Treat `LAKEBASE_URL` as optional local or legacy compatibility, not the default production authentication design.
- Do not invent Lakebase endpoint names or authentication settings before the target environment is confirmed.
- Prefer idempotent schema/setup and ingestion where practical.
- Add focused tests for important deterministic logic and MCP tool delegation.
- Do not silently change architecture without explaining why.

# Scope

MVP Lakebase entities should eventually include:

- users
- watchlists
- watchlist_tickers
- companies
- price_snapshots
- news_articles
- news_article_chunks / embeddings
- research_notes
- analysis_reports

MVP agent capabilities should eventually include:

- retrieve company information
- retrieve historical stock performance
- semantic search over financial news
- inspect a user's watchlist
- add a ticker to a watchlist
- remove a ticker from a watchlist
- save a research note
- save an analysis report

The agent may compare companies by composing primitive tools rather than requiring a dedicated compare tool.

# Non-goals for MVP

Do not add:

- brokerage/trading execution
- portfolio optimization
- automatic investment recommendations
- real-time trading infrastructure
- multi-agent orchestration unless required later
- complex technical-analysis systems
- SEC filing ingestion unless added as a later stretch goal
- custom authentication when Databricks identity is sufficient

# Working style

- Work in small phases.
- Before changing code, inspect relevant existing files.
- Do not implement future phases unless explicitly requested.
- After each phase, report:
  1. files created/changed
  2. major design decisions
  3. commands/tests run
  4. test results
  5. unresolved issues
  6. recommended next step
- Never commit or push unless explicitly told to do so.

# Validation environment

- Never assume `.venv` is portable across Codex execution environments; prefer it only when executable.
- If `.venv` is unusable, discover an available Python interpreter, create/use `.codex-venv`, and install project requirements there when needed.
- Do not skip the full test suite solely because `.venv` is unusable when another interpreter is available.
- Never alter or delete the user's `.venv` as part of the Codex fallback.
