-- Databricks AI Stock Market Research Assistant
-- Phase 1 Lakebase/PostgreSQL schema.
--
-- This file is safe to run repeatedly. It intentionally contains no sample
-- data, extensions, vector columns, or environment-specific configuration.

CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS watchlists (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_watchlists_user_name UNIQUE (user_id, name)
);

-- The unique constraint above also provides a user_id-leading index for
-- user-scoped watchlist lookups.

CREATE TABLE IF NOT EXISTS watchlist_tickers (
    watchlist_id BIGINT NOT NULL REFERENCES watchlists (id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    added_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_watchlist_tickers PRIMARY KEY (watchlist_id, ticker),
    CONSTRAINT ck_watchlist_tickers_uppercase CHECK (ticker = UPPER(ticker))
);

CREATE TABLE IF NOT EXISTS companies (
    ticker TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    legal_name TEXT,
    description TEXT,
    market_cap NUMERIC(24, 4),
    market TEXT,
    exchange TEXT,
    security_type TEXT,
    active BOOLEAN,
    list_date DATE,
    sic_code TEXT,
    sic_description TEXT,
    industry TEXT,
    homepage_url TEXT,
    currency_name TEXT,
    locale TEXT,
    raw_source_payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_companies_ticker_uppercase CHECK (ticker = UPPER(ticker))
);

CREATE TABLE IF NOT EXISTS price_snapshots (
    ticker TEXT NOT NULL,
    price_date DATE NOT NULL,
    open NUMERIC(20, 6),
    high NUMERIC(20, 6),
    low NUMERIC(20, 6),
    close NUMERIC(20, 6),
    volume NUMERIC(24, 4),
    vwap NUMERIC(20, 6),
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_price_snapshots PRIMARY KEY (ticker, price_date),
    CONSTRAINT ck_price_snapshots_ticker_uppercase CHECK (ticker = UPPER(ticker))
);

-- The primary key above is the ticker/date lookup index and prevents duplicate
-- daily snapshots without maintaining a redundant secondary index.

CREATE TABLE IF NOT EXISTS news_articles (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    author TEXT,
    publisher TEXT,
    article_url TEXT,
    published_at TIMESTAMPTZ,
    keywords TEXT[],
    raw_source_payload JSONB NOT NULL,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_news_articles_published_at
    ON news_articles (published_at DESC);

CREATE TABLE IF NOT EXISTS news_article_tickers (
    article_id TEXT NOT NULL REFERENCES news_articles (id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    sentiment TEXT,
    sentiment_reasoning TEXT,
    CONSTRAINT pk_news_article_tickers PRIMARY KEY (article_id, ticker),
    CONSTRAINT ck_news_article_tickers_uppercase CHECK (ticker = UPPER(ticker))
);

CREATE INDEX IF NOT EXISTS idx_news_article_tickers_ticker
    ON news_article_tickers (ticker);

CREATE TABLE IF NOT EXISTS news_article_chunks (
    article_id TEXT NOT NULL REFERENCES news_articles (id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_news_article_chunks PRIMARY KEY (article_id, chunk_index),
    CONSTRAINT ck_news_article_chunks_index CHECK (chunk_index >= 0)
);

CREATE TABLE IF NOT EXISTS research_notes (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    note_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_research_notes_ticker_uppercase CHECK (ticker = UPPER(ticker))
);

CREATE INDEX IF NOT EXISTS idx_research_notes_user_id
    ON research_notes (user_id);

CREATE TABLE IF NOT EXISTS analysis_reports (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    title TEXT NOT NULL,
    report_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_analysis_reports_ticker_uppercase CHECK (ticker = UPPER(ticker))
);

CREATE INDEX IF NOT EXISTS idx_analysis_reports_user_id
    ON analysis_reports (user_id);
