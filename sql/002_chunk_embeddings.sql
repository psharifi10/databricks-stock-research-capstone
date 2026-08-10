-- Phase 4B: keep each chunk and its embedding in the same logical row.
CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE;

ALTER TABLE news_article_chunks
    ADD COLUMN IF NOT EXISTS embedding VECTOR(384);

ALTER TABLE news_article_chunks
    ADD COLUMN IF NOT EXISTS embedding_model TEXT;

CREATE INDEX IF NOT EXISTS news_article_chunks_embedding_ann
    ON news_article_chunks
    USING lakebase_ann (embedding vector_cosine_ops);

-- The Serverless notebook calls this bounded function through the Lakebase
-- Data API. Arguments arrive as typed RPC values; no SQL is built in Python.
CREATE OR REPLACE FUNCTION set_news_article_chunk_embedding(
    p_article_id TEXT,
    p_chunk_index INTEGER,
    p_embedding REAL[],
    p_embedding_model TEXT
)
RETURNS VOID
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, pg_temp
AS $$
BEGIN
    IF COALESCE(array_length(p_embedding, 1), 0) <> 384 THEN
        RAISE EXCEPTION 'Embedding must contain exactly 384 values.';
    END IF;

    IF NULLIF(BTRIM(p_embedding_model), '') IS NULL THEN
        RAISE EXCEPTION 'Embedding model is required.';
    END IF;

    UPDATE public.news_article_chunks
    SET embedding = p_embedding::VECTOR(384),
        embedding_model = p_embedding_model
    WHERE article_id = p_article_id
      AND chunk_index = p_chunk_index;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'News article chunk was not found.';
    END IF;
END;
$$;
