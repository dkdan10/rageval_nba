# rageval-nba

Reference NBA hybrid RAG evaluation demo.

## Milestone 7 Corpus

The tracked corpus source list lives in `examples/corpus/articles.json`. It is a
curated metadata manifest with stable article IDs, source URLs, topics, storage
policies, and short notes. Raw fetched pages are intentionally not tracked:
`scripts/build_corpus.py --fetch` writes them to the gitignored
`data/raw/corpus/` cache, and `scripts/build_corpus.py --from-cache` builds
`articles` and `article_chunks` in the local SQLite database.

Some manifest records include concise repo-authored summaries so the demo can
remain reproducible without committing copyrighted article text or depending on
blocked source pages. Live fetches respect robots.txt and may vary by network,
source availability, and publisher controls. The demo retriever currently uses a
deterministic lexical fallback over `article_chunks`; it is not production vector
retrieval and does not require paid embedding calls.
