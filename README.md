# rageval-nba

Reference NBA hybrid RAG evaluation demo.

## Milestone 7 Corpus

For the local demo/report path, build the deterministic seed database and cached
corpus, then run the HTML report command:

```bash
uv run python scripts/build_stats_db.py
uv run python scripts/build_corpus.py --from-cache
uv run rageval run examples/nba_test_suite.yaml --output report.html
```

Rebuild the corpus after every stats DB rebuild. `build_stats_db.py` recreates
`data/nba.db`, so any previously loaded `article_chunks` are removed until
`scripts/build_corpus.py --from-cache` is run again.

Real NBA API ingestion is still available explicitly when network access and
source availability are acceptable:

```bash
uv run python scripts/build_stats_db.py --mode real
```

For a safer resumable pull, reuse raw JSON already saved under `data/raw/`, set
a network timeout, and keep a small pause between NBA Stats calls:

```bash
uv run python scripts/build_stats_db.py --mode real --resume-raw --timeout-seconds 30 --rate-limit-seconds 1
```

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

The HTML report uses Pico.css and Chart.js from public CDNs for lightweight
styling and charts. The demo CLI reports article-prefix retrieval metrics
because curated article IDs are stable while chunk indexes can change when
cached source pages change.

The report includes a per-category breakdown (mean metric scores by factual,
analytical, hybrid, and unanswerable/adversarial category) and a highlighted
failure-mode panel listing metric errors, zero-score metrics, refusal
disagreements, SQL errors, and missing routes.

For fast feedback during iteration, run a 4-5 case representative subset:

```bash
uv run rageval demo --output demo-report.html
```

Add `--verbose` to either `rageval run` or `rageval demo` to print one
case-by-case progress line (case id, route, refused flag, metric and error
counts).

Use `--metrics` to run a deterministic subset, for example
`--metrics refusal,prefix_recall@5`. The CLI also accepts `--no-cache` for plan
parity; the default deterministic demo path does not use the LLM cache. Judge
calibration is available through `rageval calibrate <judge_name>`, delegating to
the same calibration implementation used by `scripts/calibrate_judge.py`.

The seed database is intentionally small. The demo SQL path uses real table
queries where the seed fixture contains the requested rows and deterministic
literal rows for examples outside that offline fixture.
