# Release 1.0.0 — Hybrid Search & Qdrant Client 1.18

**Service:** vectorization-service

> ⚠️ **Deploy code + client together.** Old code on new client or new code on old client will break.

---

## Table of Contents

- [What's New](#whats-new)
- [Search Architecture](#search-architecture)
- [Qdrant 1.18 Migration](#qdrant-118-migration)
- [Dependencies](#dependencies)
- [Deployment](#deployment)

---

## What's New

- **Hybrid search** — dense vectors + BM25 sparse (keyword), fused server-side with RRF. Enabled by setting `SPARSE_SEARCH_ENABLED=true` (new env key this release). When disabled, the service falls back to dense similarity search only (semantic/cosine matching across the 5 named vector fields).
- **BM25 / keyword search** — using `Qdrant/bm25` sparse model via fastembed. Two sparse vectors are stored per chunk: `bm25_text` (chunk body) and `bm25_title` (normalized document title). Document titles are normalized by stripping file extensions (`.docx`, `.pdf`, …), replacing `_-./ ` with spaces, and lowercasing — so `"source_doc_MI Repository for AI Pilot_c3481f.docx"` becomes `"source doc mi repository for ai pilot c3481f"`, making every title word individually searchable.
- **Title matching** — exact, partial (prefix), and mid (infix/substring), each with a configurable score boost. BM25 title vector now covers term-level matching; Phase 1 boost multipliers are reduced accordingly (see boost table below).
- **Summary matching** — summary field now participates in keyword matching and boosting (mirrors title behaviour).
- **Hybrid ranking** — two fusion methods available, selected via `HYBRID_FUSION_METHOD`. Both modes use a pre-fusion `sparse_combined` blend of the two BM25 scores: `(SPARSE_TITLE_SHARE × norm_title + SPARSE_TEXT_SHARE × norm_text) / (TITLE_SHARE + TEXT_SHARE)`.
  - `weighted` *(default)* — dense and sparse scores are each min-max normalised independently, then combined as `0.7 × dense + 0.3 × sparse_combined`. Scores are comparable to the plain `filter_score` threshold.
  - `rrf` — Reciprocal Rank Fusion: ranks the combined dense list against the sparse list (`1/(k + dense_rank) + 1/(k + sparse_rank)`, k=60), then min-max normalised. Actual BM25 hits are identified from raw pre-normalization scores so the pool-minimum scorer (which maps to 0.0 after min-max) still receives a `sparse_rank`.
- **Debug scoring fields** — set `include_scoring_debug: true` on a request to surface per-result hybrid breakdown: `keyword_text_score` (raw `bm25_text` score, `null` if no hit), `keyword_title_score` (raw `bm25_title` score, `null` if no hit), `dense_score`, `normalized_dense`, `normalized_sparse` (`sparse_combined`), `fusion_score`, `rrf_score`, `dense_rank`, `sparse_rank`.
- **Title/summary boost multipliers**:

  | Parameter | Value |
  |---|---|
  | `EXACT_TITLE_BOOST` | 2.0 |
  | `PARTIAL_TITLE_BOOST` | 1.2 |
  | `EXACT_SUMMARY_BOOST` | 1.3 |
  | `PARTIAL_SUMMARY_BOOST` | 1.1 |

- **Ranking bugs fixed** — hybrid score was always 0.0 (RRF fusion score was ignored, now used directly); candidate limit could reach 1M (now capped at `min(top_k × 20, 10000)`); score cap at 1.0 ran before boosting (now applied once, after all boosts).
- **qdrant-client 1.18** compatibility across all call sites.

---

## Search Architecture

<details>
<summary>Collection structure</summary>

- **5 dense named vectors** (384-dim, cosine): `text`, `title`, `summary`, `tags`, `metadata`
- **2 sparse vectors** `bm25_text` and `bm25_title` (IDF modifier each) — active when `SPARSE_SEARCH_ENABLED=true`; `bm25_title` is indexed from the normalized document title, `bm25_text` from the chunk body
- **Payload indexes** created at startup (idempotent):
  - `source_id`, `metadata.company`, `tags` — keyword
  - `metadata.DOCUMENT_TYPE` — text
  - `title`, `summary` — prefix-tokenized text (`min_token_len=2, max_token_len=20, lowercase`)

</details>

<details>
<summary>Hybrid search (dense + BM25 + RRF)</summary>

When `SPARSE_SEARCH_ENABLED=true`, one batch call is issued with:
- 5 dense prefetch queries (one per named vector field)
- 2 BM25 sparse prefetch queries (`bm25_text` for body, `bm25_title` for normalized title)

Results are fused client-side via the configured `HYBRID_FUSION_METHOD` (`weighted` or `rrf`).

Falls back to dense-only if sparse encoding fails.

</details>

<details>
<summary>Dense-only search</summary>

When sparse search is disabled, 5 field queries run in a single batch and are merged with priority weights:

| Field | Weight |
|-------|--------|
| title | 0.36 |
| text | 0.27 |
| tags | 0.14 |
| summary | 0.14 |
| metadata | 0.09 |

</details>

<details>
<summary>Title & summary boosting</summary>

Applied after semantic ranking when `HYBRID_SEARCH_ENABLED=true` and `search_mode != "semantic"`:

- **Exact match** → multiply score by boost (title ×2.0, summary ×1.3), capped at 1.0
- **Partial/mid match** → multiply by lower boost (title ×1.2, summary ×1.1), capped at 1.0
- **Missing docs** (matched keyword but below semantic threshold) → injected at a floor score (0.15 × boost)

Title boost takes precedence — a doc matching both title and summary only gets the title boost.

</details>

<details>
<summary>Filtering</summary>

| API parameter | Qdrant field | Logic |
|---|---|---|
| `categories` | `tags` | OR within, AND between types |
| `organizations` | `metadata.company` | OR within, AND between types |
| `resource_type` | `metadata.DOCUMENT_TYPE` | substring match |
| `file_type` | `metadata.type` | OR within, AND between types |

Score filtering: use `filter_score` (single threshold) or `detail_filter_score` (per-field thresholds, OR logic).

</details>

---

## Qdrant 1.18 Migration

<details>
<summary>Breaking API changes fixed</summary>

| Removed / invalid in 1.18 | Fix applied |
|---|---|
| `client.search()` removed | `query_points(...).points` |
| `NearestQuery(nearest=NamedVector(...))` | `query=<vector>, using="<name>"` |
| `client.embed_sparse()` / `set_sparse_model()` | `fastembed.SparseTextEmbedding("Qdrant/bm25")` directly |
| `PointVectors` used in `upsert` | moved to `update_vectors()` |
| `FieldCondition(invert=True)` | replaced with `must_not` filter |
| `qdrant_client.http.models` (legacy path) | `qdrant_client.models` |

</details>

---

## Dependencies

- **`qdrant-client[fastembed]`** — `>=1.18.0,<2.0.0` (required; older clients are missing the APIs used here)
- **`fastembed`** — pulled in by the `[fastembed]` extra; used directly for BM25
- **Qdrant server** — `>=1.18` required
- **`Qdrant/bm25` model** — downloads on first sparse use; pre-cache in the image to avoid first-request latency

---

## Deployment

### Pre-deploy checklist

1. **Verify qdrant-client version:**
   ```bash
   pip show qdrant-client | grep Version
   # Must be >= 1.18.0
   ```

2. **Verify Qdrant server version:**
   ```bash
   curl -s http://<QDRANT_HOST>:<QDRANT_PORT>/collections | python3 -m json.tool | head
   # Or check the Qdrant dashboard — server must be >= 1.18
   ```

3. **Take a Qdrant snapshot (backup) before deploying:**
   ```bash
   # Create a snapshot of the collection
   curl -X POST "http://<QDRANT_HOST>:<QDRANT_PORT>/collections/<COLLECTION_NAME>/snapshots"
   
   # List snapshots to confirm it was created
   curl "http://<QDRANT_HOST>:<QDRANT_PORT>/collections/<COLLECTION_NAME>/snapshots"
   
   # Download the snapshot locally for safekeeping
   curl -o backup-<COLLECTION_NAME>-$(date +%Y%m%d).snapshot \
     "http://<QDRANT_HOST>:<QDRANT_PORT>/collections/<COLLECTION_NAME>/snapshots/<snapshot_name>"
   ```

### Deploy steps

4. Add the new env keys to `.env`:
   ```dotenv
   HYBRID_SEARCH_ENABLED=true
   SPARSE_SEARCH_ENABLED=true
   SPARSE_TEXT_VECTOR_NAME=bm25_text
   SPARSE_TITLE_VECTOR_NAME=bm25_title
   SPARSE_TITLE_SHARE=0.45
   SPARSE_TEXT_SHARE=0.55
   RRF_K=60
   EXACT_TITLE_BOOST=2.0
   PARTIAL_TITLE_BOOST=1.2
   EXACT_SUMMARY_BOOST=1.3
   PARTIAL_SUMMARY_BOOST=1.1
   SHORT_QUERY_THRESHOLD=3
   QDRANT_CHECK_COMPATIBILITY=false
   ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

6. Start the service 

7. **Back-fill BM25 for existing docs** (idempotent — only updates sparse vectors, does not re-embed dense). Populates both `bm25_text` and `bm25_title` per point:
   ```bash
   # Dry run first
   COLLECTION_NAME=documents1 SPARSE_SEARCH_ENABLED=true \
     python scripts/migrate_to_sparse_vectors.py --dry-run

   # Real run
   COLLECTION_NAME=documents1 SPARSE_SEARCH_ENABLED=true \
     python scripts/migrate_to_sparse_vectors.py
   ```
   > Pass `COLLECTION_NAME` explicitly — the migration script defaults to `documents`, the service uses `documents1`. New uploads get both BM25 vectors automatically; only pre-existing docs need back-fill.

### Rollback

Revert the code, pin `qdrant-client[fastembed]>=1.9.0,<1.14.0`, set `SPARSE_SEARCH_ENABLED=false`, and restart. The two sparse vector fields (`bm25_text`, `bm25_title`) and prefix indexes added by this release are inert when sparse is disabled and can be left in place.

---
