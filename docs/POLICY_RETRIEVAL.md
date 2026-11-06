# Policy retrieval and the G-10 calibration

`retrieve_policy_clauses(query, category)` answers policy questions only from the nine fictional VoxCom clauses in `src/voxcare/domain/data/policies.json` (D§8.2; `scripts/sync_policy_corpus.py` checks the file against the design text).

## Pipeline

1. **Topic filter** — the requested `category` (`WARRANTY`, `RETURNS`, `SHIPPING`, `INSTALLATION`, or `ALL`).
2. **Applicability filter** — market, channel, product categories and brands inferred from the query text (`domain/rules.py::product_categories_in`, `brands_in`). A kitchen-appliance installation question never receives the TV/AV installation clause; an LG warranty question never receives the Samsung clause.
3. **Vector search** — Chroma collection `voxcom_policy_v1`, cosine space, `similarity_score = 1 − cosine_distance`, top-k = 3 (`VOXCARE_RAG_TOP_K`). The embedded text is `title + ". " + content_en`; the stored document stays `content_en` verbatim. The collection is rebuilt automatically when the corpus, the embedding model or this recipe changes.
4. **Grounding gate** — when the best score reaches the threshold, the tool returns the clauses at or above it (with corpus metadata and applicability rules); otherwise `INSUFFICIENT_EVIDENCE` with `http: 200`, and the bot speaks the out-of-domain admission (G§7.3), never a fee or a timeline.

The spoken reply reads at most three clauses and the technician condition; the screen shows the full clause text.

## Why the threshold is not 0.70 (G-10)

The design's 0.70 threshold and its demo scores (0.88 / 0.52) belong to an embedding model the design does not name. Similarity scores are not comparable across models, so the threshold was recalibrated for the model this prototype uses.

- **Model:** `all-MiniLM-L6-v2` (Chroma's default ONNX embedding, 384 dimensions, downloaded once, about 80 MB). `VOXCARE_POLICY_EMBEDDING=minilm`.
- **Offline alternative:** a deterministic hashing embedding (`VOXCARE_POLICY_EMBEDDING=hashing`) for tests and machines without the model. It is lexical, not semantic.
- **Labeled set:** `evaluation/policy_calibration.json`, 50 queries — 32 in-domain paraphrases with the expected clause ids and 18 out-of-domain questions. The first two rows are the Scenario 4 and Scenario 5 queries.
- **Method:** `scripts/calibrate_policy_threshold.py` runs the real pipeline above for every query and every candidate threshold. An in-domain query counts as grounded when its best score reaches the threshold and an expected clause is returned; an out-of-domain query counts as refused when nothing reaches the threshold. Accuracy = (grounded + refused) / 50.

### Results (measured on 50 queries)

| Embedding | Threshold | Accuracy | In-domain grounded | Out-of-domain refused | Correct top-1 clause | Scenario 4 / 5 |
| --- | --- | --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | **0.32 (default)** | **0.82** | 27 / 32 | 14 / 18 | 31 / 32 | pass / pass |
| all-MiniLM-L6-v2 | 0.3235 (best on this set) | 0.84 | 27 / 32 | 15 / 18 | 31 / 32 | pass / pass |
| all-MiniLM-L6-v2 | 0.70 (design value) | 0.42 | 3 / 32 | 18 / 18 | 31 / 32 | **fail** / pass |
| hashing | **0.09 (default)** | **0.84** | 25 / 32 | 17 / 18 | 27 / 32 | pass / pass |
| hashing | 0.70 (design value) | 0.36 | 0 / 32 | 18 / 18 | 27 / 32 | fail / pass |

The default is the best value on the set rounded down to two decimals; a value fitted exactly to 50 queries would overstate its precision. It sits well below the Scenario 4 return-clause score (0.3766). Full per-query reports: `evaluation/results/policy_calibration_minilm.json` and `…_hashing.json`.

Scenario 4 scores with MiniLM: POL-WAR-001 0.6709, POL-RET-001 0.3766, POL-RET-002 0.3325. Scenario 5's best score is 0.2535.

### Remaining errors at 0.32 (MiniLM)

- **Missed (5 in-domain):** short installation questions ("Do you set up the TV stand for free?", 0.24; "Will the technician connect the water hoses of my washer?", 0.27; "Is installation free for my washing machine?", 0.30; the copper-pipe question, 0.29) and "Can I check the box for damage before paying the driver?" (0.16). These end in the out-of-domain admission, which is safe but unhelpful.
- **Wrongly grounded (4 out-of-domain):** "Do you offer extended warranty insurance plans for purchase?" (0.42, warranty clauses), "What is your refund policy for concert tickets?" (0.41, POL-RET-002), "Do you deliver to Cambodia?" (0.38, POL-SHIP-001), "Can I get same-day delivery by drone?" (0.3234, POL-SHIP-001). The reply then reads a real clause that does not answer the question. No fee or timeline is invented, but the caller may take the clause as an answer. This is the main limitation of a single similarity gate on this small corpus.

## Reproduce and change

```sh
uv run python scripts/calibrate_policy_threshold.py --embedding minilm          # summary
uv run python scripts/calibrate_policy_threshold.py --embedding minilm --json   # per-query report
```

`VOXCARE_RAG_THRESHOLD` overrides the default for the configured model. Recalibrate after changing the embedding model, the embedded-text recipe, or the corpus. A larger labeled set (and a second-stage check such as a clause-level entailment test) would be needed before quoting these numbers as general retrieval accuracy; they describe this 50-query set only.
