# Policy sources

Prototype corpus `src/voxcare/domain/data/policies.json` (`VOXCOM-DEMO-POLICY-V1`, `1.0-demo`, `is_fictional=true`) is loaded and validated once by `src/voxcare/domain/policy.py` (`_load_corpus`) and served by `SessionBoundary` through the model-visible `retrieve_policy_clauses` tool.

## Verbatim source of each clause

All 9 clause texts are transcribed verbatim from the accepted design document `02_research/42_voxcare_p1_detailed_design.md` §8.2 (kept in the separate DACN design repository, not in this one) (plain-text rendering of the spec's LaTeX markup; no terms changed). Each row's `source_reference` names that section:

| Clause | Spec section | Terms |
|---|---|---|
| `POL-SHIP-001` (Clause 1.1) | §8.2.1 | 500,000 VND / 10 km free radius, 10,000 VND/km 10–30 km, regional quote beyond 30 km |
| `POL-SHIP-002` (Clause 1.2) | §8.2.1 | bulky thresholds (>300L, >9kg, >55"), 2-person handling, 50,000 VND/floor from 2nd floor, COD cap 20,000,000 VND |
| `POL-INST-001` (Clause 2.1) | §8.2.2 | pedestal assembly, washer hose connections, refrigerator levelling + power-on test |
| `POL-INST-002` (Clause 2.2) | §8.2.2 | drilling/bracket/copper-beyond-3m billed per auxiliary schedule, itemised quote + confirmation first |
| `POL-WAR-001` (Clause 3.1) | §8.2.3 | Samsung 24-month general; Digital Inverter 20-year (01/11/2022–31/03/2026) / 10-year (from 01/04/2026); in-home dispatch ≥43" + major appliances; 4 exclusions |
| `POL-WAR-002` (Clause 3.2) | §8.2.3 | LG 24-month parts+labour; 10-year Inverter Direct Drive / Smart Inverter compressor; free metro inspection |
| `POL-WAR-003` (Clause 3.3) | §8.2.3 | Panasonic 24 months (12 for select TV/audio); 12-year Inverter compressor; 10-year washer motor; hotline 1800 1593 |
| `POL-RET-001` (Clause 4.1) | §8.2.4 | 30-day 1-for-1 replacement, technician-certified defect, original packaging + serial |
| `POL-RET-002` (Clause 4.2) | §8.2.4 | VoxCom pays freight for defect/wrong-SKU; change-of-mind is per-merchant review, never promised |

Stored categories are `SHIPPING` / `INSTALLATION` / `WARRANTY` / `RETURNS`; `ALL` is accepted only as the retrieval parameter, never stored. Brand/product scoping lives in `applicability_rules` per clause.

## Retrieval honesty

Retrieval is deterministic lexical overlap (`domain/policy.py`, minimum coverage 0.2), explicitly a demo simplification of the spec §8.2 vector design (`top_k=3`, similarity threshold 0.70). Scores are overlap ratios, not semantic similarities, and must never be compared with the design's 0.70 threshold. `INSUFFICIENT_EVIDENCE` / `NEEDS_CONTEXT` envelopes report honestly when no clause applies.

## Verified external sources (shape/scope only)

The external sources below only verify the shape of real-world information and warn against importing another seller's terms as VoxCom terms. No external text was copied into the fictional corpus:

1. **Vietnam Consumer Protection Law 19/2023/QH15**, National Legal Database, active from 2024-07-01: <https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=161263&Keyword=>. Article 30: warranty follows parties' agreement or mandatory law; if offered, business must publicly announce policy, including time, scope, method and exclusions. Supports asking for the applicable published policy and evidence; does not establish a general change-of-mind return entitlement. The prototype states no legal conclusion.
2. **Samsung Vietnam product warranty**: <https://www.samsung.com/vn/support/warranty/>. Date-tiered inverter component terms motivate the `POL-WAR-001` purchase-date tiering; exact model/date and authoritative manufacturer evidence are still required per query.
3. **LG Vietnam warranty center**: <https://www.lg.com/vn/tro-giup/bao-hanh/>. Product/component-specific terms (e.g. 10-year entries for selected washer motors / refrigerator compressors); no general 24-month duration for every LG device. Corpus retrieval requires exact model/date evidence and states no LG duration without it.
4. **Panasonic Vietnam product warranty periods**: <https://www.panasonic.com/vn/support/product-warranty-period.html>. Product/origin/evidence distinctions; no general duration was copied into the fictional fixture.
5. **GHN developer docs**, official carrier API docs: <https://developer.ghn.vn/vi>. Delivery status/date is dynamic carrier/OMS evidence; fee/ETA cannot be inferred from policy prose. No GHN statuses, prices, or service promises are presented as VoxCom policy.

`POL-RET-001` keeps its 30-day term only as an explicit fictional test condition, conditioned on verified delivery evidence and authorized technician assessment — not a statutory cooling-off period, actual VoxCom promise, or approval. `POL-RET-002` intentionally leaves change-of-mind, freight charge, restocking charge, refund amount and timing unresolved, tracking the OPD rule that the bot must not invent terms.
