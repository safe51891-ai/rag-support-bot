# RAG Support Bot: Evaluation Report

Documents: 5 help-center articles (refunds, shipping, accounts, cancellations, payments).
Questions: 30 tuning questions (19 answerable, 11 the docs cannot answer) and 10 held-out questions (6 answerable, 4 unanswerable).

## Tuning set

| Version | Hit@1 | Hit@3 | MRR | Unanswerable questions answered (wrong) | Answerable questions escalated (unneeded) |
|---|---|---|---|---|---|
| Baseline (exact words, no gate) | 0.895 | 0.895 | 0.895 | 1 of 11 | 5 of 19 |
| Improved (stemming, synonyms, coverage gate) | 0.895 | 0.947 | 0.921 | 1 of 11 | 3 of 19 |

## Held-out set

Read this with care: 10 questions is a small sample, and I printed these results during development, before the final threshold was set. Treat them as a sanity check, not as proof.

| Version | Hit@1 | Hit@3 | Unanswerable answered | Answerable escalated |
|---|---|---|---|---|
| Improved | 0.833 | 1.000 | 0 of 4 | 0 of 6 |

## Decisions and their trade-offs

- **Coverage gate threshold 0.3.** A stricter gate (0.5) answered no unanswerable question but escalated 9 of 19 valid ones. A looser gate (0.2) allowed 3 wrong answers. I chose 0.3 because a wrong answer misinforms a customer, while an escalation only costs a person's time.
- **Synonyms are hand-written.** They come from the tuning set, so they will overfit it. Build them from real support tickets instead.

## Known gaps

- **"Can I change my delivery address after the order is placed?"** is still answered, using the shipping doc. The answer is wrong, because the docs do not cover address changes. This is the failure to watch for: shared words such as "delivery" can pass the gate.
- Three answerable questions are still escalated in the tuning set. These are paraphrases that use words the docs never use.
- The docs are short, written for this demo. Real help centers are larger, so retrieval gets harder.
