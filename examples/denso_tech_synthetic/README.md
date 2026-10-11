# Synthetic technical tables (evaluation fixtures, not DENSO data)

Eight documents of fictional engineering tables in English, Japanese and
Vietnamese, **generated with no model** by
[`scripts/synth_tech.py`](../../scripts/synth_tech.py) from a fixed seed, so every
value is exact and the files are byte-for-byte reproducible
(`python scripts/synth_tech.py build`). Every part number and figure is made up;
each document says so on its first lines. DENSO's real engineering documents are
not public, so these only imitate the *layout* of torque charts, connector
pin-outs, fault-code lists, thermistor tables, sampling plans, supplier quality
tables and injector / ignition-coil data sheets.

| file | language | table layout | shape |
|---|---|---|---|
| `tech-torque-spec-en` | en | Markdown | 15 rows, two-level header (grade × dry/lubricated) |
| `tech-thermistor-table-en` | en | aligned text | 39 rows, three sensors |
| `tech-injector-flow-en` | en | Markdown | 21 rows × 8 models |
| `tech-connector-pinout-ja` | ja | aligned text | 54 pins, tagged by bank |
| `tech-sampling-plan-ja` | ja | aligned text | 14 lot ranges × 6 AQL levels |
| `tech-ignition-coil-ja` | ja | Markdown | 64 part numbers |
| `tech-fault-codes-vi` | vi | Markdown | 20 fault codes, two banks |
| `tech-supplier-ppm-vi` | vi | aligned text | 44 suppliers, quarter × category header |

"Aligned text" is what a PDF text extractor gives back: the header once at the
top and plain columns after it. That is the hard case, because once the table is
split into chunks the later rows have lost their column headers.

The questions ([`evals/documents/synthetic_denso_tech.jsonl`](../../evals/documents/synthetic_denso_tech.jsonl))
are ten facts, each aimed at one way a table breaks retrieval, asked in all three
languages (30 questions). The `format` field is the technique, so the dashboard's
per-format chips are a table-technique breakdown. Each document exists in one
language, so per fact one question is same-language and two are cross-language.

Load them into their own workspace, next to the real DENSO documents so they
have distractors, and score the set against that workspace:

```bash
python scripts/synth_tech.py ingest --workspace denso-tech
python -m tini ingest examples/denso_mock_docs --workspace denso-tech
```

Ten facts is a diagnostic, not a benchmark: each technique is one fact asked
three ways, so a technique's "3/3" is one success, not three. Use it to see
*which* table problem a change fixes or causes.
