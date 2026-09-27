# Synthetic DENSO documents (evaluation fixtures, not real publications)

These files are **generated**, not published by DENSO. Each is an excerpt of a
real DENSO PDF from [`../denso_mock_docs/`](../denso_mock_docs/), machine-translated
by a language model when the target language differs from the source, so that
the corpus has documents in Japanese, English and Vietnamese (the real files
have no Vietnamese, and `denso.com` blocks automated downloads).

Every excerpt exists in exactly **one** language, and the questions in
[`evals/documents/synthetic_denso_xl.jsonl`](../../evals/documents/synthetic_denso_xl.jsonl)
ask each fact in all three, so one question per fact is same-language and two
are cross-language over the same answer.

| file | source | language |
|---|---|---|
| `syn-compliance-code-vi` | Code of Conduct (en) | Vietnamese |
| `syn-governance-a-ja` | Corporate Governance Report 2025 (en), first excerpt | Japanese |
| `syn-governance-b-en` | Corporate Governance Report 2025 (en), second excerpt | English (untranslated) |
| `syn-qa-manual-ja` | DENSO TEN QA manual (en) | Japanese |
| `syn-human-rights-ja` | Human Rights Policy (en) | Japanese |
| `syn-integrated-report-a-en` | Integrated Report 2025 (ja), first excerpt | English |
| `syn-integrated-report-b-vi` | Integrated Report 2025 (ja), second excerpt | Vietnamese |
| `syn-securities-report-a-en` | Securities Report (ja), first excerpt | English |
| `syn-securities-report-b-vi` | Securities Report (ja), second excerpt | Vietnamese |

## Read the numbers with these limits in mind

- A translated excerpt is cleaner than a real PDF (no table debris, consistent
  wording), so absolute scores here are optimistic. Compare configurations and
  query languages against each other; do not quote a recall figure.
- The corpus is tiny (9 documents, ~90 chunks), so top-3 is a lot of the
  corpus. Prefer the "answer in returned chunks" measure to the document hit.
- The questions were first drafted by a model from the document text and then
  **edited by hand to one house style**, so they are uniform (rules below).
  Every answer is verified to occur verbatim in its document. They can still echo
  the document's wording.
- Translation quality is not human-checked.

## House style for the questions

Enforced by `python scripts/synth_corpus.py check` and by a test on the committed
files, so edits keep the set uniform:

- **Self-contained.** Names the subject and any fiscal period; never "according to
  the text", "this document" or "the current year". (The securities-report excerpt
  is the fiscal year ended 31 March 2023, so its questions say so.)
- **One sentence, one answer**, ending in a question mark; en 60-135 characters,
  ja 25-90, vi 55-150.
- **A distinctive answer**: verbatim from the document, at least 4 characters, with
  its unit when it is a number (`521,615 million yen`, not `521`), so it cannot
  match a chunk by accident.
- **Same question in en / ja / vi**, at the same level of detail.
- **Exactly 4 facts per document**, each asked in all 3 languages (108 questions).

## Regenerate / load

```bash
# verify the committed questions against the documents and the style rules
python scripts/synth_corpus.py check

# redraft everything with a model (needs TINI_PROVIDER / TINI_MODEL). This replaces
# the documents AND the hand-edited questions, so it needs --force
python scripts/synth_corpus.py build --force

# load into a workspace of its own, then measure
python scripts/synth_corpus.py ingest --workspace denso-xl
python scripts/document_baseline.py xl-dense --workspace denso-xl --sets synthetic_denso_xl --rerank off
python scripts/language_matrix.py evals/baselines/xl-dense.json
```
