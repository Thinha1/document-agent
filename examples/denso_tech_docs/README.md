# Public DENSO technical documents (not committed)

The PDFs here are downloaded, not stored in git. The one public DENSO product
manual that could be fetched without circumventing anything (`denso.com` returns
403 to automated clients, and this repo does not try to get around that):

| file | source |
|---|---|
| `Spark-Plug-Manual-English_v2.pdf` | https://assets.denso-am.eu/production/attachments/Spark-Plug-Manual-English_v2.pdf ("Spark Plugs: Discovering DENSO Technology", 68 pages, DENSO Aftermarket Europe) |

It is technical prose with charts and almost no tables, so it does not test table
retrieval; it is used as a real technical distractor and for the few questions in
[`evals/documents/denso_qa_tech.jsonl`](../../evals/documents/denso_qa_tech.jsonl)
(workspace `denso-tech`). For tables, see [`../denso_tech_synthetic/`](../denso_tech_synthetic/).
