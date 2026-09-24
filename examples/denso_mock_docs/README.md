# Denso mock documents

Sample administrative/governance PDFs for exercising the document-ingestion
pipeline (`tini/documents/`) end to end, including the Qdrant backend
(`tini/documents/qdrant_store.py`, `docker-compose.yml`). These are Denso
Corporation's own public disclosures — safe to use as realistic sample content.

`denso.com` returns a 403 (Akamai WAF) from this project's sandboxed dev
environment, so the PDFs couldn't be fetched automatically. Download them
yourself (a regular browser works fine) and drop them in this folder:

Already have (checked into this folder): the Code of Conduct
(`compliance-doc-employee-behavior-guideline-en.pdf`) and the 2021 Basic
Policies on Corporate Governance (`management-doc-corporate-governance-policy-2021-en.pdf`).

More candidates, same source, organized by kind:

**Governance / compliance**
- Corporate Governance Report 2025:
  https://www.denso.com/global/en/-/media/global/about-us/sustainability/governance/management/management-doc-corporate-governance-2025-en.pdf
- DENSO Group Human Rights Policy:
  https://www.denso.com/global/en/-/media/global/about-us/sustainability/society/humanrights/humanrights-doc-human-rights-policy-en.pdf
- Supplier Sustainability Guidelines:
  https://www.denso.com/global/en/-/media/global/about-us/sustainability/society/supply-chain/supply-chain-doc-sustainability-guidelines-en.pdf

**Financial / investor**
- Financial Report 2024:
  https://www.denso.com/global/en/-/media/secure-investors/settlement/2024/financial-report2024.pdf
- Integrated Report 2025 (large — good for testing chunking on a long document):
  https://www.denso.com/global/en/-/media/Global/about-us/investors/annual-report/2025/annual-report-doc-2025-viewing-en.pdf

**Other**
- Green, Social and Sustainability Bond Framework 2021:
  https://www.denso.com/global/en/-/media/global/about-us/sustainability/library/evaluation/202107_denso-framework_e.pdf

## Ingest them

```bash
docker compose up -d qdrant
pip install -e '.[qdrant]'
```

Add to `.env`:

```dotenv
TINI_DOCUMENT_INDEX=qdrant
TINI_QDRANT_URL=http://localhost:6333
TINI_EMBEDDING_PROVIDER=gemini        # or openai / openai_compatible
TINI_EMBEDDING_MODEL=gemini-embedding-2
TINI_EMBEDDING_DIMENSIONS=768
```

Then:

```bash
python -m tini ingest examples/denso_mock_docs --workspace denso
python -m tini documents --workspace denso --search "corporate governance"
```
