# Document Internal Agent

[English](README.md) · **Tiếng Việt**

Trợ lý chạy cục bộ (local-first) trả lời câu hỏi từ tài liệu nội bộ của tổ chức,
và cho biết câu trả lời lấy từ **tài liệu nào, trang nào**. Mã nguồn được giữ gọn để
một người có thể lần theo một yêu cầu từ gateway đến câu trả lời trong một buổi chiều.

> **Lưu ý về tên.** Dự án trước đây tên là *Tini Agent*. Package Python, lệnh `tini`
> và các biến `TINI_*` tạm thời giữ nguyên tên cũ để file `.env` và các lệnh hiện có
> vẫn chạy bình thường.

## Hiện tại agent làm được gì

| Khả năng | Ý nghĩa thực tế |
|---|---|
| **Nạp tài liệu nội bộ** | PDF, DOCX, Markdown, văn bản thuần, HTML, CSV, JSON, YAML; từ một file, một thư mục hoặc upload qua dashboard. Nạp lại không bị trùng (băm SHA-256); file đổi nội dung sẽ thay toàn bộ chunk cũ trong một giao dịch. |
| **Đọc PDF scan** | Trang gần như không có chữ được render và chuyển thành văn bản bằng mô hình vision (OCR), có cache để mỗi trang chỉ tốn phí một lần. |
| **Tìm theo ý nghĩa *và* theo từ khoá chính xác** | Hybrid retrieval trên Qdrant: embedding dense `bge-m3` kết hợp vector từ khoá sparse, hợp nhất bằng reciprocal-rank fusion. |
| **Rerank cục bộ** | `bge-reranker-v2-m3` tự host (text-embeddings-inference) chấm lại 50 ứng viên đầu. Nội dung tài liệu chỉ được gửi tới địa chỉ cục bộ; URL reranker nào khác sẽ bị từ chối khi khởi động. |
| **Đa ngôn ngữ Anh, Nhật, Việt** | Embedding đa ngôn ngữ, chia chunk và tìm từ khoá hỗ trợ CJK, và tuỳ chọn dịch câu hỏi tiếng Việt sang tiếng Anh cho reranker. |
| **Giữ nguyên bảng biểu** | Bảng được chia chunk theo hàng, tiêu đề, chú thích và dòng header cột được lặp lại ở mọi chunk, nên một hàng toàn số vẫn mang đủ ý nghĩa. |
| **Trích dẫn nguồn kèm số trang** | Câu trả lời dẫn `[tiêu đề#chunk, pp.A-B]`. Khoảng trang được lưu cho mọi chunk PDF. |
| **Ẩn tài liệu lỗi thời** | Tài liệu bị thay thế hoặc bị loại trừ không xuất hiện trong kết quả tìm kiếm; catalog theo dõi phiên bản và trạng thái. |
| **Trả lời đúng ngôn ngữ người hỏi** | Ngôn ngữ câu trả lời theo ngôn ngữ câu hỏi. |
| **Tìm web chỉ qua SearXNG của bạn** | `search_web` chỉ gọi `SEARXNG_BASE_URL` đã cấu hình; nếu lỗi sẽ báo lỗi trung thực chứ không tự chuyển sang nhà cung cấp công khai. |
| **Kết nối hệ thống tài liệu khác** | Mọi MCP server (file system, wiki, drive...) đều thêm được vào tool registry. |
| **Có trí nhớ** | Bộ nhớ SQLite/FTS5, tuỳ chọn backend Mem0, Zep, LangMem, Supabase hoặc Notion; kèm skill và file persona chỉnh sửa được. |
| **Nhiều kênh giao tiếp** | CLI, Telegram, Discord, WhatsApp và voice. Gateway chỉ chuyển văn bản; chính sách truy xuất nằm trong runtime. |
| **Minh bạch quá trình** | Dashboard hiển thị hội thoại, các lần gọi tool, trace, trình quản lý tri thức (upload, xem chunk, thử truy xuất), so sánh mô hình và chạy eval. |

**Theo thiết kế, agent không làm:** ghi hoặc thực hiện hành động lên hệ thống của bạn.
Các tool tích hợp chỉ để truy xuất; văn bản truy xuất được và skill từ cộng đồng đều bị
coi là đầu vào không đáng tin.

## Kết quả hiện tại

Kết quả truy xuất trên kho tài liệu DENSO (workspace `denso`, 26 tài
liệu, 1.144 chunk), chia theo hai loại tài liệu trong kho.Thiết lập: Qdrant hybrid (dense `bge-m3` + vector
từ khoá, RRF, trọng số từ khoá 0,5), reranker `bge-reranker-v2-m3` trên 50 ứng viên.

| Nhóm câu hỏi | Số câu | Tìm đúng tài liệu | Văn bản đáp án nằm trong chunk trả về |
|---|---|---|---|
| **Tài liệu hành chính** (báo cáo chứng khoán và thường niên, quản trị, tuân thủ, nhân quyền, chuỗi cung ứng, sổ tay QA) | 46 | 40 (87%) | 37 (80%) |
| **Tài liệu kỹ thuật** (đặc tả AUTOSAR CAN Driver, sổ tay bugi DENSO) | 24 | 24 (100%) | 22 (92%) |
| **Tất cả, tiếng Anh, Nhật và Việt** | 70 | 64 (91%) | 59 (84%) |

Theo ngôn ngữ, tìm được đáp án: tiếng Nhật 29/30, tiếng Anh 22/24, tiếng Việt 8/16. Nhóm hành
chính: Nhật 27/28, Anh 4/6, Việt 6/12. Nhóm kỹ thuật: Anh 18/18, Nhật 2/2, Việt 2/4. Nhóm kỹ
thuật chính là `evals/documents/denso_qa_tech.jsonl` (cũng nằm trong
`denso_qa_multilingual.jsonl`, nơi ba dòng trên được tính ra), nên các dòng không phải ba lần
chạy độc lập. Độ trễ trung vị khoảng 1,2 giây mỗi truy vấn (p95 khoảng 1,4 giây), đo trên
dashboard với bộ tổng hợp 108 câu, cùng cấu hình.

"Văn bản đáp án nằm trong chunk trả về" chặt hơn "tìm đúng tài liệu". Các bộ câu hỏi nhỏ
(nhóm kỹ thuật chỉ có hai câu tiếng Nhật và bốn câu tiếng Việt) và do chính những người chọn
tài liệu viết, nên hãy đọc như xu hướng chứ không phải benchmark. Phương pháp và toàn bộ lịch
sử thử nghiệm: [docs/document-ingestion.md](docs/document-ingestion.md),
[evals/baselines/](evals/baselines/).

## Giới hạn đã biết

- Câu hỏi tiếng Việt là điểm yếu (tìm được đáp án 8/16 ở kết quả trên): reranker chấm
  điểm kém khi ghép câu hỏi tiếng Việt với chunk tiếng Anh. Tuỳ chọn đọc câu hỏi bằng
  tiếng Anh cho reranker (`TINI_RERANK_TRANSLATE=vi`, mặc định tắt, cần mô hình nhỏ chạy
  nhanh) đã được thử lại ngày 2026-10-11 và cho đúng kết quả 8/16, nên không nằm trong
  kết quả trên; lần chạy trước đó trên một bộ câu hỏi khác từng cho thấy có cải thiện.
- Các hàng của bảng trong PDF có lớp chữ được trích ra cách nhau một dấu cách, nên
  việc chia chunk theo hàng không áp dụng được.
- Chỉ PDF mới có số trang.
- Bộ đánh giá nhỏ và do chính tác giả kho tài liệu viết; quy mô vượt vài chục tài liệu
  chưa được đo.
- Các baseline trong `evals/baselines/` có trước khi gộp workspace `denso` và cần chạy
  lại.
- Hai bộ câu hỏi (`denso_qa_tech.jsonl` và `denso_qa_multilingual.jsonl`) trùng nhau;
  chạy cả hai sẽ đếm đôi 24 câu.

## Một câu hỏi được trả lời như thế nào

```text
kênh (CLI / Telegram / Discord / WhatsApp / voice)
   -> Tini.respond()            tini/runtime/session.py   chỉ dẫn, skill, bộ nhớ, lịch sử
   -> triage (graph, tuỳ chọn)  tini/graph/               chào hỏi -> mô hình nhỏ, câu hỏi tri thức -> vòng lặp đầy đủ
   -> vòng lặp agent            tini/loop/agent.py        mô hình <-> gọi tool
        -> search_documents     tini/tools/documents.py   hybrid search -> rerank -> chunk có trích dẫn
        -> search_web           tini/tools/search.py      chỉ SearXNG của bạn
        -> tool MCP             tini/tools/mcp_client.py
   -> câu trả lời kèm nguồn và số trang
```

Đường nạp tài liệu: tìm file -> trích văn bản (OCR cho trang scan, đánh dấu trang cho
PDF) -> chia chunk theo cấu trúc -> embedding -> upsert vào Qdrant (hoặc SQLite) cùng
metadata. Xem [docs/document-ingestion.md](docs/document-ingestion.md).

## Bắt đầu nhanh

Cần Python 3.11 trở lên.

```bash
git clone https://github.com/Thinha1/document-agent
cd document-agent
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
python -m tini
```

Cấu hình một nhà cung cấp chat trong `.env`:

```dotenv
TINI_PROVIDER=openai
OPENAI_API_KEY=your-key
```

Với server cục bộ, proxy hoặc dịch vụ tương thích OpenAI, dùng provider riêng:

```dotenv
TINI_PROVIDER=custom_openai
CUSTOM_OPENAI_API_KEY=your-key
CUSTOM_OPENAI_BASE_URL=https://provider.example/v1
TINI_MODEL=your-model-id
# Chỉ cho gateway định tuyến: tên provider đã cấu hình trong gateway.
CUSTOM_OPENAI_UPSTREAM_PROVIDER=openai
# Tuỳ chọn; mặc định bằng TINI_MODEL.
TINI_SMALL_MODEL=your-small-model-id
```

`custom_openai` tách biệt với provider `openai` chính thức. Với gateway định tuyến như
Bifrost, giữ `TINI_MODEL` là ID mô hình trần và đặt `CUSTOM_OPENAI_UPSTREAM_PROVIDER`;
giá trị gửi đi sẽ là `provider/model`. Endpoint phải hỗ trợ Chat Completions với function
tool calls. Dashboard thử gọi `GET <base-url>/models` để điền danh sách mô hình; vẫn có
thể nhập ID thủ công. DeepSeek và các provider có tên khác giữ cấu hình riêng.

### Toàn bộ stack tự host (Docker)

`docker-compose.yml` chạy agent cùng Qdrant (chỉ mục vector), Ollama (embedding
`bge-m3`), reranker cục bộ tuỳ chọn và MinIO tuỳ chọn:

```bash
cp .env.example .env
docker compose up -d --build
docker compose exec ollama ollama pull bge-m3          # chỉ lần đầu
docker compose --profile reranker up -d                # thêm bge-reranker (~2,2 GB lần khởi động đầu)
docker compose run --rm tini python -m tini ingest examples/denso_mock_docs --workspace denso
```

Dashboard ở `http://localhost:9000` (đổi cổng publish trong file
`docker-compose.override.yml`, đã git-ignore, nếu cổng bị chiếm). Card NVIDIA cũng được
bật qua file override này; xem chú thích trong `docker-compose.yml`.

## Nạp tài liệu

```bash
python -m tini ingest ./internal-docs                        # đệ quy; --no-recursive để chỉ lấy một cấp
python -m tini ingest ./internal-docs --workspace engineering
python -m tini documents                                     # liệt kê thư viện
python -m tini documents --search "incident recovery"
```

Để dùng giao diện, chạy `tini dashboard` và mở **Knowledge**: kéo thả file hoặc thư
mục, theo dõi các giai đoạn nạp thực tế, xem chunk và khoảng trang, thử truy xuất và
chạy eval truy xuất lẫn sinh câu trả lời. File upload được sao chép vào
`.tini/documents/<workspace>/`; xoá một mục khỏi thư viện vẫn giữ nguồn để có thể
index lại.

Các cài đặt quan trọng nhất (đều có trong `.env.example`; mỗi cài đặt được ghi cùng số
đo trong [docs/document-ingestion.md](docs/document-ingestion.md)):

```dotenv
TINI_DOCUMENT_INDEX=qdrant            # sqlite là mặc định nhẹ
TINI_DOCUMENT_HYBRID=1                # dense + từ khoá, hợp nhất bằng RRF
TINI_EMBEDDING_PROVIDER=openai_compatible   # ví dụ bge-m3 phục vụ bởi Ollama
TINI_RERANK_PROVIDER=tei              # bge-reranker-v2-m3 cục bộ; chỉ localhost
TINI_RERANK_TRANSLATE=vi              # đọc câu hỏi tiếng Việt bằng tiếng Anh cho reranker
TINI_TABLE_CHUNKING=1                 # lặp header bảng ở mọi chunk (mặc định bật)
```

Mô hình embedding được cấu hình tách khỏi mô hình chat, nên một mô hình chat tương
thích OpenAI vẫn có thể dùng embedding Gemini hoặc `bge-m3`. Hãy index lại tài liệu khi
đổi mô hình embedding, số chiều hoặc cài đặt chia chunk. Nếu endpoint embedding ngừng
hoạt động lúc truy vấn, backend SQLite vẫn chạy được nhờ FTS5.

## Tìm kiếm web

```dotenv
SEARXNG_BASE_URL=http://localhost:8080
```

Instance phải cho phép phản hồi JSON (`GET /search?q=...&format=json`).

## Nguồn tài liệu qua MCP

```bash
pip install -e '.[mcp]'
```

Tạo `.tini/mcp.json`:

```json
{
  "servers": [
    {
      "name": "docs",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/srv/internal-docs"]
    }
  ]
}
```

Các tool MCP được đặt tên `<server>_<tool>` và tham gia registry khi khởi động.

## Dashboard và định tuyến graph

Chạy `tini dashboard` (hoặc `make dashboard`). Dashboard hiển thị hội thoại, các lần gọi
truy xuất và tool, bộ nhớ, trace, so sánh mô hình, tình trạng tích hợp, topology triage
trực tiếp và trình quản lý **Knowledge**. Đặt `TINI_GRAPH_WORKFLOWS=1` để phân loại từng
tin nhắn trước: chào hỏi dùng mô hình nhỏ, câu hỏi tài liệu đi vào vòng lặp truy xuất
đầy đủ, và graph tự quay về vòng lặp đó nếu có lỗi.

## Đánh giá

```bash
make eval          # bộ deterministic, không cần mạng
make eval-judge    # kiểm tra bằng LLM-as-judge; cần provider đã cấu hình
make gate          # release gate
make lint          # Ruff
python scripts/document_baseline.py my-change --compare evals/baselines/<file>.json
python scripts/generation_eval.py run --label flash --provider <p> --model <m> --judge-provider <p2> --judge-model <m2>
```

Bộ câu hỏi truy xuất nằm trong `evals/documents/`, bộ sinh câu trả lời trong
`evals/generation/`, các ca dùng tool trong `evals/dataset.jsonl`. Một lỗi runtime cần
có ca hồi quy trong `evals/deterministic/`; chất lượng câu trả lời mang tính chủ quan
thuộc `evals/judge/`. Hãy dùng mô hình chấm khác với mô hình sinh.

## Dữ liệu runtime và bảo mật

Trạng thái cục bộ nằm trong `.tini/` (`state.db`, trace, cache OCR, file upload được
quản lý, `SOUL.md`, `mcp.json` tuỳ chọn); thông tin xác thực nằm trong `.env`. Cả hai
đều bị git-ignore. Không bao giờ commit tài liệu nội bộ. Văn bản truy xuất được và file
`SKILL.md` từ cộng đồng được coi là đầu vào không đáng tin; không mở rộng MCP roots và
không bao giờ để lộ thông tin xác thực cho mô hình. Xem [SECURITY.md](SECURITY.md) và
[CONTRIBUTING.md](CONTRIBUTING.md).

## Tài liệu

[docs/README.md](docs/README.md) liệt kê các tài liệu về kiến trúc, nạp tài liệu, đánh
giá và bộ nhớ.

## Giấy phép

MIT.
