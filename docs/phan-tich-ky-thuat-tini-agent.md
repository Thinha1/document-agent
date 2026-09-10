# Tini-agent: đọc codebase trong 10 phút

Tini là một agent tra cứu tài liệu nội bộ, không phải bộ sưu tập automation tool.
Kiến trúc giữ ba việc tách biệt: lắp context, điều phối retrieval và đo chất lượng.

## 1. Assembly Root

[`tini/app.py`](../tini/app.py) tạo `Settings`, database, memory facade, model
client và `ToolRegistry`. Registry mặc định chỉ đăng ký `search_web`; nếu có
`.tini/mcp.json`, MCP bridge thêm các retrieval tool theo namespace server.

Gateway CLI, web hoặc messaging cuối cùng đều gọi cùng `Tini.respond()`. Vì vậy
không có một RAG implementation riêng cho dashboard và một implementation khác
cho terminal.

## 2. Một Turn Chạy Như Thế Nào?

[`tini/runtime/session.py`](../tini/runtime/session.py) lắp system instructions,
skill phù hợp, memory được chọn và recent history. Retrieval gate dùng small model
để tránh nạp memory không liên quan vào lời chào hoặc small talk.

Nếu graph được bật, triage classifier chọn:

```text
quick → small-model reply
full  → normal context assembly + agent loop
```

Graph lỗi sẽ đi vào full path. Đây là “fail open”: mất tối ưu latency nhưng không
mất khả năng trả lời.

## 3. Agent Loop là Lõi RAG

[`tini/loop/agent.py`](../tini/loop/agent.py) chạy model với tool schemas. Khi model
gọi tool, registry thực thi, nối kết quả vào transcript rồi gọi model lại. Chu trình
tiếp tục tới final answer hoặc guardrail.

Thiết kế loop phù hợp với truy vấn tài liệu vì không thể biết trước cần một hay
nhiều lần retrieve. Graph không thay thế loop; graph chỉ nên giữ những stage ổn
định như chọn corpus, rerank hoặc citation check.

## 4. Retrieval Surface

[`tini/tools/search.py`](../tini/tools/search.py) gọi `SEARXNG_BASE_URL/search` với
JSON format. Operator quyết định engine và nguồn nào SearXNG được phép truy vấn.
Không có fallback sang public search provider.

[`tini/tools/mcp_client.py`](../tini/tools/mcp_client.py) là seam cho knowledge
base nội bộ: filesystem, vector store hoặc document API. MCP server nên được scope
đến root tối thiểu và trả metadata nguồn đủ để model dẫn chứng.

Khi không cấu hình nguồn hoặc không có kết quả, tool trả lỗi/empty state rõ ràng.
Đây là điều kiện cần để benchmark đo groundedness thay vì thưởng cho hallucination.

## 5. State và Memory

SQLite là default cho chat log, semantic facts và episodic memory. FTS5 cung cấp
retrieval local không cần service. Các adapter Supabase, Notion hoặc memory arena
giữ cùng interface và nằm sau optional extras.

Consolidation chạy theo batch, rút fact/episode từ chat chưa xử lý. Nếu model lỗi,
chat log gốc vẫn còn. Working memory chỉ lấy cửa sổ gần nhất; dữ liệu cũ quay lại
qua retrieval khi có liên quan.

## 6. Provider Compatibility

[`tini/loop/models.py`](../tini/loop/models.py) chuẩn hóa hai wire format chính:
Anthropic-style và OpenAI-compatible. `TINI_PROVIDER` chọn adapter và model.
Proxy, local server hoặc vendor tương thích dùng provider `custom_openai` với
`CUSTOM_OPENAI_BASE_URL`; DeepSeek có cấu hình riêng để endpoint không rò giữa
các provider.

## 7. Graph và Dashboard

[`tini/graph/workflows/triage.py`](../tini/graph/workflows/triage.py) là workflow
shipped duy nhất. Dashboard lấy topology từ `Graph.describe()` và hiển thị event
thực, tránh một sơ đồ thủ công bị lệch code.

Dashboard hiện cho thấy triage route và retrieval/tool calls. Một flow chọn tài
liệu chi tiết chỉ nên xuất hiện sau khi runtime có workflow thật với state như
`corpus_ids`, `chunks`, `ranked_chunks` và `citations`.

## 8. Evaluation

`evals/deterministic/` kiểm tra routing, schema, tool arguments, provider settings
và state transition mà không dùng mạng. `evals/dataset.jsonl` chứa case RAG có
expected tool hoặc no-tool. `evals/judge/` chấm groundedness và chất lượng trả lời.

Compare arena chạy cùng prompt qua nhiều model trong home tạm biệt lập, ghi latency,
token, cost, tool calls và score vào history riêng. OpenTelemetry có thể xuất trace
sang hệ thống quan sát như Phoenix hoặc Langfuse mà không buộc runtime phụ thuộc
vào một dashboard cụ thể.

## 9. Quy Tắc Khi Mở Rộng

- Thêm document source qua MCP trước khi thêm core tool.
- Giữ retrieval text là dữ liệu không tin cậy, không phải instruction.
- Tách deterministic retrieval metrics khỏi model-scored answer quality.
- Dùng graph cho control flow ổn định; dùng loop cho query refinement động.
- Thêm regression test cho mọi lỗi runtime đã sửa.

Nếu giữ đúng các seam này, có thể thay model, retriever, vector store hoặc tracing
backend mà không phải viết lại agent.
