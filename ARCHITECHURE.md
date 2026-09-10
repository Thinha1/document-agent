# Data Structures & State Management

## Tổng quan

Tini là agent tra cứu tài liệu nội bộ theo hướng local-first. Một request đi qua
gateway, lớp lắp context, retrieval gate, agent loop và model provider. Graph chỉ
bổ sung routing có cấu trúc quanh loop; dashboard và eval quan sát cùng các seam
đó mà không thay đổi hành vi runtime.

```text
Gateway → Session/context → optional triage graph → agent loop → model
                         ↘ retrieval gate          ↔ retrieval tools
                                                   ↘ answer + trace
```

## 1. Runtime Objects

### `Settings`

[`tini/config.py`](tini/config.py) đọc biến môi trường và giữ cấu hình runtime:
provider, model, giới hạn loop, thư mục `.tini/`, memory backend và cờ graph.
Endpoint tùy chỉnh dùng `TINI_PROVIDER=custom_openai`, `CUSTOM_OPENAI_BASE_URL` và
`TINI_MODEL`. OpenAI và DeepSeek vẫn có cấu hình riêng; `TINI_BASE_URL` chỉ là
fallback tương thích cũ.

### `Session`

[`tini/runtime/session.py`](tini/runtime/session.py) tạo working context cho một
turn từ bốn nguồn:

1. system instructions và `SOUL.md`;
2. procedural skills phù hợp với câu hỏi;
3. memory được retrieval gate chọn;
4. một cửa sổ chat gần nhất.

Lịch sử đầy đủ vẫn nằm trong SQLite; model chỉ nhận phần cần cho turn hiện tại.

### `Tool` và `ToolRegistry`

[`tini/tools/registry.py`](tini/tools/registry.py) lưu tool theo tên, xuất JSON
schema cho model và biến exception thành kết quả có thể quan sát. Registry mặc
định chỉ có `search_web`. Khi `.tini/mcp.json` tồn tại, MCP bridge đăng ký thêm
tool theo namespace `<server>_<tool>`.

## 2. Durable State

[`tini/db.py`](tini/db.py) quản lý `state.db`. Các bảng chính lưu:

- `chat_log`: message theo session và thời gian;
- `facts`: semantic memory có thể tìm bằng SQLite FTS5;
- `episodes`: sự kiện hoặc tóm tắt theo thời gian;
- metadata phục vụ consolidation và runtime.

SearXNG và MCP là nguồn retrieval bên ngoài, không được sao chép mặc định vào
SQLite. Trace và kết quả model comparison nằm trong artifact riêng dưới `.tini/`
để benchmark không làm bẩn conversation state.

## 3. Memory và Retrieval

Tini tách ba loại context dài hạn:

- **Semantic**: fact hoặc đoạn thông tin tìm theo relevance.
- **Episodic**: sự kiện đã xảy ra, có mốc thời gian.
- **Procedural**: hướng dẫn trong `SKILL.md`, chỉ nạp khi match.

Retrieval gate dùng small model để quyết định một turn có cần memory hay không.
Nếu cần, semantic và episodic store trả về top-k item; nếu không, agent tránh đưa
context không liên quan vào prompt. Consolidation chạy theo batch sau một số
exchange, rút thông tin bền vững từ chat log. Lỗi consolidation không xóa log gốc.

Đối với tài liệu nghiệp vụ, runtime có hai seam:

- `search_web` gọi đúng SearXNG instance do operator quản lý;
- MCP nối filesystem, vector database hoặc document service nội bộ.

Mỗi adapter phải giữ metadata nguồn đủ để câu trả lời có thể dẫn chứng và phải
trả lời rõ khi không tìm thấy bằng chứng.

## 4. Agent Loop

[`tini/loop/agent.py`](tini/loop/agent.py) thực hiện vòng lặp chuẩn:

1. gửi context và tool schemas cho model;
2. nếu model gọi tool, registry thực thi và đưa kết quả trở lại transcript;
3. model có thể tinh chỉnh truy vấn hoặc trả lời;
4. loop dừng khi có text cuối cùng hoặc chạm guardrail.

Loop phù hợp với RAG vì số lượt retrieve và thứ tự query không biết trước. Giới
hạn iteration, token và timeout ngăn một turn chạy vô hạn.

## 5. Graph State

Khi `TINI_GRAPH_WORKFLOWS=1`, workflow
[`triage.py`](tini/graph/workflows/triage.py) chạy trước loop:

```text
START → classify ─┬→ quick_reply → END
                  └→ full_agent  → END
```

Classifier ghi `route` và `triage_reason` vào state. Nhánh `full_agent` gọi đúng
loop hiện có, không tạo một implementation RAG thứ hai. Classifier hoặc graph
thất bại sẽ fail open về full loop để ưu tiên khả năng trả lời.

Graph engine dùng một dictionary làm blackboard. Node chỉ trả về các key cần
merge; router là hàm Python đọc state. `max_steps` và `max_visits` giới hạn cycle.
Thiết kế này đủ cho triage hiện tại và dễ stub trong deterministic tests.

## 6. Observability và Evaluation

Mỗi run phát event cho LLM call, tool call, retrieval decision và graph node.
JSONL trace luôn có sẵn; OpenTelemetry có thể xuất cùng dữ liệu sang collector.
Dashboard đọc các artifact này để hiển thị conversation, source/tool activity,
latency, token, cost và graph topology.

Evaluation được tách theo loại bằng chứng:

- `evals/deterministic/`: routing, tool name, arguments và state transition;
- `evals/dataset.jsonl`: các tình huống RAG có expected tool hoặc no-tool;
- `evals/judge/`: độ đúng, groundedness và chất lượng câu trả lời;
- Compare arena: chạy cùng prompt qua nhiều model trong home tạm biệt lập.

## 7. Một Request Hoàn Chỉnh

Với câu hỏi “quy trình restart worker là gì?”:

1. gateway chuyển text vào `Tini.respond()`;
2. triage chọn `full`;
3. session lắp history, skill và memory phù hợp;
4. model gọi `search_web` hoặc MCP document tool;
5. kết quả gồm title, snippet, URL hoặc metadata nguồn được trả lại model;
6. model tạo câu trả lời grounded, hoặc nói rõ không đủ bằng chứng;
7. message, usage và trace được lưu vào đúng artifact.

Điểm quan trọng là mỗi lớp có một trách nhiệm: graph chọn đường đi, loop điều
phối retrieval động, adapter truy cập nguồn, memory giữ context dài hạn, và ops
đo chất lượng.
