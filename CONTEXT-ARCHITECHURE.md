# Context Architecture

## Tổng quan

Context của Tini được lắp theo nhu cầu của từng turn, thay vì đưa toàn bộ lịch sử
và mọi tài liệu vào model. Mục tiêu là giữ prompt nhỏ, có nguồn gốc rõ ràng và có
thể đánh giá độc lập.

```text
instructions + selected skills + retrieved memory + recent history + tool schemas
                                      ↓
                               provider request
```

## 1. System và Conversation Context

[`tini/runtime/session.py`](tini/runtime/session.py) là assembly point. System
context gồm hành vi nền của assistant, `SOUL.md` tùy chỉnh và procedural skill đã
match. Conversation context chỉ lấy số turn gần nhất theo `TINI_HISTORY_TURNS`.
Chat cũ không mất: nó vẫn ở `state.db` và có thể quay lại qua retrieval.

Tool context được sinh trực tiếp từ `ToolRegistry`, vì vậy model chỉ nhìn thấy
tool thực sự đăng ký. Runtime mặc định có `search_web`; MCP tools chỉ xuất hiện
khi operator cấu hình `.tini/mcp.json`.

## 2. Retrieval Selection

Retrieval gate trả lời một câu hẹp: turn hiện tại có cần memory hay không. Quyết
định này dùng small model và fail open khi không parse được kết quả. Khi gate bật,
semantic và episodic store trả về top-k item; khi gate tắt, chúng không chiếm
context hay làm lệch câu trả lời xã giao.

Document retrieval là bước động bên trong agent loop:

- `search_web` truy vấn SearXNG do operator self-host;
- MCP tools truy vấn filesystem, vector store hoặc document API nội bộ;
- model có thể đọc kết quả, đổi query và retrieve thêm trước khi trả lời.

Adapter nên giữ title, URI, collection hoặc metadata tương đương. Không có kết
quả phải được biểu diễn như thiếu bằng chứng, không phải một câu trả lời phỏng đoán.

## 3. Procedural Context

Các file `SKILL.md` là hướng dẫn theo tác vụ, không phải knowledge base. Loader
chỉ nạp skill khi metadata match câu hỏi. Skill nên mô tả cách dùng retrieval tool,
quy tắc dẫn nguồn và edge case; nội dung nghiệp vụ nên nằm trong document store để
có thể index, cập nhật và benchmark riêng.

## 4. Loop Transcript

Một turn trong [`tini/loop/agent.py`](tini/loop/agent.py) tích lũy transcript tạm:

1. model nhận context ban đầu và schemas;
2. tool call được thực thi;
3. tool result được nối vào transcript;
4. model tiếp tục retrieve hoặc tạo final answer.

Transcript chi tiết phục vụ đúng run hiện tại. Persistent chat chỉ lưu message và
summary cần thiết, tránh biến raw tool payload thành lịch sử vĩnh viễn.

## 5. Graph Context

Triage graph có state nhỏ và tách khỏi document context:

```text
{message} → classify → {route, triage_reason}
                      ├─ quick: small-model reply
                      └─ full: normal session + retrieval loop
```

Nhánh quick không nạp memory hay gọi retrieval. Nhánh full không nhận một bản
context dựng sẵn từ graph; nó gọi assembly path chuẩn để giữ một source of truth.
Graph fail open về full path nếu classifier hoặc engine lỗi.

Nếu sau này dashboard cần hiển thị “chọn corpus → retrieve → rerank → answer”,
pipeline đó nên trở thành workflow riêng với state key rõ ràng như `corpus_ids`,
`chunks`, `ranked_chunks` và `citations`. Hiện tại dashboard chỉ hiển thị triage
topology và các retrieval/tool event đã thực sự xảy ra.

## 6. Persistence và Isolation

`.tini/state.db` giữ chat và memory. `.tini/traces/` cùng compare history giữ dữ
liệu quan sát. Compare arena tạo home tạm cho từng contestant để model này không
đọc state của model khác. `.tini/mcp.json` là cấu hình operator và không được đưa
vào prompt như nội dung tài liệu.

Consolidation đọc phần chat chưa xử lý, tạo fact/episode rồi đánh dấu tiến độ. Nếu
model hoặc backend lỗi, source log vẫn còn để retry.

## 7. Portability và Guardrails

Provider adapters chuẩn hóa Anthropic-style và OpenAI-compatible requests. Context
assembly không phụ thuộc provider; chỉ wire format, endpoint, model id và usage
parsing thay đổi. Base URL theo provider ngăn endpoint của một dịch vụ bị dùng nhầm
khi chuyển `TINI_PROVIDER`.

Các guardrail cần giữ khi mở rộng RAG:

- giới hạn recent history, retrieval top-k, snippet length và loop iterations;
- coi retrieved text là dữ liệu không tin cậy, không phải system instruction;
- scope MCP server tới document root tối thiểu;
- trace source selection nhưng không ghi credential;
- đánh giá retrieval correctness riêng với answer quality;
- giữ context assembly ở một chỗ để dashboard, CLI và gateway có cùng hành vi.

Kiến trúc này cho phép thay retriever, vector store hoặc model mà không tạo thêm
một luồng prompt song song khó kiểm thử.
