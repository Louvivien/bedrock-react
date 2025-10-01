Got it — here’s a fully generic version (no mention of triPica anywhere):

---

# AI Assistant

A Next.js React app that adds an AI side-panel to an agent desktop. It streams answers, renders Markdown, and calls your backend (via a server route) to talk to an AI agent and your domain APIs.

---

## ✨ Features

* Draggable, collapsible **AI side panel**
* **Streaming** assistant responses with Markdown rendering
* **Session** stickiness (per-tab `sessionId`) to keep context
* **Overrides**: inject headers/JWT for tenant/brand/channel without changing code
* Quick-prompt buttons (account summary, billing, usage analysis, risk spotting, ticket review)
* Debug overlay (shows prompt, resolved attrs, and whether overrides are applied)

---

## 🧱 Tech Stack

* **Next.js (React, App Router)**
* **TypeScript**
* **Server route** at `/api/invoke` (Node runtime) to call your AI agent
* **AWS Bedrock Agent Runtime** client on the server route (streaming)
* **React Markdown** + GFM for formatting answers

---

## 📦 Prerequisites

* Node.js 18+
* An AWS account & credentials (for the server route to call your Bedrock Agent)
* A configured Bedrock **Agent ID** (and alias)
* Your domain backend reachable by the agent (for tools/functions), if applicable

---

## 🔐 Environment Variables

Create `.env.local` at the project root:

```bash
# --- App defaults (UI) ---
NEXT_PUBLIC_DEFAULT_X_BRAND=DEMO-DEMO
NEXT_PUBLIC_DEFAULT_X_CHANNEL=AGENT_TOOL
NEXT_PUBLIC_DEFAULT_LANG=en
NEXT_PUBLIC_DEFAULT_CUSTOMER_OUID=

# --- AWS / Bedrock (server route) ---
AWS_REGION=eu-west-1
BEDROCK_AGENT_ID=xxxxxxxx
BEDROCK_AGENT_ALIAS_ID=xxxxxxxx

# Optional: tighten server timeouts, etc.
INVOCATION_STREAM_CHUNK_TIMEOUT_MS=120000
```

**Notes**

* The UI shows “Use overrides” and lets you pass headers like brand/channel/lang and (optionally) a JWT to the server route. The route forwards these via **Bedrock session attributes** so the agent has the right context.
* Keep secrets (JWTs) off the client whenever possible. Prefer server-side injection.

---

## 🚀 Getting Started

```bash
# Install
npm install

# Dev
npm run dev

# Build
npm run build

# Start (prod)
npm start
```

Open [http://localhost:3000](http://localhost:3000).

---

## 🗂️ Project Structure (high level)

```
app/
  page.tsx                 # Main UI (side panel, chat stream, debug)
  api/
    invoke/
      route.ts             # Server route (Node runtime) calls Bedrock Agent
app/components/
  DraggableSidePanel.tsx   # Side panel wrapper
  ...                      # Other UI bits
```

---

## 🔄 How It Works

1. **User types a prompt** in the side-panel.
2. The **client POSTs** to `/api/invoke` with:

   * `prompt`, `sessionId`
   * `useOverrides` flag + optional attrs (brand/channel/lang/JWT/customer id)
3. The **server route**:

   * Builds a **Bedrock session state** (includes headers/JWT equivalents as session attributes when overrides are ON)
   * Calls **Bedrock Agent Runtime** with streaming enabled
   * Pipes **streamed tokens** back to the browser
4. The client **renders tokens** as they arrive and formats with Markdown.

---

## 🧭 Session & Overrides

* A `sessionId` (UUID) is created per browser tab so your agent keeps context (conversation + any cached customer snapshot) across turns.
* Toggle **“Use overrides”** to push tenant/brand/channel/lang (and optionally a JWT) into the conversation. This is useful when your agent calls domain APIs that expect headers like `X-Brand` or `X-Channel`.

---

## 🧪 Quick Prompts (examples)

* **Summarize account**
* **Check billing & payments**
* **Analyze usage**
* **Spot risks**
* **Review tickets**

These are intentionally short and generic so the agent can decide which tools to call and return a concise, operator-ready answer.

---

## 🧩 Backend/API Notes 
If your agent calls your domain endpoints, you’ll typically rely on a **customer snapshot/DTO** for most account questions. A reference spec would define:

* **Get customer data (DTO)** — returns a consolidated customer view; avoid re-calling if it’s already in context.
* **Example action (e.g., add goodwill / create order)** — idempotent action with a clear, short response.

> The agent/Lambda layer commonly injects `X-Channel`, `X-Brand`, and authorization so you don’t need to pass them from the UI for standard flows.

---

## 🛡️ Security

* **Never** expose long-lived JWTs in the browser; prefer short-lived, scoped tokens or purely server-side injection.
* Treat **session attributes** as sensitive; they can affect which APIs your agent can reach.
* Log only what’s needed; redact tokens and PII in server logs.

---

## 🧰 Troubleshooting

**403 on protected calls**

* Missing/invalid JWT or wrong brand/channel. Enable **Use overrides** and provide `X-Brand`, `X-Channel`, or have the server route inject a valid JWT.

**Timeouts**

* Increase the server route timeout (framework or reverse proxy).
* Check downstream API performance; heavy snapshots can take time.

**React hydration warning**

* Avoid rendering values that differ between server and client at first paint (e.g., `Date.now()` without snapshotting). Ensure Markdown output mounts only after client hydration.

**No streaming**

* Confirm the server route uses the **Node runtime** (not Edge) and that Bedrock streaming is enabled in the SDK call.

---

## ♻️ Deployment

* Any Node-capable host (Vercel/EC2/containers).
* Ensure **AWS credentials** and **BEDROCK_AGENT_* envs** are available at runtime on the server.
* If behind a proxy, enable **HTTP chunked transfer** so streams aren’t buffered.

---

## 🤝 Contributing

* Keep UI changes small and testable (the debug overlay is your friend).
* When adding quick prompts, keep them **under 5 words** and **action-oriented**.
* Prefer **server-side** integrations for anything involving secrets.

---

## 📄 License

MIT

---

## 🧠 AWS Bedrock Agent Lambda

The server route calls a Bedrock **Agent**. That agent can be fronted by a lightweight **Lambda** (“override lambda”) that standardizes auth, headers, and tool calls to your domain APIs.

### What the Lambda does

* **Header/JWT injection**
  Resolves tenant context and auth once, then forwards to backend tools (e.g., `X-Brand`, `X-Channel`, `Accept-Language`, `Authorization`).

* **Safe defaults**
  Falls back to demo values if not provided by the UI (e.g., default customer ID).

* **Snapshot/DTO reuse**
  Encourages the agent to reuse an in-context snapshot instead of re-fetching it every turn.

* **Latency/timeout control**
  Short-circuits long calls; returns clear errors for the UI to surface.

### Typical environment variables (Lambda)

```bash
# Domain backend
BASE_URL=https://api-demo.example.com
TIMEOUT_SEC=10

X_BRAND=DEMO-DEMO
X_CHANNEL=agent-tool
DEFAULT_CUSTOMER_ID=              # optional fallback

# Auth handling
STATIC_JWT=                       # optional static Bearer token
ALWAYS_USE_ENV_TOKEN=true         # if true, ignore UI JWT
VERIFY_AFTER_ACTION=false         # optional post-action verification
```

### Session attributes mapping

The client optionally sends “overrides.” The Lambda turns those into consistent headers & session state:

```json
{
  "sessionAttributes": {
    "xBrand": "DEMO-DEMO",
    "xChannel": "AGENT_TOOL",
    "lang": "en",
    "customerId": "85DA...E86",
    "jwt": "Bearer eyJ..."
  }
}
```

At tool time, the Lambda builds the outbound request:

```
GET /api/private/v1/agent/customer/customerDto/{id}
X-Brand: DEMO-DEMO
X-Channel: AGENT_TOOL
Accept-Language: en
Authorization: Bearer <token>
```

### Handler lifecycle (simplified)

1. Parse event (intent, tool route, session attrs)
2. Decide **auth mode** (UI JWT vs `STATIC_JWT`), apply headers
3. Build URL + query, enforce **timeouts**
4. Call backend; map errors (403/404/5xx) to concise messages
5. Optionally **verify** results after sensitive actions
6. Return compact JSON/body for the agent to summarize

### Troubleshooting (Lambda)

* **403 on protected routes** → Missing/invalid JWT or wrong brand/channel. Confirm overrides or enable `ALWAYS_USE_ENV_TOKEN`.
* **Timeout** → Raise `TIMEOUT_SEC` or narrow the snapshot/request.
* **Unexpected re-fetching** → Ensure the orchestration prompt tells the agent to **reuse** snapshots already in context.

---

## 🗣️ Advanced Prompt System (orchestration + post-processing)

A layered prompt strategy so answers are **actionable, safe, and terse** for operators.

### Layers

1. **System / Guardrails**
   Role, tone, formatting rules (concise, operator-ready, third person). Safety & privacy reminders.

2. **Orchestration Prompt** (router / planner)

   * Extract facts from the user turn + conversation summary
   * Prefer **one** snapshot read; reuse in memory when available
   * Parallelize independent tool calls when beneficial
   * Never invent parameters (IDs, dates, brands). Ask only if absolutely required

3. **Tool I/O Instructions**

   * Input schema, required headers, success/failure mapping
   * Encourage **idempotence** for actions
   * Truncate or redact large payloads before returning to the model

4. **Post-Processor** (final answer shaper)

   * Convert hidden tool outputs into a clean **operator summary**
   * **Third person only** (no “you”)
   * Output compact bullet points and next-best actions
   * Hide tool names, endpoints, and implementation details

### Memory & token budgeting

* Maintain a short **conversation summary** (customer context, last snapshot timestamp, plan, unpaid invoices, etc.)
* Stay under a strict token cap by preferring **summary + last 3–4 turns** instead of full history
* Evict stale blobs (full JSON) after extracting only needed fields



### Orchestration skeleton

```
- Parse user goal and required facts
- If a fresh snapshot is not in memory, fetch it once; otherwise reuse
- If an action is requested, verify eligibility and limits
- Produce a 5–8 line operator summary + clear next steps
- Keep tools invisible in the final message; no implementation details
```

---
