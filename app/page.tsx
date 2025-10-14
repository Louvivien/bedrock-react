// FULL FILE — app/page.tsx
"use client";

import DraggableSidePanel from "./components/DraggableSidePanel";
import { useEffect, useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";


type Msg = { role: "user" | "assistant"; content: string };

type DebugPayload = {
  prompt: string;
  sessionId: string;
  useOverrides: boolean;
  attrs: Record<string, string>;
  _debug_promptDefaultsWhenOff: Record<string, string> | null;
};

const DEFAULTS = {
  xBrand: process.env.NEXT_PUBLIC_DEFAULT_X_BRAND ?? "",
  xChannel: process.env.NEXT_PUBLIC_DEFAULT_X_CHANNEL ?? "AGENT_TOOL",
  lang: process.env.NEXT_PUBLIC_DEFAULT_LANG ?? "en",
  customerOuid:
    process.env.NEXT_PUBLIC_DEFAULT_CUSTOMER_OUID ??
    "",
};

export default function Home() {
  const [fatalError, setFatalError] = useState<string | null>(null);
  useEffect(() => {
    const onUnhandled = (e: ErrorEvent) => setFatalError(e.message || String(e));
    const onRejection = (e: PromiseRejectionEvent) =>
      setFatalError(e.reason?.message || String(e.reason));
    window.addEventListener("error", onUnhandled);
    window.addEventListener("unhandledrejection", onRejection);
    return () => {
      window.removeEventListener("error", onUnhandled);
      window.removeEventListener("unhandledrejection", onRejection);
    };
  }, []);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);


  // Session id persisted per tab
  const [sessionId, setSessionId] = useState<string>("");

  // ---- KEEP THIS EXACTLY (your requested snippet) ----
  // -- line before --
  useEffect(() => {
    try {
      // Persist within a tab so refreshes keep the same id
      const KEY = "bra_session_id";
      const existing =
        typeof window !== "undefined"
          ? window.sessionStorage.getItem(KEY)
          : null;
      if (existing) {
        setSessionId(existing);
      } else {
        const id = uuidv4();
        if (typeof window !== "undefined") {
          window.sessionStorage.setItem(KEY, id);
        }
        setSessionId(id);
      }
    } catch {
      // Fallback if sessionStorage unavailable
      setSessionId(uuidv4());
    }
  }, []);
  // -- line after --

  // Settings state (collapsed section)
  const [useOverrides, setUseOverrides] = useState(false);
  const [jwt, setJwt] = useState("");
  const [customerOuid, setCustomerOuid] = useState(DEFAULTS.customerOuid);
  const [xBrand, setXBrand] = useState(DEFAULTS.xBrand);
  const [xChannel, setXChannel] = useState(DEFAULTS.xChannel);
  const [lang, setLang] = useState(DEFAULTS.lang);
  const [maxTokens, setMaxTokens] = useState<number>(300);

    // timing
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const tickRef = useRef<number | null>(null);
  const t0Ref = useRef<number | null>(null);

  useEffect(() => {
  return () => {
    if (tickRef.current) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
  };
}, []);


  const [billingAccountOuid, setBA] = useState("");
  const [parentOuid, setParent] = useState("");
  const [offeringOuid, setOffering] = useState("");
  const [specOuid, setSpec] = useState("");
  const [msisdn, setMsisdn] = useState("");
  const [goodwillSizeGb, setSize] = useState<number>(2);
  const [goodwillReason, setReason] = useState("boosterOrPassRefund");

  const [lastPayload, setLastPayload] = useState<DebugPayload | null>(null);

  const chatRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    chatRef.current?.scrollTo(0, chatRef.current.scrollHeight);
  }, [messages, isStreaming]);

  const quickPrompts = [
    "🧾 Summarize account",
    "💳 Check billing & payments",
    "📊 Analyze consumption; recommend plan/booster",
    "🚨 Spot risks; suggest actions",
    "🎟️ Review tickets; propose next steps",
  ];

  function buildAttrs(): Record<string, string> {
    if (!useOverrides) return {};
    const attrs: Record<string, string> = {};
    if (jwt) attrs.jwt = jwt;
    if (customerOuid) attrs.customerOuid = customerOuid;

    // call context
    if (lang) attrs.lang = lang;
    if (xBrand) attrs.xBrand = xBrand;
    if (xChannel) attrs.xChannel = xChannel;

    

    // goodwill (optional)
    if (billingAccountOuid) attrs.billingAccountOuid = billingAccountOuid;
    if (parentOuid) attrs.parentOuid = parentOuid;
    if (offeringOuid) attrs.offeringOuid = offeringOuid;
    if (specOuid) attrs.specOuid = specOuid;
    if (msisdn) attrs.msisdn = msisdn;

    // always include goodwill sizing if overrides enabled
    attrs.goodwillSizeGb = String(Math.max(1, Number(goodwillSizeGb || 2)));
    if (goodwillReason) attrs.goodwillReason = goodwillReason;

    return attrs;
  }

  async function send(prompt: string) {
    if (!prompt.trim() || isStreaming || !sessionId) return;

    // add user message
    setMessages((m) => [...m, { role: "user", content: prompt }]);
    setInput("");

    // pre-create assistant slot to stream into
    const idx = messages.length + 1;
    setMessages((m) => [...m, { role: "assistant", content: "" }]);
    setIsStreaming(true);

    // start timing
    t0Ref.current = performance.now();
    setElapsedMs(0);
    if (tickRef.current) clearInterval(tickRef.current);
    tickRef.current = window.setInterval(() => {
      if (t0Ref.current != null) {
        setElapsedMs(performance.now() - t0Ref.current);
      }
    }, 100);



    const attrs = buildAttrs();

    // Build baseline prompt attrs when overrides are OFF
    const baselinePromptAttrs = !useOverrides
      ? {
          xBrand: DEFAULTS.xBrand,
          xChannel: DEFAULTS.xChannel,
          lang: DEFAULTS.lang,
        }
      : null;

    const payload: DebugPayload = {
      prompt,
      sessionId,
      useOverrides,
      attrs,
      _debug_promptDefaultsWhenOff: baselinePromptAttrs,
    };
    setLastPayload(payload);

    try {
      // -- line before --
      const res = await fetch("/api/invoke", {
        method: "POST",
        body: JSON.stringify({
          prompt,
          sessionId,
          useOverrides,
          attrs,
        }),
      });
      const reader = res.body?.getReader();
      const decoder = new TextDecoder();

      let assistant = "";
      while (reader) {
        const { value, done } = await reader.read();
        if (done) break;
        assistant += decoder.decode(value);
        setMessages((m) => {
          const copy = [...m];
          copy[idx] = { role: "assistant", content: assistant };
          return copy;
        });
      }
      // -- line after --
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setMessages((m) => {
        const copy = [...m];
        copy[idx] = { role: "assistant", content: `⚠️ Error: ${msg}` };
        return copy;
      });
    } finally {
      
        if (tickRef.current) {
          clearInterval(tickRef.current);
          tickRef.current = null;
        }
        if (t0Ref.current != null) {
          setElapsedMs(performance.now() - t0Ref.current);
          t0Ref.current = null;
        }
        setIsStreaming(false);
    }
  }

  return (
  <DraggableSidePanel
    initialWidth={480}
    minWidth={320}
    maxWidth={900}
    side="right" // <- handle on the LEFT edge of the panel (visible when panel is on the right)
    className="border-neutral-200"
  >
    <main className="min-h-screen bg-neutral-50 text-neutral-900">
      <div style={{ position: "relative", zIndex: 0, pointerEvents: "auto" }}>
      <div className="mx-auto max-w-3xl px-4 py-6">
        <h1 className="text-2xl font-semibold mb-1">🤖 AI Assistant</h1>

        {fatalError && (
          <div className="mb-3 rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-800">
            Client error: {fatalError}
          </div>
        )}

        <p className="text-sm text-neutral-600 mb-4">
          Talk to your Bedrock Agent. 
        </p>

        {/* SETTINGS (collapsed) */}
        <details className="mb-4 rounded-lg border border-neutral-200 bg-white">
          <summary className="cursor-pointer list-none px-4 py-3 font-medium">
            ⚙️ Settings 
          </summary>
          <div className="px-4 pb-4">
            <p className="text-xs text-neutral-500 mb-3">
              Toggle overrides to send your params as session attributes; otherwise minimal defaults are sent as prompt attributes.
            </p>

            <label className="flex items-center gap-2 mb-3">
              <input
                type="checkbox"
                checked={useOverrides}
                onChange={(e) => setUseOverrides(e.target.checked)}
              />
              <span>Use these overrides in calls (sessionAttributes &amp; promptSessionAttributes)</span>
            </label>

            <hr className="my-3" />
            <h3 className="font-semibold mb-2">🔐 Auth</h3>
            <div className="mb-3">
              <label className="block text-sm mb-1">JWT (include “Bearer …”)</label>
              <input
                className="w-full rounded border px-3 py-2"
                value={jwt}
                onChange={(e) => setJwt(e.target.value)}
                placeholder="Bearer eyJhbGciOiJIUzUxMiJ9.…"
              />
            </div>

            <hr className="my-3" />
            <h3 className="font-semibold mb-2">👤 Customer context</h3>
            <div className="mb-3">
              <label className="block text-sm mb-1">customerOuid</label>
              <input
                className="w-full rounded border px-3 py-2"
                value={customerOuid}
                onChange={(e) => setCustomerOuid(e.target.value)}
                placeholder={DEFAULTS.customerOuid}
              />
            </div>

            <hr className="my-3" />
            <h3 className="font-semibold mb-2">🌐 Call context</h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div>
                <label className="block text-sm mb-1">X-Brand</label>
                <input
                  className="w-full rounded border px-3 py-2"
                  value={xBrand}
                  onChange={(e) => setXBrand(e.target.value)}
                  placeholder="DEMO-DEMO"
                />
              </div>
              <div>
                <label className="block text-sm mb-1">X-Channel</label>
                <input
                  className="w-full rounded border px-3 py-2"
                  value={xChannel}
                  onChange={(e) => setXChannel(e.target.value)}
                  placeholder="AGENT_TOOL"
                />
              </div>
              <div>
                <label className="block text-sm mb-1">Accept-Language</label>
                <select
                  className="w-full rounded border px-3 py-2"
                  value={lang}
                  onChange={(e) => setLang(e.target.value)}
                >
                  <option value="en">en</option>
                  <option value="fr">fr</option>
                </select>
              </div>
            </div>

            <hr className="my-3" />
            <h3 className="font-semibold mb-2">🎁 Goodwill parameters</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-sm mb-1">parentOuid</label>
                <input
                  className="w-full rounded border px-3 py-2"
                  value={parentOuid}
                  onChange={(e) => setParent(e.target.value)}
                  placeholder="Subscription OUID"
                />
              </div>
              <div>
                <label className="block text-sm mb-1">specOuid</label>
                <input
                  className="w-full rounded border px-3 py-2"
                  value={specOuid}
                  onChange={(e) => setSpec(e.target.value)}
                  placeholder="8B3C73498520F7048BC00F449DBAE447"
                />
              </div>
              <div>
                <label className="block text-sm mb-1">msisdn</label>
                <input
                  className="w-full rounded border px-3 py-2"
                  value={msisdn}
                  onChange={(e) => setMsisdn(e.target.value)}
                  placeholder="0613423341"
                />
              </div>
              <div>
                <label className="block text-sm mb-1">billingAccountOuid</label>
                <input
                  className="w-full rounded border px-3 py-2"
                  value={billingAccountOuid}
                  onChange={(e) => setBA(e.target.value)}
                  placeholder="B5C34D432E67E1543F02255AACB06057"
                />
              </div>
              <div>
                <label className="block text-sm mb-1">offeringOuid</label>
                <input
                  className="w-full rounded border px-3 py-2"
                  value={offeringOuid}
                  onChange={(e) => setOffering(e.target.value)}
                  placeholder="47F2CE64A772B870D62F5BD19ED02196"
                />
              </div>
              <div>
                <label className="block text-sm mb-1">goodwillSizeGb</label>
                <input
                  type="number"
                  min={1}
                  max={1000}
                  className="w-full rounded border px-3 py-2"
                  value={goodwillSizeGb}
                  onChange={(e) => setSize(Number(e.target.value))}
                />
              </div>
              <div className="sm:col-span-2">
                <label className="block text-sm mb-1">goodwillReason</label>
                <input
                  className="w-full rounded border px-3 py-2"
                  value={goodwillReason}
                  onChange={(e) => setReason(e.target.value)}
                  placeholder="boosterOrPassRefund"
                />
              </div>
            </div>
          </div>
        </details>

        {/* Quick prompts */}
        <div className="mb-3 grid grid-cols-1 sm:grid-cols-3 gap-2">
          {quickPrompts.map((q) => (
          <button
            key={q}
            type="button"
            className="rounded-md border bg-white px-3 py-2 text-sm hover:bg-neutral-50"
            onClick={() => send(q)}
            disabled={isStreaming}
          >
                        {q}
            </button>
          ))}
        </div>

        {/* Debug */}
        <details className="mb-4 rounded border border-neutral-200 bg-white">
          <summary className="cursor-pointer list-none px-4 py-2 text-sm font-medium">
            🧪 Debug — payload being sent
          </summary>
          <pre className="px-4 py-3 text-xs overflow-x-auto">
            {JSON.stringify(
              {
                sessionId,
                useOverrides,
                payload: lastPayload,
              },
              null,
              2
            )}
          </pre>
        </details>

    {/* Chat box */}
    {/* Status / loading */}
{isStreaming && (
  <div className="mb-2 flex items-center gap-2 text-sm text-neutral-700">
    <span
      className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-neutral-300 border-t-neutral-900"
      aria-hidden
    />
    <span>Processing… {elapsedMs != null ? (elapsedMs / 1000).toFixed(1) : "0.0"}s</span>
  </div>
)}

{!isStreaming && elapsedMs != null && (
  <div className="mb-2 text-xs text-neutral-500">
    Last response took {(elapsedMs / 1000).toFixed(2)}s
  </div>
)}
    <div
      ref={chatRef}
      className="h-[50vh] overflow-auto rounded-lg border border-neutral-200 bg-white p-4"
    >
      {messages.map((m, i) => (
        <div key={i} className="mb-3">
          <div className="text-xs text-neutral-500 mb-1">
            {m.role === "user" ? "You" : "Assistant"}
          </div>
          {m.role === "assistant" ? (
          <div className="prose prose-sm max-w-none [&_pre]:bg-neutral-50 [&_pre]:p-3 [&_pre]:rounded-md [&_code]:font-mono">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                a: (props) => <a {...props} target="_blank" rel="noreferrer" />,
              }}
            >
              {m.content}
            </ReactMarkdown>
          </div>
          ) : (
            <div className="whitespace-pre-wrap">{m.content}</div>
          )}
        </div>
      ))}

      {messages.length === 0 && (
        <div className="text-sm text-neutral-500">
          Ask something or click a quick prompt above.
        </div>
      )}
    </div>

    {/* Input */}
    <form
          className="mt-3 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <input
            className="flex-1 rounded border px-3 py-2"
            value={input}
            placeholder="Ask me anything…"
            onChange={(e) => setInput(e.target.value)}
            disabled={isStreaming}
          />
          <button
            type="submit"
            className="rounded bg-black px-4 py-2 text-white disabled:opacity-50"
            disabled={isStreaming || !input.trim() || !sessionId}
          >
            Send
          </button>
        </form>
      </div>
    </div>
    </main>
      </DraggableSidePanel>

  );
}
