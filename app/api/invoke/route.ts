// FULL FILE — app/api/invoke/route.ts
import { NextRequest } from "next/server";
import {
  BedrockAgentRuntimeClient,
  InvokeAgentCommand,
  InvokeAgentCommandInput,
  InvokeAgentCommandOutput,
} from "@aws-sdk/client-bedrock-agent-runtime";

export const runtime = "nodejs"; // ensure Node runtime (not Edge)

type Payload = {
  prompt: string;
  sessionId: string;
  useOverrides: boolean;
  attrs?: Record<string, string>;
};

type SessionState = {
  sessionAttributes?: Record<string, string>;
  promptSessionAttributes?: Record<string, string>;
};

export async function POST(req: NextRequest) {
  const { prompt, sessionId, useOverrides, attrs = {} } = (await req.json()) as Payload;

  const region = process.env.AWS_REGION || "eu-west-1";
  const client = new BedrockAgentRuntimeClient({ region });

  // Build session state (sticky + turn) or minimal baseline when OFF
  // -- line before --
  const sessionState: SessionState = {};
  // -- line after --
  if (useOverrides && Object.keys(attrs).length > 0) {
    sessionState.sessionAttributes = attrs;       // sticky across turns
    sessionState.promptSessionAttributes = attrs; // visible to current turn/tools
  } else {
    sessionState.promptSessionAttributes = {
      xBrand: process.env.DEFAULT_X_BRAND ?? "DEMO-DEMO",
      xChannel: process.env.DEFAULT_X_CHANNEL ?? "AGENT_TOOL",
      lang: process.env.DEFAULT_LANG ?? "en",
    };
  }

  const streamFinal =
    (process.env.STREAM_FINAL_RESPONSE ?? "true").toLowerCase() === "true";
  const guardrailInterval = Number(process.env.APPLY_GUARDRAIL_INTERVAL ?? "50");

  const input: InvokeAgentCommandInput = {
    agentId: process.env.AGENT_ID!,
    agentAliasId: process.env.AGENT_ALIAS_ID!,
    sessionId,
    inputText: prompt,
    enableTrace: false,
    streamingConfigurations: {
      applyGuardrailInterval: guardrailInterval,
      streamFinalResponse: streamFinal,
    },
    sessionState,
  };

  const cmd = new InvokeAgentCommand(input);

  // Stream Bedrock chunks -> HTTP response
  const textDecoder = new TextDecoder();

  const stream = new ReadableStream({
    async start(controller) {
      try {
        // -- line before --
        const resp: InvokeAgentCommandOutput = await client.send(cmd);
        for await (const event of resp.completion ?? []) {
          if (event.chunk?.bytes) {
            controller.enqueue(textDecoder.decode(event.chunk.bytes));
          }
        }
        // -- line after --
      } catch (err: unknown) {
        // -- line before --
        const message = err instanceof Error ? err.message : String(err);
        // -- line after --
        controller.enqueue(`\n[error] ${message}`);
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-store",
      "Transfer-Encoding": "chunked",
    },
  });
}
