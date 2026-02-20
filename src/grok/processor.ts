import type { GrokSettings, GlobalSettings } from "../settings";

type GrokNdjson = Record<string, unknown>;

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

async function readWithTimeout(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  ms: number,
): Promise<ReadableStreamReadResult<Uint8Array> | { timeout: true }> {
  if (ms <= 0) return { timeout: true };
  return Promise.race([
    reader.read(),
    sleep(ms).then(() => ({ timeout: true }) as const),
  ]);
}

function makeChunk(
  id: string,
  created: number,
  model: string,
  content: string,
  finish_reason?: "stop" | "error" | null,
): string {
  const payload: Record<string, unknown> = {
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [
      {
        index: 0,
        delta: content ? { role: "assistant", content } : {},
        finish_reason: finish_reason ?? null,
      },
    ],
  };
  return `data: ${JSON.stringify(payload)}\n\n`;
}

function makeDone(): string {
  return "data: [DONE]\n\n";
}

function toImgProxyUrl(globalCfg: GlobalSettings, origin: string, path: string): string {
  const baseUrl = (globalCfg.base_url ?? "").trim() || origin;
  return `${baseUrl}/images/${path}`;
}

function buildVideoTag(src: string): string {
  return `<video src="${src}" controls="controls" width="500" height="300"></video>\n`;
}

function buildVideoPosterPreview(videoUrl: string, posterUrl?: string): string {
  const href = String(videoUrl || "").replace(/"/g, "&quot;");
  const poster = String(posterUrl || "").replace(/"/g, "&quot;");
  if (!href) return "";
  if (!poster) return `<a href="${href}" target="_blank" rel="noopener noreferrer">${href}</a>\n`;
  return `<a href="${href}" target="_blank" rel="noopener noreferrer" style="display:inline-block;position:relative;max-width:100%;text-decoration:none;">
  <img src="${poster}" alt="video" style="max-width:100%;height:auto;border-radius:12px;display:block;" />
  <span style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;">
    <span style="width:64px;height:64px;border-radius:9999px;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;">
      <span style="width:0;height:0;border-top:12px solid transparent;border-bottom:12px solid transparent;border-left:18px solid #fff;margin-left:4px;"></span>
    </span>
  </span>
</a>\n`;
}

function buildVideoHtml(args: { videoUrl: string; posterUrl?: string; posterPreview: boolean }): string {
  if (args.posterPreview) return buildVideoPosterPreview(args.videoUrl, args.posterUrl);
  return buildVideoTag(args.videoUrl);
}

function base64UrlEncode(input: string): string {
  const bytes = new TextEncoder().encode(input);
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function encodeAssetPath(raw: string): string {
  try {
    const u = new URL(raw);
    // Keep full URL (query etc.) to avoid lossy pathname-only encoding (some URLs may encode the real path in query).
    return `u_${base64UrlEncode(u.toString())}`;
  } catch {
    const p = raw.startsWith("/") ? raw : `/${raw}`;
    return `p_${base64UrlEncode(p)}`;
  }
}

function normalizeGeneratedAssetUrls(input: unknown): string[] {
  if (!Array.isArray(input)) return [];

  const out: string[] = [];
  for (const v of input) {
    if (typeof v !== "string") continue;
    const s = v.trim();
    if (!s) continue;
    if (s === "/") continue;

    try {
      const u = new URL(s);
      if (u.pathname === "/" && !u.search && !u.hash) continue;
    } catch {
      // ignore (path-style strings are allowed)
    }

    out.push(s);
  }

  return out;
}

function formatToolCall(tag: string, content: string): string {
  let data: Record<string, unknown>;
  try {
    data = JSON.parse(content) as Record<string, unknown>;
  } catch {
    return "";
  }
  if (tag === "function_call") {
    const name = typeof data.name === "string" ? data.name : "";
    let args: Record<string, unknown> = {};
    if (typeof data.arguments === "string") {
      try { args = JSON.parse(data.arguments) as Record<string, unknown>; } catch { /* ignore */ }
    } else if (data.arguments && typeof data.arguments === "object") {
      args = data.arguments as Record<string, unknown>;
    }
    if (name === "web_search" || name === "search") {
      const query = typeof args.query === "string" ? args.query : "";
      return query ? `\n🔍 搜索: ${query}\n` : `\n🔍 ${name}\n`;
    } else if (name === "browse" || name === "browse_web") {
      const url = typeof args.url === "string" ? args.url : "";
      return url ? `\n🌐 浏览: ${url}\n` : `\n🌐 ${name}\n`;
    } else if (name === "code_execution") {
      return "\n🖥️ 执行代码\n";
    } else if (name) {
      return `\n🔧 ${name}\n`;
    }
  } else if (tag === "raw_function_result") {
    if (typeof data === "object" && data !== null) {
      if (data.error || data.success === false) return "\n❌ 执行失败\n";
    }
    return "\n✅ 执行成功\n";
  }
  return "";
}

export function createOpenAiStreamFromGrokNdjson(
  grokResp: Response,
  opts: {
    cookie: string;
    settings: GrokSettings;
    global: GlobalSettings;
    origin: string;
    onFinish?: (result: { status: number; duration: number }) => Promise<void> | void;
  },
): ReadableStream<Uint8Array> {
  const { settings, global, origin } = opts;
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();

  const id = `chatcmpl-${crypto.randomUUID()}`;
  const created = Math.floor(Date.now() / 1000);

  const filteredTags = (settings.filtered_tags ?? "")
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);
  const showThinking = settings.show_thinking !== false;
  const showToolCalls = settings.show_tool_calls !== false;

  const firstTimeoutMs = Math.max(0, (settings.stream_first_response_timeout ?? 30) * 1000);
  const chunkTimeoutMs = Math.max(0, (settings.stream_chunk_timeout ?? 120) * 1000);
  const totalTimeoutMs = Math.max(0, (settings.stream_total_timeout ?? 600) * 1000);

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      const body = grokResp.body;
      if (!body) {
        controller.enqueue(encoder.encode(makeChunk(id, created, "grok-4-mini-thinking-tahoe", "Empty response", "error")));
        controller.enqueue(encoder.encode(makeDone()));
        controller.close();
        return;
      }

      const reader = body.getReader();
      const startTime = Date.now();
      let finalStatus = 200;
      let lastChunkTime = startTime;
      let firstReceived = false;

      let currentModel = "grok-4-mini-thinking-tahoe";
      let isImage = false;
      let isThinking = false;
      let thinkingFinished = false;
      let videoProgressStarted = false;
      let lastVideoProgress = -1;
      let toolCallBuffer = "";
      let toolCallTag: string | null = null;

      let buffer = "";

      const flushStop = () => {
        controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, "", "stop")));
        controller.enqueue(encoder.encode(makeDone()));
      };

      try {
        // eslint-disable-next-line no-constant-condition
        while (true) {
          const now = Date.now();
          const elapsed = now - startTime;
          if (!firstReceived && elapsed > firstTimeoutMs) {
            flushStop();
            if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
            controller.close();
            return;
          }
          if (totalTimeoutMs > 0 && elapsed > totalTimeoutMs) {
            flushStop();
            if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
            controller.close();
            return;
          }
          const idle = now - lastChunkTime;
          if (firstReceived && idle > chunkTimeoutMs) {
            flushStop();
            if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
            controller.close();
            return;
          }

          const perReadTimeout = Math.min(
            firstReceived ? chunkTimeoutMs : firstTimeoutMs,
            totalTimeoutMs > 0 ? Math.max(0, totalTimeoutMs - elapsed) : Number.POSITIVE_INFINITY,
          );

          const res = await readWithTimeout(reader, perReadTimeout);
          if ("timeout" in res) {
            flushStop();
            if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
            controller.close();
            return;
          }

          const { value, done } = res;
          if (done) break;
          if (!value) continue;
          buffer += decoder.decode(value, { stream: true });

          let idx: number;
          while ((idx = buffer.indexOf("\n")) !== -1) {
            const line = buffer.slice(0, idx).trim();
            buffer = buffer.slice(idx + 1);
            if (!line) continue;

            let data: GrokNdjson;
            try {
              data = JSON.parse(line) as GrokNdjson;
            } catch {
              continue;
            }

            firstReceived = true;
            lastChunkTime = Date.now();

            const err = (data as any).error;
            if (err?.message) {
              finalStatus = 500;
              controller.enqueue(
                encoder.encode(makeChunk(id, created, currentModel, `Error: ${String(err.message)}`, "stop")),
              );
              controller.enqueue(encoder.encode(makeDone()));
              if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
              controller.close();
              return;
            }

            const grok = (data as any).result?.response;
            if (!grok) continue;

            const userRespModel = grok.userResponse?.model;
            if (typeof userRespModel === "string" && userRespModel.trim()) currentModel = userRespModel.trim();

            // Video generation stream
            const videoResp = grok.streamingVideoGenerationResponse;
            if (videoResp) {
              const progress = typeof videoResp.progress === "number" ? videoResp.progress : 0;
              const videoUrl = typeof videoResp.videoUrl === "string" ? videoResp.videoUrl : "";
              const thumbUrl = typeof videoResp.thumbnailImageUrl === "string" ? videoResp.thumbnailImageUrl : "";

              if (progress > lastVideoProgress) {
                lastVideoProgress = progress;
                if (showThinking) {
                  let msg = "";
                  if (!videoProgressStarted) {
                    msg = `<think>视频已生成${progress}%\n`;
                    videoProgressStarted = true;
                  } else if (progress < 100) {
                    msg = `视频已生成${progress}%\n`;
                  } else {
                    msg = `视频已生成${progress}%</think>\n`;
                  }
                  controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, msg)));
                }
              }

              if (videoUrl) {
                const videoPath = encodeAssetPath(videoUrl);
                const src = toImgProxyUrl(global, origin, videoPath);

                let poster: string | undefined;
                if (thumbUrl) {
                  const thumbPath = encodeAssetPath(thumbUrl);
                  poster = toImgProxyUrl(global, origin, thumbPath);
                }

                controller.enqueue(
                  encoder.encode(
                    makeChunk(
                      id,
                      created,
                      currentModel,
                      buildVideoHtml({
                        videoUrl: src,
                        posterPreview: settings.video_poster_preview === true,
                        ...(poster ? { posterUrl: poster } : {}),
                      }),
                    ),
                  ),
                );
              }
              continue;
            }

            if (grok.imageAttachmentInfo) isImage = true;
            const rawToken = grok.token;

            if (isImage) {
              const modelResp = grok.modelResponse;
              if (modelResp) {
                const urls = normalizeGeneratedAssetUrls(modelResp.generatedImageUrls);
                if (urls.length) {
                  const linesOut: string[] = [];
                  for (const u of urls) {
                    const imgPath = encodeAssetPath(u);
                    const imgUrl = toImgProxyUrl(global, origin, imgPath);
                    linesOut.push(`![Generated Image](${imgUrl})`);
                  }
                  controller.enqueue(
                    encoder.encode(makeChunk(id, created, currentModel, linesOut.join("\n"), "stop")),
                  );
                  controller.enqueue(encoder.encode(makeDone()));
                  if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
                  controller.close();
                  return;
                }
              } else if (typeof rawToken === "string" && rawToken) {
                controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, rawToken)));
              }
              continue;
            }

            // 提取专家ID、消息标签和思考状态
            const rolloutId = typeof grok.rolloutId === "string" ? grok.rolloutId : "";
            const prefix = rolloutId ? `[${rolloutId}] ` : "";
            const messageTag = typeof grok.messageTag === "string" ? grok.messageTag : "";
            const currentIsThinking = Boolean(grok.isThinking);

            // 处理工具调用（结构化字段，Expert 模式）
            if (messageTag === "function_call" && grok.functionCall && typeof grok.functionCall === "object") {
              if (showThinking) {
                const fc = grok.functionCall as Record<string, unknown>;
                const toolName = typeof fc.name === "string" ? fc.name : "";
                let toolArgs: Record<string, unknown> = {};
                if (typeof fc.arguments === "string") {
                  try { toolArgs = JSON.parse(fc.arguments) as Record<string, unknown>; } catch { /* ignore */ }
                } else if (fc.arguments && typeof fc.arguments === "object") {
                  toolArgs = fc.arguments as Record<string, unknown>;
                }
                if (!isThinking) {
                  controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, `<think>\n`)));
                  isThinking = true;
                }
                if (toolName === "web_search") {
                  const query = typeof toolArgs.query === "string" ? toolArgs.query : "";
                  if (query) controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, `${prefix}🔍 搜索: ${query}\n`)));
                } else if (toolName === "web_browse") {
                  const url = typeof toolArgs.url === "string" ? toolArgs.url : "";
                  if (url) controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, `${prefix}🌐 浏览: ${url}\n`)));
                } else if (toolName === "chatroom_send") {
                  const to = typeof toolArgs.to === "string" ? toolArgs.to : "";
                  const msg = typeof toolArgs.message === "string" ? toolArgs.message : "";
                  if (msg) {
                    const shortMsg = msg.length > 100 ? `${msg.slice(0, 100)}...` : msg;
                    controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, `${prefix}💬 → ${to}: ${shortMsg}\n`)));
                  }
                } else if (toolName) {
                  controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, `${prefix}🔧 ${toolName}\n`)));
                }
              }
              continue;
            }

            // 处理工具执行结果（结构化字段，Expert 模式）
            if (messageTag === "raw_function_result" && (grok.webSearchResults || grok.codeExecutionResult)) {
              if (showThinking) {
                if (!isThinking) {
                  controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, `<think>\n`)));
                  isThinking = true;
                }
                const webResults = grok.webSearchResults;
                if (webResults) {
                  let resultsList: unknown[] = [];
                  if (Array.isArray(webResults)) {
                    resultsList = webResults;
                  } else if (typeof webResults === "object" && webResults !== null) {
                    const r = (webResults as Record<string, unknown>).results;
                    if (Array.isArray(r)) resultsList = r;
                  }
                  if (resultsList.length > 0) {
                    controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, `${prefix}📄 找到 ${resultsList.length} 条结果\n`)));
                  }
                }
                const codeResult = grok.codeExecutionResult;
                if (codeResult && typeof codeResult === "object") {
                  const cr = codeResult as Record<string, unknown>;
                  const exitCode = typeof cr.exitCode === "number" ? cr.exitCode : -1;
                  if (exitCode === 0) {
                    const stdout = typeof cr.stdout === "string" ? cr.stdout.trim() : "";
                    if (stdout) {
                      const shortOut = stdout.length > 200 ? `${stdout.slice(0, 200)}...` : stdout;
                      controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, `${prefix}✅ 执行成功: ${shortOut}\n`)));
                    } else {
                      controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, `${prefix}✅ 执行成功\n`)));
                    }
                  } else {
                    const stderr = typeof cr.stderr === "string" ? cr.stderr.trim() : "";
                    const lastLine = stderr ? stderr.split("\n").at(-1) ?? "未知错误" : "未知错误";
                    controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, `${prefix}❌ 执行失败: ${lastLine}\n`)));
                  }
                }
              }
              continue;
            }

            // Text chat stream
            if (Array.isArray(rawToken)) continue;
            if (typeof rawToken !== "string" || !rawToken) continue;
            let token = rawToken;

            if (filteredTags.some((t) => token.includes(t))) continue;

            if (thinkingFinished && currentIsThinking) continue;

            // Flush tool call buffer when tag changes
            if (toolCallTag && messageTag !== toolCallTag) {
              if (showToolCalls && toolCallBuffer) {
                const formatted = formatToolCall(toolCallTag, toolCallBuffer);
                if (formatted) controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, formatted)));
              }
              toolCallBuffer = "";
              toolCallTag = null;
            }

            // Accumulate tool call tokens (legacy / no structured field)
            if (messageTag === "function_call" || messageTag === "raw_function_result") {
              if (showToolCalls) {
                toolCallTag = messageTag;
                toolCallBuffer += token;
              }
              continue;
            }

            if (grok.toolUsageCardId && grok.webSearchResults?.results && Array.isArray(grok.webSearchResults.results)) {
              if (currentIsThinking) {
                if (showThinking) {
                  let appended = "";
                  for (const r of grok.webSearchResults.results) {
                    const title = typeof r.title === "string" ? r.title : "";
                    const url = typeof r.url === "string" ? r.url : "";
                    const preview = typeof r.preview === "string" ? r.preview.replace(/\n/g, "") : "";
                    appended += `\n- [${title}](${url} \"${preview}\")`;
                  }
                  token += `${appended}\n`;
                } else {
                  continue;
                }
              } else {
                continue;
              }
            }

            let content = token;
            if (messageTag === "header") content = `\n\n${token}\n\n`;

            let shouldSkip = false;
            if (!isThinking && currentIsThinking) {
              if (showThinking) content = `<think>\n${prefix}${content}`;
              else shouldSkip = true;
            } else if (isThinking && !currentIsThinking) {
              if (showThinking) content = `\n</think>\n${content}`;
              thinkingFinished = true;
            } else if (currentIsThinking && !showThinking) {
              shouldSkip = true;
            } else if (currentIsThinking && showThinking && prefix) {
              content = `${prefix}${content}`;
            }

            if (!shouldSkip) controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, content)));
            isThinking = currentIsThinking;
          }
        }

        // Flush any pending tool call buffer
        if (toolCallTag && toolCallBuffer && showToolCalls) {
          const formatted = formatToolCall(toolCallTag, toolCallBuffer);
          if (formatted) controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, formatted)));
        }

        controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, "", "stop")));
        controller.enqueue(encoder.encode(makeDone()));
        if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
        controller.close();
      } catch (e) {
        finalStatus = 500;
        controller.enqueue(
          encoder.encode(
            makeChunk(id, created, currentModel, `处理错误: ${e instanceof Error ? e.message : String(e)}`, "error"),
          ),
        );
        controller.enqueue(encoder.encode(makeDone()));
        if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
        controller.close();
      } finally {
        try {
          reader.releaseLock();
        } catch {
          // ignore
        }
      }
    },
  });
}

export async function parseOpenAiFromGrokNdjson(
  grokResp: Response,
  opts: { cookie: string; settings: GrokSettings; global: GlobalSettings; origin: string; requestedModel: string },
): Promise<Record<string, unknown>> {
  const { global, origin, requestedModel, settings } = opts;
  const text = await grokResp.text();
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);

  let content = "";
  let model = requestedModel;
  for (const line of lines) {
    let data: GrokNdjson;
    try {
      data = JSON.parse(line) as GrokNdjson;
    } catch {
      continue;
    }

    const err = (data as any).error;
    if (err?.message) throw new Error(String(err.message));

    const grok = (data as any).result?.response;
    if (!grok) continue;

    const videoResp = grok.streamingVideoGenerationResponse;
    if (videoResp?.videoUrl && typeof videoResp.videoUrl === "string") {
      const videoPath = encodeAssetPath(videoResp.videoUrl);
      const src = toImgProxyUrl(global, origin, videoPath);

      let poster: string | undefined;
      if (typeof videoResp.thumbnailImageUrl === "string" && videoResp.thumbnailImageUrl) {
        const thumbPath = encodeAssetPath(videoResp.thumbnailImageUrl);
        poster = toImgProxyUrl(global, origin, thumbPath);
      }

      content = buildVideoHtml({
        videoUrl: src,
        posterPreview: settings.video_poster_preview === true,
        ...(poster ? { posterUrl: poster } : {}),
      });
      model = requestedModel;
      break;
    }

    const modelResp = grok.modelResponse;
    if (!modelResp) continue;
    if (typeof modelResp.error === "string" && modelResp.error) throw new Error(modelResp.error);

    if (typeof modelResp.model === "string" && modelResp.model) model = modelResp.model;
    if (typeof modelResp.message === "string") content = modelResp.message;

    const rawUrls = modelResp.generatedImageUrls;
    const urls = normalizeGeneratedAssetUrls(rawUrls);
    if (urls.length) {
      for (const u of urls) {
        const imgPath = encodeAssetPath(u);
        const imgUrl = toImgProxyUrl(global, origin, imgPath);
        content += `\n![Generated Image](${imgUrl})`;
      }
      break;
    }

    // If upstream emits placeholder/empty generatedImageUrls in intermediate frames, keep scanning.
    if (Array.isArray(rawUrls)) continue;

    // For normal chat replies, the first modelResponse is enough.
    break;
  }

  return {
    id: `chatcmpl-${crypto.randomUUID()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
      },
    ],
    usage: null,
  };
}
