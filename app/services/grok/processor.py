"""
OpenAI 响应格式处理器
"""
import time
import uuid
import random
import html
import re
import orjson
from typing import Any, AsyncGenerator, Optional, AsyncIterable, List

from app.core.config import get_config
from app.core.logger import logger
from app.services.grok.assets import DownloadService


ASSET_URL = "https://assets.grok.com/"

# Compiled patterns for strict grok:render block stripping (streaming only)
_GROK_RENDER_OPEN_RE = re.compile(r"<grok:render\b[^>]*>", re.IGNORECASE)
_GROK_RENDER_CLOSE_TAG = "</grok:render>"
_GROK_RENDER_TAG_PREFIX = "<grok:render"
# Maximum size of the cross-token lookahead buffer for grok:render tag detection
_GROK_RENDER_BUF_MAX = 8192


def _find_partial_tag_suffix(text: str) -> int:
    """Return the length of the longest suffix of text that is a prefix of '<grok:render'.

    Args:
        text: The string to examine.

    Returns:
        Length of the trailing partial tag prefix (0 if none found).
    """
    tag = _GROK_RENDER_TAG_PREFIX
    max_check = min(len(tag), len(text))
    for length in range(max_check, 0, -1):
        if text.endswith(tag[:length]):
            return length
    return 0


def _format_tool_call(tag: str, content: str) -> str:
    """格式化工具调用信息"""
    try:
        data = orjson.loads(content)
    except Exception:
        return ""

    if tag == "function_call":
        name = data.get("name", "")
        args = data.get("arguments", {})
        if isinstance(args, str):
            try:
                args = orjson.loads(args)
            except Exception:
                args = {}
        if name in ("web_search", "search"):
            query = args.get("query", "")
            return f"\n🔍 搜索: {query}\n" if query else f"\n🔍 {name}\n"
        elif name in ("browse", "browse_web"):
            url = args.get("url", "")
            return f"\n🌐 浏览: {url}\n" if url else f"\n🌐 {name}\n"
        elif name == "code_execution":
            return "\n🖥️ 执行代码\n"
        elif name:
            return f"\n🔧 {name}\n"
    elif tag == "raw_function_result":
        if isinstance(data, dict):
            if data.get("error") or data.get("success") is False:
                return "\n❌ 执行失败\n"
        return "\n✅ 执行成功\n"
    return ""


def _build_video_poster_preview(video_url: str, thumbnail_url: str = "") -> str:
    """将 <video> 替换为可点击的 Poster 预览图（用于前端展示）"""
    safe_video = html.escape(video_url or "", quote=True)
    safe_thumb = html.escape(thumbnail_url or "", quote=True)

    if not safe_video:
        return ""

    if not safe_thumb:
        return f'<a href="{safe_video}" target="_blank" rel="noopener noreferrer">{safe_video}</a>'

    return f'''<a href="{safe_video}" target="_blank" rel="noopener noreferrer" style="display:inline-block;position:relative;max-width:100%;text-decoration:none;">
  <img src="{safe_thumb}" alt="video" style="max-width:100%;height:auto;border-radius:12px;display:block;" />
  <span style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;">
    <span style="width:64px;height:64px;border-radius:9999px;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;">
      <span style="width:0;height:0;border-top:12px solid transparent;border-bottom:12px solid transparent;border-left:18px solid #fff;margin-left:4px;"></span>
    </span>
  </span>
</a>'''


def _parse_tool_usage_card(token: str) -> tuple[str, dict]:
    """Extract tool name/args from tool_usage_card token payload."""
    if not token:
        return "", {}
    tool_name = ""
    tool_args: dict = {}

    tool_match = re.search(r"<xai:tool_name>([^<]+)</xai:tool_name>", token)
    if tool_match:
        tool_name = tool_match.group(1).strip()

    args_match = re.search(r"<!\[CDATA\[(.+?)\]\]>", token, re.DOTALL)
    if args_match:
        try:
            parsed = orjson.loads(args_match.group(1))
            if isinstance(parsed, dict):
                tool_args = parsed
        except Exception:
            tool_args = {}
    return tool_name, tool_args


class BaseProcessor:
    """基础处理器"""
    
    def __init__(self, model: str, token: str = ""):
        self.model = model
        self.token = token
        self.created = int(time.time())
        self.app_url = get_config("app.app_url", "")
        self._dl_service: Optional[DownloadService] = None

    def _get_dl(self) -> DownloadService:
        """获取下载服务实例（复用）"""
        if self._dl_service is None:
            self._dl_service = DownloadService()
        return self._dl_service

    async def close(self):
        """释放下载服务资源"""
        if self._dl_service:
            await self._dl_service.close()
            self._dl_service = None

    async def process_url(self, path: str, media_type: str = "image") -> str:
        """处理资产 URL"""
        # 处理可能的绝对路径
        if path.startswith("http"):
            from urllib.parse import urlparse
            path = urlparse(path).path
            
        if not path.startswith("/"):
            path = f"/{path}"

        # Invalid root path is not a displayable image URL.
        if path in {"", "/"}:
            return ""

        # Always materialize to local cache endpoint so callers don't rely on
        # direct assets.grok.com access (often blocked without upstream cookies).
        dl_service = self._get_dl()
        await dl_service.download(path, self.token, media_type)
        local_path = f"/v1/files/{media_type}{path}"
        if self.app_url:
            return f"{self.app_url.rstrip('/')}{local_path}"
        return local_path
            
    def _sse(self, content: str = "", role: str = None, finish: str = None) -> str:
        """构建 SSE 响应 (StreamProcessor 通用)"""
        if not hasattr(self, 'response_id'):
            self.response_id = None
        if not hasattr(self, 'fingerprint'):
            self.fingerprint = ""
            
        delta = {}
        if role:
            delta["role"] = role
            delta["content"] = ""
        elif content:
            delta["content"] = content
        
        chunk = {
            "id": self.response_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.fingerprint if hasattr(self, 'fingerprint') else "",
            "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]
        }
        return f"data: {orjson.dumps(chunk).decode()}\n\n"


class StreamProcessor(BaseProcessor):
    """流式响应处理器"""
    
    def __init__(self, model: str, token: str = "", think: bool = None):
        super().__init__(model, token)
        self.response_id: Optional[str] = None
        self.fingerprint: str = ""
        self.think_opened: bool = False
        self.role_sent: bool = False
        self.filter_tags = get_config("grok.filter_tags", [])
        self.image_format = get_config("app.image_format", "url")
        self.show_tool_calls: bool = get_config("grok.show_tool_calls", True)
        self.is_thinking: bool = False
        self.thinking_finished: bool = False
        self._tool_buf: str = ""
        self._tool_tag: Optional[str] = None
        self._last_rollout_id: str = ""
        # State for cross-token <grok:render ...>...</grok:render> block stripping
        self._in_grok_render_block: bool = False
        self._grok_render_pending: str = ""

        if think is None:
            self.show_think = get_config("grok.thinking", False)
        else:
            self.show_think = think
    
    async def process(self, response: AsyncIterable[bytes]) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        try:
            async for line in response:
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                
                result = data.get("result", {})
                if not isinstance(result, dict):
                    result = {}
                resp = result.get("response", {})
                if not isinstance(resp, dict):
                    resp = {}
                
                # 元数据
                if (llm := resp.get("llmInfo")) and not self.fingerprint:
                    self.fingerprint = llm.get("modelHash", "")
                if rid := resp.get("responseId"):
                    self.response_id = rid
                
                # 首次发送 role
                if not self.role_sent:
                    yield self._sse(role="assistant")
                    self.role_sent = True
                
                # 图像生成进度
                if img := resp.get("streamingImageGenerationResponse"):
                    if self.show_think:
                        if not self.think_opened:
                            yield self._sse("<think>\n")
                            self.think_opened = True
                        idx = img.get('imageIndex', 0) + 1
                        progress = img.get('progress', 0)
                        yield self._sse(f"正在生成第{idx}张图片中，当前进度{progress}%\n")
                    continue
                
                # modelResponse
                if mr := resp.get("modelResponse"):
                    if self.think_opened and self.show_think:
                        if msg := mr.get("message"):
                            yield self._sse(msg + "\n")
                        yield self._sse("</think>\n")
                        self.think_opened = False
                    
                    # 处理生成的图片
                    for url in mr.get("generatedImageUrls", []):
                        parts = url.split("/")
                        img_id = parts[-2] if len(parts) >= 2 else "image"
                        
                        if self.image_format == "base64":
                            dl_service = self._get_dl()
                            base64_data = await dl_service.to_base64(url, self.token, "image")
                            if base64_data:
                                yield self._sse(f"![{img_id}]({base64_data})\n")
                            else:
                                final_url = await self.process_url(url, "image")
                                yield self._sse(f"![{img_id}]({final_url})\n")
                        else:
                            final_url = await self.process_url(url, "image")
                            yield self._sse(f"![{img_id}]({final_url})\n")
                    
                    if (meta := mr.get("metadata", {})).get("llm_info", {}).get("modelHash"):
                        self.fingerprint = meta["llm_info"]["modelHash"]
                    continue
                
                # 提取专家ID、消息标签和思考状态
                rollout_id = ""
                for obj in (resp, result, data):
                    rid = obj.get("rolloutId") if isinstance(obj, dict) else ""
                    if isinstance(rid, str) and rid:
                        rollout_id = rid
                        break
                if rollout_id:
                    self._last_rollout_id = rollout_id
                elif self._last_rollout_id and (self.is_thinking or self.think_opened):
                    rollout_id = self._last_rollout_id
                prefix = f"[{rollout_id}] " if rollout_id else ""
                message_tag = resp.get("messageTag") or result.get("messageTag") or ""
                is_thinking = bool(
                    resp.get("isThinking", result.get("isThinking", data.get("isThinking", False)))
                )
                token = resp.get("token")
                if token is None:
                    token = result.get("token")
                function_call = resp.get("functionCall")
                if not function_call:
                    function_call = result.get("functionCall")
                web_results_data = resp.get("webSearchResults")
                if web_results_data is None:
                    web_results_data = result.get("webSearchResults")
                code_result = resp.get("codeExecutionResult")
                if code_result is None:
                    code_result = result.get("codeExecutionResult")
                tool_usage_card_id = resp.get("toolUsageCardId") or result.get("toolUsageCardId")
                if get_config("grok.debug_stream_fields", False):
                    logger.info(
                        "Grok stream fields",
                        extra={
                            "rolloutId": rollout_id,
                            "messageTag": message_tag,
                            "isThinking": is_thinking,
                            "hasFunctionCall": bool(function_call),
                            "hasToolUsageCardId": bool(tool_usage_card_id),
                            "hasWebSearchResults": bool(web_results_data),
                            "hasCodeExecutionResult": bool(code_result),
                            "tokenLen": len(token) if isinstance(token, str) else 0,
                        },
                    )
                    if self.show_think:
                        debug_parts = [
                            f"tag={message_tag or '-'}",
                            f"rollout={rollout_id or '-'}",
                            f"fn={1 if function_call else 0}",
                            f"card={1 if tool_usage_card_id else 0}",
                            f"web={1 if web_results_data else 0}",
                            f"code={1 if code_result else 0}",
                            f"think={1 if is_thinking else 0}",
                        ]
                        if not self.think_opened:
                            yield self._sse("<think>\n")
                            self.think_opened = True
                        yield self._sse(f"[debug] {' '.join(debug_parts)}\n")

                # 处理工具调用（结构化字段，Expert 模式）
                if message_tag == "function_call" and function_call:
                    if self.show_think:
                        tool_name = function_call.get("name", "") if isinstance(function_call, dict) else ""
                        tool_args = function_call.get("arguments", {}) if isinstance(function_call, dict) else {}
                        if isinstance(tool_args, str):
                            try:
                                tool_args = orjson.loads(tool_args)
                            except Exception:
                                tool_args = {}
                        if not self.think_opened:
                            yield self._sse("<think>\n")
                            self.think_opened = True
                        if tool_name == "web_search":
                            query = tool_args.get("query", "")
                            if query:
                                yield self._sse(f"{prefix}🔍 搜索: {query}\n")
                        elif tool_name == "web_browse":
                            url = tool_args.get("url", "")
                            if url:
                                yield self._sse(f"{prefix}🌐 浏览: {url}\n")
                        elif tool_name == "chatroom_send":
                            to = tool_args.get("to", "")
                            msg = tool_args.get("message", "")
                            if msg:
                                short_msg = msg[:100] + ("..." if len(msg) > 100 else "")
                                yield self._sse(f"{prefix}💬 → {to}: {short_msg}\n")
                        elif tool_name:
                            yield self._sse(f"{prefix}🔧 {tool_name}\n")
                    continue

                # 处理工具执行结果（结构化字段，Expert 模式）
                if message_tag == "tool_usage_card" and token:
                    if self.show_think:
                        tool_name, tool_args = _parse_tool_usage_card(token if isinstance(token, str) else "")
                        if not self.think_opened:
                            yield self._sse("<think>\n")
                            self.think_opened = True
                        if tool_name == "web_search":
                            query = tool_args.get("query", "")
                            if query:
                                yield self._sse(f"{prefix}[tool] search: {query}\n")
                        elif tool_name in ("web_browse", "browse_page"):
                            url = tool_args.get("url", "")
                            if url:
                                yield self._sse(f"{prefix}[tool] browse: {url}\n")
                        elif tool_name == "chatroom_send":
                            to = tool_args.get("to", "")
                            msg = tool_args.get("message", "")
                            if msg:
                                short_msg = msg[:100] + ("..." if len(msg) > 100 else "")
                                yield self._sse(f"{prefix}[expert] -> {to}: {short_msg}\n")
                        elif tool_name:
                            yield self._sse(f"{prefix}[tool] {tool_name}\n")
                    continue

                if message_tag == "raw_function_result" and (web_results_data or code_result):
                    if self.show_think:
                        if not self.think_opened:
                            yield self._sse("<think>\n")
                            self.think_opened = True
                        if web_results_data:
                            if isinstance(web_results_data, dict):
                                results_list = web_results_data.get("results", [])
                            elif isinstance(web_results_data, list):
                                results_list = web_results_data
                            else:
                                results_list = []
                            if results_list:
                                yield self._sse(f"{prefix}📄 找到 {len(results_list)} 条结果\n")
                        if code_result:
                            exit_code = code_result.get("exitCode", -1)
                            if exit_code == 0:
                                stdout = code_result.get("stdout", "").strip()
                                if stdout:
                                    short_out = stdout[:200] + ("..." if len(stdout) > 200 else "")
                                    yield self._sse(f"{prefix}✅ 执行成功: {short_out}\n")
                                else:
                                    yield self._sse(f"{prefix}✅ 执行成功\n")
                            else:
                                stderr = code_result.get("stderr", "").strip()
                                last_line = stderr.split('\n')[-1] if stderr else "未知错误"
                                yield self._sse(f"{prefix}❌ 执行失败: {last_line}\n")
                    continue

                # 普通 token
                if web_results_data and self.show_think:
                    if not self.think_opened:
                        yield self._sse("<think>\n")
                        self.think_opened = True
                    if isinstance(web_results_data, dict):
                        results_list = web_results_data.get("results", [])
                    elif isinstance(web_results_data, list):
                        results_list = web_results_data
                    else:
                        results_list = []
                    if results_list:
                        yield self._sse(f"{prefix}[tool] found {len(results_list)} results\n")
                    continue

                if token is not None:
                    if not token:
                        continue

                    # Flush tool buffer when tag changes
                    if self._tool_tag and message_tag != self._tool_tag:
                        formatted = _format_tool_call(self._tool_tag, self._tool_buf)
                        if formatted and self.show_tool_calls:
                            yield self._sse(formatted)
                        self._tool_buf = ""
                        self._tool_tag = None

                    # Accumulate tool call tokens (legacy / no structured field)
                    if message_tag in ("function_call", "raw_function_result", "tool_usage_card"):
                        if self.show_tool_calls:
                            self._tool_tag = message_tag
                            self._tool_buf += token
                        continue

                    # Skip thinking tokens if thinking phase is already done
                    if self.thinking_finished and is_thinking:
                        continue

                    # Handle thinking state transitions
                    if not self.is_thinking and is_thinking:
                        if self.show_think and not self.think_opened:
                            yield self._sse("<think>\n")
                            self.think_opened = True
                    elif self.is_thinking and not is_thinking:
                        if self.show_think and self.think_opened:
                            yield self._sse("</think>\n")
                            self.think_opened = False
                        self.thinking_finished = True

                    self.is_thinking = is_thinking

                    # Append web search results if present
                    web_results = web_results_data if isinstance(web_results_data, dict) else {}
                    if tool_usage_card_id and isinstance(web_results.get("results"), list):
                        if is_thinking and self.show_think:
                            appended = ""
                            for r in web_results["results"]:
                                title = r.get("title", "")
                                url = r.get("url", "")
                                preview = r.get("preview", "").replace("\n", "")
                                appended += f"\n- [{title}]({url} \"{preview}\")"
                            token += f"{appended}\n"
                        else:
                            continue

                    # Strictly strip <grok:render ...>...</grok:render> blocks across token boundaries
                    merged = self._grok_render_pending + token
                    if len(merged) > _GROK_RENDER_BUF_MAX:
                        merged = merged[-_GROK_RENDER_BUF_MAX:]
                    out_parts: list[str] = []
                    s = merged
                    while s:
                        if not self._in_grok_render_block:
                            m = _GROK_RENDER_OPEN_RE.search(s)
                            if not m:
                                # No complete open tag; hold back any partial tag prefix at end
                                partial_len = _find_partial_tag_suffix(s)
                                if partial_len:
                                    out_parts.append(s[:-partial_len])
                                    s = s[-partial_len:]
                                else:
                                    out_parts.append(s)
                                    s = ""
                                break
                            if m.start() > 0:
                                out_parts.append(s[: m.start()])
                            s = s[m.end():]
                            self._in_grok_render_block = True
                            continue
                        # Inside block: look for close tag
                        idx = s.find(_GROK_RENDER_CLOSE_TAG)
                        if idx == -1:
                            # Retain potential partial close tag at end for next iteration
                            tail_len = len(_GROK_RENDER_CLOSE_TAG) - 1
                            s = s[-tail_len:] if len(s) > tail_len else s
                            break
                        s = s[idx + len(_GROK_RENDER_CLOSE_TAG):]
                        self._in_grok_render_block = False
                    self._grok_render_pending = s
                    token = "".join(out_parts)
                    if not token:
                        continue

                    # Apply filter tags
                    if self.filter_tags and any(t in token for t in self.filter_tags):
                        continue

                    # Skip thinking content if not showing
                    if is_thinking and not self.show_think:
                        continue

                    yield self._sse(f"{prefix}{token}" if (is_thinking and prefix) else token)
                        
            # Flush any pending tool call buffer
            if self._tool_buf and self._tool_tag:
                formatted = _format_tool_call(self._tool_tag, self._tool_buf)
                if formatted and self.show_tool_calls:
                    yield self._sse(formatted)
                self._tool_buf = ""
                self._tool_tag = None
            if self.think_opened:
                yield self._sse("</think>\n")
            yield self._sse(finish="stop")
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream processing error: {e}", extra={"model": self.model})
            raise
        finally:
            await self.close()


class CollectProcessor(BaseProcessor):
    """非流式响应处理器"""
    
    def __init__(self, model: str, token: str = ""):
        super().__init__(model, token)
        self.image_format = get_config("app.image_format", "url")
    
    async def process(self, response: AsyncIterable[bytes]) -> dict[str, Any]:
        """处理并收集完整响应"""
        response_id = ""
        fingerprint = ""
        content = ""
        
        try:
            async for line in response:
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                
                resp = data.get("result", {}).get("response", {})
                
                if (llm := resp.get("llmInfo")) and not fingerprint:
                    fingerprint = llm.get("modelHash", "")
                
                if mr := resp.get("modelResponse"):
                    response_id = mr.get("responseId", "")
                    content = mr.get("message", "")
                    
                    if urls := mr.get("generatedImageUrls"):
                        content += "\n"
                        for url in urls:
                            parts = url.split("/")
                            img_id = parts[-2] if len(parts) >= 2 else "image"
                            
                            if self.image_format == "base64":
                                dl_service = self._get_dl()
                                base64_data = await dl_service.to_base64(url, self.token, "image")
                                if base64_data:
                                    content += f"![{img_id}]({base64_data})\n"
                                else:
                                    final_url = await self.process_url(url, "image")
                                    content += f"![{img_id}]({final_url})\n"
                            else:
                                final_url = await self.process_url(url, "image")
                                content += f"![{img_id}]({final_url})\n"
                    
                    if (meta := mr.get("metadata", {})).get("llm_info", {}).get("modelHash"):
                        fingerprint = meta["llm_info"]["modelHash"]
                            
        except Exception as e:
            logger.error(f"Collect processing error: {e}", extra={"model": self.model})
        finally:
            await self.close()
        
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": fingerprint,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content, "refusal": None, "annotations": []},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": 0, "audio_tokens": 0, "image_tokens": 0},
                "completion_tokens_details": {"text_tokens": 0, "audio_tokens": 0, "reasoning_tokens": 0}
            }
        }


class VideoStreamProcessor(BaseProcessor):
    """视频流式响应处理器"""
    
    def __init__(self, model: str, token: str = "", think: bool = None):
        super().__init__(model, token)
        self.response_id: Optional[str] = None
        self.think_opened: bool = False
        self.role_sent: bool = False
        self.video_format = get_config("app.video_format", "url")
        
        if think is None:
            self.show_think = get_config("grok.thinking", False)
        else:
            self.show_think = think
    
    def _build_video_html(self, video_url: str, thumbnail_url: str = "") -> str:
        """构建视频 HTML 标签"""
        if get_config("grok.video_poster_preview", False):
            return _build_video_poster_preview(video_url, thumbnail_url)
        poster_attr = f' poster="{thumbnail_url}"' if thumbnail_url else ""
        return f'''<video id="video" controls="" preload="none"{poster_attr}>
  <source id="mp4" src="{video_url}" type="video/mp4">
</video>'''
    
    async def process(self, response: AsyncIterable[bytes]) -> AsyncGenerator[str, None]:
        """处理视频流式响应"""
        try:
            async for line in response:
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                
                resp = data.get("result", {}).get("response", {})
                
                if rid := resp.get("responseId"):
                    self.response_id = rid
                
                # 首次发送 role
                if not self.role_sent:
                    yield self._sse(role="assistant")
                    self.role_sent = True
                
                # 视频生成进度
                if video_resp := resp.get("streamingVideoGenerationResponse"):
                    progress = video_resp.get("progress", 0)
                    
                    if self.show_think:
                        if not self.think_opened:
                            yield self._sse("<think>\n")
                            self.think_opened = True
                        yield self._sse(f"正在生成视频中，当前进度{progress}%\n")
                    
                    if progress == 100:
                        video_url = video_resp.get("videoUrl", "")
                        thumbnail_url = video_resp.get("thumbnailImageUrl", "")
                        
                        if self.think_opened and self.show_think:
                            yield self._sse("</think>\n")
                            self.think_opened = False
                        
                        if video_url:
                            final_video_url = await self.process_url(video_url, "video")
                            final_thumbnail_url = ""
                            if thumbnail_url:
                                final_thumbnail_url = await self.process_url(thumbnail_url, "image")
                            
                            video_html = self._build_video_html(final_video_url, final_thumbnail_url)
                            yield self._sse(video_html)
                            
                            logger.info(f"Video generated: {video_url}")
                    continue
                        
            if self.think_opened:
                yield self._sse("</think>\n")
            yield self._sse(finish="stop")
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Video stream processing error: {e}", extra={"model": self.model})
        finally:
            await self.close()


class VideoCollectProcessor(BaseProcessor):
    """视频非流式响应处理器"""
    
    def __init__(self, model: str, token: str = ""):
        super().__init__(model, token)
        self.video_format = get_config("app.video_format", "url")
    
    def _build_video_html(self, video_url: str, thumbnail_url: str = "") -> str:
        if get_config("grok.video_poster_preview", False):
            return _build_video_poster_preview(video_url, thumbnail_url)
        poster_attr = f' poster="{thumbnail_url}"' if thumbnail_url else ""
        return f'''<video id="video" controls="" preload="none"{poster_attr}>
  <source id="mp4" src="{video_url}" type="video/mp4">
</video>'''
    
    async def process(self, response: AsyncIterable[bytes]) -> dict[str, Any]:
        """处理并收集视频响应"""
        response_id = ""
        content = ""
        
        try:
            async for line in response:
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                
                resp = data.get("result", {}).get("response", {})
                
                if video_resp := resp.get("streamingVideoGenerationResponse"):
                    if video_resp.get("progress") == 100:
                        response_id = resp.get("responseId", "")
                        video_url = video_resp.get("videoUrl", "")
                        thumbnail_url = video_resp.get("thumbnailImageUrl", "")
                        
                        if video_url:
                            final_video_url = await self.process_url(video_url, "video")
                            final_thumbnail_url = ""
                            if thumbnail_url:
                                final_thumbnail_url = await self.process_url(thumbnail_url, "image")
                            
                            content = self._build_video_html(final_video_url, final_thumbnail_url)
                            logger.info(f"Video generated: {video_url}")
                            
        except Exception as e:
            logger.error(f"Video collect processing error: {e}", extra={"model": self.model})
        finally:
            await self.close()
        
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content, "refusal": None},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }


class ImageStreamProcessor(BaseProcessor):
    """图片生成流式响应处理器"""
    
    def __init__(
        self,
        model: str,
        token: str = "",
        n: int = 1,
        response_format: str = "b64_json",
    ):
        super().__init__(model, token)
        self.partial_index = 0
        self.n = n
        self.target_index = random.randint(0, 1) if n == 1 else None
        self.response_format = (response_format or "b64_json").lower()
        if self.response_format == "url":
            self.response_field = "url"
        elif self.response_format == "base64":
            self.response_field = "base64"
        else:
            self.response_field = "b64_json"
    
    def _sse(self, event: str, data: dict) -> str:
        """构建 SSE 响应 (覆盖基类)"""
        return f"event: {event}\ndata: {orjson.dumps(data).decode()}\n\n"
    
    async def process(self, response: AsyncIterable[bytes]) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        final_images = []
        
        try:
            async for line in response:
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                
                resp = data.get("result", {}).get("response", {})
                
                # 图片生成进度
                if img := resp.get("streamingImageGenerationResponse"):
                    image_index = img.get("imageIndex", 0)
                    progress = img.get("progress", 0)
                    
                    if self.n == 1 and image_index != self.target_index:
                        continue
                    
                    out_index = 0 if self.n == 1 else image_index
                    
                    yield self._sse("image_generation.partial_image", {
                        "type": "image_generation.partial_image",
                        self.response_field: "",
                        "index": out_index,
                        "progress": progress
                    })
                    continue
                
                # modelResponse
                if mr := resp.get("modelResponse"):
                    if urls := mr.get("generatedImageUrls"):
                        for url in urls:
                            if self.response_format == "url":
                                processed = await self.process_url(url, "image")
                                if processed:
                                    final_images.append(processed)
                                continue
                            dl_service = self._get_dl()
                            base64_data = await dl_service.to_base64(url, self.token, "image")
                            if base64_data:
                                if "," in base64_data:
                                    b64 = base64_data.split(",", 1)[1]
                                else:
                                    b64 = base64_data
                                final_images.append(b64)
                    continue
                    
            for index, b64 in enumerate(final_images):
                if self.n == 1:
                    if index != self.target_index:
                        continue
                    out_index = 0
                else:
                    out_index = index
                
                yield self._sse("image_generation.completed", {
                    "type": "image_generation.completed",
                    self.response_field: b64,
                    "index": out_index,
                    "usage": {
                        "total_tokens": 50,
                        "input_tokens": 25,
                        "output_tokens": 25,
                        "input_tokens_details": {"text_tokens": 5, "image_tokens": 20}
                    }
                })
        except Exception as e:
            logger.error(f"Image stream processing error: {e}")
            raise
        finally:
            await self.close()


class ImageCollectProcessor(BaseProcessor):
    """图片生成非流式响应处理器"""
    
    def __init__(
        self,
        model: str,
        token: str = "",
        response_format: str = "b64_json",
    ):
        super().__init__(model, token)
        self.response_format = (response_format or "b64_json").lower()
    
    async def process(self, response: AsyncIterable[bytes]) -> List[str]:
        """处理并收集图片"""
        images = []
        
        try:
            async for line in response:
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                
                resp = data.get("result", {}).get("response", {})
                
                if mr := resp.get("modelResponse"):
                    if urls := mr.get("generatedImageUrls"):
                        for url in urls:
                            if self.response_format == "url":
                                processed = await self.process_url(url, "image")
                                if processed:
                                    images.append(processed)
                                continue
                            dl_service = self._get_dl()
                            base64_data = await dl_service.to_base64(url, self.token, "image")
                            if base64_data:
                                if "," in base64_data:
                                    b64 = base64_data.split(",", 1)[1]
                                else:
                                    b64 = base64_data
                                images.append(b64)
                                
        except Exception as e:
            logger.error(f"Image collect processing error: {e}")
        finally:
            await self.close()
        
        return images


__all__ = [
    "StreamProcessor",
    "CollectProcessor",
    "VideoStreamProcessor",
    "VideoCollectProcessor",
    "ImageStreamProcessor",
    "ImageCollectProcessor",
]
