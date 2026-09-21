"""Anthropic Messages API on top of an OpenAI Chat Completions backend.

llama-server answers /v1/messages itself; halogen-flash-server does not, and
Claude Code speaks nothing else. proxy.py uses this module to translate for
such backends (LLM_TRANSLATE_MESSAGES=1): a Messages request becomes a Chat
Completions request, and the answer — whole or streamed — becomes a Messages
response again.

What is mapped:
  system, text, images (base64 and URL), tool_use / tool_result, tools and
  tool_choice, thinking (passed on as `thinking`, which halogen reads), the
  effort of output_config (as reasoning_effort), stop_sequences, samplers,
  usage including cached prompt tokens.
What is not: server tools (web search and the like — the backend has none),
  citations, and thinking signatures (sent back empty; the backend never
  checks them).
"""
import json
import uuid

# Anthropic's effort names → the reasoning_effort levels of an OpenAI backend.
_EFFORT = {"low": "low", "medium": "medium", "high": "high", "max": "xhigh"}

_STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


def _msg_id():
    return "msg_" + uuid.uuid4().hex[:24]


def _blocks(content):
    """Content as a list of blocks; a plain string is one text block."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content or []


def _text(content):
    """All text in a string or a list of blocks, joined."""
    return "\n\n".join(b.get("text", "") for b in _blocks(content)
                       if b.get("type") == "text" and b.get("text"))


def _image_part(block):
    src = block.get("source") or {}
    if src.get("type") == "base64":
        url = f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
    elif src.get("type") == "url":
        url = src.get("url", "")
    else:
        return None
    return {"type": "image_url", "image_url": {"url": url}}


def _user_content(parts):
    """OpenAI user content: a plain string when there is only text, so the
    prompt stays byte-identical to what a text-only client would send."""
    if all(p["type"] == "text" for p in parts):
        return "\n\n".join(p["text"] for p in parts)
    return parts


def _tool_result_text(block):
    content = block.get("content")
    if isinstance(content, str):
        text = content
    else:
        pieces = []
        for b in content or []:
            if b.get("type") == "text":
                pieces.append(b.get("text", ""))
            elif b.get("type") == "image":
                pieces.append("[image]")
        text = "\n\n".join(pieces)
    if block.get("is_error"):
        text = "Error: " + text
    return text


def _convert_messages(messages):
    out = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            out.append({"role": "system", "content": _text(content)})
            continue
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if role == "assistant":
            texts, calls, thinking = [], [], []
            for b in content or []:
                t = b.get("type")
                if t == "text":
                    texts.append(b.get("text", ""))
                elif t == "tool_use":
                    calls.append({
                        "id": b.get("id"),
                        "type": "function",
                        "function": {"name": b.get("name"),
                                     "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)},
                    })
                elif t == "thinking" and b.get("thinking"):
                    thinking.append(b["thinking"])
            m = {"role": "assistant", "content": "".join(texts) or None}
            if calls:
                m["tool_calls"] = calls
            if thinking:
                m["reasoning_content"] = "\n\n".join(thinking)
            out.append(m)
            continue

        # user: tool results become tool messages, which must directly follow
        # the assistant turn that called them; everything else stays one user
        # message after them.
        parts = []
        for b in content or []:
            t = b.get("type")
            if t == "tool_result":
                out.append({"role": "tool", "tool_call_id": b.get("tool_use_id"),
                            "content": _tool_result_text(b)})
                # A tool message holds text only; images it returned travel in
                # the user message that follows.
                if isinstance(b.get("content"), list):
                    for inner in b["content"]:
                        if inner.get("type") == "image":
                            p = _image_part(inner)
                            if p:
                                parts.append(p)
            elif t == "text":
                parts.append({"type": "text", "text": b.get("text", "")})
            elif t == "image":
                p = _image_part(b)
                if p:
                    parts.append(p)
            elif t == "document":
                src = b.get("source") or {}
                if src.get("type") == "text":
                    parts.append({"type": "text", "text": src.get("data", "")})
        if parts:
            out.append({"role": "user", "content": _user_content(parts)})
    return out


def _place_system_turns(messages):
    """Chat templates such as Qwen's take system text only at the very start,
    while the Messages API lets a system turn follow others (Claude Code sends
    its environment that way). Leading system turns merge into one; a later one
    joins the user turn before it, or becomes a user turn of its own. Earlier
    turns are never rewritten, so the prompt prefix — and with it the prompt
    cache — stays stable as a conversation grows."""
    out = []
    for m in messages:
        if m["role"] != "system":
            out.append(m)
            continue
        text = m.get("content") or ""
        if not text:
            continue
        if all(x["role"] == "system" for x in out):
            if out:
                out[0]["content"] += "\n\n" + text
            else:
                out.append({"role": "system", "content": text})
        elif out[-1]["role"] == "user" and isinstance(out[-1]["content"], str):
            out[-1]["content"] += "\n\n" + text
        elif out[-1]["role"] == "user":
            out[-1]["content"].append({"type": "text", "text": text})
        else:
            out.append({"role": "user", "content": text})
    return out


def _convert_tools(tools):
    out = []
    for t in tools or []:
        # Server tools (web_search_…, code_execution_…) carry no schema and
        # nothing on this side could run them.
        if "input_schema" not in t:
            continue
        out.append({"type": "function", "function": {
            "name": t.get("name"),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
        }})
    return out


def _convert_tool_choice(tc):
    kind = (tc or {}).get("type")
    if kind == "any":
        return "required"
    if kind == "tool":
        return {"type": "function", "function": {"name": tc.get("name")}}
    if kind == "none":
        return "none"
    return "auto"


def messages_to_chat(body, max_tokens_cap=None):
    """Translate a Messages request body into a Chat Completions one."""
    chat = {"model": body.get("model"), "messages": []}

    system = body.get("system")
    if system:
        text = system if isinstance(system, str) else _text(system)
        if text:
            chat["messages"].append({"role": "system", "content": text})
    chat["messages"].extend(_convert_messages(body.get("messages") or []))
    chat["messages"] = _place_system_turns(chat["messages"])

    max_tokens = body.get("max_tokens")
    if max_tokens is not None:
        if max_tokens_cap:
            max_tokens = min(int(max_tokens), max_tokens_cap)
        chat["max_tokens"] = max_tokens
    for key in ("temperature", "top_p", "top_k"):
        if key in body:
            chat[key] = body[key]
    if body.get("stop_sequences"):
        chat["stop"] = body["stop_sequences"]

    tools = _convert_tools(body.get("tools"))
    if tools:
        chat["tools"] = tools
        tc = body.get("tool_choice")
        if tc:
            chat["tool_choice"] = _convert_tool_choice(tc)
            if tc.get("disable_parallel_tool_use"):
                chat["parallel_tool_calls"] = False

    thinking = body.get("thinking") or {}
    if thinking.get("type") == "disabled":
        chat["thinking"] = {"type": "disabled"}
    elif thinking.get("type") == "enabled":
        chat["thinking"] = {"type": "enabled"}
        if thinking.get("budget_tokens"):
            chat["thinking"]["budget_tokens"] = thinking["budget_tokens"]
    # "adaptive" (and no thinking field at all) leave it to the server default.

    effort = (body.get("output_config") or {}).get("effort")
    if effort in _EFFORT and chat.get("thinking", {}).get("type") != "disabled":
        chat["reasoning_effort"] = _EFFORT[effort]

    if body.get("stream"):
        chat["stream"] = True
        chat["stream_options"] = {"include_usage": True}
    return chat


def _parse_args(arguments):
    if isinstance(arguments, dict):
        return arguments
    try:
        value = json.loads(arguments or "{}")
        return value if isinstance(value, dict) else {"value": value}
    except json.JSONDecodeError:
        return {"_raw": arguments}


def _usage(usage, timings=None):
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    if not cached and timings:
        cached = int(timings.get("cache_n") or 0)
    cached = min(cached, prompt)
    return {
        "input_tokens": prompt - cached,
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,
        "output_tokens": int(usage.get("completion_tokens") or 0),
    }


def chat_to_message(resp, model):
    """Translate a (non-streamed) Chat Completions response into a Message."""
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = []
    if msg.get("reasoning_content"):
        content.append({"type": "thinking", "thinking": msg["reasoning_content"], "signature": ""})
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        content.append({"type": "tool_use", "id": tc.get("id") or "toolu_" + uuid.uuid4().hex[:24],
                        "name": fn.get("name"), "input": _parse_args(fn.get("arguments"))})
    return {
        "id": _msg_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _STOP_REASON.get(choice.get("finish_reason") or "", "end_turn"),
        "stop_sequence": None,
        "usage": _usage(resp.get("usage"), resp.get("timings")),
    }


def error_body(status, message):
    kind = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
            404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error",
            529: "overloaded_error"}.get(status, "api_error")
    return {"type": "error", "error": {"type": kind, "message": message}}


def openai_error_message(raw):
    """The message of an OpenAI-style error body, or the raw text."""
    try:
        err = json.loads(raw)
        if isinstance(err.get("error"), dict):
            return err["error"].get("message") or raw
        if isinstance(err.get("error"), str):
            return err["error"]
        if "detail" in err:
            return str(err["detail"])
    except (ValueError, AttributeError):
        pass
    return raw


def estimate_tokens(body):
    """count_tokens without a tokenizer: ~3.5 characters per token over
    everything the prompt carries. Clients use it to decide when to compact,
    so erring high is the safe side."""
    chat = messages_to_chat(body)
    size = len(json.dumps(chat["messages"], ensure_ascii=False))
    size += len(json.dumps(chat.get("tools", []), ensure_ascii=False))
    return {"input_tokens": int(size / 3.5) + 1}


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


class StreamTranslator:
    """Turns Chat Completions stream chunks into Messages stream events."""

    def __init__(self, model):
        self.model = model
        self.index = -1          # index of the open content block
        self.kind = None         # "thinking" | "text" | "tool" of the open block
        self.tool_blocks = {}    # OpenAI tool_call index → our block index
        self.finish = None
        self.usage = None
        self.timings = None

    def start(self):
        return sse("message_start", {"type": "message_start", "message": {
            "id": _msg_id(), "type": "message", "role": "assistant", "model": self.model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}}})

    def _close(self):
        out = b""
        if self.kind is None:
            return out
        if self.kind == "thinking":
            out += sse("content_block_delta", {"type": "content_block_delta", "index": self.index,
                                               "delta": {"type": "signature_delta", "signature": ""}})
        out += sse("content_block_stop", {"type": "content_block_stop", "index": self.index})
        self.kind = None
        return out

    def _open(self, kind, block):
        out = self._close()
        self.index += 1
        self.kind = kind
        out += sse("content_block_start", {"type": "content_block_start", "index": self.index,
                                           "content_block": block})
        return out

    def _delta(self, delta):
        return sse("content_block_delta", {"type": "content_block_delta", "index": self.index,
                                           "delta": delta})

    def feed(self, chunk):
        out = b""
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        if chunk.get("timings"):
            self.timings = chunk["timings"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                if self.kind != "thinking":
                    out += self._open("thinking", {"type": "thinking", "thinking": "", "signature": ""})
                out += self._delta({"type": "thinking_delta", "thinking": delta["reasoning_content"]})
            if delta.get("content"):
                if self.kind != "text":
                    out += self._open("text", {"type": "text", "text": ""})
                out += self._delta({"type": "text_delta", "text": delta["content"]})
            for tc in delta.get("tool_calls") or []:
                i = tc.get("index", 0)
                fn = tc.get("function") or {}
                if i not in self.tool_blocks:
                    out += self._open("tool", {"type": "tool_use",
                                               "id": tc.get("id") or "toolu_" + uuid.uuid4().hex[:24],
                                               "name": fn.get("name"), "input": {}})
                    self.tool_blocks[i] = self.index
                # Arguments of an earlier call arriving late cannot be placed:
                # Messages streams one block at a time. Backends send calls in
                # order, so this does not happen in practice.
                if fn.get("arguments") and self.tool_blocks[i] == self.index:
                    out += self._delta({"type": "input_json_delta", "partial_json": fn["arguments"]})
            if choice.get("finish_reason"):
                self.finish = choice["finish_reason"]
        return out

    def end(self):
        out = self._close()
        usage = _usage(self.usage, self.timings)
        out += sse("message_delta", {"type": "message_delta",
                                     "delta": {"stop_reason": _STOP_REASON.get(self.finish or "", "end_turn"),
                                               "stop_sequence": None},
                                     "usage": usage})
        out += sse("message_stop", {"type": "message_stop"})
        return out
