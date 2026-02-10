import inspect
import json
import time
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Type

from openai import OpenAI
from .func_call_shema import as_json_schema
from .func_call_shema import doc as D




# =========================
# OpenAI Client
# =========================

# -------------------------
# OpenAI Client with Chat + Tools
# -------------------------
class OpenAIClient:
    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: int = 600,
        max_retries: int = 3,
        tools_inject = {}
    ):
        client_kwargs = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if api_base is not None:
            client_kwargs["base_url"] = api_base

        self.client = OpenAI(**client_kwargs)
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.tools_inject = tools_inject
        self.tools_inject["_openai_client"] = self

        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: List[Dict[str, Any]] = []

    # -------------------------
    # Register a Python function as a tool
    # -------------------------
    def register_tool(self, func: Callable):
        name = func.__name__
        if name in self._tools:
            raise ValueError(f"Tool `{name}` already registered")
        self._tools[name] = func
        self._tool_schemas.append({
            "type": "function",
            "function": as_json_schema(func)
        })

    # -------------------------
    # Basic Chat
    # -------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        max_tokens: int = 2000,
    ) -> str:
        resp = self._request(messages=messages, temperature=temperature, max_tokens=max_tokens)
        return resp.choices[0].message.content

    # -------------------------
    # Chat with tools
    # -------------------------
    def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        tool_choice = "auto"
    ) -> str:
        """
        Agent loop that automatically executes tool calls until the model returns
        a normal assistant message.
        """
        resp = self._request(
            messages=messages,
            temperature=temperature,
            tools=self._tool_schemas,
            tool_choice=tool_choice
        )
        msg = resp.choices[0].message

        return {"content": msg.content.strip() if msg.content else "", "tool_calls": msg.tool_calls}

        #     # If it's not a tool call, return the assistant text
        #     if msg.get("role") == "assistant" and "function_call" not in msg:
        #         return msg["content"]

        #     # Execute tool call
        #     tool_call = msg.get("function_call")
        #     if not tool_call:
        #         raise RuntimeError("Unexpected message without tool_call")

        #     tool_name = tool_call["name"]
        #     args = json.loads(tool_call.get("arguments", "{}"))

        #     if tool_name not in self._tools:
        #         raise RuntimeError(f"Unknown tool `{tool_name}`")

        #     result = self._tools[tool_name](**self.tools_inject,**args)

        #     # Append tool result back into the conversation
        #     messages.append({
        #         "role": "tool",
        #         "name": tool_name,
        #         "content": json.dumps(result, ensure_ascii=False)
        #     })

        # raise RuntimeError("Tool loop exceeded max steps")

    # -------------------------
    # Low-level request with retry
    # -------------------------
    def _request(self, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return self.client.chat.completions.create(
                    model=self.model_name,
                    **kwargs
                )
            except Exception as e:
                print(e)
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)