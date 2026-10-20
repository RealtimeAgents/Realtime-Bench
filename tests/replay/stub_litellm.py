"""The chat completions stub as the generic litellm API reaches it, with request headers kept.

With the ``openai/`` provider prefix LiteLLM dispatches to its OpenAI handler
(litellm/main.py:5906-5916, 2567-2705). This API passes LiteLLM no client
(providers/litellm.py:411, 586-588, 615-617), so LiteLLM builds its own AsyncOpenAI client on its
aiohttp transport (litellm/llms/openai/openai.py:404-425; litellm/llms/openai/common_utils.py:
305-330) and posts OpenAI chat completions to ``{api_base}/chat/completions``: the wire format
ChatCompletionsStub speaks. This API never streams (providers/litellm.py:412), so the request has
no ``stream`` field and every reply is one JSON completion (stub.py:80-83).

The stub also keeps each request's headers from a fixed allow-list: those that change what the
server does. The others only describe the client or the machine (User-Agent, X-Stainless-*, the
Host port, Content-Length). Names are compared lowercased; their case depends on the HTTP stack
(the aiohttp transport sends httpx's lowercased names: litellm/llms/custom_httpx/
aiohttp_transport.py:383-386).
"""

from __future__ import annotations

from .stub import ChatCompletionsStub

COMPARED_HEADERS = frozenset(
    {
        "accept",
        "authorization",  # the key the driver resolved (providers/litellm.py:496-505, 611-612)
        "content-type",
        # The SDK sends these when an organization or project is set (openai/_client.py:1146-1151).
        "openai-organization",
        "openai-project",
    }
)


class LiteLLMStub(ChatCompletionsStub):
    """ChatCompletionsStub that also records the allow-listed headers of every request, in order."""

    def __init__(self, replies):
        self.request_headers: list[dict[str, str]] = []
        super().__init__(replies)

    def _handler(self):
        stub = self
        base = super()._handler()

        class Handler(base):
            def do_POST(self):
                stub.request_headers.append(
                    {
                        name.lower(): value
                        for name, value in self.headers.items()
                        if name.lower() in COMPARED_HEADERS
                    }
                )
                super().do_POST()

        return Handler
