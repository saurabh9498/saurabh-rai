"""
Model interface, plus a scripted stand-in for tests.

The pipeline needs two things from a model: the generated text, and the log-probability
of each generated token. Log-probabilities are how the Builder's confidence is measured
(nl2sql/confidence.py). Not every hosted API returns them; OpenAI-compatible endpoints,
including vLLM and TensorRT-LLM servers, do via `logprobs: true`.

ScriptedLLM replays canned completions so the control flow can be tested deterministically
without a model or an API key. It says nothing about how well a real model writes SQL.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Completion:
    text: str
    token_logprobs: list[float]


class LLM(Protocol):
    def complete(self, prompt: str) -> Completion: ...


@dataclass
class ScriptedLLM:
    """Returns the next scripted completion each call and records every prompt."""
    script: list[Completion]
    prompts: list[str] = field(default_factory=list)

    def complete(self, prompt: str) -> Completion:
        self.prompts.append(prompt)
        if not self.script:
            raise RuntimeError("ScriptedLLM ran out of completions")
        return self.script.pop(0)


@dataclass
class OpenAICompatibleLLM:
    """Chat-completions client that requests token log-probabilities. Not run in this repo."""
    base_url: str = os.environ.get("NL2SQL_BASE_URL", "http://localhost:8000/v1")
    model: str = os.environ.get("NL2SQL_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    api_key: str = os.environ.get("NL2SQL_API_KEY", "none")
    temperature: float = 0.0

    def complete(self, prompt: str) -> Completion:
        body = json.dumps({"model": self.model, "temperature": self.temperature, "logprobs": True,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=60) as r:
            choice = json.loads(r.read())["choices"][0]
        content = choice["logprobs"]["content"] or []
        return Completion(choice["message"]["content"], [t["logprob"] for t in content])
