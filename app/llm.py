"""LLM providers: huggingface (local), ollama (local), openai-compatible (API), extractive (no LLM)."""
from __future__ import annotations

import re

import requests

from .config import Settings
from .prompts import NOT_FOUND, build_messages


class BaseLLM:
    name = "base"

    def chat(self, messages: list[dict]) -> str:
        raise NotImplementedError

    def answer(self, question: str, contexts: list[dict]) -> str:
        return self.chat(build_messages(question, contexts)).strip()


class ExtractiveLLM(BaseLLM):
    """No generation at all: returns the most relevant sentences verbatim.

    Zero hallucination risk, instant, no downloads. Good fallback and test double.
    """

    name = "extractive"
    _STOP = {"what", "which", "when", "where", "who", "whom", "does", "this", "that",
             "with", "from", "have", "about", "into", "than", "then", "there", "their",
             "are", "the", "and", "for", "how", "why", "was", "were", "is", "of", "in"}

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def answer(self, question: str, contexts: list[dict]) -> str:
        q = self._tokens(question) - self._STOP
        scored = []
        for i, c in enumerate(contexts, start=1):
            for sent in re.split(r"(?<=[.!?])\s+", c["text"]):
                overlap = len(self._tokens(sent) & q)
                if overlap:
                    scored.append((overlap, -i, i, sent.strip()))
        if not scored:
            return NOT_FOUND
        scored.sort(reverse=True)
        return "\n".join(f"- {s} [{i}]" for _, _, i, s in scored[:3])


class OllamaLLM(BaseLLM):
    name = "ollama"

    def __init__(self, s: Settings):
        self.url, self.model, self.max_tokens = s.ollama_url.rstrip("/"), s.ollama_model, s.max_new_tokens

    def chat(self, messages: list[dict]) -> str:
        r = requests.post(
            f"{self.url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False,
                  "options": {"temperature": 0, "num_predict": self.max_tokens}},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]


class OpenAILLM(BaseLLM):
    """Any OpenAI-compatible endpoint (OpenAI, Groq, Together, LM Studio, vLLM...)."""

    name = "openai"

    def __init__(self, s: Settings):
        if not s.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (required for LLM_PROVIDER=openai).")
        self.s = s

    def chat(self, messages: list[dict]) -> str:
        r = requests.post(
            f"{self.s.openai_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.s.openai_api_key}"},
            json={"model": self.s.openai_model, "messages": messages,
                  "temperature": 0, "max_tokens": self.s.max_new_tokens},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class HuggingFaceLLM(BaseLLM):
    """Runs an instruction-tuned model locally with Hugging Face Transformers."""

    name = "huggingface"

    def __init__(self, s: Settings):
        self.model_name, self.max_tokens = s.hf_llm_model, s.max_new_tokens
        self._tok = self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype="auto", device_map="auto"
            )
        return self._tok, self._model

    def chat(self, messages: list[dict]) -> str:
        tok, model = self._load()
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def build_llm(s: Settings) -> BaseLLM:
    providers = {
        "huggingface": HuggingFaceLLM,
        "ollama": OllamaLLM,
        "openai": OpenAILLM,
        "extractive": lambda _s: ExtractiveLLM(),
    }
    if s.llm_provider not in providers:
        raise ValueError(f"Unknown LLM_PROVIDER '{s.llm_provider}'. Choose from {list(providers)}.")
    return providers[s.llm_provider](s)
