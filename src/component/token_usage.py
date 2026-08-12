# -*- coding: utf-8 -*-
"""模型 token 用量监控：回调采集每次 LLM 调用 usage，输出结构化日志（不落库）。

- TokenUsageCallback: BaseCallbackHandler，on_llm_end 采集 usage（含上下文缓存命中/未命中）与延迟
- estimate_cost: 按模型单价估算成本（元）
"""
import json
import time
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler

from component.logger import get_logger

_log = get_logger("token_usage")

# 模型单价（元 / 百万 token）：输入命中 / 输入未命中 / 输出。可按需覆盖。
DEFAULT_PRICES: Dict[str, Dict[str, float]] = {
    "deepseek-v4-flash": {"input_hit": 0.5, "input_miss": 2.0, "output": 8.0},
}


def estimate_cost(
    model: str,
    usage: Dict[str, int],
    prices: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """估算成本（元）。命中/未命中缺省时按全部输入视为未命中。"""
    p = (prices or DEFAULT_PRICES).get(model, {})
    hit = usage.get("hit", 0) or 0
    miss = usage.get("miss", 0) or 0
    inp = usage.get("input", 0) or 0
    out = usage.get("output", 0) or 0
    if hit + miss == 0:
        hit, miss = 0, inp
    return round(
        (hit * p.get("input_hit", 0.5)
         + miss * p.get("input_miss", 2.0)
         + out * p.get("output", 8.0)) / 1_000_000,
        6,
    )


class TokenUsageCallback(BaseCallbackHandler):
    """采集每次 LLM 调用 usage（含缓存命中），打印一行 JSON，并累加会话级汇总。"""

    def __init__(
        self,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        prices: Optional[Dict[str, Dict[str, float]]] = None,
        logger=None,
    ):
        self.session_id = session_id
        self.model = model
        self.prices = prices or DEFAULT_PRICES
        self._log = logger or _log
        self._starts: Dict[str, float] = {}
        self.round_total = {"input": 0, "output": 0, "hit": 0, "miss": 0}

    def reset(self) -> None:
        self.round_total = {"input": 0, "output": 0, "hit": 0, "miss": 0}
        self._starts.clear()

    # ---------- 采集 ----------

    def on_llm_start(self, serialized: Dict[str, Any], prompts, **kwargs) -> None:
        run_id = kwargs.get("run_id")
        if run_id:
            self._starts[run_id] = time.time()

    def on_llm_end(self, response, **kwargs) -> None:
        run_id = kwargs.get("run_id")
        start = self._starts.pop(run_id, None) if run_id else None
        latency_ms = round((time.time() - start) * 1000, 1) if start else None
        usage = self._extract_usage(response)
        model = self.model or self._model_of(response)
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": self.session_id,
            "model": model,
            "input": usage.get("input", 0),
            "output": usage.get("output", 0),
            "total": usage.get("total", 0),
            "hit": usage.get("hit", 0),
            "miss": usage.get("miss", 0),
            "latency_ms": latency_ms,
        }
        self._log.info("token_usage %s", json.dumps(
            record, ensure_ascii=False))
        for k in ("input", "output", "hit", "miss"):
            self.round_total[k] += usage.get(k, 0)

    def _extract_usage(self, response) -> Dict[str, int]:
        u: Dict[str, Any] = {}
        llm_output = getattr(response, "llm_output", None) or {}
        u = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if not u:
            gens = getattr(response, "generations", None) or []
            if gens and gens[0]:
                info = getattr(gens[0][0], "generation_info", None) or {}
                u = info.get("token_usage") or info.get("usage") or {}
        return {
            "input": u.get("prompt_tokens", 0) if u else 0,
            "output": u.get("completion_tokens", 0) if u else 0,
            "total": u.get("total_tokens", 0) if u else 0,
            "hit": u.get("prompt_cache_hit_tokens", 0) if u else 0,
            "miss": u.get("prompt_cache_miss_tokens", 0) if u else 0,
        }

    def _model_of(self, response) -> str:
        llm_output = getattr(response, "llm_output", None) or {}
        return llm_output.get("model_name") or "unknown"

    # ---------- 会话级汇总 ----------

    def round_summary(self) -> Dict[str, Any]:
        t = self.round_total
        hit_rate = (t["hit"] / (t["hit"] + t["miss"])
                    ) if (t["hit"] + t["miss"]) else 0.0
        usage = {"input": t["input"], "output": t["output"],
                 "hit": t["hit"], "miss": t["miss"]}
        return {
            "session_id": self.session_id,
            "model": self.model,
            "total_input": t["input"],
            "total_output": t["output"],
            "total_tokens": t["input"] + t["output"],
            "cache_hit": t["hit"],
            "cache_miss": t["miss"],
            "hit_rate": round(hit_rate, 4),
            "cost_yuan": estimate_cost(self.model or "", usage, self.prices),
        }

    def log_round_summary(self) -> None:
        self._log.info("session_summary %s", json.dumps(
            self.round_summary(), ensure_ascii=False))
