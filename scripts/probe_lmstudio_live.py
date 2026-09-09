"""Live LM Studio compatibility probe (synthetic data only).

Verifies structured CallLLMOutput against the loaded Qwen 3.5 model at the
configured context.  Prints metadata only — never prompts or transcripts.
Run manually against a running LM Studio instance; not part of CI.
"""

from __future__ import annotations

import json
import sys
import time

import httpx

from src.llm.schemas import CallLLMOutput

BASE = "http://127.0.0.1:1234"
MODEL = "qwen3.5-9b-the-defiant-fable-uncensored-heretic-neo-imatrix-max-mtp"


def status() -> dict[str, object]:
    r = httpx.get(f"{BASE}/api/v1/models", timeout=30, trust_env=False)
    r.raise_for_status()
    data = r.json()
    entries = data.get("models") or []
    entry = next((m for m in entries if m.get("key") == MODEL), None)
    if not entry:
        return {"loaded": False, "model": MODEL}
    instances = entry.get("loaded_instances") or []
    inst = instances[0] if instances else None
    config = (inst.get("config") if isinstance(inst, dict) else {}) or {}
    caps = entry.get("capabilities") or {}
    reasoning = caps.get("reasoning") if isinstance(caps, dict) else {}
    quant = entry.get("quantization") or {}
    return {
        "loaded": bool(inst),
        "loaded_context": config.get("context_length"),
        "max_context": entry.get("max_context_length"),
        "quantization": quant.get("name") if isinstance(quant, dict) else None,
        "reasoning_default": reasoning.get("default") if isinstance(reasoning, dict) else None,
    }


def probe(schema: dict, label: str) -> dict[str, object]:
    prompt = (
        "Du är en svensk callcenter-analysator. Returnera ENDAST giltig JSON enligt schemat. "
        "Analysera följande syntetiska samtal:\n"
        "Agent: Hej, vad gäller saken?\n"
        "Kund: Mitt abonnemang dubbeldebiteras sedan tre månader.\n"
        "Agent: Jag beklagar, jag ska undersöka det.\n"
        "Kund: Det är frustrerande, ni har inte löst det än.\n"
        "Agent: Jag eskalerar till faktureringsteamet.\n"
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Returnera endast JSON enligt angivet schema."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "CallLLMOutput",
                "strict": True,
                "schema": schema,
            },
        },
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    t0 = time.perf_counter()
    r = httpx.post(f"{BASE}/v1/chat/completions", json=payload, timeout=300, trust_env=False)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    c = (d.get("choices") or [{}])[0]
    msg = c.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    usage = d.get("usage") or {}
    parsed = None
    pydantic_ok = False
    parse_err = None
    if content:
        try:
            parsed = json.loads(content)
            CallLLMOutput.model_validate(parsed)
            pydantic_ok = True
        except Exception as exc:  # noqa: BLE001
            parse_err = f"{type(exc).__name__}: {exc}"
    return {
        "label": label,
        "http_status": r.status_code,
        "finish_reason": c.get("finish_reason"),
        "elapsed_s": round(dt, 1),
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "json_parsed": parsed is not None,
        "pydantic_valid": pydantic_ok,
        "parse_error": parse_err,
    }


def main() -> int:
    st = status()
    print("STATUS", json.dumps(st, ensure_ascii=False))
    if not st.get("loaded"):
        print("Model not loaded — start LM Studio and load the target model first.")
        return 1
    schema = CallLLMOutput.model_json_schema()
    res = probe(schema, "CallLLMOutput")
    print("PROBE", json.dumps(res, ensure_ascii=False))
    if not res["pydantic_valid"]:
        print("FAIL: structured output did not validate as CallLLMOutput.")
        return 2
    if res["content_chars"] == 0:
        print("FAIL: empty content (reasoning-only response).")
        return 3
    print("OK: structured output valid, content non-empty.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
