"""Model-output parsing and guarded generation.

Copied from DIGIVAX vlm_tests/test_scripts/utils.py (extract_json, salvage_entries,
generate_extraction) so predictions match the evaluated pipeline exactly.
"""
import json
import re

_RECORD_KEYS = ("date", "vaccine_name", "status", "note")


def _iter_json_objects(text: str, start: int):
    """Yield every complete top-level ``{...}`` block found from ``start`` on.

    String-aware brace matching, so nested objects (e.g. a record's stickers)
    stay inside their parent and a half-written object at the end of a
    truncated generation is simply not yielded.
    """
    depth = 0
    in_str = esc = False
    begin = None
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                begin = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and begin is not None:
                yield text[begin:i + 1]
                begin = None
            elif depth < 0:
                return
        elif c == "]" and depth == 0:
            return  # entries array closed — nothing useful after it


def salvage_entries(text: str) -> list:
    """Recover the valid prefix of the entries list from a truncated output.

    Greedy decoding occasionally degenerates into a repetition loop on
    information-dense pages: the model keeps emitting near-identical records
    until it runs into max_new_tokens, so the JSON never closes. The records
    generated *before* the loop are usually correct, so we keep every
    record-shaped object and cut at the first repeat (ignoring b_box).
    """
    m = re.search(r'"entries"\s*:\s*\[', text)
    start = m.end() if m else 0
    records = []
    for chunk in _iter_json_objects(text, start):
        try:
            obj = json.loads(chunk)
        except Exception:
            continue
        if isinstance(obj, dict) and any(k in obj for k in _RECORD_KEYS):
            records.append(obj)
    seen, out = set(), []
    for r in records:
        key = json.dumps({k: v for k, v in r.items() if k != "b_box"},
                         sort_keys=True, ensure_ascii=False)
        if key in seen:
            break  # repetition loop starts here — everything after it is noise
        seen.add(key)
        out.append(r)
    return out


def _clean(text: str) -> str:
    text = text.strip()
    # strip closed and unclosed <think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()
    # strip markdown fences
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text


def extract_json(text: str, quiet: bool = False) -> list:
    """Best-effort extraction of the entries list from model output.

    Accepts either the full label object {"country": ..., "entries": [...]}
    or a bare JSON array as fallback.
    """
    text = _clean(text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("entries"), list):
            return [r for r in obj["entries"] if isinstance(r, dict)]
        if isinstance(obj, list) and all(isinstance(r, dict) for r in obj):
            return obj
    except Exception:
        pass
    # Try each top-level {...} block (largest first)
    candidates_obj = re.findall(r"\{.*\}", text, re.DOTALL)
    for candidate in sorted(candidates_obj, key=len, reverse=True):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and isinstance(obj.get("entries"), list):
                return [r for r in obj["entries"] if isinstance(r, dict)]
        except Exception:
            pass
    # Truncated / degenerated generation: keep the valid prefix of the records.
    # Must run before the bare-array fallback below, which would otherwise
    # "rescue" a nested "stickers": [...] array and report it as the prediction.
    salvaged = salvage_entries(text)
    if salvaged:
        if not quiet:
            print(f"  [salvage] output not parsable as a whole — recovered "
                  f"{len(salvaged)} record(s) from the truncated prefix.")
        return salvaged
    # Fallback: try bare array blocks
    candidates = re.findall(r"\[.*?\]", text, re.DOTALL)
    candidates_long = re.findall(r"\[.*\]", text, re.DOTALL)
    for candidate in sorted(set(candidates + candidates_long), key=len, reverse=True):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list) and len(obj) > 0 and all(isinstance(r, dict) for r in obj):
                return obj
        except Exception:
            pass
    if not quiet:
        print(f"  [warn] could not parse JSON from model output.\n"
              f"    first 300: {text[:300]}\n    last  200: {text[-200:]}")
    return []


def extract_country(text: str) -> str:
    """The top-level "country" field, or "" — extract_json only returns entries."""
    m = re.search(r'"country"\s*:\s*"((?:[^"\\]|\\.)*)"', _clean(text))
    if not m:
        return ""
    try:
        return json.loads(f'"{m.group(1)}"')
    except Exception:
        return m.group(1)


def generate_extraction(model, processor, inputs, max_new_tokens: int = 4096,
                        retry_repetition_penalty: float = 1.05, tag: str = "") -> str:
    """Greedy JSON generation with one automatic anti-degeneration retry.

    Normal case: plain greedy decoding. Failure case: on a few information-dense
    pages the fine-tuned model falls into a repetition loop, emits records until
    it hits ``max_new_tokens`` and the JSON never closes. That is detected here —
    output ran into the token cap, or nothing parsable came out — and the image
    is retried once with a mild repetition penalty (still deterministic). The
    attempt with more parsable records wins.
    """
    input_len = inputs["input_ids"].shape[-1]
    pad_id = getattr(processor.tokenizer, "eos_token_id", None)
    attempts = ({}, {"repetition_penalty": retry_repetition_penalty})
    best_text, best_n = "", -1
    for i, extra in enumerate(attempts):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=pad_id, **extra)
        n_new = out.shape[-1] - input_len
        text = processor.batch_decode(out[:, input_len:], skip_special_tokens=True)[0]
        n_entries = len(extract_json(text, quiet=True))
        if n_entries > best_n:
            best_text, best_n = text, n_entries
        hit_cap = n_new >= max_new_tokens
        if n_entries and not hit_cap:
            return text
        if i + 1 < len(attempts):
            print(f"  [degenerate]{tag} {n_new} new tokens"
                  f"{' (hit the ' + str(max_new_tokens) + '-token cap)' if hit_cap else ''}, "
                  f"{n_entries} parsable record(s) — retrying with "
                  f"repetition_penalty={retry_repetition_penalty}", flush=True)
        else:
            print(f"  [degenerate]{tag} retry gave {n_entries} record(s); "
                  f"keeping the best attempt ({best_n} record(s))", flush=True)
    return best_text
