"""Database verification of extracted records (the pipeline's "MCP refine" stage).

Copied from DIGIVAX finetuning/testing/test_qwen35_9b_lora_4gpu_mcp.py
(mcp_enrich, _needs_refine, surgical_merge, refine_and_enrich, infer_pass2_refine)
with the second VLM pass injected as a callable instead of a module-level model.

  enrich   — every sticker trade name is looked up in the PEI reference DB
             (ref_vaccines.db): manufacturer corrected, trade-name typos fixed
             (<= 2 edits), missing indications appended when unambiguous.
  pass 2   — optional (off unless a run_pass2 callable is given), and only when
             a lookup was ambiguous: the VLM is asked again with the DB context
             in the prompt, and ONLY trade_name / vaccine_name are spliced from
             its answer into the pass-1 records.
"""
import re
from typing import Callable, Optional

from .mcp_tools import (
    execute_tool, build_db_context, extract_trade_names_from_prediction,
    lev_similarity, best_lev_match, best_trade_name_match, db_indications, _token_match,
)
from .parse import extract_json

_SEASON_RE = re.compile(r"\s*\d{4}/\d{4}")


def _strip_season(name: str) -> str:
    return _SEASON_RE.sub("", name).strip()


def mcp_enrich(entries):
    """
    Two-step DB enrichment for a list of predicted vaccination entries.

    Step 1 – Levenshtein correction of unrecognized field values:
      • vaccine_name  – each item corrected against all DB indications if not token-matched
      • trade_name    – season-strip fallback first, then Levenshtein (max 2 edits)
      • manufacturer  – corrected to DB canonical when similarity > 50 % and lookup succeeded

    Step 2 – DB consistency:
      • lookup_vaccine per sticker → db_match annotation
      • check_vaccine_consistency  → confirmed / extra / missing
      • missing indications appended only when every named sticker resolved unambiguously

    Returns {"validated_entries": [...], "tool_calls": [...]}.
    """
    tool_calls = []
    enriched   = []
    _all_inds  = db_indications()

    for entry in entries:

        # ── Step 1a: correct vaccine_name items against DB indications ────────
        lev_corrections: list[tuple] = []
        corrected_vnames = []
        for vname in entry.get("vaccine_name", []):
            if not _token_match(vname, _all_inds):
                fixed = best_lev_match(vname, _all_inds)
                if fixed:
                    lev_corrections.append(("vaccine_name", vname, fixed))
                    corrected_vnames.append(fixed)
                    continue
            corrected_vnames.append(vname)
        predicted_vaccine_names = corrected_vnames

        # ── Step 1b + Step 2: per-sticker lookup ─────────────────────────────
        sticker_trade_names = []
        new_stickers = []

        for sticker in entry.get("stickers", []):
            trade_name = sticker.get("trade_name", "").strip()
            if not trade_name or trade_name.lower() == "n/a":
                new_stickers.append(sticker)
                continue

            result = execute_tool("lookup_vaccine", {"trade_name": trade_name})

            if not result.get("found"):
                # Season-strip fallback: "Vaxigrip Tetra 2020/2021" → "Vaxigrip Tetra"
                stripped = _strip_season(trade_name)
                has_season = stripped != trade_name
                if has_season:
                    result = execute_tool("lookup_vaccine", {"trade_name": stripped})
                # Levenshtein fallback only for genuine typos (<= 2 edits, shared
                # 4-char prefix). Skipped when the name carries a season year.
                if not result.get("found") and not has_season:
                    fixed = best_trade_name_match(trade_name)
                    if fixed:
                        lev_corrections.append(("trade_name", trade_name, fixed))
                        trade_name = fixed
                        result = execute_tool("lookup_vaccine", {"trade_name": trade_name})

            tool_calls.append({"tool": "lookup_vaccine",
                               "args": {"trade_name": trade_name}, "result": result})

            new_sticker = dict(sticker)
            new_sticker["trade_name"] = sticker.get("trade_name", trade_name)  # always keep original
            if result.get("found"):
                db_mfr = result["manufacturer"]
                sticker_mfr = sticker.get("manufacturer", "").strip()
                if sticker_mfr and sticker_mfr.lower() != "n/a":
                    if lev_similarity(sticker_mfr, db_mfr) > 50 and sticker_mfr.lower() != db_mfr.lower():
                        lev_corrections.append(("manufacturer", sticker_mfr, db_mfr))
                        new_sticker["manufacturer"] = db_mfr
                new_sticker["db_match"] = {
                    "canonical_trade_name": result["trade_name"],
                    "trade_name_ok":        result["trade_name"].lower() == trade_name.lower(),
                    "manufacturer":         db_mfr,
                    "indications":          result["indications"],
                    "approval_number":      result["approval_number"],
                    "approval_date":        result["approval_date"],
                }
            else:
                new_sticker["db_match"] = None
            sticker_trade_names.append(trade_name)
            new_stickers.append(new_sticker)

        # ── Step 2: consistency check + conditionally append missing ──────────
        consistency_args = {
            "trade_names":             sticker_trade_names,
            "predicted_vaccine_names": predicted_vaccine_names,
        }
        consistency = execute_tool("check_vaccine_consistency", consistency_args)
        tool_calls.append({"tool": "check_vaccine_consistency",
                           "args": consistency_args, "result": consistency})

        missing = consistency.get("missing", [])
        extra   = consistency.get("extra", [])
        # Append the DB's missing indications only when every named sticker in the
        # record resolved to an unambiguous DB row.
        named_stickers = [
            s for s in new_stickers
            if (s.get("trade_name") or "").strip().lower() not in ("", "n/a")
        ]
        all_matched_ok = bool(named_stickers) and all(
            s.get("db_match") and s["db_match"].get("trade_name_ok")
            for s in named_stickers
        )
        safe_additions = missing if all_matched_ok else []

        new_entry = dict(entry)
        new_entry["stickers"]      = new_stickers
        new_entry["vaccine_name"]  = predicted_vaccine_names + safe_additions
        new_entry["vaccine_name_check"] = {
            "db_indications_union": consistency.get("db_indications_union", []),
            "confirmed":            consistency.get("confirmed", []),
            "extra":                extra,
            "missing":              missing,
            "consistent":           consistency.get("consistent", None),
            "lev_corrections":      lev_corrections,
        }
        enriched.append(new_entry)

    return {"validated_entries": enriched, "tool_calls": tool_calls}


def needs_refine(enriched_entries: list) -> bool:
    """True when any sticker trade name failed or fuzzy-matched the DB lookup,
    or the consistency check flagged extra vaccine names."""
    for e in enriched_entries:
        if e.get("vaccine_name_check", {}).get("extra"):
            return True
        for s in e.get("stickers", []):
            tn = (s.get("trade_name") or "").strip()
            if not tn or tn.lower() == "n/a":
                continue
            dbm = s.get("db_match")
            if dbm is None or not dbm.get("trade_name_ok"):
                return True
    return False


def surgical_merge(entries1: list, entries2: list) -> tuple[list, int]:
    """Splice ONLY vaccine_name and stickers[].trade_name from pass-2 entries
    into the pass-1 records. Pass-2 entries are matched to pass-1 entries by
    date (each used once), falling back to list position. Returns the merged
    entries and the number of spliced fields."""
    merged, n_spliced = [], 0
    used: set[int] = set()

    def _match(i: int, e1: dict) -> Optional[int]:
        d1 = (e1.get("date") or "").strip()
        if d1:
            for j, e2 in enumerate(entries2):
                if j not in used and (e2.get("date") or "").strip() == d1:
                    return j
        return i if i < len(entries2) and i not in used else None

    for i, e1 in enumerate(entries1):
        j = _match(i, e1)
        if j is None:
            merged.append(e1)
            continue
        used.add(j)
        e2  = entries2[j]
        new = dict(e1)
        vn2 = e2.get("vaccine_name")
        if isinstance(vn2, list) and vn2 and vn2 != e1.get("vaccine_name"):
            new["vaccine_name"] = vn2
            n_spliced += 1
        st1, st2 = e1.get("stickers", []), e2.get("stickers", [])
        new_st = []
        for k, s1 in enumerate(st1):
            s = dict(s1)
            if k < len(st2) and isinstance(st2[k], dict):
                tn2 = (st2[k].get("trade_name") or "").strip()
                if tn2 and tn2.lower() != "n/a" and tn2 != s1.get("trade_name"):
                    s["trade_name"] = tn2
                    n_spliced += 1
            new_st.append(s)
        new["stickers"] = new_st
        merged.append(new)
    return merged, n_spliced


def refine_and_enrich(entries1: list, run_pass2: Optional[Callable[[str], str]]):
    """Enrich pass-1 entries; run pass 2 only when needed and merge it surgically.

    ``run_pass2(db_context)`` runs the VLM on the same image with the DB context
    injected and returns its raw text; None disables pass 2 (enrichment only).
    Returns (enrich_result, raw_pass2, refine_info).
    """
    enrich1 = mcp_enrich(entries1)
    if not needs_refine(enrich1["validated_entries"]):
        return enrich1, None, {"applied": False, "reason": "all DB lookups unambiguous"}
    if run_pass2 is None:
        return enrich1, None, {"applied": False, "reason": "pass 2 off (a lookup was ambiguous)"}
    db_context = build_db_context(extract_trade_names_from_prediction(entries1))
    if not db_context:
        return enrich1, None, {"applied": False, "reason": "no trade names to look up"}
    raw2     = run_pass2(db_context)
    entries2 = extract_json(raw2) if raw2 else []
    if not entries2:
        return enrich1, raw2, {"applied": False, "reason": "empty pass-2 output"}
    merged, n_spliced = surgical_merge(entries1, entries2)
    if not n_spliced:
        return enrich1, raw2, {"applied": False, "reason": "pass 2 confirmed pass 1"}
    return mcp_enrich(merged), raw2, {"applied": True, "n_fields_spliced": n_spliced}
