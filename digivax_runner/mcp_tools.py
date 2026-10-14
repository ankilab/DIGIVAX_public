"""
MCP tool definitions and implementations for the vaccination reference database.

Import in any test script:
    from vaccination_db.mcp_tools import TOOL_SCHEMAS, execute_tool, build_db_context

Exposed tools:
  lookup_vaccine(trade_name)                 – vaccine info + indications, fuzzy match
  search_vaccines_by_indication(indication)  – all vaccines for a disease
  validate_trade_names(trade_names)          – batch match extracted names against DB
  list_indications()                         – all known indication names
"""

import functools
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).parent / "ref_vaccines.db"

# ---------------------------------------------------------------------------
# Tool schemas — compatible with Anthropic MCP / OpenAI function-calling format
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "check_vaccine_consistency",
        "description": (
            "Cross-check a vaccination record's predicted vaccine names against the "
            "disease indications known for its sticker trade names in the reference DB. "
            "Returns which predicted names are DB-confirmed, which are extra (hallucinated "
            "or OCR error), and which DB indications are missing from the prediction. "
            "Also validates each trade name (canonical form) and reports the manufacturer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "trade_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Trade names of all stickers in this vaccination record.",
                },
                "predicted_vaccine_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Vaccine name list extracted by the VLM for this record.",
                },
            },
            "required": ["trade_names", "predicted_vaccine_names"],
        },
    },
    {
        "name": "lookup_vaccine",
        "description": (
            "Look up a single vaccine trade name in the PEI reference database. "
            "Returns manufacturer, vaccine type, approval info, and all disease indications. "
            "Supports fuzzy matching so minor OCR errors are tolerated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "trade_name": {
                    "type": "string",
                    "description": "Trade name of the vaccine as read from the certificate.",
                }
            },
            "required": ["trade_name"],
        },
    },
    {
        "name": "search_vaccines_by_indication",
        "description": (
            "Find all vaccines in the database approved for a given disease or indication "
            "(e.g. 'Tetanus', 'Influenza', 'Hepatitis B', 'Masern')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "indication": {
                    "type": "string",
                    "description": "Disease / indication name to search for.",
                }
            },
            "required": ["indication"],
        },
    },
    {
        "name": "validate_trade_names",
        "description": (
            "Validate a batch of vaccine trade names extracted from a certificate. "
            "Returns which names are known in the DB and their canonical forms, "
            "and which are unrecognised (potential OCR errors or unknown vaccines)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "trade_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of trade names to validate.",
                }
            },
            "required": ["trade_names"],
        },
    },
    {
        "name": "list_indications",
        "description": "List all disease indications available in the vaccination reference database.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _connect():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _normalize(name: str) -> str:
    """Lowercase, strip whitespace/punctuation for fuzzy matching."""
    return re.sub(r"[\s\-_.,;:()]+", "", name.strip().lower())


def _fuzzy_query(conn: sqlite3.Connection, trade_name: str) -> Optional[sqlite3.Row]:
    norm = _normalize(trade_name)
    # Exact normalised match first
    row = conn.execute(
        "SELECT * FROM vaccines WHERE trade_name_norm = ?", (norm,)
    ).fetchone()
    if row:
        return row
    # Prefix match
    row = conn.execute(
        "SELECT * FROM vaccines WHERE trade_name_norm LIKE ?",
        (f"{norm}%",),
    ).fetchone()
    if row and len(norm) >= len(row["trade_name_norm"]) * 0.6:
        return row
    # Substring match — only trust if query covers ≥60% of candidate length
    rows = conn.execute(
        "SELECT * FROM vaccines WHERE trade_name_norm LIKE ?",
        (f"%{norm}%",),
    ).fetchall()
    for row in rows:
        if len(norm) >= len(row["trade_name_norm"]) * 0.6:
            return row
    return None


def _indications_for(conn, vaccine_id):
    rows = conn.execute(
        """SELECT i.name FROM indications i
           JOIN vaccine_indications vi ON vi.indication_id = i.id
           WHERE vi.vaccine_id = ?""",
        (vaccine_id,),
    ).fetchall()
    return [r["name"] for r in rows]


def _vaccine_row_to_dict(conn, row):
    mfr = conn.execute(
        "SELECT name FROM manufacturers WHERE id = ?", (row["manufacturer_id"],)
    ).fetchone()
    return {
        "trade_name": row["trade_name"],
        "manufacturer": mfr["name"] if mfr else "unknown",
        "vaccine_type": row["vaccine_type"],
        "approval_number": row["approval_number"],
        "approval_date": row["approval_date"],
        "indications": _indications_for(conn, row["id"]),
    }

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def lookup_vaccine(trade_name):
    """Return DB info for a single trade name, or a not-found dict."""
    with _connect() as conn:
        row = _fuzzy_query(conn, trade_name)
        if row is None:
            return {"found": False, "query": trade_name}
        result = _vaccine_row_to_dict(conn, row)
        result["found"] = True
        result["query"] = trade_name
        return result


def search_vaccines_by_indication(indication):
    """Return all vaccines matching an indication (case-insensitive substring)."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT v.* FROM vaccines v
               JOIN vaccine_indications vi ON vi.vaccine_id = v.id
               JOIN indications i ON i.id = vi.indication_id
               WHERE LOWER(i.name) LIKE ?""",
            (f"%{indication.lower()}%",),
        ).fetchall()
        vaccines = [_vaccine_row_to_dict(conn, r) for r in rows]
    return {"indication_query": indication, "count": len(vaccines), "vaccines": vaccines}


def validate_trade_names(trade_names):
    """Batch-validate a list of trade names against the DB."""
    matched, unmatched = [], []
    with _connect() as conn:
        for name in trade_names:
            row = _fuzzy_query(conn, name)
            if row:
                matched.append({
                    "query": name,
                    "canonical_trade_name": row["trade_name"],
                    "indications": _indications_for(conn, row["id"]),
                })
            else:
                unmatched.append(name)
    return {
        "matched": matched,
        "unmatched": unmatched,
        "match_rate": len(matched) / len(trade_names) if trade_names else 0.0,
    }


def list_indications():
    """Return all indication names in the DB, sorted."""
    with _connect() as conn:
        rows = conn.execute("SELECT name FROM indications ORDER BY name").fetchall()
    return {"indications": [r["name"] for r in rows]}


# Token-based vaccine name matching so "Hib" matches
# "Haemophilus influenzae Typ b (Hib)", etc.
_VACCINE_SYNONYMS = {
    "hib":            "haemophilusinfluenzaetypb",
    "mmr":            "masernmumpsroeteln",
    "mmrv":           "masernmumpsroetelnvarizellen",
    "fsme":           "fruehsommermeningoenzephalitis",
    "rsv":            "respiratorischersynzytialvirus",
    "hpv":            "humanerpapillomvirus",
    "diphtheria":     "diphtherie",
    "pertussis":      "keuchhusten",
    "varicella":      "varizellen",
    "measles":        "masern",
    "rubella":        "roeteln",
    "meningococcal":  "meningokokken",
    "pneumococcal":   "pneumokokken",
    "polio":          "poliomyelitis",
    # Everyday German vs. the clinical vocabulary the PEI register uses. Certificates
    # are written either way and the ground truth follows the certificate, so without
    # these the same disease reads as a disagreement — see the vocabulary-drift figure
    # in finetuning/testing/analysis_mcp/.
    "kinderlaehmung":   "poliomyelitis",
    "ipv":              "poliomyelitis",
    "opv":              "poliomyelitis",
    "wundstarrkrampf":  "tetanus",
    "starrkrampf":      "tetanus",
    "keuchhusten":      "keuchhusten",
    "morbilli":         "masern",
    "parotitis":        "mumps",
    "ziegenpeter":      "mumps",
    "rubeola":          "roeteln",
    "windpocken":       "varizellen",
    "guertelrose":      "herpeszoster",
    "zoster":           "herpeszoster",
    "shingles":         "herpeszoster",
    "rabies":           "tollwut",
    "lyssa":            "tollwut",
    "grippe":           "influenza",
    "typhoid":          "typhus",
    "anthrax":          "milzbrand",
    "variola":          "pocken",
    "mpox":             "pocken",
    "smallpox":         "pocken",
    "tbe":              "fruehsommermeningoenzephalitis",
    "hepatitisinfektiosa": "hepatitis",   # old certificate wording for hepatitis A/B
}


# Word-tokens that name a disease *family* rather than a disease. Emitting them
# alongside the full name makes "Hepatitis A" and "Hepatitis B" overlap on
# "hepatitis" and compare equal, which silently suppressed the DB correction in
# every A-vs-B mix-up. They are dropped whenever the full name says more than the
# stem alone — a record that really only says "Hepatitis" still keeps the token.
#
# Deliberately only "hepatitis". "meningokokken" was tried here too and measured
# clearly worse (+0.0008 vs +0.0027 indications_f1): serotype labels are used
# loosely on certificates, so splitting "Meningokokken" from "Meningokokken C"
# makes the DB append a serotype the ground truth does not carry. Anything added
# to this set must be re-measured with analysis_mcp/verify_fix.py.
_AMBIGUOUS_STEMS = {"hepatitis"}


def _vac_tokens(names):
    """Normalise a vaccine name list to a set of comparable tokens."""
    tokens = set()
    for name in names:
        # lower, remove special chars, split on non-alnum
        cleaned = re.sub(r"[^a-z0-9]+", "", name.lower().replace("ä", "ae")
                         .replace("ö", "oe").replace("ü", "ue").replace("ß", "ss"))
        tokens.add(cleaned)
        # also add each word individually so "Masern, Mumps, Röteln (MMR)" splits nicely
        for tok in re.split(r"[^a-zäöüß0-9]+", name.lower()):
            t = re.sub(r"[^a-z0-9]", "", tok.replace("ä", "ae").replace("ö", "oe")
                       .replace("ü", "ue").replace("ß", "ss"))
            if len(t) > 2:
                t = _VACCINE_SYNONYMS.get(t, t)
                if t in _AMBIGUOUS_STEMS and cleaned != t:
                    continue
                tokens.add(t)
    return tokens - {"", "na", "nva"}


def _token_match(pred_name, db_indications):
    """True if a predicted vaccine name token-overlaps with any DB indication."""
    pred_toks = _vac_tokens([pred_name])
    for ind in db_indications:
        if pred_toks & _vac_tokens([ind]):
            return True
    return False


def check_vaccine_consistency(trade_names, predicted_vaccine_names):
    """
    For a single vaccination record (entry):
    1. Look up each sticker trade name → canonical name, manufacturer, indications
    2. Build the union of DB indications across all found stickers
    3. Cross-check predicted vaccine_names against that union (token-based)

    Returns:
      sticker_checks     – per-sticker: found, canonical_trade_name, manufacturer, indications,
                           trade_name_ok (predicted == canonical, case-insensitive)
      db_indications_union – all indications expected for these stickers
      confirmed          – predicted names that match a DB indication
      extra              – predicted names NOT covered by any sticker in the DB
                           (possible hallucination or unlisted vaccine)
      missing            – DB indications not found in the prediction
      consistent         – True only when extra==[] and missing==[]
    """
    sticker_checks = []
    db_indications_union = []
    seen_inds = set()

    with _connect() as conn:
        for tn in trade_names:
            row = _fuzzy_query(conn, tn)
            if row is None:
                sticker_checks.append({
                    "query": tn,
                    "found": False,
                    "canonical_trade_name": None,
                    "trade_name_ok": False,
                    "manufacturer": None,
                    "indications": [],
                })
                continue
            inds = _indications_for(conn, row["id"])
            mfr  = conn.execute(
                "SELECT name FROM manufacturers WHERE id = ?", (row["manufacturer_id"],)
            ).fetchone()
            canonical = row["trade_name"]
            sticker_checks.append({
                "query": tn,
                "found": True,
                "canonical_trade_name": canonical,
                "trade_name_ok": _normalize(tn) == _normalize(canonical),
                "manufacturer": mfr["name"] if mfr else "unknown",
                "indications": inds,
            })
            for ind in inds:
                key = ind.lower()
                if key not in seen_inds:
                    db_indications_union.append(ind)
                    seen_inds.add(key)

    confirmed, extra = [], []
    for pname in predicted_vaccine_names:
        if _token_match(pname, db_indications_union):
            confirmed.append(pname)
        else:
            extra.append(pname)

    missing = [
        ind for ind in db_indications_union
        if not _token_match(ind, predicted_vaccine_names)
    ]

    return {
        "sticker_checks":        sticker_checks,
        "db_indications_union":  db_indications_union,
        "confirmed":             confirmed,
        "extra":                 extra,
        "missing":               missing,
        "consistent":            len(extra) == 0 and len(missing) == 0,
    }


# ---------------------------------------------------------------------------
# Levenshtein correction helpers
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def lev_similarity(a: str, b: str) -> float:
    """(1 − dist/max_len) × 100.  Returns 100.0 if both strings are empty."""
    ml = max(len(a), len(b))
    return (1 - _levenshtein(a.lower(), b.lower()) / ml) * 100 if ml else 100.0


def best_lev_match(value: str, candidates, threshold: float = 50.0, max_abs_dist: int = 4,
                   min_prefix: int = 0) -> Optional[str]:
    """Return the highest-similarity candidate within max_abs_dist edits and above threshold, or None.

    min_prefix additionally requires that many leading characters to be identical.
    Use it whenever the candidate space is short, morphologically similar strings —
    see best_trade_name_match below for why.
    """
    best, top = None, 0.0
    for c in candidates:
        if abs(len(value) - len(c)) > max_abs_dist:
            continue
        if min_prefix and _shared_prefix(value, c) < min_prefix:
            continue
        dist = _levenshtein(value.lower(), c.lower())
        if dist > max_abs_dist:
            continue
        s = lev_similarity(value, c)
        if s > top:
            top, best = s, c
    return best if top > threshold else None


def _shared_prefix(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a.lower(), b.lower()):
        if ca != cb:
            break
        n += 1
    return n


# Fuzzy matching of trade names needs far tighter bounds than the generic default.
# Vaccine brand names are short (7-10 chars) and share morphology (-rix, -ix), so at
# the old 4-edit / 50 % setting entirely different products were neighbours and got
# silently substituted, taking their indications with them:
#   Twinrix (Hep A+B) -> Shingrix (shingles)      distance 3, 21 records
#   VAXIGRIP (influenza) -> Varilrix (varicella)  distance 3, 21 records
#   Pandemrix (H1N1) -> Fendrix (Hep B)           distance 4,  7 records
#   FSME-IMMUN (TBE) -> Td-IMMUN (tetanus/diph.)  distance 4,  5 records
# Genuine OCR typos sit at distance 1 and keep their opening characters
# (Prioris -> Priorix, Boosterix -> Boostrix), so distance <= 2 plus a 4-character
# shared prefix keeps every real fix and drops every brand confusion above.
# Measured in finetuning/testing/analysis_mcp/tuning.md.
TRADE_NAME_MAX_DIST = 2
TRADE_NAME_MIN_PREFIX = 4


def best_trade_name_match(value: str) -> Optional[str]:
    """Nearest DB trade name that is plausibly the same product misread, or None."""
    return best_lev_match(value, db_trade_names(),
                          max_abs_dist=TRADE_NAME_MAX_DIST,
                          min_prefix=TRADE_NAME_MIN_PREFIX)


@functools.lru_cache(maxsize=1)
def db_trade_names() -> tuple:
    with _connect() as conn:
        return tuple(r["trade_name"] for r in conn.execute("SELECT trade_name FROM vaccines").fetchall())


@functools.lru_cache(maxsize=1)
def db_manufacturers() -> tuple:
    with _connect() as conn:
        return tuple(r["name"] for r in conn.execute("SELECT name FROM manufacturers").fetchall())


@functools.lru_cache(maxsize=1)
def db_indications() -> tuple:
    with _connect() as conn:
        return tuple(r["name"] for r in conn.execute("SELECT name FROM indications").fetchall())


# ---------------------------------------------------------------------------
# Dispatcher — the single entry point for any caller
# ---------------------------------------------------------------------------

_TOOL_FN = {
    "check_vaccine_consistency":    lambda args: check_vaccine_consistency(**args),
    "lookup_vaccine":               lambda args: lookup_vaccine(**args),
    "search_vaccines_by_indication": lambda args: search_vaccines_by_indication(**args),
    "validate_trade_names":         lambda args: validate_trade_names(**args),
    "list_indications":             lambda args: list_indications(),
}


def execute_tool(tool_name, tool_args):
    """Dispatch a tool call by name and return its result dict."""
    fn = _TOOL_FN.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return fn(tool_args)
    except Exception as exc:
        return {"error": str(exc), "tool": tool_name, "args": tool_args}

# ---------------------------------------------------------------------------
# Prompt helpers — inject DB context into model prompts
# ---------------------------------------------------------------------------

def build_db_context(trade_names):
    """
    Given a list of trade names (extracted from a first-pass inference),
    return a short text block summarising the DB lookup results.
    Ready to inject into a follow-up prompt.
    """
    if not trade_names:
        return ""
    result = validate_trade_names(trade_names)
    lines = ["=== Vaccination DB Reference ==="]
    for m in result["matched"]:
        inds = ", ".join(m["indications"]) or "—"
        lines.append(
            f"• {m['query']!r} → canonical: {m['canonical_trade_name']!r}  |  indications: {inds}"
        )
    for u in result["unmatched"]:
        lines.append(f"• {u!r} → NOT FOUND in reference DB (possible OCR error or unknown vaccine)")
    lines.append("=================================")
    return "\n".join(lines)


def extract_trade_names_from_prediction(entries):
    """Pull all sticker trade_names out of a parsed prediction."""
    names = []
    for entry in entries:
        for sticker in entry.get("stickers", []):
            tn = sticker.get("trade_name", "").strip()
            if tn and tn.lower() != "n/a":
                names.append(tn)
    return names


if __name__ == "__main__":
    import json, sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list_indications"
    args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    print(json.dumps(execute_tool(cmd, args), indent=2, ensure_ascii=False))
