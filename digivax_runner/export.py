"""Write the user-facing outputs of one image from its full record.

    python -m digivax_runner.export <out_dir>     # (re)build all exports from technical/raw

    standard/<key>.json       the label: {country, entries[status, vaccine_name, stickers,
                              date, note]} — no bounding boxes, no DB-internal fields
    fhir/<key>.json           FHIR R4 Bundle (type collection) of Immunization resources
    digital_card/<key>.png    rendered vaccination card (needs Pillow)

FHIR mapping per record:
  status        completed → completed, not-done → not-done, unclear → completed
                (FHIR has no "unclear"; the note says so)
  vaccineCode   CVX code (http://hl7.org/fhir/sid/cvx) when the set of vaccine
                names matches exactly one CVX entry, otherwise text only; no names
                and no sticker → v3 NullFlavor UNK. Codes from the CDC CVX table.
  occurrence    occurrenceDateTime, or data-absent-reason "unknown" when the date
                is missing or not a valid date
  lotNumber / manufacturer   only when the record has exactly one sticker value
"""
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

from .prepare import read_images_tsv, done_path

STANDARD_ENTRY_KEYS = ("status", "vaccine_name", "stickers", "date", "note")
STICKER_KEYS = ("trade_name", "lot_no", "manufacturer")
MISSING = {"", "n/a", "na", "none", "unknown", "null"}

CVX_SYSTEM = "http://hl7.org/fhir/sid/cvx"
NULLFLAVOR_SYSTEM = "http://terminology.hl7.org/CodeSystem/v3-NullFlavor"
ABSENT_REASON_URL = "http://hl7.org/fhir/StructureDefinition/data-absent-reason"

# Disease tokens per vaccine name, matched on the lower-cased name. Order matters:
# "meningokokken b" before "meningokokken", "masern, mumps, röteln" before "masern".
_NAME_TOKENS = [
    (("mmr", "masern, mumps, röteln"), {"measles", "mumps", "rubella"}),
    (("masern",), {"measles"}), (("mumps",), {"mumps"}), (("röteln",), {"rubella"}),
    (("tetanus", "wundstarrkrampf"), {"tetanus"}), (("diphther",), {"diphtheria"}),
    (("pertussis", "keuchhusten", "azellulär"), {"pertussis"}),
    (("polio", "kinderlähmung"), {"polio"}), (("haemophilus", "hib"), {"hib"}),
    (("hepatitis b",), {"hepb"}), (("hepatitis a",), {"hepa"}),
    (("varizellen", "windpocken"), {"varicella"}),
    (("meningokokken acwy",), {"men_acwy"}), (("meningokokken b",), {"men_b"}),
    (("meningokokken c",), {"men_c"}), (("meningokokken a",), {"men_a"}),
    (("meningokokken",), {"men"}),
    (("pneumokokken",), {"pneumo"}), (("rotavirus",), {"rota"}), (("influenza", "grippe"), {"flu"}),
    (("covid",), {"covid"}), (("zoster", "gürtelrose"), {"zoster"}), (("papillom", "hpv"), {"hpv"}),
    (("fsme",), {"tbe"}), (("gelbfieber",), {"yellow_fever"}), (("tollwut",), {"rabies"}),
    (("typhus",), {"typhoid"}), (("japanische enzephalitis",), {"je"}),
    (("cholera",), {"cholera"}), (("dengue",), {"dengue"}), (("ebola",), {"ebola"}),
    (("pocken", "mpox"), {"mpox"}), (("milzbrand", "anthrax"), {"anthrax"}),
    (("synzytial", "rsv"), {"rsv"}), (("chikungunya",), {"chikungunya"}),
]

# Exact disease set → (CVX code, CVX short description). "Unspecified formulation"
# codes, because the booklet shows diseases, not the formulation. Sets that map to
# several products (T+D: DT or Td; T+D+P: DTaP or Tdap; T+D+P+IPV: DTaP-IPV or
# Tdap-IPV) are left uncoded rather than guessed.
_CVX = {
    frozenset({"tetanus"}): ("112", "tetanus toxoid, unspecified formulation"),
    frozenset({"pertussis"}): ("11", "pertussis"),
    frozenset({"polio"}): ("89", "polio, unspecified formulation"),
    frozenset({"hib"}): ("17", "Hib, unspecified formulation"),
    frozenset({"hepb"}): ("45", "Hep B, unspecified formulation"),
    frozenset({"hepa"}): ("85", "Hep A, unspecified formulation"),
    frozenset({"hepa", "hepb"}): ("104", "Hep A-Hep B"),
    frozenset({"tetanus", "diphtheria", "pertussis", "hib"}): ("50", "DTaP-Hib"),
    frozenset({"tetanus", "diphtheria", "pertussis", "hib", "polio"}): ("120", "DTaP-Hib-IPV"),
    frozenset({"tetanus", "diphtheria", "pertussis", "polio", "hepb"}): ("110", "DTaP-Hep B-IPV"),
    frozenset({"tetanus", "diphtheria", "pertussis", "hib", "polio", "hepb"}):
        ("146", "DTaP,IPV,Hib,HepB"),
    frozenset({"measles", "mumps", "rubella"}): ("03", "MMR"),
    frozenset({"measles", "mumps", "rubella", "varicella"}): ("94", "MMRV"),
    frozenset({"measles"}): ("05", "measles"),
    frozenset({"mumps"}): ("07", "mumps"),
    frozenset({"rubella"}): ("06", "rubella"),
    frozenset({"measles", "rubella"}): ("04", "M/R"),
    frozenset({"mumps", "rubella"}): ("38", "rubella/mumps"),
    frozenset({"varicella"}): ("21", "varicella"),
    frozenset({"men_acwy"}): ("108", "meningococcal ACWY, unspecified formulation"),
    frozenset({"men_b"}): ("164", "meningococcal B, unspecified"),
    frozenset({"men_c"}): ("103", "meningococcal C conjugate"),
    frozenset({"pneumo"}): ("109", "pneumococcal, unspecified formulation"),
    frozenset({"rota"}): ("122", "rotavirus, unspecified formulation"),
    frozenset({"flu"}): ("88", "influenza, unspecified formulation"),
    frozenset({"covid"}): ("213", "SARS-COV-2 (COVID-19) vaccine, UNSPECIFIED"),
    frozenset({"zoster"}): ("188", "zoster, unspecified formulation"),
    frozenset({"hpv"}): ("137", "HPV, unspecified formulation"),
    frozenset({"tbe"}): ("222", "Tick-borne encephalitis, unspecified"),
    frozenset({"yellow_fever"}): ("184", "Yellow fever, unspecified"),
    frozenset({"rabies"}): ("90", "rabies, unspecified formulation"),
    frozenset({"typhoid"}): ("91", "typhoid, unspecified formulation"),
    frozenset({"je"}): ("129", "Japanese Encephalitis, unspecified formulation"),
    frozenset({"cholera"}): ("26", "cholera, unspecified formulation"),
    frozenset({"dengue"}): ("330", "Dengue fever, unspecified"),
    frozenset({"ebola"}): ("214", "Ebola, unspecified"),
    frozenset({"mpox"}): ("325", "Vaccinia (smallpox, monkeypox), unspecified"),
    frozenset({"anthrax"}): ("319", "Anthrax vaccine, unspecified"),
    frozenset({"rsv"}): ("314", "Respiratory syncytial virus (RSV) vaccine, unspecified"),
}


def _present(value) -> bool:
    return isinstance(value, str) and value.strip().lower() not in MISSING


def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _stickers(entry: dict) -> list[dict]:
    return [s for s in entry.get("stickers") or [] if isinstance(s, dict)]


def _names(entry: dict) -> list[str]:
    return [n for n in entry.get("vaccine_name") or [] if isinstance(n, str) and n.strip()]


# --- standard --------------------------------------------------------------------

def to_standard(record: dict) -> dict:
    return {
        "country": record.get("country") or "n/a",
        "entries": [
            {**{k: e.get(k, [] if k in ("vaccine_name", "stickers") else "n/a")
                for k in STANDARD_ENTRY_KEYS},
             "stickers": [{k: s.get(k, "n/a") for k in STICKER_KEYS} for s in _stickers(e)]}
            for e in record.get("entries") or []
        ],
    }


# --- FHIR --------------------------------------------------------------------------

def cvx_coding(names: list[str]):
    # A name may list several diseases ("Tetanus, Diphtherie, Pertussis"). Each match
    # is cut out of the string so that "Windpocken" does not also hit "pocken" and
    # "Haemophilus influenzae" does not also hit "influenza".
    tokens = set()
    for name in names:
        low, hit = name.lower(), False
        for keys, toks in _NAME_TOKENS:
            for k in keys:
                if k in low:
                    low, hit = low.replace(k, " "), True
                    tokens |= toks
        if not hit:
            return None                      # unknown name → do not guess a code
    code = _CVX.get(frozenset(tokens))
    return {"system": CVX_SYSTEM, "code": code[0], "display": code[1]} if code else None


def _valid_date(value) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", value):
        return False
    parts = [int(p) for p in value.split("-")] + [1, 1]
    try:
        date(parts[0], parts[1], parts[2])
    except ValueError:
        return False
    return True


def _immunization(entry: dict) -> dict:
    names, stickers = _names(entry), _stickers(entry)
    trades = [s["trade_name"] for s in stickers if _present(s.get("trade_name"))]
    lots = [s["lot_no"] for s in stickers if _present(s.get("lot_no"))]
    makers = sorted({s["manufacturer"] for s in stickers if _present(s.get("manufacturer"))})
    status = entry.get("status")

    if trades and names:
        text = f"{' / '.join(trades)} ({', '.join(names)})"
    else:
        text = " / ".join(trades) or ", ".join(names)
    vaccine_code = {"text": text} if text else {"text": "unknown"}
    coding = cvx_coding(names) if names else None
    if coding:
        vaccine_code = {"coding": [coding], **vaccine_code}
    elif not text:
        vaccine_code["coding"] = [{"system": NULLFLAVOR_SYSTEM, "code": "UNK", "display": "unknown"}]

    res = {
        "resourceType": "Immunization",
        "status": "not-done" if status == "not-done" else "completed",
        "patient": {"display": "Anonymous"},
        "vaccineCode": vaccine_code,
    }
    d = entry.get("date")
    if _valid_date(d):
        res["occurrenceDateTime"] = d
    else:
        res["_occurrenceDateTime"] = {
            "extension": [{"url": ABSENT_REASON_URL, "valueCode": "unknown"}]}
    if len(lots) == 1:
        res["lotNumber"] = lots[0]
    if len(makers) == 1:
        res["manufacturer"] = {"display": makers[0]}

    notes = []
    if status == "unclear":
        notes.append("Entry marked as unclear in the booklet transcription.")
    if _present(d) and not _valid_date(d):
        notes.append(f"Date as read: {d}")
    if len(lots) > 1:
        notes.append("Stickers: " + "; ".join(
            f"{s.get('trade_name', 'n/a')} (lot {s.get('lot_no', 'n/a')})" for s in stickers))
    if _present(entry.get("note")):
        notes.append(entry["note"])
    if notes:
        res["note"] = [{"text": " ".join(notes)}]
    return res


def to_fhir(record: dict) -> dict:
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": _immunization(e)} for e in record.get("entries") or []],
    }


# --- all exports -------------------------------------------------------------------

def write_exports(out_dir: Path, key: str, record: dict):
    from .card import render_card       # Pillow only needed here

    out_dir = Path(out_dir)
    _write_text(out_dir / "standard" / f"{key}.json",
                json.dumps(to_standard(record), indent=2, ensure_ascii=False) + "\n")
    _write_text(out_dir / "fhir" / f"{key}.json",
                json.dumps(to_fhir(record), indent=2, ensure_ascii=False) + "\n")
    card = out_dir / "digital_card" / f"{key}.png"
    card.parent.mkdir(parents=True, exist_ok=True)
    tmp = card.with_name(f".{card.name}.{os.getpid()}.tmp")
    render_card(record).save(tmp, format="PNG", optimize=True)
    os.replace(tmp, card)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python -m digivax_runner.export <out_dir>")
    out_dir = Path(sys.argv[1])
    n = 0
    for key, _ in read_images_tsv(out_dir):
        p = done_path(out_dir, key)
        if p.exists():
            write_exports(out_dir, key, json.loads(p.read_text(encoding="utf-8")))
            n += 1
    print(f"exported {n} image(s) to {out_dir}/{{standard,fhir,digital_card}}")


if __name__ == "__main__":
    main()
