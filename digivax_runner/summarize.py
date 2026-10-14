"""Collect per-image records into technical/results.csv, summary.json, failed.txt (stdlib only).

    python -m digivax_runner.summarize <out_dir>

Safe to run repeatedly and from several array tasks at once: every file is
written to a temp name and atomically renamed.
"""
import csv
import io
import json
import os
import sys
from pathlib import Path

from .prepare import done_path, read_images_tsv, tech_dir

CSV_COLUMNS = [
    "image", "record", "sticker", "date", "status", "vaccine_name",
    "trade_name", "lot_no", "manufacturer",
    "db_trade_name", "db_trade_name_ok", "db_indications",
    "vaccine_name_consistent", "country", "note",
]


def _atomic_write(path: Path, text: str, encoding: str = "utf-8"):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def _join(values) -> str:
    if isinstance(values, list):
        return "; ".join(str(v) for v in values)
    return "" if values is None else str(values)


def _rows(pred: dict):
    """One CSV row per sticker; a record without stickers still gets one row."""
    image = Path(pred.get("image", "")).name
    for r_idx, entry in enumerate(pred.get("entries", []), 1):
        check = entry.get("vaccine_name_check") or {}
        base = {
            "image": image, "record": r_idx,
            "date": entry.get("date", ""), "status": entry.get("status", ""),
            "vaccine_name": _join(entry.get("vaccine_name", [])),
            "vaccine_name_consistent": _join(check.get("consistent")),
            "country": pred.get("country", ""), "note": entry.get("note", ""),
        }
        stickers = [s for s in entry.get("stickers", []) if isinstance(s, dict)] or [{}]
        for s_idx, s in enumerate(stickers, 1):
            dbm = s.get("db_match") or {}
            yield {**base,
                   "sticker": s_idx if s else "",
                   "trade_name": s.get("trade_name", ""), "lot_no": s.get("lot_no", ""),
                   "manufacturer": s.get("manufacturer", ""),
                   "db_trade_name": dbm.get("canonical_trade_name", ""),
                   "db_trade_name_ok": _join(dbm.get("trade_name_ok")) if dbm else "",
                   "db_indications": _join(dbm.get("indications", [])) if dbm else ""}


def summarize(out_dir: Path) -> dict:
    tech = tech_dir(out_dir)
    selected = read_images_tsv(out_dir)
    fail_dir = tech / "failed"

    preds, failed, missing = [], [], []
    for key, path in selected:
        p = done_path(out_dir, key)
        if p.exists():
            preds.append(json.loads(p.read_text(encoding="utf-8")))
        elif (fail_dir / f"{key}.json").exists():
            err = json.loads((fail_dir / f"{key}.json").read_text(encoding="utf-8"))
            failed.append((path, err.get("error", "?")))
        else:
            missing.append(path)

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    w.writeheader()
    for pred in preds:
        w.writerows(_rows(pred))
    # utf-8-sig so Excel shows umlauts (Röteln, Masern …) correctly
    _atomic_write(tech / "results.csv", buf.getvalue(), encoding="utf-8-sig")

    _atomic_write(tech / "failed.txt",
                  "".join(f"{path}\t{err}\n" for path, err in failed)
                  + "".join(f"{path}\tnot processed\n" for path in missing))

    n_pass2 = sum(1 for p in preds if (p.get("verification") or {}).get("pass2_applied"))
    summary = {
        "images_selected":   len(selected),
        "images_done":       len(preds),
        "images_failed":     len(failed),
        "images_not_processed": len(missing),
        "images_no_records": sum(1 for p in preds if not p.get("entries")),
        "records":           sum(len(p.get("entries", [])) for p in preds),
        "images_verified":   sum(1 for p in preds if p.get("verification")),
        "pass2_applied":     n_pass2,
        "seconds_total":     round(sum((p.get("timing_s") or {}).get("total", 0) for p in preds), 1),
    }
    _atomic_write(tech / "summary.json", json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python -m digivax_runner.summarize <out_dir>")
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))


if __name__ == "__main__":
    main()
