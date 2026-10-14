"""Render a record as a digital vaccination card (PNG, Pillow only).

Entries are sorted by date (unreadable dates last). Every row shows date, the
vaccines it protects against, product(s) with manufacturer, lot number(s) and
status; a record's note is printed under its row.
"""
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .export import _names, _present, _stickers, _valid_date

WIDTH, MARGIN, PAD = 1400, 40, 40
CONTENT_W = WIDTH - 2 * MARGIN - 2 * PAD
COLUMNS = [("Date", 160), ("Vaccination against", 400), ("Product", 340), ("Lot", 170), ("Status", 170)]

PAGE_BG, CARD_BG = "#E6ECEF", "#FFFFFF"
HEADER_BG, HEADER_FG, HEADER_SUB = "#1F4E5F", "#FFFFFF", "#CFE3EA"
TEXT, MUTED, RULE, ZEBRA = "#1A1F24", "#5F6B73", "#D5DDE1", "#F4F7F8"
STATUS_STYLE = {                      # label, chip background, chip text
    "completed": ("completed", "#E3F1E5", "#1E6B2A"),
    "unclear":   ("unclear",   "#FBEFD9", "#8A5300"),
    "not-done":  ("not done",  "#ECEFF1", "#55616A"),
}

_FONT_DIRS = ["/usr/share/fonts/dejavu-sans-fonts", "/usr/share/fonts/dejavu",
              "/usr/share/fonts/truetype/dejavu"]


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in _FONT_DIRS:
        if (Path(d) / name).exists():
            return ImageFont.truetype(str(Path(d) / name), size)
    return ImageFont.load_default(size=size)


F_TITLE, F_SUB = _font(44, True), _font(22)
F_HEAD, F_CELL, F_CELL_B, F_SMALL = _font(18, True), _font(21), _font(21, True), _font(17)


def _wrap(draw, text: str, font, width: int) -> list[str]:
    lines = []
    for para in str(text).split("\n"):
        line = ""
        for word in para.split():
            cand = f"{line} {word}".strip()
            if draw.textlength(cand, font=font) <= width or not line:
                line = cand
            else:
                lines.append(line)
                line = word
            while draw.textlength(line, font=font) > width and len(line) > 1:   # very long word
                cut = len(line)
                while cut > 1 and draw.textlength(line[:cut], font=font) > width:
                    cut -= 1
                lines.append(line[:cut])
                line = line[cut:]
        lines.append(line)
    return lines


def _line_h(font) -> int:
    asc, desc = font.getmetrics()
    return asc + desc + 6


def _fmt_date(value) -> str:
    if _valid_date(value):
        parts = value.split("-")
        return ".".join(reversed(parts))          # 2021-07-12 → 12.07.2021
    return value if _present(value) else "unknown"


def _cells(entry: dict) -> list[list[tuple[str, object, str]]]:
    """Per column: list of (text, font, colour) lines before wrapping."""
    stickers = _stickers(entry)
    product, lots = [], []
    for s in stickers:
        product.append((s.get("trade_name") if _present(s.get("trade_name")) else "unnamed sticker",
                        F_CELL_B, TEXT))
        if _present(s.get("manufacturer")):
            product.append((s["manufacturer"], F_SMALL, MUTED))
        lots.append((s["lot_no"], F_CELL, TEXT) if _present(s.get("lot_no")) else ("–", F_CELL, MUTED))
    names = _names(entry)
    return [
        [(_fmt_date(entry.get("date")), F_CELL_B, TEXT)],
        [(n, F_CELL, TEXT) for n in names] or [("not specified", F_CELL, MUTED)],
        product or [("no sticker", F_CELL, MUTED)],
        lots or [("–", F_CELL, MUTED)],
        [],                                        # status chip drawn separately
    ]


def _sort_key(entry: dict):
    d = entry.get("date")
    return (0, d) if _valid_date(d) else (1, "")


def render_card(record: dict) -> Image.Image:
    entries = sorted(record.get("entries") or [], key=_sort_key)
    scratch = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    gap = 16

    # --- lay out rows first, to know the height -------------------------------
    rows = []
    for e in entries:
        cols = []
        for (_, w), lines in zip(COLUMNS, _cells(e)):
            wrapped = [(t, f, c) for text, f, c in lines for t in _wrap(scratch, text, f, w - gap)]
            cols.append(wrapped)
        note = e.get("note")
        note_lines = _wrap(scratch, f"Note: {note}", F_SMALL, CONTENT_W - 2 * 12) if _present(note) else []
        body_h = max([sum(_line_h(f) for _, f, _ in col) for col in cols] + [_line_h(F_CELL) + 8])
        rows.append((e, cols, note_lines, body_h + 24 + len(note_lines) * _line_h(F_SMALL)))

    header_h, info_h, table_head_h = 150, 64, 46
    footer_h = 90
    body_h = sum(r[3] for r in rows) if rows else 80
    card_h = header_h + info_h + table_head_h + body_h + footer_h
    img = Image.new("RGB", (WIDTH, card_h + 2 * MARGIN), PAGE_BG)
    draw = ImageDraw.Draw(img)
    x0, y0, x1 = MARGIN, MARGIN, WIDTH - MARGIN

    # --- card + header -----------------------------------------------------------
    draw.rounded_rectangle((x0, y0, x1, y0 + card_h), radius=28, fill=CARD_BG)
    draw.rounded_rectangle((x0, y0, x1, y0 + header_h), radius=28, fill=HEADER_BG)
    draw.rectangle((x0, y0 + header_h - 30, x1, y0 + header_h), fill=HEADER_BG)
    cx = x0 + PAD
    draw.text((cx, y0 + 34), "Vaccination Card", font=F_TITLE, fill=HEADER_FG)
    country = record.get("country")
    draw.text((cx, y0 + 96), "Digitaler Impfpass" + (f"  ·  {country}" if _present(country) else ""),
              font=F_SUB, fill=HEADER_SUB)
    n_done = sum(1 for e in entries if e.get("status") == "completed")
    summary = f"{len(entries)} entries  ·  {n_done} completed"
    draw.text((x1 - PAD - draw.textlength(summary, font=F_SUB), y0 + 96), summary,
              font=F_SUB, fill=HEADER_SUB)

    # --- source line ---------------------------------------------------------------
    y = y0 + header_h + 22
    source = Path(record.get("image") or "").name or "unknown"
    info = f"Source image: {source}    ·    Generated: {date.today():%d.%m.%Y}"
    draw.text((cx, y), _wrap(draw, info, F_SMALL, CONTENT_W)[0], font=F_SMALL, fill=MUTED)
    y = y0 + header_h + info_h

    # --- table head -----------------------------------------------------------------
    x = cx
    for title, w in COLUMNS:
        draw.text((x, y + 12), title.upper(), font=F_HEAD, fill=MUTED)
        x += w
    y += table_head_h
    draw.line((cx, y - 2, cx + CONTENT_W, y - 2), fill=RULE, width=2)

    if not rows:
        draw.text((cx, y + 26), "No vaccination records found on this image.", font=F_CELL, fill=MUTED)
        y += 80

    # --- rows -------------------------------------------------------------------------
    for i, (e, cols, note_lines, h) in enumerate(rows):
        if i % 2:
            draw.rectangle((cx - 12, y, cx + CONTENT_W + 12, y + h), fill=ZEBRA)
        x = cx
        for (_, w), col in zip(COLUMNS, cols):
            ly = y + 12
            for text, f, c in col:
                draw.text((x, ly), text, font=f, fill=c)
                ly += _line_h(f)
            x += w
        label, bg, fg = STATUS_STYLE.get(e.get("status"), (str(e.get("status") or "unknown"),
                                                            "#ECEFF1", "#55616A"))
        chip_w = draw.textlength(label, font=F_SMALL) + 28
        sx = cx + sum(w for _, w in COLUMNS[:-1])
        draw.rounded_rectangle((sx, y + 10, sx + chip_w, y + 10 + 32), radius=16, fill=bg)
        draw.text((sx + 14, y + 16), label, font=F_SMALL, fill=fg)
        ny = y + h - 12 - len(note_lines) * _line_h(F_SMALL)
        for line in note_lines:
            draw.text((cx, ny), line, font=F_SMALL, fill=MUTED)
            ny += _line_h(F_SMALL)
        y += h
        draw.line((cx, y, cx + CONTENT_W, y), fill=RULE, width=1)

    # --- footer ---------------------------------------------------------------------
    disclaimer = ("Automatically extracted from a photo of the vaccination booklet (DIGIVAX). "
                  "Not an official document — check against the original booklet.")
    fy = y0 + card_h - footer_h + 24
    for line in _wrap(draw, disclaimer, F_SMALL, CONTENT_W):
        draw.text((cx, fy), line, font=F_SMALL, fill=MUTED)
        fy += _line_h(F_SMALL)
    return img
