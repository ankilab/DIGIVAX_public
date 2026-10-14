"""Prompts for the fine-tuned adapter.

The adapter was trained with the "timopt" prompt pair, so inference must use it
verbatim. Copied from DIGIVAX prompt_rag_engineering/prompts.py
(REFINED_SYSTEM_PROMPT / REFINED_USER_PROMPT).
"""

SYSTEM_PROMPT = (
    "You are a precise medical document parser specialising in vaccination booklets (Impfpass). "
    "Extract vaccination records exactly as they appear. "
    "Output only a valid JSON object — no markdown, no explanations, no surrounding text."
)

USER_PROMPT = """\
Extract all vaccination records from this vaccination booklet image.

DOCUMENT STRUCTURE
The page shows a vaccination table split into two sections:
Section columns (left → right): Date | Sticker/Vignette column | indication x | indication y | indication z | ... (each column corresponds to a vaccine type, e.g. Tetanus, Diphtherie, Pertussis etc. or there is a free field where the vaccine name is handwritten)
Each horizontal row represents one vaccination event on one date.

HOW TO READ EACH ROW
1. DATE — Read the handwritten date in the leftmost column. Convert to YYYY-MM-DD.
   If only the year is legible, use YYYY-01-01.
2. STICKERS (Vignetten) — Adhesive labels pasted in the sticker column show the trade name and lot number.
   The lot number is printed after "Ch.-B.", "Ch-B", "Lot-Nr.", or "Lot/".
   One row may have multiple stickers; list every one of them.
   If no sticker is present: the vaccine name may be written by hand in the sticker column — treat it as the trade_name with lot_no "n/a".
3. VACCINE COLUMNS — An × (cross / X) mark in a column means that vaccine was administered.
   An ○ (open circle) mark means the entry is planned or unclear — do NOT treat it as administered.
   List every column that carries an × as a separate entry in vaccine_name.
4. MENINGOKOKKEN SUBTYPE — If a handwritten "B" or "C" appears next to the Meningokokken column, record the vaccine_name as "Meningokokken B" or "Meningokokken C" respectively.

STATUS RULES
- "completed" : sticker(s) or handwritten vaccine name present AND × marks in the vaccine columns AND a valid past or present date.
- "unclear"   : entry exists but is ambiguous — missing sticker, only circle marks (○) instead of ×, illegible date, or otherwise incomplete/contradictory data.
- "not-done"  : entry consists only of circle marks (no sticker, no ×), typically a future-dated planning entry.

OUTPUT FIELDS
- country      : Name of the country issuing the booklet.
- vaccine_name : List of vaccine types derived from the × marks in the column headers. Use the column header names exactly (e.g. "Tetanus", "Diphtherie", "Pertussis", "Poliomyelitis", "Hib (Haemophilus influenzae b)", "Hepatitis B", "Masern, Mumps, Röteln (MMR)", "Varizellen", "Meningokokken", "Pneumokokken", "Rotavirus", "Influenza", "COVID-19").
- stickers     : List of {trade_name, lot_no, manufacturer}. Set manufacturer to "n/a" if not shown on the sticker.
- date         : "YYYY-MM-DD" or "n/a" if completely illegible.
- note         : Describe any anomaly: handwritten additions, circle-only marks, overlapping or tilted stickers, partial entries. Use "n/a" if nothing unusual.
- b_box        : Approximate pixel bounding box of the row in the image as [[x, y, width, height]] where x,y is the top-left corner.

Return ONLY this JSON object — no markdown fences, no extra keys, no commentary:
{
  "country": "<country name>",
  "entries": [
    {
      "status": "completed" | "unclear" | "not-done",
      "vaccine_name": ["<vaccine type>", ...],
      "stickers": [
        {"trade_name": "<trade name>", "lot_no": "<lot number or n/a>", "manufacturer": "<manufacturer or n/a>"}
      ],
      "date": "YYYY-MM-DD" or "n/a",
      "note": "<anomaly description or n/a>",
      "b_box": [[x, y, width, height]]
    }
  ]
}

Keep all vaccine names, trade names, lot numbers, and notes in the original language from the image. Do not translate.\
"""


def refine_prompt(db_context: str) -> str:
    """User prompt for the verification pass (pass 2) with DB context injected."""
    return (
        f"{USER_PROMPT}\n\n"
        f"Use the reference data below to correct vaccine trade names and indications "
        f"if the image text was ambiguous:\n\n{db_context}"
    )
