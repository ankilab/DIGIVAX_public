# DIGIVAX_runnable

Extracts vaccination records from photos/scans of vaccination certificates
(Impfpass) with the DIGIVAX pipeline. Runs on the NHR@FAU **Alex** cluster.

For every image:

1. **Extraction**: fine-tuned Qwen3.5-9B (LoRA adapter, "timopt" prompt) reads
   the page and returns one record per vaccination: date, status, vaccine names,
   stickers (trade name, lot number, manufacturer), note.
2. **Verification**: every sticker trade name is looked up in the PEI
   reference database (`digivax_runner/ref_vaccines.db`). Manufacturer names are
   corrected, trade-name typos fixed, and missing indications added when the
   match is unambiguous.
3. **Second pass** (only with `--pass2`): when a lookup was ambiguous (about 1 in
   3 images), the model reads the image again with the database entries in the prompt. Only
   trade names and vaccine names are taken from that answer.

No training happens here: the adapter is the fold-1 checkpoint of DIGIVAX run
`20260818T221350Z`.

**Not included in this repository:** the fine-tuned LoRA adapter and any
vaccination-certificate images (they contain personal health data). The base
model (`Qwen/Qwen3.5-9B`) is downloaded from Hugging Face by `install.sh`; set
`ADAPTER_SOURCE` in `config.env` to your own LoRA adapter directory, and put your
own images into `images/<name>/`.

## Install (once)

On the Alex login node:

```bash
cd DIGIVAX_runnable
ADAPTER_SOURCE=/path/to/lora_adapter ./install.sh
```

This creates a Python venv, makes sure the base model is present (downloads it
if not, ~19 GB) and copies the adapter. Everything large goes to
`$WORK/DIGIVAX_runnable/`, because the `/home/hpc` quota is too small for it.
All paths are in [`config.env`](config.env).

## Run

```bash
./run.sh                              # all images in images/test1 (the default)
./run.sh mybatch                      # all images in images/mybatch
./run.sh mybatch -n 20                # only the first 20 (sorted by name)
./run.sh /path/to/images              # any other folder
```

`run.sh` submits a GPU job and then shows what is happening, e.g.:

```
[16:23:25] Waiting for a GPU — other jobs are ahead in the queue; Slurm's estimated start: 16:40
[16:31:02] GPU assigned on a0322 — starting the job
[16:31:20] Loading the model on NVIDIA A40 …
[16:32:32] Model loaded in 72 s — processing 5 image(s)
[16:34:43] ✓ image 1/5  page_001  6 records · DB 5/5 found · 131 s
⠹ [████░░░░░░░░░░░░░░░░] 1/5 images  │ image 2/5 page_002: pass 1: reading the image 0:42  │ ~8:44 left
```

The last line is a live progress bar (loading bar while the model loads). When the
job finishes, `run.sh` prints a summary.

Ctrl-C cancels the job (`scancel`). Images that were finished are kept; run
again with `-o <output folder>` to process the rest. Closing the terminal only
stops the display — the job keeps running.

| Option | |
|---|---|
| `-n N` | only the first N images |
| `-o DIR` | output folder. Pass an existing one to **resume**: finished images are skipped and failed ones retried |
| `-p K` | split across K GPUs in parallel (default: one GPU per 450 images) |
| `-t HH:MM:SS` | walltime per job (default: estimated, 3 min per image + 15 min) |
| `--pass2` | second pass: re-read images whose DB lookup was ambiguous, with the DB entries in the prompt (slower) |
| `--no-verify` | extraction only, no database verification (cannot be combined with `--pass2`) |
| `--direct` | run in the current shell, when you are already on a GPU node (`salloc`) |

Input: `.jpg .jpeg .png .webp .tif .tiff .bmp` files directly in the folder
(subfolders are not searched; PDFs are not supported). EXIF rotation is applied.

Runtime on one A40: 20 s to 4.5 min per image (about 2 min on average with `--pass2`; dense pages
that need the second pass take longest), plus about 1 minute to load the model.

## Output

`output/<YYYYMMDD_HHMMSS>_<folder>/`, e.g. `output/20261002_153012_mybatch/` (or the folder given with `-o`):

```
fhir/<image>.json           FHIR R4 Bundle (type collection), one Immunization per record
standard/<image>.json       the label: country + entries (status, vaccine_name, stickers,
                            date, note), without bounding boxes
digital_card/<image>.png    the records rendered as a vaccination card
technical/
  raw/<image>.json          full record: final entries incl. db_match per sticker, model text
                            of both passes, pass-1 records, DB tool calls, timing
  results.csv               one row per sticker (records without a sticker get one row)
  summary.json              counts: images done/failed, records, second passes
  failed.txt                images that failed or were not processed (with the error)
  failed/<image>.json       error and traceback per failed image
  run_config.json           model, adapter and settings used
  images.tsv, todo.tsv      selected images / images still to do
  logs/                     Slurm job logs
```

FHIR mapping: `vaccineCode` gets a CVX code (`http://hl7.org/fhir/sid/cvx`, the
"unspecified formulation" codes) only when the vaccine names map to exactly one CVX
entry; combinations that fit several products (Td vs. DT, DTaP vs. Tdap, …) keep only
`text`. FHIR has no "unclear" status, so unclear records become `completed` with a
note. Missing or invalid dates use the `data-absent-reason` extension.

To rebuild `fhir/`, `standard/` and `digital_card/` from `technical/raw` (e.g. after
changing the converter), no GPU needed:

```bash
source config.env && py -m digivax_runner.export output/<run folder>
```

`results.csv` columns: `image, record, sticker, date, status, vaccine_name,
trade_name, lot_no, manufacturer, db_trade_name, db_trade_name_ok,
db_indications, vaccine_name_consistent, country, note`. Multiple vaccine names
are separated with `; `. The file is UTF-8 with BOM, so Excel shows umlauts correctly.

## Configuration

Every value in `config.env` can be overridden from the environment, e.g. to run
on an A100 instead of an A40:

```bash
SLURM_PARTITION=a100 SLURM_GRES=gpu:a100:1 ./run.sh images/
```

To use a different adapter, point `ADAPTER_SOURCE` and `ADAPTER_DIR` at it and run
`./install.sh` again. The adapter must have been trained with the timopt prompt
(`digivax_runner/prompts.py`).

## Layout

```
install.sh, run.sh, job.sbatch, config.env, requirements.txt
images/<name>/   input folders, one per batch; images/test1 is the default
                 (not tracked by git — images stay local)
digivax_runner/
  infer.py       GPU worker: model loading, per-image pipeline
  verify.py      database verification + second pass (from DIGIVAX stage 3)
  parse.py       JSON extraction from model output, decode-loop retry
  prompts.py     timopt prompt pair
  prepare.py     image selection (images.tsv / todo.tsv), stdlib only
  summarize.py   results.csv / summary.json / failed.txt, stdlib only
  watch.py       live progress display for run.sh (queue, model loading, images), stdlib only
  export.py      standard/ and fhir/ outputs (label → FHIR R4 Bundle with CVX codes)
  card.py        digital_card/ PNG rendering (Pillow)
  mcp_tools.py   reference-DB lookups (copied from DIGIVAX vaccination_db/)
  ref_vaccines.db
```
