"""GPU worker: run the extraction pipeline on the images listed in <out>/technical/todo.tsv.

    python -m digivax_runner.infer --out-dir <out> --base-model <dir> --adapter <dir>
                                   [--task-id i --num-tasks k] [--no-verify | --pass2]

Per image:
  pass 1   fine-tuned Qwen3.5-9B + LoRA adapter, timopt prompt, greedy decoding
  verify   DB lookup of every sticker trade name (verify.mcp_enrich)
  pass 2   only with --pass2, and only when a lookup was ambiguous:
           VLM again with DB context, surgical merge

Writes standard/, fhir/ and digital_card/ for every image (export.py), then
technical/raw/<key>.json: final records incl. DB matches, model text of both
passes, pass-1 records and DB tool calls. That file is written last, so images
that have it are skipped. Errors go to technical/failed/<key>.json and the run
continues. With --num-tasks k each array task takes every k-th image of todo.tsv.

Progress (phase, current image and step, finished images) goes to
technical/status/task_<i>.json, which run.sh's watcher (watch.py) displays.
"""
import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

from .export import write_exports
from .parse import extract_json, extract_country, generate_extraction
from .prepare import done_path, read_images_tsv, tech_dir
from .prompts import SYSTEM_PROMPT, USER_PROMPT, refine_prompt
from .summarize import summarize
from .verify import mcp_enrich, refine_and_enrich

MAX_NEW_TOKENS = 4096


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num-tasks", type=int, default=1)
    ap.add_argument("--no-verify", action="store_true", help="Skip DB verification and pass 2.")
    ap.add_argument("--pass2", action="store_true",
                    help="Re-read images whose DB lookup was ambiguous with the DB context (pass 2).")
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    args = ap.parse_args()
    if args.no_verify and args.pass2:
        ap.error("--pass2 needs DB verification; drop --no-verify")
    for name in ("base_model", "adapter"):
        if not getattr(args, name).is_dir():
            sys.exit(f"ERROR: --{name.replace('_', '-')} not found: {getattr(args, name)}")
    return args


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def status_writer(out: Path, task_id: int):
    """Returns update(**fields): merges fields into the task's status file."""
    path = tech_dir(out) / "status" / f"task_{task_id}.json"
    state = {"job": os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "direct",
             "task": task_id, "host": socket.gethostname(), "results": []}

    def update(**fields):
        state.update(fields)
        state["updated"] = time.time()
        try:
            _write_json(path, state)
        except OSError:
            pass                                   # progress display only — never fail the run
    return update


def load_model(base_model: Path, adapter: Path):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    processor  = AutoProcessor.from_pretrained(str(base_model), trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(
        str(base_model), dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    if isinstance(getattr(base, "_no_split_modules", None), set):
        base._no_split_modules = list(base._no_split_modules)
    model = PeftModel.from_pretrained(base, str(adapter))
    model.eval()
    return model, processor


def main():
    args = parse_args()
    out = args.out_dir
    status = status_writer(out, args.task_id)
    status(phase="starting", since=time.time())
    mine = read_images_tsv(out, "todo.tsv")[args.task_id::args.num_tasks]
    todo = [(k, p) for k, p in mine if not done_path(out, k).exists()]
    print(f"task {args.task_id}/{args.num_tasks}: {len(mine)} image(s) assigned, "
          f"{len(mine) - len(todo)} already done, {len(todo)} to run  |  "
          f"verify={'off' if args.no_verify else 'on'}{'  pass2=on' if args.pass2 else ''}", flush=True)
    if not todo:
        summarize(out)
        status(phase="finished", ok=0, failed=0)
        return 0

    import torch
    from PIL import Image, ImageOps

    t0 = time.time()
    status(phase="loading", since=t0, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "?")
    print("loading model …", flush=True)
    model, processor = load_model(args.base_model, args.adapter)
    device = next(model.parameters()).device
    print(f"model loaded in {time.time() - t0:.0f}s  |  base={args.base_model}  adapter={args.adapter}",
          flush=True)
    status(phase="inference", model_seconds=round(time.time() - t0), total=len(todo))

    n_failed, results = 0, []
    for i, (key, path) in enumerate(todo, 1):
        tag = f"[{i}/{len(todo)}] {key}"
        t_img = time.time()
        status(index=i, key=key, image_since=t_img, step="pass 1: reading the image", step_since=t_img)
        print(f"{tag}  started", flush=True)
        try:
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")

            def run_vlm(user_prompt: str) -> str:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_prompt},
                    ]},
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
                with torch.inference_mode():
                    return generate_extraction(model, processor, inputs,
                                               max_new_tokens=MAX_NEW_TOKENS, tag=f" {key}")

            raw1 = run_vlm(USER_PROMPT)
            entries1 = extract_json(raw1)
            t_pass1 = time.time() - t_img

            if args.no_verify:
                result, raw2, refine_info = {"validated_entries": entries1, "tool_calls": []}, None, None
            else:
                status(step="checking stickers against the DB", step_since=time.time())
                def run_pass2(db_context: str) -> str:
                    status(step="pass 2: re-reading with DB context", step_since=time.time())
                    return run_vlm(refine_prompt(db_context))
                result, raw2, refine_info = refine_and_enrich(entries1, run_pass2 if args.pass2 else None)
            entries = result["validated_entries"]
            lookups = [tc for tc in result["tool_calls"] if tc["tool"] == "lookup_vaccine"]
            n_found = sum(1 for tc in lookups if tc["result"].get("found"))
            t_total = time.time() - t_img

            record = {
                "image": path,
                "country": extract_country(raw1),
                "entries": entries,
                "verification": None if args.no_verify else {
                    "db_lookups": len(lookups), "db_found": n_found,
                    "pass2_applied": bool(refine_info and refine_info.get("applied")),
                    "pass2_info": refine_info,
                },
                "timing_s": {"pass1": round(t_pass1, 1), "total": round(t_total, 1)},
                "raw_pass1": raw1, "raw_pass2": raw2,
                "pass1_entries": entries1, "db_tool_calls": result["tool_calls"],
            }
            status(step="writing outputs", step_since=time.time())
            write_exports(out, key, record)
            _write_json(done_path(out, key), record)
            (tech_dir(out) / "failed" / f"{key}.json").unlink(missing_ok=True)
            p2 = "" if args.no_verify else f"  db={n_found}/{len(lookups)}"
            if args.pass2:
                p2 += "  pass2=" + ("applied" if refine_info.get("applied") else f"no ({refine_info.get('reason')})")
            print(f"{tag}  records={len(entries)}{p2}  {t_total:.0f}s", flush=True)
            results.append({
                "index": i, "key": key, "ok": True, "records": len(entries), "seconds": round(t_total),
                "db": None if args.no_verify else f"{n_found}/{len(lookups)}",
                "pass2": bool(refine_info.get("applied")) if args.pass2 else None})
            status(results=results)

        except Exception as exc:
            n_failed += 1
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            _write_json(tech_dir(out) / "failed" / f"{key}.json", {
                "image": path, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            print(f"{tag}  FAILED: {type(exc).__name__}: {exc}", flush=True)
            results.append({"index": i, "key": key, "ok": False, "seconds": round(time.time() - t_img),
                            "error": f"{type(exc).__name__}: {exc}"[:300]})
            status(results=results)

    s = summarize(out)
    status(phase="finished", ok=len(todo) - n_failed, failed=n_failed)
    print(f"task {args.task_id} finished: {len(todo) - n_failed} ok, {n_failed} failed  |  "
          f"run total so far: {s['images_done']}/{s['images_selected']} done", flush=True)
    return 0 if n_failed == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
