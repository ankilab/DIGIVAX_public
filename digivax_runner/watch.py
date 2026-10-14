"""Show what a submitted run is doing, until its job has left the queue (stdlib only).

    python -m digivax_runner.watch <out_dir> <job_id> <num_tasks>

Phases: waiting for a GPU (squeue) → starting up → loading the model (progress
from the Slurm log) → inference per image (technical/status/task_<i>.json, written
by infer.py). Events are printed as lines; on a terminal the last line is a live
progress bar. Exit 0 when the job has left the queue, 130 on Ctrl-C (run.sh then
cancels the job), 2 when squeue keeps failing.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .prepare import done_path, read_images_tsv, tech_dir

POLL_S, SQUEUE_EVERY_S = 2, 15
DEFAULT_SECONDS_PER_IMAGE = 130            # measured A40 average, used until the first image is done
TTY = sys.stdout.isatty()
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
REASONS = {
    "Priority": "other jobs are ahead in the queue",
    "Resources": "all matching GPUs are busy",
    "QOSMaxJobsPerUserLimit": "you have reached the job limit",
    "QOSMaxGRESPerUser": "you have reached the GPU limit",
    "BeginTime": "scheduled start time not reached",
    "Dependency": "waiting for another job",
}


def clock() -> str:
    return time.strftime("%H:%M:%S")


def dur(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def bar(frac: float, width: int = 20) -> str:
    n = round(max(0.0, min(1.0, frac)) * width)
    return "█" * n + "░" * (width - n)


def short(key: str, n: int = 24) -> str:
    return key if len(key) <= n else key[:n - 1] + "…"


class Display:
    """Event lines above, one live line at the bottom (terminal only)."""

    def __init__(self):
        self.live = ""

    def event(self, msg: str):
        if TTY and self.live:
            sys.stdout.write("\r\033[K")
        print(f"[{clock()}] {msg}", flush=True)
        if TTY and self.live:
            sys.stdout.write(self.live)
            sys.stdout.flush()

    def set_live(self, text: str):
        if not TTY:
            return
        self.live = text[:shutil.get_terminal_size((120, 20)).columns - 1]
        sys.stdout.write("\r\033[K" + self.live)
        sys.stdout.flush()

    def clear(self):
        if TTY and self.live:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
        self.live = ""


def squeue(jid: str):
    """Rows (task_id_field, state, reason, est_start, node) of this job, or None on error.
    Queried by user, not `squeue -j`, which errors once a finished job is purged."""
    try:
        r = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%F|%i|%T|%r|%S|%N"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    rows = [line.split("|") for line in r.stdout.splitlines()]
    return [row[1:] for row in rows if len(row) == 6 and row[0] == jid]


def task_of(array_id: str):
    m = re.fullmatch(r"\d+_(\d+)", array_id)
    return int(m.group(1)) if m else None      # pending groups look like 123_[0-3]


def loading_percent(log: Path):
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 20000))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    hits = re.findall(r"Loading weights:\s*(\d+)%", tail)
    return int(hits[-1]) if hits else None


def read_status(path: Path, jid: str):
    try:
        st = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return st if str(st.get("job")) == jid else None   # ignore files from an earlier job


def main():
    if len(sys.argv) != 4:
        sys.exit("usage: python -m digivax_runner.watch <out_dir> <job_id> <num_tasks>")
    out, jid, n_tasks = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    tech = tech_dir(out)
    keys = [k for k, _ in read_images_tsv(out)]
    single = n_tasks == 1
    who = (lambda t: "") if single else (lambda t: f"task {t}: ")

    disp = Display()
    t_submit = time.time()
    rows, last_q, q_errors = [], 0.0, 0
    reason_shown = None
    seen = {t: {"running": False, "since": None, "phase": None, "results": 0} for t in range(n_tasks)}
    durations: list[float] = []
    tick = 0

    while True:
        now = time.time()
        if now - last_q >= SQUEUE_EVERY_S or last_q == 0:
            q = squeue(jid)
            last_q = now
            if q is None:
                q_errors += 1
                if q_errors >= 20:
                    disp.clear()
                    print(f"squeue keeps failing — check the job yourself: squeue -j {jid}", file=sys.stderr)
                    return 2
            else:
                q_errors, rows = 0, q
                if not rows:
                    break

        running = {task_of(r[0]): r[4] for r in rows if r[1] == "RUNNING"}
        pending = [r for r in rows if r[1] == "PENDING"]

        # --- queue events ----------------------------------------------------------
        if pending and not running:
            reason, start = pending[0][2], pending[0][3]
            if reason != reason_shown:
                why = REASONS.get(reason, reason if reason not in ("None", "") else "")
                eta = ""
                if re.match(r"\d{4}-\d\d-\d\dT", start):
                    eta = f"; Slurm's estimated start: {start[11:16]}"
                disp.event(f"Waiting for a GPU{f' — {why}' if why else ''}{eta}")
                reason_shown = reason
        for t, node in running.items():
            if t is not None and t in seen and not seen[t]["running"]:
                seen[t].update(running=True, since=now)
                disp.event(f"{who(t)}GPU assigned on {node} — starting the job")

        # --- task status events ---------------------------------------------------------
        statuses = {}
        for t in range(n_tasks):
            st = read_status(tech / "status" / f"task_{t}.json", jid)
            if st is None:
                continue
            statuses[t] = st
            s = seen[t]
            if not s["running"]:                       # started between two squeue polls
                s.update(running=True, since=now)
                disp.event(f"{who(t)}GPU assigned on {st.get('host', '?')} — starting the job")
            phase = st.get("phase")
            if phase != s["phase"]:
                if phase == "loading":
                    disp.event(f"{who(t)}Loading the model on {st.get('gpu', 'GPU')} …")
                elif phase == "inference":
                    disp.event(f"{who(t)}Model loaded in {st.get('model_seconds', '?')} s — "
                               f"processing {st.get('total', '?')} image(s)")
                s["phase"] = phase
            results = st.get("results") or []
            for r in results[s["results"]:]:
                total = st.get("total", "?")
                if r.get("ok"):
                    durations.append(r.get("seconds", 0))
                    extra = ""
                    if r.get("db") is not None:
                        extra = f" · DB {r['db']} found"
                        if r.get("pass2") is not None:
                            extra += f" · 2nd pass {'applied' if r['pass2'] else 'no'}"
                    disp.event(f"{who(t)}✓ image {r['index']}/{total}  {r['key']}  "
                               f"{r.get('records', 0)} records{extra} · {r.get('seconds', 0)} s")
                else:
                    disp.event(f"{who(t)}✗ image {r['index']}/{total}  {r['key']}  FAILED: {r.get('error', '?')}")
            s["results"] = len(results)
            if phase == "finished" and s.get("finished_shown") is None:
                s["finished_shown"] = True
                disp.event(f"{who(t)}Finished: {st.get('ok', 0)} ok, {st.get('failed', 0)} failed")

        # --- live line -------------------------------------------------------------------
        tick += 1
        spin = SPINNER[tick % len(SPINNER)]
        done = sum(1 for k in keys if done_path(out, k).exists())
        overall = f"[{bar(done / max(1, len(keys)))}] {done}/{len(keys)} images"
        active = [t for t in range(n_tasks) if seen[t]["running"] and seen[t]["phase"] != "finished"]
        n_wait = sum(1 for t in range(n_tasks) if not seen[t]["running"])

        if not active and n_wait:
            live = f"{spin} Waiting for a GPU  {dur(now - t_submit)}"
        elif single or len(active) == 1:
            t = active[0] if active else 0
            st = statuses.get(t, {})
            phase = st.get("phase")
            if phase == "loading":
                pct = loading_percent(tech / "logs" / f"slurm_{jid}_{t}.log")
                prog = f"[{bar(pct / 100)}] {pct:3d}%" if pct is not None else "preparing"
                live = f"{spin} Loading model  {prog}  {dur(now - st.get('since', now))}"
            elif phase == "inference" and st.get("key"):
                avg = sum(durations) / len(durations) if durations else DEFAULT_SECONDS_PER_IMAGE
                remaining = len(keys) - done
                eta = (f"  │ ~{dur(remaining * avg)} left" if durations and remaining else "")
                live = (f"{spin} {overall}  │ image {st.get('index')}/{st.get('total')} "
                        f"{short(st['key'])}: {st.get('step', '')} {dur(now - st.get('step_since', now))}"
                        f"{eta}")
            elif phase == "finished":
                live = f"{spin} {overall}  │ writing summary"
            else:
                live = f"{spin} Starting up (Python, CUDA)  {dur(now - (seen[t]['since'] or now))}"
        else:
            phases = [statuses.get(t, {}).get("phase") or "starting" for t in active]
            counts = ", ".join(f"{phases.count(p)} {p}" for p in dict.fromkeys(phases))
            live = f"{spin} {overall}  │ tasks: {counts}" + (f", {n_wait} waiting for GPU" if n_wait else "")
        disp.set_live(live)

        time.sleep(POLL_S)

    disp.clear()
    disp.event(f"Job {jid} has left the queue")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        if TTY:
            sys.stdout.write("\r\033[K")
        sys.exit(130)
