#!/usr/bin/env python3
"""
Dashboard Server — 轻看板服务 (FastAPI + SSE)
实时监控 RL-MEC Benchmark 训练状态

用法:
    python scripts/serve_dashboard.py --logs-dir logs --host 127.0.0.1 --port 8088
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional
import threading
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RL-MEC Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_DIR = Path("logs")
HOST = "127.0.0.1"
PORT = 8088

STALL_THRESHOLD_SEC = 120

_run_states: dict[str, "RunState"] = {}
_state_lock = threading.Lock()


class RunState:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.status = "idle"
        self.current_algorithm = ""
        self.current_step = 0
        self.total_step = 100000
        self.progress_pct = 0.0
        self.it_per_sec = 0.0
        self.eta_seconds = 0
        self.elapsed_seconds = 0
        self.update_count = 0
        self.completed_algorithms: list[str] = []
        self.results: list[dict] = []
        self.last_error = ""
        self.updated_at = time.time()
        self.log_offsets: dict[str, int] = {}
        self.last_log_time = time.time()

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "current_algorithm": self.current_algorithm,
            "current_step": self.current_step,
            "total_step": self.total_step,
            "progress_pct": self.progress_pct,
            "it_per_sec": self.it_per_sec,
            "eta_seconds": self.eta_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "update_count": self.update_count,
            "completed_algorithms": self.completed_algorithms,
            "results": self.results,
            "last_error": self.last_error,
            "updated_at": self.updated_at,
        }


def parse_elapsed_from_tqdm(tqdm_bar: str) -> float:
    m = re.search(r"(\d+)s", tqdm_bar)
    return float(m.group(1)) if m else 0.0


def parse_eta_from_tqdm(tqdm_bar: str) -> int:
    m = re.search(r"<(\d+):(\d+):(\d+)", tqdm_bar)
    if m:
        h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return h * 3600 + mn * 60 + s
    m2 = re.search(r"<(\d+):(\d+)", tqdm_bar)
    if m2:
        mn, s = int(m2.group(1)), int(m2.group(2))
        return mn * 60 + s
    return 0


def parse_step_from_tqdm(line: str) -> Optional[tuple]:
    m = re.search(r"Training \w+Agent: .*?(\d+)/(\d+).*?([\d.]+)\sit/s", line)
    if m:
        cur = int(m.group(1))
        tot = int(m.group(2))
        ips = float(m.group(3))
        return cur, tot, ips
    return None


def parse_algo_switch(line: str) -> Optional[str]:
    m = re.search(r"Algorithm:\s*(\w+)", line, re.IGNORECASE)
    return m.group(1) if m else None


def parse_result(line: str) -> Optional[dict]:
    m = re.search(r"\[(\w+)\].*?reward=([-\d.]+).*?time=([\d.]+)s", line, re.IGNORECASE)
    if m:
        return {
            "algorithm": m.group(1),
            "reward": float(m.group(2)),
            "train_time": float(m.group(3)),
        }
    return None


def parse_update_count(line: str) -> Optional[int]:
    m = re.search(r"update_count[=:]?\s*(\d+)", line, re.IGNORECASE)
    return int(m.group(1)) if m else None


def scan_logs(log_dir: Path, state: RunState):
    if not log_dir.exists():
        return

    stdout_files = sorted(log_dir.glob("benchmark*.log"))
    stderr_files = sorted(log_dir.glob("benchmark*.err.log"))

    for lf in stderr_files:
        try:
            with open(lf, "r", encoding="utf-8", errors="replace") as f:
                f.seek(state.log_offsets.get(str(lf), 0))
                for line in f:
                    step_info = parse_step_from_tqdm(line)
                    if step_info:
                        state.current_step, state.total_step, state.it_per_sec = step_info
                        if state.it_per_sec > 0:
                            remaining = state.total_step - state.current_step
                            state.eta_seconds = int(remaining / state.it_per_sec)
                        state.last_log_time = time.time()

                    eta = parse_eta_from_tqdm(line)
                    if eta > 0:
                        state.eta_seconds = eta

                    elapsed = parse_elapsed_from_tqdm(line)
                    if elapsed > 0:
                        state.elapsed_seconds = elapsed

                    uc = parse_update_count(line)
                    if uc is not None:
                        state.update_count = uc

                    algo = parse_algo_switch(line)
                    if algo:
                        if state.current_algorithm and state.current_algorithm != algo:
                            if algo not in state.completed_algorithms:
                                state.completed_algorithms.append(state.current_algorithm)
                        state.current_algorithm = algo
                state.log_offsets[str(lf)] = f.tell()
        except Exception:
            pass

    for lf in stdout_files:
        try:
            with open(lf, "r", encoding="utf-8", errors="replace") as f:
                f.seek(state.log_offsets.get(str(lf), 0))
                for line in f:
                    result = parse_result(line)
                    if result:
                        state.results.append(result)
                        if result["algorithm"] not in state.completed_algorithms:
                            state.completed_algorithms.append(result["algorithm"])

                    algo = parse_algo_switch(line)
                    if algo:
                        state.current_algorithm = algo
                        state.last_log_time = time.time()
                state.log_offsets[str(lf)] = f.tell()
        except Exception:
            pass

    now = time.time()
    if state.current_step > 0 and state.total_step > 0:
        state.progress_pct = round(state.current_step / state.total_step * 100, 2)

    if state.current_step >= state.total_step and state.total_step > 0:
        state.status = "finished"
    elif now - state.last_log_time > STALL_THRESHOLD_SEC:
        state.status = "stalled"
    else:
        state.status = "running" if state.current_step > 0 else "idle"

    state.updated_at = time.time()


def background_scan(log_dir: Path, interval: float = 1.0):
    def loop():
        while True:
            with _state_lock:
                for run_id, state in _run_states.items():
                    scan_logs(log_dir, state)
            time.sleep(interval)

    t = threading.Thread(target=loop, daemon=True)
    t.start()


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "monitor_dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Dashboard not found")


@app.get("/api/runs")
async def list_runs():
    with _state_lock:
        runs = [{"run_id": k, "updated_at": v.updated_at} for k, v in _run_states.items()]
    runs.sort(key=lambda x: x["updated_at"], reverse=True)
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    with _state_lock:
        if run_id not in _run_states:
            raise HTTPException(status_code=404, detail="Run not found")
        return _run_states[run_id].to_dict()


@app.get("/api/runs/{run_id}/events")
async def stream_events(run_id: str, request: Request):
    async def event_generator():
        last_state = ""
        while True:
            if await request.is_disconnected():
                break
            with _state_lock:
                if run_id not in _run_states:
                    break
                state = _run_states[run_id]
                current_state = json.dumps(state.to_dict(), sort_keys=True)
            if current_state != last_state:
                last_state = current_state
                yield f"event: snapshot\ndata: {current_state}\n\n"
            time.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.on_event("startup")
async def startup():
    run_id = "latest"
    with _state_lock:
        _run_states[run_id] = RunState(run_id)
    background_scan(LOG_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="RL-MEC Dashboard Server")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Log directory path")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8088, help="Port to bind")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    LOG_DIR = Path(args.logs_dir)
    HOST = args.host
    PORT = args.port
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
