#!/usr/bin/env python3
"""Terminal UI launcher for model refusal ablation."""

from __future__ import annotations

import curses
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class LauncherState:
    model_path: str = "models/input_model.gguf"
    output_path: str = "models/ablated_model.gguf"
    heretic_config: str = "config/heretic_refusal.yaml"
    method: str = "zero"
    strategy: str = "ablation"
    auto_detect: bool = False
    apply_heretic_edits: bool = False


METHOD_OPTIONS = ("zero", "prune", "clamp")
STRATEGY_OPTIONS = ("ablation", "heretic")


def _status_to_progress(log_line: str, previous: int) -> int:
    """Map script output to progress percentage."""
    mapping = {
        "Auto-detecting refusal neurons": 25,
        "Identified": 35,
        "Aborting to": 0,
        "Ablating": 55,
        "Post-ablation compliance": 75,
        "Saved": 90,
        "Ablation report saved": 95,
        "Heretic-style refusal ablation completed": 95,
        "completed with report": 95,
    }
    for token, value in mapping.items():
        if token in log_line:
            return max(previous, value)
    return previous


def _run_pipeline(state: LauncherState, output_queue: queue.Queue, state_queue: queue.Queue) -> None:
    """Run the underlying ablation pipeline in a background thread."""
    ts = int(time.time())
    report_path = f"exports/ablation_reports/refusal_ablation_{ts}.json"
    output_queue.put(("meta", f"Report: {report_path}"))

    if not os.path.exists("exports/ablation_reports"):
        Path("exports/ablation_reports").mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        sys.executable,
        "src/aegis_lab/editing/model_refusal_ablation.py",
        "--model",
        state.model_path,
        "--output",
        state.output_path,
        "--method",
        state.method,
        "--strategy",
        state.strategy,
        "--report",
        report_path,
    ]

    if state.auto_detect:
        cmd.append("--auto-detect")

    if state.strategy == "heretic":
        cmd.extend(["--heretic-config", state.heretic_config])
        if state.apply_heretic_edits:
            cmd.append("--apply-heretic-edits")

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
    except Exception as exc:  # pragma: no cover - start-up failure path
        output_queue.put(("log", f"Failed to start process: {exc}"))
        state_queue.put(("done", 1, "start-failed"))
        return

    state_queue.put(("pid", proc.pid))

    if proc.stdout is not None:
        for line in proc.stdout:
            output_queue.put(("log", line.rstrip("\n")))
    rc = proc.wait()
    state_queue.put(("done", rc, "completed" if rc == 0 else "failed"))


def _draw_checkbox(window: Any, row: int, col: int, label: str, checked: bool, focused: bool) -> None:
    marker = "[x]" if checked else "[ ]"
    prefix = "> " if focused else "  "
    window.addstr(row, col, f"{prefix}{marker} {label}")


def _draw_radio(window: Any, row: int, col: int, label: str, selected: bool, focused: bool) -> None:
    marker = "(*)" if selected else "( )"
    prefix = "> " if focused else "  "
    window.addstr(row, col, f"{prefix}{marker} {label}")


def _draw_input(window: Any, row: int, col: int, label: str, value: str, width: int,
                focused: bool, placeholder: str = "") -> None:
    prefix = "> " if focused else "  "
    value_display = value or placeholder
    width = max(10, width)
    window.addstr(row, col, f"{prefix}{label}: ")
    window.addstr(row, col + len(label) + 4, value_display[: width - 1])


def _draw_progress_bar(window: Any, row: int, col: int, width: int, progress: int, label: str) -> None:
    width = max(10, width)
    fill = max(1, (width - 2) * progress // 100)
    bar = "[" + ("="#fill) + (" " * max(0, width - 2 - fill)) + "]"
    window.addstr(row, col, f"{label}: {progress:>3}% {bar}")


def _validate_input(state: LauncherState) -> Tuple[bool, str]:
    if not state.model_path:
        return False, "Model path is required"
    if not os.path.isfile(state.model_path):
        return False, f"Model not found: {state.model_path}"
    if not state.output_path:
        return False, "Output path is required"
    if state.strategy == "heretic" and not state.heretic_config:
        return False, "Heretic config is required for Heretic strategy"
    return True, ""


def _build_ui_lines() -> List[Dict[str, str]]:
    return [
        {"type": "text", "key": "model", "label": "Model path"},
        {"type": "text", "key": "output", "label": "Output path"},
        {"type": "radio", "key": "method_zero", "label": "Method: zero", "group": "method"},
        {"type": "radio", "key": "method_prune", "label": "Method: prune", "group": "method"},
        {"type": "radio", "key": "method_clamp", "label": "Method: clamp", "group": "method"},
        {"type": "radio", "key": "strategy_ablation", "label": "Strategy: ablation", "group": "strategy"},
        {"type": "radio", "key": "strategy_heretic", "label": "Strategy: heretic", "group": "strategy"},
        {"type": "check", "key": "auto_detect", "label": "Auto-detect layers"},
        {"type": "check", "key": "apply_heretic_edits", "label": "Apply Heretic edits"},
        {"type": "text", "key": "heretic_config", "label": "Heretic config"},
        {"type": "action", "key": "run", "label": "Run ablation"},
        {"type": "action", "key": "quit", "label": "Quit"},
    ]


def _selected_radio_group(state: LauncherState, group: str, key: str) -> bool:
    if group == "method":
        return state.method in key
    if group == "strategy":
        return state.strategy in key
    return False


def _set_radio_group(state: LauncherState, group: str, key: str) -> None:
    if group == "method":
        if key.endswith("zero"):
            state.method = "zero"
        elif key.endswith("prune"):
            state.method = "prune"
        elif key.endswith("clamp"):
            state.method = "clamp"
    elif group == "strategy":
        if key.endswith("ablation"):
            state.strategy = "ablation"
        elif key.endswith("heretic"):
            state.strategy = "heretic"


def _get_state_value(state: LauncherState, key: str) -> str:
    return {
        "model": state.model_path,
        "output": state.output_path,
        "heretic_config": state.heretic_config,
    }[key]


def _set_state_value(state: LauncherState, key: str, value: str) -> None:
    if key == "model":
        state.model_path = value
    elif key == "output":
        state.output_path = value
    elif key == "heretic_config":
        state.heretic_config = value


def _get_checkbox_value(state: LauncherState, key: str) -> bool:
    if key == "auto_detect":
        return state.auto_detect
    if key == "apply_heretic_edits":
        return state.apply_heretic_edits
    return False


def _set_checkbox_value(state: LauncherState, key: str, value: bool) -> None:
    if key == "auto_detect":
        state.auto_detect = value
    elif key == "apply_heretic_edits":
        state.apply_heretic_edits = value


def run_tui(stdscr: Any) -> int:
    curses.curs_set(0)
    stdscr.timeout(100)

    state = LauncherState()
    lines = _build_ui_lines()
    focus = 0
    logs = deque(maxlen=8)
    status = "Ready"
    progress = 0
    spinner = ["-", "\\", "|", "/"]
    spinner_idx = 0
    running = False
    done = False
    process_pid = None
    queue_lines: queue.Queue = queue.Queue()
    queue_state: queue.Queue = queue.Queue()
    process_thread: threading.Thread | None = None

    while True:
        height, width = stdscr.getmaxyx()
        stdscr.erase()
        if width < 60 or height < 24:
            stdscr.addstr(0, 0, "Terminal too small. Resize to at least 60x24")
            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord("q"), ord("Q"), 27):
                break
            continue

        stdscr.addstr(0, 2, "Model Refusal Ablation TUI Launcher")
        stdscr.addstr(1, 2, "Use arrow keys to navigate, space/enter to toggle/execute.")
        stdscr.addstr(2, 2, "q to quit while idle; while running, q will cancel.")

        # Model input section
        y = 4
        _draw_input(stdscr, y, 2, "Model path", state.model_path, width - 30, focus == 0)
        y += 2
        _draw_input(stdscr, y, 2, "Output path", state.output_path, width - 30, focus == 1)

        # Method (checkbox-style single-select)
        y += 3
        stdscr.addstr(y, 2, "Ablation method")
        y += 1
        _draw_radio(stdscr, y, 4, "zero", state.method == "zero", focus == 2)
        y += 1
        _draw_radio(stdscr, y, 4, "prune", state.method == "prune", focus == 3)
        y += 1
        _draw_radio(stdscr, y, 4, "clamp", state.method == "clamp", focus == 4)

        # Strategy
        y += 2
        stdscr.addstr(y, 2, "Refusal strategy")
        y += 1
        _draw_radio(stdscr, y, 4, "ablation", state.strategy == "ablation", focus == 5)
        y += 1
        _draw_radio(stdscr, y, 4, "heretic", state.strategy == "heretic", focus == 6)

        # Checkboxes
        y += 2
        _draw_checkbox(stdscr, y, 2, "Auto-detect layers", state.auto_detect, focus == 7)
        y += 1
        _draw_checkbox(stdscr, y, 2, "Apply Heretic edits", state.apply_heretic_edits, focus == 8)

        # Heretic config section
        y += 2
        heretic_label = (
            "Heretic config" if state.strategy == "heretic" else "Heretic config (only used in heretic strategy)"
        )
        _draw_input(stdscr, y, 2, heretic_label, state.heretic_config, width - 30, focus == 9)

        # Action buttons
        y += 2
        stdscr.addstr(y, 2, "> Run ablation" if focus == 10 else "  Run ablation")
        y += 1
        stdscr.addstr(y, 2, "> Quit" if focus == 11 else "  Quit")

        # Progress and logs
        log_start = 18
        y = max(log_start, y)
        status_label = status
        if running:
            status_label = f"Running ({spinner[spinner_idx]}) {status_label}"
            spinner_idx = (spinner_idx + 1) % len(spinner)
            progress = min(progress + 1, 98)

        _draw_progress_bar(stdscr, log_start, 2, min(48, width - 12), progress, "Progress")
        stdscr.addstr(log_start + 1, 2, f"Status: {status_label}"[: width - 4])
        stdscr.addstr(log_start + 2, 2, "Logs:")
        line_top = log_start + 3
        for idx in range(8):
            if idx < len(logs):
                stdscr.addstr(line_top + idx, 2, logs[idx][: width - 5])

        stdscr.refresh()

        # Process worker queues
        while True:
            try:
                event_type, payload = queue_lines.get_nowait()
            except queue.Empty:
                break
            if event_type == "log":
                logs.append(payload)
                progress = _status_to_progress(payload, progress)
                status = payload[-min(len(payload), 64):]
            elif event_type == "meta":
                logs.append(payload)

        while True:
            try:
                event = queue_state.get_nowait()
            except queue.Empty:
                break
            if event[0] == "pid":
                process_pid = event[1]
            elif event[0] == "done":
                running = False
                status = "Done" if event[1] == 0 else "Failed"
                progress = 100 if event[1] == 0 else min(progress, 99)
                done = True if event[1] == 0 else False
                if event[2] == "failed":
                    logs.append("Process exited with non-zero status.")

        if not running:
            key = stdscr.getch()
            if key in (-1, curses.ERR):
                continue

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (curses.KEY_UP, ord("k")):
                focus = (focus - 1) % len(lines)
                continue
            if key in (curses.KEY_DOWN, ord("j")):
                focus = (focus + 1) % len(lines)
                continue
            if key in (ord("\t"),):
                focus = (focus + 1) % len(lines)
                continue

            if focus <= 1 and key == curses.KEY_LEFT:
                focus = (focus - 1) % len(lines)

            item = lines[focus]
            item_key = item["key"]

            if key == ord(" "):
                if item["type"] == "radio":
                    _set_radio_group(state, item["group"], item_key)
                    if item_key == "strategy_ablation":
                        state.apply_heretic_edits = False
                    continue
                if item["type"] == "check":
                    value = _get_checkbox_value(state, item_key)
                    _set_checkbox_value(state, item_key, not value)
                    continue

            if item["type"] in ("text", "action") and key in (curses.KEY_ENTER, 10, 13):
                if item_key == "run":
                    if not _validate_input(state)[0]:
                        valid, msg = _validate_input(state)
                        status = msg
                        continue
                    if running:
                        continue
                    running = True
                    done = False
                    status = "Starting"
                    progress = 0
                    logs.clear()
                    queue_lines = queue.Queue()
                    queue_state = queue.Queue()
                    process_thread = threading.Thread(
                        target=_run_pipeline,
                        args=(state, queue_lines, queue_state),
                        daemon=True,
                    )
                    process_thread.start()
                    continue
                if item_key == "quit":
                    break

            if item["type"] == "text":
                if key in (curses.KEY_BACKSPACE, 127, 8):
                    value = _get_state_value(state, item_key)
                    _set_state_value(state, item_key, value[:-1])
                elif 32 <= key <= 126:
                    _set_state_value(state, item_key, _get_state_value(state, item_key) + chr(key))

            continue

        else:
            # running case
            key = stdscr.getch()
            if key in (ord("q"), ord("Q"), 27):
                if process_pid:
                    try:
                        os.kill(process_pid, 15)
                    except Exception:
                        pass
                break

    return 0


def main() -> int:
    return curses.wrapper(run_tui)


if __name__ == "__main__":
    raise SystemExit(main())
