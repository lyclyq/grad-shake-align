# src/loggingx.py
from __future__ import annotations

import csv
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class CSVLogger:
    """
    Writes a flat dict row to CSV (fail-open, flush each write),
    but merges multiple log() calls with the same step into ONE row.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: list[str] = []

        # --- NEW: step aggregation ---
        self._pending_step: Optional[int] = None
        self._pending_row: Dict[str, Any] = {}

    def _ensure_writer(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        else:
            # extend header if new keys appear
            new_keys = [k for k in row.keys() if k not in self._fieldnames]
            if new_keys:
                self._fieldnames += new_keys
                assert self._writer is not None
                self._writer.fieldnames = self._fieldnames

    def _flush_pending(self) -> None:
        if self._pending_step is None:
            return
        row = {"step": int(self._pending_step), **self._pending_row}

        # ensure writer + header covers all keys
        self._ensure_writer(row)

        # fill missing keys
        for k in self._fieldnames:
            row.setdefault(k, "")

        assert self._writer is not None
        self._writer.writerow(row)
        self._file.flush()

        # clear
        self._pending_step = None
        self._pending_row = {}

    def log(self, step: int, data: Dict[str, Any]) -> None:
        step = int(step)

        # first log
        if self._pending_step is None:
            self._pending_step = step
            self._pending_row = dict(data)
            return

        # same step -> merge
        if step == self._pending_step:
            self._pending_row.update(data)
            return

        # step advanced (or went backwards): flush old then start new
        self._flush_pending()
        self._pending_step = step
        self._pending_row = dict(data)

    def close(self) -> None:
        try:
            self._flush_pending()
        except Exception:
            pass
        try:
            self._file.close()
        except Exception:
            pass


class SwanLabLogger:
    """
    Fail-open SwanLab logger (async background thread).
    If swanlab import/log fails, it disables itself and training continues.
    """
    def __init__(self, enabled: bool, project: str, run_name: str, config: Dict[str, Any]):
        self.enabled = bool(enabled)
        self.failed = False

        self._q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=2048)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sl = None

        if not self.enabled:
            return

        try:
            import swanlab  # type: ignore
            self._sl = swanlab
            self._sl.init(project=project, experiment_name=run_name, config=config)
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        except Exception:
            self.enabled = False
            self.failed = True

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                return
            try:
                if self._sl is not None:
                    self._sl.log(item)
            except Exception:
                self.enabled = False
                self.failed = True
                return

    def log(self, data: Dict[str, Any]) -> None:
        if (not self.enabled) or self.failed:
            return
        try:
            self._q.put_nowait(dict(data))
        except Exception:
            pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except Exception:
            pass


@dataclass
class RunLogger:
    """
    Runner-side unified logger:
      logger.log(step, {"train/loss":..., "val/acc":...})
    """
    csv: CSVLogger
    swan: SwanLabLogger

    def log(self, step: int, data: Dict[str, Any]) -> None:
        self.csv.log(step, data)
        self.swan.log({"step": int(step), **data})

    def close(self) -> None:
        self.csv.close()
        self.swan.close()


class ExperimentLogger:
    """
    Trainer-side compatibility wrapper:
      logger.log_step({"train/loss": ...}, step=xxx)
    """
    def __init__(self, csv_logger: CSVLogger, swan_logger: SwanLabLogger):
        self._r = RunLogger(csv=csv_logger, swan=swan_logger)

    def log_step(self, data: Dict[str, Any], step: int) -> None:
        self._r.log(step, data)

    def close(self) -> None:
        self._r.close()
