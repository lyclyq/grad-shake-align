from __future__ import annotations

import csv
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class CSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open('w', newline='', encoding='utf-8')
        self._writer: Optional[csv.DictWriter] = None

    def log(self, step: int, data: Dict[str, Any]):
        row = {'step': step}
        row.update(data)
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        else:
            # if new keys appear, rewrite header is messy; so we fill missing keys.
            for k in row.keys():
                if k not in self._writer.fieldnames:
                    # extend fieldnames
                    self._writer.fieldnames = list(self._writer.fieldnames) + [k]
        # ensure all keys exist
        for k in self._writer.fieldnames:
            row.setdefault(k, '')
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass


class SwanLabLogger:
    """Fail-open SwanLab logger.

    SwanLab sometimes blocks / hangs depending on environment.
    We push logs through a background thread; on any exception, we disable.
    """

    def __init__(self, enabled: bool, project: str, run_name: str, config: Dict[str, Any]):
        self.enabled = enabled
        self.failed = False
        self._q: 'queue.Queue[Dict[str, Any]]' = queue.Queue(maxsize=1024)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._sl = None

        if not enabled:
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

    def _worker(self):
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
                self.failed = True
                self.enabled = False
                return

    def log(self, data: Dict[str, Any]):
        if not self.enabled or self.failed:
            return
        try:
            self._q.put_nowait(dict(data))
        except Exception:
            # drop if queue is full
            pass

    def close(self):
        self._stop.set()
        try:
            self._q.put_nowait(None)  # type: ignore
        except Exception:
            pass


@dataclass
class RunLogger:
    csv: CSVLogger
    swan: SwanLabLogger

    def log(self, step: int, data: Dict[str, Any]):
        self.csv.log(step, data)
        self.swan.log({'step': step, **data})

    def close(self):
        self.csv.close()
        self.swan.close()
