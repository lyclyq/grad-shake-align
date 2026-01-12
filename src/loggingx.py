from __future__ import annotations

import csv
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open('w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


class SwanLabLogger:
    """Fail-open SwanLab logger with async queue and error fuse."""

    def __init__(self, cfg: Dict[str, Any], run_name: str, config: Dict[str, Any]):
        sl_cfg = cfg.get('swanlab', {})
        self.enabled = bool(sl_cfg.get('enabled', False))
        self.failed = False
        self.max_queue = int(sl_cfg.get('max_queue', 2000))
        self.fail_after = int(sl_cfg.get('fail_after_errors', 3))
        self._errors = 0
        self._q: 'queue.Queue[Dict[str, Any]]' = queue.Queue(maxsize=self.max_queue)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._sl = None

        if not self.enabled:
            return
        try:
            import swanlab  # type: ignore
            self._sl = swanlab
            self._sl.init(project=sl_cfg.get('project', 'ShakeAlign'), experiment_name=run_name, config=config)
            if bool(sl_cfg.get('async', True)):
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
                self._errors += 1
                if self._errors >= self.fail_after:
                    self.failed = True
                    self.enabled = False
                    return

    def log(self, data: Dict[str, Any]) -> None:
        if not self.enabled or self.failed:
            return
        if self._thread is None and self._sl is not None:
            try:
                self._sl.log(data)
            except Exception:
                self._errors += 1
                if self._errors >= self.fail_after:
                    self.failed = True
                    self.enabled = False
            return
        try:
            self._q.put_nowait(dict(data))
        except Exception:
            pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)  # type: ignore[arg-type]
        except Exception:
            pass


@dataclass
class ExperimentLogger:
    run_dir: Path
    run_name: str
    cfg: Dict[str, Any]
    seed: int
    trial_id: Optional[int]
    csv: CSVLogger
    swan: SwanLabLogger
    events_path: Path

    @classmethod
    def create(
        cls,
        run_dir: str | Path,
        run_name: str,
        cfg: Dict[str, Any],
        seed: int,
        trial_id: Optional[int] = None,
    ) -> 'ExperimentLogger':
        run_dir = Path(run_dir)
        fields = ['step', 'epoch', 'split', 'metric', 'value', 'seed', 'trial_id']
        csv_logger = CSVLogger(run_dir / 'metrics.csv', fieldnames=fields)
        swan_logger = SwanLabLogger(cfg.get('log', {}), run_name, config=cfg)
        events_path = run_dir / 'events.log'
        events_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(run_dir, run_name, cfg, seed, trial_id, csv_logger, swan_logger, events_path)

    def log_metric(self, split: str, metric: str, value: float, step: int, epoch: int | None = None) -> None:
        row = {
            'step': step,
            'epoch': epoch if epoch is not None else '',
            'split': split,
            'metric': metric,
            'value': float(value),
            'seed': self.seed,
            'trial_id': '' if self.trial_id is None else self.trial_id,
        }
        self.csv.log(row)
        self.swan.log({'step': step, f'{split}/{metric}': float(value)})

    def log_metrics(self, split: str, metrics: Dict[str, float], step: int, epoch: int | None = None) -> None:
        for k, v in metrics.items():
            self.log_metric(split, k, v, step=step, epoch=epoch)

    def log_event(self, message: str, level: str = 'INFO') -> None:
        with self.events_path.open('a', encoding='utf-8') as f:
            f.write(f'[{level}] {message}\n')

    def close(self) -> None:
        self.csv.close()
        self.swan.close()
