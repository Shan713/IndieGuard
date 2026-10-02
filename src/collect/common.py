"""Shared helpers for the collectors: throttled HTTP session, logging, atomic writes, keep-awake."""
from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
LOGS = ROOT / "logs"

UA = "IndieGuard-capstone-research/1.0 (academic project; contact via GitHub)"
# Age-gate cookies so mature titles return their store page instead of the age check.
STORE_COOKIES = {"birthtime": "568022401", "lastagecheckage": "1-0-1988", "mature_content": "1", "wants_mature_content": "1"}


def get_logger(name: str) -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOGS / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (no system settings are changed)."""
    if os.name == "nt":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


class Throttled:
    """requests.Session wrapper: minimum gap between calls, backoff on 429/5xx/network errors."""

    def __init__(self, min_interval: float, log: logging.Logger, block_wait: float = 300.0, max_tries: int = 8,
                 adaptive: bool = False, floor: float = 1.25, ceiling: float = 3.0):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.s.cookies.update(STORE_COOKIES)
        self.min_interval = min_interval
        self.block_wait = block_wait
        self.max_tries = max_tries
        self.log = log
        self._last = 0.0
        self.n_requests = 0
        self.n_429 = 0
        # Adaptive pacing: slow down 15% after a 429, speed up 3% after 300 clean requests.
        self.adaptive, self.floor, self.ceiling = adaptive, floor, ceiling
        self._clean = 0

    def _wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()

    def get(self, url: str, params: dict | None = None, timeout: float = 60) -> requests.Response | None:
        """Returns the response (2xx, or a 4xx other than 429), or None after max_tries failures."""
        for attempt in range(1, self.max_tries + 1):
            self._wait()
            try:
                r = self.s.get(url, params=params, timeout=timeout)
                self.n_requests += 1
            except requests.RequestException as e:
                wait = min(30 * attempt, 300)
                self.log.warning("network error %s (attempt %d), sleeping %ds: %s", url, attempt, wait, e)
                time.sleep(wait)
                continue
            if r.status_code == 429:
                self.n_429 += 1
                if self.adaptive:
                    self._clean = 0
                    self.min_interval = min(self.min_interval * 1.15, self.ceiling)
                    self.log.info("pacing now %.2fs/request", self.min_interval)
                ra = r.headers.get("Retry-After")
                # Escalating wait: blocks often clear well before the worst-case ~4.5 min.
                wait = int(ra) + 5 if ra and ra.isdigit() else min(self.block_wait * attempt, 600)
                self.log.warning("429 on %s (attempt %d), sleeping %ds", url, attempt, wait)
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                wait = min(20 * attempt, 300)
                self.log.warning("HTTP %d on %s (attempt %d), sleeping %ds", r.status_code, url, attempt, wait)
                time.sleep(wait)
                continue
            if self.adaptive:
                self._clean += 1
                if self._clean >= 300 and self.min_interval > self.floor:
                    self._clean = 0
                    self.min_interval = max(self.min_interval * 0.97, self.floor)
                    self.log.info("pacing now %.2fs/request", self.min_interval)
            return r
        self.log.error("giving up on %s %s", url, params)
        return None
