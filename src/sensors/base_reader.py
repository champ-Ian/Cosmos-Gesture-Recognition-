#!/usr/bin/env python3
"""
Shared interface every sensor reader (`imu_reader.py`, `uwb_reader.py`,
`mmwave_reader.py`, `rfid_reader.py`) implements.

Each reader opens its device/connection in `__init__` and immediately starts
a background thread that keeps streaming samples into an internal buffer,
tagged with `time.monotonic()` receive timestamps -- the same host clock
`collect.py`'s coordinator uses for its `events.csv` markers. `window()`
lets the coordinator (live, during collection) or the offline cutting step
in `extract_features.py` (from a continuous per-sensor log) slice out only
the samples that fall inside a given time range, without needing hardware-
level sync between boards.

`window()`'s return shape is reader-specific (mmWave/UWB return dicts of
packed arrays; IMU/RFID return `list[(relative_time_s, raw_line)]`) --
there's no single sample schema that fits a radar frame and a text line
equally well, so this base class doesn't try to force one.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseReader(ABC):
    name: str

    @abstractmethod
    def check_error(self) -> None:
        """Raise RuntimeError if the background thread hit an unrecoverable error."""

    @property
    @abstractmethod
    def sample_count(self) -> int:
        """Total samples/frames/lines received so far this session."""

    @abstractmethod
    def window(self, start_time_s: float, end_time_s: float) -> Any:
        """Return samples received in [start_time_s, end_time_s]. Shape is reader-specific."""

    @abstractmethod
    def close(self) -> None:
        """Stop the background thread and release the underlying hardware/socket."""
