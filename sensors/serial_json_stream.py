#!/usr/bin/env python3
"""
Generic background-thread reader for the ESP32-based IoT sensors (IMU, UWB,
RFID) shipped in the COSMOS project box.

Unlike the mmWave radar, these boards do not have a single documented binary
protocol here, and each group's firmware may still change field names or
sample rate while the project is being built. Rather than guessing a schema
and silently dropping anything that does not match it, this reader stores the
**raw newline-delimited text** each board prints over USB serial, tagged with
a `time.monotonic()` receive timestamp. That raw log is never lossy: whatever
the firmware prints is exactly what ends up in the trial `.npz` file, and
feature extraction (parsing JSON, picking out fields) can be written or fixed
later without recollecting data.

Expected firmware contract (recommended, not enforced):
    One JSON object per line, newline-terminated, e.g.:
        IMU  : {"t_ms": 12345, "ax": 0.01, "ay": 0.98, "az": 0.03,
                 "gx": 1.2, "gy": -0.3, "gz": 0.1}
        UWB  : {"t_ms": 12345, "ranges_m": {"anchor_0": 1.23, "anchor_1": 0.87}}
        RFID : {"t_ms": 12345, "tag_id": "E200...", "rssi": -41.5}
If your firmware prints plain CSV or something else, that is fine too -- the
raw line is stored either way. Update `sensors/parsing.py` (or write your own)
to match whatever your boards actually send.
"""
from __future__ import annotations

import threading
import time

import serial


class SerialLineStream:
    """Reads newline-terminated text lines from a serial port on a background thread."""

    def __init__(self, name: str, port_path: str, baud: int = 115200, read_timeout_s: float = 0.2) -> None:
        self.name = name
        self.port_path = port_path
        self.port = serial.Serial(port_path, baud, timeout=read_timeout_s)

        self._lock = threading.Lock()
        self._buffer: list[tuple[float, str]] = []
        self._stop_event = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"{name}-reader")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                raw = self.port.readline()
            except serial.SerialException as error:
                self._error = error
                return

            if not raw:
                continue

            recv_time = time.monotonic()
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue

            with self._lock:
                self._buffer.append((recv_time, text))

    def check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"{self.name} serial stream failed: {self._error}") from self._error

    @property
    def line_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def latest_line(self) -> str | None:
        with self._lock:
            return self._buffer[-1][1] if self._buffer else None

    def window(self, start_time_s: float, end_time_s: float) -> list[tuple[float, str]]:
        """Return (relative_time_s, raw_line) pairs received in [start_time_s, end_time_s]."""
        with self._lock:
            return [
                (recv_time - start_time_s, line)
                for recv_time, line in self._buffer
                if start_time_s <= recv_time <= end_time_s
            ]

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self.port.close()
