#!/usr/bin/env python3
"""
Background-thread reader for the ESP32-based IMU board (`IMU_lab_students`)
shipped in the COSMOS project box.

UWB and RFID are NOT read this way -- UWB is driven by `sensors/uwb_reader.py`
(subprocess-based FiRa ranging, not raw serial text) and RFID is a network
device read by `sensors/rfid_reader.py` (TCP socket, not serial).

This reader stores the **raw newline-delimited text** the IMU board prints
over USB serial, tagged with a `time.monotonic()` receive timestamp, rather
than parsing at capture time. That raw log is never lossy: whatever the
firmware prints is exactly what ends up in the continuous session log (or
the cut trial `.npz` file), and feature extraction (`extract_features.py`)
can be fixed later without recollecting data.

Firmware contract (`IMU_lab_students/main/main.c`): one line per sample,
newline-terminated, e.g.:
    accel[g] x= 0.012 y=-0.034 z= 0.998 | gyro[dps] x= 0.10 y=-0.20 z= 0.05
If your group's firmware prints something else, the raw line is still stored
either way -- update the parser in `extract_features.py` (`_IMU_SAMPLE_RE`)
to match whatever your board actually sends.
"""
from __future__ import annotations

import threading
import time

import serial

from sensors.base_reader import BaseReader


class ImuReader(BaseReader):
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
    def sample_count(self) -> int:
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
