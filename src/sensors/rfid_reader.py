#!/usr/bin/env python3
"""
Background-thread reader for the RFID reader used in `RFID_Lab`.

Unlike IMU (USB serial), the RFID reader is a network device: it streams
newline-terminated text reports over a **TCP socket** (see
`RFID_Lab/reading_from_TCP.py` / `touch_detector_gui.py`, both of which
connect to `192.168.137.1:9055` by default -- the reader's own network
address, not a serial port). Each line reports one tag read:

    <EPC> <timestamp...> <RSSI> <read_count>

e.g. `E2806995000040154D38514E 2024-01-01 12:00:00.123 -45 3`. The reader
streams a line for every tag it sees in range, not just tags you care
about -- filter by EPC downstream (see `extract_features.py`) if your
gestures use specific tags.

Like `ImuReader`, this stores the **raw line** tagged with a
`time.monotonic()` receive timestamp and does no parsing at capture time --
`.window()` returns the same `list[(relative_time_s, raw_line)]` shape so
`collect.py` and `realtime_demo.py` can treat it the same way as the serial
sensors.
"""
from __future__ import annotations

import socket
import threading
import time

from sensors.base_reader import BaseReader


class RfidReader(BaseReader):
    """Reads newline-terminated text lines from the RFID reader's TCP socket on a background thread."""

    def __init__(
        self,
        host: str,
        port: int,
        connect_timeout_s: float = 10.0,
        read_timeout_s: float = 0.5,
    ) -> None:
        self.name = "rfid"
        self.host = host
        self.port = port
        self.socket = socket.create_connection((host, port), timeout=connect_timeout_s)
        self.socket.settimeout(read_timeout_s)

        self._lock = threading.Lock()
        self._buffer: list[tuple[float, str]] = []
        self._stop_event = threading.Event()
        self._error: Exception | None = None
        self._recv_buffer = b""
        self._thread = threading.Thread(target=self._run, daemon=True, name="rfid-reader")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                data = self.socket.recv(4096)
            except socket.timeout:
                continue
            except OSError as error:
                if not self._stop_event.is_set():
                    self._error = error
                return

            if not data:
                if not self._stop_event.is_set():
                    self._error = ConnectionError("RFID reader closed the TCP connection.")
                return

            recv_time = time.monotonic()
            self._recv_buffer += data
            while b"\n" in self._recv_buffer:
                raw_line, self._recv_buffer = self._recv_buffer.split(b"\n", 1)
                text = raw_line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                with self._lock:
                    self._buffer.append((recv_time, text))

    def check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"{self.name} TCP stream failed: {self._error}") from self._error

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
        self.socket.close()
