#!/usr/bin/env python3
"""
Background reader for a Qorvo DWM3001CDK FiRa TWR ranging pair.

Unlike the mmWave radar (one board, one serial port, binary UART frames) the
UWB kit for this project is two DWM3001CDK boards -- a controller/initiator
and a controlee/responder, each on its own serial port -- running the FiRa
ranging demo from the vendored `uwb-qorvo-tools` CLI. There is no direct
Python UART protocol here: ranging is started and read by launching
`run_fira_twr.py` as a subprocess per board (exactly as `UWB_lab`'s
`ranging_experiment_wrapper.py` / `collect_dataset.py` do), then parsing the
controller's stdout with `RangeLogParser` for `distance: X cm` lines.

Only the controller side reports distance; the controlee subprocess just
needs to be running so the controller has someone to range against.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from uwb.uwb_io import (
    RangeLogParser,
    compute_ranging_span_ms,
    process_env,
    reset_devices,
    stop_process,
    twr_command,
    validate_timing,
)


class UwbStream:
    """Runs a controller+controlee FiRa TWR pair and buffers parsed range samples."""

    def __init__(
        self,
        controller_port: str,
        controlee_port: str,
        group_id: int,
        log_dir: Path,
        preamble_code: int = 10,
        channel: int = 9,
        fps: float = 50.0,
        ranging_span_ms: int | None = None,
        slot_span: int = 2400,
        slots_per_rr: int = 6,
        python_exe: str | None = None,
        reset_devices_first: bool = True,
        startup_delay_s: float = 3.0,
        session_duration_s: int = 3600,
    ) -> None:
        self.controller_port = controller_port
        self.controlee_port = controlee_port
        self.group_id = group_id
        self.python_exe = python_exe or sys.executable
        self.ranging_span_ms = compute_ranging_span_ms(fps, ranging_span_ms)
        validate_timing(slot_span, slots_per_rr, self.ranging_span_ms)

        self.log_dir = Path(log_dir)
        controller_dir = self.log_dir / "controller"
        controlee_dir = self.log_dir / "controlee"
        controller_dir.mkdir(parents=True, exist_ok=True)
        controlee_dir.mkdir(parents=True, exist_ok=True)

        if reset_devices_first:
            print(f"Resetting UWB devices ({controller_port}, {controlee_port})...")
            reset_devices(
                self.python_exe,
                [controller_port, controlee_port],
                self.log_dir / "device_reset_log.txt",
            )

        self._lock = threading.Lock()
        self._buffer: list[tuple[float, object]] = []
        self._stop_event = threading.Event()
        self._error: Exception | None = None
        self._closing = False

        controlee_duration = session_duration_s + 5 + int(startup_delay_s)
        controlee_cmd = twr_command(
            self.python_exe,
            controlee_port,
            preamble_code,
            controlee_duration,
            slot_span,
            slots_per_rr,
            self.ranging_span_ms,
            channel=channel,
            controlee=True,
            stats=True,
        )
        controller_cmd = twr_command(
            self.python_exe,
            controller_port,
            preamble_code,
            session_duration_s,
            slot_span,
            slots_per_rr,
            self.ranging_span_ms,
            channel=channel,
            controlee=False,
            stats=True,
        )

        print(f"Starting UWB controlee on {controlee_port}...")
        self._controlee_log = open(controlee_dir / "controlee_terminal_log.txt", "w", buffering=1)
        self._controlee_proc = subprocess.Popen(
            controlee_cmd,
            cwd=controlee_dir,
            stdout=self._controlee_log,
            stderr=subprocess.STDOUT,
            text=True,
            env=process_env(),
            start_new_session=True,
        )
        time.sleep(startup_delay_s)

        print(f"Starting UWB controller on {controller_port}...")
        self._controller_log = open(controller_dir / "controller_terminal_log.txt", "w", buffering=1)
        self._controller_proc = subprocess.Popen(
            controller_cmd,
            cwd=controller_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env(),
            start_new_session=True,
        )

        self._parser = RangeLogParser()
        self._thread = threading.Thread(target=self._run, daemon=True, name="uwb-controller-reader")
        self._thread.start()

    def _run(self) -> None:
        try:
            assert self._controller_proc.stdout is not None
            for line in self._controller_proc.stdout:
                self._controller_log.write(line)
                sample = self._parser.feed(line)
                if sample:
                    recv_time = time.monotonic()
                    with self._lock:
                        self._buffer.append((recv_time, sample))
        except Exception as error:  # noqa: BLE001 - surfaced via check_error()
            self._error = error
        finally:
            self._controller_log.close()

    def check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"UWB ranging stream failed: {self._error}") from self._error
        if not self._closing and self._controller_proc.poll() is not None:
            raise RuntimeError(
                f"UWB controller process exited early with code {self._controller_proc.returncode}"
            )

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def window(self, start_time_s: float, end_time_s: float) -> dict[str, np.ndarray]:
        """Return samples received in [start_time_s, end_time_s] as packed arrays."""
        with self._lock:
            samples = [
                (recv_time - start_time_s, sample)
                for recv_time, sample in self._buffer
                if start_time_s <= recv_time <= end_time_s
            ]

        return {
            "time_s": np.array([t for t, _ in samples], dtype=float),
            "sequence": np.array(
                [s.sequence if s.sequence is not None else -1 for _, s in samples], dtype=np.int32
            ),
            "status": np.array([s.status for _, s in samples], dtype=object),
            "distance_cm": np.array([s.distance_cm for _, s in samples], dtype=float),
        }

    def close(self, final_reset: bool = True) -> None:
        self._closing = True
        self._stop_event.set()
        stop_process(self._controller_proc)
        self._thread.join(timeout=3)
        stop_process(self._controlee_proc, self._controlee_log)

        if final_reset:
            print("Resetting UWB devices after stream close...")
            try:
                reset_devices(
                    self.python_exe,
                    [self.controller_port, self.controlee_port],
                    self.log_dir / "final_device_reset_log.txt",
                )
            except RuntimeError as error:
                print(f"Warning: final UWB device reset failed: {error}")
