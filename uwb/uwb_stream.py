#!/usr/bin/env python3
"""
Background reader for a Qorvo DWM3001CDK FiRa TWR ranging setup: one worn
"tag" board ranging against one or more fixed "anchor" boards.

There is no direct Python UART protocol here: ranging is started and read by
launching `run_fira_twr.py` (vendored in `uwb/uwb-qorvo-tools/`) as a
subprocess per board -- one for the tag (FiRa "controller" role) and one per
anchor (FiRa "controlee" role) -- then parsing the tag's stdout with
`RangeLogParser` for `distance: X cm` / `mac address: ...` lines.

Two node modes, both driven by the same class:

- One anchor (`len(anchor_ports) == 1`): plain FiRa unicast TWR, exactly the
  controller/controlee pair `UWB_lab`'s lab exercise and tooling use. This
  path matches a documented, working lab flow.
- Multiple anchors (`len(anchor_ports) > 1`): FiRa "one-to-many" ranging
  (`--node onetomany`), where the tag is the one controller and each anchor
  is a controlee with a distinct MAC address (`uwb-qorvo-tools`'s own
  `--n_controlees`/`--mac`/`--dest-mac` flags). This mode is real and
  supported by the vendored CLI/UCI stack, but it is **not** exercised by
  `UWB_lab`'s documented lab and has not been verified against physical
  DWM3001CDK hardware here -- smoke-test it (short `--stats` run, check that
  every anchor's MAC shows up with `status: Ok`) before trusting it for real
  data collection, and re-check `--slots-per-rr` if ranging looks unstable:
  more controlees need more slots per ranging round than the single-anchor
  default.

Only the tag/controller side reports distance; anchor/controlee subprocesses
just need to be running so the tag has someone to range against.
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
    """Runs a tag+anchor(s) FiRa TWR setup and buffers parsed range samples."""

    def __init__(
        self,
        tag_port: str,
        anchor_ports: list[str],
        group_id: int,
        log_dir: Path,
        preamble_code: int = 10,
        channel: int = 9,
        fps: float = 50.0,
        ranging_span_ms: int | None = None,
        slot_span: int = 2400,
        slots_per_rr: int | None = None,
        python_exe: str | None = None,
        reset_devices_first: bool = True,
        startup_delay_s: float = 3.0,
        session_duration_s: int = 3600,
    ) -> None:
        if not anchor_ports:
            raise ValueError("UwbStream needs at least one anchor port.")

        self.tag_port = tag_port
        self.anchor_ports = list(anchor_ports)
        self.group_id = group_id
        self.python_exe = python_exe or sys.executable
        self.ranging_span_ms = compute_ranging_span_ms(fps, ranging_span_ms)

        n_anchors = len(self.anchor_ports)
        self.multi_anchor = n_anchors > 1
        if slots_per_rr is None:
            # Single-anchor default matches UWB_lab's documented lab exercise.
            # The one-to-many heuristic below has NOT been hardware-verified;
            # tune it if ranging is unstable with more anchors.
            slots_per_rr = 6 if not self.multi_anchor else 6 * n_anchors
        validate_timing(slot_span, slots_per_rr, self.ranging_span_ms)
        self.slots_per_rr = slots_per_rr

        self.log_dir = Path(log_dir)
        tag_dir = self.log_dir / "tag"
        tag_dir.mkdir(parents=True, exist_ok=True)
        anchor_dirs = []
        for i in range(n_anchors):
            anchor_dir = self.log_dir / f"anchor_{i}"
            anchor_dir.mkdir(parents=True, exist_ok=True)
            anchor_dirs.append(anchor_dir)

        if reset_devices_first:
            print(f"Resetting UWB devices (tag {tag_port}, anchors {self.anchor_ports})...")
            reset_devices(
                self.python_exe,
                [tag_port, *self.anchor_ports],
                self.log_dir / "device_reset_log.txt",
            )

        self._lock = threading.Lock()
        self._buffer: list[tuple[float, object]] = []
        self._stop_event = threading.Event()
        self._error: Exception | None = None
        self._closing = False

        anchor_duration = session_duration_s + 5 + int(startup_delay_s)
        node_mode = "onetomany" if self.multi_anchor else "unicast"
        # Anchor MACs are 0x1..0xN; the tag stays at the CLI's own default (0x0)
        # unless we're in one-to-many mode, where it must be set explicitly
        # alongside --dest-mac / --n_controlees.
        anchor_macs = [f"0x{i + 1}" for i in range(n_anchors)]

        self._anchor_procs: list[subprocess.Popen] = []
        self._anchor_logs: list = []
        for anchor_port, anchor_dir, anchor_mac in zip(self.anchor_ports, anchor_dirs, anchor_macs):
            print(f"Starting UWB anchor on {anchor_port} (mac {anchor_mac})...")
            anchor_cmd = twr_command(
                self.python_exe,
                anchor_port,
                preamble_code,
                anchor_duration,
                slot_span,
                slots_per_rr,
                self.ranging_span_ms,
                channel=channel,
                controlee=True,
                stats=True,
                node_mode=node_mode,
                n_controlees=n_anchors,
                mac=anchor_mac if self.multi_anchor else None,
                dest_mac="[0x0]" if self.multi_anchor else None,
            )
            anchor_log = open(anchor_dir / "anchor_terminal_log.txt", "w", buffering=1)
            anchor_proc = subprocess.Popen(
                anchor_cmd,
                cwd=anchor_dir,
                stdout=anchor_log,
                stderr=subprocess.STDOUT,
                text=True,
                env=process_env(),
                start_new_session=True,
            )
            self._anchor_procs.append(anchor_proc)
            self._anchor_logs.append(anchor_log)
        time.sleep(startup_delay_s)

        print(f"Starting UWB tag on {tag_port}...")
        tag_cmd = twr_command(
            self.python_exe,
            tag_port,
            preamble_code,
            session_duration_s,
            slot_span,
            slots_per_rr,
            self.ranging_span_ms,
            channel=channel,
            controlee=False,
            stats=True,
            node_mode=node_mode,
            n_controlees=n_anchors,
            mac="0x0" if self.multi_anchor else None,
            dest_mac=f"[{','.join(anchor_macs)}]" if self.multi_anchor else None,
        )
        self._tag_log = open(tag_dir / "tag_terminal_log.txt", "w", buffering=1)
        self._tag_proc = subprocess.Popen(
            tag_cmd,
            cwd=tag_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env(),
            start_new_session=True,
        )

        self._parser = RangeLogParser()
        self._thread = threading.Thread(target=self._run, daemon=True, name="uwb-tag-reader")
        self._thread.start()

    def _run(self) -> None:
        try:
            assert self._tag_proc.stdout is not None
            for line in self._tag_proc.stdout:
                self._tag_log.write(line)
                sample = self._parser.feed(line)
                if sample:
                    recv_time = time.monotonic()
                    with self._lock:
                        self._buffer.append((recv_time, sample))
        except Exception as error:  # noqa: BLE001 - surfaced via check_error()
            self._error = error
        finally:
            self._tag_log.close()

    def check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"UWB ranging stream failed: {self._error}") from self._error
        if not self._closing and self._tag_proc.poll() is not None:
            raise RuntimeError(
                f"UWB tag process exited early with code {self._tag_proc.returncode}"
            )

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def window(self, start_time_s: float, end_time_s: float) -> dict[str, np.ndarray]:
        """Return samples received in [start_time_s, end_time_s] as packed arrays.

        One row per (ranging round, anchor) measurement -- `mac_address` is
        the only thing that distinguishes anchors from each other when more
        than one is configured.
        """
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
            "mac_address": np.array([s.mac_address for _, s in samples], dtype=object),
            "status": np.array([s.status for _, s in samples], dtype=object),
            "distance_cm": np.array([s.distance_cm for _, s in samples], dtype=float),
        }

    def close(self, final_reset: bool = True) -> None:
        self._closing = True
        self._stop_event.set()
        stop_process(self._tag_proc)
        self._thread.join(timeout=3)
        for anchor_proc, anchor_log in zip(self._anchor_procs, self._anchor_logs):
            stop_process(anchor_proc, anchor_log)

        if final_reset:
            print("Resetting UWB devices after stream close...")
            try:
                reset_devices(
                    self.python_exe,
                    [self.tag_port, *self.anchor_ports],
                    self.log_dir / "final_device_reset_log.txt",
                )
            except RuntimeError as error:
                print(f"Warning: final UWB device reset failed: {error}")
