#!/usr/bin/env python3
"""
Background reader for a Qorvo DWM3001CDK FiRa TWR ranging setup: one fixed
"anchor" board ranging against one or more worn "node" boards.

There is no direct Python UART protocol here: ranging is started and read by
launching `run_fira_twr.py` (vendored in `uwb/uwb-qorvo-tools/`) as a
subprocess per board -- one for the anchor (FiRa "controller" role) and one
per node (FiRa "controlee" role) -- then parsing the anchor's stdout with
`RangeLogParser` for `distance: X cm` / `mac address: ...` lines.

Two modes, both driven by the same class:

- One node (`len(node_ports) == 1`): plain FiRa unicast TWR, exactly the
  controller/controlee pair `UWB_lab`'s lab exercise and tooling use. This
  path matches a documented, working lab flow.
- Multiple nodes (`len(node_ports) > 1`, e.g. this project's 1 anchor + 2
  worn nodes): FiRa "one-to-many" ranging (`--node onetomany`), where the
  anchor is the one controller and each node is a controlee with a distinct
  MAC address (`uwb-qorvo-tools`'s own `--n_controlees`/`--mac`/`--dest-mac`
  flags). This mode is real and supported by the vendored CLI/UCI stack, but
  it is **not** exercised by `UWB_lab`'s documented lab and has not been
  verified against physical DWM3001CDK hardware here -- smoke-test it (short
  `--stats` run, check that every node's MAC shows up with `status: Ok`)
  before trusting it for real data collection, and re-check
  `--slots-per-rr` if ranging looks unstable: more controlees need more
  slots per ranging round than the single-node default.

Only the anchor/controller side reports distance; node/controlee
subprocesses just need to be running so the anchor has someone to range
against.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from sensors.base_reader import BaseReader
from uwb.uwb_io import (
    RangeLogParser,
    compute_ranging_span_ms,
    process_env,
    reset_devices,
    stop_process,
    twr_command,
    validate_timing,
)


class UwbReader(BaseReader):
    """Runs an anchor+node(s) FiRa TWR setup and buffers parsed range samples."""

    def __init__(
        self,
        anchor_port: str,
        node_ports: list[str],
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
        if not node_ports:
            raise ValueError("UwbReader needs at least one node port.")

        self.name = "uwb"
        self.anchor_port = anchor_port
        self.node_ports = list(node_ports)
        self.group_id = group_id
        self.python_exe = python_exe or sys.executable
        self.ranging_span_ms = compute_ranging_span_ms(fps, ranging_span_ms)

        n_nodes = len(self.node_ports)
        self.multi_node = n_nodes > 1
        if slots_per_rr is None:
            # Single-node default matches UWB_lab's documented lab exercise.
            # The one-to-many heuristic below has NOT been hardware-verified;
            # tune it if ranging is unstable with more nodes.
            slots_per_rr = 6 if not self.multi_node else 6 * n_nodes
        validate_timing(slot_span, slots_per_rr, self.ranging_span_ms)
        self.slots_per_rr = slots_per_rr

        self.log_dir = Path(log_dir)
        anchor_dir = self.log_dir / "anchor"
        anchor_dir.mkdir(parents=True, exist_ok=True)
        node_dirs = []
        for i in range(n_nodes):
            node_dir = self.log_dir / f"node_{i}"
            node_dir.mkdir(parents=True, exist_ok=True)
            node_dirs.append(node_dir)

        if reset_devices_first:
            print(f"Resetting UWB devices (anchor {anchor_port}, nodes {self.node_ports})...")
            reset_devices(
                self.python_exe,
                [anchor_port, *self.node_ports],
                self.log_dir / "device_reset_log.txt",
            )

        self._lock = threading.Lock()
        self._buffer: list[tuple[float, object]] = []
        self._stop_event = threading.Event()
        self._error: Exception | None = None
        self._closing = False

        node_duration = session_duration_s + 5 + int(startup_delay_s)
        node_mode = "onetomany" if self.multi_node else "unicast"
        # Node MACs are 0x1..0xN; the anchor stays at the CLI's own default
        # (0x0) unless we're in one-to-many mode, where it must be set
        # explicitly alongside --dest-mac / --n_controlees.
        node_macs = [f"0x{i + 1}" for i in range(n_nodes)]

        self._node_procs: list[subprocess.Popen] = []
        self._node_logs: list = []
        for node_port, node_dir, node_mac in zip(self.node_ports, node_dirs, node_macs):
            print(f"Starting UWB node on {node_port} (mac {node_mac})...")
            node_cmd = twr_command(
                self.python_exe,
                node_port,
                preamble_code,
                node_duration,
                slot_span,
                slots_per_rr,
                self.ranging_span_ms,
                channel=channel,
                controlee=True,
                stats=True,
                node_mode=node_mode,
                n_controlees=n_nodes,
                mac=node_mac if self.multi_node else None,
                dest_mac="[0x0]" if self.multi_node else None,
            )
            node_log = open(node_dir / "node_terminal_log.txt", "w", buffering=1)
            node_proc = subprocess.Popen(
                node_cmd,
                cwd=node_dir,
                stdout=node_log,
                stderr=subprocess.STDOUT,
                text=True,
                env=process_env(),
                start_new_session=True,
            )
            self._node_procs.append(node_proc)
            self._node_logs.append(node_log)
        time.sleep(startup_delay_s)

        print(f"Starting UWB anchor on {anchor_port}...")
        anchor_cmd = twr_command(
            self.python_exe,
            anchor_port,
            preamble_code,
            session_duration_s,
            slot_span,
            slots_per_rr,
            self.ranging_span_ms,
            channel=channel,
            controlee=False,
            stats=True,
            node_mode=node_mode,
            n_controlees=n_nodes,
            mac="0x0" if self.multi_node else None,
            dest_mac=f"[{','.join(node_macs)}]" if self.multi_node else None,
        )
        self._anchor_log = open(anchor_dir / "anchor_terminal_log.txt", "w", buffering=1)
        self._anchor_proc = subprocess.Popen(
            anchor_cmd,
            cwd=anchor_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env(),
            start_new_session=True,
        )

        self._parser = RangeLogParser()
        self._thread = threading.Thread(target=self._run, daemon=True, name="uwb-anchor-reader")
        self._thread.start()

    def _run(self) -> None:
        try:
            assert self._anchor_proc.stdout is not None
            for line in self._anchor_proc.stdout:
                self._anchor_log.write(line)
                sample = self._parser.feed(line)
                if sample:
                    recv_time = time.monotonic()
                    with self._lock:
                        self._buffer.append((recv_time, sample))
        except Exception as error:  # noqa: BLE001 - surfaced via check_error()
            self._error = error
        finally:
            self._anchor_log.close()

    def check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"UWB ranging stream failed: {self._error}") from self._error
        if not self._closing and self._anchor_proc.poll() is not None:
            raise RuntimeError(
                f"UWB anchor process exited early with code {self._anchor_proc.returncode}"
            )

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def window(self, start_time_s: float, end_time_s: float) -> dict[str, np.ndarray]:
        """Return samples received in [start_time_s, end_time_s] as packed arrays.

        One row per (ranging round, node) measurement -- `mac_address` is
        the only thing that distinguishes nodes from each other when more
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
        stop_process(self._anchor_proc)
        self._thread.join(timeout=3)
        for node_proc, node_log in zip(self._node_procs, self._node_logs):
            stop_process(node_proc, node_log)

        if final_reset:
            print("Resetting UWB devices after stream close...")
            try:
                reset_devices(
                    self.python_exe,
                    [self.anchor_port, *self.node_ports],
                    self.log_dir / "final_device_reset_log.txt",
                )
            except RuntimeError as error:
                print(f"Warning: final UWB device reset failed: {error}")
