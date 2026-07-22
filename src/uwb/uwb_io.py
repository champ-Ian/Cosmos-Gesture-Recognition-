#!/usr/bin/env python3
"""
Qorvo DWM3001CDK FiRa two-way-ranging (TWR) helpers.

Adapted from UCLA COSMOS `UWB_lab` (uwb_lab_common.py). The real UWB hardware
distributed for this project is a pair of Qorvo DWM3001CDK boards running the
FiRa UCI ranging demo, not a single tag that prints JSON lines: one board is
the "controller" (initiator), the other the "controlee" (responder), each on
its own serial port. Ranging is driven by launching the vendored
`uwb-qorvo-tools` CLI (`run_fira_twr.py`) as a subprocess per board; the
controller's stdout prints text lines such as:

    sequence n: 12
    ranging interval: 20.00 ms
    status: Ok (0x0)
    distance: 87.3 cm

`RangeLogParser` turns that text stream into structured samples the same way
`UWB_lab/uwb_lab_common.py` does for offline log files, just fed line-by-line
from a live subprocess instead of a saved log.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

UWB_DIR = Path(__file__).resolve().parent
QORVO_ROOT = UWB_DIR / "uwb-qorvo-tools"
RUN_FIRA_TWR = QORVO_ROOT / "scripts" / "fira" / "run_fira_twr" / "run_fira_twr.py"
RESET_DEVICE = QORVO_ROOT / "scripts" / "device" / "reset_device" / "reset_device.py"


def process_env() -> dict:
    """Environment for the vendored Qorvo CLI: its `uci`/`uqt_utils` libs on PYTHONPATH."""
    env = os.environ.copy()
    repo_paths = [
        str(QORVO_ROOT),
        str(QORVO_ROOT / "lib" / "uwb-uci"),
        str(QORVO_ROOT / "lib" / "uqt-utils"),
    ]
    old_pythonpath = env.get("PYTHONPATH")
    if old_pythonpath:
        repo_paths.append(old_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(repo_paths)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def compute_ranging_span_ms(fps: float, ranging_span: int | None) -> int:
    if ranging_span is not None:
        return int(ranging_span)
    return max(1, int(round(1000.0 / float(fps))))


def validate_timing(slot_span: int, slots_per_rr: int, ranging_span: int) -> None:
    slot_ms = float(slot_span) / 1200.0
    minimum_ms = slot_ms * int(slots_per_rr)
    if ranging_span < minimum_ms:
        raise ValueError(
            f"ranging span {ranging_span} ms is shorter than "
            f"{slots_per_rr} slots * {slot_ms:.3f} ms = {minimum_ms:.3f} ms"
        )


def twr_command(
    python_exe: str,
    port: str,
    preamble_code: int,
    duration_s: float,
    slot_span: int,
    slots_per_rr: int,
    ranging_span: int,
    channel: int = 9,
    controlee: bool = False,
    stats: bool = True,
    node_mode: str = "unicast",
    n_controlees: int = 1,
    mac: str | None = None,
    dest_mac: str | None = None,
) -> list[str]:
    """Build a `run_fira_twr.py` invocation.

    `node_mode="onetomany"` plus `n_controlees` > 1 is FiRa's multi-node
    ranging mode: one controller (here, the worn tag) ranges against several
    controlees (here, the fixed anchors) within a single MAC round, using
    `mac`/`dest_mac` to address each side. This mode is exercised by the
    vendored CLI's own `--node`/`--n_controlees`/`--mac`/`--dest-mac` flags,
    but is **not** what `UWB_lab`'s single controller/controlee lab exercises
    -- treat it as unverified against real DWM3001CDK hardware until you've
    smoke-tested it (short `--stats` run) yourself.
    """
    cmd = [
        python_exe,
        "-u",
        str(RUN_FIRA_TWR),
        "-p",
        str(port),
        "--channel",
        str(channel),
        "--preamble-idx",
        str(preamble_code),
        "--aoa-report",
        "all-disabled",
        "--slot-span",
        str(slot_span),
        "--slots-per-rr",
        str(slots_per_rr),
        "--ranging-span",
        str(ranging_span),
        "-t",
        str(int(duration_s)),
    ]
    if controlee:
        cmd.append("--controlee")
    if stats:
        cmd.append("--stats")
    if node_mode != "unicast":
        cmd.extend(["--node", node_mode])
    if n_controlees != 1:
        cmd.extend(["--n_controlees", str(n_controlees)])
    if mac is not None:
        cmd.extend(["--mac", mac])
    if dest_mac is not None:
        cmd.extend(["--dest-mac", dest_mac])
    return cmd


def stop_process(proc: subprocess.Popen | None, log_file=None) -> None:
    try:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
    finally:
        if log_file:
            log_file.close()


def run_device_command(cmd: list[str], log_path: Path) -> str:
    completed = subprocess.run(
        cmd,
        cwd=QORVO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=process_env(),
    )
    with open(log_path, "a") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.write(completed.stdout)
        if not completed.stdout.endswith("\n"):
            log.write("\n")
        log.write(f"return_code={completed.returncode}\n\n")
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-20:])
        raise RuntimeError(
            "device command failed\n"
            f"command: {' '.join(cmd)}\n"
            f"return_code: {completed.returncode}\n"
            f"last output:\n{tail}"
        )
    return completed.stdout


def reset_devices(python_exe: str, ports: list[str], log_path: Path) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    for port in ports:
        run_device_command([python_exe, str(RESET_DEVICE), "-p", str(port)], log_path)
        time.sleep(0.75)
    time.sleep(1.0)


@dataclass
class RangeSample:
    sequence: int | None
    interval_ms: float | None
    status: str
    status_code: str
    distance_cm: float
    mac_address: str


class RangeLogParser:
    """Incrementally parses `run_fira_twr.py --stats` controller stdout lines.

    Each ranging round prints one `sequence n:` / `ranging interval:` line
    followed by one `# Measurement <i>:` block per controlee (one block for a
    plain unicast controller<->controlee pair, several blocks in
    `--node onetomany` mode), each with its own `status:`, `mac address:`,
    and `distance:` lines in that order (see
    `uwb-qorvo-tools/lib/uwb-uci/uci/qorvo_msg.py`'s `RangingTwrData.__str__`).
    `mac_address` is what tells samples from different anchors apart in
    one-to-many mode; it's parsed as the raw hex string printed by the
    device and not otherwise validated.
    """

    sequence_re = re.compile(r"sequence n:\s*(\d+)")
    interval_re = re.compile(r"ranging interval:\s*([0-9.]+)\s*ms")
    status_re = re.compile(r"status:\s*([A-Za-z0-9_]+)\s*\((0x[0-9a-fA-F]+)\)")
    mac_re = re.compile(r"mac address:\s*([0-9a-fA-F:]+)\s*hex")
    distance_re = re.compile(r"distance:\s*([-+]?[0-9]*\.?[0-9]+)\s*cm")

    def __init__(self) -> None:
        self.sequence: int | None = None
        self.interval_ms: float | None = None
        self.status: str | None = None
        self.status_code: str | None = None
        self.mac_address: str | None = None

    def feed(self, line: str) -> RangeSample | None:
        match = self.sequence_re.search(line)
        if match:
            self.sequence = int(match.group(1))
            self.status = None
            self.status_code = None
            self.mac_address = None
            return None

        match = self.interval_re.search(line)
        if match:
            self.interval_ms = float(match.group(1))
            return None

        match = self.status_re.search(line)
        if match:
            self.status = match.group(1)
            self.status_code = match.group(2)
            return None

        match = self.mac_re.search(line)
        if match:
            self.mac_address = match.group(1)
            return None

        match = self.distance_re.search(line)
        if match:
            sample = RangeSample(
                sequence=self.sequence,
                interval_ms=self.interval_ms,
                status=self.status or "unknown",
                status_code=self.status_code or "",
                distance_cm=float(match.group(1)),
                mac_address=self.mac_address or "",
            )
            self.status = None
            self.status_code = None
            self.mac_address = None
            return sample

        return None
