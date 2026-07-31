#!/usr/bin/env python3
"""
Standalone diagnostic: continuously shows "GESTURE" the instant any sensor's
activity crosses --trigger-z-threshold std-devs above its calibrated idle
baseline, and "no gesture" otherwise -- updating live, independent of the
slower window-seconds capture+classify cycle realtime_demo.py uses.

This directly answers "when does it actually start recording": it shows the
literal live trigger condition (the same activity_triggered() check
realtime_demo.py's trigger mode uses to decide when to start a capture), with
no model, no classification, and no window-length delay in the way -- pure
timing verification of the trigger mechanism by itself. Move/gesture and
watch the console or --gui window -- the state should flip to GESTURE the
moment you start moving, and back to "no gesture" once you stop.

Run from the repo root:
    python src/trigger_monitor.py \\
        --mmwave-port /dev/cu.usbserial-XXXX --imu-port /dev/cu.usbserial-YYYY \\
        --uwb-anchor-port /dev/cu.usbmodemAAAA --uwb-node-port /dev/cu.usbmodemBBBB \\
        --uwb-group-id 1 --duration 60 --gui
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

from realtime_demo import (
    DEFAULT_MMWAVE_CFG,
    activity_triggered,
    calibrate_idle_baseline,
    close_streams,
    open_streams,
)
from sensors.common import REPO_DIR

BG = "#0a0c12"
PANEL_BG = "#161a24"
TEXT_DIM = "#9ca3af"
TEXT_BRIGHT = "#f5f5f7"
NO_GESTURE_COLOR = "#3a3f4b"
LIVE_GREEN = "#2ecc71"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=60.0, help="Seconds to monitor.")
    parser.add_argument("--calibration-seconds", type=float, default=2.0, help="Idle period at startup -- stay still.")
    parser.add_argument("--trigger-window-seconds", type=float, default=0.4)
    parser.add_argument("--trigger-check-interval", type=float, default=0.05, help="Faster than realtime_demo.py's "
        "default -- this script isn't doing any capture/classify work, so it can poll quickly.")
    parser.add_argument("--trigger-z-threshold", type=float, default=3.0, help="Same meaning as realtime_demo.py's "
        "--trigger-z-threshold -- match whatever value you're actually using there to test the same sensitivity.")
    parser.add_argument("--gui", action="store_true", help="Show a big color-coded GESTURE/no gesture window "
        "instead of terminal-only output.")

    mmwave_group = parser.add_argument_group("mmWave radar (TI xWRL6432)")
    mmwave_group.add_argument("--mmwave-port")
    mmwave_group.add_argument("--mmwave-cfg", type=Path, default=DEFAULT_MMWAVE_CFG)
    mmwave_group.add_argument("--mmwave-baud", type=int, default=115200)
    mmwave_group.add_argument("--no-mmwave-warm-reset", action="store_true")

    imu_group = parser.add_argument_group("IMU (ESP32 Core2)")
    imu_group.add_argument("--imu-port")
    imu_group.add_argument("--imu-baud", type=int, default=115200)

    uwb_group = parser.add_argument_group("UWB (Qorvo DWM3001CDK FiRa TWR)")
    uwb_group.add_argument("--uwb-anchor-port")
    uwb_group.add_argument("--uwb-node-port", action="append")
    uwb_group.add_argument("--uwb-group-id", type=int)
    uwb_group.add_argument("--uwb-preamble-code", type=int, default=10)
    uwb_group.add_argument("--uwb-channel", type=int, choices=[5, 9], default=9)
    uwb_group.add_argument("--uwb-fps", type=float, default=50.0)
    uwb_group.add_argument("--uwb-slot-span", type=int, default=2400)
    uwb_group.add_argument("--uwb-slots-per-rr", type=int, default=None)
    uwb_group.add_argument("--uwb-skip-device-reset", action="store_true")

    rfid_group = parser.add_argument_group("RFID (RFID_Lab reader, TCP -- not serial)")
    rfid_group.add_argument("--rfid-host", default="192.168.137.1")
    rfid_group.add_argument("--rfid-tcp-port", type=int, default=9055)
    rfid_group.add_argument("--rfid", action="store_true", help="Enable the RFID reader.")

    return parser.parse_args()


class TriggerMonitorGUI:
    """Minimal Tkinter window: one big color-coded GESTURE / no gesture label
    plus a status line. Deliberately much simpler than realtime_gui.py's
    RealtimeGestureGUI -- there's only two states here, not 15 gesture labels
    to color/tally."""

    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import font as tkfont

        self.tk = tk
        self.root = tk.Tk()
        self.root.title("Trigger Monitor -- GESTURE vs no gesture")
        self.root.configure(bg=BG)
        self.root.geometry("560x360")
        self.root.minsize(420, 280)

        big_font = tkfont.Font(family="Helvetica", size=42, weight="bold")
        status_font = tkfont.Font(family="Helvetica", size=12)
        timer_font = tkfont.Font(family="Helvetica", size=11)

        self.state_label = tk.Label(self.root, text="calibrating...", font=big_font, bg=BG, fg=TEXT_DIM)
        self.state_label.pack(expand=True, fill="both", padx=20, pady=(30, 10))

        self.status_label = tk.Label(self.root, text="", font=status_font, bg=BG, fg=TEXT_DIM)
        self.status_label.pack(pady=(0, 6))

        self.timer_label = tk.Label(self.root, text="", font=timer_font, bg=BG, fg=TEXT_DIM)
        self.timer_label.pack(pady=(0, 20))

    def _set_state(self, state: str) -> None:
        if state == "GESTURE":
            self.state_label.configure(text="GESTURE", fg=LIVE_GREEN)
        elif state == "no gesture":
            self.state_label.configure(text="no gesture", fg=TEXT_BRIGHT)
        else:
            self.state_label.configure(text=state, fg=TEXT_DIM)

    def _poll(self, update_queue: "queue.Queue", stop_event: threading.Event) -> None:
        try:
            while True:
                event = update_queue.get_nowait()
                if event.get("_status"):
                    self.status_label.configure(text=event["_status"])
                if "state" in event:
                    self._set_state(event["state"])
                if "time_s" in event:
                    self.timer_label.configure(text=f"{event['time_s']:.2f}s")
                if event.get("_stop"):
                    self._close()
                    return
        except queue.Empty:
            pass
        self.root.after(50, self._poll, update_queue, stop_event)

    def _close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self, update_queue: "queue.Queue", stop_event: threading.Event) -> None:
        self.root.protocol("WM_DELETE_WINDOW", lambda: (stop_event.set(), self._close()))
        self.root.after(50, self._poll, update_queue, stop_event)
        self.root.mainloop()


def run_monitor_loop(
    streams: dict,
    args: argparse.Namespace,
    stop_event: threading.Event,
    gui_queue: "queue.Queue | None",
) -> None:
    baseline = calibrate_idle_baseline(streams, args)
    baseline_text = "Baseline: " + ", ".join(f"{sensor}={mean:.2f}±{std:.2f}" for sensor, (mean, std) in baseline.items())
    print(baseline_text)
    print(f"Live trigger state (z >= {args.trigger_z_threshold}) -- move to test, Ctrl+C to stop:")
    if gui_queue is not None:
        gui_queue.put({"_status": f"z >= {args.trigger_z_threshold} -- move to test"})

    start = time.monotonic()
    end_time = start + args.duration
    state = None
    try:
        while time.monotonic() < end_time and not stop_event.is_set():
            now = time.monotonic()
            for stream in streams.values():
                stream.check_error()
            triggered = activity_triggered(streams, baseline, now, args)
            new_state = "GESTURE" if triggered else "no gesture"
            if new_state != state:
                state = new_state
                elapsed = now - start
                print(f"[{elapsed:6.2f}s] {state}", flush=True)
                if gui_queue is not None:
                    gui_queue.put({"state": state, "time_s": elapsed})
            elif gui_queue is not None:
                gui_queue.put({"time_s": now - start})
            time.sleep(args.trigger_check_interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if gui_queue is not None:
            gui_queue.put({"_stop": True})


def main() -> int:
    args = parse_args()
    required_sensors = []
    if args.mmwave_port:
        required_sensors.append("mmwave")
    if args.imu_port:
        required_sensors.append("imu")
    if args.uwb_anchor_port and args.uwb_node_port:
        required_sensors.append("uwb")
    if args.rfid:
        required_sensors.append("rfid")
    if not required_sensors:
        raise SystemExit(
            "Pass at least one sensor's port: --mmwave-port / --imu-port / "
            "(--uwb-anchor-port + --uwb-node-port) / --rfid."
        )

    session_dir = REPO_DIR / "sessions" / "trigger_monitor_tmp"
    session_dir.mkdir(parents=True, exist_ok=True)

    streams = open_streams(args, required_sensors, session_dir)
    stop_event = threading.Event()
    try:
        if args.gui:
            gui_queue: "queue.Queue" = queue.Queue()
            worker = threading.Thread(target=run_monitor_loop, args=(streams, args, stop_event, gui_queue), daemon=True)
            worker.start()
            gui = TriggerMonitorGUI()
            gui.run(gui_queue, stop_event)
            stop_event.set()
            worker.join(timeout=2)
        else:
            run_monitor_loop(streams, args, stop_event, None)
    finally:
        close_streams(streams)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
