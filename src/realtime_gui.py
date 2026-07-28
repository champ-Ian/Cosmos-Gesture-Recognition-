#!/usr/bin/env python3
"""
Tkinter live display for `realtime_demo.py --gui`: a window showing the
current predicted gesture in large color-coded text, a confidence bar, and
a scrolling history of recent predictions.

Tkinter is not thread-safe -- every widget update has to happen on the main
thread's event loop. `realtime_demo.py` runs the actual sensor/prediction
loop on a background thread (it has its own blocking `time.sleep()` pacing
that shouldn't share a thread with `root.mainloop()`) and pushes each
prediction through a `queue.Queue`; `RealtimeGestureGUI.run()` polls that
queue via `root.after()` instead of touching it from the worker thread.
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import font as tkfont

BG = "#0f1117"
PANEL_BG = "#1a1e29"
TEXT_MUTED = "#6b7280"
TEXT_BRIGHT = "#f5f5f7"
NO_GESTURE_COLOR = "#3a3f4b"

# Distinct, readable-on-dark-background hues -- cycled per gesture label so
# the same gesture always gets the same color for the length of a session.
ACCENT_PALETTE = [
    "#4cc9f0", "#f72585", "#7209b7", "#4361ee", "#4895ef",
    "#f4a261", "#2a9d8f", "#e76f51", "#a7c957", "#ff6b6b",
    "#c77dff", "#38b000", "#ffb703", "#fb8500", "#06d6a0",
]


class RealtimeGestureGUI:
    """Owns the Tk window. Construct and call `.run()` on the main thread only."""

    def __init__(self, model_label: str, sensors: list[str], gesture_labels: list[str], no_gesture_label: str):
        self.no_gesture_label = no_gesture_label
        self.palette = {
            label: ACCENT_PALETTE[i % len(ACCENT_PALETTE)] for i, label in enumerate(sorted(gesture_labels))
        }

        self.root = tk.Tk()
        self.root.title("COSMOS Gesture Recognition -- Live Demo")
        self.root.configure(bg=BG)
        self.root.geometry("640x600")
        self.root.minsize(480, 480)

        header_font = tkfont.Font(family="Helvetica", size=12, weight="bold")
        big_font = tkfont.Font(family="Helvetica", size=44, weight="bold")
        sub_font = tkfont.Font(family="Helvetica", size=13)
        mono_font = tkfont.Font(family="Menlo", size=11)

        tk.Label(
            self.root, text=f"{model_label}   |   {'+'.join(sensors)}",
            font=header_font, fg=TEXT_MUTED, bg=BG, pady=14,
        ).pack(fill="x")

        self.pred_label = tk.Label(self.root, text="...", font=big_font, fg=TEXT_BRIGHT, bg=BG, pady=18)
        self.pred_label.pack(fill="x")

        self.conf_canvas = tk.Canvas(self.root, height=22, bg=PANEL_BG, highlightthickness=0)
        self.conf_canvas.pack(fill="x", padx=40, pady=(0, 6))
        self.conf_bar = self.conf_canvas.create_rectangle(0, 0, 0, 22, fill=NO_GESTURE_COLOR, width=0)

        self.conf_text = tk.Label(self.root, text=" ", font=sub_font, fg=TEXT_MUTED, bg=BG)
        self.conf_text.pack(fill="x", pady=(0, 16))

        history_frame = tk.Frame(self.root, bg=BG)
        history_frame.pack(fill="both", expand=True, padx=24, pady=(0, 12))
        tk.Label(history_frame, text="RECENT", font=header_font, fg=TEXT_MUTED, bg=BG, anchor="w").pack(fill="x")
        self.history_text = tk.Text(
            history_frame, height=12, bg=PANEL_BG, fg=TEXT_BRIGHT, font=mono_font,
            borderwidth=0, highlightthickness=0, state="disabled", wrap="none",
        )
        self.history_text.pack(fill="both", expand=True, pady=(6, 0))

        self.status_label = tk.Label(self.root, text="listening...", font=sub_font, fg=TEXT_MUTED, bg=BG, pady=10)
        self.status_label.pack(fill="x")

    def _color_for(self, label: str) -> str:
        if label == self.no_gesture_label:
            return NO_GESTURE_COLOR
        return self.palette.get(label, NO_GESTURE_COLOR)

    def _apply_prediction(self, event: dict) -> None:
        label = event["prediction"]
        confidence = event.get("confidence")
        color = self._color_for(label)

        self.pred_label.configure(text=label.upper().replace("_", " "), fg=color)

        width = self.conf_canvas.winfo_width() or 560
        fraction = 0.0 if confidence is None else max(0.0, min(1.0, confidence))
        self.conf_canvas.coords(self.conf_bar, 0, 0, width * fraction, 22)
        self.conf_canvas.itemconfig(self.conf_bar, fill=color)
        self.conf_text.configure(text="" if confidence is None else f"confidence: {confidence * 100:.0f}%")

        self.history_text.configure(state="normal")
        conf_suffix = "" if confidence is None else f"  ({confidence * 100:.0f}%)"
        self.history_text.insert("1.0", f"[{event['time_s']:6.1f}s]  {label}{conf_suffix}\n")
        # Cap history so the widget doesn't grow unbounded over a long session.
        line_count = int(self.history_text.index("end-1c").split(".")[0])
        if line_count > 300:
            self.history_text.delete("300.0", "end")
        self.history_text.configure(state="disabled")

    def _poll(self, update_queue: "queue.Queue", stop_event: threading.Event) -> None:
        try:
            while True:
                event = update_queue.get_nowait()
                if event.get("_status"):
                    self.status_label.configure(text=event["_status"])
                    continue
                if event.get("_stop"):
                    self.root.destroy()
                    return
                self._apply_prediction(event)
        except queue.Empty:
            pass
        self.root.after(50, self._poll, update_queue, stop_event)

    def run(self, update_queue: "queue.Queue", stop_event: threading.Event) -> None:
        """Blocks in the Tk event loop until the window is closed (which sets
        `stop_event` so the background prediction thread knows to stop too) or
        the worker thread signals it's done by pushing `{"_stop": True}`."""
        self.root.protocol("WM_DELETE_WINDOW", lambda: (stop_event.set(), self.root.destroy()))
        self.root.after(50, self._poll, update_queue, stop_event)
        self.root.mainloop()
