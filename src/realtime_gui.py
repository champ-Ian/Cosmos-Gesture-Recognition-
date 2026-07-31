#!/usr/bin/env python3
"""
Tkinter live display for `realtime_demo.py --gui`.

Purely a display layer -- it only ever reads events already computed by
`capture_and_classify()` (prediction, confidence, vote breakdown, etc.) and
renders them. It never touches feature extraction, thresholds, or the model,
so nothing here can change what gets predicted, only how it's shown.

Tkinter is not thread-safe -- every widget update has to happen on the main
thread's event loop. `realtime_demo.py` runs the actual sensor/prediction
loop on a background thread and pushes each prediction through a
`queue.Queue`; `RealtimeGestureGUI.run()` polls that queue via `root.after()`
instead of touching it from the worker thread.
"""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import font as tkfont

BG = "#0a0c12"
PANEL_BG = "#161a24"
TEXT_MUTED = "#6b7280"
TEXT_DIM = "#9ca3af"
TEXT_BRIGHT = "#f5f5f7"
NO_GESTURE_COLOR = "#3a3f4b"
UNKNOWN_GESTURE_COLOR = "#f4a261"  # amber -- deliberately distinct from NO_GESTURE_COLOR: movement WAS
# confirmed, the classifier just isn't sure which gesture, which is a different situation than no
# movement having happened at all (see realtime_demo.py's NO_GESTURE_LABEL vs UNKNOWN_GESTURE_LABEL).
LIVE_GREEN = "#2ecc71"

# Distinct, readable-on-dark-background hues -- cycled per gesture label so
# the same gesture always gets the same color for the length of a session.
ACCENT_PALETTE = [
    "#4cc9f0", "#f72585", "#7209b7", "#4361ee", "#4895ef",
    "#f4a261", "#2a9d8f", "#e76f51", "#a7c957", "#ff6b6b",
    "#c77dff", "#38b000", "#ffb703", "#fb8500", "#06d6a0",
]

TALLY_ROWS = 6


def _lerp_color(hex_a: str, hex_b: str, t: float) -> str:
    """Linear-interpolate two '#rrggbb' colors at t in [0, 1] (0=a, 1=b)."""
    a = hex_a.lstrip("#")
    b = hex_b.lstrip("#")
    ar, ag, ab = int(a[0:2], 16), int(a[2:4], 16), int(a[4:6], 16)
    br, bg, bb = int(b[0:2], 16), int(b[2:4], 16), int(b[4:6], 16)
    r = round(ar + (br - ar) * t)
    g = round(ag + (bg - ag) * t)
    bch = round(ab + (bb - ab) * t)
    return f"#{r:02x}{g:02x}{bch:02x}"


def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _rounded_rect(canvas: tk.Canvas, x1: float, y1: float, x2: float, y2: float, radius: float, **kwargs):
    """A flat, solid-fill rounded rectangle -- unlike stacked gradient ovals,
    a single smoothed polygon renders as one clean shape with no banding."""
    points = [
        x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
        x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
        x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1, x1 + radius, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


class RealtimeGestureGUI:
    """Owns the Tk window. Construct and call `.run()` on the main thread only."""

    def __init__(
        self,
        model_label: str,
        sensors: list[str],
        gesture_labels: list[str],
        no_gesture_label: str,
        duration_s: float | None = None,
        unknown_gesture_label: str | None = None,
    ):
        self.no_gesture_label = no_gesture_label
        self.unknown_gesture_label = unknown_gesture_label
        self.total_duration_s = duration_s
        self.palette = {
            label: ACCENT_PALETTE[i % len(ACCENT_PALETTE)] for i, label in enumerate(sorted(gesture_labels))
        }
        self.counts: dict[str, int] = {}
        self._closed = False
        self._blink_on = True

        self.root = tk.Tk()
        self.root.title("COSMOS Gesture Recognition -- Live Demo")
        self.root.configure(bg=BG)
        self.root.geometry("700x760")
        self.root.minsize(560, 640)

        title_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
        header_font = tkfont.Font(family="Helvetica", size=11, weight="bold")
        sub_font = tkfont.Font(family="Helvetica", size=12)
        dim_font = tkfont.Font(family="Helvetica", size=10)
        mono_font = tkfont.Font(family="Menlo", size=11)
        self.card_label_font = tkfont.Font(family="Helvetica", size=11, weight="bold")
        self.big_font = tkfont.Font(family="Helvetica", size=36, weight="bold")

        # ---- Header: title + live pulse dot -----------------------------
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=24, pady=(18, 4))
        tk.Label(header, text="COSMOS", font=title_font, fg=TEXT_BRIGHT, bg=BG).pack(side="left")
        tk.Label(header, text=" · LIVE GESTURE RECOGNITION", font=header_font, fg=TEXT_MUTED, bg=BG).pack(side="left")
        self.live_canvas = tk.Canvas(header, width=14, height=14, bg=BG, highlightthickness=0)
        self.live_canvas.pack(side="right", pady=2)
        self.live_dot = self.live_canvas.create_oval(2, 2, 12, 12, fill=LIVE_GREEN, outline="")
        tk.Label(header, text="LIVE", font=dim_font, fg=TEXT_MUTED, bg=BG).pack(side="right", padx=(0, 6))

        tk.Label(
            self.root, text=f"{model_label}   |   {'+'.join(sensors)}", font=dim_font, fg=TEXT_MUTED, bg=BG, anchor="w"
        ).pack(fill="x", padx=24, pady=(0, 10))

        # ---- Session timer -------------------------------------------------
        strip = tk.Frame(self.root, bg=BG)
        strip.pack(fill="x", padx=24, pady=(0, 6))
        self.timer_label = tk.Label(strip, text="", font=dim_font, fg=TEXT_MUTED, bg=BG)
        self.timer_label.pack(side="right")

        self.timer_canvas = tk.Canvas(self.root, height=4, bg=PANEL_BG, highlightthickness=0)
        self.timer_canvas.pack(fill="x", padx=24, pady=(0, 14))
        self.timer_bar = self.timer_canvas.create_rectangle(0, 0, 0, 4, fill=LIVE_GREEN, width=0)

        # ---- Main stage: glow + emoji + gesture name ---------------------
        self.stage = tk.Canvas(self.root, height=180, bg=BG, highlightthickness=0)
        self.stage.pack(fill="x", padx=40)
        self._stage_bounds = (0, 0, 620, 180)  # updated on <Configure>, see below
        self.stage.bind("<Configure>", self._on_stage_resize)

        self.subtext_label = tk.Label(self.root, text=" ", font=dim_font, fg=TEXT_MUTED, bg=BG)
        self.subtext_label.pack(fill="x", pady=(0, 4))

        # ---- Confidence bar ------------------------------------------------
        self.conf_canvas = tk.Canvas(self.root, height=18, bg=PANEL_BG, highlightthickness=0)
        self.conf_canvas.pack(fill="x", padx=40, pady=(4, 4))
        self.conf_bar = self.conf_canvas.create_rectangle(0, 0, 0, 18, fill=NO_GESTURE_COLOR, width=0)

        self.conf_text = tk.Label(self.root, text=" ", font=sub_font, fg=TEXT_MUTED, bg=BG)
        self.conf_text.pack(fill="x", pady=(0, 14))

        # ---- Live tally panel ----------------------------------------------
        tally_frame = tk.Frame(self.root, bg=BG)
        tally_frame.pack(fill="x", padx=24, pady=(0, 10))
        tk.Label(tally_frame, text="SESSION TALLY", font=header_font, fg=TEXT_MUTED, bg=BG, anchor="w").pack(fill="x")
        self._tally_rows = []
        for _ in range(TALLY_ROWS):
            row = tk.Frame(tally_frame, bg=BG)
            row.pack(fill="x", pady=1)
            name = tk.Label(row, text="", font=dim_font, fg=TEXT_DIM, bg=BG, width=16, anchor="w")
            name.pack(side="left")
            bar_canvas = tk.Canvas(row, height=10, bg=PANEL_BG, highlightthickness=0)
            bar_canvas.pack(side="left", fill="x", expand=True, padx=(4, 8))
            bar_item = bar_canvas.create_rectangle(0, 0, 0, 10, fill=NO_GESTURE_COLOR, width=0)
            count_label = tk.Label(row, text="", font=dim_font, fg=TEXT_DIM, bg=BG, width=3, anchor="e")
            count_label.pack(side="left")
            self._tally_rows.append((row, name, bar_canvas, bar_item, count_label))

        # ---- Recent history --------------------------------------------------
        history_frame = tk.Frame(self.root, bg=BG)
        history_frame.pack(fill="both", expand=True, padx=24, pady=(4, 10))
        tk.Label(history_frame, text="RECENT", font=header_font, fg=TEXT_MUTED, bg=BG, anchor="w").pack(fill="x")
        self.history_text = tk.Text(
            history_frame, height=8, bg=PANEL_BG, fg=TEXT_BRIGHT, font=mono_font,
            borderwidth=0, highlightthickness=0, state="disabled", wrap="none",
        )
        self.history_text.pack(fill="both", expand=True, pady=(6, 0))
        self.history_text.tag_configure("dim", foreground=TEXT_MUTED)

        self.status_label = tk.Label(self.root, text="listening...", font=sub_font, fg=TEXT_MUTED, bg=BG, pady=10)
        self.status_label.pack(fill="x")

        self._draw_stage(no_gesture_label, None)

    # ------------------------------------------------------------------ #

    def _color_for(self, label: str) -> str:
        if label == self.no_gesture_label:
            return NO_GESTURE_COLOR
        if label == self.unknown_gesture_label:
            return UNKNOWN_GESTURE_COLOR
        return self.palette.get(label, NO_GESTURE_COLOR)

    def _on_stage_resize(self, event) -> None:
        self._stage_bounds = (0, 0, event.width, event.height)
        self._draw_stage(self._last_label, self._last_confidence)

    def _draw_stage(self, label: str, confidence: float | None) -> None:
        """A single flat rounded card: dark-tinted fill in the gesture's accent
        color, a bright border in that same color, and the gesture name centered
        in plain bold text. No gradients (Tk canvas ovals band badly with no real
        alpha blending) and no emoji (Tk's emoji glyph metrics don't center
        reliably) -- just one clean shape that reads well at a glance."""
        self._last_label, self._last_confidence = label, confidence
        color = self._color_for(label)
        x1, y1, x2, y2 = self._stage_bounds
        if x2 - x1 < 10:
            return
        pad = 8
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        self.stage.delete("all")
        card_fill = _lerp_color(color, "#000000", 0.6)
        _rounded_rect(self.stage, x1 + pad, y1 + pad, x2 - pad, y2 - pad, radius=20, fill=card_fill, outline=color, width=2)
        self.stage.create_text(cx, cy - 24, text="GESTURE", font=self.card_label_font, fill=TEXT_DIM)
        self.stage.create_text(cx, cy + 16, text=label.upper().replace("_", " "), font=self.big_font, fill=TEXT_BRIGHT)

    def _update_tally(self) -> None:
        ranked = sorted(self.counts.items(), key=lambda kv: -kv[1])[:TALLY_ROWS]
        max_count = ranked[0][1] if ranked else 1
        width = 300
        for i, (_, name_label, bar_canvas, bar_item, count_label) in enumerate(self._tally_rows):
            if i < len(ranked):
                label, count = ranked[i]
                name_label.configure(text=label.replace("_", " "))
                w = bar_canvas.winfo_width() or width
                bar_canvas.coords(bar_item, 0, 0, w * (count / max_count), 10)
                bar_canvas.itemconfig(bar_item, fill=self._color_for(label))
                count_label.configure(text=str(count))
            else:
                name_label.configure(text="")
                bar_canvas.coords(bar_item, 0, 0, 0, 10)
                count_label.configure(text="")

    def _apply_prediction(self, event: dict) -> None:
        label = event["prediction"]
        confidence = event.get("confidence")
        color = self._color_for(label)

        self._draw_stage(label, confidence)

        width = self.conf_canvas.winfo_width() or 560
        fraction = 0.0 if confidence is None else max(0.0, min(1.0, confidence))
        self.conf_canvas.coords(self.conf_bar, 0, 0, width * fraction, 18)
        self.conf_canvas.itemconfig(self.conf_bar, fill=color)
        vote_window = event.get("vote_window") or 1
        vote_fraction = event.get("vote_fraction")
        conf_bits = []
        if confidence is not None:
            conf_bits.append(f"confidence {confidence * 100:.0f}%")
        if vote_window > 1 and vote_fraction is not None:
            conf_bits.append(f"vote agreement {vote_fraction * 100:.0f}% (window={vote_window})")
        self.conf_text.configure(text="   ·   ".join(conf_bits))

        # If this got gated to no_gesture for being below --confidence-threshold,
        # show what the raw (ungated) guess actually was -- purely informational,
        # doesn't change what got logged/voted on.
        if event.get("below_threshold") and event.get("raw_prediction"):
            raw_conf = event.get("raw_confidence")
            raw_conf_text = f" ({raw_conf * 100:.0f}%)" if raw_conf is not None else ""
            self.subtext_label.configure(text=f"raw guess: {event['raw_prediction']}{raw_conf_text} -- below threshold")
        else:
            self.subtext_label.configure(text=" ")

        if label != self.no_gesture_label:
            self.counts[label] = self.counts.get(label, 0) + 1
            self._update_tally()

        self.history_text.configure(state="normal")
        conf_suffix = "" if confidence is None else f"  ({confidence * 100:.0f}%)"
        self.history_text.insert("1.0", f"[{event['time_s']:6.1f}s]  {label}{conf_suffix}\n")
        line_count = int(self.history_text.index("end-1c").split(".")[0])
        if line_count > 300:
            self.history_text.delete("300.0", "end")
        self.history_text.configure(state="disabled")

    def _tick_live_dot(self) -> None:
        if self._closed:
            return
        self._blink_on = not self._blink_on
        self.live_canvas.itemconfig(self.live_dot, fill=LIVE_GREEN if self._blink_on else PANEL_BG)
        self.root.after(700, self._tick_live_dot)

    def _tick_timer(self, session_start: float) -> None:
        if self._closed:
            return
        elapsed = time.monotonic() - session_start
        width = self.timer_canvas.winfo_width() or 620
        if self.total_duration_s:
            frac = max(0.0, min(1.0, elapsed / self.total_duration_s))
            self.timer_label.configure(text=f"{_fmt_time(elapsed)} / {_fmt_time(self.total_duration_s)}")
        else:
            frac = 0.0
            self.timer_label.configure(text=_fmt_time(elapsed))
        self.timer_canvas.coords(self.timer_bar, 0, 0, width * frac, 4)
        self.root.after(200, self._tick_timer, session_start)

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.root.destroy()

    def _poll(self, update_queue: "queue.Queue", stop_event: threading.Event) -> None:
        try:
            while True:
                event = update_queue.get_nowait()
                if event.get("_status"):
                    self.status_label.configure(text=event["_status"])
                    continue
                if event.get("_stop"):
                    self._close()
                    return
                self._apply_prediction(event)
        except queue.Empty:
            pass
        self.root.after(50, self._poll, update_queue, stop_event)

    def run(self, update_queue: "queue.Queue", stop_event: threading.Event) -> None:
        """Blocks in the Tk event loop until the window is closed (which sets
        `stop_event` so the background prediction thread knows to stop too) or
        the worker thread signals it's done by pushing `{"_stop": True}`."""
        self.root.protocol("WM_DELETE_WINDOW", lambda: (stop_event.set(), self._close()))
        self.root.after(50, self._poll, update_queue, stop_event)
        self.root.after(700, self._tick_live_dot)
        self.root.after(200, self._tick_timer, time.monotonic())
        self.root.mainloop()
