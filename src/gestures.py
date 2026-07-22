#!/usr/bin/env python3
"""
Canonical gesture registry for the COSMOS gesture-recognition final project.

Keeping this list in one place means every script (collector, combiner,
future training/eval code) uses the same label spelling. `--gesture` on
`collect.py` accepts these canonical (snake_case) names.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GestureSpec:
    name: str
    display_name: str
    instruction: str
    suggested_sensors: tuple[str, ...]
    # "discrete": one clean instance per trial, clear start/end (collect.py
    #   prompts trial-by-trial, --duration each).
    # "periodic": repeated cycles with no natural single boundary --
    #   collect.py records one long continuous take instead, and
    #   extract_features.py's cut step segments it into fixed-length windows
    #   afterward. Grouping follows the final-project implementation hints
    #   (Shanmu Wang): periodic = clapping, boxing, palm up-down, soli.
    group: str = "discrete"


# Suggested sensor combination per gesture, based on the group's sensing plan:
# mmwave = radar range/point-cloud, imu = ESP32 Core2 IMU, uwb = the 3 UWB
# ranging modules, rfid = RFID reader + tag/copper-wire near-field sensing.
GESTURES: dict[str, GestureSpec] = {
    spec.name: spec
    for spec in (
        GestureSpec(
            "pull",
            "Pull",
            "Extend your arm forward, then pull your hand straight back toward your body.",
            ("mmwave", "uwb"),
        ),
        GestureSpec(
            "push",
            "Push",
            "Push your hand straight forward, away from your body and toward the sensors.",
            ("mmwave", "uwb"),
        ),
        GestureSpec(
            "clockwise",
            "Clockwise",
            "Trace a clockwise circle in the air in front of the radar.",
            ("mmwave", "imu"),
        ),
        GestureSpec(
            "anti_clockwise",
            "Anti-clockwise",
            "Trace a counter-clockwise circle in the air in front of the radar.",
            ("mmwave", "imu"),
        ),
        GestureSpec(
            "right",
            "Right",
            "Sweep your hand/arm from left to right in front of the sensors.",
            ("mmwave", "uwb"),
        ),
        GestureSpec(
            "left",
            "Left",
            "Sweep your hand/arm from right to left in front of the sensors.",
            ("mmwave", "uwb"),
        ),
        GestureSpec(
            "bye_bye",
            "Bye-Bye",
            "Wave goodbye by rotating your wrist back and forth. Strap the IMU to the back of your hand.",
            ("imu",),
        ),
        GestureSpec(
            "one_arm_boxing",
            "One-Arm Boxing",
            "Throw repeated punches forward with one arm, retracting between each. IMU on the "
            "punching wrist, UWB tag tracks distance change to the anchor.",
            ("imu", "uwb"),
            group="periodic",
        ),
        GestureSpec(
            "clapping",
            "Clapping",
            "Clap your hands together repeatedly in front of the sensors.",
            ("mmwave", "imu"),
            group="periodic",
        ),
        GestureSpec(
            "two_arm_boxing",
            "Two-Arm Boxing",
            "Throw alternating punches with both arms; UWB distance-to-anchor should change with each punch.",
            ("imu", "uwb"),
            group="periodic",
        ),
        GestureSpec(
            "t_arm",
            "T-Arm",
            "Hold both arms straight out to the sides, forming a T, and hold the pose.",
            ("uwb", "imu"),
        ),
        GestureSpec(
            "raise_arms",
            "Raise Arms",
            "Raise both arms straight overhead, then lower them.",
            ("uwb", "imu"),
        ),
        GestureSpec(
            "soli",
            "Soli",
            "Move your fingers/hand near the RFID copper-wire antenna without touching it, "
            "repeatedly (micro-gesture, like Google Soli).",
            ("rfid",),
            group="periodic",
        ),
        GestureSpec(
            "fist_open",
            "Making Fist and Open",
            "Hold your hand near the RFID reader and repeatedly close into a fist, then open it.",
            ("rfid",),
        ),
        GestureSpec(
            "palm_up_down",
            "Palm Up-Down",
            "With the IMU strapped to the back of your hand, rotate your palm face-up then face-down, repeatedly.",
            ("imu",),
            group="periodic",
        ),
    )
}

DEFAULT_GESTURES: tuple[str, ...] = tuple(GESTURES.keys())


def normalize_gestures(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_GESTURES)

    gestures: list[str] = []
    for value in values:
        for part in value.split(","):
            label = part.strip().lower().replace("-", "_").replace(" ", "_")
            if not label:
                continue
            if label not in GESTURES:
                valid = ", ".join(sorted(GESTURES))
                raise SystemExit(f"Unknown gesture '{label}'. Valid gestures: {valid}")
            gestures.append(label)
    if not gestures:
        raise SystemExit("At least one gesture is required.")
    return gestures
