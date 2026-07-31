#!/usr/bin/env python3
"""
Score a realtime_demo.py predictions.csv from a STILLNESS-ONLY test session --
i.e. a run where you deliberately did not perform any real gesture the whole
time, so every single logged row's correct answer is known in advance:
"no_gesture". That makes it trivial to measure the true false-positive rate
of the no-movement gates (--min-activity-z, --settle-window-seconds /
--max-end-activity-z, --confidence-threshold) instead of eyeballing scrolling
console output.

Record the test:
    python src/realtime_demo.py --model models/cnn_raw_with_resting.joblib \\
        <same sensor/gate flags you're using> --duration 90 --session-name stillness_test
    (then just stand there -- don't perform any gesture at all for the whole 90s)

Score it:
    python src/analyze_stillness_test.py sessions/eval_stillness_test/realtime_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions_csv", help="realtime_demo.py's logged predictions.csv from a stillness-only run.")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.predictions_csv, newline="")))
    if not rows:
        print("No rows in that CSV -- did the session actually capture anything?")
        return 1

    total = len(rows)
    correct = [r for r in rows if r["gated_prediction"] == "no_gesture"]
    false_positives = [r for r in rows if r["gated_prediction"] != "no_gesture"]

    gate_counts = Counter(r["gate_reason"] for r in correct)

    print(f"Total captured windows during stillness: {total}")
    print(f"Correctly shown as no_gesture: {len(correct)} ({100 * len(correct) / total:.1f}%)")
    print(f"False positives (showed a real gesture while you did nothing): {len(false_positives)} "
          f"({100 * len(false_positives) / total:.1f}%)")
    print()
    print("Of the correct no_gesture calls, which gate caught them:")
    for reason, count in gate_counts.most_common():
        label = {
            "min_activity": "min-activity-z (no motion at all)",
            "not_settled": "settle gate (never settled by window end)",
            "confidence": "confidence threshold (low-confidence raw guess)",
            "none": "model itself predicted no_gesture/resting natively",
        }.get(reason, reason)
        print(f"  {count:4d}  {label}")

    if false_positives:
        print()
        print("False positives -- these are what nothing currently catches:")
        for r in false_positives:
            print(
                f"  [{r['time_s']}s] raw guess: {r['raw_prediction']} "
                f"(confidence {r['raw_confidence']}, threshold was {r['confidence_threshold']})"
            )
        max_conf = max(float(r["raw_confidence"]) for r in false_positives if r["raw_confidence"])
        print()
        print(f"To catch ALL of these via --confidence-threshold alone, you'd need to raise it to "
              f"at least {max_conf:.2f} -- but check whether that's still low enough to accept your "
              f"real gestures before doing that (compare against a real-gesture test run's confidences).")
    else:
        print()
        print("Zero false positives in this stillness test -- the no-movement gates are working for "
              "this session's conditions.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
