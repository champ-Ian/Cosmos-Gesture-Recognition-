#!/usr/bin/env python3
"""
Background-thread reader for the TI xWRL6432 mmWave radar.

Runs `radar_io.read_frame` continuously on its own thread so the radar can be
captured in parallel with the other IoT sensors (IMU / UWB / RFID), each on
its own serial port. Frames are timestamped with `time.monotonic()` so a
collection script can slice out the frames that fall inside a given gesture
trial window, the same way the text-line sensors are windowed.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import serial

from mmwave.radar_io import (
    PointCloud,
    RangeConfig,
    load_configuration,
    parse_range_config,
    point_cloud_from_tlvs,
    range_profile_from_tlvs,
    read_frame,
    remove_leading_sensor_stop,
    send_configuration,
    stop_and_drain,
    warm_reset_demo,
)
from sensors.base_reader import BaseReader


@dataclass
class RadarFrameRecord:
    recv_time_s: float
    frame_number: int
    range_profile: np.ndarray | None
    points_xyz: np.ndarray
    points_velocity: np.ndarray


class MmwaveReader(BaseReader):
    """Continuously reads radar frames on a background thread."""

    def __init__(
        self,
        port_path: str,
        cfg_path: Path,
        baud: int = 115200,
        frame_timeout_s: float = 5.0,
        warm_reset: bool = True,
    ) -> None:
        self.name = "mmwave"
        self.port_path = port_path
        self.cfg_path = cfg_path
        self.frame_timeout_s = frame_timeout_s

        commands = load_configuration(cfg_path)
        self.range_config: RangeConfig | None = parse_range_config(commands)
        expected_bytes = (
            None if self.range_config is None else self.range_config.num_range_bins * 4
        )
        self._expected_range_profile_bytes = expected_bytes

        self.port = serial.Serial(port_path, baud, timeout=0.2)
        stop_and_drain(self.port)
        if warm_reset:
            warm_reset_demo(self.port)
        send_configuration(
            self.port,
            remove_leading_sensor_stop(commands),
            use_cfg_baud_rate=False,
        )

        self._lock = threading.Lock()
        self._buffer: list[RadarFrameRecord] = []
        self._stop_event = threading.Event()
        self._error: Exception | None = None
        self._consecutive_warnings = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame_number, tlvs = read_frame(
                    self.port,
                    self.frame_timeout_s,
                    self._expected_range_profile_bytes,
                )
            except (TimeoutError, ValueError, RuntimeError) as error:
                self._consecutive_warnings += 1
                if self._consecutive_warnings >= 5:
                    self._error = error
                continue

            self._consecutive_warnings = 0
            recv_time = time.monotonic()
            profile = range_profile_from_tlvs(tlvs)
            cloud: PointCloud = point_cloud_from_tlvs(tlvs)
            record = RadarFrameRecord(
                recv_time_s=recv_time,
                frame_number=frame_number,
                range_profile=profile,
                points_xyz=(
                    np.column_stack((cloud.x, cloud.y, cloud.z))
                    if len(cloud.x)
                    else np.empty((0, 3), dtype=float)
                ),
                points_velocity=cloud.velocity,
            )
            with self._lock:
                self._buffer.append(record)

    @property
    def frame_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def sample_count(self) -> int:
        return self.frame_count

    def check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"mmWave radar stream failed: {self._error}") from self._error

    def window(self, start_time_s: float, end_time_s: float) -> dict[str, np.ndarray]:
        """Return frames received in [start_time_s, end_time_s] as packed arrays."""
        with self._lock:
            records = [
                record
                for record in self._buffer
                if start_time_s <= record.recv_time_s <= end_time_s
            ]

        frame_count = len(records)
        if frame_count == 0:
            return {
                "frame_number": np.zeros(0, dtype=np.uint32),
                "time_s": np.zeros(0, dtype=float),
                "range_profile": np.zeros((0, 0), dtype=float),
                "point_count": np.zeros(0, dtype=np.uint16),
                "points_xyz": np.zeros((0, 0, 3), dtype=float),
                "points_velocity": np.zeros((0, 0), dtype=float),
            }

        range_bin_count = 0
        for record in records:
            if record.range_profile is not None:
                range_bin_count = max(range_bin_count, len(record.range_profile))

        max_points = max((len(record.points_xyz) for record in records), default=0)

        frame_number = np.array([record.frame_number for record in records], dtype=np.uint32)
        time_s = np.array([record.recv_time_s - start_time_s for record in records], dtype=float)
        range_profile = np.zeros((frame_count, range_bin_count), dtype=float)
        point_count = np.zeros(frame_count, dtype=np.uint16)
        points_xyz = np.full((frame_count, max_points, 3), np.nan, dtype=float)
        points_velocity = np.full((frame_count, max_points), np.nan, dtype=float)

        for index, record in enumerate(records):
            if record.range_profile is not None:
                length = len(record.range_profile)
                range_profile[index, :length] = record.range_profile
            count = len(record.points_xyz)
            point_count[index] = count
            if count:
                points_xyz[index, :count, :] = record.points_xyz
                points_velocity[index, :count] = record.points_velocity

        return {
            "frame_number": frame_number,
            "time_s": time_s,
            "range_profile": range_profile,
            "point_count": point_count,
            "points_xyz": points_xyz,
            "points_velocity": points_velocity,
        }

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        print("> sensorStop 0")
        stop_and_drain(self.port)
        self.port.close()
