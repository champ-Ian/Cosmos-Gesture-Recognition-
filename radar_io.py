#!/usr/bin/env python3
"""
TI xWRL6432 mmWave UART protocol helpers.

Adapted from UCLA COSMOS `mmwave_lab` (get_range_profile.py, point_cloud_viewer.py,
near_field_gesture_viewer.py). Trimmed down to the parsing/config primitives needed
for headless background-thread data collection: no plotting, no CLI entry point.
"""
from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import serial

# The four uint16 words {0x0102, 0x0304, 0x0506, 0x0708}
# appear in little-endian byte order on UART.
MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"

# xWRL6432 extended range-profile TLVs.
RANGE_PROFILE_MAJOR = 302
RANGE_PROFILE_MINOR = 303

# xWRL6432 point-cloud TLVs.
POINT_CLOUD_FLOAT = 1
POINT_CLOUD_FIXED_TYPES = {301, 1020}

SPEED_OF_LIGHT = 299_792_458.0
OPTIONAL_UNSUPPORTED_COMMANDS = {"cfarScndPassCfg", "compressionCfg"}
CLI_FAILURE_PATTERNS = ("error", "not recognized", "invalid", "failed")
CLI_OK_PATTERNS = ("done", "mmwdemo:", "skipped")


@dataclass(frozen=True)
class RangeConfig:
    bin_spacing_m: float
    fft_size: int
    num_range_bins: int


@dataclass(frozen=True)
class PointCloud:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    velocity: np.ndarray


def empty_point_cloud() -> PointCloud:
    empty = np.array([], dtype=float)
    return PointCloud(empty, empty, empty, empty)


# ---------------------------------------------------------------------------
# Low-level UART framing
# ---------------------------------------------------------------------------


def read_exact(port: serial.Serial, count: int, timeout_s: float) -> bytes:
    """Read exactly count bytes, waiting through serial timeouts."""
    output = bytearray()
    deadline = time.monotonic() + timeout_s

    while len(output) < count:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out while reading {count} bytes ({len(output)} received)."
            )
        chunk = port.read(count - len(output))
        if chunk:
            output.extend(chunk)

    return bytes(output)


def wait_for_magic_word(port: serial.Serial, timeout_s: float) -> None:
    """Discard ASCII/debug data until the binary frame sync word appears."""
    window = bytearray()
    discarded = bytearray()
    deadline = time.monotonic() + timeout_s

    while True:
        if time.monotonic() > deadline:
            message = "Timed out waiting for a UART frame magic word."
            text = bytes(discarded[-512:]).decode("ascii", errors="ignore").strip()
            if text:
                message += f" Last UART text: {text}"
            raise TimeoutError(message)

        value = port.read(1)
        if not value:
            continue

        window.extend(value)
        discarded.extend(value)

        recent_text = bytes(discarded[-256:]).decode("ascii", errors="ignore").lower()
        if "return error" in recent_text or "assert" in recent_text:
            raise RuntimeError(
                "The radar firmware reported an error while starting: "
                f"{recent_text.strip()}"
            )

        if len(window) > len(MAGIC_WORD):
            del window[0]

        if bytes(window) == MAGIC_WORD:
            return


def parse_tlvs(
    packet: bytes,
    number_of_tlvs: int,
    expected_range_profile_bytes: int | None = None,
) -> list[tuple[int, bytes]]:
    """
    Parse xWRL6432 TLVs.

    Some TI docs label the frame header as 52 bytes, while the documented
    fields and observed xWRL6432 frames are 40 bytes. Try both offsets, but
    prefer candidates matching the expected range-profile payload size.
    """
    candidates: list[tuple[int, list[tuple[int, bytes]]]] = []

    for header_size in (40, 52):
        position = header_size
        output: list[tuple[int, bytes]] = []
        valid = True

        for _ in range(number_of_tlvs):
            if position + 8 > len(packet):
                valid = False
                break

            tlv_type, tlv_length = struct.unpack_from("<II", packet, position)
            position += 8

            if tlv_type == 0 or tlv_length > len(packet) - position:
                valid = False
                break

            payload = packet[position : position + tlv_length]
            output.append((tlv_type, payload))
            position += tlv_length

        if valid:
            score = 0
            if header_size == 40:
                score += 1
            if position == len(packet):
                score += 4
            trailing = packet[position:]
            if trailing and not any(trailing):
                score += 2
            if expected_range_profile_bytes is not None:
                for tlv_type, payload in output:
                    if (
                        tlv_type in {RANGE_PROFILE_MAJOR, RANGE_PROFILE_MINOR}
                        and len(payload) == expected_range_profile_bytes
                    ):
                        score += 16
                        break
            candidates.append((score, output))

    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    raise ValueError("Could not identify the TLV start position.")


def read_frame(
    port: serial.Serial,
    timeout_s: float,
    expected_range_profile_bytes: int | None = None,
) -> tuple[int, list[tuple[int, bytes]]]:
    wait_for_magic_word(port, timeout_s)

    first_40_bytes = MAGIC_WORD + read_exact(port, 32, timeout_s)

    (
        _version,
        total_packet_length,
        _platform,
        frame_number,
        _cpu_cycles,
        _number_of_objects,
        number_of_tlvs,
        _subframe_number,
    ) = struct.unpack_from("<8I", first_40_bytes, 8)

    if not 40 <= total_packet_length <= 2_000_000:
        raise ValueError(f"Implausible packet length: {total_packet_length}")

    if number_of_tlvs > 64:
        raise ValueError(f"Implausible TLV count: {number_of_tlvs}")

    packet = first_40_bytes + read_exact(port, total_packet_length - 40, timeout_s)

    return frame_number, parse_tlvs(packet, number_of_tlvs, expected_range_profile_bytes)


# ---------------------------------------------------------------------------
# TLV decoding
# ---------------------------------------------------------------------------


def range_profile_from_tlvs(tlvs: list[tuple[int, bytes]]) -> np.ndarray | None:
    for tlv_type, payload in tlvs:
        if tlv_type in {RANGE_PROFILE_MAJOR, RANGE_PROFILE_MINOR}:
            return np.frombuffer(payload, dtype="<u4").astype(float)
    return None


def decode_float_points(payload: bytes) -> PointCloud:
    count = len(payload) // 16
    if count == 0:
        return empty_point_cloud()

    values = np.frombuffer(payload[: count * 16], dtype="<f4").reshape(count, 4)
    return PointCloud(
        x=values[:, 0].astype(float),
        y=values[:, 1].astype(float),
        z=values[:, 2].astype(float),
        velocity=values[:, 3].astype(float),
    )


def decode_fixed_points(payload: bytes) -> PointCloud:
    if len(payload) < 20:
        return empty_point_cloud()

    xyz_unit, doppler_unit, _snr_unit, _noise_unit = struct.unpack_from("<ffff", payload, 0)
    num_major_points, _num_minor_points = struct.unpack_from("<HH", payload, 16)

    x_values: list[float] = []
    y_values: list[float] = []
    z_values: list[float] = []
    velocities: list[float] = []

    offset = 20
    for _ in range(num_major_points):
        if offset + 10 > len(payload):
            break
        x, y, z, doppler, _snr, _noise = struct.unpack_from("<hhhhBB", payload, offset)
        x_values.append(x * xyz_unit)
        y_values.append(y * xyz_unit)
        z_values.append(z * xyz_unit)
        velocities.append(doppler * doppler_unit)
        offset += 10

    return PointCloud(
        x=np.array(x_values, dtype=float),
        y=np.array(y_values, dtype=float),
        z=np.array(z_values, dtype=float),
        velocity=np.array(velocities, dtype=float),
    )


def point_cloud_from_tlvs(tlvs: list[tuple[int, bytes]]) -> PointCloud:
    for tlv_type, payload in tlvs:
        if tlv_type == POINT_CLOUD_FLOAT:
            return decode_float_points(payload)
        if tlv_type in POINT_CLOUD_FIXED_TYPES:
            return decode_fixed_points(payload)
    return empty_point_cloud()


# ---------------------------------------------------------------------------
# CLI configuration
# ---------------------------------------------------------------------------


def load_configuration(path: Path) -> list[str]:
    commands: list[str] = []

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # TI config files commonly use % for comments.
        if line.startswith(("%", "#")):
            continue
        commands.append(line)

    if not any(line.startswith("sensorStart") for line in commands):
        raise ValueError("The configuration file has no sensorStart command.")

    return commands


def parse_range_config(commands: list[str]) -> RangeConfig | None:
    """
    Estimate range metadata from the xWRL6432 CLI configuration.

    Fs = 100 MHz / DigOutputSampRate
    delta_r = c * Fs / (2 * slope * N_FFT)
    """
    sample_rate_divider: float | None = None
    adc_samples: int | None = None
    slope_mhz_per_us: float | None = None

    for command in commands:
        fields = command.split()
        if fields[0] == "chirpComnCfg" and len(fields) >= 5:
            sample_rate_divider = float(fields[1])
            adc_samples = int(fields[4])
        elif fields[0] == "chirpTimingCfg" and len(fields) >= 5:
            slope_mhz_per_us = float(fields[4])

    if sample_rate_divider is None or adc_samples is None or slope_mhz_per_us is None or slope_mhz_per_us == 0:
        return None

    if sample_rate_divider <= 0:
        raise ValueError(f"Invalid chirpComnCfg DigOutputSampRate: {sample_rate_divider}")
    if adc_samples <= 0:
        raise ValueError(f"Invalid chirpComnCfg NumOfAdcSamples: {adc_samples}")

    sampling_rate_hz = 100e6 / sample_rate_divider
    fft_size = 1 << (adc_samples - 1).bit_length()
    slope_hz_per_second = abs(slope_mhz_per_us) * 1e12

    bin_spacing_m = SPEED_OF_LIGHT * sampling_rate_hz / (2.0 * slope_hz_per_second * fft_size)

    return RangeConfig(
        bin_spacing_m=bin_spacing_m,
        fft_size=fft_size,
        num_range_bins=fft_size // 2,
    )


def read_text_until_quiet(port: serial.Serial, quiet_time: float = 0.15, max_time: float = 2.0) -> str:
    """Read UART text until there has been no input for quiet_time."""
    start = time.monotonic()
    last_rx = start
    chunks: list[bytes] = []

    while time.monotonic() - start < max_time:
        waiting = port.in_waiting
        if waiting:
            chunks.append(port.read(waiting))
            last_rx = time.monotonic()
            continue
        if time.monotonic() - last_rx >= quiet_time:
            break
        time.sleep(0.01)

    return b"".join(chunks).decode("ascii", errors="ignore")


def write_cli_command(port: serial.Serial, command: str) -> None:
    port.write((command + "\n").encode("ascii"))
    port.flush()


def cli_response_failed(reply: str) -> bool:
    reply_lower = reply.lower()
    return any(pattern in reply_lower for pattern in CLI_FAILURE_PATTERNS)


def cli_response_ok(reply: str) -> bool:
    reply_lower = reply.lower()
    return any(pattern in reply_lower for pattern in CLI_OK_PATTERNS)


def require_plausible_cli_response(command: str, reply: str) -> None:
    if cli_response_ok(reply) or cli_response_failed(reply):
        return

    if reply.strip():
        raise RuntimeError(
            "Unexpected CLI response while sending "
            f"{command!r}. This usually means the UART baud rate is wrong "
            "or the sensor was already streaming binary frames. Reset or "
            "power-cycle the EVM to return the demo CLI to 115200."
        )

    raise RuntimeError(
        "No CLI response while sending "
        f"{command!r}. Check the serial port and baud rate; reset or "
        "power-cycle the EVM if the port is correct."
    )


def send_configuration(port: serial.Serial, commands: list[str], use_cfg_baud_rate: bool) -> None:
    start_command: str | None = None

    for command in commands:
        if command.startswith("sensorStart"):
            start_command = command
            continue

        command_name = command.split()[0]

        if command_name == "baudRate" and not use_cfg_baud_rate:
            print(f"> {command} (skipped; keeping host UART at {port.baudrate} baud)")
            continue

        print(f"> {command}")
        write_cli_command(port, command)

        if command_name == "baudRate":
            fields = command.split()
            if len(fields) != 2:
                raise ValueError(f"Invalid baudRate command: {command}")
            new_baud = int(fields[1])
            time.sleep(0.15)
            port.baudrate = new_baud
            print(f"Host UART changed to {new_baud} baud.")
            reply = read_text_until_quiet(port, max_time=1.0)
            if reply:
                print(reply, end="")
            if cli_response_failed(reply):
                raise RuntimeError(f"Device returned an error for command: {command}")
            continue

        reply = read_text_until_quiet(port)
        if reply and (cli_response_ok(reply) or cli_response_failed(reply)):
            print(reply, end="")

        require_plausible_cli_response(command, reply)

        if "not recognized" in reply.lower() and command_name in OPTIONAL_UNSUPPORTED_COMMANDS:
            print(f"Skipping unsupported optional command: {command_name}")
            continue

        if cli_response_failed(reply):
            raise RuntimeError(f"Device returned an error for command: {command}")

    if start_command is None:
        raise ValueError("No sensorStart command found.")

    port.reset_input_buffer()
    print(f"> {start_command}")
    write_cli_command(port, start_command)


def remove_leading_sensor_stop(commands: list[str]) -> list[str]:
    if commands and commands[0].split()[0] == "sensorStop":
        return commands[1:]
    return commands


def stop_and_drain(port: serial.Serial) -> None:
    """Recover the CLI even if binary frames are still streaming."""
    port.reset_input_buffer()

    deadline = time.monotonic() + 4.0
    last_rx = time.monotonic()
    next_stop = 0.0
    text_tail = ""

    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_stop:
            port.write(b"sensorStop 0\r\n")
            port.flush()
            next_stop = now + 0.25

        waiting = port.in_waiting
        if waiting:
            data = port.read(waiting)
            text_tail = (text_tail + data.decode("ascii", errors="ignore"))[-512:]
            last_rx = time.monotonic()
        elif "done" in text_tail.lower() and time.monotonic() - last_rx > 0.25:
            break
        elif "mmwdemo:/>" in text_tail.lower() and time.monotonic() - last_rx > 0.5:
            break
        else:
            time.sleep(0.02)

    quiet_deadline = time.monotonic() + 0.6
    while time.monotonic() < quiet_deadline:
        waiting = port.in_waiting
        if waiting:
            port.read(waiting)
            quiet_deadline = time.monotonic() + 0.6
        else:
            time.sleep(0.02)

    port.reset_input_buffer()
    time.sleep(0.2)


def warm_reset_demo(port: serial.Serial, timeout_s: float = 8.0) -> None:
    """Reload the flashed demo so the UART is back at a clean CLI prompt."""
    print("> sensorWarmRst")
    port.reset_input_buffer()
    write_cli_command(port, "sensorWarmRst")

    chunks: list[bytes] = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        waiting = port.in_waiting
        if waiting:
            chunks.append(port.read(waiting))
            text = b"".join(chunks[-8:]).decode("ascii", errors="ignore").lower()
            if "mmwdemo:/>" in text:
                port.reset_input_buffer()
                return
        else:
            time.sleep(0.05)

    text = b"".join(chunks).decode("ascii", errors="ignore")
    if "not supported" in text.lower() or "error" in text.lower():
        raise RuntimeError(f"sensorWarmRst failed: {text.strip()}")

    raise RuntimeError("Timed out waiting for mmwDemo prompt after sensorWarmRst.")
